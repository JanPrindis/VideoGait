import torch
import os
import copy
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from utils.logger import log


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, f1_tolerance_frames):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': []
        }

        # Hybrid saving state
        self.f1_tolerance_frames = f1_tolerance_frames
        self.best_val_f1 = -1.0             # We are maximizing F1 score
        self.best_val_loss = float('inf')   # We are minimizing loss if F1 score stays the same
        self.best_model_state = copy.deepcopy(self.model.state_dict())

    def _create_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Creates a boolean mask from sequence lengths."""
        mask = torch.arange(max_len, device=self.device)[None, :] < lengths[:, None]
        return mask

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Computes loss, handling potential length mismatches from sequence packing
        and applying a mask for variable-length sequences.
        """
        # Get the sequence length from the model's output
        b, t, c = logits.shape

        # Slice labels to match the output length from the model
        labels = labels[:, :t, :]

        # This prevents the mask from being larger than the tensor it's applied to.
        lengths = torch.clamp(lengths, max=t)

        # Check if the batch contains sequences of different lengths
        is_padded = len(torch.unique(lengths)) > 1

        if is_padded:
            # --- PADDED CASE ---
            mask = self._create_mask(lengths, t)

            active_logits = logits[mask]
            active_labels = labels[mask]

            if active_logits.numel() == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            return self.criterion(active_logits, active_labels)
        else:
            # --- UNPADDED CASE ---
            return self.criterion(logits.reshape(-1, c), labels.reshape(-1, c))

    def _compute_windowed_f1(self, logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
                             tolerance: int = 3):
        """
        Computes a windowed F1 score, where a prediction is considered a true positive
        if it falls within a specified tolerance window around a ground truth event.

        Args:
            logits (torch.Tensor): The raw output from the model (B, T, C).
            labels (torch.Tensor): The ground truth labels (B, T, C).
            lengths (torch.Tensor): The actual sequence lengths for each item in the batch (B,).
            tolerance (int): The number of frames on either side of an event to
                             consider a match valid (e.g., tolerance=3 means a 7-frame window).

        Returns:
            float: The calculated windowed F1 score for the batch.
        """
        with torch.no_grad():
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()  # (B, T, C)

            b, t, c = logits.shape  #
            labels = labels[:, :t, :]  #

            # lengths shape (B,), mask shape (B, T)
            mask = self._create_mask(lengths, t)  #
            mask = mask.unsqueeze(-1).expand_as(predictions)  # (B, T, C)

            # Masking - ignore padding
            predictions = predictions * mask
            labels = labels * mask

            predictions_flat = predictions.permute(0, 2, 1).reshape(b * c, 1, t)  # (B*C, 1, T)
            labels_flat = labels.permute(0, 2, 1).reshape(b * c, 1, t)  # (B*C, 1, T)

            kernel_size = 2 * tolerance + 1
            padding = tolerance

            # Dilation
            labels_dilated = F.max_pool1d(labels_flat, kernel_size=kernel_size, stride=1, padding=padding)
            labels_dilated = labels_dilated[:, :, :t]

            predictions_dilated = F.max_pool1d(predictions_flat, kernel_size=kernel_size, stride=1, padding=padding)
            predictions_dilated = predictions_dilated[:, :, :t]

            #  TP, FP, FN
            true_positives_prec = (predictions_flat * labels_dilated).sum().item()
            predicted_positives = predictions_flat.sum().item()

            true_positives_rec = (labels_flat * predictions_dilated).sum().item()
            actual_positives = labels_flat.sum().item()
            epsilon = 1e-7

            # Precision
            precision = true_positives_prec / (predicted_positives + epsilon)

            # Recall
            recall = true_positives_rec / (actual_positives + epsilon)

            f1 = 2 * (precision * recall) / (precision + recall + epsilon)

            return f1

    def train_epoch(self):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        total_f1 = 0.0

        for features, labels, lengths in self.train_loader:
            features, labels, lengths = features.to(self.device), labels.to(self.device), lengths.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(features, lengths.cpu())

            loss = self._compute_loss(logits, labels, lengths)
            f1 = self._compute_windowed_f1(logits, labels, lengths, tolerance=self.f1_tolerance_frames)

            if loss.item() > 0:
                loss.backward()
                self.optimizer.step()

                # OneCycleLR is called per batch
                if self.scheduler is not None and isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()

            total_loss += loss.item()
            total_f1 += f1

        return total_loss / len(self.train_loader), total_f1 / len(self.train_loader)

    def validate_epoch(self):
        """Runs a single validation epoch."""
        self.model.eval()
        total_loss = 0.0
        total_f1 = 0.0

        with torch.no_grad():
            for features, labels, lengths in self.val_loader:
                features, labels, lengths = features.to(self.device), labels.to(self.device), lengths.to(self.device)

                logits = self.model(features, lengths.cpu())

                loss = self._compute_loss(logits, labels, lengths)
                f1 = self._compute_windowed_f1(logits, labels, lengths, tolerance=self.f1_tolerance_frames)

                total_loss += loss.item()
                total_f1 += f1

        return total_loss / len(self.val_loader), total_f1 / len(self.val_loader)

    def fit(self, num_epochs: int, save_path: str | None = None):
        log("TRAINER", "Starting training... Press Ctrl+C to stop early.", level="info")
        try:
            for epoch in range(num_epochs):
                train_loss, train_f1 = self.train_epoch()
                val_loss, val_f1 = self.validate_epoch()

                if self.scheduler is not None and not isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_f1'].append(train_f1)
                self.history['val_f1'].append(val_f1)

                msg = (f"Epoch {epoch + 1}/{num_epochs} | "
                       f"LR: {current_lr:.6f} | "
                       f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                       f"F1: {train_f1:.4f}/{val_f1:.4f}")

                is_best = False
                reason = ""

                # if F1 score is higher -> save
                if val_f1 > self.best_val_f1:
                    is_best = True
                    reason = f"New Best F1 ({val_f1:.4f})"

                # if F1 score stays the same, but the loss is lower -> save
                elif val_f1 == self.best_val_f1:
                    if val_loss < self.best_val_loss:
                        is_best = True
                        reason = f"Tie-break: Lower Loss ({val_loss:.4f})"

                if is_best:
                    self.best_val_f1 = val_f1
                    self.best_val_loss = val_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    log("TRAINER", f"{msg} -> SAVED! {reason}", level="success")
                else:
                    log("TRAINER", msg, level="info")

        except KeyboardInterrupt:
            log("TRAINER", "Training interrupted by user...", level="warning")

        except Exception as e:
            log("TRAINER", f"Something went wrong during training: {e}", level="error")
            raise

        finally:
            if self.best_val_f1 == -1.0:
                log("TRAINER", "No best model to save.", level="warning")
            else:
                log("TRAINER", f"Loading best model (F1: {self.best_val_f1:.4f}, Loss: {self.best_val_loss:.4f})", level="info")
                self.model.load_state_dict(self.best_model_state)

                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.best_model_state, save_path)
                    log("TRAINER", f"Best model saved to {save_path}", level="success")

            log("TRAINER", "Training finished.", level="success")
            return self.history
