import torch
import os
import copy

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': []
        }

        # Hybrid saving state
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

    def _compute_f1(self, logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor):
        """
        Computes F1 score.
        """
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            b, t, c = logits.shape
            labels = labels[:, :t, :]

            # Mask padding
            lengths = torch.clamp(lengths, max=t)
            mask = self._create_mask(lengths, t)

            mask_expanded = mask.unsqueeze(-1).expand_as(preds)

            active_preds = preds[mask_expanded]
            active_labels = labels[mask_expanded]

            if active_preds.numel() == 0:
                return 0.0

            # TP, FP, FN calculation
            tp = (active_preds * active_labels).sum().item()
            fp = (active_preds * (1 - active_labels)).sum().item()
            fn = ((1 - active_preds) * active_labels).sum().item()

            # F1 Score formula
            epsilon = 1e-7
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
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
            f1 = self._compute_f1(logits, labels, lengths)

            if loss.item() > 0:
                loss.backward()
                self.optimizer.step()

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
                f1 = self._compute_f1(logits, labels, lengths)

                total_loss += loss.item()
                total_f1 += f1

        return total_loss / len(self.val_loader), total_f1 / len(self.val_loader)

    def fit(self, num_epochs: int, save_path: str | None = None):
        print("Starting training... Press Ctrl+C to stop early.")
        try:
            for epoch in range(num_epochs):
                train_loss, train_f1 = self.train_epoch()
                val_loss, val_f1 = self.validate_epoch()

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_f1'].append(train_f1)
                self.history['val_f1'].append(val_f1)

                print(f"Epoch {epoch + 1}/{num_epochs} | "
                      f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                      f"F1: {train_f1:.4f}/{val_f1:.4f}", end="")

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
                    print(f" -> SAVED! {reason}")
                else:
                    print()

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user...")

        except Exception as e:
            print(f"\n\nSomething went wrong during training: {e}")
            pass

        finally:
            if self.best_val_f1 == -1.0:
                print("\nNo best model to save.")
            else:
                print(f"\nLoading best model (F1: {self.best_val_f1:.4f}, Loss: {self.best_val_loss:.4f})")
                self.model.load_state_dict(self.best_model_state)

                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.best_model_state, save_path)
                    print(f"Best model saved to {save_path}")

            print("Training finished.")
            return self.history
