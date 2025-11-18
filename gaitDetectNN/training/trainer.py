# In D:/Semestralka/GaitPhaseDetection/gaitDetectNN/training/trainer.py

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
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
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

        # --- KEY FIX 1: Slice labels to match the output length from the model ---
        labels = labels[:, :t, :]

        # --- KEY FIX 2: Clamp lengths to ensure they don't exceed the logits' length ---
        # This prevents the mask from being larger than the tensor it's applied to.
        lengths = torch.clamp(lengths, max=t)

        # Check if the batch contains sequences of different lengths
        is_padded = len(torch.unique(lengths)) > 1

        if is_padded:
            # --- PADDED CASE (from collate_pad) ---
            mask = self._create_mask(lengths, t)

            active_logits = logits[mask]
            active_labels = labels[mask]

            if active_logits.numel() == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            return self.criterion(active_logits, active_labels)
        else:
            # --- UNPADDED CASE (from collate_sliding) ---
            return self.criterion(logits.reshape(-1, c), labels.reshape(-1, c))

    def train_epoch(self):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        for features, labels, lengths in self.train_loader:
            features, labels, lengths = features.to(self.device), labels.to(self.device), lengths.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(features, lengths.cpu())
            loss = self._compute_loss(logits, labels, lengths)

            if loss.item() > 0:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        """Runs a single validation epoch."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for features, labels, lengths in self.val_loader:
                features, labels, lengths = features.to(self.device), labels.to(self.device), lengths.to(self.device)

                logits = self.model(features, lengths.cpu())
                loss = self._compute_loss(logits, labels, lengths)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def fit(self, num_epochs: int, save_path: str | None = None):
        print("Starting training... Press Ctrl+C to stop early.")
        try:
            for epoch in range(num_epochs):
                train_loss = self.train_epoch()
                val_loss = self.validate_epoch()

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)

                print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}",
                      end="")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    print(" -> New best model saved!")
                else:
                    print()

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user...")

        except Exception as e:
            print(f"\n\nSomething went wrong during training: {e}")

        finally:
            if self.best_val_loss == float('inf'):
                print("\nNo best model to save. Training did not complete a full validation epoch.")
            else:
                print(f"\nLoading best model with validation loss: {self.best_val_loss:.4f}")
                self.model.load_state_dict(self.best_model_state)

                if save_path:
                    try:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(self.best_model_state, save_path)
                        print(f"Best model weights saved to {save_path}")
                    except Exception as e:
                        print(f"Error saving model to {save_path}: {e}")

            print("Training finished.")
            return self.history
