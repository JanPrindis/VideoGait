import torch
import numpy as np


class Predictor:
    """
    A generic inference pipeline for running gait detection models on pre-processed data.
    """

    def __init__(self, model: torch.nn.Module, device: str):
        """
        Args:
            model (torch.nn.Module): The trained PyTorch model, already loaded.
            device (str): The device to run inference on ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.model.eval()  # Set the model to evaluation mode
        self.device = device

    def predict_on_clip(self, features_clip: np.ndarray) -> np.ndarray:
        """
        Runs inference on a single, pre-processed feature clip.

        Args:
            features_clip (np.ndarray): A numpy array of shape (seq_len, num_features)
                                        containing the keypoint data for one clip.

        Returns:
            np.ndarray: A numpy array of shape (seq_len, num_classes) containing
                        the predicted probabilities for each event.
        """
        with torch.no_grad():
            # Numpy -> Tensor
            features_tensor = torch.from_numpy(features_clip).float().to(self.device)

            # Add a "batch" dimension. The model expects a batch of clips.
            # Shape becomes: (1, seq_len, num_features)
            features_tensor = features_tensor.unsqueeze(0)

            # Create a lengths tensor for this single clip
            lengths_tensor = torch.tensor([features_clip.shape[0]], device='cpu')

            # Forward pass
            logits = self.model(features_tensor, lengths_tensor)

            # Sigmoid (Logits -> Probabilities)
            probabilities = torch.sigmoid(logits)

            # Back to CPU
            predicted_probs = probabilities.squeeze(0).cpu().numpy()

            return predicted_probs
