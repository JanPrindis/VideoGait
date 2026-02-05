import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
import numpy as np

DatasetItem = Tuple[np.ndarray, np.ndarray]


def collate_pad(batch: List[DatasetItem]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch of variable-length sequences by padding them.
    """
    feature_matrices, label_matrices = zip(*batch)

    features_tensor = [torch.from_numpy(x).float() for x in feature_matrices]
    labels_tensor = [torch.from_numpy(y).float() for y in label_matrices]

    lengths = torch.tensor([len(x) for x in features_tensor], dtype=torch.long)

    features_padded = pad_sequence(features_tensor, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels_tensor, batch_first=True, padding_value=0.0)  # Padding is 0.0

    return features_padded, labels_padded, lengths


def collate_sliding(batch: List[DatasetItem], window: int = 64, stride: int = 16) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch by creating fixed-size sliding windows.
    """
    X_windows, y_windows = [], []

    for X_np, y_np in batch:
        if X_np.shape[0] < window:
            continue

        X = torch.from_numpy(X_np).float()
        y = torch.from_numpy(y_np).float()

        for start in range(0, X.shape[0] - window + 1, stride):
            X_windows.append(X[start: start + window])
            y_windows.append(y[start: start + window])

    if not X_windows:
        num_features = batch[-1][0].shape[1] if batch else 0
        num_classes = batch[-1][1].shape[1] if batch else 0
        return torch.empty(0, window, num_features), torch.empty(0, window, num_classes), torch.empty(0,
                                                                                                      dtype=torch.long)

    X_stacked = torch.stack(X_windows)
    y_stacked = torch.stack(y_windows)

    lengths = torch.full((len(X_stacked),), window, dtype=torch.long)

    return X_stacked, y_stacked, lengths
