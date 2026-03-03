import os

import numpy as np
from torch.utils.data import Dataset
from utils.logger import log

class GaitDataset(Dataset):
    """
    A PyTorch Dataset for loading gait analysis data.

    This dataset takes pairs of keypoint and annotation file paths, applies a
    preprocessing function to generate feature and label matrices, and makes
    them available for training a neural network.
    """
    def __init__(self, file_paths: list[tuple[str, str]], preprocessing_fn: callable, **preprocess_kwargs):
        """
        Initializes the dataset by processing all provided file paths.

        Args:
            file_paths (list[tuple[str, str]]): A list of tuples, where each tuple
                contains the path to a keypoint JSON file and its corresponding
                annotation JSON file.
            preprocessing_fn (callable): The function to call to process a single
                keypoint/annotation pair into feature and label matrices.
            **preprocess_kwargs: Additional keyword arguments to pass to the
                `preprocessing_fn`.
        """
        self.data = []

        for kpt_path, ann_path in file_paths:
            if not os.path.exists(kpt_path) or not os.path.exists(ann_path):
                log("DATASET", f"Skipping missing file pair: {kpt_path}, {ann_path}", level="warning")
                continue

            feature_matrices, label_matrices, _ = preprocessing_fn(
                keypoints_path=kpt_path,
                annotations_path=ann_path,
                **preprocess_kwargs
            )

            if feature_matrices and label_matrices:
                self.data.extend(zip(feature_matrices, label_matrices))

        log("DATASET", f"Dataset created with {len(self.data)} total clips.", level="success")

    def __len__(self) -> int:
        """Returns the total number of clips in the dataset."""
        return len(self.data)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves a single clip (feature and label matrices) from the dataset.

        Args:
            idx (int): The index of the clip to retrieve.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the feature matrix
                and the corresponding label matrix.
        """
        feature_matrix, label_matrix = self.data[idx]
        return feature_matrix, label_matrix
