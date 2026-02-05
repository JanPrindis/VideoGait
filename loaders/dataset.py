import os

import numpy as np
from torch.utils.data import Dataset

class GaitDataset(Dataset):
    def __init__(self, file_paths: list[tuple[str, str]], preprocessing_fn: callable, **preprocess_kwargs):
        self.data = []

        for kpt_path, ann_path in file_paths:
            if not os.path.exists(kpt_path) or not os.path.exists(ann_path):
                print(f"[Warning] Skipping missing file pair: {kpt_path}, {ann_path}")
                continue

            feature_matrices, label_matrices, _ = preprocessing_fn(
                keypoints_path=kpt_path,
                annotations_path=ann_path,
                **preprocess_kwargs
            )

            if feature_matrices and label_matrices:
                self.data.extend(zip(feature_matrices, label_matrices))

        print(f"Dataset created with {len(self.data)} total clips.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        feature_matrix, label_matrix = self.data[idx]
        return feature_matrix, label_matrix
