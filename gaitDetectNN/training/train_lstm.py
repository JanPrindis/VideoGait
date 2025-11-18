import os
import random
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from Skeletons.halpe_skeleton import HALPE_SKELETON
from gaitDetectNN.models.gaitLSTM import GaitLSTM
from dataset import GaitDataset
from collate import collate_pad
from gaitDetectNN.training.trainer import Trainer
from utils.data import find_matching_annotation
from utils.preprocessing import generate_features

# =============================================================================
# Training Hyperparameters
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 200
RANDOM_SEED = 3

# Data and Model Configuration
SKELETON_DEFINITION = HALPE_SKELETON

KEYPOINTS_USED = [
    "HIP", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_SHOULDER", "RIGHT_SHOULDER"
]
KINEMATICS_USED = [
    "HIP", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]
ANGLES_USED = [
    ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"), ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL", "LEFT_FOOT_INDEX"), ("RIGHT_ANKLE", "RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
    ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"), ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE")
]
DISTANCES_USED = [
    ("HIP", "LEFT_HEEL"), ("HIP", "RIGHT_HEEL"), ("HIP", "LEFT_FOOT_INDEX"), ("HIP", "RIGHT_FOOT_INDEX"),
    ("LEFT_KNEE", "RIGHT_KNEE")
]

# Path Configuration
PROJECT_ROOT = "../.."
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "PROCESSED")
ANNOTATION_ROOT = os.path.join(PROJECT_ROOT, "annotations")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "gaitDetectNN", "checkpoints", "lstm_best.pth")

# =============================================================================
# Data Preparation
# =============================================================================
def main():
    print("Preparing data...")
    keypoint_files = glob(os.path.join(DATASET_ROOT, "60/KEYPOINTS/*.json"))

    if not keypoint_files:
        raise FileNotFoundError("No keypoint files found. Check your DATASET_ROOT path.")

    # Match keypoints with annotations
    file_paths = []
    for kp_path in keypoint_files:
        ann_path = find_matching_annotation(kp_path, ANNOTATION_ROOT)
        if ann_path and os.path.exists(ann_path):
            file_paths.append((kp_path, ann_path))

    # Shuffle the dataset
    random.seed(RANDOM_SEED)
    random.shuffle(file_paths)

    # 80/20 split for training and validation
    split_index = int(len(file_paths) * 0.8)
    train_paths = file_paths[:split_index]
    val_paths = file_paths[split_index:]

    print(f"Found {len(file_paths)} total files. Using {len(train_paths)} for training and {len(val_paths)} for validation.")

    # --- Create Datasets ---
    preprocess_args = {
        "skeleton_definition": SKELETON_DEFINITION,
        "confidence_threshold": 0.5,
        "exclude_ratio": 0.1,
        # Pass the feature definitions
        "keypoints": KEYPOINTS_USED,
        "kinematics_keypoints": KINEMATICS_USED,
        "angle_triplets": ANGLES_USED,
        "distance_pairs": DISTANCES_USED
    }

    train_dataset = GaitDataset(train_paths, preprocessing_fn=generate_features, **preprocess_args)
    val_dataset = GaitDataset(val_paths, preprocessing_fn=generate_features, **preprocess_args)

    # --- Create DataLoaders ---
    # and pin_memory for faster data transfer to the GPU.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_pad, num_workers=0, pin_memory=True)

    # =============================================================================
    # Model, Loss, and Optimizer
    # =============================================================================
    if not train_dataset.data:
        raise ValueError("Training dataset is empty after preprocessing. Cannot determine model input size.")

    num_features = train_dataset.data[0][0].shape[1]
    print(f"Initializing model on {DEVICE} with {num_features} input features.")
    model = GaitLSTM(input_size=num_features)

    # Calculate class weights for the loss function
    positives = 0
    total_samples = 0
    for _, labels in train_dataset:
        positives += np.sum(labels)
        total_samples += labels.size

    negatives = total_samples - positives
    pos_weight_value = negatives / positives if positives > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_value], device=DEVICE)

    print(f"Positive Samples: {positives}, Negative Samples: {negatives}")
    print(f"Applying a positive weight of {pos_weight_value:.2f} to the loss function.")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # =============================================================================
    # Training
    # =============================================================================
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, DEVICE)
    history = trainer.fit(
        num_epochs=NUM_EPOCHS,
        save_path=CHECKPOINT_PATH
    )

    print("Training complete.")


if __name__ == "__main__":
    main()
