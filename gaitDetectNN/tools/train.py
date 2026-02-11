import argparse
import yaml
import sys
import os
import torch
import numpy as np
import random
import json
import shutil
from glob import glob

from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- PATH SETUP ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- IMPORTS ---
from gaitDetectNN.builder import build_model
from gaitDetectNN.engine.trainer import Trainer

from loaders import GaitDataset, collate_pad
from utils.preprocessing import generate_features
from utils.data import find_matching_annotation
from skeletons import get_skeleton_by_name
from utils.logger import log

def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, training_cfg):
    opt_type = training_cfg.get('optimizer', 'AdamW')  # Default AdamW
    lr = float(training_cfg['learning_rate'])
    weight_decay = float(training_cfg.get('weight_decay', 0.01))  # Default 0.01

    log("TRAIN", f"Using {opt_type} (lr={lr}, weight_decay={weight_decay})", level="info")

    if opt_type == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif opt_type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif opt_type == 'SGD':
        momentum = float(training_cfg.get('momentum', 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")


def build_scheduler(optimizer, training_cfg, steps_per_epoch):
    """
    Creates a Learning Rate Scheduler based on config.
    """
    sched_type = training_cfg.get('scheduler', None)

    if not sched_type:
        return None

    # Load scheduler specific config
    sched_params = training_cfg.get('scheduler_config', {})
    if sched_params is None: sched_params = {}

    log("TRAIN", f"Initializing {sched_type} with params: {sched_params}", level="info")

    if sched_type == 'OneCycleLR':
        return OneCycleLR(
            optimizer,
            max_lr=float(training_cfg['learning_rate']),
            epochs=int(training_cfg['epochs']),
            steps_per_epoch=steps_per_epoch,
            **sched_params
        )

    elif sched_type == 'StepLR':
        return StepLR(optimizer, **sched_params)

    else:
        log("TRAIN", f"Unknown scheduler type '{sched_type}'. No scheduler used.", level="warning")
        return None


def save_history(history, output_dir):
    """
    Saves the training history dictionary to a JSON file.

    Args:
        history (dict): A dictionary containing training metrics (e.g., loss, accuracy).
                        Values should be lists of numbers.
        output_dir (str): The directory where 'history.json' will be saved.
    """
    history_path = os.path.join(output_dir, "history.json")
    clean_history = {}
    for k, v in history.items():
        clean_history[k] = [float(x) for x in v]

    with open(history_path, 'w') as f:
        json.dump(clean_history, f, indent=4)
    log("TRAIN", f"History saved to {history_path}", level="success")


def plot_training_curves(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # LOSS
    axs[0].plot(epochs, history['train_loss'], label='Train Loss', color='blue', linestyle='--')
    axs[0].plot(epochs, history['val_loss'], label='Val Loss', color='blue', linewidth=2)

    # Mark minimal loss
    min_loss_idx = np.argmin(history['val_loss'])
    min_loss_val = history['val_loss'][min_loss_idx]
    axs[0].scatter(min_loss_idx + 1, min_loss_val, color='cyan', s=100, zorder=5, label='Min Loss')

    axs[0].set_ylabel('BCE Loss (Weighted)')
    axs[0].set_title(f'Loss Curve (Min Val: {min_loss_val:.4f} at Ep {min_loss_idx + 1})')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # F1 SCORE
    axs[1].plot(epochs, history['train_f1'], label='Train F1', color='green', linestyle='--')
    axs[1].plot(epochs, history['val_f1'], label='Val F1', color='green', linewidth=2)

    # Mark max F1 score
    best_f1_idx = np.argmax(history['val_f1'])
    best_f1_val = history['val_f1'][best_f1_idx]

    # Red dot marks saved model
    axs[1].scatter(best_f1_idx + 1, best_f1_val, color='red', s=150, zorder=5, label='Best Checkpoint (Saved)')

    axs[1].set_ylabel('F1 Score')
    axs[1].set_title(f'F1 Score (Max Val: {best_f1_val:.4f} at Ep {best_f1_idx + 1})')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.xlabel('Epochs')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "training_plot.png")
    plt.savefig(plot_path)
    log("TRAIN", f"Training plot saved to {plot_path}", level="success")


def main():
    """
    Main execution function for the gait detection training script.

    This function handles:
    1. Parsing command-line arguments for configuration.
    2. Setting up the experiment directory and logging.
    3. Preparing the dataset (finding files, pairing keypoints with annotations).
    4. Splitting data into training and validation sets.
    5. Initializing the data loaders with specified preprocessing.
    6. Calculating class weights for handling class imbalance.
    7. Building the model architecture.
    8. Configuring the optimizer and loss function.
    9. Running the training loop via the Trainer class.
    10. Saving the training history and final model.
    """
    parser = argparse.ArgumentParser(description="Gait Detection Training Script")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    # SETUP OUTPUT DIRECTORY
    log("TRAIN", f"Loading configuration from: {args.config}", level="info")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Create folder for experiments
    output_dir_root = "experiments"
    experiment_name = cfg.get('experiment_name', 'default_experiment')
    output_dir = os.path.join(PROJECT_ROOT, output_dir_root, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config copy (for easier transfer)
    shutil.copy(args.config, os.path.join(output_dir, "config.yaml"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = cfg.get('training', {}).get('seed', 3)
    set_seed(seed)

    log("TRAIN", f"Experiment: {experiment_name}", level="info")
    log("TRAIN", f"Output Dir: {output_dir}", level="info")
    log("TRAIN", f"Device: {device} | Seed: {seed}", level="info")

    # DATA PREPARATION
    log("TRAIN", "Preparing dataset...", level="info")
    dataset_root = os.path.join(PROJECT_ROOT, cfg['data']['dataset_root'])
    annotation_root = os.path.join(PROJECT_ROOT, cfg['data'].get('annotation_root', 'annotations'))
    framerate = cfg['data']['framerate']

    # F1 Tolerance Window
    tolerance_ms = cfg['training'].get('f1_window_size_ms', 50)
    tolerance_frames = int(round((tolerance_ms / 1000.0) * framerate))

    search_pattern = os.path.join(dataset_root, str(framerate), "KEYPOINTS", "*.json")
    log("TRAIN", f"Searching: {search_pattern}", level="info")
    keypoint_files = glob(search_pattern, recursive=True)

    if not keypoint_files:
        raise FileNotFoundError(f"No files found at {search_pattern}")

    # Pairing
    file_paths = []
    for kp_path in keypoint_files:
        ann_path = find_matching_annotation(kp_path, annotation_root)
        if ann_path and os.path.exists(ann_path):
            file_paths.append((kp_path, ann_path))

    log("TRAIN", f"Found {len(file_paths)} valid pairs.", level="success")

    # Shuffle & Split
    random.shuffle(file_paths)
    split_ratio = cfg['data'].get('train_split', 0.8)
    split_idx = int(len(file_paths) * split_ratio)
    train_paths = file_paths[:split_idx]
    val_paths = file_paths[split_idx:]

    # Preprocessing Config
    skeleton_name = cfg['data']['skeleton']
    skeleton_def = get_skeleton_by_name(skeleton_name)

    features_cfg = cfg['data'].get('features', {})

    preprocess_args = {
        "skeleton_definition": skeleton_def,
        "confidence_threshold": cfg['preprocessing'].get('confidence_threshold', 0.4),
        "exclude_ratio": cfg['preprocessing'].get('exclude_ratio', 0.1),
        "min_segment_length": cfg['preprocessing'].get('min_segment_length', 60),
        "outlier_ratio": cfg['preprocessing'].get('outlier_ratio', 0.2),
        "filter_cutoff": cfg['preprocessing'].get('filter_cutoff', 6),
        "filter_order": cfg['preprocessing'].get('filter_order', 4),
        "keypoints": features_cfg.get('keypoints'),
        "kinematics_keypoints": features_cfg.get('kinematics'),
        "angle_triplets": features_cfg.get('angles'),
        "distance_pairs": features_cfg.get('distances'),
    }

    # Loaders
    batch_size = cfg['training']['batch_size']
    num_workers = cfg['data'].get('num_workers', 0)

    train_dataset = GaitDataset(train_paths, preprocessing_fn=generate_features, **preprocess_args)
    val_dataset = GaitDataset(val_paths, preprocessing_fn=generate_features, **preprocess_args)

    if not train_dataset.data:
        raise RuntimeError("Train dataset empty!")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pad,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_pad, num_workers=num_workers,
                            pin_memory=True)

    # CLASS WEIGHTS
    log("TRAIN", "Calculating class weights...", level="info")
    positives = 0
    total_samples = 0
    for _, labels in train_dataset.data:
        positives += np.sum(labels)
        total_samples += labels.size

    negatives = total_samples - positives
    pos_weight = negatives / positives if positives > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    log("TRAIN", f"Pos Weight: {pos_weight:.2f}", level="info")

    # Adjacency matrix flag
    if cfg['data'].get('requires_adj_matrix', False):
        log("TRAIN", "'requires_adj_matrix' is True -> Injecting feature config to model.", level="info")

        features_cfg = cfg['data']['features']

        # Requirements: Config must not contain angles or distances
        if features_cfg.get('angles') or features_cfg.get('distances'):
            raise ValueError("[Config] ERROR: 'requires_adj_matrix=true' does not support 'angles' or 'distances'.")

        # Inject parameters into model - so model can build adjacency matrix
        if 'params' not in cfg['model']:
            cfg['model']['params'] = {}

        cfg['model']['params']['features_config'] = features_cfg
        cfg['model']['params']['skeleton_name'] = cfg['data']['skeleton']

    # BUILD MODEL
    input_size = train_dataset.data[0][0].shape[1]
    log("TRAIN", f"Building '{cfg['model']['type']}' (Input: {input_size})", level="info")
    model = build_model(cfg['model'], input_size=input_size)

    # TRAINING
    optimizer = build_optimizer(model, cfg['training'])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    scheduler = build_scheduler(
        optimizer,
        cfg['training'],
        steps_per_epoch=len(train_loader)
    )

    # Result path
    model_save_path = os.path.join(output_dir, "best_model.pth")
    num_epochs = cfg['training']['epochs']

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, device, tolerance_frames)

    # Run training
    history = trainer.fit(num_epochs=num_epochs, save_path=model_save_path)

    # SAVE RESULTS
    save_history(history, output_dir)
    plot_training_curves(history, output_dir)
    log("TRAIN", f"Experiment '{experiment_name}' finished.", level="success")
    log("TRAIN", f"Results saved in: {output_dir}", level="info")


if __name__ == "__main__":
    main()
