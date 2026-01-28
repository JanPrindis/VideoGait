import argparse
import yaml
import sys
import os
import torch
import optuna
import numpy as np
import random
from glob import glob
from torch import nn
from torch.utils.data import DataLoader

# --- IMPORTS ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import gaitDetectNN.models
from gaitDetectNN.utils.builder import build_model
from gaitDetectNN.engine.trainer import Trainer
from dataset import GaitDataset
from collate import collate_pad
from utils.preprocessing import generate_features
from utils.data import find_matching_annotation
from skeletons import get_skeleton_by_name


# ==============================================================================
# SEARCH SPACE REGISTRY
# ==============================================================================

def suggest_bilstm_params(trial):
    return {
        # Architecture
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        # "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),

        # Training
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    }

def suggest_gru_params(trial):
    return {
        # Architecture
        # "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [512, 768, 1024]),
        "num_layers": trial.suggest_int("num_layers", 2, 3),
        # "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "dropout": trial.suggest_float("dropout", 0.12, 0.5),
        "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),

        # Training
        # "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    }

def suggest_tcn_params(trial):
    # Layer calculation
    num_layers = trial.suggest_int("num_layers", 3, 6)
    channel_size = trial.suggest_categorical("channel_size", [32, 64, 128])
    # channel_size = trial.suggest_categorical("channel_size", [128, 256, 512])

    # Create a list of channels, for example [64, 64, 64]
    num_channels = [channel_size] * num_layers

    return {
        # Architecture
        "num_channels": num_channels,
        "kernel_size": trial.suggest_categorical("kernel_size", [7, 9, 11]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),

        # Training
        "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    }

SEARCH_SPACES = {
    #"GaitLSTM": suggest_lstm_params,
    "GaitBiLSTM": suggest_bilstm_params,
    "GaitBiGRU": suggest_gru_params,
    "GaitTCN": suggest_tcn_params,
    #"GaitSTGCN": suggest_stgcn_params,
}


# ==============================================================================
# OBJECTIVE FUNCTION
# ==============================================================================

def objective(trial, base_cfg, train_loader, val_loader, input_size, pos_weight, device, f1_window_frame):
    model_type = base_cfg['model']['type']

    # Get tuning parameters
    if model_type not in SEARCH_SPACES:
        raise ValueError(f"Model '{model_type}' does not have defined search space inside tune.py!")

    suggested_params = SEARCH_SPACES[model_type](trial)

    # Separate optimizer parameters from model
    lr = suggested_params.pop("lr", 0.001)
    weight_decay = suggested_params.pop("weight_decay", 0.01)

    # Build Model
    model_params = base_cfg['model'].get('params', {}).copy()
    model_params.update(suggested_params)

    # Modify training config for build_model
    current_model_cfg = {
        "type": model_type,
        "params": model_params
    }

    model = build_model(current_model_cfg, input_size)

    # Setup Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Use pre-calculated weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, f1_window_frame)

    # Training
    tuning_epochs = 15

    try:
        val_f1 = None
        for epoch in range(tuning_epochs):
            train_loss, train_f1 = trainer.train_epoch()
            val_loss, val_f1 = trainer.validate_epoch()

            # Report result to Optuna
            trial.report(val_f1, epoch)

            # Pruning - if current run is bad, stop it
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_f1

    except RuntimeError as e:
        print(f"Trial failed: {e}")
        return 0.0


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning with Optuna")
    parser.add_argument("--config", required=True, help="Path to base YAML config")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--study_name", type=str, default="gait_optimization")
    args = parser.parse_args()

    # Load training config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Tuning: {cfg['model']['type']} | Device: {device} ---")

    # --------------------------------------------------------------------------
    # DATASET
    # --------------------------------------------------------------------------
    dataset_root = os.path.join(PROJECT_ROOT, cfg['data']['dataset_root'])
    annotation_root = os.path.join(PROJECT_ROOT, cfg['data'].get('annotation_root', 'annotations'))
    framerate = cfg['data']['framerate']

    # F1 Tolerance Window
    tolerance_ms = cfg['training'].get('f1_window_size_ms', 50)
    tolerance_frames = int(round((tolerance_ms / 1000.0) * framerate))

    # Find files
    search_pattern = os.path.join(dataset_root, str(framerate), "KEYPOINTS", "*.json")
    keypoint_files = glob(search_pattern, recursive=True)

    file_paths = []
    for kp_path in keypoint_files:
        ann_path = find_matching_annotation(kp_path, annotation_root)
        if ann_path and os.path.exists(ann_path):
            file_paths.append((kp_path, ann_path))

    # Shuffle & Split
    random.seed(cfg['training'].get('seed', 42))
    random.shuffle(file_paths)
    split_idx = int(len(file_paths) * cfg['data'].get('train_split', 0.8))
    train_paths = file_paths[:split_idx]
    val_paths = file_paths[split_idx:]

    # Preprocessing Config
    skeleton_def = get_skeleton_by_name(cfg['data']['skeleton'])
    preprocess_args = {
        "skeleton_definition": skeleton_def,
        "confidence_threshold": cfg['preprocessing'].get('confidence_threshold', 0.4),
        "exclude_ratio": cfg['preprocessing'].get('exclude_ratio', 0.1),
        "min_segment_length": cfg['preprocessing'].get('min_segment_length', 60),
        "outlier_ratio": cfg['preprocessing'].get('outlier_ratio', 0.2),
        "filter_cutoff": cfg['preprocessing'].get('filter_cutoff', 6),
        "filter_order": cfg['preprocessing'].get('filter_order', 4),
        "keypoints": cfg['data']['features']['keypoints'],
        "kinematics_keypoints": cfg['data']['features']['kinematics'],
        "angle_triplets": cfg['data']['features']['angles'],
        "distance_pairs": cfg['data']['features']['distances'],
    }

    print("Loading datasets...")
    train_dataset = GaitDataset(train_paths, preprocessing_fn=generate_features, **preprocess_args)  #
    val_dataset = GaitDataset(val_paths, preprocessing_fn=generate_features, **preprocess_args)  #

    if not train_dataset.data:
        raise RuntimeError("Train dataset empty!")

    # DataLoaders
    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pad, num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_pad, num_workers=0, pin_memory=True)

    # Calculate class weights
    positives = 0
    total_samples = 0
    for _, labels in train_dataset.data:
        positives += np.sum(labels)
        total_samples += labels.size

    negatives = total_samples - positives
    pos_weight_val = negatives / positives if positives > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"Calculated Pos Weight: {pos_weight_val:.2f}")

    # Determine input size
    input_size = train_dataset.data[0][0].shape[1]

    # --------------------------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------------------------
    print(f"\nStarting Optuna Study: {args.study_name} ({args.trials} trials)")

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )

    study.optimize(
        lambda trial: objective(trial, cfg, train_loader, val_loader, input_size, pos_weight, device, tolerance_frames),
        n_trials=args.trials
    )

    # --------------------------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"TUNING FINISHED. Best F1: {study.best_value:.4f}")
    print("=" * 50)
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best parameters into config file
    output_path = os.path.join(os.path.dirname(args.config), "best_params.yaml")

    best_config_dump = {
        "best_f1": float(study.best_value),
        "params": study.best_params
    }

    with open(output_path, 'w') as f:
        yaml.dump(best_config_dump, f)

    print(f"\nBest parameters saved to: {output_path}")


if __name__ == "__main__":
    main()
