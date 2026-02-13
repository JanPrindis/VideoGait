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
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

# --- IMPORTS ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from gaitDetectNN.builder import build_model
from gaitDetectNN.engine.trainer import Trainer
from loaders import GaitDataset, collate_pad
from utils.preprocessing import generate_features
from utils.data import find_matching_annotation
from train import build_scheduler
from utils.logger import log
from utils.config_models import TrainConfig
from utils.config_utils import load_and_validate_yaml, build_preprocess_args_from_train_config


# ==============================================================================
# SEARCH SPACE REGISTRY
# ==============================================================================

def suggest_bilstm_params(trial):
    return {
        # Architecture
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 768]),
        # "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.2, 0.6),
        "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),

        # Training
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    }

def suggest_gru_params(trial):
    return {
        # Architecture
        # "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 768]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
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
    num_layers = trial.suggest_int("num_layers", 2, 5)
    channel_size = trial.suggest_categorical("channel_size", [64, 128, 256, 512])
    # channel_size = trial.suggest_categorical("channel_size", [128, 256, 512])

    # Create a list of channels, for example [64, 64, 64]
    num_channels = [channel_size] * num_layers

    return {
        # Architecture
        "num_channels": num_channels,
        "kernel_size": trial.suggest_categorical("kernel_size", [5, 7, 9, 11]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),

        # Training
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    }

def suggest_transformer_params(trial):
    # Get model width (d_model)
    d_model = trial.suggest_categorical("d_model", [32, 64])

    # Pick number of heads based on model width
    if d_model == 64:
        n_head = trial.suggest_categorical("n_head_64", [2, 4])
    elif d_model == 128:
        n_head = trial.suggest_categorical("n_head_128", [2, 4, 8])
    else:  # 256
        n_head = trial.suggest_categorical("n_head_256", [4, 8])

    # Kernel Size (Feature Tokenizer)
    kernel_size = trial.suggest_categorical("kernel_size", [5, 9, 11, 15])
    padding = kernel_size // 2

    return {
        # Architecture Params
        "d_model": d_model,
        "n_head": n_head,
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        "kernel_size": kernel_size,
        "padding": padding,

        # Training Params
        "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    }


def suggest_stgcn_params(trial):
    return {
        # Architecture
        "hidden_channels": trial.suggest_categorical("hidden_channels", [32, 64, 128]),
        # "hidden_channels": trial.suggest_categorical("hidden_channels", [128]),
        "num_layers": trial.suggest_int("num_layers", 4, 9),
        # "tcn_kernel_size": trial.suggest_categorical("tcn_kernel_size", [15]),
        "tcn_kernel_size": trial.suggest_categorical("tcn_kernel_size", [7, 9, 15, 21]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),

        # Graph Strategy
        "graph_strategy": trial.suggest_categorical("graph_strategy", ["uniform", "spatial"]),

        # Training
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    }

SEARCH_SPACES = {
    #"GaitLSTM": suggest_lstm_params,
    "GaitBiLSTM": suggest_bilstm_params,
    "GaitBiGRU": suggest_gru_params,
    "GaitTCN": suggest_tcn_params,
    "GaitTransformer": suggest_transformer_params,
    "GaitSTGCN": suggest_stgcn_params,
}


# ==============================================================================
# OBJECTIVE FUNCTION
# ==============================================================================

def objective(trial, base_cfg: TrainConfig, train_loader, val_loader, input_size, pos_weight, device, f1_window_frame):
    model_type = base_cfg.model.type

    # Get tuning parameters
    if model_type not in SEARCH_SPACES:
        raise ValueError(f"Model '{model_type}' does not have defined search space inside tune.py!")

    suggested_params = SEARCH_SPACES[model_type](trial)

    # Separate optimizer parameters from model
    lr = suggested_params.pop("lr", 0.001)
    weight_decay = suggested_params.pop("weight_decay", 0.01)

    current_cfg = base_cfg.model_copy(deep=True)
    current_cfg.model.params.update(suggested_params)

    if current_cfg.data.requires_adj_matrix:
        current_cfg.model.params['features_config'] = current_cfg.data.features.model_dump()
        current_cfg.model.params['skeleton_name'] = current_cfg.data.skeleton

    # Build Model
    model = build_model(current_cfg.model, input_size)

    # Setup Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Use pre-calculated weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training
    tuning_epochs = 15

    # Config hack
    current_cfg.training.learning_rate = lr
    current_cfg.training.epochs = tuning_epochs

    scheduler = build_scheduler(
        optimizer,
        current_cfg.training,
        steps_per_epoch=len(train_loader)
    )

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, device, f1_window_frame)

    try:
        val_f1 = None
        for epoch in range(tuning_epochs):
            train_loss, train_f1 = trainer.train_epoch()
            val_loss, val_f1 = trainer.validate_epoch()

            # Epoch based schedulers stepping
            if scheduler is not None and not isinstance(scheduler, OneCycleLR):
                scheduler.step()

            # Report result to Optuna
            trial.report(val_f1, epoch)

            # Pruning - if current run is bad, stop it
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_f1

    except RuntimeError as e:
        log("TUNE", f"Trial failed: {e}", level="error")
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
    cfg: TrainConfig = load_and_validate_yaml(args.config, TrainConfig)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("TUNE", f"Tuning: {cfg.model.type} | Device: {device}", level="info")

    # --------------------------------------------------------------------------
    # DATASET
    # --------------------------------------------------------------------------
    dataset_root = os.path.join(PROJECT_ROOT, cfg.data.dataset_root)
    annotation_root = os.path.join(PROJECT_ROOT, cfg.data.annotation_root)
    framerate = cfg.data.framerate

    # F1 Tolerance Window
    tolerance_ms = cfg.training.f1_window_size_ms
    tolerance_frames = int(round((tolerance_ms / 1000.0) * framerate))

    # Find files
    search_pattern = os.path.join(dataset_root, str(int(framerate)), "KEYPOINTS", "*.json")
    keypoint_files = glob(search_pattern, recursive=True)

    file_paths = []
    for kp_path in keypoint_files:
        ann_path = find_matching_annotation(kp_path, annotation_root)
        if ann_path and os.path.exists(ann_path):
            file_paths.append((kp_path, ann_path))

    # Shuffle & Split
    random.seed(cfg.training.seed)
    random.shuffle(file_paths)
    split_idx = int(len(file_paths) * cfg.data.train_split)
    train_paths = file_paths[:split_idx]
    val_paths = file_paths[split_idx:]

    # Preprocessing Config
    preprocess_args = build_preprocess_args_from_train_config(cfg)

    log("TUNE", "Loading datasets...", level="info")
    train_dataset = GaitDataset(train_paths, preprocessing_fn=generate_features, **preprocess_args)
    val_dataset = GaitDataset(val_paths, preprocessing_fn=generate_features, **preprocess_args)

    if not train_dataset.data:
        raise RuntimeError("Train dataset empty!")

    # DataLoaders
    batch_size = cfg.training.batch_size
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
    log("TUNE", f"Calculated Pos Weight: {pos_weight_val:.2f}", level="info")

    # Determine input size
    input_size = train_dataset.data[0][0].shape[1]

    # --------------------------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------------------------
    log("TUNE", f"Starting Optuna Study: {args.study_name} ({args.trials} trials)", level="info")

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
    log("TUNE", f"TUNING FINISHED. Best F1: {study.best_value:.4f}", level="success")
    log("TUNE", "Best Params:", level="info")
    for key, value in study.best_params.items():
        log("TUNE", f"  {key}: {value}", level="info")

    # Save best parameters into config file
    output_path = os.path.join(os.path.dirname(args.config), "best_params.yaml")

    best_config_dump = {
        "best_f1": float(study.best_value),
        "params": study.best_params
    }

    with open(output_path, 'w') as f:
        yaml.dump(best_config_dump, f)

    log("TUNE", f"Best parameters saved to: {output_path}", level="success")


if __name__ == "__main__":
    main()
