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
    # --- PHASE 1: COARSE SEARCH ---
    # "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
    # "num_layers": trial.suggest_int("num_layers", 1, 3),
    # "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),
    # "dropout": trial.suggest_float("dropout", 0.1, 0.7),
    # "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
    # "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    # ----------------------------------------------

    # --- PHASE 2: FINE-TUNING ---
    return {
        "hidden_dim": 512,
        "num_layers": 3,
        "dense_units": 128,

        "dropout": trial.suggest_float("dropout", 0.4, 0.6),
        "lr": trial.suggest_float("lr", 5e-4, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-2, log=True)
    }

def suggest_gru_params(trial):
    # --- PHASE 1: COARSE SEARCH ---
    # "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
    # "num_layers": trial.suggest_int("num_layers", 1, 3),
    # "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),
    # "dropout": trial.suggest_float("dropout", 0.1, 0.7),
    # "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
    # "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    # ----------------------------------------------

    # --- PHASE 2: FINE-TUNING ---
    return {
        "hidden_dim": 512,
        "num_layers": 3,
        "dense_units": 128,

        "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        "lr": trial.suggest_float("lr", 2e-4, 8e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)
    }

def suggest_tcn_params(trial):
    # --- PHASE 1: COARSE SEARCH ---
    # num_layers = trial.suggest_int("num_layers", 2, 6)
    # channel_size = trial.suggest_categorical("channel_size", [32, 64, 128, 256])
    # num_channels = [channel_size] * num_layers
    # return {
    #     "num_channels": num_channels,
    #     "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7, 9, 11]),
    #     "dropout": trial.suggest_float("dropout", 0.1, 0.6),
    #     "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
    #     "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    # }
    # ----------------------------------------------

    # --- PHASE 2: FINE-TUNING ---
    num_layers = 6
    channel_size = 128
    num_channels = [channel_size] * num_layers

    return {
        "num_channels": num_channels,
        "kernel_size": 11,

        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "lr": trial.suggest_float("lr", 2e-3, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 2e-4, log=True)
    }

def suggest_transformer_params(trial):
    # --- PHASE 1: COARSE SEARC ---
    # "d_model": trial.suggest_categorical("d_model", [16, 32, 64, 128]),
    # "num_layers": trial.suggest_int("num_layers", 1, 4),
    # "dim_feedforward": trial.suggest_categorical("dim_feedforward", [32, 64, 128, 256]),
    # "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 9, 15]),
    # "dropout": trial.suggest_float("dropout", 0.1, 0.6),
    # "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
    # "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    # ----------------------------------------------

    # --- PHASE 1.1: SHIFTED COARSE SEARCH ---
    # "d_model": trial.suggest_categorical("d_model", [128, 256]),
    # "num_layers": trial.suggest_int("num_layers", 2, 4),
    # "dim_feedforward": trial.suggest_categorical("dim_feedforward", [128, 256, 512]),
    # "kernel_size": trial.suggest_categorical("kernel_size", [11, 15, 21, 31]),
    # "dropout": trial.suggest_float("dropout", 0.05, 0.25),
    # "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
    # "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)
    # ----------------------------------------------

    # --- PHASE 2: FINE-TUNING  ---
    d_model = 256
    n_head = 8
    num_layers = 2
    dim_feedforward = 512
    kernel_size = 21
    padding = kernel_size // 2
    
    return {
        "d_model": d_model,
        "n_head": n_head,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "kernel_size": kernel_size,
        "padding": padding,

        "dropout": trial.suggest_float("dropout", 0.05, 0.15),
        "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-5, log=True)
    }


def suggest_stgcn_params(trial):
    # --- ST-GCN KEYPOINTS ---
    # --- PHASE 1: COARSE SEARCH ---
    # "hidden_channels": trial.suggest_categorical("hidden_channels", [16, 32, 64, 128]),
    # "num_layers": trial.suggest_int("num_layers", 2, 8),
    # "tcn_kernel_size": trial.suggest_categorical("tcn_kernel_size", [5, 9, 15, 21]),
    # "dropout": trial.suggest_float("dropout", 0.1, 0.6),
    # "graph_strategy": trial.suggest_categorical("graph_strategy", ["uniform", "spatial"]),
    # "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
    # "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    # ----------------------------------------------

    # --- PHASE 2: FINE-TUNING (ST-GCN Keypoints)  ---
    # "hidden_channels": 64, "num_layers": 7, "tcn_kernel_size": 9, "graph_strategy": "uniform"
    # "dropout": [0.25, 0.5], "lr": [2e-3, 5e-3], "weight_decay": [1e-3, 1e-2]
    # ----------------------------------------------

    # --- ST-GCN KINEMATICS ---
    # --- PHASE 1: COARSE SEARCH ---
    # "hidden_channels": trial.suggest_categorical("hidden_channels", [16, 32, 64, 128]),
    # "num_layers": trial.suggest_int("num_layers", 2, 8),
    # "tcn_kernel_size": trial.suggest_categorical("tcn_kernel_size", [5, 9, 15, 21]),
    # "dropout": trial.suggest_float("dropout", 0.1, 0.6),
    # "graph_strategy": trial.suggest_categorical("graph_strategy", ["uniform", "spatial"]),
    # "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
    # "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    # ----------------------------------------------

    # --- PHASE 2: FINE-TUNING (ST-GCN Kinematics) ---
    return {
        "hidden_channels": 128,
        "num_layers": 3,
        "tcn_kernel_size": 21,
        "graph_strategy": "uniform",

        "dropout": trial.suggest_float("dropout", 0.15, 0.35),
        "lr": trial.suggest_float("lr", 2e-3, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)
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

    trainer = Trainer(
        model, train_loader, val_loader, criterion, 
        optimizer, scheduler, device, f1_window_frame,
        noise_std=current_cfg.training.noise_std
    )

    try:
        best_val_loss = float('inf')
        best_val_f1 = 0.0

        for epoch in range(tuning_epochs):
            train_loss, train_f1 = trainer.train_epoch()
            val_loss, val_f1 = trainer.validate_epoch()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_f1 = val_f1

            # Epoch based schedulers stepping
            if scheduler is not None and not isinstance(scheduler, OneCycleLR):
                scheduler.step()

            # Report result to Optuna
            trial.report(val_loss, epoch)

            # Pruning - if current run is bad, stop it
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Save F1 score to trial attributes for later analysis
        trial.set_user_attr("val_f1", float(best_val_f1))
        return best_val_loss

    except RuntimeError as e:
        log("TUNE", f"Trial failed: {e}", level="error")
        return float('inf')


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
    total_files = len(file_paths)
    train_idx = int(total_files * cfg.data.train_split)
    val_idx = train_idx + int(total_files * cfg.data.val_split)

    train_paths = file_paths[:train_idx]
    val_paths = file_paths[train_idx:val_idx]

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
        direction="minimize",
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
    log("TUNE", "TUNING FINISHED.", level="success")

    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    
    if not complete_trials:
        log("TUNE", "No trials completed successfully.", level="warning")
        return
        
    complete_trials.sort(key=lambda t: t.value)

    top_n = min(5, len(complete_trials))
    log("TUNE", f"Top {top_n} Configurations:", level="info")
    for i in range(top_n):
        t = complete_trials[i]
        val_f1 = t.user_attrs.get("val_f1")
        f1_str = f"{val_f1:.4f}" if val_f1 is not None else "N/A"
        log("TUNE", f"  Rank {i+1} (Trial {t.number}) | Loss: {t.value:.4f} | F1: {f1_str}", level="info")
        for key, value in t.params.items():
            log("TUNE", f"    {key}: {value}", level="info")

    # Save best parameters into config file
    output_path = os.path.join(os.path.dirname(args.config), "best_params.yaml")

    best_config_dump = {
        "best_val_loss": float(study.best_value),
        "best_val_f1": float(study.best_trial.user_attrs.get("val_f1", 0.0)),
        "params": study.best_params
    }

    with open(output_path, 'w') as f:
        yaml.dump(best_config_dump, f)

    log("TUNE", f"Best parameters saved to: {output_path}", level="success")


if __name__ == "__main__":
    main()
