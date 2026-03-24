import argparse
import sys
import os
import torch
import matplotlib.pyplot as plt
from glob import glob

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.config_utils import load_and_validate_yaml, build_preprocess_args_from_train_config
from utils.config_models import TrainConfig
from utils.preprocessing import generate_features

def main():
    parser = argparse.ArgumentParser(description="Visualize Data Augmentation Noise")
    parser.add_argument("--config", required=True, help="Path to train config")
    args = parser.parse_args()

    # Load config
    cfg = load_and_validate_yaml(args.config, TrainConfig)
    dataset_root = os.path.join(PROJECT_ROOT, cfg.data.dataset_root)
    framerate = cfg.data.framerate
    noise_std = cfg.training.noise_std
    
    search_pattern = os.path.join(dataset_root, str(int(framerate)), "KEYPOINTS", "*.json")
    sample_files = glob(search_pattern, recursive=True)
    
    if not sample_files:
        print(f"Error: No files found in {search_pattern}")
        sys.exit(1)

    # Load Preprocessing Args
    preprocess_args = build_preprocess_args_from_train_config(cfg)
    
    # Get Feature Matrix for the first file
    feature_matrices, _, _ = generate_features(
        keypoints_path=sample_files[0],
        annotations_path=None,
        frame_rate=framerate,
        **preprocess_args
    )

    if not feature_matrices:
        print("Error: Could not generate features.")
        sys.exit(1)

    # Get the first clip
    features_raw = torch.tensor(feature_matrices[0], dtype=torch.float32)
    
    # Apply exactly the same noise as Trainer does
    noise = torch.randn_like(features_raw) * noise_std
    features_noisy = features_raw + noise

    # --- Plotting ---
    # Pick 3 random features to plot
    features_to_plot = [0, 10, min(20, features_raw.shape[1] - 1)]
    
    fig, axs = plt.subplots(len(features_to_plot), 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Data Augmentation Noise (std = {noise_std}) on Normalized Features", fontsize=14)

    for i, f_idx in enumerate(features_to_plot):
        ax = axs[i]
        ax.plot(features_raw[:, f_idx].numpy(), label="Original Signal", color='blue', linewidth=2)
        ax.plot(features_noisy[:, f_idx].numpy(), label="Noisy Signal (To Model)", color='red', alpha=0.7, linestyle='--')
        ax.set_title(f"Feature Index {f_idx}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.xlabel("Frame")
    plt.tight_layout()
    plt.savefig("noise_visualization.png", dpi=150)
    print("Plot saved to noise_visualization.png")

if __name__ == "__main__":
    main()
