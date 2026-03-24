import argparse
import sys
import os
from glob import glob

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.config_utils import load_and_validate_yaml, build_preprocess_args_from_train_config
from utils.config_models import TrainConfig
from gaitDetectNN.builder import build_model
from utils.preprocessing import generate_features


def main():
    parser = argparse.ArgumentParser(description="Count model parameters")
    parser.add_argument("--config", required=True, help="Path to train config")
    args = parser.parse_args()

    cfg = load_and_validate_yaml(args.config, TrainConfig)

    dataset_root = os.path.join(PROJECT_ROOT, cfg.data.dataset_root)
    framerate = cfg.data.framerate
    
    search_pattern = os.path.join(dataset_root, str(int(framerate)), "KEYPOINTS", "*.json")
    sample_files = glob(search_pattern, recursive=True)
    
    if not sample_files:
        print(f"Error: No keypoint files found in {search_pattern} to determine input_size.")
        sys.exit(1)

    preprocess_args = build_preprocess_args_from_train_config(cfg)
    
    if cfg.data.requires_adj_matrix:
        cfg.model.params['features_config'] = cfg.data.features.model_dump()
        cfg.model.params['skeleton_name'] = cfg.data.skeleton

    feature_matrices, _, _ = generate_features(
        keypoints_path=sample_files[0],
        annotations_path=None,
        frame_rate=framerate,
        **preprocess_args
    )

    if not feature_matrices:
        print("Error: Could not generate features from the sample file.")
        sys.exit(1)

    input_size = feature_matrices[0].shape[1]
    print(f"Resolved input_size:  {input_size} (from {os.path.basename(sample_files[0])})")

    print(f"Building: {cfg.model.type}")
    model = build_model(cfg.model, input_size=input_size)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-" * 40)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print("-" * 40)


if __name__ == "__main__":
    main()

    