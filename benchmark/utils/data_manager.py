"""
This module handles data loading and splitting for the benchmarking pipeline.

It ensures that the benchmark runs on the correct subset of data (validation set)
consistent with how the model was trained, or on the full dataset if requested.
"""
import os
import sys
import yaml
import random
from glob import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data import find_matching_annotation


def get_data_settings(cfg):
    """
    Resolves data configuration parameters based on the detection method.

    If the method is 'NeuralNet', it loads the original training configuration
    to ensure the benchmark uses the exact same data split and framerate as training.
    If 'Heuristic', it uses parameters directly from the benchmark config.

    Args:
        cfg (dict): The benchmark configuration dictionary.

    Returns:
        dict: A dictionary containing:
            - dataset_root (str): Path to the dataset root.
            - annotation_root (str): Path to the annotations root.
            - use_full (bool): Whether to use the full dataset.
            - framerate (int): The target framerate.
            - seed (int): Random seed for splitting.
            - split_ratio (float): The train/val split ratio used.
    """
    method = cfg['event_detector']['method']
    data_root = os.path.join(PROJECT_ROOT, cfg['data']['dataset_root'])

    settings = {
        "dataset_root": data_root,
        "annotation_root": os.path.join(PROJECT_ROOT, cfg['data']['annotation_root']),
        "use_full": cfg['data'].get('use_full_dataset', False)
    }

    # For NN, pull the config from training config, so it matches what the model expects
    if method == "NeuralNet":
        nn_cfg = cfg['event_detector']['neural_net']
        exp_path = os.path.join(PROJECT_ROOT, nn_cfg['experiment_path'])
        train_cfg_path = os.path.join(exp_path, "config.yaml")

        if not os.path.exists(train_cfg_path):
            raise FileNotFoundError(f"Training config missing for NeuralNet: {train_cfg_path}")

        print(f"[Data] Auto-resolving data settings from experiment: {nn_cfg['experiment_path']}")
        with open(train_cfg_path) as f:
            train_cfg = yaml.safe_load(f)

        settings["framerate"] = train_cfg['data']['framerate']
        settings["seed"] = train_cfg['training']['seed']
        settings["split_ratio"] = train_cfg['data'].get('train_split', 0.8)

    # For heuristic methods, pull the config from the config file
    elif method == "Heuristic":
        heuristics_cfg = cfg['event_detector']['heuristic']
        print(f"[Data] Using explicit settings from Heuristic config.")

        settings["framerate"] = cfg['preprocessing']['framerate']
        settings["seed"] = heuristics_cfg['seed']
        settings["split_ratio"] = heuristics_cfg['train_split']

    return settings


def get_benchmark_files(cfg):
    """
    Retrieves the list of file pairs (keypoints + annotations) for benchmarking.

    This function replicates the random split used during training to ensure
    that the benchmark runs on the 'validation' portion of the data (unseen during training),
    unless 'use_full_dataset' is enabled.

    Args:
        cfg (dict): The benchmark configuration dictionary.

    Returns:
        tuple: A tuple containing:
            - list[dict]: A list of dictionaries, each with keys 'kp' (keypoints path),
                          'ann' (annotation path), and 'name' (filename).
            - int: The global framerate resolved from settings.
    """
    settings = get_data_settings(cfg)

    # Find files based on FPS
    # The expected file structure is DATASET_ROOT/{FPS}/KEYPOINTS/file.json
    fps = settings["framerate"]
    search_pattern = os.path.join(settings["dataset_root"], str(fps), "KEYPOINTS", "*.json")

    print(f"[Data] Searching: {search_pattern}")
    files = glob(search_pattern, recursive=True)

    # Find matching annotations
    paired_files = []
    for kp in files:
        ann = find_matching_annotation(kp, settings["annotation_root"])
        if ann and os.path.exists(ann):
            paired_files.append({"kp": kp, "ann": ann, "name": os.path.basename(kp)})

    print(f"[Data] Found {len(paired_files)} valid pairs.")

    # If using the full dataset, return
    if settings["use_full"]:
        print("[Data] Using FULL dataset.")
        return paired_files, settings["framerate"]

    # Otherwise replicate the training/validation split
    seed = settings["seed"]
    ratio = settings["split_ratio"]

    random.seed(seed)
    random.shuffle(paired_files)

    split_idx = int(len(paired_files) * ratio)

    # First part is Training, second is Validation
    test_files = paired_files[split_idx:]


    for file in test_files:
        print(f"[DATA] Filtered file: {file['name']}")


    print(f"[Data] Using VALIDATION split (Seed: {seed}, Ratio: {ratio})")
    print(f"       -> Train set (ignored): {split_idx}")
    print(f"       -> Test set (used):     {len(test_files)}")

    return test_files, settings["framerate"]
