"""
This module provides utility functions for handling and resolving configuration parameters.

It specifically handles the logic for determining which skeleton definition to use
based on the selected event detection method (Heuristic vs NeuralNet) or pose detector configuration.
"""
import os
import yaml
from skeletons import get_skeleton_by_name

def resolve_skeleton_name_from_config(config: dict) -> str:
    """
    Extracts the name of the skeleton (e.g., 'HALPE', 'COCO') from the configuration.

    It searches in the following order:
    1. Pose Detector config (if linked in app config).
    2. Heuristic event detector config.
    3. Neural Network training data config.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        str: The uppercase name of the skeleton.

    Raises:
        ValueError: If the skeleton name cannot be found in any expected location.
    """
    skeleton_name = None

    # App config - try to find pose detector config
    pose_cfg_path = config.get('pose_detector', {}).get('config_path')
    if pose_cfg_path and os.path.exists(pose_cfg_path):
        with open(pose_cfg_path, 'r') as f:
            pose_cfg = yaml.safe_load(f)
            skeleton_name = pose_cfg.get('skeleton')

    # Heuristic benchmark
    if not skeleton_name:
        skeleton_name = config.get('event_detector', {}).get('heuristic', {}).get('skeleton')

    # Train/benchmark NeuralNet
    if not skeleton_name:
        skeleton_name = config.get('data', {}).get('skeleton')

    # 4. Fallback / Error
    if not skeleton_name:
        raise ValueError("Unable to find 'skeleton' definition in config file!")

    return skeleton_name.upper()


def resolve_skeleton_from_config(config: dict):
    """
    Resolves the full SkeletonDefinition object based on the configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        SkeletonDefinition: The resolved skeleton definition object containing keypoints and links.
    """
    skeleton_name = resolve_skeleton_name_from_config(config)
    return get_skeleton_by_name(skeleton_name)
