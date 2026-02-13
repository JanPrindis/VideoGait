"""
This module provides utility functions for handling and resolving configuration parameters.

It specifically handles the logic for determining which skeleton definition to use
based on the selected event detection method (Heuristic vs NeuralNet) or pose detector configuration.
"""
import os
import yaml
from typing import Union
from pydantic import ValidationError

from skeletons import get_skeleton_by_name
from utils.config_models import AppConfig, BenchmarkConfig, TrainConfig, DetectorConfig

ConfigType = Union[AppConfig, BenchmarkConfig, TrainConfig]

def load_and_validate_yaml(path: str, model_class):
    """
    Loads a YAML file and validates it against a Pydantic model.

    Args:
        path (str): The file path to the YAML configuration.
        model_class (Type[BaseModel]): The Pydantic model class to validate against.

    Returns:
        BaseModel: An instance of the provided model class populated with data from the YAML file.

    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
        ValueError: If the YAML content fails validation against the model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r') as f:
        raw_dict = yaml.safe_load(f)

    try:
        return model_class(**raw_dict)
    except ValidationError as e:
        raise ValueError(f"Validation failed for '{path}':\n{e}")


def resolve_skeleton_name_from_config(config: ConfigType) -> str:
    """
    Extracts the name of the skeleton (e.g., 'HALPE', 'COCO') from the configuration.
    Args:
        config (Union[AppConfig, BenchmarkConfig, TrainConfig]): The validated configuration object.

    Returns:
        str: The uppercase name of the skeleton.

    Raises:
        ValueError: If the skeleton name cannot be found.
    """
    skeleton_name = None

    # AppConfig -> Get skeleton definition from pose detector
    if isinstance(config, AppConfig):
        if config.event_detector.method == "Heuristic" and config.event_detector.heuristic and config.event_detector.heuristic.skeleton:
            skeleton_name = config.event_detector.heuristic.skeleton
        else:
            pose_cfg_path = config.pose_detector.config_path
            try:
                detector_cfg: DetectorConfig = load_and_validate_yaml(pose_cfg_path, DetectorConfig)
                skeleton_name = detector_cfg.skeleton
            except FileNotFoundError:
                raise ValueError(
                    f"Cannot resolve skeleton: Pose detector config '{pose_cfg_path}' not found, and no fallback skeleton provided in event_detector.")

    elif isinstance(config, BenchmarkConfig):
        if config.event_detector.method == "Heuristic" and config.event_detector.heuristic:
            skeleton_name = config.event_detector.heuristic.skeleton

    # TrainingConfig -> Contains skeleton config
    elif isinstance(config, TrainConfig):
        skeleton_name = config.data.skeleton

    # Fallback / Error
    if not skeleton_name:
        raise ValueError("Unable to find 'skeleton' definition in the provided config object!")

    return skeleton_name.upper()


def resolve_skeleton_from_config(config: ConfigType):
    """
    Resolves the full SkeletonDefinition object based on the configuration.

    Args:
        config (Union[AppConfig, BenchmarkConfig, TrainConfig]): The validated configuration object.

    Returns:
        SkeletonDefinition: The resolved skeleton definition object containing keypoints and links.
    """
    skeleton_name = resolve_skeleton_name_from_config(config)
    return get_skeleton_by_name(skeleton_name)


def build_preprocess_args_from_train_config(train_cfg: TrainConfig) -> dict:
    """
    Constructs a dictionary of preprocessing arguments from the training configuration.

    This includes the skeleton definition, filtering parameters, and feature selection
    settings required for the data preprocessing pipeline.

    Args:
        train_cfg (TrainConfig): The training configuration object.

    Returns:
        dict: A dictionary containing parameters for the preprocessing pipeline.
    """
    skeleton_def = get_skeleton_by_name(train_cfg.data.skeleton)
    feat_def = train_cfg.data.features

    return {
        "skeleton_definition": skeleton_def,
        "confidence_threshold": train_cfg.preprocessing.confidence_threshold,
        "exclude_ratio": train_cfg.preprocessing.exclude_ratio,
        "min_segment_length": train_cfg.preprocessing.min_segment_length,
        "outlier_ratio": train_cfg.preprocessing.outlier_ratio,
        "filter_cutoff": train_cfg.preprocessing.filter_cutoff,
        "filter_order": train_cfg.preprocessing.filter_order,

        # Features
        "keypoints": feat_def.keypoints,
        "kinematics_keypoints": feat_def.kinematics,
        "angle_triplets": feat_def.angles,
        "distance_pairs": feat_def.distances,
    }
