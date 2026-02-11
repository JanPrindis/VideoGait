"""
This module provides functionality to build pose detectors from configuration files.

It uses the detector registry to instantiate the correct detector class based on the
'type' field specified in the YAML configuration.
"""
import yaml
import os
import detectors
from utils.registry import POSE_DETECTORS
from utils.logger import log


def build_detector_from_file(config_path: str):
    """
    Instantiates a pose detector based on a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
                           The file must contain a 'type' field (e.g., 'RTMLib', 'AlphaPose')
                           and optionally a 'params' dictionary.

    Returns:
        BaseDetector: An instance of the requested pose detector.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is missing the 'type' field.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Detector config not found at: {config_path}")

    with open(config_path, 'r') as f:
        cfg_content = yaml.safe_load(f)

    detector_type = cfg_content.get("type")
    if not detector_type:
        raise ValueError(f"Config file {config_path} missing 'type' field.")

    detector_class = POSE_DETECTORS.get(detector_type)

    params = cfg_content.get("params", {})

    log("BUILDER", f"Building detector: {detector_type} from {config_path}", level="info")
    return detector_class(params)
