"""
This module provides functionality to build pose detectors from configuration files.

It uses the detector registry to instantiate the correct detector class based on the
'type' field specified in the YAML configuration.
"""
import os
import detectors
from utils.config_models import DetectorConfig
from utils.config_utils import load_and_validate_yaml
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

    try:
        cfg_content: DetectorConfig = load_and_validate_yaml(config_path, DetectorConfig)
    except ValueError as e:
        log("CONFIG", str(e), level="error")
        return None

    detector_type = cfg_content.type
    detector_class = POSE_DETECTORS.get(detector_type)
    params = cfg_content.params

    log("BUILDER", f"Building detector: {detector_type} from {config_path}", level="info")
    return detector_class(params)
