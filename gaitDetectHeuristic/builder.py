from utils.config_models import AppConfig
from utils.registry import HEURISTICS
import gaitDetectHeuristic.methods


def build_heuristic_detector(app_config: AppConfig):
    """
    Builds and returns a heuristic detector instance based on the provided configuration.

    Args:
        app_config (AppConfig): The application configuration object.

    Returns:
        The instantiated heuristic detector class.

    Raises:
        ValueError: If the configuration is missing 'event_detector.heuristic.method'.
    """
    heuristic_cfg = app_config.event_detector.heuristic
    method_name = heuristic_cfg.method

    if not method_name:
        raise ValueError("Config missing 'event_detector.heuristic.method'")

    detector_cls = HEURISTICS.get(method_name)

    return detector_cls(app_config)
