from .registry import HEURISTICS

from .. import methods

def build_heuristic_detector(full_config):
    heuristic_cfg = full_config.get('event_detector', {}).get('heuristic', {})
    method_name = heuristic_cfg.get('method')

    if not method_name:
        raise ValueError("Config missing 'event_detector.heuristic.method'")

    detector_cls = HEURISTICS.get(method_name)

    return detector_cls(full_config)
