import os
import sys

# Path hack
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gaitDetectHeuristic.tools.inference import run_heuristic_inference

class HeuristicWrapper:
    def __init__(self, preprocessing_cfg, heuristic_cfg):
        self.preprocessing_cfg = preprocessing_cfg
        self.heuristic_cfg = heuristic_cfg

    def predict(self, input_path):
        mock_config = {
            "preprocessing": self.preprocessing_cfg,
            "event_detector": {
                "method": "Heuristic",
                "heuristic": {
                    "method": self.heuristic_cfg.get("method", "Zeni"),
                    "skeleton": self.heuristic_cfg.get("skeleton", "HALPE"),
                    "params": self.heuristic_cfg.get("params", {})
                }
            },
            "output": {
                "save_json": False,
                "save_plot": False
            }
        }

        return run_heuristic_inference(mock_config, input_path, output_dir=None)
