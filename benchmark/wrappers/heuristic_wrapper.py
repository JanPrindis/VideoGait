"""
This module provides a wrapper for the Heuristic-based gait event detector.

It adapts the standalone inference pipeline to the interface expected by the benchmark engine.
"""
import os
import sys

from utils.config_models import AppConfig, PreprocessingConfig, PoseDetectorRef, EventDetectorConfig, HeuristicConfig, \
    OutputConfig

# Path hack
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gaitDetectHeuristic.tools.inference import run_heuristic_inference

class HeuristicWrapper:
    """
    Wraps the inference pipeline to provide a unified interface for the benchmark.
    """
    def __init__(self, preprocessing_cfg: PreprocessingConfig, heuristic_cfg: HeuristicConfig):
        self.preprocessing_cfg = preprocessing_cfg
        self.heuristic_cfg = heuristic_cfg

    def predict(self, input_path):
        """
        Runs the inference pipeline on a single input file.

        Args:
            input_path (str): Path to the keypoints JSON file.

        Returns:
            dict | None: The result dictionary containing predictions and events, or None if failed.
        """
        mock_config = AppConfig(
            preprocessing=self.preprocessing_cfg,
            pose_detector=PoseDetectorRef(config_path="mock/path.yaml"),
            event_detector=EventDetectorConfig(
                method="Heuristic",
                heuristic=HeuristicConfig(
                    method=self.heuristic_cfg.method,
                    skeleton=self.heuristic_cfg.skeleton,
                    params=self.heuristic_cfg.params
                )
            ),
            output=OutputConfig(
                save_video=False,
                save_report=False
            )
        )

        # Output dir = None -> we are not saving the analysis
        return run_heuristic_inference(mock_config, input_path, output_dir=None)
