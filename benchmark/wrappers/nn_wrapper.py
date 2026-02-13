"""
This module provides a wrapper for the Neural Network-based gait event detector.

It adapts the standalone inference pipeline to the interface expected by the benchmark engine.
"""
import os
import sys

from utils.config_models import PreprocessingConfig, NeuralNetConfig, AppConfig, PoseDetectorRef, EventDetectorConfig, \
    OutputConfig

# Path hack
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gaitDetectNN.tools.inference import run_nn_inference


class NeuralNetWrapper:
    """
    Wraps the inference pipeline to provide a unified interface for the benchmark.
    """
    def __init__(self, preprocessing_cfg: PreprocessingConfig, neural_net_cfg: NeuralNetConfig):
        self.preprocessing_cfg = preprocessing_cfg
        self.nn_cfg = neural_net_cfg

    def predict(self, input_path):
        """
        Runs the inference pipeline on a single input file.

        Args:
            input_path (str): Path to the keypoints JSON file.

        Returns:
            dict | None: The result dictionary containing predictions and events, or None if failed.
        """
        # Create "Mock" app config file that required for inference
        mock_config = AppConfig(
            preprocessing=self.preprocessing_cfg,
            pose_detector=PoseDetectorRef(config_path="mock/path.yaml"),
            event_detector=EventDetectorConfig(
                method="NeuralNet",
                neural_net=self.nn_cfg
            ),
            output=OutputConfig(
                save_video=False,
                save_report=False
            )
        )

        # Output dir = None -> we are not saving the analysis
        return run_nn_inference(mock_config, input_path, output_dir=None)
