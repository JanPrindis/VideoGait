"""
This module defines the abstract base class for all pose detectors.

It enforces a common interface (`detect`) that all specific detector implementations
(e.g., RTMLib, AlphaPose) must adhere to.
"""
from abc import ABC, abstractmethod
from pathlib import Path


class BaseDetector(ABC):
    """
    Abstract base class for pose detection wrappers.

    All specific detector implementations must inherit from this class and implement
    the `detect` method.
    """
    def __init__(self, config: dict):
        """
        Initializes the detector with a configuration dictionary.

        Args:
            config (dict): Configuration parameters specific to the detector.
        """
        self.config = config

    @abstractmethod
    def detect(self, video_path: str, output_path: str, keypoint_filename_override: str = None):
        """
        Runs pose detection on a video file.

        Args:
            video_path (str): Path to the input video file.
            output_path (str): Path to the directory where results should be saved.
            keypoint_filename_override (str, optional): Override for the output keypoint JSON filename.
        """
        pass

    @staticmethod
    def get_paths(video_path, output_path):
        """
        Helper to resolve absolute paths for input and output.

        Args:
            video_path (str): Input video path.
            output_path (str): Output directory path.

        Returns:
            tuple: (resolved_video_path, resolved_output_path) as strings.
        """
        return str(Path(video_path).resolve()), str(Path(output_path).resolve())
