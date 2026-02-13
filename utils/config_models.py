"""
This module defines Pydantic models for configuration validation across the application.

It ensures type safety and structure for:
- Application Config (Inference)
- Training Config
- Benchmark Config
- Detector Config
"""
from typing import Optional, Literal, Dict, Any, List

from pydantic import BaseModel, ConfigDict, model_validator, Field


# ==========================================
# SHARED
# ==========================================

class PreprocessingConfig(BaseModel):
    """
    Configuration for data preprocessing steps.
    """
    model_config = ConfigDict(extra='allow')

    framerate: float = 60.0
    confidence_threshold: float = 0.4
    exclude_ratio: float = 0.1
    min_segment_length: int = 60
    outlier_ratio: float = 0.2
    filter_cutoff: int = 6
    filter_order: int = 4
    use_filter: bool = True


class NNPostProcessing(BaseModel):
    """
    Configuration for Neural Network post-processing (event extraction).
    """
    model_config = ConfigDict(extra='allow')

    threshold: float = 0.5
    min_distance_sec: float = 0.25


class NeuralNetConfig(BaseModel):
    """
    Configuration specific to Neural Network event detection.
    """
    model_config = ConfigDict(extra='allow')

    experiment_path: str
    checkpoint: str
    post_processing: NNPostProcessing = Field(default_factory=NNPostProcessing)


class HeuristicConfig(BaseModel):
    """
    Configuration specific to Heuristic event detection.
    """
    model_config = ConfigDict(extra='allow')

    method: str
    skeleton: Optional[str] = None
    seed: Optional[int] = 3
    train_split: Optional[float] = 0.8
    params: Dict[str, Any] = Field(default_factory=dict)


class EventDetectorConfig(BaseModel):
    """
    Main configuration for the Event Detector component.
    """
    model_config = ConfigDict(extra='allow')

    method: Literal["NeuralNet", "Heuristic"]
    neural_net: Optional[NeuralNetConfig] = None
    heuristic: Optional[HeuristicConfig] = None

    @model_validator(mode='after')
    def check_method_dependencies(self):
        if self.method == "NeuralNet" and self.neural_net is None:
            raise ValueError('method="NeuralNet", ale chybí sekce "neural_net"!')
        if self.method == "Heuristic" and self.heuristic is None:
            raise ValueError('method="Heuristic", ale chybí sekce "heuristic"!')
        return self


# ==========================================
# APP CONFIG
# ==========================================

class PoseDetectorRef(BaseModel):
    """
    Reference to the Pose Detector configuration file.
    """
    model_config = ConfigDict(extra='allow')
    config_path: str


class OutputConfig(BaseModel):
    """
    Configuration for output generation.
    """
    model_config = ConfigDict(extra='allow')

    output_root_dir: str = "results"
    save_video: bool = True
    save_report: bool = True


class VideoOutputsConfig(BaseModel):
    """
    Toggles for specific visualization layers in the output video.
    """
    model_config = ConfigDict(extra='allow')

    enable_overlay: bool = True
    enable_kinematics: bool = True
    enable_logic: bool = True


class VisualizationConfig(BaseModel):
    """
    Configuration for visualization appearance and behavior.
    """
    model_config = ConfigDict(extra='allow')

    text_output_type: Literal["pdf", "md", "console", "interactive", "html"] = "pdf"
    event_detector_debug_plots: bool = False

    colors: Dict[str, Any] = Field(default_factory=dict)

    line_thickness: int = 3
    keypoint_radius: int = 5
    trail_thickness: int = 2
    footprint_duration: int = 60

    outputs: VideoOutputsConfig = Field(default_factory=VideoOutputsConfig)


class AppConfig(BaseModel):
    """
    Root configuration for the main application (Inference Pipeline).
    """
    model_config = ConfigDict(extra='allow')

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    pose_detector: PoseDetectorRef
    event_detector: EventDetectorConfig

    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# ==========================================
# BENCHMARK CONFIG
# ==========================================

class EvaluationConfig(BaseModel):
    """
    Configuration for benchmarking evaluation metrics.
    """
    model_config = ConfigDict(extra='allow')

    strict_tolerance_ms: int = 50
    loose_tolerance_ms: int = 150


class BenchmarkDataConfig(BaseModel):
    """
    Configuration for benchmark dataset loading.
    """
    model_config = ConfigDict(extra='allow')

    dataset_root: str
    annotation_root: str = "annotations"
    use_full_dataset: bool = False


class BenchmarkConfig(BaseModel):
    """
    Root configuration for the Benchmark script.
    """
    model_config = ConfigDict(extra='allow')

    experiment_name: str
    output_dir: str
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    event_detector: EventDetectorConfig
    data: BenchmarkDataConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ==========================================
# TRAINING CONFIG
# ==========================================

class TrainFeaturesConfig(BaseModel):
    """
    Configuration for input features used in training.
    """
    model_config = ConfigDict(extra='allow')

    keypoints: Optional[List[str]] = None
    kinematics: Optional[List[str]] = None
    angles: Optional[List[List[str]]] = None
    distances: Optional[List[List[str]]] = None


class TrainDataConfig(BaseModel):
    """
    Configuration for training data loading and processing.
    """
    model_config = ConfigDict(extra='allow')

    dataset_root: str
    annotation_root: str = "annotations"
    framerate: float = 60.0
    skeleton: str
    requires_adj_matrix: bool = False
    train_split: float = 0.8
    num_workers: int = 0
    features: TrainFeaturesConfig


class ModelDefinition(BaseModel):
    """
    Configuration for the Neural Network model architecture.
    """
    model_config = ConfigDict(extra='allow')

    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class TrainingParams(BaseModel):
    """
    Hyperparameters for the training loop.
    """
    model_config = ConfigDict(extra='allow')

    optimizer: str = "AdamW"
    learning_rate: float
    epochs: int
    batch_size: int

    weight_decay: Optional[float] = None
    momentum: Optional[float] = None
    f1_window_size_ms: Optional[int] = 50
    seed: Optional[int] = 3

    scheduler: Optional[str] = None
    scheduler_config: Dict[str, Any] = Field(default_factory=dict)


class TrainConfig(BaseModel):
    """
    Root configuration for the Training script.
    """
    model_config = ConfigDict(extra='allow')

    experiment_name: str
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    data: TrainDataConfig
    model: ModelDefinition
    training: TrainingParams


# ==========================================
# KEYPOINT DETECTOR CONFIG
# ==========================================

class DetectorConfig(BaseModel):
    """
    Configuration for Pose Detectors (e.g., RTMLib, AlphaPose).
    """
    model_config = ConfigDict(extra='allow')

    type: str
    skeleton: str
    params: Dict[str, Any] = Field(default_factory=dict)
