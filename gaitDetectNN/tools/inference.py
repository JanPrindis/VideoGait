import shutil
from typing import Union, Dict, List

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from utils.config_models import AppConfig, TrainConfig
from utils.config_utils import load_and_validate_yaml, build_preprocess_args_from_train_config, \
    resolve_skeleton_from_config

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- IMPORTS ---
from gaitDetectNN.builder import build_model
from gaitDetectNN.engine.predictor import Predictor

from utils.preprocessing import generate_features
from utils.gait_structs import GaitEvent, GaitEventType
from utils.logger import log

EVENT_ORDER = ["Left Heel Strike", "Left Toe Off", "Right Heel Strike", "Right Toe Off"]


def load_train_config_and_model(experiment_path, checkpoint_name, input_size, device):
    """
    Loads the training configuration and the trained model weights.

    Args:
        experiment_path (str): Path to the experiment directory containing 'config.yaml'.
        checkpoint_name (str): Name of the checkpoint file (e.g., 'best_model.pth').
        input_size (int): The size of the input feature vector expected by the model.
        device (str or torch.device): The device to load the model onto ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The loaded model with weights applied.
            - train_cfg (dict): The configuration dictionary used for training.

    Raises:
        FileNotFoundError: If the config or checkpoint file does not exist.
    """
    train_cfg_path = os.path.join(experiment_path, "config.yaml")
    weights_path = os.path.join(experiment_path, checkpoint_name)

    if not os.path.exists(train_cfg_path):
        raise FileNotFoundError(f"Train config not found at: {train_cfg_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    # Load training config
    try:
        train_cfg: TrainConfig = load_and_validate_yaml(train_cfg_path, TrainConfig)
    except ValueError as e:
        log("CONFIG", str(e), level="error")
        return None

    # Adjacency matrix injection
    if train_cfg.data.requires_adj_matrix:
        log("INFERENCE", "'requires_adj_matrix' is True -> Injecting feature config to model.", level="info")

        features_dict = train_cfg.data.features.model_dump()
        train_cfg.model.params['features_config'] = features_dict
        train_cfg.model.params['skeleton_name'] = train_cfg.data.skeleton

    # Build model
    log("INFERENCE", f"Building architecture: {train_cfg.model.type} (Input Size: {input_size})", level="info")
    model = build_model(train_cfg.model, input_size=input_size)

    # Load weights
    log("INFERENCE", f"Loading weights from: {weights_path}", level="info")
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model, train_cfg


def extract_gait_events(
        predictions: np.ndarray,
        threshold: float = 0.5,
        min_distance_frames: int = 20
) -> Dict[str, List[GaitEvent]]:
    """
    Extracts discrete gait events from continuous probability predictions using peak detection.

    Args:
        predictions (np.ndarray): A (N, 4) array of probability scores for each frame.
                                  Columns order: [L_HS, L_TO, R_HS, R_TO].
        threshold (float): The minimum probability required to consider a peak a valid event.
        min_distance_frames (int): The minimum number of frames between two events of the same type.

    Returns:
        Dict[str, List[GaitEvent]]: A dictionary with keys 'left' and 'right', containing
                                    lists of detected GaitEvent objects sorted by frame index.
    """
    events = {
        "left": [],
        "right": []
    }

    # Order must match [L_HS, L_TO, R_HS, R_TO]
    column_mapping = {
        0: ("left", GaitEventType.HEEL_STRIKE),
        1: ("left", GaitEventType.TOE_OFF),
        2: ("right", GaitEventType.HEEL_STRIKE),
        3: ("right", GaitEventType.TOE_OFF)
    }

    for col_idx in range(4):
        side, event_type = column_mapping[col_idx]
        signal = predictions[:, col_idx]

        # Find confidence peaks
        peaks, properties = find_peaks(
            signal,
            height=threshold,
            distance=min_distance_frames
        )

        for i, frame_idx in enumerate(peaks):
            event = GaitEvent(
                frame=int(frame_idx),
                event_type=event_type
            )
            events[side].append(event)

    # Sort events by frame index
    events["left"].sort(key=lambda x: x.frame)
    events["right"].sort(key=lambda x: x.frame)

    return events


def visualize_confidences(
        predictions: np.ndarray,
        events: Dict[str, List[GaitEvent]],
        app_config: AppConfig,
        save_path: str
):
    """
    Generates and saves a plot visualizing the model's confidence scores and detected events.

    Args:
        predictions (np.ndarray): A (N, 4) array of probability scores.
        events (Dict[str, List[GaitEvent]]): The dictionary of extracted gait events.
        app_config (dict): The application configuration dictionary (used for threshold visualization).
        save_path (str): The file path where the plot image will be saved.

    Returns:
        None
    """
    nn_cfg = app_config.event_detector.neural_net
    threshold = nn_cfg.post_processing.threshold
    experiment_name = nn_cfg.experiment_path.rstrip("/").split("/")[-1]

    num_classes = predictions.shape[1]
    fig, axs = plt.subplots(num_classes, 1, figsize=(12, 14), sharex=True)
    if num_classes == 1: axs = [axs]

    # Color Palette
    col_hs_sig = 'blue'
    col_to_sig = 'green'
    col_hs_marker = 'red'
    col_to_marker = 'orange'

    # Mapping index -> (Side, Type, Title, ColorSignal, ColorMarker, MarkerShape)
    layer_config = {
        0: ("left", GaitEventType.HEEL_STRIKE, "Left Heel Strike", col_hs_sig, col_hs_marker, 'v'),
        1: ("left", GaitEventType.TOE_OFF, "Left Toe Off", col_to_sig, col_to_marker, '^'),
        2: ("right", GaitEventType.HEEL_STRIKE, "Right Heel Strike", col_hs_sig, col_hs_marker, 'v'),
        3: ("right", GaitEventType.TOE_OFF, "Right Toe Off", col_to_sig, col_to_marker, '^')
    }

    for i in range(num_classes):
        ax = axs[i]
        side, ev_type, title, col_sig, col_marker, marker_shape = layer_config[i]

        signal = predictions[:, i]

        # Probability curve
        ax.plot(signal, label='Confidence', color=col_sig, linewidth=2)
        ax.fill_between(range(len(signal)), signal, 0, color=col_sig, alpha=0.2)

        # Threshold line
        ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1, alpha=0.8,
                   label=f'Threshold ({threshold})')

        # Detected events
        relevant_events = [e for e in events[side] if e.event_type == ev_type]

        if relevant_events:
            frames = [e.frame for e in relevant_events]
            values = signal[frames]

            # Scatter Marker
            ax.scatter(frames, values, c=col_marker, marker=marker_shape,
                       s=100, zorder=10, edgecolors='black', label='Detected Event')

            # Text Labels
            for f, v in zip(frames, values):
                ax.text(f, v + 0.05, str(f), fontsize=10, ha='center', va='bottom',
                        fontweight='bold', color='black', zorder=11)

        # Styling
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend (only first HS and TO)
        if i == 0 or i == 1:
            ax.legend(loc='lower left', frameon=True, framealpha=0.9)

    plt.xlabel("Frame Index", fontsize=12)
    plt.suptitle(f"Neural Network Gait Event Detection - {experiment_name}", fontsize=16, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log("INFERENCE", f"Plot saved to: {save_path}", level="success")


def run_nn_inference(
    app_config: AppConfig,
    input_path: str,
    output_dir: str = None
):
    """
    Executes the full gait analysis inference pipeline.

    Args:
        app_config (Union[str, Dict[str, Any]]): Path to the app config YAML or the loaded dict.
        input_path (str): Path to the input keypoints JSON file.
        output_dir (str, optional): Directory to save output plots. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - "predictions": Raw probability array.
            - "events": Extracted gait events.
            - "framerate": The framerate used for analysis.
            - "event_names": List of event names corresponding to prediction columns.
            - "global_ranges": List of valid frame ranges processed.
    """
    if not os.path.isabs(input_path):
        input_path = os.path.join(PROJECT_ROOT, input_path)

    # Determine settings
    target_fps = app_config.preprocessing.framerate
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Safety Check
    method = app_config.event_detector.method
    if method != "NeuralNet":
        raise ValueError(
            f"Inference pipeline called with method '{method}'. This module supports 'NeuralNet' only.")

    # Load & Sync with Training Configuration
    nn_cfg = app_config.event_detector.neural_net
    exp_path = nn_cfg.experiment_path
    full_exp_path = os.path.join(PROJECT_ROOT, exp_path)

    # Load training config
    train_cfg_path = os.path.join(full_exp_path, "config.yaml")
    if not os.path.exists(train_cfg_path):
        raise FileNotFoundError(f"Training config missing at: {train_cfg_path}")

    try:
        train_cfg: TrainConfig = load_and_validate_yaml(train_cfg_path, TrainConfig)
    except ValueError as e:
        log("CONFIG", str(e), level="error")
        return None

    # Sync FPS
    trained_fps = train_cfg.data.framerate
    if trained_fps != target_fps:
        log("INFERENCE", f"Overriding App FPS ({target_fps}) -> Model Trained FPS ({trained_fps})", level="warning")
        target_fps = trained_fps

    # Sync features
    preprocess_args = build_preprocess_args_from_train_config(train_cfg)

    # Override the skeleton with detector's skeleton to allow cross skeleton detection
    # This ensures we parse the keypoint JSON using the indices of the actual detector
    # Missing features will be zero-padded
    try:
        detector_skeleton = resolve_skeleton_from_config(app_config)
    except ValueError:
        # Fallback for benchmarking where app_config is mocked and has no valid pose detector
        detector_skeleton = resolve_skeleton_from_config(train_cfg)

    preprocess_args["skeleton_definition"] = detector_skeleton

    preprocess_args["confidence_threshold"] = app_config.preprocessing.confidence_threshold
    preprocess_args["exclude_ratio"] = app_config.preprocessing.exclude_ratio

    # Run data preprocessing
    feature_matrices, _, global_ranges = generate_features(
        keypoints_path=input_path,
        annotations_path=None,  # During inference, we don't have the ground truth data
        frame_rate=target_fps,
        **preprocess_args
    )

    if not feature_matrices:
        log("INFERENCE", "Warning: No valid clips generated (low confidence or short video).", level="warning")
        return None

    # Load Model & Predictor
    input_size = feature_matrices[0].shape[1]

    model, _ = load_train_config_and_model(
        full_exp_path,
        nn_cfg.checkpoint,
        input_size,
        device
    )
    predictor = Predictor(model, device)

    # Run Prediction & Stitching
    # Calculate total frames of the video
    total_frames = max(end for start, end in global_ranges) + 1

    # Predict the first clip, so we get the number of classes
    first_probs = predictor.predict_on_clip(feature_matrices[0])
    num_classes = first_probs.shape[1]

    # Initialize the full prediction array
    full_prediction = np.zeros((total_frames, num_classes))

    for idx, clip_features in enumerate(feature_matrices):
        probs = predictor.predict_on_clip(clip_features)

        # Calculate global time offset
        start, end = global_ranges[idx]
        length = end - start + 1

        # Trim shapes to match
        probs = probs[:length]
        valid_len = min(len(probs), length)

        full_prediction[start: start + valid_len] = probs[:valid_len]

    # Post-processing (Event Extraction)
    post_proc_cfg = nn_cfg.post_processing
    threshold = post_proc_cfg.threshold
    min_dist_sec = post_proc_cfg.min_distance_sec
    min_dist_frames = int(min_dist_sec * target_fps)

    log("INFERENCE", f"Extracting events (Threshold: {threshold}, Min Dist: {min_dist_frames} frames)...", level="info")
    structured_events = extract_gait_events(
        predictions=full_prediction,
        threshold=threshold,
        min_distance_frames=min_dist_frames
    )

    # Save output
    save_debug_plot = app_config.visualization.event_detector_debug_plots

    if output_dir and save_debug_plot:
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(PROJECT_ROOT, output_dir)

        # Create output folder
        debug_dir = os.path.join(output_dir, "event_detector_debug")

        # If debug folder exists, remove
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)

        # Create clean debug dir (does not contain old files)
        os.makedirs(debug_dir, exist_ok=True)

        # Save the prediction plot
        plot_path = os.path.join(debug_dir, "gait_confidences_plot.png")
        visualize_confidences(full_prediction, structured_events, app_config, plot_path)

    return {
        "predictions": full_prediction,
        "events": structured_events,
        "framerate": target_fps,
        "event_names": EVENT_ORDER,
        "global_ranges": global_ranges
    }
