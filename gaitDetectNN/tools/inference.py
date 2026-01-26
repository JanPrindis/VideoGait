from typing import Union, Dict, Any, List

import yaml
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- IMPORTS ---
import gaitDetectNN.models
from gaitDetectNN.utils.builder import build_model
from gaitDetectNN.engine.predictor import Predictor

from skeletons import get_skeleton_by_name
from utils.preprocessing import generate_features

from gaitStructs import GaitEvent, GaitEventType

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
    with open(train_cfg_path, 'r') as f:
        train_cfg = yaml.safe_load(f)

    # Build model
    print(f"[Model] Building architecture: {train_cfg['model']['type']} (Input Size: {input_size})")
    model = build_model(train_cfg['model'], input_size=input_size)

    # Load weights
    print(f"[Model] Loading weights from: {weights_path}")
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


def visualize_results(predictions, events, cfg, save_path):
    """
    Generates and saves a plot visualizing the model's confidence scores and detected events.

    Args:
        predictions (np.ndarray): A (N, 4) array of probability scores.
        events (Dict[str, List[GaitEvent]]): The dictionary of extracted gait events.
        cfg (dict): The application configuration dictionary (used for threshold visualization).
        save_path (str): The file path where the plot image will be saved.

    Returns:
        None
    """
    nn_cfg = cfg['event_detector'].get('neural_net', {})
    post_proc = nn_cfg.get('post_processing', {})
    threshold = post_proc.get('threshold', 0.5)

    num_classes = predictions.shape[1]
    fig, axs = plt.subplots(num_classes, 1, figsize=(12, 12), sharex=True)
    if num_classes == 1: axs = [axs]

    # Event to plot mapping
    plot_mapping = {
        0: ("left", GaitEventType.HEEL_STRIKE),
        1: ("left", GaitEventType.TOE_OFF),
        2: ("right", GaitEventType.HEEL_STRIKE),
        3: ("right", GaitEventType.TOE_OFF)
    }

    for i in range(num_classes):
        # Confidence line
        axs[i].plot(predictions[:, i], label='Probability', color='blue', linewidth=1.5)

        # Confidence threshold
        axs[i].axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Threshold ({threshold})')

        # Axis config
        axs[i].set_ylabel(EVENT_ORDER[i])
        axs[i].set_ylim(-0.1, 1.2)
        axs[i].grid(True, alpha=0.3)

        # Detected events
        side, ev_type = plot_mapping[i]
        relevant_events = [e for e in events[side] if e.event_type == ev_type]

        if relevant_events:
            frames = [e.frame for e in relevant_events]
            values = predictions[frames, i]

            # Dot
            axs[i].scatter(frames, values, color='lime', edgecolors='black', s=50, zorder=5, label='Detected Peak')

            # Event frame number
            for f, v in zip(frames, values):
                axs[i].text(f, v + 0.08, str(f), fontsize=9, ha='center', va='bottom', fontweight='bold',
                            color='darkgreen')

        if i == 0:
            axs[i].legend(loc='lower right')

    plt.xlabel("Frame Index")
    plt.suptitle("Gait Event Detection")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Output] Plot saved to: {save_path}")
    plt.close(fig)


def run_nn_inference(
    app_config: Union[str, Dict[str, Any]],
    input_path: str,
    output_dir: str = None
):
    """
    Executes the full gait analysis inference pipeline.

    This includes:
    1. Loading application and training configurations.
    2. Preprocessing input keypoint data (normalization, feature extraction).
    3. Loading the trained neural network model.
    4. Running predictions on the processed data.
    5. Post-processing predictions to extract discrete gait events.
    6. Visualizing and saving the prediction results for debug use.

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
    # Load config
    if isinstance(app_config, str):
        if not os.path.isabs(app_config):
            app_config = os.path.join(PROJECT_ROOT, app_config)

        print(f"[Inference] Loading App Config from: {app_config}")

        with open(app_config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = app_config


    if not os.path.isabs(input_path):
        input_path = os.path.join(PROJECT_ROOT, input_path)

    # Determine settings
    target_fps = cfg.get('preprocessing', {}).get('framerate', 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Safety Check
    if cfg['event_detector']['method'] != 'NeuralNet':
        raise ValueError(
            f"Inference pipeline called with method '{cfg['event_detector']['method']}'. This module supports 'NeuralNet' only.")

    # Load & Sync with Training Configuration
    nn_cfg = cfg['event_detector']['neural_net']
    exp_path = nn_cfg['experiment_path']
    full_exp_path = os.path.join(PROJECT_ROOT, exp_path)

    # Load training config
    train_cfg_path = os.path.join(full_exp_path, "config.yaml")
    if not os.path.exists(train_cfg_path):
        raise FileNotFoundError(f"Training config missing at: {train_cfg_path}")

    with open(train_cfg_path, 'r') as f:
        train_cfg = yaml.safe_load(f)

    # Sync FPS
    if 'framerate' in train_cfg.get('data', {}):
        trained_fps = train_cfg['data']['framerate']
        if trained_fps != target_fps:
            print(f"[Inference] Overriding App FPS ({target_fps}) -> Model Trained FPS ({trained_fps})")
            target_fps = trained_fps

    # Sync features
    feat_def = train_cfg['data']['features']
    skeleton_def = get_skeleton_by_name(train_cfg['data']['skeleton'])

    preprocess_args = {
        "skeleton_definition": skeleton_def,
        "confidence_threshold": cfg['preprocessing'].get('confidence_threshold', 0.5),
        "exclude_ratio": cfg['preprocessing'].get('exclude_ratio', 0.1),
        "keypoints": feat_def['keypoints'],
        "kinematics_keypoints": feat_def['kinematics'],
        "angle_triplets": feat_def['angles'],
        "distance_pairs": feat_def['distances'],
        "filter_cutoff": train_cfg['preprocessing'].get('filter_cutoff', 6),
        "filter_order": train_cfg['preprocessing'].get('filter_order', 4),
        "min_segment_length": train_cfg['preprocessing'].get('min_segment_length', 60),
        "outlier_ratio": train_cfg['preprocessing'].get('outlier_ratio', 0.2)
    }

    # Run data preprocessing
    feature_matrices, _, global_ranges = generate_features(
        keypoints_path=input_path,
        annotations_path=None,  # During inference, we don't have the ground truth data
        frame_rate=target_fps,
        **preprocess_args
    )

    if not feature_matrices:
        print("[Inference] Warning: No valid clips generated (low confidence or short video).")
        return None

    # Load Model & Predictor
    input_size = feature_matrices[0].shape[1]

    model, _ = load_train_config_and_model(
        full_exp_path,
        nn_cfg['checkpoint'],
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
    post_proc_cfg = nn_cfg.get('post_processing', {})
    threshold = post_proc_cfg.get('threshold', 0.5)
    min_dist_sec = post_proc_cfg.get('min_distance_sec', 0.25)
    min_dist_frames = int(min_dist_sec * target_fps)

    print(f"[Inference] Extracting events (Threshold: {threshold}, Min Dist: {min_dist_frames} frames)...")
    structured_events = extract_gait_events(
        predictions=full_prediction,
        threshold=threshold,
        min_distance_frames=min_dist_frames
    )

    # Save output
    if output_dir:
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(PROJECT_ROOT, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Save the prediction plot
        plot_path = os.path.join(output_dir, "gait_confidences_plot.png")
        visualize_results(full_prediction, structured_events, cfg, plot_path)

    return {
        "predictions": full_prediction,
        "events": structured_events,
        "framerate": target_fps,
        "event_names": EVENT_ORDER,
        "global_ranges": global_ranges
    }


if __name__ == "__main__":
    config = "configs/apps/analyze_video.yaml"
    # input_path = "results/test.json"
    input_path = "dataset/PROCESSED/60/KEYPOINTS/PD_006_MD.json"
    output_dir = "results/test_patient"

    data = run_inference_pipeline(config, input_path, output_dir)
    print("Test")

    from gaitStructs import build_phases_from_events
    from visualizeGaitPhases import visualize_gait_phases, print_statistics

    l_phases, r_phases, support_phases = build_phases_from_events(
        data["events"],
        data["global_ranges"]
    )

    print_statistics(l_phases, r_phases, support_phases)
    visualize_gait_phases(l_phases, r_phases, support_phases)