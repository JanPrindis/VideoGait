import sys
import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict, Any, List

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- IMPORTS ---
from gaitDetectHeuristic.utils.builder import build_heuristic_detector


def run_heuristic_inference(
        app_config: Union[str, Dict[str, Any]],
        input_path: str,
        output_dir: str = None
):
    """
    Executes the heuristic gait analysis inference pipeline.

    Args:
        app_config (Union[str, Dict[str, Any]]): Path to the app config YAML or the loaded dict.
        input_path (str): Path to the input keypoints JSON file.
        output_dir (str, optional): Directory to save output plots. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - "events": Extracted gait events.
            - "framerate": The framerate used.
            - "global_ranges": List of valid frame ranges processed.
            - "predictions": None (Heuristics do not output continuous probabilities).
    """
    # Load Config
    if isinstance(app_config, str):
        if not os.path.isabs(app_config):
            app_config = os.path.join(PROJECT_ROOT, app_config)

        print(f"[Inference] Loading App Config from: {app_config}")
        with open(app_config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = app_config

    if not os.path.isabs(input_path):
        input_path = os.path.join(PROJECT_ROOT, input_path)

    # Safety Check
    method = cfg.get('event_detector', {}).get('method')
    if method != 'Heuristic':
        print(f"[Warning] Config specifies method '{method}', but this is the Heuristic pipeline. Proceeding anyway.")

    # Build Detector using the Builder Pattern
    try:
        detector = build_heuristic_detector(cfg)
        print(f"[Inference] Initialized Heuristic Detector: {detector.__class__.__name__}")
    except Exception as e:
        print(f"[Error] Failed to initialize detector: {e}")
        return None

    # Run Inference
    # The detector handles data loading, preprocessing, and range looping internally
    print(f"[Inference] Processing {os.path.basename(input_path)}...")
    result = detector.run_inference(input_path)

    events = result['events']
    ranges = result['global_ranges']
    fps = result['framerate']

    l_count = len(events['left'])
    r_count = len(events['right'])
    print(f"[Results] FPS: {fps} | Walking Segments: {len(ranges)}")
    print(f"          Left Events: {l_count}, Right Events: {r_count}")

    # Return structured dictionary matching NN pipeline format
    return {
        "predictions": None,  # Heuristics don't provide frame-by-frame probabilities
        "events": events,
        "framerate": fps,
        "event_names": ["Left HS", "Left TO", "Right HS", "Right TO"],  # Constant for consistency
        "global_ranges": ranges

    }


if __name__ == "__main__":
    config = "configs/apps/analyze_video_zeni.yaml"
    # input_path = "results/test.json"
    input_path = "dataset/PROCESSED/60/KEYPOINTS/KOA_003_SV.json"
    output_dir = "results/test_patient_heuristic"

    data = run_heuristic_inference(config, input_path, output_dir)
    print("Test")

    from gaitStructs import build_phases_from_events
    from visualizeGaitPhases import visualize_gait_phases, print_statistics

    l_phases, r_phases, support_phases = build_phases_from_events(
        data["events"],
        data["global_ranges"]
    )

    print_statistics(l_phases, r_phases, support_phases)
    visualize_gait_phases(l_phases, r_phases, support_phases)

