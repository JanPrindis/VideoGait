import sys
import os
import yaml
from typing import Union, Dict, Any

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- IMPORTS ---
from gaitDetectHeuristic.builder import build_heuristic_detector
from utils.logger import log


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

        log("INFERENCE", f"Loading App Config from: {app_config}", level="info")
        with open(app_config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = app_config

    if not os.path.isabs(input_path):
        input_path = os.path.join(PROJECT_ROOT, input_path)

    if output_dir is not None and not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)

    # Safety Check
    method = cfg.get('event_detector', {}).get('method')
    if method != 'Heuristic':
        log("INFERENCE", f"Config specifies method '{method}', but this is the Heuristic pipeline. Proceeding anyway.", level="warning")

    # Build Detector using the Builder Pattern
    try:
        detector = build_heuristic_detector(cfg)
        log("INFERENCE", f"Initialized Heuristic Detector: {detector.__class__.__name__}", level="info")
    except Exception as e:
        log("INFERENCE", f"Failed to initialize detector: {e}", level="error")
        return None

    # Run Inference
    # The detector handles data loading, preprocessing, and range looping internally
    log("INFERENCE", f"Processing {os.path.basename(input_path)}...", level="info")
    result = detector.run_inference(input_path, output_dir)

    events = result['events']
    ranges = result['global_ranges']
    fps = result['framerate']

    l_count = len(events['left'])
    r_count = len(events['right'])
    log("INFERENCE", f"FPS: {fps} | Walking Segments: {len(ranges)}", level="info")
    log("INFERENCE", f"Left Events: {l_count}, Right Events: {r_count}", level="info")

    # Return structured dictionary matching NN pipeline format
    return {
        "predictions": None,  # Heuristics don't provide frame-by-frame probabilities
        "events": events,
        "framerate": fps,
        "event_names": ["Left HS", "Left TO", "Right HS", "Right TO"],  # Constant for consistency
        "global_ranges": ranges

    }


if __name__ == "__main__":
    methods = ["bonci", "desailly", "ghoussayni", "hreljac", "hsue", "oconnor", "zeni"]
    visualize = False

    for method in methods:
        config = f"configs/apps/analyze_video_{method}.yaml"

        input_path = "dataset/PROCESSED/60/KEYPOINTS/KOA_003_SV.json"
        output_dir = f"results/test_patient_{method}"

        data = run_heuristic_inference(config, input_path, output_dir)
        log("INFERENCE", f"Testing done: {method}", level="success")

        if visualize:
            from utils.gait_structs import build_phases_from_events
            from tools.debug_viz import visualize_gait_phases, print_statistics

            l_phases, r_phases, support_phases = build_phases_from_events(
                data["events"],
                data["global_ranges"]
            )

            print_statistics(l_phases, r_phases, support_phases)
            visualize_gait_phases(l_phases, r_phases, support_phases)
