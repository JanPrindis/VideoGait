import sys
import os

from utils.config_models import AppConfig

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- IMPORTS ---
from gaitDetectHeuristic.builder import build_heuristic_detector
from utils.logger import log


def run_heuristic_inference(
        app_config: AppConfig,
        input_path: str,
        output_dir: str = None
):
    """
    Executes the heuristic gait analysis inference pipeline.

    Args:
        app_config (AppConfig): The application configuration object.
        input_path (str): Path to the input keypoints JSON file.
        output_dir (str, optional): Directory to save output plots. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - "events": Extracted gait events.
            - "framerate": The framerate used.
            - "global_ranges": List of valid frame ranges processed.
            - "predictions": None (Heuristics do not output continuous probabilities).
    """
    if not os.path.isabs(input_path):
        input_path = os.path.join(PROJECT_ROOT, input_path)

    if output_dir is not None and not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)

    # Safety Check
    method = app_config.event_detector.method
    if method != 'Heuristic':
        log("INFERENCE", f"Config specifies method '{method}', but this is the Heuristic pipeline. Proceeding anyway.", level="warning")

    # Build Detector using the Builder Pattern
    try:
        detector = build_heuristic_detector(app_config)
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
