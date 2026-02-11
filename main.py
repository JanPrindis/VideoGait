"""
Main entry point for the Gait Analysis Pipeline.

This script orchestrates the entire workflow:
1. Video Interpolation (Smart FPS adjustment)
2. Pose Estimation (Keypoint extraction)
3. Gait Event Detection (Heuristic or Neural Network)
4. Visualization (Video overlay)
5. Analysis (Kinematics, Spatiotemporal metrics)
6. Reporting (PDF/HTML generation)
"""
import os
import sys
import yaml
from pathlib import Path
from utils.logger import log

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

from utils.gait_structs import build_phases_from_events
from utils.video_processing import smart_interpolate
from detectors.inference import run_pose_extraction
from gaitDetectHeuristic.tools.inference import run_heuristic_inference
from gaitDetectNN.tools.inference import run_nn_inference
from utils.visualizer import GaitVisualizer
from utils.analysis import GaitAnalyzer
from utils.plotting import GaitPlotter
from utils.report_generator import ReportGenerator

def load_yaml(path):
    """Helper to load a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_analysis_pipeline(
    input_video_path: str,
    app_config_path: str,
    analysis_name_override: str = None,
    output_root_override: str = None,
    output_format: str = 'pdf'
):
    """
    Executes the complete gait analysis pipeline on a single video.

    Args:
        input_video_path (str): Path to the input video file.
        app_config_path (str): Path to the application configuration YAML.
        analysis_name_override (str, optional): Custom name for the analysis output folder.
                                                Defaults to video filename.
        output_root_override (str, optional): Custom root directory for outputs.
                                              Defaults to 'results' defined in config.
        output_format (str, optional): Format of the final report ('pdf', 'html', 'interactive').
                                       Defaults to 'pdf'.

    Returns:
        str | None: Path to the output directory if successful, None otherwise.
    """
    log("PIPELINE", "Starting analysis...", level="info")
    log("PIPELINE", f"Input Video: {input_video_path}", level="info")
    log("PIPELINE", f"Config: {app_config_path}", level="info")

    # --- CONFIG LOADING ---
    if not os.path.exists(app_config_path):
        log("PIPELINE", f"Config not found at {app_config_path}", level="error")
        return None

    app_config = load_yaml(app_config_path)

    # --- OUTPUT DIRECTORY SETUP ---
    # Get output root folder
    res_root = output_root_override or app_config.get('output', {}).get('output_root_dir', 'results')

    # Get analysis folder name
    if analysis_name_override:
        an_name = analysis_name_override
    else:
        # Use name
        an_name = Path(input_video_path).stem

    # Final path: project_root / results / analysis_name
    output_dir = Path(res_root) / an_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log("PIPELINE", f"Output Directory: {output_dir}", level="info")

    # --- TARGET FPS RESOLUTION ---
    detector_cfg = app_config.get('event_detector', {})
    method = detector_cfg.get('method', 'Heuristic')

    if method == 'Heuristic':
        target_fps = app_config.get('preprocessing', {}).get('framerate', 60)
        log("CONFIG", f"Method: Heuristic -> Using AppConfig framerate: {target_fps}", level="info")

    elif method == 'NeuralNet':
        exp_path = detector_cfg.get('neural_net', {}).get('experiment_path', '')
        nn_config_path = os.path.join(exp_path, 'config.yaml')

        if os.path.exists(nn_config_path):
            nn_config = load_yaml(nn_config_path)
            target_fps = nn_config.get('data', {}).get('framerate', 60)  # Default 60
            log("CONFIG", f"Method: NeuralNet -> Loaded from {nn_config_path}", level="info")
            log("CONFIG", f"Target FPS: {target_fps}", level="info")
        else:
            log("CONFIG", f"ERROR: Neural Net config not found at {nn_config_path}", level="error")
            return None
    else:
        log("CONFIG", f"Unknown method: {method}", level="error")
        return None

    # --- VIDEO INTERPOLATION ---
    current_video_path = str(output_dir / f"{an_name}.mp4")

    if os.path.exists(current_video_path):
        log("VIDEO", f"Interpolated video already exists at {current_video_path}. Skipping.", level="warning")
    else:
        log("VIDEO", f"Interpolating video...", level="info")
        try:
            smart_interpolate(
                input_path=input_video_path,
                output_path=current_video_path,
                target_fps=target_fps
            )
            log("VIDEO", "Interpolation done.", level="success")
        except Exception as e:
            log("VIDEO", f"Interpolation FAILED: {e}", level="error")
            import traceback
            traceback.print_exc()
            return None

    log("PIPELINE", f"Working Video: {current_video_path}", level="info")
    log("PIPELINE", f"Working FPS: {target_fps}", level="info")

    # --- KEYPOINT EXTRACTION ---
    run_pose_extraction(
        app_config=app_config,
        video_path=current_video_path,
        output_root=res_root,
        run_name=an_name,
    )

    # --- GAIT EVENT DETECTION ---
    event_detector_func = run_heuristic_inference if method == "Heuristic" else run_nn_inference
    event_data = event_detector_func(
        app_config=app_config,
        input_path=str(output_dir / "pose_detector_data" / f"{an_name}.json"),
        output_dir=str(output_dir)
    )

    if event_data is None:
        log("PIPELINE", "Gait event detection failed (returned None). Aborting analysis.", level="error")
        return None

    l_phases, r_phases, support_phases = build_phases_from_events(
        event_data["events"],
        event_data["global_ranges"]
    )

    # Ensure phases are lists (handle potential None returns)
    l_phases = l_phases or []
    r_phases = r_phases or []
    support_phases = support_phases or []

    # --- VIDEO VISUALIZATION ---
    visualizer = GaitVisualizer(
        app_config=app_config
    )

    gait_data = {
        "events": event_data["events"],
        "left_phases": l_phases,
        "right_phases": r_phases,
        "support_phases": support_phases
    }

    visualizer.process_video(
        video_path=current_video_path,
        output_root=str(output_dir),
        keypoints_data=str(output_dir / "pose_detector_data" / f"{an_name}.json"),
        valid_ranges=event_data["global_ranges"],
        gait_data=gait_data
    )

    # --- RUN ANALYSIS ---
    analyzer = GaitAnalyzer(app_config)
    analysis_report = analyzer.run_analysis(
        keypoints_path=str(output_dir / "pose_detector_data" / f"{an_name}.json"),
        events_data=event_data,
        gait_data=gait_data,
        output_dir=str(output_dir)
    )

    if analysis_report is None:
        log("PIPELINE", "Gait Analysis failed (returned None). Skipping plots and report.", level="error")
        return None

    # --- GENERATE PLOTS ---
    if output_format == "interactive":
        log("PIPELINE", "Interactive HTML selected, skipping plots generation.", level="warning")
    else:
        plotter = GaitPlotter(str(output_dir / "plots"))
        plotter.generate_plots_from_json(analysis_source=analysis_report)

    # --- EXPORT ---
    # TODO: Annoying setup - document: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation
    report_generator = ReportGenerator(
        app_config=app_config,
        output_dir=str(output_dir),
    )

    report_generator.export(
        analysis_source=analysis_report,
        graphs_dir=str(output_dir / "plots"),
        video_dir=str(output_dir),
        filename_base=f"{an_name}_report",
        output_type=output_format
    )

    return str(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VideoGait Analysis Pipeline")
    parser.add_argument("--video", "-v", required=True, help="Path to input video file")
    parser.add_argument("--config", "-c", required=True, help="Path to app config file (YAML)")
    parser.add_argument("--output", "-o", default=None, help="Override output root directory")
    parser.add_argument("--name", "-n", default=None, help="Override analysis name (folder name)")
    parser.add_argument("--format", "-f", default="pdf", choices=["pdf", "html", "interactive"], help="Report output format")

    args = parser.parse_args()

    result = run_analysis_pipeline(
        input_video_path=args.video,
        app_config_path=args.config,
        analysis_name_override=args.name,
        output_root_override=args.output,
        output_format=args.format
    )

    if not result:
        sys.exit(1)
