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
import json
from pathlib import Path

from utils.config_models import AppConfig, TrainConfig
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
from utils.config_utils import resolve_skeleton_name_from_config, load_and_validate_yaml


def _setup_pipeline_environment(
        app_config_path: str,
        input_video_path: str,
        analysis_name_override: str,
        output_root_override: str
):
    """Loads config and sets up output paths common to all pipelines."""
    # --- CONFIG LOADING ---
    if not os.path.exists(app_config_path):
        log("PIPELINE", f"Config not found at {app_config_path}", level="error")
        return None, None, None

    try:
        app_config: AppConfig = load_and_validate_yaml(app_config_path, AppConfig)
    except ValueError as e:
        log("CONFIG", str(e), level="error")
        return None, None, None

    # --- OUTPUT DIRECTORY SETUP ---
    res_root = output_root_override or app_config.output.output_root_dir
    if analysis_name_override:
        an_name = analysis_name_override
    else:
        an_name = Path(input_video_path).stem

    output_dir = Path(res_root) / an_name
    return app_config, output_dir, an_name


def _generate_plots_and_report(app_config: AppConfig, output_dir: Path, analysis_report: dict, an_name: str, output_format: str):
    """Helper to generate plots and the final report file."""
    if "metadata" in analysis_report:
        analysis_report["metadata"]["file"] = an_name

    # --- GENERATE PLOTS ---
    if output_format == "interactive":
        log("PIPELINE", "Interactive HTML selected, skipping plots generation.", level="warning")
    else:
        plotter = GaitPlotter(str(output_dir / "plots"))
        plotter.generate_plots_from_json(analysis_source=analysis_report)

    # --- EXPORT ---
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

    # --- SETUP ---
    app_config, output_dir, an_name = _setup_pipeline_environment(
        app_config_path, input_video_path, analysis_name_override, output_root_override
    )
    if not app_config:
        return None

    # Create directory if it doesn't exist for the main pipeline
    output_dir.mkdir(parents=True, exist_ok=True)
    log("PIPELINE", f"Output Directory: {output_dir}", level="info")

    # --- TARGET FPS RESOLUTION AND USED SKELETONS ---
    detector_cfg = app_config.event_detector
    method = detector_cfg.method

    # Check what skeleton is used by the detector
    try:
        produced_skeleton_name = resolve_skeleton_name_from_config(app_config)
    except Exception as e:
        log("PIPELINE", f"Failed to resolve skeleton name: {e}", level="error")
        return None

    if method == 'Heuristic':
        target_fps = app_config.preprocessing.framerate
        log("CONFIG", f"Method: Heuristic -> Using AppConfig framerate: {target_fps}", level="info")
        log("PIPELINE", f"Heuristics will use skeleton: {produced_skeleton_name}", level="info")

    elif method == 'NeuralNet':
        exp_path = detector_cfg.neural_net.experiment_path
        nn_config_path = os.path.join(exp_path, 'config.yaml')

        if os.path.exists(nn_config_path):
            try:
                nn_config: TrainConfig = load_and_validate_yaml(nn_config_path, TrainConfig)
            except ValueError as e:
                log("CONFIG", str(e), level="error")
                return None

            target_fps = nn_config.data.framerate
            expected_skeleton_name = nn_config.data.skeleton

            log("CONFIG", f"Method: NeuralNet -> Loaded from {nn_config_path}", level="info")
            log("CONFIG", f"Target FPS: {target_fps}", level="info")

            # Skeleton handshake check
            if expected_skeleton_name.upper() != produced_skeleton_name:
                log("PIPELINE",
                    f"SKELETON MISMATCH WARNING! Detector uses '{produced_skeleton_name}', but NeuralNet expects '{expected_skeleton_name.upper()}'!",
                    level="warning")
                log("PIPELINE", "Proceeding with Cross-Skeleton Inference. Missing keypoints will be zero-padded.",
                    level="warning")
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
        output_root=str(output_dir.parent),
        run_name=an_name,
    )

    # --- GAIT EVENT DETECTION ---
    event_detector_func = run_heuristic_inference if method == "Heuristic" else run_nn_inference
    event_data = event_detector_func(
        app_config=app_config,
        input_path=str(output_dir / "keypoints.json"),
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
        keypoints_data=str(output_dir / "keypoints.json"),
        valid_ranges=event_data["global_ranges"],
        gait_data=gait_data
    )

    # --- RUN ANALYSIS ---
    analyzer = GaitAnalyzer(app_config)
    analysis_report = analyzer.run_analysis(
        keypoints_path=str(output_dir / "keypoints.json"),
        events_data=event_data,
        gait_data=gait_data,
        output_dir=str(output_dir)
    )

    if analysis_report is None:
        log("PIPELINE", "Gait Analysis failed (returned None). Skipping plots and report.", level="error")
        return None

    # --- GENERATE PLOTS & REPORT ---
    _generate_plots_and_report(
        app_config=app_config,
        output_dir=output_dir,
        analysis_report=analysis_report,
        an_name=an_name,
        output_format=output_format
    )

    return str(output_dir)


def regenerate_report_pipeline(
    input_video_path: str,
    app_config_path: str,
    analysis_name_override: str = None,
    output_root_override: str = None,
    output_format: str = 'pdf'
):
    log("PIPELINE", "Regenerating report...", level="info")

    # --- SETUP ---
    app_config, output_dir, an_name = _setup_pipeline_environment(
        app_config_path, input_video_path, analysis_name_override, output_root_override
    )
    if not app_config:
        return None

    # Check for existing directory in regenerate pipeline
    if not output_dir.exists():
        log("PIPELINE", f"Output directory not found: {output_dir}", level="error")
        return None

    # --- LOAD ANALYSIS RESULTS ---
    analysis_path = output_dir / "analysis.json"
    if not analysis_path.exists():
        log("PIPELINE", f"Analysis file not found: {analysis_path}. Cannot regenerate report.", level="error")
        return None

    with open(analysis_path, 'r') as f:
        analysis_report = json.load(f)

    # --- GENERATE PLOTS & REPORT ---
    _generate_plots_and_report(
        app_config=app_config,
        output_dir=output_dir,
        analysis_report=analysis_report,
        an_name=an_name,
        output_format=output_format
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
