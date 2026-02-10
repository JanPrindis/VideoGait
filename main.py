import os
import sys
import yaml
from pathlib import Path

import logging

from utils.gait_structs import build_phases_from_events

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.video_processing import smart_interpolate
from detectors.inference import run_pose_extraction
from gaitDetectHeuristic.tools.inference import run_heuristic_inference
from gaitDetectNN.tools.inference import run_nn_inference
from utils.visualizer import GaitVisualizer
from utils.analysis import GaitAnalyzer
from utils.plotting import GaitPlotter
from utils.report_generator import ReportGenerator

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    input_video_path = "videos/Test1.mov"
    app_config_path = "configs/app/analyze_video_test.yaml"

    analysis_name_override = None
    output_root_override = None

    print(f"--- STARTING PIPELINE ---")
    print(f"Video: {input_video_path}")
    print(f"Config: {app_config_path}")

    # --- CONFIG LOADING ---
    if not os.path.exists(app_config_path):
        print(f"Error: Config not found at {app_config_path}")
        return

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
    print(f"Output Directory: {output_dir}")

    # --- TARGET FPS RESOLUTION ---
    detector_cfg = app_config.get('event_detector', {})
    method = detector_cfg.get('method', 'Heuristic')

    target_fps = None

    if method == 'Heuristic':
        target_fps = app_config.get('preprocessing', {}).get('framerate', 60)
        print(f"Method: Heuristic -> Using AppConfig framerate: {target_fps}")

    elif method == 'NeuralNet':
        exp_path = detector_cfg.get('neural_net', {}).get('experiment_path', '')
        nn_config_path = os.path.join(exp_path, 'config.yaml')

        if os.path.exists(nn_config_path):
            nn_config = load_yaml(nn_config_path)
            target_fps = nn_config.get('data', {}).get('framerate', 60)  # Default 60
            print(f"Method: NeuralNet -> Loaded from {nn_config_path}")
            print(f"Target FPS: {target_fps}")
        else:
            print(f"ERROR: Neural Net config not found at {nn_config_path}")
            return
    else:
        print(f"Unknown method: {method}")
        return

    # --- VIDEO INTERPOLATION ---
    current_video_path = str(output_dir / f"{an_name}.mp4")

    if os.path.exists(current_video_path):
        print(f"Interpolated video already exists at {current_video_path}. Skipping.")
    else:
        print(f"Interpolating video...")
        try:
            smart_interpolate(
                input_path=input_video_path,
                output_path=current_video_path,
                target_fps=target_fps
            )
            print("Interpolation done.")
        except Exception as e:
            print(f"Interpolation FAILED: {e}")
            import traceback
            traceback.print_exc()
            return

    print("-" * 30)
    print(f"Working Video: {current_video_path}")
    print(f"Working FPS: {target_fps}")
    print("-" * 30)

    # --- KEYPOINT EXTRACTION ---
    run_pose_extraction(
        app_config=app_config,
        video_path=current_video_path,
        output_root=res_root,
        run_name=an_name,
    )

    print("-" * 30)

    # --- GAIT EVENT DETECTION ---
    event_detector_func = run_heuristic_inference if method == "Heuristic" else run_nn_inference
    event_data = event_detector_func(
        app_config=app_config,
        input_path=str(output_dir / "pose_detector_data" / f"{an_name}.json"),
        output_dir=str(output_dir)
    )

    l_phases, r_phases, support_phases = build_phases_from_events(
        event_data["events"],
        event_data["global_ranges"]
    )

    # TODO: Add NONE checks

    print("-" * 30)

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

    # --- GENERATE PLOTS ---
    plotter = GaitPlotter(str(output_dir / "plots"))
    plotter.generate_plots_from_json(str(output_dir / "analysis.json"))

    print("-" * 30)

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
        filename_base=f"{an_name}_report"
    )


if __name__ == "__main__":
    main()
