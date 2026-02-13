"""
This module provides functionality for running pose estimation on video files.

It handles configuration loading, path resolution, and execution of the configured pose detector.
"""
import sys
import argparse
from pathlib import Path

from utils.config_models import AppConfig
from utils.config_utils import load_and_validate_yaml

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from detectors.builder import build_detector_from_file
from utils.logger import log


def resolve_path(path_str: str, root: Path):
    """
    Resolves a file path, handling both absolute and relative paths.

    Args:
        path_str (str): The path string to resolve.
        root (Path): The root path to use if `path_str` is relative.

    Returns:
        Path | None: The resolved absolute path, or None if input is empty.
    """
    if not path_str:
        return None

    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    else:
        return (root / p).resolve()


def run_pose_extraction(app_config: AppConfig, video_path: str, output_root: str = None, run_name: str = None):
    """
    Runs the pose extraction pipeline on a single video.

    It loads the detector configuration, initializes the detector, and processes the video.
    Results are saved to the configured output directory.

    Args:
        app_config (AppConfig): The application configuration object.
        video_path (str): Path to the input video file.
        output_root (str, optional): Override for the output root directory.
        run_name (str, optional): Name for the output subdirectory (defaults to video filename).

    Raises:
        FileNotFoundError: If the video file or detector config is not found.
        ValueError: If output path is not defined or config is invalid.
    """
    # Input file path validation
    vid_path = resolve_path(video_path, project_root)
    if not vid_path or not vid_path.exists():
        raise FileNotFoundError(f"Video file not found: {vid_path}")

    # Resolve ROOT Output path
    root_out_path = None

    # Direct input form CLI has priority
    if output_root:
        root_out_path = resolve_path(output_root, project_root)

    # Otherwise use path from app config
    if not root_out_path:
        cfg_out_str = app_config.output.output_root_dir
        if cfg_out_str:
            root_out_path = resolve_path(cfg_out_str, project_root)

    if not root_out_path:
        raise ValueError("Output root path not defined! Set it in 'output.output_root_dir' or via CLI.")

    # Determine Sub-directory name
    # If run_name specified, use it, otherwise use video name
    sub_dir_name = run_name if run_name else vid_path.stem

    # Construct Final Path
    final_out_path = root_out_path / sub_dir_name

    # Safety check & Create
    if not final_out_path.exists():
        log("INFERENCE", f"Creating output directory: {final_out_path}", level="info")
        final_out_path.mkdir(parents=True, exist_ok=True)

    # Load detector config file
    pose_det_section = app_config.pose_detector
    det_config_rel = pose_det_section.config_path

    if not det_config_rel:
        raise ValueError("App config missing 'pose_detector.config_path'")

    det_config_path = resolve_path(det_config_rel, project_root)

    if not det_config_path.exists():
        raise FileNotFoundError(f"Detector config not found: {det_config_path}")

    # Build & Run
    log("BUILDER", f"Building detector from: {det_config_path.name}", level="info")
    detector = build_detector_from_file(str(det_config_path))

    log("DETECTOR", f"Processing: {vid_path.name}", level="info")
    try:
        detector.detect(str(vid_path), str(final_out_path))

    except Exception as e:
        log("DETECTOR", f"Critical Error: {e}", level="error")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--video", "-v", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default=None, help="Override output root")

    args = parser.parse_args()

    cfg_path = resolve_path(args.config, project_root)

    if not cfg_path.exists():
        log("INFERENCE", f"Error: Config not found at {cfg_path}", level="error")
        sys.exit(1)

    app_config: AppConfig = load_and_validate_yaml(str(cfg_path), AppConfig)

    run_pose_extraction(app_config, args.video, args.output)
