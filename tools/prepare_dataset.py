import os
import shutil
import sys
import time
from glob import glob
from pathlib import Path
from tqdm import tqdm
import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from detectors.builder import build_detector_from_file
from utils.json_serializer import AnnotationSerializer
from utils.video_processing import smart_interpolate, create_video_writer
from utils.visualizer import GaitVisualizer
from utils.logger import log
from utils.config_models import AppConfig, PreprocessingConfig, VisualizationConfig, VideoOutputsConfig, \
    PoseDetectorRef, EventDetectorConfig, HeuristicConfig

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.signal_count = 0

    def exit_gracefully(self, signum, frame):
        self.signal_count += 1
        if self.signal_count >= 2:
            print()
            log("DATASET", "Force kill received! Exiting immediately.", level="error")
            log("DATASET", "WARNING: Check the output folder for potentially corrupted/incomplete files.", level="warning")
            os._exit(1)

        self.kill_now = True
        print()  # Newline to clear tqdm line
        log("DATASET", "Stop signal received! Finishing current task before exiting...", level="warning")
        log("DATASET", "Press Ctrl+C again to force quit immediately.", level="warning")

# ==========================================
# HELPERS
# ==========================================

def _parse_filename(filename):
    """Parses patient info from filename: ID_Disease_Direction_Severity.ext"""
    base = os.path.basename(filename)
    name_only = os.path.splitext(base)[0] # Remove extension
    parts = name_only.split("_")

    if len(parts) < 3:
        return None # Invalid format

    patient_id = parts[0]
    disease_type = parts[1]
    direction = parts[2]  # 01 / 02
    severity = parts[3] if len(parts) > 3 else None

    return patient_id, disease_type, direction, severity

def _get_detector_name(config_path):
    """Extracts detector name from config path"""
    stem = Path(config_path).stem
    return stem.replace("_config", "").replace("config_", "")


def _create_minimal_config(config_path, confidence=0.4):
    """
    Creates a basic app configuration object to satisfy GaitVisualizer.
    """
    return AppConfig(
        preprocessing=PreprocessingConfig(
            confidence_threshold=confidence,
        ),

        visualization=VisualizationConfig(
            colors={
                "left": "#00BFFF",
                "right": "#FF4500",
                "center": "#ADFF2F",
                "dimmed": [128, 128, 128]
            },
            line_thickness=2,
            keypoint_radius=4,
            footprint_duration=0,
            outputs=VideoOutputsConfig(
                enable_overlay=True,
                enable_kinematics=False,
                enable_logic=False
            )
        ),

        pose_detector=PoseDetectorRef(
            config_path=config_path
        ),

        # Dummy event detector
        event_detector=EventDetectorConfig(
            method="Heuristic",
            heuristic=HeuristicConfig(method="dummy")
        )
    )

# ==========================================
# MAIN PIPELINE STEPS
# ==========================================

def merge_videos(dataset_root, blacklist, killer=None):
    """Merges 01 and 02 videos into one continuous shot."""
    log("DATASET", "Merging videos...", level="info")

    # Input: Recursive search for MOV
    all_videos = glob(f"{dataset_root}/**/*.MOV", recursive=True)
    all_videos = [v for v in all_videos if os.path.basename(v) not in blacklist]

    # Output dir
    out_dir = os.path.join(dataset_root, "MERGED")
    os.makedirs(out_dir, exist_ok=True)

    # Group by patient/disease/severity
    pairs = {}
    for vid in all_videos:
        parsed = _parse_filename(vid)
        if not parsed: continue

        pid, dtype, direction, sev = parsed
        key = (pid, dtype, sev)

        if key not in pairs: pairs[key] = {}
        pairs[key][direction] = vid

    # Process
    for (pid, dtype, sev), dct in tqdm(pairs.items(), desc="Merging pairs"):
        if killer and killer.kill_now:
            log("DATASET", "Graceful exit during merging.", level="warning")
            return

        suffix = f"_{sev}" if sev else ""

        # Define output name
        if "01" in dct and "02" in dct:
            out_name = f"{dtype}_{pid}{suffix}.mp4"
            out_path = os.path.join(out_dir, out_name)

            if os.path.exists(out_path):
                continue

            _join_videos(dct["01"], dct["02"], out_path)

        elif "01" in dct:
            out_name = f"{dtype}_{pid}{suffix}_01.mp4"
            out_path = os.path.join(out_dir, out_name)
            if os.path.exists(out_path):
                continue
            shutil.copy(dct["01"], out_path)

        elif "02" in dct:
            out_name = f"{dtype}_{pid}{suffix}_02.mp4"
            out_path = os.path.join(out_dir, out_name)
            if os.path.exists(out_path):
                continue
            shutil.copy(dct["02"], out_path)


def _join_videos(vid1, vid2, out_path):
    """Helper for joining two videos."""
    import cv2
    caps = [cv2.VideoCapture(vid1), cv2.VideoCapture(vid2)]

    fps = caps[0].get(cv2.CAP_PROP_FPS)
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = create_video_writer(out_path, fps, w, h)

    for cap in caps:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            writer.write(frame)
        cap.release()
    writer.release()


def interpolate(dataset_root, target_fps_list, killer=None):
    """
    Interpolates merged videos to target framerate.
    Uses chaining: Output of lower FPS is used as input for higher FPS to save time.
    """
    # Order target framerate for chaining
    target_fps_list = sorted(target_fps_list)
    log("DATASET", f"Interpolating to {target_fps_list} FPS (Chained)...", level="info")

    merged_dir = os.path.join(dataset_root, "MERGED")
    interp_root = os.path.join(dataset_root, "INTERPOLATED")

    # Find all videos in MERGED
    src_videos = glob(os.path.join(merged_dir, "*.mp4"))

    # Map: File name -> Path to latest version
    current_source_map = {os.path.basename(v): v for v in src_videos}

    for fps in target_fps_list:
        out_dir = os.path.join(interp_root, str(fps))
        os.makedirs(out_dir, exist_ok=True)

        if killer and killer.kill_now:
            return

        for filename, src_path in tqdm(current_source_map.items(), desc=f"Interpolating to {fps}"):
            if killer and killer.kill_now:
                log("DATASET", "Graceful exit during interpolation.", level="warning")
                return

            out_path = os.path.join(out_dir, filename)

            if os.path.exists(out_path):
                # If exists - just update the map
                current_source_map[filename] = out_path
                continue

            try:
                # Interpolate from previous version
                smart_interpolate(src_path, out_path, target_fps=fps)

                # Success - update map
                current_source_map[filename] = out_path

            except Exception as e:
                log("DATASET", f"Failed to interpolate {filename}: {e}", level="error")
                # On error, we don't update the map as a fallback measure


def detect_and_visualize(dataset_root, target_fps_list, detector_configs, skip_visualization=False, killer=None):
    """Runs pose detection and basic visualization for all combinations."""
    log("DATASET", "Running detection and visualization...", level="info")

    interp_root = os.path.join(dataset_root, "INTERPOLATED")
    processed_root = os.path.join(dataset_root, "PROCESSED")

    # Iterate over FPS
    for fps in target_fps_list:
        if killer and killer.kill_now:
            return

        src_dir = os.path.join(interp_root, str(fps))
        if not os.path.exists(src_dir):
            log("DATASET", f"Skipping {fps} FPS (directory not found)", level="warning")
            continue

        videos = glob(os.path.join(src_dir, "*.mp4"))

        # Iterate over Detectors
        for config_path, conf_thresh in detector_configs:
            if killer and killer.kill_now:
                return

            det_name = _get_detector_name(config_path)

            log("DATASET", f"Processing: FPS={fps} | Detector={det_name}", level="info")

            # Prepare paths
            # PROCESSED/{detector}/{fps}/KEYPOINTS
            kp_out_dir = os.path.join(processed_root, det_name, str(int(fps)), "KEYPOINTS")
            # PROCESSED/{detector}/{fps}/ANNOTATED
            vis_out_dir = os.path.join(processed_root, det_name, str(int(fps)), "ANNOTATED")

            os.makedirs(kp_out_dir, exist_ok=True)
            os.makedirs(vis_out_dir, exist_ok=True)

            # Build detector
            try:
                detector = build_detector_from_file(config_path)
            except Exception as e:
                log("DATASET", f"Could not build detector {det_name}: {e}", level="error")
                continue

            # Build Visualizer with fake config
            fake_config = _create_minimal_config(config_path=config_path, confidence=conf_thresh)

            try:
                visualizer = GaitVisualizer(fake_config)
            except Exception as e:
                log("DATASET", f"Could not initialize Visualizer for {det_name}: {e}", level="error")
                continue

            # Process Videos
            for video_path in tqdm(videos, unit="vid"):
                if killer and killer.kill_now:
                    log("DATASET", "Graceful exit during detection/visualization.", level="warning")
                    return

                base_name = os.path.basename(video_path)
                json_name = base_name.replace(".mp4", ".json")

                json_out_path = os.path.join(kp_out_dir, json_name)

                # Detection
                if not os.path.exists(json_out_path):
                    try:
                        # Output path for detector is the directory, or full path depending on implementation.
                        # RTMLib wrapper typically takes output_path as directory.
                        detector.detect(video_path, kp_out_dir)
                    except Exception as e:
                        log("DATASET", f"Detection failed for {base_name}: {e}", level="error")
                        continue

                # Visualization
                # Visualizer creates its own filenames, usually {stem}_inspection.mp4

                if not skip_visualization:
                    # We need to check if it already exists to avoid re-rendering
                    expected_vis_output = os.path.join(vis_out_dir, base_name.replace(".mp4", "_inspection.mp4"))

                    if not os.path.exists(expected_vis_output):
                        try:
                            visualizer.process_video(
                                video_path=video_path,
                                output_root=vis_out_dir,
                                keypoints_data=json_out_path
                            )
                        except Exception as e:
                            log("DATASET", f"Visualization failed for {base_name}: {e}", level="error")


def recalculate_annotations(annotations_root_path, killer=None):
    all_files = glob(annotations_root_path + "/ORIGINAL/*.json")

    for out_framerate in [60, 120]:
        if killer and killer.kill_now:
            return

        out_path = os.path.join(annotations_root_path, f"{out_framerate}")
        os.makedirs(out_path, exist_ok=True)

        pbar = tqdm(all_files, desc = f"Recalculating annotations to {out_framerate}fps", unit = "video", leave=False)
        for file in all_files:
            if killer and killer.kill_now:
                log("DATASET", "Graceful exit during annotation recalculation.", level="warning")
                pbar.close()
                return

            base_name = os.path.basename(file)

            data = AnnotationSerializer.load(file)
            original_fps = int(data["metadata"]["fps"])
            ratio = out_framerate / original_fps
            serializer = AnnotationSerializer(
                output_path=out_path,
                output_file_name=base_name,
                fps=out_framerate
            )

            for side in ["left", "right"]:
                for event in data["annotations"][side]:
                    event.frame = round(event.frame * ratio)
                    serializer.add_event(side, event)

            serializer.save()
            pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    annotations_root_path = "../annotations"
    dataset_root_path = "../dataset"
    blacklist = [
        "002_NM_01.MOV", # Bad crop
        "004_NM_01.MOV", # Bad crop
        "015_NM_02.MOV", # Corrupted

        "001_PD_01_SV.MOV", # The detection fails if there are multiple people walking,
        "001_PD_02_SV.MOV", # so patients with severe Parkinson's Disease are excluded
        "002_PD_01_SV.MOV", #
        "003_PD_01_SV.MOV", #
        "003_PD_02_SV.MOV", #
    ]


    detector_configs = [
        (f"{PROJECT_ROOT}/configs/detectors/rtmlib_config.yaml", 0.3),
        (f"{PROJECT_ROOT}/configs/detectors/alphapose_config.yaml", 0.01),
        (f"{PROJECT_ROOT}/configs/detectors/mediapipe_config.yaml", 0.2),
    ]

    # Fix ONNX not finding CUDA dlls
    import onnxruntime
    # Preload DLLs from NVIDIA site packages
    onnxruntime.preload_dlls(directory="")

    killer = GracefulKiller()

    log("DATASET", "This takes a really long time, even on GPU!", level="warning")
    log("DATASET", "You can stop this script and resume later by pressing Ctrl+C.", level="warning")
    log("DATASET", "Press Ctrl+C again to force quit immediately.", level="warning")
    time.sleep(5)

    # Merge
    merge_videos(dataset_root_path, blacklist, killer=killer)

    # Interpolate
    # This process can be slow. To optimize, first interpolate videos to 60fps.
    # Then, replace the original videos with these 60fps versions before interpolating to 120fps.
    # This strategy leverages RIFE for the second interpolation, which is significantly faster with a CUDA-enabled GPU,
    # as it avoids repeated use of FFmpeg's minterpolate.
    interpolate(dataset_root_path, target_fps_list=[60, 120], killer=killer)

    # Process and visualize
    detect_and_visualize(
        dataset_root=dataset_root_path,
        detector_configs=detector_configs,
        # target_fps_list=[60, 120],
        target_fps_list=[120],
        skip_visualization=False,    # If you don't care about visualization, you can skip it
        killer=killer
    )

    # Update annotations
    # This is not required, as the annotations are already pre-calculated.
    # Only use if the 60/120 folders are missing in the annotations folder.
    # recalculate_annotations(annotations_root_path, killer=killer)
