import os
import shutil
from glob import glob
import cv2
from tqdm import tqdm
import subprocess

from skeletons.halpe_skeleton import HALPE_SKELETON
from rtmlib.infer import RTMLib
from interpolate import RIFE_interpolate
from utils.jsonSerializer import AnnotationSerializer
from utils.visualizer import Visualizer

_is_nvenc_available = None

def is_nvenc_available():
    """Checks if NVIDIA NVENC encoder is available in FFmpeg."""
    global _is_nvenc_available
    if _is_nvenc_available is None:
        try:
            # Run ffmpeg to list encoders and capture output
            result = subprocess.run(
                ['ffmpeg', '-encoders'],
                capture_output=True,
                text=True,
                check=True
            )
            # Check for the h264_nvenc encoder in the output
            _is_nvenc_available = 'h264_nvenc' in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg is not found or the command fails
            _is_nvenc_available = False
    return _is_nvenc_available


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def join_videos(first, second, output_path, output_name):
    os.makedirs(output_path, exist_ok=True)

    caps = [cv2.VideoCapture(first), cv2.VideoCapture(second)]
    tot_frame = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT)) + int(caps[1].get(cv2.CAP_PROP_FRAME_COUNT))
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(f"{output_path}/{output_name}", fourcc, fps, (width, height))

    pbar = tqdm(total=tot_frame, leave=False, desc=f"Merging", unit="Frame")
    for cap in caps:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            writer.write(frame)
            pbar.update(1)
        cap.release()

    pbar.close()
    writer.release()


def parse_filename(filename):
    base = os.path.basename(filename).replace(".MOV", "")
    parts = base.split("_")

    patient_id = parts[0]
    disease_type = parts[1]
    direction = parts[2]  # 01 / 02

    severity = None
    if len(parts) > 3:
        severity = parts[3]

    return patient_id, disease_type, direction, severity


def merge_videos(dataset_root_path, blacklist=None):
    all_videos = glob(f"{dataset_root_path}/**/*.MOV", recursive=True)

    # Filter out blacklisted videos
    blacklist = set(blacklist or [])
    all_videos = [v for v in all_videos if not any(bad in os.path.basename(v) for bad in blacklist)]

    pairs = {}
    message_log = []

    for vid in all_videos:
        patient_id, dtype, direction, severity = parse_filename(vid)
        key = (patient_id, dtype, severity)

        if key not in pairs:
            pairs[key] = {}

        pairs[key][direction] = vid

    total = len(pairs)
    pbar = tqdm(total=total, desc="Processing videos", unit="Patient")

    # The output directory is now flat, without subfolders for disease type
    out_dir = os.path.join(dataset_root_path, "MERGED")
    os.makedirs(out_dir, exist_ok=True)

    for (pid, dtype, sev), dct in pairs.items():
        suffix = f"_{sev}" if sev else ""

        if "01" in dct and "02" in dct:
            # Merge
            out_name = f"{dtype}_{pid}{suffix}.mp4"
            join_videos(dct["01"], dct["02"], out_dir, out_name)

        elif "01" in dct:
            out_name = f"{dtype}_{pid}{suffix}_01.mp4"
            shutil.copy(dct["01"], os.path.join(out_dir, out_name))
            message_log.append(f"[SingleCopy] {pid} {dtype} {sev}")

        elif "02" in dct:
            out_name = f"{dtype}_{pid}{suffix}_02.mp4"
            shutil.copy(dct["02"], os.path.join(out_dir, out_name))
            message_log.append(f"[SingleCopy] {pid} {dtype} {sev}")

        else:
            message_log.append(f"Something went terribly wrong for {pid} {dtype} {sev}")

        pbar.update(1)

    pbar.close()

    if message_log:
        print("Log:")
        for msg in message_log:
            print(msg)


def interpolate_video_fps(orig_video_path, out_video_path, target_fps):
    orig_fps = get_video_fps(orig_video_path)
    temp_video_path = None

    if orig_fps == target_fps:
        shutil.copy(orig_video_path, out_video_path)
        return

    encoder = "h264_nvenc" if is_nvenc_available() else "libx264"

    def run_ffmpeg_with_progress(command, total_frames):
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8')

        pbar = tqdm(total=total_frames, desc="minterpolate", unit="frame", leave=False)

        for line in process.stdout:
            if "frame=" in line:
                try:
                    parts = line.split()
                    frame_index = parts.index("frame=")
                    current_frame = int(parts[frame_index + 1])
                    pbar.update(current_frame - pbar.n)
                except (ValueError, IndexError):
                    pass # Ignore malformed lines
        pbar.close()
        process.wait()

    if orig_fps == 30:
        if target_fps == 60:
            RIFE_interpolate(video=orig_video_path, output=out_video_path, exp=1, ext="mp4")
        elif target_fps == 120:
            RIFE_interpolate(video=orig_video_path, output=out_video_path, exp=2, ext="mp4")
        else:
            raise ValueError(f"Unsupported target FPS for 30fps video: {target_fps}")
    elif orig_fps == 50:
        cap = cv2.VideoCapture(orig_video_path)
        orig_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if target_fps == 60:
            total_frames = int(orig_frame_count * (60 / orig_fps))
            command = ["ffmpeg", "-i", orig_video_path, "-vf", "minterpolate=fps=60", "-c:v", encoder, "-y", out_video_path]
            run_ffmpeg_with_progress(command, total_frames)
        elif target_fps == 120:
            temp_video_path = out_video_path.replace(".mp4", "_temp.mp4")
            total_frames = int(orig_frame_count * (60 / orig_fps))
            command = ["ffmpeg", "-i", orig_video_path, "-vf", "minterpolate=fps=60", "-c:v", encoder, "-y", temp_video_path]
            run_ffmpeg_with_progress(command, total_frames)
            RIFE_interpolate(video=temp_video_path, output=out_video_path, exp=1, ext="mp4")
        else:
            raise ValueError(f"Unsupported target FPS for 50fps video: {target_fps}")
    elif orig_fps == 24:
        cap = cv2.VideoCapture(orig_video_path)
        orig_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        temp_video_path_30 = out_video_path.replace(".mp4", "_temp_30.mp4")
        total_frames = int(orig_frame_count * (30 / orig_fps))
        command = ["ffmpeg", "-i", orig_video_path, "-vf", "minterpolate=fps=30", "-c:v", encoder, "-y", temp_video_path_30]
        run_ffmpeg_with_progress(command, total_frames)
        if target_fps == 60:
            RIFE_interpolate(video=temp_video_path_30, output=out_video_path, exp=1, ext="mp4")
        elif target_fps == 120:
            RIFE_interpolate(video=temp_video_path_30, output=out_video_path, exp=2, ext="mp4")
        else:
            raise ValueError(f"Unsupported target FPS for 24fps video: {target_fps}")
        temp_video_path = temp_video_path_30
    elif orig_fps == 60:
        if target_fps == 120:
            RIFE_interpolate(video=orig_video_path, output=out_video_path, exp=1, ext="mp4")
        else:
            raise ValueError(f"Unsupported target FPS for 60fps video: {target_fps}")
    else:
        raise ValueError(f"Unsupported original FPS: {orig_fps}")

    if temp_video_path and os.path.exists(temp_video_path):
        os.remove(temp_video_path)


def interpolate_all_videos(dataset_root_path, interpolate_to: list[int]):
    merged_dir = os.path.join(dataset_root_path, "MERGED")
    interpolated_root = os.path.join(dataset_root_path, "INTERPOLATED")

    all_videos = glob(f"{merged_dir}/*.mp4", recursive=True)

    for target_fps in interpolate_to:
        interpolated_dir = os.path.join(interpolated_root, str(target_fps))
        os.makedirs(interpolated_dir, exist_ok=True)

        pbar = tqdm(all_videos, desc=f"Interpolating videos to {target_fps}fps", unit="video")
        for video_path in pbar:
            # No subdirectories needed, filename contains all info
            base_name = os.path.basename(video_path)
            out_file = os.path.join(interpolated_dir, base_name)

            try:
                interpolate_video_fps(video_path, out_file, target_fps)
            except ValueError as e:
                print(f"Skipping {video_path}: {e}")


def process_videos(dataset_root_path, detector, visualizer=None):
    processed_root = os.path.join(dataset_root_path, "PROCESSED")

    modes_to_process = [
        {
            "name": "INTERPOLATED 60",
            "input_dir": os.path.join(dataset_root_path, "INTERPOLATED", "60"),
            "output_dir": os.path.join(processed_root, "60")
        },
        {
            "name": "INTERPOLATED 120",
            "input_dir": os.path.join(dataset_root_path, "INTERPOLATED", "120"),
            "output_dir": os.path.join(processed_root, "120")
        }
    ]

    for mode in modes_to_process:
        input_dir = mode["input_dir"]
        output_dir = mode["output_dir"]
        mode_name = mode["name"]

        all_videos = glob(f"{input_dir}/**/*.mp4", recursive=True)
        pbar = tqdm(all_videos, desc=f"Running inference ({mode_name})", unit="video")

        for video_path in pbar:
            keypoint_dir = os.path.join(output_dir, "KEYPOINTS")
            annotated_dir = os.path.join(output_dir, "ANNOTATED")
            os.makedirs(keypoint_dir, exist_ok=True)
            os.makedirs(annotated_dir, exist_ok=True)

            base_name = os.path.basename(video_path)
            json_name = base_name.replace('.mp4', '.json')
            annotated_name = base_name

            try:
                detector.detect(video_path=video_path, output_path=keypoint_dir)
            except Exception as e:
                print(f"[ERROR] Inference failed for {video_path}: {e}")
                continue

            try:
                if visualizer is not None:
                    visualizer.visualize(
                        original_video=video_path,
                        visualizer_output_path=annotated_dir,
                        visualizer_output_file=annotated_name,
                        detector_output_path=keypoint_dir,
                        detector_output_file=json_name,
                        confidence_threshold=0.6
                    )
            except Exception as e:
                print(f"[ERROR] Visualization failed for {video_path}: {e}")

        pbar.close()


def recalculate_annotations(annotations_root_path):
    all_files = glob(annotations_root_path + "/ORIGINAL/*.json")

    for out_framerate in [60, 120]:
        out_path = os.path.join(annotations_root_path, f"{out_framerate}")
        os.makedirs(out_path, exist_ok=True)

        pbar = tqdm(all_files, desc = f"Recalculating annotations to {out_framerate}fps", unit = "video", leave=False)
        for file in all_files:
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
    annotations_root_path = "./annotations"
    dataset_root_path = "./dataset"
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

    # Fix ONNX not finding CUDA dlls
    import onnxruntime
    # Preload DLLs from NVIDIA site packages
    onnxruntime.preload_dlls(directory="")

    # Merge
    merge_videos(dataset_root_path, blacklist)

    # Interpolate
    # This process can be slow. To optimize, first interpolate videos to 60fps.
    # Then, replace the original videos with these 60fps versions before interpolating to 120fps.
    # This strategy leverages RIFE for the second interpolation, which is significantly faster with a CUDA-enabled GPU,
    # as it avoids repeated use of FFmpeg's minterpolate.
    interpolate_all_videos(dataset_root_path, interpolate_to=[60, 120])

    # Process and visualize
    rtmlib = RTMLib()                                       # Can be any keypoint detector
    vis = Visualizer(skeleton_definition=HALPE_SKELETON)    # Skeleton is based on the detector's output

    # To skip visualization, pass None as visualizer in the parameter
    process_videos(
        dataset_root_path,
        detector=rtmlib,
        visualizer=vis
    )

    # Update annotations
    # This is not required, as the annotations are already pre-calculated.
    # Only use if the 60/120 folders are missing in the annotations folder.
    # recalculate_annotations(annotations_root_path)
