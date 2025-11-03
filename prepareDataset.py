import os
import shutil
from glob import glob
import cv2
from tqdm import tqdm

from Skeletons.halpe_skeleton import HALPE_SKELETON
from inferVideo import RTMLib
from utils.data import create_folder_if_not_exists
from utils.visualizer import Visualizer


def join_videos(first, second, output_path, output_name):
    create_folder_if_not_exists(output_path)

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

    for (pid, dtype, sev), dct in pairs.items():
        out_dir = os.path.join(dataset_root_path, "MERGED", dtype)
        os.makedirs(out_dir, exist_ok=True)

        suffix = f"{sev}" if sev else ""

        if "01" in dct and "02" in dct:
            # Merge
            out_name = f"{pid}_{suffix}.mp4"
            join_videos(dct["01"], dct["02"], out_dir, out_name)

        elif "01" in dct:
            out_name = f"{pid}_{suffix}_01.mp4"
            shutil.copy(dct["01"], os.path.join(out_dir, out_name))
            message_log.append(f"[SingleCopy] {pid} {dtype} {sev}")

        elif "02" in dct:
            out_name = f"{pid}_{suffix}_02.mp4"
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


def process_videos(dataset_root_path):
    merged_dir = os.path.join(dataset_root_path, "MERGED")
    keypoint_root = os.path.join(dataset_root_path, "PROCESSED", "KEYPOINTS")
    annotated_root = os.path.join(dataset_root_path, "PROCESSED", "ANNOTATED")

    create_folder_if_not_exists(keypoint_root)
    create_folder_if_not_exists(annotated_root)

    rtmlib = RTMLib()
    vis = Visualizer(skeleton_definition=HALPE_SKELETON)

    all_videos = glob(f"{merged_dir}/**/*.mp4", recursive=True)
    pbar = tqdm(all_videos, desc="Running inference", unit="video")

    for video_path in pbar:
        rel_dir = os.path.relpath(os.path.dirname(video_path), merged_dir)
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        # Output paths
        keypoint_dir = os.path.join(keypoint_root, rel_dir)
        annotated_dir = os.path.join(annotated_root, rel_dir)
        create_folder_if_not_exists(keypoint_dir)
        create_folder_if_not_exists(annotated_dir)

        json_name = f"{base_name}.json"
        annotated_name = f"{base_name}.mp4"

        # Infer
        pbar.set_description(f"Infer: {base_name}")
        try:
            rtmlib.inferVideo(in_path=video_path, out_path=keypoint_dir)
        except Exception as e:
            print(f"[ERROR] Failed inference for {video_path}: {e}")
            continue

        # Visualization
        pbar.set_description(f"Annotate: {base_name}")
        try:
            vis.visualize(
                original_video=video_path,
                visualizer_output_path=annotated_dir,
                visualizer_output_file=annotated_name,
                detector_output_path=keypoint_dir,
                detector_output_file=json_name,
                confidence_threshold=0.6
            )
        except Exception as e:
            print(f"[ERROR] Failed visualization for {video_path}: {e}")

    pbar.close()


if __name__ == "__main__":
    dataset_root_path = ""
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

    # merge_videos(dataset_root_path, blacklist)
    process_videos(dataset_root_path)
