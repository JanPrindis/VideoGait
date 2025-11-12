import numpy as np

from Skeletons.halpe_skeleton import HALPE_SKELETON
from utils.data import get_valid_range, get_keypoints
from utils.jsonSerializer import KeypointSerializer, AnnotationSerializer


def split_video(keypoint_json_path, skeleton_definition, frame_rate=60, exclude_ratio=0.1):
    keypoints, valid_indices = get_keypoints(
        keypoint_json_path,
        skeleton_definition,
        ["HIP"])

    hip = np.array([coord[0] for coord in keypoints["HIP"]])
    trimmed_valid_range = get_valid_range(hip, frame_rate, exclude_ratio)
    global_valid_range = [(start + valid_indices[0], end + valid_indices[0]) for start, end in trimmed_valid_range]

    json = KeypointSerializer.load(keypoint_json_path)
    split_keypoint_data = []
    for (start, end) in global_valid_range:
        clip = json[start:end + 1]
        split_keypoint_data.append(clip)

    return split_keypoint_data


def get_valid_annotations(annotations_json_path, video_keypoints):
    if len(video_keypoints) < 2:
        return

    # Get frame numbers from image_id attribute ("FRAME_NUMBER.jpg")
    valid_range = range(
        int(video_keypoints[0]["image_id"].split(".")[0]),
        int(video_keypoints[-1]["image_id"].split(".")[0]))

    annotations_data = AnnotationSerializer.load(annotations_json_path)

    for side in ["left", "right"]:
        # Iterate backwards to safely delete elements
        for i in range(len(annotations_data["annotations"][side]) - 1, -1, -1):
            if annotations_data["annotations"][side][i].frame not in valid_range:
                del annotations_data["annotations"][side][i]

    return annotations_data


if __name__ == "__main__":
    data = split_video(
        keypoint_json_path="../results/test.json",
        skeleton_definition=HALPE_SKELETON,
        frame_rate=60,
        exclude_ratio=0.1
    )

    annotations = get_valid_annotations("../annotations/NM/001_.json", data[0])
    left_events = annotations["annotations"]["left"]
    right_events = annotations["annotations"]["right"]

    # TODO: Training pre-processing
    # 1. Split video into clips
    # 2. Filter out annotations for each clip
    # 3. Get required keypoints
    # 4. Normalize keypoints to height
    # 5. Calculate relative keypoint coords to hip, mirror if needed
    # 6. Calculate other input params

    # TODO: ML infer pre-processing
    # 1. Split video into clips
    # 2. Get required keypoints
    # 3. Normalize keypoints to height
    # 4. Calculate relative keypoint coords to hip, mirror if needed
    # 5. Calculate other input params

    # TODO: Basic pre-processing
    # 1. Split into clips
    # 2. Get required keypoints

    # TODO: Functions
    # [x] split_videos
    # [x] filter_annotations
    # [x] get_keypoints (from input params)
    # [ ] normalize_to_height - wont read the whole JSON, will be used on the go
    # [ ] get_relative_coords - wont read the whole JSON, will be used on the go

    pass
