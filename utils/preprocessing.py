import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Skeletons.halpe_skeleton import HALPE_SKELETON
from gaitStructs import GaitEventType
from utils.data import get_valid_range, get_keypoints, cubic_interpolate_nan, butterworth_filter, average_with_nones, \
    calculate_torso_height
from utils.jsonSerializer import KeypointSerializer, AnnotationSerializer


def split_video(keypoint_json_path, valid_range):
    json = KeypointSerializer.load(keypoint_json_path)
    split_keypoint_data = []
    for (start, end) in valid_range:
        clip = json[start:end + 1]
        split_keypoint_data.append(clip)

    return split_keypoint_data


def get_valid_annotations(annotations_json_path, valid_range):
    """
    Loads annotations and filters them to keep only those within a given range.

    Args:
        annotations_json_path (str): Path to the annotation JSON file.
        valid_range (range): A range object (e.g., range(start, end)) to filter by.

    Returns:
        dict: A new dictionary containing only the filtered annotations.
    """
    full_annotations = AnnotationSerializer.load(annotations_json_path)
    filtered_annotations = {"left": [], "right": []}

    for side in ["left", "right"]:
        for event in full_annotations["annotations"][side]:
            if event.frame in valid_range:
                filtered_annotations[side].append(event)

    return {"metadata": full_annotations["metadata"], "annotations": filtered_annotations}


def preprocess_keypoints(keypoints: dict, frame_rate: float):
    """
    Applies cubic interpolation and a Butterworth filter to smooth keypoint data.

    Args:
        keypoints (dict): The dictionary of keypoint lists.
        frame_rate (float): The frame rate of the video for the filter.

    Returns:
        dict: The dictionary with smoothed keypoint data.
    """
    processed_keypoints = {}
    for name, coords_list in keypoints.items():
        if not any(c is not None and c[0] is not None for c in coords_list):
            processed_keypoints[name] = coords_list
            continue

        # Separate x, y, and confidence scores
        x_coords = [c[0] if c is not None else None for c in coords_list]
        y_coords = [c[1] if c is not None else None for c in coords_list]
        conf_scores = [c[2] if c is not None else None for c in coords_list]

        # Interpolate and filter X coordinates
        x_processed = cubic_interpolate_nan(x_coords)
        x_processed = butterworth_filter(x_processed, cutoff=5, order=4, fs=frame_rate)

        # Interpolate and filter Y coordinates
        y_processed = cubic_interpolate_nan(y_coords)
        y_processed = butterworth_filter(y_processed, cutoff=5, order=4, fs=frame_rate)

        # Recombine into list of tuples, keeping original confidence
        processed_keypoints[name] = list(zip(x_processed, y_processed, conf_scores))

    return processed_keypoints


def normalize_coords(keypoints: dict):
    """
    Normalizes keypoint coordinates based on torso height and makes them
    relative to the 'HIP' keypoint (local coordinate system).

    Args:
        keypoints (dict): A dictionary of keypoint lists, where each list
                          contains (x, y, confidence) tuples for each frame.

    Returns:
        dict: A new dictionary with the normalized and localized keypoints.
    """
    torso_heights = calculate_torso_height(keypoints["HIP"], keypoints["NECK"])
    num_frames = len(keypoints["HIP"])

    # Create a new dictionary to store the normalized keypoints
    normalized_keypoints = {name: [(None, None, None)] * num_frames for name in keypoints.keys()}

    for i in range(num_frames):
        hip_coord = keypoints["HIP"][i]
        torso_h = torso_heights[i]

        # If we don't have a hip or a torso height for this frame, we can't normalize.
        if hip_coord is None or hip_coord[0] is None or torso_h is None or torso_h == 0:
            continue

        hip_x, hip_y, _ = hip_coord

        for keypoint_name, keypoint_list in keypoints.items():
            current_coord = keypoint_list[i]
            if current_coord is not None and current_coord[0] is not None:
                x, y, conf = current_coord
                normalized_x = (x - hip_x) / torso_h
                normalized_y = (y - hip_y) / torso_h
                normalized_keypoints[keypoint_name][i] = (normalized_x, normalized_y, conf)

    return normalized_keypoints


def visualize_normalized_skeleton(normalized_keypoints: dict, skeleton_definition):
    fig, ax = plt.subplots(figsize=(8, 10))
    plt.suptitle("Normalized Skeleton")

    frame_index = 0
    num_frames = len(next(iter(normalized_keypoints.values())))

    # Get skeleton links from the definition
    links = skeleton_definition.links

    def draw_frame(idx):
        ax.clear()
        ax.set_title(f"Frame: {idx}")

        # Collect valid points for the current frame
        points = {}
        for name, coords_list in normalized_keypoints.items():
            coord = coords_list[idx]
            if coord and coord[0] is not None:
                points[name] = (coord[0], coord[1])

        if not points:
            ax.text(0.5, 0.5, 'No data for this frame', ha='center', va='center', transform=ax.transAxes)
        else:
            # Draw skeleton links (bones)
            for link_name, (p1_name, p2_name) in links.items():
                p1_name, p2_name = p1_name.name, p2_name.name
                if p1_name in points and p2_name in points:
                    ax.plot([points[p1_name][0], points[p2_name][0]],
                            [points[p1_name][1], points[p2_name][1]], 'r-')

            # Draw keypoints (joints)
            x_coords, y_coords = zip(*points.values())
            ax.scatter(x_coords, y_coords, s=50, zorder=3)

        # Set plot limits and aspect ratio
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(1.5, -1.5)  # Invert Y-axis to match image coordinates
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        fig.canvas.draw()

    def on_press(event):
        nonlocal frame_index
        if event.key == 'right':
            frame_index = (frame_index + 1) % num_frames
        elif event.key == 'left':
            frame_index = (frame_index - 1 + num_frames) % num_frames
        draw_frame(frame_index)

    fig.canvas.mpl_connect('key_press_event', on_press)
    draw_frame(frame_index)
    plt.show()


def create_clips(normalized_keypoints: dict, original_keypoints: dict, valid_ranges: list):
    """
    Splits normalized keypoints into clips, detects walking direction, and mirrors
    clips to ensure all appear to walk in the same direction (right).

    Args:
        normalized_keypoints (dict): The dictionary of normalized keypoint data.
        original_keypoints (dict): The dictionary of original (pre-normalization)
                                   keypoint data, used to detect direction.
        valid_ranges (list): A list of (start, end) tuples for each clip.

    Returns:
        list: A list of clips. Each clip is a dictionary of keypoint lists.
    """
    clips = []
    for start, end in valid_ranges:
        # Determine walking direction from the original, unfiltered hip movement.
        # We use original data to get a clear start/end signal without filter artifacts.
        hip_start_x = original_keypoints["HIP"][start][0]
        hip_end_x = original_keypoints["HIP"][end][0]

        # If start or end is None, we can't determine direction, so we skip.
        if hip_start_x is None or hip_end_x is None:
            continue

        # Assume we want all clips to look like the person is walking to the right.
        # If x at the end is less than x at the start, the person was walking left.
        should_mirror = hip_end_x < hip_start_x

        # Create the clip by slicing the normalized data
        clip_data = {}
        for name, coords_list in normalized_keypoints.items():
            clip_data[name] = coords_list[start:end + 1]

        if should_mirror:
            for name, coords_list in clip_data.items():
                # Mirror the x-axis by negating the x-coordinate.
                # The y-coordinate and confidence remain unchanged.
                clip_data[name] = [(-x, y, conf) if x is not None else (None, None, None)
                                   for x, y, conf in coords_list]

        clips.append(clip_data)

    return clips


def create_processed_clips(
        keypoints_path: str,
        annotations_path: str,
        skeleton_definition,
        required_keypoints: list,
        confidence_threshold: float = 0.5,
        exclude_ratio: float = 0.1,
        create_labels: bool = True
):
    """
    A full pipeline to load, process, normalize, and split keypoints into standardized clips.

    This function serves as a base for both ML training (with labels) and other
    processing tasks (without labels).

    Args:
        keypoints_path (str): Path to the keypoints JSON file.
        annotations_path (str): Path to the annotations JSON file.
        skeleton_definition: The skeleton definition object.
        required_keypoints (list): A list of keypoint names to load.
        confidence_threshold (float): Minimum confidence to consider a keypoint valid.
        exclude_ratio (float): The percentage of the walking path to exclude from the start/end.
        create_labels (bool): If True, generates one-hot encoded labels for each clip.

    Returns:
        tuple: A tuple containing:
            - list: A list of processed clips. Each clip is a dictionary of keypoint lists.
            - list or None: A list of corresponding labels for each clip if create_labels is True, otherwise None.
    """
    # 1. Load and prepare keypoints
    # Essential keypoints needed for core processing (normalization, direction detection)
    essential_keypoints = {"HIP", "LEFT_HIP", "RIGHT_HIP", "NECK", "LEFT_SHOULDER", "RIGHT_SHOULDER"}

    # Combine user-requested keypoints with essential ones, ensuring no duplicates
    all_required_keypoints = list(set(required_keypoints) | essential_keypoints)

    keypoints, valid_indices = get_keypoints(
        keypoint_json_path=keypoints_path,
        skeleton_definition=skeleton_definition,
        items_to_get=all_required_keypoints,
        confidence_threshold=confidence_threshold
    )

    # Copy the original keypoints for direction detection later
    original_keypoints_for_direction = {name: list(coords) for name, coords in keypoints.items()}

    # If the "HIP" keypoint data is all None, calculate it by averaging
    if all(coord[0] is None for coord in keypoints["HIP"]):
        lh = keypoints["LEFT_HIP"]
        rh = keypoints["RIGHT_HIP"]
        keypoints["HIP"] = average_with_nones(lh, rh)

    # If the "NECK" keypoint data is all None, calculate it by averaging
    if all(coord[0] is None for coord in keypoints["NECK"]):
        ls = keypoints["LEFT_SHOULDER"]
        rs = keypoints["RIGHT_SHOULDER"]
        keypoints["NECK"] = average_with_nones(ls, rs)

    # 2. Smooth and normalize the keypoints
    frame_rate = AnnotationSerializer.load(annotations_path)["metadata"]["fps"]
    keypoints = preprocess_keypoints(keypoints, frame_rate)
    normalized_keypoints = normalize_coords(keypoints)

    # 3. Identify and split into clips
    trimmed_valid_range = get_valid_range(
        np.array([coord[0] for coord in keypoints["HIP"]]),
        frame_rate,
        exclude_ratio
    )
    clips = create_clips(
        normalized_keypoints=normalized_keypoints,
        original_keypoints=original_keypoints_for_direction,
        valid_ranges=trimmed_valid_range
    )

    # 4. Optionally create labels
    all_labels = None
    if create_labels:
        all_labels = []
        global_valid_range = [(start + valid_indices[0], end + valid_indices[0]) for start, end in trimmed_valid_range]

        for start, end in global_valid_range:
            segment_size = end - start + 1
            labels = [[0, 0, 0, 0] for _ in range(segment_size)]
            annotations = get_valid_annotations(
                annotations_json_path=annotations_path,
                valid_range=range(start, end + 1)
            )
            for side in ["left", "right"]:
                for annotation in annotations["annotations"][side]:
                    state = [0, 0, 0, 0]
                    if annotation.event_type == GaitEventType.HEEL_STRIKE:
                        state = [1, 0, 0, 0] if side == "left" else [0, 0, 1, 0]
                    elif annotation.event_type == GaitEventType.TOE_OFF:
                        state = [0, 1, 0, 0] if side == "left" else [0, 0, 0, 1]
                    labels[annotation.frame - start] = state
            all_labels.append(labels)

    return clips, all_labels


if __name__ == "__main__":
    exclude_ratio = 0.1
    keypoints_path = "../dataset/PROCESSED/60/KEYPOINTS/NM_001.json"
    annotations_path = "../annotations/60/NM_001.json"
    skeleton_definition = HALPE_SKELETON
    required_keypoints = ["HIP", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_SHOULDER", "RIGHT_SHOULDER", "NECK"]

    clips, labels = create_processed_clips(
        keypoints_path=keypoints_path,
        annotations_path=annotations_path,
        skeleton_definition=skeleton_definition,
        required_keypoints=required_keypoints,
        exclude_ratio=exclude_ratio,
        create_labels=True
    )

    # Visualization
    if clips:
        print(f"Created {len(clips)} standardized clip(s). Visualizing the first one.")
        visualize_normalized_skeleton(clips[0], skeleton_definition)

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
