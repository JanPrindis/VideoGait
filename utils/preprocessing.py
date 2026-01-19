import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skeletons.halpe_skeleton import HALPE_SKELETON
from gaitStructs import GaitEventType
from utils.data import get_valid_range, get_keypoints, cubic_interpolate_nan, butterworth_filter, average_with_nones, \
    calculate_torso_height, calculate_distance, calculate_angle
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
        processed_keypoints[name] = list(zip(x_processed.tolist(), y_processed.tolist(), conf_scores))

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
        ax.set_ylim(1.8, -1.8)  # Invert Y-axis to match image coordinates
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
        skeleton_definition,
        required_keypoints: list,
        confidence_threshold: float = 0.5,
        exclude_ratio: float = 0.1,
        annotations_path: str | None = None,  # Now fully optional
        frame_rate: float | None = None,      # Optional, but required if annotations are missing
        create_labels: bool = True
):
    has_annotations = annotations_path and os.path.exists(annotations_path)

    if has_annotations:
        # If annotations exist, they are the source of truth for FPS.
        determined_frame_rate = AnnotationSerializer.load(annotations_path)["metadata"]["fps"]
    elif frame_rate is not None:
        # If no annotations, use the manually provided frame rate.
        determined_frame_rate = frame_rate
    else:
        # If neither is available, we cannot proceed.
        raise ValueError("`frame_rate` must be provided when `annotations_path` is not specified.")

    # Disable label creation if we don't have annotations to create them from.
    if not has_annotations:
        create_labels = False

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

    if not valid_indices:
        print(f"No valid frames found in {keypoints_path} with confidence > {confidence_threshold}. Skipping.")
        return [], [], [], determined_frame_rate

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

    # Copy the original keypoints for direction detection later
    original_keypoints_for_direction = {name: list(coords) for name, coords in keypoints.items()}
    original_keypoints_for_direction = preprocess_keypoints(original_keypoints_for_direction, determined_frame_rate)

    # 2. Smooth and normalize the keypoints
    keypoints = preprocess_keypoints(keypoints, determined_frame_rate)
    normalized_keypoints = normalize_coords(keypoints)

    # 3. Identify and split into clips
    trimmed_valid_range = get_valid_range(
        np.array([coord[0] for coord in keypoints["HIP"]]),
        determined_frame_rate,
        exclude_ratio
    )
    clips = create_clips(
        normalized_keypoints=normalized_keypoints,
        original_keypoints=original_keypoints_for_direction,
        valid_ranges=trimmed_valid_range
    )

    # Calculate global ranges that correspond to the original video's frame indices
    global_valid_range = [(start + valid_indices[0], end + valid_indices[0]) for start, end in trimmed_valid_range]

    # 4. Optionally create labels
    # 4. Optionally create labels
    all_labels = None
    if create_labels:
        all_labels = []
        # We need to filter the ranges list just like we filter the clips list
        final_clips = []
        final_global_ranges = []

        for idx, (start, end) in enumerate(global_valid_range):
            annotations = get_valid_annotations(
                annotations_json_path=annotations_path,
                valid_range=range(start, end + 1)
            )

            # If a clip has no events, it's a bad segment. Skip it.
            if len(annotations["annotations"]["left"]) == 0 and len(annotations["annotations"]["right"]) == 0:
                continue

            # This clip is valid, so we build its labels and keep it
            segment_size = end - start + 1
            labels = [[0, 0, 0, 0] for _ in range(segment_size)]
            for side in ["left", "right"]:
                for annotation in annotations["annotations"][side]:
                    state = [0, 0, 0, 0]
                    if annotation.event_type == GaitEventType.HEEL_STRIKE:
                        state = [1, 0, 0, 0] if side == "left" else [0, 0, 1, 0]
                    elif annotation.event_type == GaitEventType.TOE_OFF:
                        state = [0, 1, 0, 0] if side == "left" else [0, 0, 0, 1]

                    frame_idx_in_clip = annotation.frame - start
                    if 0 <= frame_idx_in_clip < len(labels):
                        labels[frame_idx_in_clip] = state

            all_labels.append(labels)
            final_clips.append(clips[idx])
            final_global_ranges.append(global_valid_range[idx])

        clips = final_clips
        global_valid_range = final_global_ranges

    print(f"{os.path.basename(keypoints_path)} resulted in {len(clips)} clips!")
    return clips, all_labels, global_valid_range, determined_frame_rate


def generate_keypoint_features(clip_dict: dict, keypoint_names: list) -> dict:
    """Extracts (x, y) coordinates for specified keypoints."""
    features = {}
    for name in keypoint_names:
        coords = np.array(clip_dict[name], dtype=np.float32)
        # Replace any NaNs that resulted from None values
        coords[np.isnan(coords)] = 0.0

        features[f"{name}_x"] = coords[:, 0]
        features[f"{name}_y"] = coords[:, 1]
    return features


def generate_kinematic_features(clip_dict: dict, keypoint_names: list, frame_rate: float) -> dict:
    """Generates velocities and accelerations for specified keypoints."""
    features = {}
    dt = 1.0 / frame_rate
    for name in keypoint_names:
        coords = np.array(clip_dict[name], dtype=np.float32)[:, :2]

        # Replace any remaining NaNs with 0.0 before differentiation
        coords[np.isnan(coords)] = 0.0

        # Calculate velocity
        velocity = np.gradient(coords, dt, axis=0)
        features[f"{name}_vx"] = velocity[:, 0]
        features[f"{name}_vy"] = velocity[:, 1]

        # Calculate acceleration
        acceleration = np.gradient(velocity, dt, axis=0)
        features[f"{name}_ax"] = acceleration[:, 0]
        features[f"{name}_ay"] = acceleration[:, 1]
    return features


def generate_angle_features(clip_dict: dict, angle_triplets: list[tuple[str, str, str]]) -> dict:
    """Generates joint angles based on user-defined triplets."""
    features = {}
    kp_arrays = {name: np.array(coords, dtype=np.float32)[:, :2] for name, coords in clip_dict.items()}

    for p1_name, p2_name, p3_name in angle_triplets:
        feature_name = f"angle_{p1_name}-{p2_name}-{p3_name}"
        features[feature_name] = calculate_angle(kp_arrays[p1_name], kp_arrays[p2_name], kp_arrays[p3_name])
    return features


def generate_distance_features(clip_dict: dict, distance_pairs: list[tuple[str, str]]) -> dict:
    """Generates distances based on user-defined pairs."""
    features = {}
    kp_arrays = {name: np.array(coords)[:, :2] for name, coords in clip_dict.items()}

    for p1_name, p2_name in distance_pairs:
        feature_name = f"dist_{p1_name}-{p2_name}"
        features[feature_name] = calculate_distance(kp_arrays[p1_name], kp_arrays[p2_name])
    return features


def build_feature_matrix(
        clip_dict: dict,
        frame_rate: float,
        keypoints: list = None,
        kinematics_keypoints: list = None,
        angle_triplets: list = None,
        distance_pairs: list = None
) -> tuple[np.ndarray, list[str]]:
    """
    Constructs a feature matrix and corresponding labels matrix (optional).
    """
    all_features_dict = {}

    # Generate each requested feature type
    if keypoints:
        all_features_dict.update(generate_keypoint_features(clip_dict, keypoints))
    if kinematics_keypoints:
        all_features_dict.update(generate_kinematic_features(clip_dict, kinematics_keypoints, frame_rate))
    if angle_triplets:
        all_features_dict.update(generate_angle_features(clip_dict, angle_triplets))
    if distance_pairs:
        all_features_dict.update(generate_distance_features(clip_dict, distance_pairs))

    # Get the final ordered list of feature names
    feature_names = sorted(all_features_dict.keys())
    if not feature_names:
        return np.array([]), []

    # Get the number of frames from the first available feature
    num_frames = len(next(iter(all_features_dict.values())))
    num_features = len(feature_names)

    # Assemble the final matrix
    feature_matrix = np.zeros((num_frames, num_features), dtype=np.float32)
    for i, name in enumerate(feature_names):
        feature_matrix[:, i] = all_features_dict[name]

    # Final cleanup of any NaNs that may have been generated
    feature_matrix[np.isnan(feature_matrix)] = 0.0

    return feature_matrix, feature_names


def generate_features(
        keypoints_path: str,
        skeleton_definition,
        confidence_threshold: float,
        exclude_ratio: float,
        annotations_path: str | None = None,
        frame_rate: float | None = None,
        keypoints: list | None = None,
        kinematics_keypoints: list | None = None,
        angle_triplets: list | None = None,
        distance_pairs: list | None = None
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
    """
     A unified preprocessor that generates feature matrices based on a list of desired feature types.
     """
    clips_raw, all_labels_raw, global_ranges, determined_frame_rate = create_processed_clips(
        keypoints_path=keypoints_path,
        annotations_path=annotations_path,
        frame_rate=frame_rate, # Pass it down
        skeleton_definition=skeleton_definition,
        required_keypoints=keypoints,
        confidence_threshold=confidence_threshold,
        exclude_ratio=exclude_ratio,
        create_labels=bool(annotations_path) # Only create labels if path is given
    )

    if not clips_raw:
        return [], [], []

    all_feature_matrices = []
    all_label_matrices = []

    # Build the arguments for the matrix builder based on feature_types
    builder_kwargs = {}
    if keypoints:
        builder_kwargs["keypoints"] = keypoints
    if kinematics_keypoints:
        builder_kwargs["kinematics_keypoints"] = kinematics_keypoints
    if angle_triplets:
        builder_kwargs["angle_triplets"] = angle_triplets
    if distance_pairs:
        builder_kwargs["distance_pairs"] = distance_pairs

    for i, clip_dict_raw in enumerate(clips_raw):
        feature_matrix, _ = build_feature_matrix(
            clip_dict=clip_dict_raw,
            frame_rate=determined_frame_rate,
            **builder_kwargs
        )

        if feature_matrix.size == 0:
            continue

        all_feature_matrices.append(feature_matrix)
        # Only append labels if they were generated
        if all_labels_raw:
            all_label_matrices.append(np.array(all_labels_raw[i], dtype=np.float32))

    return all_feature_matrices, all_label_matrices, global_ranges


if __name__ == "__main__":
    exclude_ratio = 0.1
    keypoints_path = "../dataset/PROCESSED/60/KEYPOINTS/KOA_001_EL.json"
    annotations_path = "../annotations/60/KOA_001_EL.json"
    skeleton_definition = HALPE_SKELETON
    required_keypoints = ["HIP", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_SHOULDER", "RIGHT_SHOULDER", "NECK"]
    required_kinematics = ["HIP", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_SHOULDER", "RIGHT_SHOULDER", "NECK"]
    required_angles = [("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"), ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")]
    required_distances = [("HIP", "LEFT_FOOT_INDEX"), ("HIP", "RIGHT_FOOT_INDEX")]

    features, labels, ranges = generate_features(
        keypoints_path=keypoints_path,
        skeleton_definition=skeleton_definition,
        confidence_threshold=0.5,
        exclude_ratio=exclude_ratio,
        annotations_path=annotations_path,
        frame_rate=None,
        keypoints=required_keypoints,
        kinematics_keypoints=required_kinematics,
        angle_triplets=required_angles,
        distance_pairs=required_distances,
    )


    clips, labels, _ = create_processed_clips(
        keypoints_path=keypoints_path,
        annotations_path=annotations_path,
        skeleton_definition=skeleton_definition,
        required_keypoints=required_keypoints,
        exclude_ratio=exclude_ratio,
        create_labels=True
    )

    # Visualization
    if clips:
        print(f"Created {len(clips)} standardized clip(s)")
        for clip in clips:
            visualize_normalized_skeleton(clip, skeleton_definition)

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