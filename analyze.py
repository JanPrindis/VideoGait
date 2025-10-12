from enum import Enum

import numpy as np

from gaitDetectors.zeni import gait_detect_zeni
from gaitStructs import PhaseType, GaitPhase, Leg, compute_support_phases, GaitEvent, GaitEventType, build_phases_from_events
from utils.jsonSerializer import KeypointSerializer
from Skeletons.halpe_skeleton import HALPE_SKELETON
from utils.data import extract_keypoints, get_valid_range, trim
from utils.preprocessing import cubic_interpolate_nan, butterworth_filter, find_minima_maxima
from visualizeGaitPhases import visualize_gait_phases, print_statistics

# Result metadata
result_name = "test"
json = KeypointSerializer.load(f"results/{result_name}.json")
frame_rate = 60
total_frames = len(json)

valid_indices = []
hip = [(None, None, None)] * total_frames
left_toe = [(None, None, None)] * total_frames
right_toe = [(None, None, None)] * total_frames
left_heel = [(None, None, None)] * total_frames
right_heel = [(None, None, None)] * total_frames
confidence_threshold = 0.5

# Extract required keypoints
for i, frame_data in enumerate(json):

    # Check if any keypoint is under the confidence threshold
    if any([conf < confidence_threshold for conf in frame_data["keypoints"][2::3]]):
        continue

    valid_indices.append(i)
    keypoints = frame_data["keypoints"]
    hip[i] = extract_keypoints(keypoints, HALPE_SKELETON.keypoints["HIP"])
    left_toe[i] = extract_keypoints(keypoints, HALPE_SKELETON.keypoints["LEFT_FOOT_INDEX"])
    right_toe[i] = extract_keypoints(keypoints, HALPE_SKELETON.keypoints["RIGHT_FOOT_INDEX"])
    left_heel[i] = extract_keypoints(keypoints, HALPE_SKELETON.keypoints["LEFT_HEEL"])
    right_heel[i] = extract_keypoints(keypoints, HALPE_SKELETON.keypoints["RIGHT_HEEL"])

# Convert to numpy arrays for easier manipulation
hip = np.array([coord[0] for coord in hip])
left_toe = np.array([coord[0] for coord in left_toe])
right_toe = np.array([coord[0] for coord in right_toe])
left_heel = np.array([coord[0] for coord in left_heel])
right_heel = np.array([coord[0] for coord in right_heel])

# Trim
trimmed_range = range(valid_indices[0], valid_indices[-1] + 1)
trimmed_range_len = trimmed_range.stop - trimmed_range.start

hip = trim(hip, valid_indices[0], valid_indices[-1])
left_toe = trim(left_toe, valid_indices[0], valid_indices[-1])
right_toe = trim(right_toe, valid_indices[0], valid_indices[-1])
left_heel = trim(left_heel, valid_indices[0], valid_indices[-1])
right_heel = trim(right_heel, valid_indices[0], valid_indices[-1])

excluded_ranges = get_valid_range(hip, frame_rate)

# Detect gait events
left_events, right_events = gait_detect_zeni(
    hip, left_toe, right_toe, left_heel, right_heel,
    frame_rate=frame_rate,)

# Calculate gait phases
l_phases, r_phases, support_phases = build_phases_from_events(
    left_events, right_events,
    excluded_ranges, total_frames, valid_indices[0])

# Visualize
print_statistics(l_phases, r_phases, support_phases)
visualize_gait_phases(l_phases, r_phases, support_phases)
