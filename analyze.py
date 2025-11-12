from enum import Enum

import numpy as np

from gaitDetectors.ghoussayni import gait_detect_ghoussayni
from gaitDetectors.hsue import gait_detect_hsue
from gaitDetectors.zeni import gait_detect_zeni
from gaitStructs import build_phases_from_events
from utils.jsonSerializer import KeypointSerializer
from Skeletons.halpe_skeleton import HALPE_SKELETON
from utils.data import get_valid_range, trim, get_keypoints
from visualizeGaitPhases import visualize_gait_phases, print_statistics

# Result metadata
result_name = "test"
json_path = f"results/{result_name}.json"

frame_rate = 60
# total_frames = len(json)

# Extract required keypoints
required_keypoints = ["HIP", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_HEEL", "RIGHT_HEEL"]
extracted_keypoints, valid_indices = get_keypoints(json_path, HALPE_SKELETON, required_keypoints, confidence_threshold=0.5)

# Convert to numpy arrays for easier manipulation
hip = np.array([coord[0] for coord in extracted_keypoints["HIP"]])
left_toe = np.array([coord[0] for coord in extracted_keypoints["LEFT_FOOT_INDEX"]])
right_toe = np.array([coord[0] for coord in extracted_keypoints["RIGHT_FOOT_INDEX"]])
left_heel = np.array([coord[0] for coord in extracted_keypoints["LEFT_HEEL"]])
right_heel = np.array([coord[0] for coord in extracted_keypoints["RIGHT_HEEL"]])

total_frames = len(hip)

# Trim
trimmed_range = range(valid_indices[0], valid_indices[-1] + 1)
trimmed_range_len = trimmed_range.stop - trimmed_range.start

print(f"first_valid {valid_indices[0]}")

hip = trim(hip, valid_indices[0], valid_indices[-1])
left_toe = trim(left_toe, valid_indices[0], valid_indices[-1])
right_toe = trim(right_toe, valid_indices[0], valid_indices[-1])
left_heel = trim(left_heel, valid_indices[0], valid_indices[-1])
right_heel = trim(right_heel, valid_indices[0], valid_indices[-1])

valid_ranges = get_valid_range(hip, frame_rate, exclude_percent=0.05)

# Detect gait events

# Using Zeni et al. method
# left_events, right_events = gait_detect_zeni(
#     hip, left_toe, right_toe, left_heel, right_heel,
#     frame_rate=frame_rate,)

# Using Ghoussayni et al. method
# left_events, right_events = gait_detect_ghoussayni(
#     left_toe, right_toe, left_heel, right_heel,
#     frame_rate=frame_rate,
#     debug=True)

# Using Hsue et al. method
left_events, right_events = gait_detect_hsue(
    hip, left_toe, right_toe, left_heel, right_heel,
    frame_rate=frame_rate,
    debug=True)

# TODO: Bonci et al. method

# Calculate gait phases
l_phases, r_phases, support_phases = build_phases_from_events(
    left_events, right_events,
    valid_ranges, total_frames, valid_indices[0])

# Visualize
print_statistics(l_phases, r_phases, support_phases)
visualize_gait_phases(l_phases, r_phases, support_phases)
