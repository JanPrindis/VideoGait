from enum import Enum

import numpy as np

from gaitStructs import PhaseType, GaitPhase, Leg, compute_support_phases
from utils.jsonSerializer import KeypointSerializer
from Skeletons.halpe_skeleton import HALPE_SKELETON
from utils.data import extract_keypoints
from utils.preprocessing import cubic_interpolate_nan, butterworth_filter, find_minima_maxima
from visualizeGaitPhases import visualize_gait_phases, print_statistics


def trim(data, first_valid, last_valid):
    return data[first_valid:last_valid + 1]

result_name = "test"
json = KeypointSerializer.load(f"results/{result_name}.json")

total_frames = len(json)

hip = [(None, None, None)] * total_frames
left_toe = [(None, None, None)] * total_frames
right_toe = [(None, None, None)] * total_frames
left_heel = [(None, None, None)] * total_frames
right_heel = [(None, None, None)] * total_frames
valid_indices = []

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

# Interpolate missing values
hip = cubic_interpolate_nan(hip)
left_toe = cubic_interpolate_nan(left_toe)
right_toe = cubic_interpolate_nan(right_toe)
left_heel = cubic_interpolate_nan(left_heel)
right_heel = cubic_interpolate_nan(right_heel)

# Butterworth filter
frame_rate = 60.0
hip_f = butterworth_filter(hip, cutoff=5, order=4, fs=frame_rate)
left_toe_f = butterworth_filter(left_toe, cutoff=5, order=4, fs=frame_rate)
right_toe_f = butterworth_filter(right_toe, cutoff=5, order=4, fs=frame_rate)
left_heel_f = butterworth_filter(left_heel, cutoff=5, order=4, fs=frame_rate)
right_heel_f = butterworth_filter(right_heel, cutoff=5, order=4, fs=frame_rate)

# Fit hip pos to range 0, 1
hip_f_norm = (hip_f - np.min(hip_f)) / (np.max(hip_f) - np.min(hip_f))

# Calculations
grad = np.gradient(hip_f)
grad_abs = np.abs(grad)

grad_norm = (grad_abs - np.min(grad_abs)) / (np.max(grad_abs) - np.min(grad_abs))

vel = [val > 0  for val in grad]
lt_hip_diff = left_toe_f - hip_f
rt_hip_diff = right_toe_f - hip_f
lh_hip_diff = left_heel_f - hip_f
rh_hip_diff = right_heel_f - hip_f

# Detect turning point
# TODO: Mozna vyuzit min-max xoveho rozsahu panve a orezat 10% z obou stran pro detekci turning pointu
# TODO: Mozna gradient je shit
turning_point_threshold = 0.33
turning_point_indices = [i for i, val in enumerate(grad_norm) if val < turning_point_threshold]

# Get X movement range of hip
hip_min, hip_max = min(hip_f), max(hip_f)
exclude_percent = 0.1  # 10 %

def contiguous_ranges(indices):
    if not indices:
        return []
    indices = np.array(indices)
    groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    return [(int(g[0]), int(g[-1])) for g in groups]

# Get margin indices
exclude_range = (hip_max - hip_min) * exclude_percent
exclude_min = hip_min + exclude_range
exclude_max = hip_max - exclude_range

# Excluded ranges - start, turning, end
excluded_indices = [i for i, x in enumerate(hip_f) if x <= exclude_min or x >= exclude_max]
excluded_ranges = contiguous_ranges(excluded_indices)


# Find local maxima and minima
distance = 20
prominence = 10

lt_min, lt_max = find_minima_maxima(lt_hip_diff, distance=distance, prominence=prominence)
rt_min, rt_max = find_minima_maxima(rt_hip_diff, distance=distance, prominence=prominence)
lh_min, lh_max = find_minima_maxima(lh_hip_diff, distance=distance, prominence=prominence)
rh_min, rh_max = find_minima_maxima(rh_hip_diff, distance=distance, prominence=prominence)


def in_excluded(i):
    return any(start <= i <= end for start, end in excluded_ranges)

l_current = PhaseType.UNKNOWN
r_current = PhaseType.UNKNOWN
l_last = l_current
r_last = r_current

l_phases = []
r_phases = []
l_start = 0
r_start = 0

for i in range(trimmed_range_len):

    if in_excluded(i):
        # If phase active, stop
        if l_last != PhaseType.UNKNOWN:
            l_phases.append(GaitPhase(Leg.LEFT, l_last, l_start + valid_indices[0], i - 1 + valid_indices[0], i - l_start, valid=False))
            l_last = PhaseType.UNKNOWN
            l_current = PhaseType.UNKNOWN
        if r_last != PhaseType.UNKNOWN:
            r_phases.append(GaitPhase(Leg.RIGHT, r_last, r_start + valid_indices[0], i - 1 + valid_indices[0], i - r_start, valid=False))
            r_last = PhaseType.UNKNOWN
            r_current = PhaseType.UNKNOWN
        continue

    # Left leg
    if vel[i]:
        if lt_max[i] is not None:
            l_current = PhaseType.STANCE
        if lh_min[i] is not None:
            l_current = PhaseType.SWING
    else:
        if lt_min[i] is not None:
            l_current = PhaseType.STANCE
        if lh_max[i] is not None:
            l_current = PhaseType.SWING

    # Right leg
    if vel[i]:
        if rt_max[i] is not None:
            r_current = PhaseType.STANCE
        if rh_min[i] is not None:
            r_current = PhaseType.SWING
    else:
        if rt_min[i] is not None:
            r_current = PhaseType.STANCE
        if rh_max[i] is not None:
            r_current = PhaseType.SWING

    # If phase changed, end last active phase
    if l_current != l_last:
        if l_last != PhaseType.UNKNOWN:
            # Mark as valid if it did not start/end in excluded zones
            valid = not any(not (i - 1 < start or l_start > end) for start, end in excluded_ranges)
            l_phases.append(GaitPhase(Leg.LEFT, l_last, l_start + valid_indices[0], i - 1 + valid_indices[0], i - l_start, valid))
        l_start = i

    if r_current != r_last:
        if r_last != PhaseType.UNKNOWN:
            valid = not any(not (i - 1 < start or r_start > end) for start, end in excluded_ranges)
            r_phases.append(GaitPhase(Leg.RIGHT, r_last, r_start + valid_indices[0], i - 1 + valid_indices[0], i - r_start, valid))
        r_start = i

    l_last, r_last = l_current, r_current

# Close last segment
if l_last != PhaseType.UNKNOWN:
    valid = not any(not (trimmed_range_len - 1 < s or l_start > e) for s, e in excluded_ranges)
    l_phases.append(GaitPhase(Leg.LEFT, l_last, l_start + valid_indices[0], trimmed_range_len - 1 + valid_indices[0], trimmed_range_len - l_start, valid))
if r_last != PhaseType.UNKNOWN:
    valid = not any(not (trimmed_range_len - 1 < s or r_start > e) for s, e in excluded_ranges)
    r_phases.append(GaitPhase(Leg.RIGHT, r_last, r_start + valid_indices[0], trimmed_range_len - 1 + valid_indices[0], trimmed_range_len - r_start, valid))

# Filter out valid segments
l_phases = [seg for seg in l_phases if seg.valid]
r_phases = [seg for seg in r_phases if seg.valid]

# Support phases
support_phases = compute_support_phases(l_phases, r_phases, total_frames)

# Visualize
print_statistics(l_phases, r_phases, support_phases)
visualize_gait_phases(l_phases, r_phases, support_phases)
