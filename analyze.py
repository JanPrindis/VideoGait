from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, find_peaks

from utils.jsonSerializer import KeypointSerializer
from Skeletons.halpe_skeleton import HALPE_SKELETON

class GaitPhase(Enum):
    UNKNOWN = 0
    STANCE = 1
    SWING = 2
    SINGLE_SUPPORT = 11
    DOUBLE_SUPPORT = 12

def extract_keypoints(keypoint_list, index):
    return keypoint_list[index * 3], keypoint_list[index * 3 + 1], keypoint_list[index * 3 + 2]

def cubic_interpolate_nan(y):
    y = np.array([np.nan if v is None else v for v in y], dtype=float)
    mask = ~np.isnan(y)
    if np.sum(mask) < 2:
        return y # Not enough points to interpolate

    first_valid = np.argmax(mask)
    last_valid = len(y) - np.argmax(mask[::-1]) - 1

    x = np.arange(len(y))
    x_seg = x[first_valid:last_valid + 1]
    y_seg = y[first_valid:last_valid + 1]

    mask_seg = ~np.isnan(y_seg)
    cs = CubicSpline(x_seg[mask_seg], y_seg[mask_seg])
    y_interp = cs(x_seg)

    return np.array(y_interp)

def butterworth_filter(data, cutoff=5, fs=60.0, order=5):
    normal_cutoff = cutoff / (fs / 2)
    b, a = butter(order, normal_cutoff, btype='low')
    y = filtfilt(b, a, data)
    return np.array(y)

def find_minima_maxima(data, distance=20, prominence=5):
    minima, _ = find_peaks(-data, distance=distance, prominence=prominence)
    maxima, _ = find_peaks(data, distance=distance, prominence=prominence)
    minima_out = [None] * len(data)
    maxima_out = [None] * len(data)
    for p in minima:
        minima_out[p] = data[p]
    for p in maxima:
        maxima_out[p] = data[p]
    return np.array(minima_out), np.array(maxima_out)

def trim(data, first_valid, last_valid):
    return data[first_valid:last_valid + 1]

result_name = "test"
json = KeypointSerializer.load(f"results/{result_name}.json")

hip = []
left_toe = []
right_toe = []
left_heel = []
right_heel = []
valid_indices = []

total_frames = 0
confidence_threshold = 0.5

# Extract required keypoints
for i, frame_data in enumerate(json):
    total_frames += 1

    # Check if any keypoint is under the confidence threshold
    if any([conf < confidence_threshold for conf in frame_data["keypoints"][2::3]]):
        # TODO: Could be better if all were initialized as None to begin with
        hip.append((None, None, None))
        left_toe.append((None, None, None))
        right_toe.append((None, None, None))
        left_heel.append((None, None, None))
        right_heel.append((None, None, None))
        continue

    valid_indices.append(i)
    keypoints = frame_data["keypoints"]
    hip.append(extract_keypoints(keypoints, HALPE_SKELETON.keypoints["HIP"]))
    left_toe.append(extract_keypoints(keypoints, HALPE_SKELETON.keypoints["LEFT_FOOT_INDEX"]))
    right_toe.append(extract_keypoints(keypoints, HALPE_SKELETON.keypoints["RIGHT_FOOT_INDEX"]))
    left_heel.append(extract_keypoints(keypoints, HALPE_SKELETON.keypoints["LEFT_HEEL"]))
    right_heel.append(extract_keypoints(keypoints, HALPE_SKELETON.keypoints["RIGHT_HEEL"]))

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
# TODO: Mozna gradient je shit
turning_point_threshold = 0.33
turning_point_indices = [i for i, val in enumerate(grad_norm) if val < turning_point_threshold]

# Cutoff values under threshold
for i in turning_point_indices:
    lt_hip_diff[i] = None
    rt_hip_diff[i] = None
    lh_hip_diff[i] = None
    rh_hip_diff[i] = None

# Find local maxima and minima
distance = 20
prominence = 10

lt_min, lt_max = find_minima_maxima(lt_hip_diff, distance=distance, prominence=prominence)
rt_min, rt_max = find_minima_maxima(rt_hip_diff, distance=distance, prominence=prominence)
lh_min, lh_max = find_minima_maxima(lh_hip_diff, distance=distance, prominence=prominence)
rh_min, rh_max = find_minima_maxima(rh_hip_diff, distance=distance, prominence=prominence)

# Try to detect gait
l_leg_gait_phase = np.zeros(trimmed_range_len, dtype=int)
r_leg_gait_phase = np.zeros(trimmed_range_len, dtype=int)

l_phase = GaitPhase.UNKNOWN
r_phase = GaitPhase.UNKNOWN
l_last = l_phase
r_last = r_phase

for i in range(trimmed_range_len):
    if i in turning_point_indices:
        l_last = GaitPhase.UNKNOWN
        r_last = GaitPhase.UNKNOWN
        l_leg_gait_phase[i] = l_last.value
        r_leg_gait_phase[i] = r_last.value
        continue

    # Going right
    if vel[i]:
        if lt_max[i] is not None:
            l_phase = GaitPhase.STANCE
            l_last = GaitPhase.STANCE
        if rt_max[i] is not None:
            r_phase = GaitPhase.STANCE
            r_last = GaitPhase.STANCE
        if lh_min[i] is not None:
            l_phase = GaitPhase.SWING
            l_last = GaitPhase.SWING
        if rh_min[i] is not None:
            r_phase = GaitPhase.SWING
            r_last = GaitPhase.SWING
    # Going left
    else:
        if lt_min[i] is not None:
            l_phase = GaitPhase.STANCE
            l_last = GaitPhase.STANCE
        if rt_min[i] is not None:
            r_phase = GaitPhase.STANCE
            r_last = GaitPhase.STANCE
        if lh_max[i] is not None:
            l_phase = GaitPhase.SWING
            l_last = GaitPhase.SWING
        if rh_max[i] is not None:
            r_phase = GaitPhase.SWING
            r_last = GaitPhase.SWING

    l_leg_gait_phase[i] = l_phase.value
    r_leg_gait_phase[i] = r_phase.value
    l_last, r_last = l_phase, r_phase

# Trim gait phases
for i in range(trimmed_range_len):
    if l_leg_gait_phase[i] == GaitPhase.UNKNOWN.value or r_leg_gait_phase[i] == GaitPhase.UNKNOWN.value:
        l_leg_gait_phase[i] = GaitPhase.UNKNOWN.value
        r_leg_gait_phase[i] = GaitPhase.UNKNOWN.value

# Double stance calculation
double_stance = []
for i in range(trimmed_range_len):
    if l_leg_gait_phase[i] == GaitPhase.UNKNOWN.value or r_leg_gait_phase[i] == GaitPhase.UNKNOWN.value:
        double_stance.append(GaitPhase.UNKNOWN.value)
        continue
    if l_leg_gait_phase[i] == GaitPhase.STANCE.value and r_leg_gait_phase[i] == GaitPhase.STANCE.value:
        double_stance.append(GaitPhase.DOUBLE_SUPPORT.value)
    else:
        double_stance.append(GaitPhase.SINGLE_SUPPORT.value)

# Calculate gait phase durations
l_total_valid = len([x for x in l_leg_gait_phase if x != GaitPhase.UNKNOWN.value])
r_total_valid = len([x for x in r_leg_gait_phase if x != GaitPhase.UNKNOWN.value])

stance_valid = len([x for x in double_stance if x != GaitPhase.UNKNOWN.value])

gait_phase_durations = {
    "left_leg": {
        "stance_count": len([x for x in l_leg_gait_phase if x == GaitPhase.STANCE.value]),
        "swing_count": len([x for x in l_leg_gait_phase if x == GaitPhase.SWING.value]),
        "stance_percentage": len([x for x in l_leg_gait_phase if x == GaitPhase.STANCE.value])
                             / l_total_valid * 100 if l_total_valid > 0 else 0,
        "swing_percentage": len([x for x in l_leg_gait_phase if x == GaitPhase.SWING.value])
                            / l_total_valid * 100 if l_total_valid > 0 else 0,
    },
    "right_leg": {
        "stance_count": len([x for x in r_leg_gait_phase if x == GaitPhase.STANCE.value]),
        "swing_count": len([x for x in r_leg_gait_phase if x == GaitPhase.SWING.value]),
        "stance_percentage": len([x for x in r_leg_gait_phase if x == GaitPhase.STANCE.value])
                             / r_total_valid * 100 if r_total_valid > 0 else 0,
        "swing_percentage": len([x for x in r_leg_gait_phase if x == GaitPhase.SWING.value])
                            / r_total_valid * 100 if r_total_valid > 0 else 0,
    },
    "stance": {
        "single_support": len([x for x in double_stance if x == GaitPhase.SINGLE_SUPPORT.value]),
        "double_support": len([x for x in double_stance if x == GaitPhase.DOUBLE_SUPPORT.value]),
        "single_support_percentage": len([x for x in double_stance if x == GaitPhase.SINGLE_SUPPORT.value])
                                      / stance_valid * 100 if stance_valid > 0 else 0,
        "double_support_percentage": len([x for x in double_stance if x == GaitPhase.DOUBLE_SUPPORT.value])
                                      / stance_valid * 100 if stance_valid > 0 else 0,
    }
}

print(gait_phase_durations)

# Visualization
x_start = trimmed_range.start
x_end = trimmed_range.stop

colors = {
    GaitPhase.UNKNOWN.value: 'lightgray',
    GaitPhase.STANCE.value: 'red',
    GaitPhase.SWING.value: 'green'
}
labels = {
    GaitPhase.UNKNOWN.value: 'unknown',
    GaitPhase.STANCE.value: 'stance',
    GaitPhase.SWING.value: 'swing'
}

support_colors = {
    GaitPhase.UNKNOWN.value: 'lightgray',
    GaitPhase.SINGLE_SUPPORT.value: 'orange',
    GaitPhase.DOUBLE_SUPPORT.value: 'blue'
}
support_labels = {
    GaitPhase.UNKNOWN.value: 'unknown',
    GaitPhase.SINGLE_SUPPORT.value: 'single support',
    GaitPhase.DOUBLE_SUPPORT.value: 'double support'
}

fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

# Left foot gait phases
left_phases = np.array(l_leg_gait_phase)
start = 0
for i in range(1, len(left_phases)):
    if left_phases[i] != left_phases[start]:
        axs[0].axvspan(x_start + start, x_start + i,
                       color=colors[left_phases[start]], alpha=0.5)
        start = i
axs[0].axvspan(x_start + start, x_end, color=colors[left_phases[start]], alpha=0.5)
axs[0].set_ylabel("Left Foot")
axs[0].set_yticks([])
axs[0].set_title("Left Foot Gait Phases")

patches_left = [Patch(color=c, label=l) for v, (c, l) in zip(colors.keys(), zip(colors.values(), labels.values()))]
axs[0].legend(handles=patches_left, loc='upper right')


# Right foot gait phases
right_phases = np.array(r_leg_gait_phase)
start = 0
for i in range(1, len(right_phases)):
    if right_phases[i] != right_phases[start]:
        axs[1].axvspan(x_start + start, x_start + i,
                       color=colors[right_phases[start]], alpha=0.5)
        start = i
axs[1].axvspan(x_start + start, x_end, color=colors[right_phases[start]], alpha=0.5)
axs[1].set_ylabel("Right Foot")
axs[1].set_yticks([])
axs[1].set_title("Right Foot Gait Phases")

patches_right = [Patch(color=c, label=l) for v, (c, l) in zip(colors.keys(), zip(colors.values(), labels.values()))]
axs[1].legend(handles=patches_right, loc='upper right')


# Support phases
support_phases = np.array(double_stance)
start = 0
for i in range(1, len(support_phases)):
    if support_phases[i] != support_phases[start]:
        axs[2].axvspan(x_start + start, x_start + i,
                       color=support_colors[support_phases[start]], alpha=0.5)
        start = i
axs[2].axvspan(x_start + start, x_end, color=support_colors[support_phases[start]], alpha=0.5)

axs[2].set_yticks([])
axs[2].set_xlabel('Frame index')
axs[2].set_title('Support Phases')

patches_support = [Patch(color=c, label=l) for v, (c, l) in zip(support_colors.keys(), zip(support_colors.values(), support_labels.values()))]
axs[2].legend(handles=patches_support, loc='upper right')
axs[2].set_xlim(x_start, x_end)

plt.tight_layout()
plt.show()