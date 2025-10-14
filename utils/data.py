import numpy as np

from utils.preprocessing import cubic_interpolate_nan, butterworth_filter


def trim(data, first_valid, last_valid):
    return data[first_valid:last_valid + 1]


def extract_keypoints(keypoint_list, index):
    return keypoint_list[index * 3], keypoint_list[index * 3 + 1], keypoint_list[index * 3 + 2]


def get_valid_range(hip: np.array, frame_rate: float, exclude_percent: float = 0.1):
    # Preprocessing
    hip = cubic_interpolate_nan(hip)
    hip_f = butterworth_filter(hip, cutoff=5, order=4, fs=frame_rate)

    # Get X movement range of hip
    hip_min, hip_max = min(hip_f), max(hip_f)

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

    return excluded_ranges