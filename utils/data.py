import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks, butter, filtfilt

from utils.jsonSerializer import KeypointSerializer


def create_folder_if_not_exists(folder_path):
    import os
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


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

    # Convert excluded ranges to valid ranges
    valid_ranges = []
    last_end = 0
    for start, end in excluded_ranges:
        if start > last_end:
            valid_ranges.append((last_end, start - 1))
        last_end = max(last_end, end + 1)

    # Add the last valid range if it exists
    if last_end < len(hip_f):
        valid_ranges.append((last_end, len(hip_f) - 1))

    # Filter out invalid ranges
    valid_ranges = [(start, end) for start, end in valid_ranges if start <= end]

    # If the first excluded range starts at 0, remove the initial (0, -1) valid range
    if excluded_ranges and excluded_ranges[0][0] == 0 and valid_ranges and valid_ranges[0] == (0, -1):
        valid_ranges.pop(0)

    return valid_ranges


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


def find_minima_maxima(data, distance=20, prominence=None, rel_prominence=0.3):
    """
    Detect local minima and maxima in a 1D signal.

    Parameters
    ----------
    data : np.ndarray
        1D array representing the input signal.
    distance : int, optional
        Minimum number of samples between consecutive peaks (default: 20).
        Helps to avoid detecting multiple peaks too close to each other.
    prominence : float or None, optional
        Minimum prominence of peaks to be considered valid. If None, it is computed
        automatically as percentage (`rel_prominence`) of the total peak-to-peak amplitude of the signal.
    rel_prominence : float, optional (default: 0.3)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays of the same length as `data`:
        - First array contains detected **minima** values at their positions, and `None` elsewhere.
        - Second array contains detected **maxima** values at their positions, and `None` elsewhere.
    """

    if prominence is None:
        prominence = np.ptp(data) * rel_prominence

    minima, _ = find_peaks(-data, distance=distance, prominence=prominence)
    maxima, _ = find_peaks(data, distance=distance, prominence=prominence)
    minima_out = [None] * len(data)
    maxima_out = [None] * len(data)
    for p in minima:
        minima_out[p] = data[p]
    for p in maxima:
        maxima_out[p] = data[p]
    return np.array(minima_out), np.array(maxima_out)


def get_keypoints(keypoint_json_path, skeleton_definition, items_to_get, confidence_threshold=0.5):

    def get_specific_keypoint(keypoint_list, index):
        return keypoint_list[index * 3], keypoint_list[index * 3 + 1], keypoint_list[index * 3 + 2]

    json = KeypointSerializer.load(keypoint_json_path)
    total_frames = len(json)

    valid_indices = []
    extracted_keypoints = {item: [(None, None, None)] * total_frames for item in items_to_get}

    # Extract specified keypoints
    for i, frame_data in enumerate(json):
        # Check if any keypoint is under the confidence threshold
        if any([conf < confidence_threshold for conf in frame_data["keypoints"][2::3]]):
            continue

        valid_indices.append(i)
        keypoints = frame_data["keypoints"]
        for item in items_to_get:
            try:
                extracted_keypoints[item][i] = get_specific_keypoint(keypoints, skeleton_definition.keypoints[item])
            except KeyError:
                # if skeleton_definition does not contain the requested keypoint, it will be (None, None, None)
                pass


    return extracted_keypoints, valid_indices
