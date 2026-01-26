import os

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


def get_valid_range(
        hip: np.array,
        frame_rate: float,
        exclude_percent: float = 0.1,
        min_segment_length: int = 60,
        outlier_ratio: float = 0.2,
        filter_cutoff: int = 5,
        filter_order: int = 4
):
    # Safety check - length too short
    if len(hip) < max(20, min_segment_length):
        return []

    # Preprocessing
    hip = cubic_interpolate_nan(hip)

    try:
        hip_f = butterworth_filter(hip, cutoff=filter_cutoff, order=filter_order, fs=frame_rate)
    except ValueError:
        return []

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

    # Filter out short segments
    valid_ranges = [
        (s, e) for s, e in valid_ranges
        if (e - s + 1) >= min_segment_length
    ]

    # Filter out outliers
    if len(valid_ranges) > 1:
        lengths = [(e - s + 1) for s, e in valid_ranges]
        max_len = max(lengths)
        threshold = max_len * outlier_ratio

        valid_ranges = [
            r for r, l in zip(valid_ranges, lengths)
            if l >= threshold
        ]

    return valid_ranges


def cubic_interpolate_nan(data):
    y = np.array(data, dtype=float)
    nans = np.isnan(y)

    # If there aren't any NaN values, return original
    if not nans.any():
        return y

    non_nans_indices = np.where(~nans)[0]
    n_valid = len(non_nans_indices)

    # Edge case - no valid data
    if n_valid == 0:
        return np.zeros_like(y)

    # Edge case - only one valid data point
    if n_valid == 1:
        # Cannot interpolate, fill with the same value
        val = y[non_nans_indices[0]]
        y[:] = val
        return y

    x = np.arange(len(y))

    try:
        # Interpolate on known data
        cs = CubicSpline(x[~nans], y[~nans], extrapolate=False)

        y_interp = y.copy()
        mask_interp = nans
        y_interp[mask_interp] = cs(x[mask_interp])

        # NaNs on edges are filled with the closest valid value
        if np.isnan(y_interp).any():
            # Forward Fill
            mask = np.isnan(y_interp)
            idx = np.where(~mask, np.arange(mask.shape[0]), 0)
            np.maximum.accumulate(idx, axis=0, out=idx)
            y_interp = y_interp[idx]

            # Backward Fill
            mask = np.isnan(y_interp)
            idx = np.where(~mask, np.arange(mask.shape[0]), mask.shape[0] - 1)
            idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
            y_interp = y_interp[idx]

        return y_interp

    except Exception as e:
        print(f"[Warning] Interpolation failed: {e}")
        return y


def butterworth_filter(data, cutoff=5, fs=60.0, order=5):

    # Check if data exists
    if data is None or len(data) == 0:
        return np.array([]) if data is None else data

    y = np.array(data)

    # Check data length
    # filtfilt requires 'padlen', which is 3 * (max(len(a), len(b)) - 1).
    if len(y) <= 3 * order:
        # Not enough data
        print("[Warning] Butterworth filter: Input data sequence to short, returning original!")
        return y

    # Nyquist frequency check
    nyquist = fs / 2
    if cutoff >= nyquist:
        # Frequency is higher than what we are able to filter - Fallback to .99 * nyquist
        cutoff = 0.99 * nyquist
        print(f"[Warning] Butterworth filter: Cutoff frequency is too high, using 0.99 * nyquist = {cutoff}!")

    if cutoff <= 0:
        return y

    # Filtration
    try:
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low')
        y_filtered = filtfilt(b, a, y)
        return np.array(y_filtered)
    except ValueError as e:
        print(f"[Error] Butterworth filter: {e}")
        return y


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

    trimmed_keypoints = {}
    for item in items_to_get:
        trimmed_keypoints[item] = trim(extracted_keypoints[item], valid_indices[0], valid_indices[-1])

    return trimmed_keypoints, valid_indices


def average_with_nones(list1, list2):
    """
    Averages two lists element-wise, handling None values.

    This function can handle lists of numbers or lists of tuples (keypoints).
    - If both elements are valid, their average is taken.
    - If one element is valid and the other is None, the valid element is taken.
    - If both are None, None is returned.
    """
    if len(list1) != len(list2):
        raise ValueError("Input lists must have the same length!")

    result = []
    for item1, item2 in zip(list1, list2):
        # Check if the item is a keypoint tuple (x, y, conf) or a simple number
        is_item1_valid = item1 is not None and (not isinstance(item1, tuple) or item1[0] is not None)
        is_item2_valid = item2 is not None and (not isinstance(item2, tuple) or item2[0] is not None)

        if is_item1_valid and is_item2_valid:
            if isinstance(item1, tuple):
                result.append(tuple((v1 + v2) / 2 for v1, v2 in zip(item1, item2)))
            else:
                result.append((item1 + item2) / 2)
        elif is_item1_valid:
            result.append(item1)
        elif is_item2_valid:
            result.append(item2)
        else:
            result.append(None)
    return result


def calculate_torso_height(hip_coords, neck_coords):
    """
    Calculates the Euclidean distance between hip and neck for each frame to serve as a stable torso height.
    """
    heights = []
    for hip, neck in zip(hip_coords, neck_coords):
        # Ensure both hip and neck coordinates and their components are valid for the current frame
        if hip and hip[0] is not None and hip[1] is not None and \
                neck and neck[0] is not None and neck[1] is not None:

            # Calculate the Euclidean distance
            dist = np.sqrt((hip[0] - neck[0]) ** 2 + (hip[1] - neck[1]) ** 2)
            heights.append(dist)
        else:
            # If either keypoint is missing, the height for this frame is unknown
            heights.append(None)
    return heights


def find_matching_annotation(keypoint_json_path: str, annotations_root: str) -> str | None:
    """
    Finds the corresponding annotation file path for a given keypoint file path
    based on a specific directory structure.

    Args:
        keypoint_json_path (str): The full path to the keypoint JSON file.
            Expected format: .../PROCESSED/{FPS}/KEYPOINTS/{file_name}.json
        annotations_root (str): The root directory where annotations are stored.

    Returns:
        str | None: The full path to the corresponding annotation file,
                    or None if the keypoint path format is incorrect.
    """
    try:
        # Normalize path separators
        norm_path = os.path.normpath(keypoint_json_path)
        parts = norm_path.split(os.sep)

        # Extract file_name
        file_name = parts[-1]
        # FPS, which is the third to last part
        fps = parts[-3]

        # Sanity check if path is as expected
        if parts[-2].upper() != 'KEYPOINTS' or parts[-4].upper() != 'PROCESSED':
             print(f"[Warning] Keypoint path '{keypoint_json_path}' does not seem to match the expected structure.")
             return None

        # Construct the new path using the extracted parts
        annotation_path = os.path.join(annotations_root, fps, file_name)
        return annotation_path

    except IndexError:
        print(f"[Warning] Could not parse keypoint path: '{keypoint_json_path}'.")
        return None


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Calculates the angle at point p2 in degrees for N frames."""
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.degrees(np.arctan2(v2[:, 1], v2[:, 0]) - np.arctan2(v1[:, 1], v1[:, 0]))
    angle = np.abs(angle)
    angle[angle > 180] = 360.0 - angle[angle > 180]
    return angle


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Calculates the absolute horizontal distance (x-offset) between p1 and p2 for N frames."""
    return np.abs(p1[:, 0] - p2[:, 0])
