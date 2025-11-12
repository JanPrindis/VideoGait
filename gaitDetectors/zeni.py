from typing import List, Tuple
from gaitStructs import GaitEvent, GaitEventType
from utils.data import cubic_interpolate_nan, butterworth_filter, find_minima_maxima

import numpy as np


def gait_detect_zeni(
    hip: np.ndarray,
    left_toe: np.ndarray,
    right_toe: np.ndarray,
    left_heel: np.ndarray,
    right_heel: np.ndarray,
    frame_rate: int = 60,
    distance: int = 20,
    prominence: int = 10,
) -> Tuple[List[GaitEvent], List[GaitEvent]]:
    """
    Detect gait events (heel-strike and toe-off) using the Zeni et al. method.

    Parameters
    ----------
    hip : np.ndarray
        Nx3 array of hip marker positions over time.
    left_toe : np.ndarray
        Nx3 array of left toe marker positions.
    right_toe : np.ndarray
        Nx3 array of right toe marker positions.
    left_heel : np.ndarray
        Nx3 array of left heel marker positions.
    right_heel : np.ndarray
        Nx3 array of right heel marker positions.
    frame_rate : int, optional
        Sampling frequency of the motion capture data (default: 60 Hz).
    distance : int, optional
        Minimum distance between detected peaks (used in `find_peaks`).
    prominence : int, optional
        Minimum prominence of peaks (used in `find_peaks`).

    Returns
    -------
    Tuple[List[GaitEvent], List[GaitEvent]]
        Two lists of detected gait events:
        - First list: events for the **left leg**
        - Second list: events for the **right leg**
    """

    # Interpolate missing values
    hip = cubic_interpolate_nan(hip)
    left_toe = cubic_interpolate_nan(left_toe)
    right_toe = cubic_interpolate_nan(right_toe)
    left_heel = cubic_interpolate_nan(left_heel)
    right_heel = cubic_interpolate_nan(right_heel)

    # Butterworth filter
    hip_f = butterworth_filter(hip, cutoff=5, order=4, fs=frame_rate)
    left_toe_f = butterworth_filter(left_toe, cutoff=5, order=4, fs=frame_rate)
    right_toe_f = butterworth_filter(right_toe, cutoff=5, order=4, fs=frame_rate)
    left_heel_f = butterworth_filter(left_heel, cutoff=5, order=4, fs=frame_rate)
    right_heel_f = butterworth_filter(right_heel, cutoff=5, order=4, fs=frame_rate)

    # Fit hip pos to range 0, 1
    # hip_f_norm = (hip_f - np.min(hip_f)) / (np.max(hip_f) - np.min(hip_f))

    # Gradient calculation
    grad = np.gradient(hip_f)
    # grad_abs = np.abs(grad)
    # grad_norm = (grad_abs - np.min(grad_abs)) / (np.max(grad_abs) - np.min(grad_abs))

    # Hip-feet differences
    vel = [val > 0 for val in grad]
    lt_hip_diff = left_toe_f - hip_f
    rt_hip_diff = right_toe_f - hip_f
    lh_hip_diff = left_heel_f - hip_f
    rh_hip_diff = right_heel_f - hip_f

    # Find local maxima and minima
    lt_min, lt_max = find_minima_maxima(lt_hip_diff, distance=distance, prominence=prominence)
    rt_min, rt_max = find_minima_maxima(rt_hip_diff, distance=distance, prominence=prominence)
    lh_min, lh_max = find_minima_maxima(lh_hip_diff, distance=distance, prominence=prominence)
    rh_min, rh_max = find_minima_maxima(rh_hip_diff, distance=distance, prominence=prominence)

    left_events = []
    right_events = []

    # We measure both ways - so we use local minima when going the other way
    # Heel-strike occurs at the local maximum of the hip–toe horizontal distance
    # Toe-off occurs at the local maximum of the hip–heel horizontal distance

    # Left leg
    for i in [idx for idx, val in enumerate(lt_max) if val is not None and vel[idx]]:
        left_events.append(GaitEvent(i, GaitEventType.HEEL_STRIKE))

    for i in [idx for idx, val in enumerate(lh_min) if val is not None and vel[idx]]:
        left_events.append(GaitEvent(i, GaitEventType.TOE_OFF))

    for i in [idx for idx, val in enumerate(lt_min) if val is not None and not vel[idx]]:
        left_events.append(GaitEvent(i, GaitEventType.HEEL_STRIKE))

    for i in [idx for idx, val in enumerate(lh_max) if val is not None and not vel[idx]]:
        left_events.append(GaitEvent(i, GaitEventType.TOE_OFF))

    # Right leg
    for i in [idx for idx, val in enumerate(rt_max) if val is not None and vel[idx]]:
        right_events.append(GaitEvent(i, GaitEventType.HEEL_STRIKE))

    for i in [idx for idx, val in enumerate(rh_min) if val is not None and vel[idx]]:
        right_events.append(GaitEvent(i, GaitEventType.TOE_OFF))

    for i in [idx for idx, val in enumerate(rt_min) if val is not None and not vel[idx]]:
        right_events.append(GaitEvent(i, GaitEventType.HEEL_STRIKE))

    for i in [idx for idx, val in enumerate(rh_max) if val is not None and not vel[idx]]:
        right_events.append(GaitEvent(i, GaitEventType.TOE_OFF))

    return left_events, right_events
