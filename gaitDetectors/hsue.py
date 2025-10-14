from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from gaitStructs import GaitEvent, GaitEventType
from utils.preprocessing import cubic_interpolate_nan, butterworth_filter, find_minima_maxima
from scipy.signal import savgol_filter


def gait_detect_hsue(
    hip: np.ndarray,
    left_toe: np.ndarray,
    right_toe: np.ndarray,
    left_heel: np.ndarray,
    right_heel: np.ndarray,
    frame_rate: int = 60,
    debug: bool = False,
) -> Tuple[List[GaitEvent], List[GaitEvent]]:
    """
    Detect gait events (heel-strike and toe-off) using the Hsue et al. method.

    Parameters
    ----------
    hip : np.ndarray
        Nx3 array of hip marker positions.
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
    debug : bool, optional
        If True, plots the antero-posterior acceleration and detected peaks.

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
    hip = butterworth_filter(hip, cutoff=5, order=4, fs=frame_rate)

    left_toe_f = butterworth_filter(left_toe, cutoff=2, order=2, fs=frame_rate)
    right_toe_f = butterworth_filter(right_toe, cutoff=2, order=2, fs=frame_rate)
    left_heel_f = butterworth_filter(left_heel, cutoff=2, order=2, fs=frame_rate)
    right_heel_f = butterworth_filter(right_heel, cutoff=2, order=2, fs=frame_rate)

    # Hip velocity for direction
    grad = np.gradient(hip)
    vel = [val > 0 for val in grad]

    # Compute antero-posterior acceleration
    left_toe_accel = np.gradient(np.gradient(left_toe_f, 1.0 / frame_rate), 1.0 / frame_rate)
    right_toe_accel = np.gradient(np.gradient(right_toe_f, 1.0 / frame_rate), 1.0 / frame_rate)
    left_heel_accel = np.gradient(np.gradient(left_heel_f, 1.0 / frame_rate), 1.0 / frame_rate)
    right_heel_accel = np.gradient(np.gradient(right_heel_f, 1.0 / frame_rate), 1.0 / frame_rate)

    # Detect local maxima and minima using dynamic prominence to get rid of left over noise
    lt_min, lt_max = find_minima_maxima(left_toe_accel, distance=20, prominence=None)
    rt_min, rt_max = find_minima_maxima(right_toe_accel, distance=20, prominence=None)
    lh_min, lh_max = find_minima_maxima(left_heel_accel, distance=20, prominence=None)
    rh_min, rh_max = find_minima_maxima(right_heel_accel, distance=20, prominence=None)

    left_events = []
    right_events = []

    # Heel strike - local minimum of the heel marker when hip velocity is positive
    # Toe off - local maximum of the toe marker when hip velocity is positive
    # Left leg
    for i in [idx for idx, val in enumerate(lt_max) if val is not None and vel[idx]]:
        left_events.append(GaitEvent(i, GaitEventType.TOE_OFF))

    for i in [idx for idx, val in enumerate(lh_min) if val is not None and vel[idx]]:
        left_events.append(GaitEvent(i, GaitEventType.HEEL_STRIKE))

    for i in [idx for idx, val in enumerate(lt_min) if val is not None and not vel[idx]]:
        left_events.append(GaitEvent(i, GaitEventType.TOE_OFF))

    for i in [idx for idx, val in enumerate(lh_max) if val is not None and not vel[idx]]:
        left_events.append(GaitEvent(i, GaitEventType.HEEL_STRIKE))

    # Right leg
    for i in [idx for idx, val in enumerate(rt_max) if val is not None and vel[idx]]:
        right_events.append(GaitEvent(i, GaitEventType.TOE_OFF))

    for i in [idx for idx, val in enumerate(rh_min) if val is not None and vel[idx]]:
        right_events.append(GaitEvent(i, GaitEventType.HEEL_STRIKE))

    for i in [idx for idx, val in enumerate(rt_min) if val is not None and not vel[idx]]:
        right_events.append(GaitEvent(i, GaitEventType.TOE_OFF))

    for i in [idx for idx, val in enumerate(rh_max) if val is not None and not vel[idx]]:
        right_events.append(GaitEvent(i, GaitEventType.HEEL_STRIKE))

    if debug:
        plt.subplot(2,1,1)
        plt.plot(left_toe_accel, label="Left Toe Accel", color="red")
        plt.plot(left_heel_accel, label="Left Heel Accel", color="blue")
        plt.plot(lt_min, "x", color='red')
        plt.plot(lt_max, "x", color='green')
        plt.plot(lh_min, "x", color='red')
        plt.plot(lh_max, "x", color='green')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(right_toe_accel, label="Right Toe Accel", color="red")
        plt.plot(right_heel_accel, label="Right Heel Accel", color="blue")
        plt.plot(rt_min, "x", color='red')
        plt.plot(rt_max, "x", color='green')
        plt.plot(rh_min, "x", color='red')
        plt.plot(rh_max, "x", color='green')
        plt.legend()

        plt.show()

    return left_events, right_events
