from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from gaitStructs import GaitEvent, GaitEventType
from utils.preprocessing import cubic_interpolate_nan, butterworth_filter


def gait_detect_ghoussayni(
    left_toe: np.ndarray,
    right_toe: np.ndarray,
    left_heel: np.ndarray,
    right_heel: np.ndarray,
    frame_rate: int = 60,
    threshold: int = None,
    threshold_percentage: float = 0.15,
    debug: bool = False,
) -> Tuple[List[GaitEvent], List[GaitEvent]]:
    """
    Detect gait events (heel-strike and toe-off) using the Ghoussayni et al. method.

    Parameters
    ----------
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
    threshold : int, optional
        Velocity threshold for event detection. If None, a dynamic threshold is used.
    threshold_percentage : float, optional
        Percentage of the maximum toe vertical velocity to set as threshold if `threshold` is None (default: 0.15).
    debug : bool, optional
        If True, plots the velocity profiles, threshold and detected events for debugging purposes.

    Returns
    -------
    Tuple[List[GaitEvent], List[GaitEvent]]
        Two lists of detected gait events:
        - First list: events for the **left leg**
        - Second list: events for the **right leg**
    """


    # Interpolate missing values
    left_toe = cubic_interpolate_nan(left_toe)
    right_toe = cubic_interpolate_nan(right_toe)
    left_heel = cubic_interpolate_nan(left_heel)
    right_heel = cubic_interpolate_nan(right_heel)

    # Butterworth filter
    left_toe_f = butterworth_filter(left_toe, cutoff=5, order=2, fs=frame_rate)
    right_toe_f = butterworth_filter(right_toe, cutoff=5, order=2, fs=frame_rate)
    left_heel_f = butterworth_filter(left_heel, cutoff=5, order=2, fs=frame_rate)
    right_heel_f = butterworth_filter(right_heel, cutoff=5, order=2, fs=frame_rate)

    left_toe_grad = np.abs(np.gradient(left_toe_f, 1.0 / frame_rate))
    right_toe_grad = np.abs(np.gradient(right_toe_f, 1.0 / frame_rate))
    left_heel_grad = np.abs(np.gradient(left_heel_f, 1.0 / frame_rate))
    right_heel_grad = np.abs(np.gradient(right_heel_f, 1.0 / frame_rate))

    # Dynamic threshold
    if threshold is None:
        threshold = threshold_percentage * np.max(left_toe_grad)


    left_events = []
    right_events = []

    # Left leg
    for val in np.where((left_toe_grad[:-1] <= threshold) & (left_toe_grad[1:] > threshold))[0]:
        left_events.append(GaitEvent(frame=val, event_type=GaitEventType.TOE_OFF))

    for val in np.where((left_heel_grad[:-1] > threshold) & (left_heel_grad[1:] <= threshold))[0]:
        left_events.append(GaitEvent(frame=val, event_type=GaitEventType.HEEL_STRIKE))

    # Right leg
    for val in np.where((right_toe_grad[:-1] <= threshold) & (right_toe_grad[1:] > threshold))[0]:
        right_events.append(GaitEvent(frame=val, event_type=GaitEventType.TOE_OFF))

    for val in np.where((right_heel_grad[:-1] > threshold) & (right_heel_grad[1:] <= threshold))[0]:
        right_events.append(GaitEvent(frame=val, event_type=GaitEventType.HEEL_STRIKE))


    # DEBUG THRESHOLD PLOT
    if debug:
        plt.plot(left_heel_grad, label="Left Heel Grad")
        plt.plot(left_toe_grad, label="Left Toe Grad")
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')

        for ev in left_events:
            color = 'g' if ev.event_type == GaitEventType.HEEL_STRIKE else 'm'
            plt.axvline(x=ev.frame, color=color, linestyle=':', alpha=0.7)

        plt.legend()

    return left_events, right_events
