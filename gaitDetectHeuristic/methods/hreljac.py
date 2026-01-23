import numpy as np
from ..utils.registry import HEURISTICS
from ..base import BaseHeuristicDetector
from gaitStructs import GaitEvent, GaitEventType
from utils.data import find_minima_maxima, butterworth_filter


@HEURISTICS.register
class Hreljac(BaseHeuristicDetector):
    """
    Implementation of Hreljac et al. (2000).
    HS: Local maximum (impact) of Vertical Heel Acceleration.
    TO: Local maximum (propulsion) of Horizontal Toe Acceleration.
    """

    def get_required_keypoints(self):
        return ["HIP", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_HEEL", "RIGHT_HEEL"]

    def detect_events(self, data):
        # Unpack pre-processed data from the base class
        hip_x = data["HIP"][:, 0]
        raw_l_heel_y = data["LEFT_HEEL"][:, 1]
        raw_r_heel_y = data["RIGHT_HEEL"][:, 1]
        raw_l_toe_x = data["LEFT_FOOT_INDEX"][:, 0]
        raw_r_toe_x = data["RIGHT_FOOT_INDEX"][:, 0]

        # --- Algorithm Params ---
        acc_cutoff = self.algorithm_params.get("accel_filter_cutoff", 6.0)
        acc_order = self.algorithm_params.get("accel_filter_order", 2)

        min_dist = self.algorithm_params.get("min_peak_distance", 20)
        prominence = self.algorithm_params.get("prominence", None)

        # Determine the direction of walking
        overall_displacement = hip_x[-1] - hip_x[0]
        is_moving_right = overall_displacement > 0

        dt = 1.0 / self.framerate

        # Helper function: Filtering + Double Derivative
        def get_acceleration(arr):
            pos_filtered = butterworth_filter(
                arr,
                cutoff=acc_cutoff,
                fs=self.framerate,
                order=acc_order
            )
            # 1. Derivative (Velocity)
            vel = np.gradient(pos_filtered, dt)
            # 2. Derivative (Acceleration)
            acc = np.gradient(vel, dt)
            return acc

        # Acceleration calculation
        l_acc_y = get_acceleration(raw_l_heel_y)  # Vertical Heel
        r_acc_y = get_acceleration(raw_r_heel_y)

        l_acc_x = get_acceleration(raw_l_toe_x)  # Horizontal Toe
        r_acc_x = get_acceleration(raw_r_toe_x)

        # Find gait event candidates
        lhy_min, lhy_max = find_minima_maxima(l_acc_y, distance=min_dist, prominence=prominence)
        rhy_min, rhy_max = find_minima_maxima(r_acc_y, distance=min_dist, prominence=prominence)

        ltx_min, ltx_max = find_minima_maxima(l_acc_x, distance=min_dist, prominence=prominence)
        rtx_min, rtx_max = find_minima_maxima(r_acc_x, distance=min_dist, prominence=prominence)

        left_events = []
        right_events = []

        # Helper function for adding events
        def add_events(indices, event_type, target_list):
            for i, val in enumerate(indices):
                if val is not None:
                    target_list.append(GaitEvent(frame=i, event_type=event_type))


        # --- DETECTION LOGIC ---

        # --- HEEL STRIKE (Vertical Heel Accel) ---
        # HS: Max vertical acceleration
        # Y-down data: Negative acceleration = MINIMUM.
        add_events(lhy_min, GaitEventType.HEEL_STRIKE, left_events)
        add_events(rhy_min, GaitEventType.HEEL_STRIKE, right_events)

        # --- TOE OFF (Horizontal Toe Accel) ---
        # Max horizontal acceleration
        if is_moving_right:
            # ->
            # +X -> Searching for MAXIMUM
            add_events(ltx_max, GaitEventType.TOE_OFF, left_events)
            add_events(rtx_max, GaitEventType.TOE_OFF, right_events)
        else:
            # <-
            # -X -> Searching for MINIMUM
            add_events(ltx_min, GaitEventType.TOE_OFF, left_events)
            add_events(rtx_min, GaitEventType.TOE_OFF, right_events)

        # Sort and return values
        left_events.sort(key=lambda x: x.frame)
        right_events.sort(key=lambda x: x.frame)

        # --- DEBUG PLOT START ---
        # import matplotlib.pyplot as plt
        #
        # fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        # fig.suptitle(f"Hreljac Debug | Moving Right: {is_moving_right} | Cutoff: {acc_cutoff}Hz", fontsize=14)
        #
        # --- HEEL VERTICAL ACCELERATION (HS) ---
        # # Left Heel
        # axs[0, 0].set_title("Left Heel Vertical Accel (Target: HS = Min)")
        # axs[0, 0].plot(l_acc_y, color='#2980b9', alpha=0.8, label='Accel Y')
        # axs[0, 0].plot(lhy_min, "v", color='red', markersize=10, label='HS Candidate')  # HS is always Min (Impact)
        # axs[0, 0].grid(True, alpha=0.3)
        # axs[0, 0].legend(loc='upper right')
        #
        # # Right Heel
        # axs[0, 1].set_title("Right Heel Vertical Accel (Target: HS = Min)")
        # axs[0, 1].plot(r_acc_y, color='#2980b9', alpha=0.8, label='Accel Y')
        # axs[0, 1].plot(rhy_min, "v", color='red', markersize=10, label='HS Candidate')
        # axs[0, 1].grid(True, alpha=0.3)
        # axs[0, 1].legend(loc='upper right')
        #
        # # --- TOE HORIZONTAL ACCELERATION (TO) ---
        # # Logic depends on direction
        # to_marker = "^" if is_moving_right else "v"
        # to_label = "Max" if is_moving_right else "Min"
        #
        # # Left Toe
        # axs[1, 0].set_title(f"Left Toe Horizontal Accel (Target: TO = {to_label})")
        # axs[1, 0].plot(l_acc_x, color='#27ae60', alpha=0.8, label='Accel X')
        # if is_moving_right:
        #     axs[1, 0].plot(ltx_max, to_marker, color='red', markersize=10, label='TO Candidate')
        # else:
        #     axs[1, 0].plot(ltx_min, to_marker, color='red', markersize=10, label='TO Candidate')
        # axs[1, 0].grid(True, alpha=0.3)
        # axs[1, 0].legend(loc='upper right')
        #
        # # Right Toe
        # axs[1, 1].set_title(f"Right Toe Horizontal Accel (Target: TO = {to_label})")
        # axs[1, 1].plot(r_acc_x, color='#27ae60', alpha=0.8, label='Accel X')
        # if is_moving_right:
        #     axs[1, 1].plot(rtx_max, to_marker, color='red', markersize=10, label='TO Candidate')
        # else:
        #     axs[1, 1].plot(rtx_min, to_marker, color='red', markersize=10, label='TO Candidate')
        # axs[1, 1].grid(True, alpha=0.3)
        # axs[1, 1].legend(loc='upper right')
        #
        # plt.tight_layout()
        # plt.show()
        # --- DEBUG PLOT END ---

        return left_events, right_events
