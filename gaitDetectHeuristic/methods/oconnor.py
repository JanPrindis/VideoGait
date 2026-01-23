import numpy as np
from ..utils.registry import HEURISTICS
from ..base import BaseHeuristicDetector
from gaitStructs import GaitEvent, GaitEventType
from utils.data import find_minima_maxima, butterworth_filter


@HEURISTICS.register
class OConnor(BaseHeuristicDetector):
    """
    Implementation of O'Connor et al. (2007).
    Based on vertical velocity of the geometric midpoint between Heel and Toe

    Logic adapted for Computer Vision (Y-axis points DOWN):
    - HS: Maximum Vertical Velocity (fastest descent towards ground).
    - TO: Minimum Vertical Velocity (fastest ascent/lift form ground).
    """

    def get_required_keypoints(self):
        return ["LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_HEEL", "RIGHT_HEEL"]

    def detect_events(self, data):
        # Unpack pre-processed data from the base class (Vertical Y only)
        l_heel_y = data["LEFT_HEEL"][:, 1]
        r_heel_y = data["RIGHT_HEEL"][:, 1]
        l_toe_y = data["LEFT_FOOT_INDEX"][:, 1]
        r_toe_y = data["RIGHT_FOOT_INDEX"][:, 1]

        # --- Algorithm Params ---
        vel_cutoff = self.algorithm_params.get("velocity_filter_cutoff", 5.0)
        vel_order = self.algorithm_params.get("velocity_filter_order", 2)

        min_dist = self.algorithm_params.get("min_peak_distance", 20)
        prominence = self.algorithm_params.get("prominence", None)
        prominence_factor = self.algorithm_params.get("adaptive_prominence_factor", 0.8)

        dt = 1.0 / self.framerate

        # Helper function for calculating velocity of midpoint
        def get_midpoint_velocity(heel_arr, toe_arr):
            midpoint = (heel_arr + toe_arr) / 2.0

            mid_filtered = butterworth_filter(
                midpoint,
                cutoff=vel_cutoff,
                fs=self.framerate,
                order=vel_order
            )

            vel = np.gradient(mid_filtered, dt)
            return vel

        # Calculate velocities for both feet
        l_mid_vel = get_midpoint_velocity(l_heel_y, l_toe_y)
        r_mid_vel = get_midpoint_velocity(r_heel_y, r_toe_y)

        def get_adaptive_prominence(signal):
            # Calculate standard deviation
            sig_std = np.std(signal)
            if sig_std < 1e-6:
                return 0.5  # Fallback
            return sig_std * prominence_factor

        if prominence is None:
            prom_l = get_adaptive_prominence(l_mid_vel)
            prom_r = get_adaptive_prominence(r_mid_vel)
        else:
            prom_l = prominence
            prom_r = prominence

        # Left Leg
        l_min, l_max = find_minima_maxima(l_mid_vel, distance=min_dist, prominence=prom_l)

        # Right Leg
        r_min, r_max = find_minima_maxima(r_mid_vel, distance=min_dist, prominence=prom_r)

        left_events = []
        right_events = []

        # Helper function for adding events
        def add_events(indices, event_type, target_list):
            for i, val in enumerate(indices):
                if val is not None:
                    target_list.append(GaitEvent(frame=i, event_type=event_type))

        # --- DETECTION LOGIC ---
        # --- HEEL STRIKE (HS) ---
        # Local Minimum
        # Peak Descent = Negative velocity (Y down) -> MAXIMUM
        add_events(l_max, GaitEventType.HEEL_STRIKE, left_events)
        add_events(r_max, GaitEventType.HEEL_STRIKE, right_events)

        # # --- TOE OFF (TO) ---
        # Local Maximum
        # Peak Ascent = Negative velocity (Y down) -> MINIMUM
        add_events(l_min, GaitEventType.TOE_OFF, left_events)
        add_events(r_min, GaitEventType.TOE_OFF, right_events)

        # Sort
        left_events.sort(key=lambda x: x.frame)
        right_events.sort(key=lambda x: x.frame)

        # # --- DEBUG PLOT START (Optional) ---
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # ax[0].set_title("Left Foot Midpoint Vertical Velocity")
        # ax[0].plot(l_mid_vel, label="Vel Y")
        # ax[0].plot(l_max, "v", color="red", label="HS (Max - Descent)")
        # ax[0].plot(l_min, "^", color="green", label="TO (Min - Ascent)")
        # ax[0].legend()
        # ax[0].grid(True)
        #
        # ax[1].set_title("Right Foot Midpoint Vertical Velocity")
        # ax[1].plot(r_mid_vel, label="Vel Y")
        # ax[1].plot(r_max, "v", color="red", label="HS (Max - Descent)")
        # ax[1].plot(r_min, "^", color="green", label="TO (Min - Ascent)")
        # ax[1].legend()
        # ax[1].grid(True)
        # plt.show()
        # --- DEBUG PLOT END ---

        return left_events, right_events
