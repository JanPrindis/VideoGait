import numpy as np
from ..utils.registry import HEURISTICS
from ..base import BaseHeuristicDetector
from gaitStructs import GaitEvent, GaitEventType
from utils.data import butterworth_filter


@HEURISTICS.register
class Ghoussayni(BaseHeuristicDetector):
    """
    Implementation of Ghoussayni et al. (2004) with updated thresholds (Bruening et al., 2014).
    Based on sagittal velocity thresholds.
    """

    def get_required_keypoints(self):
        return ["LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_HEEL", "RIGHT_HEEL"]

    def detect_events(self, data):
        # Unpack data
        l_toe = data["LEFT_FOOT_INDEX"]
        r_toe = data["RIGHT_FOOT_INDEX"]
        l_heel = data["LEFT_HEEL"]
        r_heel = data["RIGHT_HEEL"]

        # --- Algorithm Params ---
        # Option 1: Fixed threshold (in case we have calibrated data in cm/s)
        # Bruening recommends 50 cm/s
        fixed_threshold = self.algorithm_params.get("velocity_threshold", None)

        # Option 2: Relative threshold (in case we have data in px)
        # Default 0.15 is about 15 % of maximal swing velocity
        rel_threshold = self.algorithm_params.get("velocity_threshold_ratio", 0.15)

        # Butterworth filter params (for smoothing input data for velocity calculation)
        vel_cutoff = self.algorithm_params.get("velocity_filter_cutoff", 6)
        vel_order = self.algorithm_params.get("velocity_filter_order", 2)

        # (dt = 1/fps)
        dt = 1.0 / self.framerate

        def get_velocity(pos_arr):
            pos_f = pos_arr.copy()
            pos_f[:, 0] = butterworth_filter(pos_arr[:, 0], cutoff=vel_cutoff, fs=self.framerate, order=vel_order)
            pos_f[:, 1] = butterworth_filter(pos_arr[:, 1], cutoff=vel_cutoff, fs=self.framerate, order=vel_order)

            # Gradient -> Velocity vector (vx, vy)
            grad = np.gradient(pos_f, axis=0) / dt

            # Sagittal velocity magnitude
            # sqrt(vx^2 + vy^2)
            speed = np.linalg.norm(grad, axis=1)
            return speed

        # Calculate velocities
        l_toe_vel = get_velocity(l_toe)
        r_toe_vel = get_velocity(r_toe)
        l_heel_vel = get_velocity(l_heel)
        r_heel_vel = get_velocity(r_heel)

        # --- Determine Threshold ---
        def get_thresh(signal):
            if fixed_threshold is not None:
                return fixed_threshold
            return np.max(signal) * rel_threshold

        # --- Event Logic ---
        left_events = []
        right_events = []

        # Helper function for detecting events
        def find_events(heel_speed, toe_speed, target_list):
            h_th = get_thresh(heel_speed)
            t_th = get_thresh(toe_speed)

            # Heel Strike: Heel velocity falls BELOW threshold
            # (Velocity is decreasing: v[i-1] > th a v[i] <= th)
            hs_indices = np.where((heel_speed[:-1] > h_th) & (heel_speed[1:] <= h_th))[0]
            for idx in hs_indices:
                target_list.append(GaitEvent(frame=idx, event_type=GaitEventType.HEEL_STRIKE))

            # 2. Toe Off: Toe velocity rises ABOVE threshold
            # (Velocity is increasing: v[i-1] <= th a v[i] > th)
            to_indices = np.where((toe_speed[:-1] <= t_th) & (toe_speed[1:] > t_th))[0]
            for idx in to_indices:
                target_list.append(GaitEvent(frame=idx, event_type=GaitEventType.TOE_OFF))

        # Run detection
        find_events(l_heel_vel, l_toe_vel, left_events)
        find_events(r_heel_vel, r_toe_vel, right_events)

        # Sort and return values
        left_events.sort(key=lambda x: x.frame)
        right_events.sort(key=lambda x: x.frame)

        return left_events, right_events
