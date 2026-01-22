import numpy as np
from ..utils.registry import HEURISTICS
from ..base import BaseHeuristicDetector
from gaitStructs import GaitEvent, GaitEventType
from utils.data import find_minima_maxima, butterworth_filter


@HEURISTICS.register
class Hsue(BaseHeuristicDetector):
    """
    Implementation of the Hsue et al. method based on antero-posterior acceleration.
    """

    def get_required_keypoints(self):
        return ["HIP", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_HEEL", "RIGHT_HEEL"]

    def detect_events(self, data):
        # Unpack pre-processed data from the base class
        hip_x = data["HIP"][:, 0]
        l_toe_x = data["LEFT_FOOT_INDEX"][:, 0]
        r_toe_x = data["RIGHT_FOOT_INDEX"][:, 0]
        l_heel_x = data["LEFT_HEEL"][:, 0]
        r_heel_x = data["RIGHT_HEEL"][:, 0]

        # --- Algorithm Params ---
        # Peak detection parameters
        min_dist = self.algorithm_params.get("min_peak_distance", 20)
        prominence = self.algorithm_params.get("prominence", None)

        # Acceleration curve smoothing parameters
        foot_cutoff = self.algorithm_params.get("foot_accel_filter_cutoff", 2)
        foot_order = self.algorithm_params.get("foot_accel_filter_order", 2)

        # Determine the direction of walking
        overall_displacement = hip_x[-1] - hip_x[0]
        is_moving_right = overall_displacement > 0

        # Apply stronger Butterworth filter to smooth out the data for second derivation
        l_toe_f = butterworth_filter(l_toe_x, cutoff=foot_cutoff, fs=self.framerate, order=foot_order)
        r_toe_f = butterworth_filter(r_toe_x, cutoff=foot_cutoff, fs=self.framerate, order=foot_order)
        l_heel_f = butterworth_filter(l_heel_x, cutoff=foot_cutoff, fs=self.framerate, order=foot_order)
        r_heel_f = butterworth_filter(r_heel_x, cutoff=foot_cutoff, fs=self.framerate, order=foot_order)

        # Acceleration Calculation
        # (dt = 1/fps)
        dt = 1.0 / self.framerate

        def get_accel(arr):
            return np.gradient(np.gradient(arr, dt), dt)

        l_toe_acc = get_accel(l_toe_f)
        r_toe_acc = get_accel(r_toe_f)
        l_heel_acc = get_accel(l_heel_f)
        r_heel_acc = get_accel(r_heel_f)

        # Find gait event candidates
        lt_min, lt_max = find_minima_maxima(l_toe_acc, distance=min_dist, prominence=prominence)
        rt_min, rt_max = find_minima_maxima(r_toe_acc, distance=min_dist, prominence=prominence)

        lh_min, lh_max = find_minima_maxima(l_heel_acc, distance=min_dist, prominence=prominence)
        rh_min, rh_max = find_minima_maxima(r_heel_acc, distance=min_dist, prominence=prominence)

        left_events = []
        right_events = []

        # Helper function for adding events
        def add_events(indices, event_type, target_list):
            for i, val in enumerate(indices):
                if val is not None:
                    target_list.append(GaitEvent(frame=i, event_type=event_type))

        # --- DETECTION LOGIC ---
        if is_moving_right:
            # ->

            # Heel Strike: Heel has the minimal local velocity (forward movement velocity is +)
            add_events(lh_min, GaitEventType.HEEL_STRIKE, left_events)  # HS = Valley
            add_events(rh_min, GaitEventType.HEEL_STRIKE, right_events)


            # Toe Off: Toe has the maximum local velocity (forward movement velocity is +)
            add_events(lt_max, GaitEventType.TOE_OFF, left_events)      # TO = Peak
            add_events(rt_max, GaitEventType.TOE_OFF, right_events)

        else:
            # <-

            # Heel Strike: Heel has the maximum local velocity (forward movement velocity is -)
            add_events(lh_max, GaitEventType.HEEL_STRIKE, left_events)   # HS = Peak
            add_events(rh_max, GaitEventType.HEEL_STRIKE, right_events)

            # Toe Off: Toe has the minimal local velocity (forward movement velocity is -)
            add_events(lt_min, GaitEventType.TOE_OFF, left_events)      # TO = Valley
            add_events(rt_min, GaitEventType.TOE_OFF, right_events)

        # Sort and return values
        left_events.sort(key=lambda x: x.frame)
        right_events.sort(key=lambda x: x.frame)


        return left_events, right_events
