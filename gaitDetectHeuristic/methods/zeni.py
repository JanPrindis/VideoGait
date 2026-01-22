import numpy as np
from ..utils.registry import HEURISTICS
from ..base import BaseHeuristicDetector
from gaitStructs import GaitEvent, GaitEventType
from utils.data import find_minima_maxima

@HEURISTICS.register
class Zeni(BaseHeuristicDetector):
    """
    Implementation of the Zeni et al. method
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
        min_dist = self.algorithm_params.get("min_peak_distance", 20)
        prominence = self.algorithm_params.get("prominence", 10)

        # Determine the direction of walking
        overall_displacement = hip_x[-1] - hip_x[0]
        is_moving_right = overall_displacement > 0

        # Heel/Toe relative distances to hip
        l_toe_dist = l_toe_x - hip_x
        l_heel_dist = l_heel_x - hip_x
        r_toe_dist = r_toe_x - hip_x
        r_heel_dist = r_heel_x - hip_x

        # Find gait event candidates
        lt_min, lt_max = find_minima_maxima(l_toe_dist, distance=min_dist, prominence=prominence)
        lh_min, lh_max = find_minima_maxima(l_heel_dist, distance=min_dist, prominence=prominence)

        rt_min, rt_max = find_minima_maxima(r_toe_dist, distance=min_dist, prominence=prominence)
        rh_min, rh_max = find_minima_maxima(r_heel_dist, distance=min_dist, prominence=prominence)

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

            # Heel Strike: Heel has a maximum distance to the hip (Max X)
            add_events(lh_max, GaitEventType.HEEL_STRIKE, left_events)
            add_events(rh_max, GaitEventType.HEEL_STRIKE, right_events)

            # Toe Off: Toe has a minimum distance to the hip (Min X)
            add_events(lt_min, GaitEventType.TOE_OFF, left_events)
            add_events(rt_min, GaitEventType.TOE_OFF, right_events)

        else:
            # <-

            # Heel Strike: Heel has a minimum distance to the hip (Min X)
            add_events(lh_min, GaitEventType.HEEL_STRIKE, left_events)
            add_events(rh_min, GaitEventType.HEEL_STRIKE, right_events)

            # Toe Off: Toe has a maximum distance to the hip (Max X)
            add_events(lt_max, GaitEventType.TOE_OFF, left_events)
            add_events(rt_max, GaitEventType.TOE_OFF, right_events)

        # Sort and return values
        left_events.sort(key=lambda x: x.frame)
        right_events.sort(key=lambda x: x.frame)

        return left_events, right_events
