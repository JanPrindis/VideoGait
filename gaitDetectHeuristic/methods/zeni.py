import os

import numpy as np
from ..utils.registry import HEURISTICS
from ..base import BaseHeuristicDetector
from gaitStructs import GaitEvent, GaitEventType
from utils.data import find_minima_maxima


@HEURISTICS.register
class Zeni(BaseHeuristicDetector):
    """
    Implementation of Zeni et al. (2008).
    Also known as the "Coordinate Based Algorithm" (CBA). Based on the relative distance to the sacrum/pelvis.

    Detection Logic:
    - HS: Maximum anterior distance between Heel and Hip (Heel furthest forward).
    - TO: Maximum posterior distance between Toe and Hip (Toe furthest back).
    """

    def get_required_keypoints(self):
        return ["HIP", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_HEEL", "RIGHT_HEEL"]

    def detect_events(self, processed_data, plot_path: str = None, sequence_number: int = 1):
        """
        Detects gait events using the Zeni et al. method.

        Args:
            processed_data (dict): A dictionary of processed keypoint data (numpy arrays).
            plot_path (str, optional): Path to save debug plots. Defaults to None.
            sequence_number (int, optional): Sequence identifier for file naming. Defaults to 1.

        Returns:
            tuple: Two lists (left_events, right_events) containing detected GaitEvent objects.
        """
        # Unpack pre-processed data from the base class
        hip_x = processed_data["HIP"][:, 0]
        l_toe_x = processed_data["LEFT_FOOT_INDEX"][:, 0]
        r_toe_x = processed_data["RIGHT_FOOT_INDEX"][:, 0]
        l_heel_x = processed_data["LEFT_HEEL"][:, 0]
        r_heel_x = processed_data["RIGHT_HEEL"][:, 0]

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

        # --- DEBUG PLOT START ---
        if plot_path is not None and self.save_debug_plot:
            file_name = "zeni_clip_" + str(sequence_number) + ".png"
            path = os.path.join(plot_path, file_name)

            import matplotlib.pyplot as plt

            # Color palette
            COL_HEEL = 'blue'
            COL_TOE = 'green'
            COL_HS = 'red'
            COL_TO = 'orange'

            fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
            fig.suptitle(f"Zeni (Hip-Relative Distance) | Moving Right: {is_moving_right} | Min Dist: {min_dist}",
                         fontsize=14)

            def plot_zeni_leg(ax, signal, peaks, title, line_color, marker_color, peak_type):
                ax.set_title(title)
                ax.set_ylabel('Distance from Hip (px)', color=line_color)
                ax.set_xlabel('Frame Index')

                # Main signal line
                ax.plot(signal, color=line_color, alpha=0.9, linewidth=1.5, label='Rel. Distance')

                # Hip Reference (Zero Line)
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Hip Center (0)')

                # Event candidates
                px = [i for i, x in enumerate(peaks) if x is not None]
                py = [x for x in peaks if x is not None]

                marker = "v" if peak_type == "min" else "^"

                if len(px) > 0:
                    ax.scatter(px, py, c=marker_color, marker=marker, s=80, zorder=5, edgecolors='black',
                               label='Event Candidate')

                # Legend
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)

            # --- HEEL STRIKE ---
            # Moving Right -> HS is Max Forward Distance (Peak)
            hs_type = "max" if is_moving_right else "min"
            l_hs_peaks = lh_max if is_moving_right else lh_min
            r_hs_peaks = rh_max if is_moving_right else rh_min

            plot_zeni_leg(axs[0, 0], l_heel_dist, l_hs_peaks,
                          f"Left Heel: Hip-Rel Dist (HS={hs_type})", COL_HEEL, COL_HS, hs_type)

            plot_zeni_leg(axs[0, 1], r_heel_dist, r_hs_peaks,
                          f"Right Heel: Hip-Rel Dist (HS={hs_type})", COL_HEEL, COL_HS, hs_type)

            # --- TOE OFF ---
            # Moving Right -> TO is Max Backward Distance (Valley/Min)
            to_type = "min" if is_moving_right else "max"
            l_to_peaks = lt_min if is_moving_right else lt_max
            r_to_peaks = rt_min if is_moving_right else rt_max

            plot_zeni_leg(axs[1, 0], l_toe_dist, l_to_peaks,
                          f"Left Toe: Hip-Rel Dist (TO={to_type})", COL_TOE, COL_TO, to_type)

            plot_zeni_leg(axs[1, 1], r_toe_dist, r_to_peaks,
                          f"Right Toe: Hip-Rel Dist (TO={to_type})", COL_TOE, COL_TO, to_type)

            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"[Output] Plot saved to: {path}")
        # --- DEBUG PLOT END ---

        return left_events, right_events
