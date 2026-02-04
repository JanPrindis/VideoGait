import os

import numpy as np
from utils.registry import HEURISTICS
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

    def detect_events(self, processed_data, plot_path: str = None, sequence_number: int = 1):
        """
        Detects gait events using the O'Connor et al. method.

        Args:
            processed_data (dict): A dictionary of processed keypoint data (numpy arrays).
            plot_path (str, optional): Path to save debug plots. Defaults to None.
            sequence_number (int, optional): Sequence identifier for file naming. Defaults to 1.

        Returns:
            tuple: Two lists (left_events, right_events) containing detected GaitEvent objects.
        """
        # Unpack pre-processed data from the base class (Vertical Y only)
        l_heel_y = processed_data["LEFT_HEEL"][:, 1]
        r_heel_y = processed_data["RIGHT_HEEL"][:, 1]
        l_toe_y = processed_data["LEFT_FOOT_INDEX"][:, 1]
        r_toe_y = processed_data["RIGHT_FOOT_INDEX"][:, 1]

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

        # --- DEBUG PLOT START ---
        if plot_path is not None and self.save_debug_plot:
            file_name = "oconnor_clip_" + str(sequence_number) + ".png"
            path = os.path.join(plot_path, file_name)

            import matplotlib.pyplot as plt

            # Calculate the midpoint positions again
            l_mid_pos = (l_heel_y + l_toe_y) / 2.0
            r_mid_pos = (r_heel_y + r_toe_y) / 2.0

            used_prom_l = prom_l if isinstance(prom_l, float) else np.mean(prom_l) if prom_l is not None else 0

            fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f"O'Connor et al. | Cutoff: {vel_cutoff}Hz | Prom: ~{used_prom_l:.2f}",
                         fontsize=14)

            def plot_oconnor_leg(ax, vel_signal, pos_signal, hs_peaks, to_peaks, title):
                # Left axis - velocity
                ax.set_title(title)
                ax.set_xlabel('Frame Index')
                ax.set_ylabel('Vert. Velocity (px/s)', color='#8e44ad')
                line1, = ax.plot(vel_signal, color='#8e44ad', alpha=0.9, linewidth=1.5, label='Midpoint Velocity')

                # HS (Max Descent -> Peak in our data because Y is down)
                px_hs = [i for i, x in enumerate(hs_peaks) if x is not None]
                py_hs = [x for x in hs_peaks if x is not None]

                scatter_hs = None
                if len(px_hs) > 0:
                    scatter_hs = ax.scatter(px_hs, py_hs, c='red', marker='v', s=80, zorder=5, edgecolors='black',
                                            label='HS Candidate (Max Vel)')

                # TO (Max Ascent -> Min in our data because Y is down)
                px_to = [i for i, x in enumerate(to_peaks) if x is not None]
                py_to = [x for x in to_peaks if x is not None]

                scatter_to = None
                if len(px_to) > 0:
                    scatter_to = ax.scatter(px_to, py_to, c='orange', marker='^', s=80, zorder=5, edgecolors='black',
                                            label='TO Candidate (Min Vel)')

                ax.grid(True, alpha=0.3)

                # Right axis - Y Position
                ax2 = ax.twinx()
                ax2.set_ylabel('Position Y (px)', color='gray', alpha=0.8)
                line2, = ax2.plot(pos_signal, color='gray', alpha=0.3, linestyle='--', label='Position Y Trace')

                # Legend
                handles = [line1, line2]
                if scatter_hs: handles.append(scatter_hs)
                if scatter_to: handles.append(scatter_to)
                labels = [h.get_label() for h in handles]

                ax.legend(handles, labels, loc='upper right')

            # LEFT LEG
            plot_oconnor_leg(axs[0], l_mid_vel, l_mid_pos, l_max, l_min,
                             "Left Leg: Midpoint Vertical Dynamics")

            # RIGHT LEG
            plot_oconnor_leg(axs[1], r_mid_vel, r_mid_pos, r_max, r_min,
                             "Right Leg: Midpoint Vertical Dynamics")

            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"[Output] Plot saved to: {path}")
        # --- DEBUG PLOT END ---

        return left_events, right_events
