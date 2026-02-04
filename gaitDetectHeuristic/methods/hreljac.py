import os

import numpy as np
from utils.registry import HEURISTICS
from ..base import BaseHeuristicDetector
from gaitStructs import GaitEvent, GaitEventType
from utils.data import find_minima_maxima, butterworth_filter


@HEURISTICS.register
class Hreljac(BaseHeuristicDetector):
    """
    Implementation of Hreljac et al. (2000).
    Based on identifying acceleration peaks caused by impact and propulsion forces.

    Detection Logic:
    - HS: Peak vertical acceleration of the Heel (impact shock).
    - TO: Peak horizontal acceleration of the Toe (propulsion).
    """

    def get_required_keypoints(self):
        return ["HIP", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_HEEL", "RIGHT_HEEL"]

    def detect_events(self, processed_data, plot_path: str = None, sequence_number: int = 1):
        """
        Detects gait events using the Hreljac et al. method.

        Args:
            processed_data (dict): A dictionary of processed keypoint data (numpy arrays).
            plot_path (str, optional): Path to save debug plots. Defaults to None.
            sequence_number (int, optional): Sequence identifier for file naming. Defaults to 1.

        Returns:
            tuple: Two lists (left_events, right_events) containing detected GaitEvent objects.
        """
        # Unpack pre-processed data from the base class
        hip_x = processed_data["HIP"][:, 0]
        raw_l_heel_y = processed_data["LEFT_HEEL"][:, 1]
        raw_r_heel_y = processed_data["RIGHT_HEEL"][:, 1]
        raw_l_toe_x = processed_data["LEFT_FOOT_INDEX"][:, 0]
        raw_r_toe_x = processed_data["RIGHT_FOOT_INDEX"][:, 0]

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
        if plot_path is not None and self.save_debug_plot:
            file_name = "hreljac_clip_" + str(sequence_number) + ".png"
            path = os.path.join(plot_path, file_name)

            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
            fig.suptitle(f"Hreljac et al. | Moving Right: {is_moving_right} | Cutoff: {acc_cutoff}Hz", fontsize=14)

            def plot_hreljac_leg(ax, acc_signal, pos_signal, peaks, title, acc_color, pos_color, peak_type):
                # Left axis - Acceleration
                ax.set_title(title)
                ax.set_ylabel('Acceleration ($px/s^2$)', color=acc_color)
                ax.set_xlabel('Frame Index')
                line1, = ax.plot(acc_signal, color=acc_color, alpha=0.9, linewidth=1.5, label='Acceleration')

                px = [i for i, x in enumerate(peaks) if x is not None]
                py = [x for x in peaks if x is not None]

                marker = "v" if peak_type == "min" else "^"
                color = "red" if peak_type == "min" else "orange"
                scatter = None
                if len(px) > 0:
                    scatter = ax.scatter(px, py, c=color, marker=marker, s=80, zorder=5, edgecolors='black',
                                         label='Event Candidate')

                ax.grid(True, alpha=0.3)

                # Right axis - Position
                ax2 = ax.twinx()
                ax2.set_ylabel('Position (px)', color='gray', alpha=0.8)
                line2, = ax2.plot(pos_signal, color=pos_color, alpha=0.3, linestyle='--', label='Position Trace')

                handles = [line1, line2]
                if scatter:
                    handles.append(scatter)

                labels = [h.get_label() for h in handles]
                ax.legend(handles, labels, loc='upper left')

            # --- HEEL STRIKE (Vertical) ---
            plot_hreljac_leg(axs[0, 0], l_acc_y, raw_l_heel_y, lhy_min,
                             "Left Heel: Vertical Accel (HS)", 'blue', 'gray', "min")

            plot_hreljac_leg(axs[0, 1], r_acc_y, raw_r_heel_y, rhy_min,
                             "Right Heel: Vertical Accel (HS)", 'blue', 'gray', "min")

            # --- TOE OFF (Horizontal) ---
            to_peak_type = "max" if is_moving_right else "min"
            l_target = ltx_max if is_moving_right else ltx_min
            r_target = rtx_max if is_moving_right else rtx_min

            plot_hreljac_leg(axs[1, 0], l_acc_x, raw_l_toe_x, l_target,
                             "Left Toe: Horiz Accel (TO)", 'green', 'gray', to_peak_type)

            plot_hreljac_leg(axs[1, 1], r_acc_x, raw_r_toe_x, r_target,
                             "Right Toe: Horiz Accel (TO)", 'green', 'gray', to_peak_type)

            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"[Output] Plot saved to: {path}")
        # --- DEBUG PLOT END ---

        return left_events, right_events
