import os

import numpy as np
from utils.registry import HEURISTICS
from ..base import BaseHeuristicDetector
from gaitStructs import GaitEvent, GaitEventType
from utils.data import find_minima_maxima, butterworth_filter


@HEURISTICS.register
class Hsue(BaseHeuristicDetector):
    """
    Implementation of Hsue et al. (2007).
    Based on the Antero-Posterior (AP) acceleration patterns of the foot markers.

    Detection Logic:
    - HS: Local extremum of Heel AP acceleration (rapid deceleration at contact).
    - TO: Local extremum of Toe AP acceleration (rapid acceleration at push-off).
    """

    def get_required_keypoints(self):
        return ["HIP", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_HEEL", "RIGHT_HEEL"]

    def detect_events(self, processed_data, plot_path: str = None, sequence_number: int = 1):
        """
        Detects gait events using the Hsue et al. method.

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

        # --- DEBUG PLOT START ---
        if plot_path is not None and self.save_debug_plot:
            file_name = "hsue_clip_" + str(sequence_number) + ".png"
            path = os.path.join(plot_path, file_name)

            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
            fig.suptitle(f"Hsue et al. | Moving Right: {is_moving_right} | Cutoff: {foot_cutoff}Hz", fontsize=14)

            def plot_hsue_leg(ax, acc_signal, pos_signal, peaks, title, line_color, marker_color, peak_type):
                # Left axis - Acceleration
                ax.set_title(title)
                ax.set_xlabel('Frame Index')
                ax.set_ylabel('AP Acceleration ($px/s^2$)', color=line_color)
                line1, = ax.plot(acc_signal, color=line_color, alpha=0.9, linewidth=1.5, label='Acceleration')

                px = [i for i, x in enumerate(peaks) if x is not None]
                py = [x for x in peaks if x is not None]

                marker = "v" if peak_type == "min" else "^"
                scatter = None
                if len(px) > 0:
                    scatter = ax.scatter(px, py, c=marker_color, marker=marker, s=80, zorder=5, edgecolors='black',
                                         label='Event Candidate')

                ax.grid(True, alpha=0.3)

                # Right axis - X Position
                ax2 = ax.twinx()
                ax2.set_ylabel('Position X (px)', color='gray', alpha=0.8)
                line2, = ax2.plot(pos_signal, color='gray', alpha=0.3, linestyle='--', label='Position X Trace')

                # Legend
                handles = [line1, line2]
                if scatter: handles.append(scatter)
                labels = [h.get_label() for h in handles]
                loc_pos = 'upper left' if peak_type == 'min' else 'lower left'

                ax.legend(handles, labels, loc=loc_pos)

            # Color palette
            COL_HEEL = 'blue'
            COL_TOE = 'green'
            COL_HS = 'red'
            COL_TO = 'orange'

            # --- HEEL STRIKE ---
            # Logic: Moving Right -> HS is Min Accel
            hs_type = "min" if is_moving_right else "max"
            l_hs_peaks = lh_min if is_moving_right else lh_max
            r_hs_peaks = rh_min if is_moving_right else rh_max

            plot_hsue_leg(axs[0, 0], l_heel_acc, l_heel_x, l_hs_peaks,
                          f"Left Heel: AP Accel (HS={hs_type})", COL_HEEL, COL_HS, hs_type)

            plot_hsue_leg(axs[0, 1], r_heel_acc, r_heel_x, r_hs_peaks,
                          f"Right Heel: AP Accel (HS={hs_type})", COL_HEEL, COL_HS, hs_type)

            # --- TOE OFF ---
            # Logic: Moving Right -> TO is Max Accel
            to_type = "max" if is_moving_right else "min"
            l_to_peaks = lt_max if is_moving_right else lt_min
            r_to_peaks = rt_max if is_moving_right else rt_min

            plot_hsue_leg(axs[1, 0], l_toe_acc, l_toe_x, l_to_peaks,
                          f"Left Toe: AP Accel (TO={to_type})", COL_TOE, COL_TO, to_type)

            plot_hsue_leg(axs[1, 1], r_toe_acc, r_toe_x, r_to_peaks,
                          f"Right Toe: AP Accel (TO={to_type})", COL_TOE, COL_TO, to_type)

            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"[Output] Plot saved to: {path}")
        # --- DEBUG PLOT END ---

        return left_events, right_events
