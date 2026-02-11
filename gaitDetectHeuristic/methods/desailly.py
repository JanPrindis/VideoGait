import os

import numpy as np
import scipy.signal as signal
from utils.registry import HEURISTICS
from ..base import BaseHeuristicDetector
from utils.gait_structs import GaitEvent, GaitEventType
from utils.data import find_minima_maxima, butterworth_filter
from utils.logger import log


@HEURISTICS.register
class Desailly(BaseHeuristicDetector):
    """
    Implementation of Desailly et al. (2009).
    Uses signal detrending (high-pass filtering) to isolate foot oscillations from the forward progression.

    Detection Logic:
    - HS: Maximum peak of the high-pass filtered Heel X-coordinate (max forward extension).
    - TO: Minimum peak of the high-pass filtered Toe X-coordinate (max backward extension).
    """

    def get_required_keypoints(self):
        return ["HIP", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_HEEL", "RIGHT_HEEL",
                "LEFT_KNEE", "RIGHT_KNEE"]

    def detect_events(self, processed_data, plot_path: str = None, sequence_number: int = 1):
        """
        Detects gait events using the Desailly et al. method.

        Args:
            processed_data (dict): A dictionary of processed keypoint data (numpy arrays).
            plot_path (str, optional): Path to save debug plots. Defaults to None.
            sequence_number (int, optional): Sequence identifier for file naming. Defaults to 1.

        Returns:
            tuple: Two lists (left_events, right_events) containing detected GaitEvent objects.
        """
        # Unpack pre-processed data
        hip_x = processed_data["HIP"][:, 0]
        l_heel_x = processed_data["LEFT_HEEL"][:, 0]
        r_heel_x = processed_data["RIGHT_HEEL"][:, 0]
        l_toe_x = processed_data["LEFT_FOOT_INDEX"][:, 0]
        r_toe_x = processed_data["RIGHT_FOOT_INDEX"][:, 0]

        l_knee_x = processed_data["LEFT_KNEE"][:, 0]
        r_knee_x = processed_data["RIGHT_KNEE"][:, 0]

        # Determine direction
        overall_displacement = hip_x[-1] - hip_x[0]
        is_moving_right = overall_displacement > 0

        # --- FREQUENCY ESTIMATION ---
        user_freq = self.algorithm_params.get("gait_frequency", None)
        calculated_freq = 0.8

        # For debug plot
        k_max_sparse = []
        peak_indices = []
        knee_signal = None

        if user_freq is not None:
            # We have gait frequency
            gait_freq = float(user_freq)
        else:
            # We don't have gait frequency - calculate from knee distance
            knee_signal = l_knee_x - r_knee_x

            # Smoothing
            knee_smooth = butterworth_filter(knee_signal, cutoff=2.5, fs=self.framerate, order=2)

            # Peak detection
            k_min, k_max = find_minima_maxima(knee_smooth, distance=10, prominence=5.0)
            k_max_sparse = k_max
            peak_indices = [i for i, x in enumerate(k_max) if x is not None]

            # Frequency calculation
            if len(peak_indices) > 1:
                diffs = np.diff(peak_indices)
                avg_frames = np.mean(diffs)
                if avg_frames > 0:
                    calculated_freq = self.framerate / float(avg_frames)

            gait_freq = np.clip(calculated_freq, 0.1, 5.0)

        # --- DESAILLY METHOD ---
        hs_cutoff_factor = self.algorithm_params.get("hs_cutoff_factor", 0.5)
        to_cutoff_factor = self.algorithm_params.get("to_cutoff_factor", 1.1)

        fc_hs = hs_cutoff_factor * gait_freq
        fc_to = to_cutoff_factor * gait_freq

        min_dist = self.algorithm_params.get("min_peak_distance", 20)
        prominence = self.algorithm_params.get("prominence", None)


        def apply_highpass(arr, cutoff):
            safe_cutoff = min(cutoff, (self.framerate / 2) - 0.1)
            sos = signal.butter(2, safe_cutoff, btype='highpass', fs=self.framerate, output='sos')
            return signal.sosfiltfilt(sos, arr)

        hp_l_heel = apply_highpass(l_heel_x, fc_hs)
        hp_r_heel = apply_highpass(r_heel_x, fc_hs)
        hp_l_toe = apply_highpass(l_toe_x, fc_to)
        hp_r_toe = apply_highpass(r_toe_x, fc_to)

        # Peak Detection
        l_heel_min, l_heel_max = find_minima_maxima(hp_l_heel, distance=min_dist, prominence=prominence)
        r_heel_min, r_heel_max = find_minima_maxima(hp_r_heel, distance=min_dist, prominence=prominence)
        l_toe_min, l_toe_max = find_minima_maxima(hp_l_toe, distance=min_dist, prominence=prominence)
        r_toe_min, r_toe_max = find_minima_maxima(hp_r_toe, distance=min_dist, prominence=prominence)

        left_events = []
        right_events = []

        # Helper function for adding events
        def add_events(sparse_peaks, event_type, target_list):
            for i, val in enumerate(sparse_peaks):
                if val is not None:
                    target_list.append(GaitEvent(frame=i, event_type=event_type))

        if is_moving_right:
            add_events(l_heel_max, GaitEventType.HEEL_STRIKE, left_events)
            add_events(r_heel_max, GaitEventType.HEEL_STRIKE, right_events)
            add_events(l_toe_min, GaitEventType.TOE_OFF, left_events)
            add_events(r_toe_min, GaitEventType.TOE_OFF, right_events)
        else:
            add_events(l_heel_min, GaitEventType.HEEL_STRIKE, left_events)
            add_events(r_heel_min, GaitEventType.HEEL_STRIKE, right_events)
            add_events(l_toe_max, GaitEventType.TOE_OFF, left_events)
            add_events(r_toe_max, GaitEventType.TOE_OFF, right_events)

        left_events.sort(key=lambda x: x.frame)
        right_events.sort(key=lambda x: x.frame)

        # --- DEBUG PLOT START ---
        if plot_path is not None and self.save_debug_plot:
            file_name = "desailly_clip_" + str(sequence_number) + ".png"
            path = os.path.join(plot_path, file_name)

            import matplotlib.pyplot as plt

            # Helper for getting data from sparse array
            def get_plot_data(sparse_array):
                if sparse_array is None or len(sparse_array) == 0:
                    return [], []
                x_vals = [i for i, x in enumerate(sparse_array) if x is not None]
                y_vals = [x for x in sparse_array if x is not None]
                return x_vals, y_vals

            # Cutoff frequency calculation
            disp_fc_hs = hs_cutoff_factor * gait_freq
            disp_fc_to = to_cutoff_factor * gait_freq

            # Prepare grid
            has_knee = (knee_signal is not None)
            rows = 3 if has_knee else 2

            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(rows, 2)

            src = "User" if user_freq else ("Auto" if len(peak_indices) > 1 else "Fallback")
            fig.suptitle(
                f"Desailly et al. | Moving Right: {is_moving_right} | Walking Freq: {gait_freq:.2f} Hz ({src}) | Cutoffs: HS ~{disp_fc_hs:.2f}Hz, TO ~{disp_fc_to:.2f}Hz",
                fontsize=14)

            current_row = 0

            # KNEE SIGNAL
            if has_knee:
                ax_knee = fig.add_subplot(gs[current_row, :])
                ax_knee.set_title("Frequency Estimation: Knee Distance Signal")
                ax_knee.set_ylabel('L knee to R knee distance (px)', color='gray')
                ax_knee.set_xlabel('Frame Index')

                ax_knee.plot(knee_signal, color='gray', alpha=0.7, label="Raw Knee Dist")

                kx, ky = get_plot_data(k_max_sparse)
                if len(kx) > 0:
                    ax_knee.scatter(kx, ky, c='purple', s=50, zorder=5, label=f"Cycle Peaks (n={len(kx)})")

                ax_knee.legend(loc='upper right')
                ax_knee.grid(True, alpha=0.3)
                current_row += 1

            # HEEL X-COORD (HS Detection)
            # Left Heel
            ax_l_heel = fig.add_subplot(gs[current_row, 0])
            ax_l_heel.set_title("Left Heel X-Coord (High-Pass)")
            ax_l_heel.set_ylabel('Detrended Heel X (px)', color='blue')
            ax_l_heel.set_xlabel('Frame Index')
            ax_l_heel.plot(hp_l_heel, label="Left Heel", color='blue')

            target_l = l_heel_max if is_moving_right else l_heel_min
            hx, hy = get_plot_data(target_l)
            if len(hx) > 0:
                ax_l_heel.scatter(hx, hy, c='red', marker='v', s=80, zorder=5, edgecolors='black', label="HS Event")
            ax_l_heel.grid(True, alpha=0.3)
            ax_l_heel.legend(loc='upper right')

            # Right Heel
            ax_r_heel = fig.add_subplot(gs[current_row, 1], sharex=ax_l_heel)
            ax_r_heel.set_title("Right Heel X-Coord (High-Pass)")
            ax_r_heel.set_ylabel('Detrended Heel X (px)', color='blue')
            ax_r_heel.set_xlabel('Frame Index')
            ax_r_heel.plot(hp_r_heel, label="Right heel", color='blue')

            target_r = r_heel_max if is_moving_right else r_heel_min
            rhx, rhy = get_plot_data(target_r)
            if len(rhx) > 0:
                ax_r_heel.scatter(rhx, rhy, c='red', marker='v', s=80, zorder=5, edgecolors='black', label="HS Event")
            ax_r_heel.grid(True, alpha=0.3)
            ax_r_heel.legend(loc='upper right')

            current_row += 1

            # TOE X-COORD (TO Detection)
            # Left Toe
            ax_l_toe = fig.add_subplot(gs[current_row, 0], sharex=ax_l_heel)
            ax_l_toe.set_title("Left Toe X-Coord (High-Pass)")
            ax_l_toe.set_ylabel('Detrended Toe X (px)', color='green')
            ax_l_toe.set_xlabel('Frame Index')
            ax_l_toe.plot(hp_l_toe, label="Left toe", color='green')

            target_l_to = l_toe_min if is_moving_right else l_toe_max
            tx, ty = get_plot_data(target_l_to)
            if len(tx) > 0:
                ax_l_toe.scatter(tx, ty, c='orange', marker='^', s=80, zorder=5, edgecolors='black', label="TO Event")
            ax_l_toe.grid(True, alpha=0.3)
            ax_l_toe.legend(loc='upper right')

            # Right Toe
            ax_r_toe = fig.add_subplot(gs[current_row, 1], sharex=ax_l_heel)
            ax_r_toe.set_title("Right Toe X-Coord (High-Pass)")
            ax_r_toe.set_ylabel('Detrended Toe X (px)', color='green')
            ax_r_toe.set_xlabel('Frame Index')
            ax_r_toe.plot(hp_r_toe, label="Right Toe", color='green')

            target_r_to = r_toe_min if is_moving_right else r_toe_max
            rtx, rty = get_plot_data(target_r_to)
            if len(rtx) > 0:
                ax_r_toe.scatter(rtx, rty, c='orange', marker='^', s=80, zorder=5, edgecolors='black', label="TO Event")
            ax_r_toe.grid(True, alpha=0.3)
            ax_r_toe.legend(loc='upper right')

            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            log("DESAILLY", f"Plot saved to: {path}", level="success")
        # --- DEBUG PLOT END ---

        return left_events, right_events
