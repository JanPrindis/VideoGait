import numpy as np
import scipy.signal as signal
from ..utils.registry import HEURISTICS
from ..base import BaseHeuristicDetector
from gaitStructs import GaitEvent, GaitEventType
from utils.data import find_minima_maxima, butterworth_filter


@HEURISTICS.register
class Desailly(BaseHeuristicDetector):
    """
    Implementation of the Desailly et al. method (2009).
    """

    def get_required_keypoints(self):
        return ["HIP", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_HEEL", "RIGHT_HEEL",
                "LEFT_KNEE", "RIGHT_KNEE"]

    def detect_events(self, data):
        # Unpack pre-processed data
        hip_x = data["HIP"][:, 0]
        l_heel_x = data["LEFT_HEEL"][:, 0]
        r_heel_x = data["RIGHT_HEEL"][:, 0]
        l_toe_x = data["LEFT_FOOT_INDEX"][:, 0]
        r_toe_x = data["RIGHT_FOOT_INDEX"][:, 0]

        l_knee_x = data["LEFT_KNEE"][:, 0]
        r_knee_x = data["RIGHT_KNEE"][:, 0]

        # Determine direction
        overall_displacement = hip_x[-1] - hip_x[0]
        is_moving_right = overall_displacement > 0

        # --- FREQUENCY ESTIMATION ---
        user_freq = self.algorithm_params.get("gait_frequency", None)
        calculated_freq = 0.8

        # For debug plot
        k_max_sparse = []
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

        # --- DEBUG PLOT ---
        # import matplotlib.pyplot as plt
        # ax_offset = 0
        #
        # def get_plot_data(sparse_array):
        #     if sparse_array is None or len(sparse_array) == 0:
        #         return [], []
        #     x_vals = [i for i, x in enumerate(sparse_array) if x is not None]
        #     y_vals = [x for x in sparse_array if x is not None]
        #     return x_vals, y_vals
        #
        # if knee_signal is not None:
        #     fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        #     src = "User" if user_freq else ("Auto" if len(peak_indices) > 1 else "Fallback")
        #     fig.suptitle(f"Desailly | Freq: {gait_freq:.2f} Hz | Source: {src}", fontsize=14)
        #
        #     # 1. Knee Signal
        #     axs[0].set_title("1. Frequency Est: Knee Distance")
        #     if knee_signal is not None:
        #         axs[0].plot(knee_signal, color='gray', label="Raw")
        #         kx, ky = get_plot_data(k_max_sparse)
        #         if len(kx) > 0:
        #             axs[0].scatter(kx, ky, c='purple', s=50, label="Cycle Peaks")
        #     axs[0].legend()
        #     axs[0].grid(True)
        # else:
        #     fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        #     ax_offset = 1
        #
        # # 2. HS Detection
        # axs[1 - ax_offset].set_title(f"2. HS Detection: HP Heel")
        # axs[1 - ax_offset].plot(hp_l_heel, label="L Heel HP", color='blue')
        #
        # target_sparse = l_heel_max if is_moving_right else l_heel_min
        # hx, hy = get_plot_data(target_sparse)
        # if len(hx) > 0:
        #     axs[1 - ax_offset].scatter(hx, hy, c='red', marker='v', zorder=5, label="HS")
        # axs[1 - ax_offset].legend()
        # axs[1 - ax_offset].grid(True)
        #
        # # 3. TO Detection
        # axs[2 - ax_offset].set_title(f"3. TO Detection: HP Toe")
        # axs[2 - ax_offset].plot(hp_l_toe, label="L Toe HP", color='green')
        #
        # target_sparse = l_toe_min if is_moving_right else l_toe_max
        # tx, ty = get_plot_data(target_sparse)
        # if len(tx) > 0:
        #     axs[2 - ax_offset].scatter(tx, ty, c='orange', marker='^', zorder=5, label="TO")
        # axs[2 - ax_offset].legend()
        # axs[2 - ax_offset].grid(True)
        #
        # plt.tight_layout()
        # plt.show()
        # --- DEBUG PLOT END ---

        return left_events, right_events
