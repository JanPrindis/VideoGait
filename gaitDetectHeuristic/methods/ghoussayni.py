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

    def detect_events(self, processed_data, plot_path: str = None, sequence_number: int = 1):
        """
        Detects gait events using the Ghoussayni et al. method.

        Args:
            processed_data (dict): A dictionary of processed keypoint data (numpy arrays).
            plot_path (str, optional): Path to save debug plots. Defaults to None.
            sequence_number (int, optional): Sequence identifier for file naming. Defaults to 1.

        Returns:
            tuple: Two lists (left_events, right_events) containing detected GaitEvent objects.
        """
        # Unpack data
        l_toe = processed_data["LEFT_FOOT_INDEX"]
        r_toe = processed_data["RIGHT_FOOT_INDEX"]
        l_heel = processed_data["LEFT_HEEL"]
        r_heel = processed_data["RIGHT_HEEL"]

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

        # --- DEBUG PLOT START ---
        if plot_path is not None and self.save_debug_plot:
            file_name = "ghoussayni_clip_" + str(sequence_number) + ".png"
            path = os.path.join(plot_path, file_name)

            import matplotlib.pyplot as plt

            l_h_th = get_thresh(l_heel_vel)
            l_t_th = get_thresh(l_toe_vel)
            r_h_th = get_thresh(r_heel_vel)
            r_t_th = get_thresh(r_toe_vel)

            fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
            fig.suptitle(f"Ghoussayni et al. | Ratio: {rel_threshold}", fontsize=14)

            # --- LEFT LEG ---
            # Left Heel (HS Detection)
            axs[0, 0].set_title(f"Left Heel Velocity (HS Logic) | Thresh: {l_h_th:.1f}")
            axs[0, 0].set_ylabel('Heel Velocity (px/s)', color='blue')
            axs[0, 0].set_xlabel('Frame Index')
            axs[0, 0].plot(l_heel_vel, color='blue', label="Heel Velocity")
            axs[0, 0].axhline(y=l_h_th, color='red', linestyle='--', alpha=0.7, label="Threshold (Falling)")

            for e in left_events:
                if e.event_type == GaitEventType.HEEL_STRIKE:
                    axs[0, 0].scatter(e.frame, l_heel_vel[e.frame], c='red', marker='v', s=80, zorder=5, edgecolors='black')

            axs[0, 0].legend(loc='upper right')
            axs[0, 0].grid(True, alpha=0.3)

            # Left Toe (TO Detection)
            axs[0, 1].set_title(f"Left Toe Velocity (TO Logic) | Thresh: {l_t_th:.1f}")
            axs[0, 1].set_ylabel('Toe Velocity (px/s)', color='green')
            axs[0, 1].set_xlabel('Frame Index')
            axs[0, 1].plot(l_toe_vel, color='green', label="Toe Velocity")
            axs[0, 1].axhline(y=l_t_th, color='orange', linestyle='--', alpha=0.7, label="Threshold (Rising)")

            for e in left_events:
                if e.event_type == GaitEventType.TOE_OFF:
                    axs[0, 1].scatter(e.frame, l_toe_vel[e.frame], c='orange', marker='^', s=80, zorder=5,
                                      edgecolors='black')

            axs[0, 1].legend(loc='upper right')
            axs[0, 1].grid(True, alpha=0.3)

            # --- RIGHT LEG ---
            # Right Heel (HS Detection)
            axs[1, 0].set_title(f"Right Heel Velocity (HS Logic) | Thresh: {r_h_th:.1f}")
            axs[1, 0].set_ylabel('Heel Velocity (px/s)', color='blue')
            axs[1, 0].set_xlabel('Frame Index')
            axs[1, 0].plot(r_heel_vel, color='blue', label="Heel Velocity")
            axs[1, 0].axhline(y=r_h_th, color='red', linestyle='--', alpha=0.7, label="Threshold (Falling)")

            for e in right_events:
                if e.event_type == GaitEventType.HEEL_STRIKE:
                    axs[1, 0].scatter(e.frame, r_heel_vel[e.frame], c='red', marker='v', s=80, zorder=5, edgecolors='black')

            axs[1, 0].legend(loc='upper right')
            axs[1, 0].grid(True, alpha=0.3)

            # Right Toe (TO Detection)
            axs[1, 1].set_title(f"Right Toe Velocity (TO Logic) | Thresh: {r_t_th:.1f}")
            axs[1, 1].set_ylabel('Toe Velocity (px/s)', color='green')
            axs[1, 1].set_xlabel('Frame Index')
            axs[1, 1].plot(r_toe_vel, color='green', label="Toe Velocity")
            axs[1, 1].axhline(y=r_t_th, color='orange', linestyle='--', alpha=0.7, label="Threshold (Rising)")

            for e in right_events:
                if e.event_type == GaitEventType.TOE_OFF:
                    axs[1, 1].scatter(e.frame, r_toe_vel[e.frame], c='orange', marker='^', s=80, zorder=5,
                                      edgecolors='black')

            axs[1, 1].legend(loc='upper right')
            axs[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"[Output] Plot saved to: {path}")
        # --- DEBUG PLOT END ---

        return left_events, right_events
