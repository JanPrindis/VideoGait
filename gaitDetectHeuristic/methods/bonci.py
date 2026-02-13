import copy
import os

import numpy as np
from utils.registry import HEURISTICS
from ..base import BaseHeuristicDetector
from utils.gait_structs import GaitEvent, GaitEventType
from .zeni import Zeni
from utils.data import butterworth_filter
from utils.logger import log


@HEURISTICS.register
class Bonci(BaseHeuristicDetector):
    """
        Implementation of Bonci et al. (2014).
        Refines Zeni's approach by incorporating velocity thresholds relative to walking speed to filter false positives.

        Detection Logic:
        - HS: Heel is at maximum forward distance from Hip AND Heel velocity is below a dynamic threshold.
        - TO: Toe is at maximum backward distance from Hip AND Toe velocity is above a dynamic threshold.
    """

    def get_required_keypoints(self):
        return ["HIP", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_HEEL", "RIGHT_HEEL"]

    def detect_events(self, processed_data, plot_path: str = None, sequence_number: int = 1):
        """
        Detects gait events using the Bonci et al. method.

        Args:
            processed_data (dict): A dictionary of processed keypoint data (numpy arrays).
            plot_path (str, optional): Path to save debug plots. Defaults to None.
            sequence_number (int, optional): Sequence identifier for file naming. Defaults to 1.

        Returns:
            tuple: Two lists (left_events, right_events) containing detected GaitEvent objects.
        """
        # --- Zeni et al. wrapper ---
        # Get Zeni params from config file
        zeni_specific_params = self.algorithm_params.get("zeni_params", {})
        zeni_config = self.config.model_copy(deep=True)

        zeni_config.event_detector.heuristic.params = zeni_specific_params

        # Create detector instance
        base_detector = Zeni(zeni_config)

        # Get base detector events
        base_left, base_right = base_detector.detect_events(processed_data, plot_path, sequence_number)

        # --- Calculate walking speed from hip movement ---
        dt = 1.0 / self.framerate

        # Filter parameters
        cutoff = self.algorithm_params.get("velocity_filter_cutoff", 6.0)
        order = self.algorithm_params.get("velocity_filter_order", 4)

        # Helper for velocity calculation (only 2D, because we don't have 3D data)
        def get_velocity_2d(kpt_name):
            raw_x = processed_data[kpt_name][:, 0]
            raw_y = processed_data[kpt_name][:, 1]

            # Smooth raw data
            smooth_x = butterworth_filter(raw_x, cutoff=cutoff, fs=self.framerate, order=order)
            smooth_y = butterworth_filter(raw_y, cutoff=cutoff, fs=self.framerate, order=order)

            # Velocity calculation
            vx = np.gradient(smooth_x, dt)
            vy = np.gradient(smooth_y, dt)

            # Magnitude (2D approximation of 3D velocity)
            return np.sqrt(vx ** 2 + vy ** 2)

        # Average walking speed
        hip_vel = get_velocity_2d("HIP")
        walking_speed = np.mean(hip_vel)

        # Static video fallback - makes the walking speed super high, so low velocity doesn't create fake events
        if walking_speed < 5.0:
            walking_speed = 1000.0

        # Calculate velocities of heels and toes
        l_heel_vel = get_velocity_2d("LEFT_HEEL")
        r_heel_vel = get_velocity_2d("RIGHT_HEEL")
        l_toe_vel = get_velocity_2d("LEFT_FOOT_INDEX")
        r_toe_vel = get_velocity_2d("RIGHT_FOOT_INDEX")

        # --- Thresholding ---
        # Get threshold values - default corresponds to the original paper
        hs_factor = self.algorithm_params.get("hs_velocity_factor", 0.5)
        to_factor = self.algorithm_params.get("to_velocity_factor", 0.8)

        search_window_size = self.algorithm_params.get("search_window_size", 10)

        # Calculate absolute velocity thresholds
        hs_thresh = hs_factor * walking_speed
        to_thresh = to_factor * walking_speed

        refined_left = []
        refined_right = []

        # --- REFINEMENT LOGIC ---
        def process_side(base_events, heel_vel, toe_vel, target_list):
            for event in base_events:
                frame = event.frame
                if frame >= len(heel_vel): continue  # Safety check

                if event.event_type == GaitEventType.HEEL_STRIKE:
                    # HEEL STRIKE - Heel Velocity should be low at impact
                    val = heel_vel[frame]

                    if val < hs_thresh:
                        target_list.append(event)
                    else:
                        target_list.append(event)

                elif event.event_type == GaitEventType.TOE_OFF:
                    # TOE OFF - Toe Velocity should be high (swing initiation)
                    val = toe_vel[frame]

                    if val > to_thresh:
                        # Falls above the threshold -> Adjust
                        # Try to find a peak inside the search window
                        search_window = search_window_size
                        start = max(0, frame - search_window)
                        end = min(len(heel_vel), frame + search_window)

                        roi = heel_vel[start:end]
                        # Find maximum inside the search window
                        if len(roi) > 0:
                            local_max_idx = np.argmax(roi)
                            adjusted_frame = start + local_max_idx

                            # Create new event with fine-tuned time
                            new_event = GaitEvent(frame=adjusted_frame, event_type=GaitEventType.TOE_OFF)
                            target_list.append(new_event)
                        else:
                            target_list.append(event)

                    else:
                        # Does not match the threshold -> Soft fallback (keep original)
                        target_list.append(event)

        # Refine both legs
        process_side(base_left, l_heel_vel, l_toe_vel, refined_left)
        process_side(base_right, r_heel_vel, r_toe_vel, refined_right)

        # Sort and return values
        refined_left.sort(key=lambda x: x.frame)
        refined_right.sort(key=lambda x: x.frame)

        # --- DEBUG PLOT START ---
        if plot_path is not None and self.save_debug_plot:
            file_name = "bonci_clip_" + str(sequence_number) + ".png"
            path = os.path.join(plot_path, file_name)

            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f"Bonci et al. | Walking Speed: {walking_speed:.2f} px/s", fontsize=14)

            # Helper - visualize single leg
            def plot_leg_debug(ax, title, heel_vel, toe_vel, base_events, refined_events, side_color):
                ax.set_title(title)
                ax.set_ylabel('Velocity (px/s)', color=side_color)
                ax.set_xlabel('Frame Index')

                # Velocity signals
                ax.plot(heel_vel, label="Heel Vel", color='blue', alpha=0.7, linewidth=1.5)
                ax.plot(toe_vel, label="Toe Vel", color='green', alpha=0.7, linewidth=1.5)

                # Thresholds
                ax.axhline(y=hs_thresh, color='blue', linestyle='--', alpha=0.4, label="HS Thresh")
                ax.axhline(y=to_thresh, color='green', linestyle='--', alpha=0.4, label="TO Thresh")

                # Search window
                window_label_added = False
                for e in base_events:
                    if e.event_type == GaitEventType.TOE_OFF:
                        if toe_vel[e.frame] > to_thresh:
                            start = max(0, e.frame - search_window_size)
                            end = min(len(heel_vel), e.frame + search_window_size)

                            label = "Search Window" if not window_label_added else None
                            ax.axvspan(start, end, color='gold', alpha=0.2, label=label)
                            window_label_added = True

                # Original Zeni detected events
                base_hs = [e.frame for e in base_events if e.event_type == GaitEventType.HEEL_STRIKE]
                base_to = [e.frame for e in base_events if e.event_type == GaitEventType.TOE_OFF]

                if base_hs: ax.scatter(base_hs, [heel_vel[i] for i in base_hs], c='gray', marker='v', s=100, alpha=0.8,
                                       label="Zeni Base")
                if base_to: ax.scatter(base_to, [toe_vel[i] for i in base_to], c='gray', marker='^', s=100, alpha=0.8)

                # Refined events
                for e in refined_events:
                    if e.event_type == GaitEventType.HEEL_STRIKE:
                        ax.scatter(e.frame, heel_vel[e.frame], c='red', marker='v', s=100, zorder=5, edgecolors='black',
                                   label="HS Final")
                    elif e.event_type == GaitEventType.TOE_OFF:
                        ax.scatter(e.frame, toe_vel[e.frame], c='orange', marker='^', s=100, zorder=5, edgecolors='black',
                                   label="TO Final")

                # Legend
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
                ax.grid(True, alpha=0.3)

            plot_leg_debug(axs[0], "Left Leg Analysis", l_heel_vel, l_toe_vel, base_left, refined_left, 'blue')
            plot_leg_debug(axs[1], "Right Leg Analysis", r_heel_vel, r_toe_vel, base_right, refined_right, 'red')

            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            log("BONCI", f"Plot saved to: {path}", level="success")
            # --- DEBUG PLOT END ---

        return refined_left, refined_right
