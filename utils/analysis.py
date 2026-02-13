"""
This module provides the core logic for comprehensive gait analysis.

It takes raw keypoints and detected gait events as input and computes a wide range of
clinical metrics, including spatiotemporal parameters, kinematic angles, symmetry indices,
and aggregated gait cycle statistics.
"""
import os
import numpy as np
import json
from scipy.interpolate import interp1d
from typing import Dict

from utils.config_models import AppConfig
from utils.json_serializer import KeypointSerializer
from utils.preprocessing import preprocess_keypoints
from utils.gait_structs import PhaseType
from utils.config_utils import resolve_skeleton_from_config
from utils.logger import log


class GaitAnalyzer:
    """
    Main class for performing gait analysis calculations.
    """
    def __init__(self, app_config: AppConfig):
        """
        Initializes the analyzer with application configuration.

        Args:
            app_config (AppConfig): The application configuration object.
        """
        self.cfg = app_config
        self.pre_cfg = app_config.preprocessing
        self.skeleton = resolve_skeleton_from_config(app_config)

        # Signal processing parameters
        self.filter_cutoff = self.pre_cfg.filter_cutoff
        self.filter_order = self.pre_cfg.filter_order

    def run_analysis(self, keypoints_path: str, events_data: Dict, gait_data: Dict, output_dir: str):
        """
        Executes the full analysis pipeline and saves the results to a JSON file.

        Args:
            keypoints_path (str): Path to the input keypoints JSON file.
            events_data (Dict): Dictionary containing detected discrete gait events (HS, TO).
            gait_data (Dict): Dictionary containing continuous gait phases (Stance/Swing).
            output_dir (str): Directory where the analysis report will be saved.

        Returns:
            dict | None: The complete analysis report dictionary, or None if input file is missing.
            
        """
        filename = os.path.basename(keypoints_path)
        log("ANALYZER", f"Analyzing {filename}...", level="info")

        if not os.path.exists(keypoints_path):
            log("ANALYZER", f"Error: File {keypoints_path} not found.", level="error")
            return None

        # --- LOAD AND FILTER ---
        raw_json = KeypointSerializer.load(keypoints_path)
        if not raw_json:
            return None

        parsed_kps = self._parse_serializer_data(raw_json)
        framerate = events_data.get('framerate', 30.0)
        valid_ranges = events_data.get('global_ranges', [])

        # Apply Low-Pass Butterworth filter to smooth jitter
        filtered_kps = preprocess_keypoints(
            parsed_kps,
            frame_rate=framerate,
            filter_cutoff=self.filter_cutoff,
            filter_order=self.filter_order
        )

        # --- PREPARE DATA ---
        l_phases = gait_data.get('left_phases', [])
        r_phases = gait_data.get('right_phases', [])
        support_phases = gait_data.get('support_phases', [])

        # --- CALCULATIONS ---

        # Estimate leg length in pixels for potential normalization
        leg_length_px = self._estimate_leg_length(filtered_kps, valid_ranges)

        # Spatio-Temporal Metrics (Cadence, Step Times, Support Ratio)
        spatiotemporal = self._calc_spatiotemporal(l_phases, r_phases, support_phases, valid_ranges, framerate)

        # Spatial Parameters (Distance based)
        spatial_params = self._calc_spatial_parameters(filtered_kps, l_phases, r_phases, leg_length_px, spatiotemporal)

        # Kinematics (Angles with Clinical Offsets and Direction Correction)
        kinematics_raw = self._calc_kinematics_signed(filtered_kps, valid_ranges)

        # Center of mass analysis (Hip center to virtual floor - lowest point)
        com_analysis = self._calc_com_analysis(filtered_kps, leg_length_px, valid_ranges)

        # Aggregated Cycles (Normalized to 0-100% of gait cycle)
        cycles_stats = self._aggregate_cycles(kinematics_raw, l_phases, r_phases, valid_ranges)

        # Detailed Statistics (Min/Max/Mean for every single detected step)
        detailed_stats = self._calc_detailed_stats(kinematics_raw, l_phases, r_phases, valid_ranges, framerate)

        # Symmetry Analysis (Left vs Right comparison)
        symmetry_stats = self._calc_symmetry_metrics(spatiotemporal, cycles_stats)

        # --- EXPORT ---
        phases_export = {
            "left": [self._serialize_phase(p) for p in l_phases],
            "right": [self._serialize_phase(p) for p in r_phases],
            "support": [self._serialize_support(p) for p in support_phases]
        }

        report = {
            "metadata": {
                "file": os.path.splitext(os.path.basename(filename))[0],
                "framerate": framerate,
                "leg_length_px": leg_length_px,
                "valid_ranges": valid_ranges
            },
            "spatiotemporal": spatiotemporal,
            "spatial": spatial_params,
            "gait_phases": phases_export,
            "kinematics_stats": cycles_stats,
            "detailed_statistics": detailed_stats,
            "symmetry_statistics": symmetry_stats,
            "raw_kinematics": kinematics_raw,
            "com_analysis": com_analysis
        }

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "analysis.json")

        with open(out_path, 'w') as f:
            json.dump(report, f, indent=4)

        log("ANALYZER", f"Report saved to {out_path}", level="success")
        return report

    # -------------------------------------------------------------------------
    # SPATIO-TEMPORAL LOGIC
    # -------------------------------------------------------------------------
    @staticmethod
    def _calc_spatiotemporal(l_phases, r_phases, support_phases, valid_ranges, fps):
        """
        Calculates temporal gait parameters including cadence, stride time, and support ratios.

        Args:
            l_phases (list): List of left leg GaitPhase objects.
            r_phases (list): List of right leg GaitPhase objects.
            support_phases (list): List of SupportPhase objects.
            valid_ranges (list): List of valid frame ranges.
            fps (float): Frames per second.

        Returns:
            dict: Dictionary containing temporal metrics for left/right legs and global stats.
        """
        stats = {}

        # Collector for all valid stride durations (from both legs)
        all_stride_durations = []

        def process_side(phases):
            stance_durs = []
            swing_durs = []
            stride_durs = []  # Local for average calculation

            # Sort phases by time
            sorted_phases = sorted(phases, key=lambda x: x.start_frame)

            i = 0
            while i < len(sorted_phases):
                p = sorted_phases[i]

                # Check validity against ranges
                mid_frame = (p.start_frame + p.end_frame) / 2
                is_valid = True
                if valid_ranges:
                    is_valid = False
                    for v_start, v_end in valid_ranges:
                        if v_start <= mid_frame <= v_end:
                            is_valid = True
                            break

                if not is_valid:
                    i += 1
                    continue

                dur_s = p.duration / fps
                ptype = str(p.phase_type).upper()

                if "STANCE" in ptype:
                    stance_durs.append(dur_s)

                    # Try to find the following Swing to complete a STRIDE
                    if i + 1 < len(sorted_phases):
                        p_next = sorted_phases[i + 1]
                        # Check continuity (max 2 frames gap)
                        if "SWING" in str(p_next.phase_type).upper() and (p_next.start_frame - p.end_frame) <= 2:
                            stride_s = dur_s + (p_next.duration / fps)
                            stride_durs.append(stride_s)
                            all_stride_durations.append(stride_s)  # Add to global collector

                elif "SWING" in ptype:
                    swing_durs.append(dur_s)

                i += 1

            avg_stance = float(np.mean(stance_durs)) if stance_durs else 0.0
            avg_swing = float(np.mean(swing_durs)) if swing_durs else 0.0

            # Prefer measured strides, fallback to sum of averages
            if stride_durs:
                avg_stride = float(np.mean(stride_durs))
            else:
                avg_stride = avg_stance + avg_swing

            return {
                "stride_time_avg": avg_stride,
                "stance_time_avg": avg_stance,
                "swing_time_avg": avg_swing,
                "step_count": len(stance_durs)
            }

        stats['left'] = process_side(l_phases)
        stats['right'] = process_side(r_phases)

        # --- GLOBAL CADENCE CALCULATION ---
        # Cadence = 120 / Avg_Stride_Time
        cadence = 0.0
        if all_stride_durations:
            global_avg_stride = np.mean(all_stride_durations)
            if global_avg_stride > 0:
                cadence = 120.0 / float(global_avg_stride)

        # Total steps (just for info)
        total_steps = stats['left']['step_count'] + stats['right']['step_count']

        stats['global'] = {
            "cadence": cadence,
            "total_steps": total_steps
        }

        # --- SUPPORT RATIO ---
        total_single, total_double = 0, 0
        for p in support_phases:
            # Check validity center
            mid = (p.start_frame + p.end_frame) / 2
            in_range = True
            if valid_ranges:
                in_range = any(s <= mid <= e for s, e in valid_ranges)

            if in_range:
                dur = p.duration / fps
                pt = str(p.support_type).lower()
                if "double" in pt:
                    total_double += dur
                elif any(x in pt for x in ["single", "left", "right"]):
                    total_single += dur

        stats['support_ratio'] = {"single_total_s": total_single, "double_total_s": total_double}
        return stats

    # -------------------------------------------------------------------------
    # SPATIAL PARAMETERS
    # -------------------------------------------------------------------------
    @staticmethod
    def _calc_spatial_parameters(kps, l_phases, r_phases, leg_len_px, spatiotemporal):
        """
        Calculates spatial metrics normalized by leg length (%LL).
        - Step Length: Horizontal distance between ankles at Heel Strike.
        - Stride Length: Sum of Avg Step L + Avg Step R (accounts for asymmetry).
        - Velocity: Calculated as Stride Length / Stride Time.

        Args:
            kps (dict): Keypoints dictionary.
            l_phases (list): List of left leg GaitPhase objects.
            r_phases (list): List of right leg GaitPhase objects.
            leg_len_px (float): Leg length in pixels.
            spatiotemporal (dict): Result from _calc_spatiotemporal.

        Returns:
            dict: Dictionary containing spatial metrics.
        """
        if leg_len_px <= 0: return {}

        l_ankle = kps.get('LEFT_ANKLE', [])
        r_ankle = kps.get('RIGHT_ANKLE', [])

        # Helper to calculate step length for a list of phases
        def calc_step_lengths(phases, primary_ankle, secondary_ankle):
            steps = []
            for p in phases:
                if p.phase_type == PhaseType.STANCE:
                    # Heel Strike is the start frame of Stance
                    hs_frame = int(p.start_frame)

                    if hs_frame < len(primary_ankle) and hs_frame < len(secondary_ankle):
                        p1 = primary_ankle[hs_frame]
                        p2 = secondary_ankle[hs_frame]

                        if p1 is not None and p2 is not None and p1[0] != 0 and p2[0] != 0:
                            # Horizontal distance (X-axis difference)
                            dist_px = abs(p1[0] - p2[0])
                            steps.append((dist_px / leg_len_px) * 100.0)
            return steps

        # Distances (Step Lengths)
        l_step_vals = calc_step_lengths(l_phases, l_ankle, r_ankle)
        r_step_vals = calc_step_lengths(r_phases, r_ankle, l_ankle)

        avg_step_l = float(np.mean(l_step_vals)) if l_step_vals else 0.0
        avg_step_r = float(np.mean(r_step_vals)) if r_step_vals else 0.0

        # Stride Length = Step L + Step R
        avg_stride_len = avg_step_l + avg_step_r

        # Velocity (Distance / Time)
        st_l = spatiotemporal.get('left', {}).get('stride_time_avg', 0)
        st_r = spatiotemporal.get('right', {}).get('stride_time_avg', 0)

        # Calculate real average stride time
        times = [t for t in [st_l, st_r] if t > 0]
        avg_stride_time = np.mean(times) if times else 0.0

        # Velocity = Stride Length (%LL) / Stride Time (s)
        velocity = 0.0
        if avg_stride_time > 0:
            velocity = avg_stride_len / avg_stride_time

        return {
            "left": {
                "step_length_avg_percent": avg_step_l,
                "stride_length_avg_percent": avg_stride_len,
            },
            "right": {
                "step_length_avg_percent": avg_step_r,
                "stride_length_avg_percent": avg_stride_len,
            },
            "global": {
                "step_length_avg_percent": (avg_step_l + avg_step_r) / 2.0,
                "stride_length_avg_percent": avg_stride_len,
                "velocity_percent_per_sec": velocity
            }
        }

    # -------------------------------------------------------------------------
    # SYMMETRY LOGIC
    # -------------------------------------------------------------------------
    @staticmethod
    def _calc_symmetry_metrics(spatio, cycles):
        """
        Calculates Symmetry Index (SI) as a percentage.
        Formula: SI = |L - R| / (0.5 * (L + R)) * 100
        0% indicates perfect symmetry.

        Args:
            spatio (dict): Spatiotemporal metrics dictionary.
            cycles (dict): Aggregated gait cycles dictionary.

        Returns:
            dict: Dictionary containing symmetry indices.
        """
        sym_stats = {}
        l_stats = spatio.get('left', {})
        r_stats = spatio.get('right', {})

        # Temporal Symmetry (Time based)
        for metric in ['stride_time_avg', 'stance_time_avg', 'swing_time_avg']:
            l_val = l_stats.get(metric, 0)
            r_val = r_stats.get(metric, 0)

            if l_val > 0 and r_val > 0:
                avg_val = 0.5 * (l_val + r_val)
                si = (abs(l_val - r_val) / avg_val) * 100
                sym_stats[metric] = round(si, 2)
            else:
                sym_stats[metric] = None

        # Kinematic Symmetry
        rom_symmetry = {}
        joints_def = [
            ("Hip", "SHOULDER-HIP-KNEE"),
            ("Knee", "HIP-KNEE-ANKLE"),
            ("Ankle", "KNEE-ANKLE-FOOT")
        ]

        for joint_name, keyword in joints_def:
            # Find matching keys in the cycles dictionary
            l_key = next((k for k in cycles if keyword in k and "LEFT" in k), None)
            r_key = next((k for k in cycles if keyword in k and "RIGHT" in k), None)

            if l_key and r_key:
                l_curve = np.array(cycles[l_key]['mean'])
                r_curve = np.array(cycles[r_key]['mean'])

                # Calculate Range of Motion (Max - Min)
                l_rom = np.max(l_curve) - np.min(l_curve)
                r_rom = np.max(r_curve) - np.min(r_curve)

                if l_rom > 0 and r_rom > 0:
                    avg_rom = 0.5 * (l_rom + r_rom)
                    si = (abs(l_rom - r_rom) / avg_rom) * 100
                    rom_symmetry[joint_name] = round(si, 2)

        sym_stats['kinematics_rom'] = rom_symmetry
        return sym_stats

    # -------------------------------------------------------------------------
    # STATISTICS LOGIC
    # -------------------------------------------------------------------------
    @staticmethod
    def _calc_detailed_stats(kinematics, l_phases, r_phases, valid_ranges, fps):
        """
        Extracts Mean, Min, Max, and Range for each individual phase.
        Ensures consistency with graphs by filtering valid ranges and checking hip polarity.

        Args:
            kinematics (dict): Dictionary of kinematic signals (angles).
            l_phases (list): List of left leg GaitPhase objects.
            r_phases (list): List of right leg GaitPhase objects.
            valid_ranges (list): List of valid frame ranges.
            fps (float): Frames per second.

        Returns:
            dict: Nested dictionary containing detailed stats per joint and side.
        """
        stats_out = {"left": {}, "right": {}}
        joint_map = {"Hip": ["SHOULDER", "HIP", "KNEE"], "Knee": ["HIP", "KNEE", "ANKLE"],
                     "Ankle": ["KNEE", "ANKLE", "FOOT"]}

        def process_side(side_name, phases):
            side_stats = {}
            for joint_simple, keywords in joint_map.items():
                data_key = next((k for k in kinematics if all(kw in k for kw in keywords) and side_name.upper() in k),
                                None)
                if not data_key: continue

                signal = np.array(kinematics[data_key])
                phase_details = []
                aggregators = {"Stance": {"dur": [], "mean": [], "min": [], "max": [], "range": []},
                               "Swing": {"dur": [], "mean": [], "min": [], "max": [], "range": []}}

                for i, p in enumerate(phases):
                    start, end = int(p.start_frame), int(p.end_frame)

                    # --- VALID RANGE CHECK ---
                    if valid_ranges:
                        is_in_range = False
                        for v_start, v_end in valid_ranges:
                            if start >= v_start and end <= v_end:
                                is_in_range = True
                                break
                        if not is_in_range:
                            continue  # Skip this phase in stats

                    if start >= len(signal): continue
                    seg = signal[start: end + 1]
                    seg = seg[seg is not None]
                    if len(seg) == 0: continue

                    # --- HIP POLARITY CHECK ---
                    # STANCE: Start (HS) must be positive.
                    if "HIP" in joint_simple.upper() and p.phase_type == PhaseType.STANCE:
                        if np.mean(seg[:min(5, len(seg))]) < 0:
                            seg = -seg  # Flip to match the graph logic

                    s_mean, s_min, s_max = np.mean(seg), np.min(seg), np.max(seg)
                    dur = (end - start) / fps
                    ptype = "Stance" if p.phase_type == PhaseType.STANCE else "Swing"

                    phase_details.append({
                        "id": i + 1, "type": ptype, "frames": [start, end],
                        "duration": dur, "mean": s_mean, "min": s_min, "max": s_max, "range": s_max - s_min
                    })

                    if ptype in aggregators:
                        for k, v in zip(["dur", "mean", "min", "max", "range"],
                                        [dur, s_mean, s_min, s_max, s_max - s_min]): aggregators[ptype][k].append(v)

                summary = {}
                for ptype, d in aggregators.items():
                    if d["dur"]: summary[ptype] = {"avg_duration": float(np.mean(d["dur"])),
                                                   "avg_rom": float(np.mean(d["range"])),
                                                   "avg_mean": float(np.mean(d["mean"])),
                                                   "abs_min": float(np.min(d["min"])),
                                                   "abs_max": float(np.max(d["max"]))}
                side_stats[joint_simple] = {"phases": phase_details, "summary": summary}
            return side_stats

        stats_out["left"] = process_side("left", l_phases)
        stats_out["right"] = process_side("right", r_phases)
        return stats_out

    # -------------------------------------------------------------------------
    # KINEMATICS LOGIC
    # -------------------------------------------------------------------------
    def _calc_kinematics_signed(self, kps, valid_ranges):
        """
        Calculates joint angles with direction correction.
        Includes a robustness check: if Hip mean is negative, flip the whole segment.

        Args:
            kps (dict): Keypoints dictionary.
            valid_ranges (list): List of valid frame ranges.

        Returns:
            dict: Dictionary of calculated angles (list of floats per frame).
        """
        # 1. Initial Direction Detection
        any_key = next(iter(kps))
        total_frames = len(kps[any_key])
        directions = np.ones(total_frames)

        # Detect direction per segment (valid range)
        if valid_ranges:
            for start, end in valid_ranges:
                dir_mult = self._detect_walking_direction_segment(kps, start, end)
                directions[start:end + 1] = dir_mult
        else:
            dir_mult = self._detect_walking_direction_segment(kps, 0, total_frames - 1)
            directions[:] = dir_mult

        angles_data = {}
        definitions = [
            ("angle_LEFT_SHOULDER-LEFT_HIP-LEFT_KNEE", "LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE", "hip"),
            ("angle_RIGHT_SHOULDER-RIGHT_HIP-RIGHT_KNEE", "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE", "hip"),
            ("angle_LEFT_HIP-LEFT_KNEE-LEFT_ANKLE", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "knee"),
            ("angle_RIGHT_HIP-RIGHT_KNEE-RIGHT_ANKLE", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "knee"),
            ("angle_LEFT_KNEE-LEFT_ANKLE-LEFT_FOOT_INDEX", "LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX", "ankle"),
            ("angle_RIGHT_KNEE-RIGHT_ANKLE-RIGHT_FOOT_INDEX", "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX", "ankle")
        ]

        # Calculate Angles
        for output_name, p1_name, p2_name, p3_name, joint_type in definitions:
            if any(k not in kps for k in [p1_name, p2_name, p3_name]):
                continue

            values_list = []
            for i in range(total_frames):
                p1, p2, p3 = kps[p1_name][i], kps[p2_name][i], kps[p3_name][i]

                if p1 is None or p2 is None or p3 is None:
                    values_list.append(None)
                    continue

                v1, v2, v3 = np.array(p1[:2]), np.array(p2[:2]), np.array(p3[:2])

                # Apply Direction Flip
                if directions[i] < 0:
                    v1[0] *= -1
                    v2[0] *= -1
                    v3[0] *= -1

                angle = self._calculate_2d_angle(v1, v2, v3, joint_type)
                values_list.append(angle)

            # Force Polarity
            # If the calculated hip angle is mostly negative, it means direction detection failed.
            if joint_type == "hip":
                ranges_to_check = valid_ranges if valid_ranges else [(0, total_frames - 1)]

                for start, end in ranges_to_check:
                    # Extract segment indices that have data
                    segment_indices = [i for i in range(start, min(end + 1, len(values_list))) if
                                       values_list[i] is not None]

                    if not segment_indices:
                        continue

                    # Calculate mean of the segment
                    segment_vals = [values_list[i] for i in segment_indices]
                    mean_val = np.mean(segment_vals)

                    # If Hip angle is negative on average, flip the whole segment
                    if mean_val < 0:
                        for i in segment_indices:
                            values_list[i] *= -1.0

            angles_data[output_name] = values_list

        return angles_data

    # -------------------------------------------------------------------------
    # CoM LOGIC
    # -------------------------------------------------------------------------
    @staticmethod
    def _calc_com_analysis(kps, leg_len_px, valid_ranges):
        """
        Calculates Center of Mass (CoM) vertical excursion statistics.

        The CoM is approximated as the midpoint between the hips. Its vertical
        distance from the lowest foot point (virtual floor) is calculated and
        normalized as a percentage of leg length.

        Args:
            kps (dict): Keypoints dictionary.
            leg_len_px (float): Estimated leg length in pixels for normalization.
            valid_ranges (list): List of valid frame ranges to analyze.

        Returns:
            dict: Dictionary containing 'statistics' (mean, min, max, range, std)
                  and 'raw_signal' (list of normalized heights per frame).
        """

        if leg_len_px <= 0: return {}

        l_hip = kps.get('LEFT_HIP', [])
        r_hip = kps.get('RIGHT_HIP', [])
        feet_keys = ['LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
        feet_data = [kps.get(k, []) for k in feet_keys]

        total_frames = len(l_hip)
        raw_signal = []

        # Generate Raw Signal (Normalized to % Leg Length)
        for i in range(total_frames):
            lh, rh = l_hip[i], r_hip[i]

            if lh is None or rh is None:
                raw_signal.append(None)
                continue

            com_y = (lh[1] + rh[1]) / 2.0

            frame_ys = []
            for fd in feet_data:
                if i < len(fd) and fd[i] is not None and fd[i][1] != 0:
                    frame_ys.append(fd[i][1])

            if not frame_ys:
                raw_signal.append(None)
                continue

            # Distance from CoM to the lowest foot point
            height_px = max(frame_ys) - com_y
            height_norm = (height_px / leg_len_px) * 100.0
            raw_signal.append(height_norm)

        # Compute statistics for valid ranges
        valid_values = []
        ranges_to_check = valid_ranges if valid_ranges else [(0, total_frames - 1)]

        for start, end in ranges_to_check:
            segment = raw_signal[start: min(end + 1, len(raw_signal))]
            valid_values.extend([v for v in segment if v is not None])

        stats = {}
        if valid_values:
            v_arr = np.array(valid_values)
            stats = {
                "mean_height_percent": float(np.mean(v_arr)),
                "min_height_percent": float(np.min(v_arr)),
                "max_height_percent": float(np.max(v_arr)),
                "vertical_excursion_range": float(np.max(v_arr) - np.min(v_arr)),
                "std_dev": float(np.std(v_arr))
            }

        return {
            "statistics": stats,
            "raw_signal": raw_signal
        }

    # --- HELPERS ---
    @staticmethod
    def _calculate_2d_angle(p1, p2, p3, joint_type):
        """
        Calculates the 2D angle between three points (p1-p2-p3).
        Applies clinical conventions based on joint type (e.g., knee flexion is positive).
        """
        vec1 = p2 - p1
        vec2 = p3 - p2

        dot = np.dot(vec1, vec2)
        det = vec1[0] * vec2[1] - vec1[1] * vec2[0]

        deg = np.degrees(np.arctan2(det, dot))

        if joint_type == "knee":
            return abs(deg)  # Knee flexion is always positive

        elif joint_type == "hip":
            return -deg  # Invert sign for correct Flexion(+) / Extension(-)

        elif joint_type == "ankle":
            return abs(deg) # Could also be calculated as angle - 90.0

        return 0.0

    @staticmethod
    def _detect_walking_direction_segment(kps, start_frame, end_frame):
        """
        Determines the walking direction for a specific time segment.

        Calculates the slope of the Left Hip's X-coordinate over time.

        Returns:
            float: 1.0 for Left->Right (increasing X), -1.0 for Right->Left (decreasing X).
        """
        l_hip = kps.get("LEFT_HIP", [])

        x_coords = []
        valid_indices = []

        limit = min(end_frame + 1, len(l_hip))
        for i in range(start_frame, limit):
            if l_hip[i] is not None and l_hip[i][0] != 0:
                x_coords.append(l_hip[i][0])
                valid_indices.append(i)

        if len(x_coords) < 5:
            return 1.0  # Default L->R

        slope, _ = np.polyfit(valid_indices, x_coords, 1)
        return 1.0 if slope >= 0 else -1.0

    def _parse_serializer_data(self, json_data):
        """
        Converts the raw JSON list format into a dictionary of keypoint lists.
        """
        max_frame = 0
        parsed_entries = []

        # Find max frame and valid entries
        for entry in json_data:
            try:
                fid = int(entry['image_id'].split('.')[0])
                max_frame = max(max_frame, fid)
                parsed_entries.append((fid, entry['keypoints']))
            except (ValueError, IndexError):
                continue

        # Initialize dictionary of None lists
        kps_dict = {name: [None] * (max_frame + 1) for name in self.skeleton.keypoints.__members__}

        # Fill data
        for fid, raw_kps in parsed_entries:
            for name, member in self.skeleton.keypoints.__members__.items():
                idx = int(member) * 3
                # Check bounds and validity (x!=0)
                if idx + 2 < len(raw_kps) and raw_kps[idx] != 0:
                    kps_dict[name][fid] = (raw_kps[idx], raw_kps[idx + 1], raw_kps[idx + 2])

        return kps_dict

    @staticmethod
    def _estimate_leg_length(kps, valid_ranges):
        """
        Estimates the leg length (Hip to Ankle) in pixels.
        Used for potential normalization of spatial metrics.
        """
        all_lengths = []

        # If no valid ranges are defined, scan the whole video
        if not valid_ranges:
            # Find the total length from one of the keys
            any_key = next(iter(kps))
            ranges_to_scan = [(0, len(kps[any_key]) - 1)]
        else:
            ranges_to_scan = valid_ranges

        for side in ['RIGHT', 'LEFT']:
            # Get all keypoints
            hips = kps.get(f'{side}_HIP', [])
            knees = kps.get(f'{side}_KNEE', [])
            ankles = kps.get(f'{side}_ANKLE', [])

            # Iterate only through valid segments
            for start_frame, end_frame in ranges_to_scan:
                # Ensure we don't go out of bounds
                limit = min(len(hips), end_frame + 1)

                for i in range(start_frame, limit):
                    h, k, a = hips[i], knees[i], ankles[i]

                    if h is None or k is None or a is None:
                        continue

                    if h[0] == 0 or k[0] == 0 or a[0] == 0:
                        continue

                    # Convert to numpy
                    p_hip = np.array(h[:2])
                    p_knee = np.array(k[:2])
                    p_ankle = np.array(a[:2])

                    # Calculate Segment Lengths
                    upper = np.linalg.norm(p_hip - p_knee)
                    lower = np.linalg.norm(p_knee - p_ankle)

                    total_len = upper + lower

                    # Ignore short segments
                    if total_len > 50:
                        all_lengths.append(total_len)

        # Fallback to avoid division by zero
        if not all_lengths:
            return 1.0

        # Use Median to filter out outliers
        return float(np.median(all_lengths))

    @staticmethod
    def _aggregate_cycles(kinematics, l_phases, r_phases, valid_ranges):
        """
        Aggregates kinematic signals into normalized gait cycles (0-100%).

        Args:
            kinematics (dict): Dictionary of kinematic signals.
            l_phases (list): List of left leg GaitPhase objects.
            r_phases (list): List of right leg GaitPhase objects.
            valid_ranges (list): List of valid frame ranges.

        Returns:
            dict: Dictionary containing mean/std cycles for each joint.
        """
        stats = {}

        # Helper to extract cycles
        def extract_cycles_from_phases(phases, signal):
            cycles = []
            if not phases or len(phases) < 2: return cycles

            # Sort phases by time to be sure
            sorted_phases = sorted(phases, key=lambda x: x.start_frame)

            i = 0
            while i < len(sorted_phases) - 1:
                p1 = sorted_phases[i]
                p2 = sorted_phases[i + 1]

                # Check for Stance -> Swing pattern
                is_stance_swing = (p1.phase_type == PhaseType.STANCE and p2.phase_type == PhaseType.SWING)

                # Check continuity -> 1 frame tolerance
                is_continuous = (p2.start_frame - p1.end_frame) <= 1

                if is_stance_swing and is_continuous:
                    start = int(p1.start_frame)
                    end = int(p2.end_frame)

                    in_valid_range = False
                    if not valid_ranges:
                        in_valid_range = True
                    else:
                        for v_start, v_end in valid_ranges:
                            if start >= v_start and end <= v_end:
                                in_valid_range = True
                                break

                    if in_valid_range:
                        # Extract signal segment
                        raw_cycle = signal[start: end + 1]
                        raw_cycle = [r for r in raw_cycle if r is not None]

                        if len(raw_cycle) > 5:
                            # Normalize to 100 points
                            x_orig = np.linspace(0, 1, len(raw_cycle))
                            interp = interp1d(x_orig, raw_cycle, kind='linear')
                            normalized = interp(np.linspace(0, 1, 100))
                            cycles.append(normalized)

                    # Skip p2 because it was used as the second half of this cycle
                    i += 1

                i += 1
            return cycles

        # Process each kinematic signal
        for key, signal in kinematics.items():
            # Determine side from key name
            side = "left" if "LEFT" in key else "right" if "RIGHT" in key else None
            if not side: continue

            phases_to_use = l_phases if side == "left" else r_phases

            # Extract
            raw_cycles_list = extract_cycles_from_phases(phases_to_use, signal)

            processed_cycles = []
            for cyc in raw_cycles_list:
                # HS -> must be positive
                # If negative, it means direction detection was wrong for this segment -> Flip.
                if "HIP" in key:
                    start_val = np.mean(cyc[:10])  # Check start of cycle
                    if start_val < 0:
                        cyc *= -1.0

                processed_cycles.append(cyc)

            # Compute Mean/Std if we have data
            if processed_cycles:
                stack = np.vstack(processed_cycles)
                stats[key] = {
                    "mean": np.mean(stack, axis=0).tolist(),
                    "std": np.std(stack, axis=0).tolist(),
                    "n_cycles": len(processed_cycles)
                }

        return stats

    @staticmethod
    def _serialize_phase(p):
        """Serializes a GaitPhase object to a dictionary."""
        return {
            "type": str(p.phase_type),
            "start": int(p.start_frame),
            "end": int(p.end_frame),
            "duration": int(p.duration)
        }

    @staticmethod
    def _serialize_support(p):
        """Serializes a SupportPhase object to a dictionary."""
        return {
            "type": str(p.support_type),
            "start": int(p.start_frame),
            "end": int(p.end_frame),
            "duration": int(p.duration)
        }
