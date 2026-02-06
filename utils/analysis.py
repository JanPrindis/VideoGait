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

from utils.json_serializer import KeypointSerializer
from utils.preprocessing import preprocess_keypoints
from utils.gait_structs import GaitEventType, PhaseType
from utils.config_utils import resolve_skeleton_from_config


class GaitAnalyzer:
    """
    Main class for performing gait analysis calculations.
    """
    def __init__(self, app_config):
        """
        Initializes the analyzer with application configuration.

        Args:
            app_config (dict): The application configuration dictionary containing
                               preprocessing settings (filter parameters).
        """
        self.cfg = app_config
        self.pre_cfg = app_config.get('preprocessing', {})
        self.skeleton = resolve_skeleton_from_config(app_config)

        # Signal processing parameters
        self.filter_cutoff = self.pre_cfg.get('filter_cutoff', 6)
        self.filter_order = self.pre_cfg.get('filter_order', 4)

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
        print(f"[Analyzer] Analyzing {filename}...")

        if not os.path.exists(keypoints_path):
            print(f"[Analyzer] Error: File {keypoints_path} not found.")
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
        spatiotemporal = self._calc_spatiotemporal(events_data, support_phases, framerate)

        # Kinematics (Angles with Clinical Offsets and Direction Correction)
        kinematics_raw = self._calc_kinematics_signed(filtered_kps)

        # Aggregated Cycles (Normalized to 0-100% of gait cycle)
        cycles_stats = self._aggregate_cycles(kinematics_raw, events_data, framerate)

        # Detailed Statistics (Min/Max/Mean for every single detected step)
        detailed_stats = self._calc_detailed_stats(kinematics_raw, l_phases, r_phases, framerate)

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
                "file": filename,
                "framerate": framerate,
                "leg_length_px": leg_length_px,
                "valid_ranges": valid_ranges
            },
            "spatiotemporal": spatiotemporal,
            "gait_phases": phases_export,
            "kinematics_stats": cycles_stats,
            "detailed_statistics": detailed_stats,
            "symmetry_statistics": symmetry_stats,
            "raw_kinematics": kinematics_raw
        }

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "analysis.json")

        with open(out_path, 'w') as f:
            json.dump(report, f, indent=4)

        print(f"[Analyzer] Report saved to {out_path}")
        return report

    # -------------------------------------------------------------------------
    # SPATIO-TEMPORAL LOGIC
    # -------------------------------------------------------------------------
    def _calc_spatiotemporal(self, events_data, support_phases, fps):
        """
        Calculates basic temporal metrics: Cadence, Stride/Stance/Swing durations.
        """
        stats = {}
        src = self._extract_events_dict(events_data)
        all_frames = []

        # Process Left and Right legs individually
        for side in ['left', 'right']:
            raw_events = src.get(side, [])

            # Filter and sort events
            hs_frames = sorted([e.frame for e in raw_events
                                if hasattr(e, 'event_type') and e.event_type == GaitEventType.HEEL_STRIKE])
            to_frames = sorted([e.frame for e in raw_events
                                if hasattr(e, 'event_type') and e.event_type == GaitEventType.TOE_OFF])

            if hs_frames:
                all_frames.extend(hs_frames)

            # Calculate Stride Time (Heel Strike to next Heel Strike)
            stride_times = []
            for i in range(len(hs_frames) - 1):
                duration = (hs_frames[i + 1] - hs_frames[i]) / fps
                stride_times.append(duration)

            # Calculate Stance and Swing Times
            stance_times = []
            swing_times = []

            for hs in hs_frames:
                # Find the immediate next Toe Off
                candidates_to = [t for t in to_frames if t > hs]

                if candidates_to:
                    next_to = min(candidates_to)
                    stance_duration = (next_to - hs) / fps
                    stance_times.append(stance_duration)

                    # Find the immediate next Heel Strike after that Toe Off
                    candidates_hs = [h for h in hs_frames if h > next_to]
                    if candidates_hs:
                        swing_duration = (min(candidates_hs) - next_to) / fps
                        swing_times.append(swing_duration)

            stats[side] = {
                "stride_time_avg": np.mean(stride_times) if stride_times else 0,
                "stance_time_avg": np.mean(stance_times) if stance_times else 0,
                "swing_time_avg": np.mean(swing_times) if swing_times else 0,
                "step_count": len(hs_frames)
            }

        # Calculate Global Cadence (Steps per Minute)
        cadence = 0
        total_steps = stats['left']['step_count'] + stats['right']['step_count']

        if all_frames:
            total_duration_sec = (max(all_frames) - min(all_frames)) / fps
            if total_duration_sec > 1:
                cadence = (total_steps / total_duration_sec) * 60

        stats['global'] = {"cadence": cadence, "total_steps": total_steps}

        # Calculate Support Ratio (Time spent in Single vs Double support)
        total_single = 0
        total_double = 0

        for p in support_phases:
            duration = p.duration / fps
            ptype_str = str(p.support_type).lower()

            if "double" in ptype_str:
                total_double += duration
            elif any(x in ptype_str for x in ["single", "left", "right"]):
                total_single += duration

        stats['support_ratio'] = {
            "single_total_s": total_single,
            "double_total_s": total_double
        }

        return stats

    # -------------------------------------------------------------------------
    # SYMMETRY LOGIC
    # -------------------------------------------------------------------------
    @staticmethod
    def _calc_symmetry_metrics(spatio, cycles):
        """
        Calculates Symmetry Index (SI) as a percentage.
        Formula: SI = |L - R| / (0.5 * (L + R)) * 100
        0% indicates perfect symmetry.
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
    def _calc_detailed_stats(kinematics, l_phases, r_phases, fps):
        """
        Extracts Mean, Min, Max, and Range for EACH individual phase (step).
        This data is used for detailed tables in the report.
        """
        stats_out = {"left": {}, "right": {}}

        joint_map = {
            "Hip": ["SHOULDER", "HIP", "KNEE"],
            "Knee": ["HIP", "KNEE", "ANKLE"],
            "Ankle": ["KNEE", "ANKLE", "FOOT"]
        }

        def process_side(side_name, phases):
            side_stats = {}

            for joint_simple_name, keywords in joint_map.items():
                # Find the full data key (e.g., 'angle_LEFT_KNEE...')
                data_key = next((k for k in kinematics if all(kw in k for kw in keywords) and side_name.upper() in k),
                                None)
                if not data_key:
                    continue

                full_signal = np.array(kinematics[data_key])
                phase_details_list = []

                # Aggregators for global summary (Stance vs Swing)
                aggregators = {
                    "Stance": {"dur": [], "mean": [], "min": [], "max": [], "range": []},
                    "Swing": {"dur": [], "mean": [], "min": [], "max": [], "range": []}
                }

                # Iterate through every single detected phase
                for i, p in enumerate(phases):
                    start_frame, end_frame = int(p.start_frame), int(p.end_frame)

                    if start_frame >= len(full_signal):
                        continue

                    # Extract the signal segment for this specific step
                    segment = full_signal[start_frame: end_frame + 1]
                    segment = segment[segment is not None]  # Filter out missing data

                    if len(segment) == 0:
                        continue

                    # Compute statistics
                    s_mean = np.mean(segment)
                    s_min = np.min(segment)
                    s_max = np.max(segment)
                    s_range = s_max - s_min
                    duration_s = (end_frame - start_frame) / fps

                    phase_type_str = "Stance" if p.phase_type == PhaseType.STANCE else "Swing"

                    # Store detailed info
                    phase_details_list.append({
                        "id": i + 1,
                        "type": phase_type_str,
                        "frames": [start_frame, end_frame],
                        "duration": duration_s,
                        "mean": s_mean,
                        "min": s_min,
                        "max": s_max,
                        "range": s_range
                    })

                    # Add to aggregators
                    if phase_type_str in aggregators:
                        aggregators[phase_type_str]["dur"].append(duration_s)
                        aggregators[phase_type_str]["mean"].append(s_mean)
                        aggregators[phase_type_str]["min"].append(s_min)
                        aggregators[phase_type_str]["max"].append(s_max)
                        aggregators[phase_type_str]["range"].append(s_range)

                # Calculate Summary
                summary = {}
                for ptype, vals in aggregators.items():
                    if vals["dur"]:
                        summary[ptype] = {
                            "avg_duration": float(np.mean(vals["dur"])),
                            "avg_rom": float(np.mean(vals["range"])),
                            "avg_mean": float(np.mean(vals["mean"])),
                            "abs_min": float(np.min(vals["min"])),
                            "abs_max": float(np.max(vals["max"]))
                        }

                side_stats[joint_simple_name] = {
                    "phases": phase_details_list,
                    "summary": summary
                }

            return side_stats

        stats_out["left"] = process_side("left", l_phases)
        stats_out["right"] = process_side("right", r_phases)
        return stats_out

    # -------------------------------------------------------------------------
    # KINEMATICS LOGIC
    # -------------------------------------------------------------------------
    def _calc_kinematics_signed(self, kps):
        direction = self._detect_walking_direction(kps)
        angles_data = {}

        # Definitions: (OutputKey, Point1, Point2, Point3, JointType)
        definitions = [
            ("angle_LEFT_SHOULDER-LEFT_HIP-LEFT_KNEE", "LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE", "hip"),
            ("angle_RIGHT_SHOULDER-RIGHT_HIP-RIGHT_KNEE", "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE", "hip"),
            ("angle_LEFT_HIP-LEFT_KNEE-LEFT_ANKLE", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "knee"),
            ("angle_RIGHT_HIP-RIGHT_KNEE-RIGHT_ANKLE", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "knee"),
            ("angle_LEFT_KNEE-LEFT_ANKLE-LEFT_FOOT_INDEX", "LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX", "ankle"),
            ("angle_RIGHT_KNEE-RIGHT_ANKLE-RIGHT_FOOT_INDEX", "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX", "ankle")
        ]

        # Get total frames from any keypoint list
        any_key = next(iter(kps))
        total_frames = len(kps[any_key])

        for output_name, p1_name, p2_name, p3_name, joint_type in definitions:

            # Skip if skeleton doesn't have required joints
            if any(k not in kps for k in [p1_name, p2_name, p3_name]):
                continue

            values_list = []
            for i in range(total_frames):
                p1 = kps[p1_name][i]
                p2 = kps[p2_name][i]
                p3 = kps[p3_name][i]

                if p1 is None or p2 is None or p3 is None:
                    values_list.append(None)
                    continue

                # Convert to numpy (ignore confidence)
                v1 = np.array(p1[:2])
                v2 = np.array(p2[:2])
                v3 = np.array(p3[:2])

                # Flip X coordinates if walking Right -> Left
                if direction < 0:
                    v1[0] *= -1
                    v2[0] *= -1
                    v3[0] *= -1

                angle = self._calculate_2d_angle(v1, v2, v3, joint_type)
                values_list.append(angle)

            angles_data[output_name] = values_list

        return angles_data

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
    def _detect_walking_direction(kps):
        """
        Detects the walking direction based on the slope of the hip's X-coordinate over time.
        Returns 1.0 for Left->Right, -1.0 for Right->Left.
        """
        l_hip = kps.get("LEFT_HIP", [])

        # Extract X coordinates where present
        x_coords = [p[0] for p in l_hip if p is not None]

        if len(x_coords) < 10:
            return 1.0  # Default

        # Simple linear fit to determine direction
        slope, _ = np.polyfit(np.arange(len(x_coords)), x_coords, 1)

        return 1.0 if slope >= 0 else -1.0

    # --- HELPERS ---
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
    def _extract_events_dict(events_data):
        """Helper to safely extract the inner event dictionary."""
        if isinstance(events_data, dict):
            return events_data.get('events', events_data)
        return {}

    def _aggregate_cycles(self, kinematics, events_data, fps):
        """
        Aggregates kinematic signals into normalized gait cycles (0-100%).
        
        Cuts the signal into individual strides based on Heel Strike events,
        interpolates them to a fixed length (100 points), and computes Mean/Std.
        """
        stats = {}
        src = self._extract_events_dict(events_data)

        # Map output keys to sides (Left/Right)
        angle_to_side = {k: "left" if "LEFT" in k else "right" for k in kinematics}

        # Pre-calculate Heel Strike frames for both sides
        hs_map = {}
        for side in ['left', 'right']:
            events = src.get(side, [])
            hs_map[side] = sorted([e.frame for e in events
                                   if hasattr(e, 'event_type') and e.event_type == GaitEventType.HEEL_STRIKE])

        for key, signal in kinematics.items():
            side = angle_to_side.get(key)
            if not side: continue

            heel_strikes = hs_map[side]
            cycles_list = []

            # Cut signal between consecutive Heel Strikes
            for i in range(len(heel_strikes) - 1):
                start = heel_strikes[i]
                end = heel_strikes[i + 1]

                raw_cycle = signal[start: end]
                raw_cycle = [r for r in raw_cycle if r is not None]

                if len(raw_cycle) < 5:
                    continue

                # Interpolate to exactly 100 points
                x_original = np.linspace(0, 1, len(raw_cycle))
                interpolator = interp1d(x_original, raw_cycle, kind='linear')

                x_target = np.linspace(0, 1, 100)
                normalized_cycle = interpolator(x_target)
                cycles_list.append(normalized_cycle)

            if cycles_list:
                stack = np.vstack(cycles_list)
                stats[key] = {
                    "mean": np.mean(stack, axis=0).tolist(),
                    "std": np.std(stack, axis=0).tolist(),
                    "n_cycles": len(cycles_list)
                }
        return stats

    @staticmethod
    def _serialize_phase(p):
        return {
            "type": str(p.phase_type),
            "start": int(p.start_frame),
            "end": int(p.end_frame),
            "duration": int(p.duration)
        }

    @staticmethod
    def _serialize_support(p):
        return {
            "type": str(p.support_type),
            "start": int(p.start_frame),
            "end": int(p.end_frame),
            "duration": int(p.duration)
        }
