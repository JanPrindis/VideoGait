import numpy as np
from abc import ABC, abstractmethod

from skeletons import get_skeleton_by_name
from utils.data import get_valid_range, get_keypoints, cubic_interpolate_nan, butterworth_filter, average_with_nones


class BaseHeuristicDetector(ABC):
    def __init__(self, full_config):
        self.config = full_config

        self.preprocessing_config = full_config.get("preprocessing", {})
        self.heuristic_config = full_config.get("event_detector", {}).get("heuristic", {})

        # Algorithm specific parameters
        self.algorithm_params = self.heuristic_config.get("params", {})

        # Pre-processing parameters
        self.framerate = self.preprocessing_config.get("framerate", 60)
        self.confidence_threshold = self.preprocessing_config.get("confidence_threshold", 0.4)
        self.exclude_ratio = self.preprocessing_config.get("exclude_ratio", 0.05)
        self.min_segment_length = self.preprocessing_config.get("min_segment_length", 60)
        self.outlier_ratio = self.preprocessing_config.get("outlier_ratio", 0.2)

        self.use_filter = self.preprocessing_config.get("use_filter", True)
        self.filter_cutoff = self.preprocessing_config.get("filter_cutoff", 6)
        self.filter_order = self.preprocessing_config.get("filter_order", 4)

        skel_name = self.heuristic_config.get("skeleton", "HALPE")
        try:
            self.skeleton = get_skeleton_by_name(skel_name)
        except ValueError as e:
            raise ValueError(f"[Heuristic] Skeleton config error: {e}")

    def _get_hip_reference_data(self, json_path):
        # Try to get center HIP keypoints
        raw_data, valid_indices = get_keypoints(json_path, self.skeleton, ["HIP"], self.confidence_threshold)
        global_offset = valid_indices[0] if valid_indices else 0

        if raw_data and "HIP" in raw_data and len(raw_data["HIP"]) > 0:
            return raw_data["HIP"], False, global_offset

        # Fallback - Average center hip from both sides
        try:
            raw_lr, _ = get_keypoints(json_path, self.skeleton, ["LEFT_HIP", "RIGHT_HIP"], self.confidence_threshold)
        except ValueError:
            return None, False, 0

        if "LEFT_HIP" not in raw_lr or "RIGHT_HIP" not in raw_lr:
            return None, False, 0

        l_hip_seq = raw_lr["LEFT_HIP"]
        r_hip_seq = raw_lr["RIGHT_HIP"]

        virtual_hip = []
        min_len = min(len(l_hip_seq), len(r_hip_seq))

        for i in range(min_len):
            l_item = l_hip_seq[i]
            r_item = r_hip_seq[i]

            # Unpack X and Y for average_with_nones
            lx = l_item[0] if l_item else None
            rx = r_item[0] if r_item else None

            ly = l_item[1] if l_item else None
            ry = r_item[1] if r_item else None

            mx = average_with_nones(lx, rx)
            my = average_with_nones(ly, ry)

            if mx is not None and my is not None:
                # Calculate average confidence
                confs = []
                if l_item: confs.append(l_item[2])
                if r_item: confs.append(r_item[2])
                mc = sum(confs) / len(confs) if confs else 0.0

                virtual_hip.append((mx, my, mc))
            else:
                virtual_hip.append((np.nan, np.nan, 0.0))

        return virtual_hip, True

    def _prepare_hip_and_ranges(self, json_path):
        hip_data, is_virtual_hip, global_offset = self._get_hip_reference_data(json_path)

        if not hip_data:
            return None, [], False, 0

        # Convert to X array for valid range detection
        hip_x_series = np.array(
            [kp[0] if kp is not None else np.nan for kp in hip_data],
            dtype=float
        )

        # Get all valid ranges
        valid_ranges = get_valid_range(
            hip_x_series,
            self.framerate,
            exclude_percent=self.exclude_ratio,
            min_segment_length=self.min_segment_length,
            outlier_ratio=self.outlier_ratio,
            filter_cutoff=self.filter_cutoff,
            filter_order=self.filter_order
        )

        return hip_data, valid_ranges, is_virtual_hip, global_offset

    def _extract_segment_data(self, json_path, start, end, required_keypoints, hip_data, is_virtual_hip):
        keys_to_fetch = [k for k in required_keypoints if not (k == "HIP" and is_virtual_hip)]

        # Get raw data
        raw_kps, _ = get_keypoints(json_path, self.skeleton, keys_to_fetch, self.confidence_threshold)

        if is_virtual_hip and "HIP" in required_keypoints:
            raw_kps["HIP"] = hip_data

        processed_data = {}
        for kp_name in required_keypoints:
            coords_list = raw_kps.get(kp_name, [])

            if len(coords_list) <= end:
                continue

            # Trim data for the current segment
            trimmed = coords_list[start: end + 1]

            arr_data = []
            for item in trimmed:
                if item is None:
                    arr_data.append([np.nan, np.nan])
                else:
                    arr_data.append([item[0], item[1]])

            arr = np.array(arr_data, dtype=float)

            # Interpolation
            arr[:, 0] = cubic_interpolate_nan(arr[:, 0])
            arr[:, 1] = cubic_interpolate_nan(arr[:, 1])

            # Smoothing
            if self.use_filter:
                arr[:, 0] = butterworth_filter(arr[:, 0], cutoff=self.filter_cutoff, fs=self.framerate,
                                               order=self.filter_order)
                arr[:, 1] = butterworth_filter(arr[:, 1], cutoff=self.filter_cutoff, fs=self.framerate,
                                               order=self.filter_order)

            processed_data[kp_name] = arr

        return processed_data

    @abstractmethod
    def detect_events(self, processed_data):
        pass

    @abstractmethod
    def get_required_keypoints(self):
        pass

    def run_inference(self, json_path: str, output_dir: str = None):
        """
        Executes the full inference pipeline on a keypoint JSON file.

        It handles data loading, segmentation, event detection for each segment,
        and aggregation of results.

        Args:
            json_path (str): Path to the input keypoints JSON file.
            output_dir (str, optional): Directory to save debug outputs.

        Returns:
            dict: Dictionary containing detected events, global ranges, and framerate.
        """
        # Get hip data and valid ranges
        hip_data, valid_ranges, is_virtual_hip, global_offset = self._prepare_hip_and_ranges(json_path)

        if not valid_ranges:
            return {
                "events": {"left": [], "right": []},
                "global_ranges": [],
                "framerate": self.framerate
            }

        # Create output folder
        if output_dir is not None and self.save_debug_plot:
            debug_dir = os.path.join(output_dir, "event_detector_debug")

            # If debug folder exists, remove
            if os.path.exists(debug_dir):
                shutil.rmtree(debug_dir)

            # Create clean debug dir (does not contain old files)
            os.makedirs(debug_dir, exist_ok=True)

        req_kps = self.get_required_keypoints()
        all_left_events = []
        all_right_events = []
        final_global_ranges = []

        # Iterate each segment separately
        for i, (start, end) in enumerate(valid_ranges):
            segment_data = self._extract_segment_data(json_path, start, end, req_kps, hip_data, is_virtual_hip)

            # If data is missing - skip
            if not segment_data or len(segment_data) != len(req_kps):
                continue

            # Event detection
            plot_path = os.path.join(output_dir, "event_detector_debug") if output_dir else None
            l_ev, r_ev = self.detect_events(
                processed_data=segment_data,
                plot_path=plot_path,
                sequence_number=i
            )

            # Offset calculation
            current_shift = start + global_offset

            for e in l_ev: e.frame += current_shift
            for e in r_ev: e.frame += current_shift

            # Aggregation of results
            all_left_events.extend(l_ev)
            all_right_events.extend(r_ev)

            final_global_ranges.append((start + global_offset, end + global_offset))

        # Order events by time
        all_left_events.sort(key=lambda x: x.frame)
        all_right_events.sort(key=lambda x: x.frame)

        return {
            "events": {
                "left": all_left_events,
                "right": all_right_events
            },
            "global_ranges": final_global_ranges,
            "framerate": self.framerate
        }
