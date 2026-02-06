import os

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

from tqdm import tqdm

from skeletons.skeletons import SkeletonSide
from utils.config_utils import resolve_skeleton_from_config
from utils.gait_structs import PhaseType, GaitEventType
from utils.json_serializer import KeypointSerializer


class GaitVisualizer:
    def __init__(self, app_config):
        self.cfg = app_config

        skel_name = resolve_skeleton_from_config(app_config)
        self.skel = skel_name

        viz_cfg = self.cfg.get('visualization', {})
        colors_cfg = viz_cfg.get('colors', {})

        # Load colors (RGB/HEX -> BGR)
        self.colors = {
            SkeletonSide.LEFT: self._parse_color(colors_cfg.get('left', [0, 128, 255])),
            SkeletonSide.RIGHT: self._parse_color(colors_cfg.get('right', [255, 0, 0])),
            SkeletonSide.CENTER: self._parse_color(colors_cfg.get('center', [0, 255, 0]))
        }

        self.phase_colors = {
            PhaseType.STANCE: self._parse_color(colors_cfg.get('stance', [0, 255, 0])),
            PhaseType.SWING: self._parse_color(colors_cfg.get('swing', [255, 0, 0])),
            PhaseType.UNKNOWN: self._parse_color(colors_cfg.get('unknown', [128, 128, 128])),
        }

        self.dimmed_color = self._parse_color(colors_cfg.get('dimmed', [128, 128, 128]))
        self.footprint_to_col = self._parse_color(colors_cfg.get('footprint_to', [255, 0, 255]))
        self.footprint_hs_col = self._parse_color(colors_cfg.get('footprint_hs', [255, 255, 0]))
        self.com_col = self._parse_color(colors_cfg.get('com_line', [255, 0, 255]))

        # Draw settings
        self.thickness = viz_cfg.get('line_thickness', 2)
        self.radius = viz_cfg.get('keypoint_radius', 4)
        self.fp_duration = viz_cfg.get('footprint_duration', 30)
        self.trail_thickness = viz_cfg.get('trail_thickness', 2)

        # Output toggles
        outputs = viz_cfg.get('outputs', {})
        self.do_overlay = outputs.get('enable_overlay', True)
        self.do_kinematics = outputs.get('enable_kinematics', False)
        self.do_logic = outputs.get('enable_logic', False)

        # Dynamic keypoint mapping
        self.kps_map = {
            'l_ankle': self._get_kp_idx("LEFT_ANKLE"),
            'r_ankle': self._get_kp_idx("RIGHT_ANKLE"),
            'l_hip': self._get_kp_idx("LEFT_HIP"),
            'r_hip': self._get_kp_idx("RIGHT_HIP"),
            'l_heel': self._get_kp_idx("LEFT_HEEL"),
            'r_heel': self._get_kp_idx("RIGHT_HEEL"),
            'l_toe': self._get_kp_idx("LEFT_FOOT_INDEX"),
            'r_toe': self._get_kp_idx("RIGHT_FOOT_INDEX")
        }

        # Lower body keypoints
        self.lower_body_ids = set()
        for name, member in self.skel.keypoints.__members__.items():
            n = name.upper()
            if any(x in n for x in ["KNEE", "ANKLE", "HIP", "HEEL", "FOOT", "TOE"]):
                self.lower_body_ids.add(int(member))

    def _get_kp_idx(self, name: str) -> Optional[int]:
        if name in self.skel.keypoints.__members__:
            return int(self.skel.keypoints[name])
        return None

    @staticmethod
    def _load_and_parse_json(json_path: str, video_total_frames: int) -> np.ndarray:
        """
        Loads keypoints from a JSON file and converts them to a numpy array.

        Args:
            json_path (str): Path to the keypoints JSON file.
            video_total_frames (int): Total number of frames in the video (for array allocation).

        Returns:
            np.ndarray: Array of shape (frames, keypoints, 3) containing (x, y, conf).
        """
        # Load JSON
        raw_data = KeypointSerializer.load(json_path)

        if not raw_data:
            print("[Visualizer] Warning: JSON is empty!")
            return np.zeros((video_total_frames, 17, 3))  # Fallback shape

        # Get number of keypoints
        first_kps = raw_data[0]['keypoints']
        num_keypoints = len(first_kps) // 3

        # Get length
        max_json_frame = 0
        parsed_entries = []

        for entry in raw_data:
            # image_id ex: "123.jpg" -> 123
            try:
                fid_str = entry['image_id']
                fid = int(fid_str.split('.')[0])

                max_json_frame = max(max_json_frame, fid)
                parsed_entries.append((fid, entry['keypoints']))
            except (ValueError, IndexError):
                continue

        # Allocate array (size = max(video_len, json_len))
        final_len = max(video_total_frames, max_json_frame + 1)
        full_kps = np.zeros((final_len, num_keypoints, 3), dtype=np.float32)

        # Fill with data
        for fid, kps_list in parsed_entries:
            kps_arr = np.array(kps_list, dtype=np.float32).reshape(-1, 3)

            # Sanity check - number of keypoints does not change
            if kps_arr.shape[0] == num_keypoints:
                full_kps[fid] = kps_arr

        return full_kps

    def process_video(self, video_path: str, output_root: str,
                      keypoints_data: Union[np.ndarray, str],
                      valid_ranges: List[Tuple[int, int]] = None,
                      gait_data: Dict = None):

        vid_path = Path(video_path)
        out_root = Path(output_root)
        out_root.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {vid_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if isinstance(keypoints_data, str):
            print(f"[Visualizer] Loading keypoints from JSON: {keypoints_data}")
            keypoints_data = self._load_and_parse_json(keypoints_data, total_frames)

        if len(keypoints_data) > total_frames:
            print(
                f"[Visualizer] Warning: Keypoints length ({len(keypoints_data)}) > Video frames ({total_frames}). Truncating extra.")

        # Init writers
        writers = {}
        if self.do_overlay:
            writers['inspection'] = self._create_writer(out_root, vid_path.stem, "inspection", fps, (w, h))
        if self.do_kinematics:
            writers['kinematics'] = self._create_writer(out_root, vid_path.stem, "kinematics", fps, (w, h))
        if self.do_logic and gait_data:
            writers['logic'] = self._create_writer(out_root, vid_path.stem, "logic", fps, (w, h))

        # Init buffers
        trail_buffers = {
            'left': [],
            'right': []
        }
        footprints = []

        l_heel, r_heel = self.kps_map['l_heel'], self.kps_map['r_heel']
        l_toe, r_toe = self.kps_map['l_toe'], self.kps_map['r_toe']

        frame_idx = 0
        last_valid_clip_id = -1

        with tqdm(total=total_frames, desc=f"[Visualizer] {vid_path.name}", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Check context validity
                is_valid, clip_info, current_clip_id = self._check_validity(frame_idx, valid_ranges)

                # Reset trails if clip changed
                if is_valid and current_clip_id != last_valid_clip_id:
                    trail_buffers['left'].clear()
                    trail_buffers['right'].clear()
                    last_valid_clip_id = current_clip_id

                kps = None
                if frame_idx < len(keypoints_data):
                    kps = keypoints_data[frame_idx]

                # ---------------------------------------------------------
                # INSPECTION MODE
                # ---------------------------------------------------------
                if 'inspection' in writers:
                    canvas = frame.copy()
                    if kps is not None:
                        self._draw_skeleton(canvas, kps, color_mode='side', use_dimmed=not is_valid)
                    self._draw_info_box(canvas, frame_idx, total_frames, clip_info, is_valid, "Inspection")
                    writers['inspection'].write(canvas)

                # ---------------------------------------------------------
                # KINEMATICS MODE (Trails, CoM, Footprints)
                # ---------------------------------------------------------
                if 'kinematics' in writers:
                    canvas = frame.copy()
                    if kps is not None and is_valid:
                        # Trails from ankles
                        self._update_trails(trail_buffers, kps, self.kps_map['l_ankle'], self.kps_map['r_ankle'])

                        # Footprints
                        self._update_footprints(footprints, gait_data, frame_idx, kps,
                                                      l_ankle=self.kps_map['l_ankle'], r_ankle=self.kps_map['r_ankle'],
                                                      l_heel=l_heel, r_heel=r_heel,
                                                      l_toe=l_toe, r_toe=r_toe)

                    if kps is not None:
                        self._draw_com_drop(canvas, kps, self.kps_map['l_hip'], self.kps_map['r_hip'])
                        self._draw_skeleton(canvas, kps, color_mode='side', use_dimmed=not is_valid)

                    self._draw_trails(canvas, trail_buffers)
                    self._draw_footprints_render(canvas, footprints, frame_idx)

                    self._draw_info_box(canvas, frame_idx, total_frames, clip_info, is_valid, "Kinematics")
                    writers['kinematics'].write(canvas)

                # ---------------------------------------------------------
                # LOGIC MODE (Stance/Swing)
                # ---------------------------------------------------------
                if 'logic' in writers:
                    canvas = frame.copy()
                    if kps is not None:
                        l_col, r_col = self._get_phase_colors_at_frame(frame_idx, gait_data)

                        self._draw_skeleton(canvas, kps, color_mode='override',
                                            override_colors={SkeletonSide.LEFT: l_col, SkeletonSide.RIGHT: r_col},
                                            use_dimmed=not is_valid,
                                            restrict_to_legs=True)

                    self._draw_info_box(canvas, frame_idx, total_frames, clip_info, is_valid, "Logic Check")
                    writers['logic'].write(canvas)

                pbar.update(1)
                frame_idx += 1

        # Cleanup
        cap.release()
        for w in writers.values():
            w.release()
        print(f"[Visualizer] Finished. Outputs in: {out_root}")

    # =========================================================================
    # Drawing and Logic
    # =========================================================================

    def _draw_skeleton(self, img, keypoints, color_mode='side', override_colors=None, use_dimmed=False,
                       restrict_to_legs=False):
        # LINKS
        for link_name, (kp1_enum, kp2_enum) in self.skel.links.items():
            idx1, idx2 = int(kp1_enum), int(kp2_enum)
            if idx1 >= len(keypoints) or idx2 >= len(keypoints): continue
            if keypoints[idx1][2] < 0.3 or keypoints[idx2][2] < 0.3: continue

            pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
            pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))

            # COLOR LOGIC
            if use_dimmed:
                color = self.dimmed_color
            elif color_mode == 'override' and override_colors:
                side = self.skel.get_link_side(idx1, idx2)

                is_leg_link = (idx1 in self.lower_body_ids) and (idx2 in self.lower_body_ids)

                if restrict_to_legs and not is_leg_link:
                    color = self.dimmed_color
                else:
                    color = override_colors.get(side, self.colors.get(side, self.colors[SkeletonSide.CENTER]))

            else:
                side = self.skel.get_link_side(idx1, idx2)
                color = self.colors.get(side, (255, 255, 255))

            cv2.line(img, pt1, pt2, color, self.thickness)

        # KEYPOINTS
        for i, kp_data in enumerate(keypoints):
            if kp_data[2] < 0.3: continue
            center = (int(kp_data[0]), int(kp_data[1]))

            if use_dimmed:
                color = self.dimmed_color
            elif color_mode == 'override' and override_colors:
                side = self.skel.get_keypoint_side(i)
                is_leg_point = (i in self.lower_body_ids)

                if restrict_to_legs and not is_leg_point:
                    color = self.colors.get(side, self.colors[SkeletonSide.CENTER])
                else:
                    color = override_colors.get(side, self.colors.get(side, self.colors[SkeletonSide.CENTER]))
            else:
                side = self.skel.get_keypoint_side(i)
                color = self.colors.get(side, (255, 255, 255))

            cv2.circle(img, center, self.radius, color, -1)

    def _draw_trails(self, img, buffers):
        for side_name, buffer in buffers.items():
            pts = list(buffer)
            if len(pts) < 2: continue

            if side_name == 'left':
                color = self.colors[SkeletonSide.LEFT]
            else:
                color = self.colors[SkeletonSide.RIGHT]

            cv2.polylines(img, [np.array(pts)], False, color, self.trail_thickness)

    @staticmethod
    def _update_trails(buffers, kps, l_idx, r_idx):
        # If skeleton does not have required keypoints - skip
        if l_idx is None or r_idx is None: return

        def add(idx, name):
            if idx < len(kps) and kps[idx][2] > 0.3:
                buffers[name].append((int(kps[idx][0]), int(kps[idx][1])))

        add(l_idx, 'left')
        add(r_idx, 'right')

    @staticmethod
    def _update_footprints(fp_list, gait_data, frame_idx, kps,
                           l_ankle, r_ankle, l_heel, r_heel, l_toe, r_toe):
        if not gait_data or 'events' not in gait_data: return

        def get_best_point(preferred, fallback, kps):
            # Preferred = Heel/Toe
            if preferred is not None and preferred < len(kps) and kps[preferred][2] > 0.3:
                return preferred
            # Fallback = Ankle
            if fallback is not None and fallback < len(kps) and kps[fallback][2] > 0.3:
                return fallback
            return None

        def check(events, ankle, heel, toe):
            for e in events:
                if e.frame == frame_idx:
                    # HEEL STRIKE -> Heel
                    if e.event_type == GaitEventType.HEEL_STRIKE:
                        idx = get_best_point(heel, ankle, kps)
                        if idx is not None:
                            fp_list.append((int(kps[idx][0]), int(kps[idx][1]), frame_idx, 1))

                    # TOE OFF -> Toe
                    elif e.event_type == GaitEventType.TOE_OFF:
                        idx = get_best_point(toe, ankle, kps)
                        if idx is not None:
                            fp_list.append((int(kps[idx][0]), int(kps[idx][1]), frame_idx, 2))

        # Check Left
        check(gait_data['events'].get('left', []), l_ankle, l_heel, l_toe)
        # Check Right
        check(gait_data['events'].get('right', []), r_ankle, r_heel, r_toe)

    def _draw_footprints_render(self, img, fp_list, curr_frame):
        active = []
        for (x, y, t, f_type) in fp_list:
            age = curr_frame - t
            if age < self.fp_duration:
                # Calculate radius for fade-out effect
                rad = int(10 * (1 - age / self.fp_duration))
                if rad > 0:
                    color = self.footprint_hs_col if f_type == 1 else self.footprint_to_col

                    cv2.circle(img, (x, y), rad, color, -1)
                    cv2.circle(img, (x, y), rad, (0, 0, 0), 1)

                active.append((x, y, t, f_type))
        fp_list[:] = active

    def _draw_com_drop(self, img, kps, l_hip, r_hip):
        if l_hip is None or r_hip is None: return
        if l_hip >= len(kps) or r_hip >= len(kps): return

        if kps[l_hip][2] < 0.3 or kps[r_hip][2] < 0.3: return

        mx = int((kps[l_hip][0] + kps[r_hip][0]) / 2)
        my = int((kps[l_hip][1] + kps[r_hip][1]) / 2)

        valid_ys = [p[1] for p in kps if p[2] > 0.3]
        if not valid_ys: return
        ground_y = int(max(valid_ys))

        cv2.line(img, (mx, my), (mx, ground_y), self.com_col, 2)
        cv2.circle(img, (mx, my), 6, self.com_col, -1)
        cv2.line(img, (mx - 10, ground_y), (mx + 10, ground_y), self.com_col, 2)

    @staticmethod
    def _draw_info_box(img, frame_idx, total, clip_info, is_valid, mode_name):
        # Background rectangle
        cv2.rectangle(img, (0, 0), (450, 100), (0, 0, 0), -1)

        status_col = (0, 255, 0) if is_valid else (128, 128, 128)

        font_scale = 0.9
        cv2.putText(
            img,
            f"Mode: {mode_name}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        cv2.putText(
            img,
            f"Frame: {frame_idx} / {total}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(
            img,
            clip_info,
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)

    @staticmethod
    def _check_validity(frame_idx, ranges):
        if not ranges:
            return True, "Full Video", 0

        for i, (start, end) in enumerate(ranges):
            if start <= frame_idx <= end:
                return True, f"Clip {i + 1} | Local: {frame_idx - start}", i

        return False, "Excluded", -1

    def _get_phase_colors_at_frame(self, frame_idx, gait_data):
        c_left = self.phase_colors[PhaseType.UNKNOWN]
        c_right = self.phase_colors[PhaseType.UNKNOWN]

        if not gait_data: return c_left, c_right

        def find_color(phases):
            if not phases: return self.phase_colors[PhaseType.UNKNOWN]
            for p in phases:
                if p.start_frame <= frame_idx <= p.end_frame:
                    return self.phase_colors.get(p.phase_type, self.phase_colors[PhaseType.UNKNOWN])
            return self.phase_colors[PhaseType.UNKNOWN]

        if 'left_phases' in gait_data: c_left = find_color(gait_data['left_phases'])
        if 'right_phases' in gait_data: c_right = find_color(gait_data['right_phases'])

        return c_left, c_right

    @staticmethod
    def _parse_color(color_input: Union[List[int], str]) -> Tuple[int, int, int]:
        # If list/tuple -> expect RGB input
        if isinstance(color_input, (list, tuple)):
            if len(color_input) >= 3:
                r, g, b = color_input[:3]
                return int(b), int(g), int(r) # RGB -> BGR
            return 255, 255, 255  # Fallback white

        # If string -> expect HEX input
        if isinstance(color_input, str):
            hex_str = color_input.lstrip('#')
            try:
                # Split to R, G, B
                if len(hex_str) == 6:
                    r = int(hex_str[0:2], 16)
                    g = int(hex_str[2:4], 16)
                    b = int(hex_str[4:6], 16)
                    return b, g, r  # RGB -> BGR
            except ValueError:
                print(f"[Visualizer] Warning: Invalid HEX color '{color_input}', using white.")

        return 255, 255, 255  # Fallback

    @staticmethod
    def _create_writer(root, stem, suffix, fps, size):
        path = root / f"{stem}_{suffix}.mp4"
        return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
