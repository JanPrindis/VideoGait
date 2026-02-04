import cv2
import numpy as np
import onnxruntime as ort

# Disable warnings
ort.set_default_logger_severity(3)

from tqdm import tqdm
from functools import partial
from pathlib import Path

from detectors import POSE_DETECTORS
from detectors.base import BaseDetector
from utils.jsonSerializer import KeypointSerializer
from .tools import PoseTracker, Custom


@POSE_DETECTORS.register
class RTMLib(BaseDetector):
    def __init__(self, config: dict):
        super().__init__(config)

        # Parameters
        self.device = self.config.get('device', 'cuda')
        self.backend = self.config.get('backend', 'onnxruntime')
        self.min_bbox_size = self.config.get('min_bbox_size', 250)
        self.openpose_skeleton = self.config.get('openpose_skeleton', False)

        # Tracking config
        tracking = self.config.get('tracking', True)

        # Model weight paths
        self.base_dir = Path(__file__).parent.resolve()
        det_rel_path = self.config.get('det_model', 'yolo_x.onnx')
        pose_rel_path = self.config.get('pose_model', 'rtmpose_x.onnx')

        det_path = str(self.base_dir / "models" / det_rel_path)
        pose_path = str(self.base_dir / "models" / pose_rel_path)

        # Other model specific config
        det_class = self.config.get('det_class', 'YOLOX')
        pose_class = self.config.get('pose_class', 'RTMPose')
        det_input_size = tuple(self.config.get('det_input_size', [640, 640]))
        pose_input_size = tuple(self.config.get('pose_input_size', [288, 384]))

        print(f"[RTMLib] Initializing with models:\n  Det: {det_path}\n  Pose: {pose_path}")

        # Initialize Custom Solution
        self.custom = partial(
            Custom,
            det_class=det_class,
            det=det_path,
            det_input_size=det_input_size,
            pose_class=pose_class,
            pose=pose_path,
            pose_input_size=pose_input_size,
            backend=self.backend,
            device=self.device,
            to_openpose=self.openpose_skeleton,
        )

        self.tracker = PoseTracker(
            solution=self.custom,
            det_frequency=1,
            to_openpose=self.openpose_skeleton,
            backend=self.backend,
            device=self.device,
            tracking=tracking,
        )

    def detect(self, video_path, output_path):
        v_path = Path(video_path).resolve()
        root_out_path = Path(output_path).resolve()

        # Create output directory
        save_dir = root_out_path / "pose_detector_data"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get output filename: video.mp4 -> video.json
        json_filename = f"{v_path.stem}.json"

        cap = cv2.VideoCapture(str(v_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Serializer
        serializer = KeypointSerializer(str(save_dir), json_filename)

        # TQDM Progress bar
        pbar = tqdm(total=total_frames, desc=f"RTMLib: {v_path.name}", unit="frame")

        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            keypoints, scores = self.tracker(frame)

            # Not found anything
            if len(scores) == 0:
                frame_idx += 1
                pbar.update(1)
                continue

            valid_detections = []

            # Calculate bbox size and filter oun small detections
            for i, (kpts, sc) in enumerate(zip(keypoints, scores)):
                x_min, y_min = np.min(kpts[:, 0]), np.min(kpts[:, 1])
                x_max, y_max = np.max(kpts[:, 0]), np.max(kpts[:, 1])
                width, height = x_max - x_min, y_max - y_min

                if height >= self.min_bbox_size:
                    total_score = float(np.sum(sc))
                    valid_detections.append((i, total_score, (x_min, y_min, x_max, y_max)))

            if not valid_detections:
                frame_idx += 1
                pbar.update(1)
                continue

            # Best detection
            best_idx, best_score, best_bbox = max(valid_detections, key=lambda x: x[1])

            filtered_keypoints = keypoints[np.newaxis, best_idx]
            filtered_scores = scores[np.newaxis, best_idx]

            # Flatten [x, y, score]
            frame_kpts = np.hstack([filtered_keypoints[0], filtered_scores[0][:, None]]).flatten().tolist()

            serializer.add_frame(
                frame_number=frame_idx,
                keypoints=frame_kpts
            )

            frame_idx += 1
            pbar.update(1)

        serializer.save()
        cap.release()
        pbar.close()
