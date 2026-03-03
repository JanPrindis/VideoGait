import cv2
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path

from detectors import POSE_DETECTORS
from ..base import BaseDetector
from utils.json_serializer import KeypointSerializer
from utils.logger import log


@POSE_DETECTORS.register
class PoseLandmarker(BaseDetector):
    def __init__(self, config: dict):
        super().__init__(config)

        # Model paths
        self.base_dir = Path(__file__).parent.resolve()

        # Get model name from config
        model_filename = self.config.get("model_name", "pose_landmarker_heavy.task")
        self.model_path = self.base_dir / "models" / model_filename

        if not self.model_path.exists():
            log("PoseLandmarker", f"Warning: Model not found at {self.model_path}", level="warning")

        log("PoseLandmarker", f"Loading model: {self.model_path}", level="info")

        # MediaPipe parameters
        self.min_bbox_size = self.config.get("min_bbox_size", 250)
        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))

        self.options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options = base_options,
            running_mode = mp.tasks.vision.RunningMode.VIDEO,
            num_poses = 1,
            min_pose_detection_confidence = self.config.get("min_conf", 0.5),
            min_pose_presence_confidence = self.config.get("min_presence", 0.5),
            min_tracking_confidence = self.config.get("min_tracking", 0.5)
        )

    def detect(self, video_path, output_path, keypoint_filename_override: str = None):
        v_path = Path(video_path).resolve()
        save_dir = Path(output_path).resolve()

        # Create output folder
        save_dir.mkdir(parents=True, exist_ok=True)

        json_filename = keypoint_filename_override if keypoint_filename_override else "keypoints.json"
        serializer = KeypointSerializer(str(save_dir), json_filename)

        # Load input video
        cap = cv2.VideoCapture(str(v_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Progress bar
        pbar = tqdm(total=total_frames, desc=f"MediaPipe: {v_path.name}", unit="frame")

        with mp.tasks.vision.PoseLandmarker.create_from_options(self.options) as landmarker:
            frame_idx = 0

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break  # End of video

                current_fps = fps if fps > 0 else 60.0
                frame_timestamp_ms = int((frame_idx / current_fps) * 1000.0)

                # BGR (OpenCV) -> RGB (MediaPipe)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                # Not found anyone
                if not detection_result.pose_landmarks:
                    frame_idx += 1
                    pbar.update(1)
                    continue

                valid_poses = []

                for landmarks in detection_result.pose_landmarks:
                    # Get normalized Y coords
                    ys = [lm.y for lm in landmarks]
                    min_y, max_y = min(ys), max(ys)

                    # Convert to pixels
                    pixel_height = (max_y - min_y) * height

                    if pixel_height >= self.min_bbox_size:
                        valid_poses.append(landmarks)

                if not valid_poses:
                    frame_idx += 1
                    pbar.update(1)
                    continue

                # Take the first detection - we are only detecting a single person (num_pose = 1)
                landmarks = valid_poses[0]

                # Denormalization
                # Format: [x, y, score, x, y, score, ...]
                keypoints_list = []
                for lm in landmarks:
                    keypoints_list.extend([
                        lm.x * width,
                        lm.y * height,
                        lm.presence
                    ])

                serializer.add_frame(
                    frame_number=frame_idx,
                    keypoints=keypoints_list
                )

                frame_idx += 1
                pbar.update(1)

        serializer.save()
        cap.release()
        pbar.close()
