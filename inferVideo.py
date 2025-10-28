from functools import partial
from os import path

import cv2
import numpy as np
from tqdm import tqdm

from rtmlib import PoseTracker, Custom
from utils.jsonSerializer import KeypointSerializer
from utils.data import create_folder_if_not_exists

class RTMLib:
    def __init__(self):
        self.device = 'cuda'
        self.backend = 'onnxruntime'  # opencv, onnxruntime, openvino
        self.openpose_skeleton = False  # True for openpose-style, False for mmpose-style

        self.custom = partial(
            Custom,
            det_class="YOLOX",
            det="models/yolo_x.onnx",
            det_input_size=(640, 640),
            pose_class="RTMPose",
            pose="models/rtmpose_x.onnx",
            pose_input_size=(288, 384),
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
            tracking=True,
        )

        self.MIN_BBOX_SIZE = 250

    def inferVideo(self, in_path, out_path):
        create_folder_if_not_exists(out_path)

        cap = cv2.VideoCapture(in_path)  # Video file path
        tot_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        file_name = path.basename(in_path).split(".")[0]
        serializer = KeypointSerializer(out_path, f"{file_name}.json")

        pbar = tqdm(total=tot_frame, leave=False, desc=f"Current video progress")

        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            keypoints, scores = self.tracker(frame)

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

                if height >= self.MIN_BBOX_SIZE:
                    total_score = float(np.sum(sc))
                    valid_detections.append((i, total_score, (x_min, y_min, x_max, y_max)))

            if not valid_detections:
                frame_idx += 1
                pbar.update(1)
                continue

            # Get the detection with the highest score
            best_idx, best_score, best_bbox = max(valid_detections, key=lambda x: x[1])

            filtered_keypoints = keypoints[np.newaxis, best_idx]
            filtered_scores = scores[np.newaxis, best_idx]

            frame_kpts = np.hstack([filtered_keypoints[0], filtered_scores[0][:, None]]).flatten().tolist()

            serializer.add_frame(
                frame_number=frame_idx,
                keypoints=frame_kpts
            )

            frame_idx += 1
            pbar.update(1)

        serializer.save()
        cap.release()
