from functools import partial

import cv2
import numpy as np
from tqdm import tqdm

from rtmlib import PoseTracker, Custom
from utils.jsonSerializer import KeypointSerializer

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
openpose_skeleton = False  # True for openpose-style, False for mmpose-style

custom = partial(
    Custom,
    det_class="YOLOX",
    det="models/yolo_x.onnx",
    det_input_size=(640, 640),
    pose_class="RTMPose",
    pose="models/rtmpose_x.onnx",
    pose_input_size=(288, 384),
    backend=backend,
    device=device,
    to_openpose=openpose_skeleton,
)

tracker = PoseTracker(
    solution=custom,
    det_frequency=1,
    to_openpose=openpose_skeleton,
    backend=backend,
    device=device,
    tracking=True,
)

MIN_BBOX_SIZE = 250

# cap = cv2.VideoCapture("videos/logitech-1920-60-8.avi")  # Video file path
cap = cv2.VideoCapture("videos/test_interpolated.mp4")  # Video file path
tot_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

file_name = "interpolated"
serializer = KeypointSerializer("./results/", f"{file_name}.json")

pbar = tqdm(total=tot_frame)

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    keypoints, scores = tracker(frame)

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

        if height >= MIN_BBOX_SIZE:
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
cv2.destroyAllWindows()
