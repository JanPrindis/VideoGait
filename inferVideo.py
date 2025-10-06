import time
from functools import partial

import cv2
import numpy as np

from rtmlib import BodyWithFeet, PoseTracker, RTMDet, RTMPose, Custom
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

cap = cv2.VideoCapture("videos/logitech-1920-60-8.avi")  # Video file path
file_name = "test"
serializer = KeypointSerializer("./results/", f"{file_name}.json")

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    s = time.time()
    keypoints, scores = tracker(frame)

    if len(scores) == 0:
        continue

    total_scores = [sum(s) for s in scores]
    best_idx = total_scores.index(max(total_scores))

    filtered_keypoints = keypoints[np.newaxis, best_idx]
    filtered_scores = scores[np.newaxis, best_idx]

    det_time = time.time() - s
    print('det: ', det_time)

    frame_kpts = np.hstack([filtered_keypoints[0], filtered_scores[0][:, None]]).flatten().tolist()

    serializer.add_frame(
        frame_number=frame_idx,
        keypoints=frame_kpts
    )

    frame_idx += 1

serializer.save()
cap.release()
cv2.destroyAllWindows()
