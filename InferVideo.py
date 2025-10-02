import time
import cv2
import numpy as np

from rtmlib import BodyWithFeet, PoseTracker
from utils import jsonSerializer
from utils.jsonSerializer import KeypointSerializer

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

cap = cv2.VideoCapture("videos/logitech-1920-60-8.avi")  # Video file path

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

body_feet_tracker = PoseTracker(
    BodyWithFeet,
    det_frequency=1,
    to_openpose=openpose_skeleton,
    mode='performance',  # balanced, performance, lightweight
    backend=backend,
    device=device,
    tracking=False
)

frame_idx = 0

file_name = "result"
serializer = KeypointSerializer("./inferResults/", f"{file_name}.json")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    s = time.time()
    keypoints, scores = body_feet_tracker(frame)

    if len(scores) == 0:
        continue

    total_scores = [sum(s) for s in scores]
    best_idx = total_scores.index(max(total_scores))

    filtered_keypoints = keypoints[np.newaxis, best_idx]
    filtered_scores = scores[np.newaxis, best_idx]

    det_time = time.time() - s
    print('det: ', det_time)

    frame_kpts = np.hstack([filtered_keypoints[0], filtered_scores[0][:, None]]).flatten().tolist()

    # img_show = frame.copy()
    serializer.add_frame(
        frame_number=frame_idx,
        keypoints=frame_kpts
    )

    # img_show = draw_skeleton(img_show,
    #                          filtered_keypoints,
    #                          filtered_scores,
    #                          openpose_skeleton=openpose_skeleton,
    #                          kpt_thr=0.6,
    #                          line_width=3)

    # img_show = cv2.resize(img_show, (960, 640))
    # cv2.imshow('Result', img_show)
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'): # Press 'q' to exit
    #     break

    # video.write(img_show)
    frame_idx += 1

serializer.save()
cap.release()
cv2.destroyAllWindows()
