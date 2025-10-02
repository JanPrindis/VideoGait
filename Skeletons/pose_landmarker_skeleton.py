from enum import IntEnum
from Skeletons.skeletons import SkeletonDefinition

class PoseLandmarkerKeypoints(IntEnum):
    NOSE = 0
    LEFT_EYE_IN = 1
    LEFT_EYE = 2
    LEFT_EYE_OUT = 3
    RIGHT_EYE_IN = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUT = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_L = 9
    MOUTH_R = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


__POSE_LANDMARKER_SKELETON = {
    "left_upper_leg": (PoseLandmarkerKeypoints.LEFT_HIP, PoseLandmarkerKeypoints.LEFT_KNEE),
    "left_lower_leg": (PoseLandmarkerKeypoints.LEFT_KNEE, PoseLandmarkerKeypoints.LEFT_ANKLE),
    "right_upper_leg": (PoseLandmarkerKeypoints.RIGHT_HIP, PoseLandmarkerKeypoints.RIGHT_KNEE),
    "right_lower_leg": (PoseLandmarkerKeypoints.RIGHT_KNEE, PoseLandmarkerKeypoints.RIGHT_ANKLE),

    "left_upper_arm": (PoseLandmarkerKeypoints.LEFT_SHOULDER, PoseLandmarkerKeypoints.LEFT_ELBOW),
    "left_lower_arm": (PoseLandmarkerKeypoints.LEFT_ELBOW, PoseLandmarkerKeypoints.LEFT_WRIST),
    "right_upper_arm": (PoseLandmarkerKeypoints.RIGHT_SHOULDER, PoseLandmarkerKeypoints.RIGHT_ELBOW),
    "right_lower_arm": (PoseLandmarkerKeypoints.RIGHT_ELBOW, PoseLandmarkerKeypoints.RIGHT_WRIST),

    "shoulders": (PoseLandmarkerKeypoints.LEFT_SHOULDER, PoseLandmarkerKeypoints.RIGHT_SHOULDER),
    "hips": (PoseLandmarkerKeypoints.LEFT_HIP, PoseLandmarkerKeypoints.RIGHT_HIP),
    "torso_left": (PoseLandmarkerKeypoints.LEFT_SHOULDER, PoseLandmarkerKeypoints.LEFT_HIP),
    "torso_right": (PoseLandmarkerKeypoints.RIGHT_SHOULDER, PoseLandmarkerKeypoints.RIGHT_HIP),

    "nose_to_left_eye": (PoseLandmarkerKeypoints.NOSE, PoseLandmarkerKeypoints.LEFT_EYE),
    "nose_to_right_eye": (PoseLandmarkerKeypoints.NOSE, PoseLandmarkerKeypoints.RIGHT_EYE),
    "eye_connection": (PoseLandmarkerKeypoints.LEFT_EYE, PoseLandmarkerKeypoints.RIGHT_EYE),
    "left_eye_to_ear": (PoseLandmarkerKeypoints.LEFT_EYE, PoseLandmarkerKeypoints.LEFT_EAR),
    "right_eye_to_ear": (PoseLandmarkerKeypoints.RIGHT_EYE, PoseLandmarkerKeypoints.RIGHT_EAR),

    "left_hand": (PoseLandmarkerKeypoints.LEFT_WRIST, PoseLandmarkerKeypoints.LEFT_PINKY),
    "right_hand": (PoseLandmarkerKeypoints.RIGHT_WRIST, PoseLandmarkerKeypoints.RIGHT_PINKY),
    "left_thumb_to_index": (PoseLandmarkerKeypoints.LEFT_THUMB, PoseLandmarkerKeypoints.LEFT_INDEX),
    "right_thumb_to_index": (PoseLandmarkerKeypoints.RIGHT_THUMB, PoseLandmarkerKeypoints.RIGHT_INDEX),

    "left_foot": (PoseLandmarkerKeypoints.LEFT_ANKLE, PoseLandmarkerKeypoints.LEFT_HEEL),
    "right_foot": (PoseLandmarkerKeypoints.RIGHT_ANKLE, PoseLandmarkerKeypoints.RIGHT_HEEL),
    "left_foot_to_index": (PoseLandmarkerKeypoints.LEFT_HEEL, PoseLandmarkerKeypoints.LEFT_FOOT_INDEX),
    "right_foot_to_index": (PoseLandmarkerKeypoints.RIGHT_HEEL, PoseLandmarkerKeypoints.RIGHT_FOOT_INDEX),

    "left_knee_to_ankle": (PoseLandmarkerKeypoints.LEFT_KNEE, PoseLandmarkerKeypoints.LEFT_ANKLE),
    "right_knee_to_ankle": (PoseLandmarkerKeypoints.RIGHT_KNEE, PoseLandmarkerKeypoints.RIGHT_ANKLE),
    "left_shoulder_to_ear": (PoseLandmarkerKeypoints.LEFT_SHOULDER, PoseLandmarkerKeypoints.LEFT_EAR),
    "right_shoulder_to_ear": (PoseLandmarkerKeypoints.RIGHT_SHOULDER, PoseLandmarkerKeypoints.RIGHT_EAR),
}

__POSE_LANDMARKER_COLORS = {
    "left_side_keypoint": (255, 128, 0),
    "right_side_keypoint": (0, 128, 255),
    "left_side_link": (255, 255, 255),
    "right_side_link": (255, 255, 255)
}

POSE_LANDMARKER_SKELETON = SkeletonDefinition(
    keypoints=PoseLandmarkerKeypoints,
    links=__POSE_LANDMARKER_SKELETON,
    colors=__POSE_LANDMARKER_COLORS
)
