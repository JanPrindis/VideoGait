from enum import IntEnum
from Skeletons.skeletons import SkeletonDefinition

class HalpeKeypoints(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    HEAD = 17
    NECK = 18
    HIP = 19
    LEFT_FOOT_INDEX = 20
    RIGHT_FOOT_INDEX = 21
    LEFT_FOOT_PINKY = 22
    RIGHT_FOOT_PINKY = 23
    LEFT_HEEL = 24
    RIGHT_HEEL = 25

__HALPE_SKELETON = {
    "left_upper_leg": (HalpeKeypoints.LEFT_HIP, HalpeKeypoints.LEFT_KNEE),
    "left_lower_leg": (HalpeKeypoints.LEFT_KNEE, HalpeKeypoints.LEFT_ANKLE),
    "right_upper_leg": (HalpeKeypoints.RIGHT_HIP, HalpeKeypoints.RIGHT_KNEE),
    "right_lower_leg": (HalpeKeypoints.RIGHT_KNEE, HalpeKeypoints.RIGHT_ANKLE),

    "left_upper_arm": (HalpeKeypoints.LEFT_SHOULDER, HalpeKeypoints.LEFT_ELBOW),
    "left_lower_arm": (HalpeKeypoints.LEFT_ELBOW, HalpeKeypoints.LEFT_WRIST),
    "right_upper_arm": (HalpeKeypoints.RIGHT_SHOULDER, HalpeKeypoints.RIGHT_ELBOW),
    "right_lower_arm": (HalpeKeypoints.RIGHT_ELBOW, HalpeKeypoints.RIGHT_WRIST),

    "shoulders": (HalpeKeypoints.LEFT_SHOULDER, HalpeKeypoints.RIGHT_SHOULDER),
    "hips": (HalpeKeypoints.LEFT_HIP, HalpeKeypoints.RIGHT_HIP),
    "torso_left": (HalpeKeypoints.LEFT_SHOULDER, HalpeKeypoints.LEFT_HIP),
    "torso_right": (HalpeKeypoints.RIGHT_SHOULDER, HalpeKeypoints.RIGHT_HIP),

    "nose_to_left_eye": (HalpeKeypoints.NOSE, HalpeKeypoints.LEFT_EYE),
    "nose_to_right_eye": (HalpeKeypoints.NOSE, HalpeKeypoints.RIGHT_EYE),
    "eye_connection": (HalpeKeypoints.LEFT_EYE, HalpeKeypoints.RIGHT_EYE),
    "left_eye_to_ear": (HalpeKeypoints.LEFT_EYE, HalpeKeypoints.LEFT_EAR),
    "right_eye_to_ear": (HalpeKeypoints.RIGHT_EYE, HalpeKeypoints.RIGHT_EAR),

    "head_to_neck": (HalpeKeypoints.HEAD, HalpeKeypoints.NECK),
    "left_shoulder_to_neck": (HalpeKeypoints.LEFT_SHOULDER, HalpeKeypoints.NECK),
    "right_shoulder_to_neck": (HalpeKeypoints.RIGHT_SHOULDER, HalpeKeypoints.NECK),

    "left_foot": (HalpeKeypoints.LEFT_ANKLE, HalpeKeypoints.LEFT_HEEL),
    "right_foot": (HalpeKeypoints.RIGHT_ANKLE, HalpeKeypoints.RIGHT_HEEL),
    "left_foot_to_index": (HalpeKeypoints.LEFT_HEEL, HalpeKeypoints.LEFT_FOOT_INDEX),
    "right_foot_to_index": (HalpeKeypoints.RIGHT_HEEL, HalpeKeypoints.RIGHT_FOOT_INDEX),

    "left_knee_to_ankle": (HalpeKeypoints.LEFT_KNEE, HalpeKeypoints.LEFT_ANKLE),
    "right_knee_to_ankle": (HalpeKeypoints.RIGHT_KNEE, HalpeKeypoints.RIGHT_ANKLE),

    "left_shoulder_to_ear": (HalpeKeypoints.LEFT_SHOULDER, HalpeKeypoints.LEFT_EAR),
    "right_shoulder_to_ear": (HalpeKeypoints.RIGHT_SHOULDER, HalpeKeypoints.RIGHT_EAR),
}

__HALPE_COLORS = {
    "left_side_keypoint": (255, 128, 0),
    "right_side_keypoint": (0, 128, 255),
    "left_side_link": (255, 255, 255),
    "right_side_link": (255, 255, 255)
}

HALPE_SKELETON = SkeletonDefinition(
    keypoints=HalpeKeypoints,
    links=__HALPE_SKELETON,
    colors=__HALPE_COLORS
)