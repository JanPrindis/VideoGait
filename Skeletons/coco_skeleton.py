from enum import IntEnum
from Skeletons.skeletons import SkeletonDefinition

class CocoKeypoints(IntEnum):
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


__COCO_NAMED_SKELETON = {
    "left_upper_leg": (CocoKeypoints.LEFT_HIP, CocoKeypoints.LEFT_KNEE),
    "left_lower_leg": (CocoKeypoints.LEFT_KNEE, CocoKeypoints.LEFT_ANKLE),
    "right_upper_leg": (CocoKeypoints.RIGHT_HIP, CocoKeypoints.RIGHT_KNEE),
    "right_lower_leg": (CocoKeypoints.RIGHT_KNEE, CocoKeypoints.RIGHT_ANKLE),

    "left_upper_arm": (CocoKeypoints.LEFT_SHOULDER, CocoKeypoints.LEFT_ELBOW),
    "left_lower_arm": (CocoKeypoints.LEFT_ELBOW, CocoKeypoints.LEFT_WRIST),
    "right_upper_arm": (CocoKeypoints.RIGHT_SHOULDER, CocoKeypoints.RIGHT_ELBOW),
    "right_lower_arm": (CocoKeypoints.RIGHT_ELBOW, CocoKeypoints.RIGHT_WRIST),

    "shoulders": (CocoKeypoints.LEFT_SHOULDER, CocoKeypoints.RIGHT_SHOULDER),
    "hips": (CocoKeypoints.LEFT_HIP, CocoKeypoints.RIGHT_HIP),
    "torso_left": (CocoKeypoints.LEFT_SHOULDER, CocoKeypoints.LEFT_HIP),
    "torso_right": (CocoKeypoints.RIGHT_SHOULDER, CocoKeypoints.RIGHT_HIP),

    "nose_to_left_eye": (CocoKeypoints.NOSE, CocoKeypoints.LEFT_EYE),
    "nose_to_right_eye": (CocoKeypoints.NOSE, CocoKeypoints.RIGHT_EYE),
    "eye_connection": (CocoKeypoints.LEFT_EYE, CocoKeypoints.RIGHT_EYE),
    "left_eye_to_ear": (CocoKeypoints.LEFT_EYE, CocoKeypoints.LEFT_EAR),
    "right_eye_to_ear": (CocoKeypoints.RIGHT_EYE, CocoKeypoints.RIGHT_EAR),
}


__COCO_COLORS = {
    "left_side_keypoint": (255, 128, 0),
    "right_side_keypoint": (0, 128, 255),
    "left_side_link": (255, 255, 255),
    "right_side_link": (255, 255, 255)
}


COCO_SKELETON = SkeletonDefinition(
    keypoints=CocoKeypoints,
    links=__COCO_NAMED_SKELETON,
    colors=__COCO_COLORS
)