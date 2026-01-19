"""
This module serves as a registry for supported skeleton definitions.

It imports specific skeleton configurations (like HALPE, COCO, POSE_LANDMARKER)
and provides a unified interface to retrieve them by name.
"""
from .halpe_skeleton import HALPE_SKELETON
from .coco_skeleton import COCO_SKELETON
from .pose_landmarker_skeleton import POSE_LANDMARKER_SKELETON

_SKELETON_REGISTRY = {
    "HALPE": HALPE_SKELETON,
    "COCO": COCO_SKELETON,
    "POSE_LANDMARKER": POSE_LANDMARKER_SKELETON
}

def get_skeleton_by_name(name: str):
    """
    Retrieves a skeleton definition object by its string name.

    Args:
        name (str): The name of the skeleton (e.g., "HALPE", "COCO").
                    Case-sensitive based on the registry keys.

    Returns:
        SkeletonDefinition: The requested skeleton definition object containing
                            keypoints, links, and colors.

    Raises:
        ValueError: If the provided name is not found in the registry.
    """
    if name not in _SKELETON_REGISTRY:
        available = list(_SKELETON_REGISTRY.keys())
        raise ValueError(f"Skeleton '{name}' not found. Available: {available}")
    return _SKELETON_REGISTRY[name]
