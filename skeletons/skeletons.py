"""
This module defines the core data structures for representing skeleton topologies.

It provides classes to define keypoints, connections (links), and helper methods
to determine the laterality (left/right/center) of joints and limbs.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union, Type, List, Set
from enum import IntEnum

class SkeletonSide(IntEnum):
    """
    Enum representing the side of the body a keypoint or link belongs to.
    """
    CENTER = 0
    LEFT = 1
    RIGHT = 2

@dataclass
class SkeletonDefinition:
    """
    Defines the structure of a skeleton, including keypoints and their connections.

    Attributes:
        keypoints (Type[IntEnum]): An IntEnum class defining the keypoint names and indices.
        links (Dict[str, Tuple]): A dictionary defining the bones (connections) between keypoints.
                                  Key is a descriptive name, value is a tuple of (start_kp, end_kp).
        left_keypoints (Set[int]): A set of indices corresponding to keypoints on the left side.
        right_keypoints (Set[int]): A set of indices corresponding to keypoints on the right side.
    """
    keypoints: Type[IntEnum]
    links: Dict[str, Tuple[Union[int, IntEnum], Union[int, IntEnum]]]

    left_keypoints: Set[int] = field(default_factory=set)
    right_keypoints: Set[int] = field(default_factory=set)

    def get_adjacency_list(self, active_keypoints: List[str]) -> List[Tuple[int, int]]:
        """
        Generates an adjacency list representing the skeleton's connections based on a list of active keypoints.

        This method filters the defined links to include only those where both endpoints
        are present in `active_keypoints`. It also injects virtual connections for
        'HIP' and 'NECK' if they are present but not explicitly linked in the definition
        (e.g., connecting hips to a central hip point).

        Args:
            active_keypoints (List[str]): A list of keypoint names currently present/detected.

        Returns:
            List[Tuple[int, int]]: A list of tuples (idx1, idx2) representing indices in `active_keypoints`.
        """

        adj_list = []
        kp_to_idx = {name: i for i, name in enumerate(active_keypoints)}

        # Skeleton definition links
        for p1, p2 in self.links.values():
            n1 = p1.name if isinstance(p1, IntEnum) else str(p1)
            n2 = p2.name if isinstance(p2, IntEnum) else str(p2)

            if n1 in kp_to_idx and n2 in kp_to_idx:
                adj_list.append((kp_to_idx[n1], kp_to_idx[n2]))

        # Injected HIP check (link with LEFT_HIP and RIGHT_HIP)
        if "HIP" in kp_to_idx:
            for side in ["LEFT_HIP", "RIGHT_HIP"]:
                if side in kp_to_idx:
                    pair = (kp_to_idx[side], kp_to_idx["HIP"])
                    if pair not in adj_list and (pair[1], pair[0]) not in adj_list:
                        adj_list.append(pair)

        # Injected NECK check (link with LEFT_SHOULDER and RIGHT_SHOULDER)
        if "NECK" in kp_to_idx:
            for side in ["LEFT_SHOULDER", "RIGHT_SHOULDER"]:
                if side in kp_to_idx:
                    pair = (kp_to_idx[side], kp_to_idx["NECK"])
                    if pair not in adj_list and (pair[1], pair[0]) not in adj_list:
                        adj_list.append(pair)

        return adj_list

    def get_keypoint_side(self, kp_idx: int) -> SkeletonSide:
        """
        Determines the side (Left, Right, Center) of a specific keypoint index.

        Args:
            kp_idx (int): The integer index (value) of the keypoint from the Enum.

        Returns:
            SkeletonSide: The side the keypoint belongs to.
        """
        if kp_idx in self.left_keypoints:
            return SkeletonSide.LEFT

        if kp_idx in self.right_keypoints:
            return SkeletonSide.RIGHT

        return SkeletonSide.CENTER

    def get_link_side(self, kp1_val: int, kp2_val: int) -> SkeletonSide:
        """
        Determines the side of a link (bone) connecting two keypoints.

        Args:
            kp1_val (int): The integer index of the first keypoint.
            kp2_val (int): The integer index of the second keypoint.

        Returns:
            SkeletonSide: LEFT if mostly left, RIGHT if mostly right, CENTER otherwise.
        """
        is_k1_left = kp1_val in self.left_keypoints
        is_k2_left = kp2_val in self.left_keypoints

        is_k1_right = kp1_val in self.right_keypoints
        is_k2_right = kp2_val in self.right_keypoints

        # (L-R) -> Center Color
        if (is_k1_left and is_k2_right) or (is_k1_right and is_k2_left):
            return SkeletonSide.CENTER

        # (L-L or L-C) -> Side Color (Left)
        if (is_k1_left or is_k2_left) and not (is_k1_right or is_k2_right):
            return SkeletonSide.LEFT

        # (R-R or R-C) -> Side Color (Right)
        if (is_k1_right or is_k2_right) and not (is_k1_left or is_k2_left):
            return SkeletonSide.RIGHT

        # (C-C) -> Center Color
        return SkeletonSide.CENTER
