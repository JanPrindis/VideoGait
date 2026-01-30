from dataclasses import dataclass
from typing import Dict, Tuple, Union, Type, List
from enum import IntEnum

@dataclass
class SkeletonDefinition:
    keypoints: Type[IntEnum]
    links: Dict[str, Tuple[Union[int, IntEnum], Union[int, IntEnum]]]
    colors: Dict[str, Tuple[int, int, int]]

    def get_adjacency_list(self, active_keypoints: List[str]) -> List[Tuple[int, int]]:

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
