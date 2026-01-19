from dataclasses import dataclass
from typing import Dict, Tuple, Union, Type
from enum import IntEnum

@dataclass
class SkeletonDefinition:
    keypoints: Type[IntEnum]
    links: Dict[str, Tuple[Union[int, IntEnum], Union[int, IntEnum]]]
    colors: Dict[str, Tuple[int, int, int]]

