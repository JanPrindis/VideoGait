"""
This module serves as a registry for supported skeleton definitions.

It imports specific skeleton configurations (like HALPE, COCO, POSE_LANDMARKER)
and provides a unified interface to retrieve them by name.
"""
import os
import pkgutil
import importlib
import inspect
from .skeletons import SkeletonDefinition

_SKELETON_REGISTRY = {}


def _discover_and_register_skeletons():
    package_path = os.path.dirname(__file__)
    package_name = __name__

    # Iterate over all modules in the package
    for _, module_name, _ in pkgutil.iter_modules([package_path]):
        # Skip init file and base skeleton definition (skeletons.py),
        if module_name == "skeletons":
            continue

        try:
            # Dynamic module import
            full_module_name = f"{package_name}.{module_name}"
            module = importlib.import_module(full_module_name)

            # Integrate over all members in the module
            for name, value in inspect.getmembers(module):

                # Filter out only SkeletonDefinition classes
                if isinstance(value, SkeletonDefinition):
                    # Key logic: HALPE_SKELETON -> HALPE
                    # Remove suffix and convert to uppercase
                    key_name = name.upper().replace("_SKELETON", "")

                    _SKELETON_REGISTRY[key_name] = value
                    # print(f"[SkeletonRegistry] Loaded '{key_name}' from {module_name}")

        except Exception as e:
            print(f"[SkeletonRegistry] Error loading module {module_name}: {e}")


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
    # Normalize name to uppercase
    name_upper = name.upper()

    if name_upper not in _SKELETON_REGISTRY:
        available = list(_SKELETON_REGISTRY.keys())
        raise ValueError(f"Skeleton '{name}' not found. Available: {available}")
    return _SKELETON_REGISTRY[name_upper]
