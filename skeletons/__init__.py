"""
This module initializes the skeleton registry by dynamically importing all skeleton definitions.

It scans the directory containing this file and imports all Python modules found.
It inspects each module for instances of `SkeletonDefinition` and registers them.
This allows new skeletons to be added simply by creating a new file in this directory.
"""
import os
import pkgutil
import importlib
import inspect
from .skeletons import SkeletonDefinition

_SKELETON_REGISTRY = {}


def _discover_and_register_skeletons():
    package_path = os.path.dirname(os.path.abspath(__file__))
    package_name = __name__

    # print(f"[DEBUG] Scanning for skeletons in: {package_path}")

    # Iterate over all modules in the package
    for _, module_name, _ in pkgutil.iter_modules([package_path]):
        # print(f"[SkeletonRegistry] Loading {module_name}")

        # Skip init file and base skeleton definition (skeletons.py),
        if module_name == "skeletons" or module_name.startswith("__"):
            continue
        try:
            # Dynamic module import
            module = importlib.import_module(f".{module_name}", package=package_name)

            # Integrate over all members in the module
            for name, value in inspect.getmembers(module):
                # Filter out only SkeletonDefinition classes
                if isinstance(value, SkeletonDefinition):
                    # Key logic: HALPE_SKELETON -> HALPE
                    # Remove suffix and convert to uppercase
                    key_name = name.upper().replace("_SKELETON", "")

                    _SKELETON_REGISTRY[key_name] = value

        except Exception as e:
            print(f"[SkeletonRegistry] Error loading module {module_name}: {e}")

_discover_and_register_skeletons()

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
