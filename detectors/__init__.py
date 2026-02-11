"""
This module initializes the detector registry by dynamically importing all detector wrappers.

It scans the subdirectories in the 'detectors' folder. For each subdirectory, it attempts to:
1. Import a 'wrapper.py' module if it exists. This is the standard convention for
   registering a detector class (e.g., @POSE_DETECTORS.register).
2. Fallback to importing the package itself (via __init__.py) if 'wrapper.py' is missing.

This ensures that all available pose detectors are registered and available for use
via the builder without manual import statements.
"""
import os
import importlib
from utils.registry import POSE_DETECTORS
from utils.logger import log

detectors_dir = os.path.dirname(os.path.abspath(__file__))

for item in os.listdir(detectors_dir):
    path = os.path.join(detectors_dir, item)

    if os.path.isdir(path) and not item.startswith("__") and item != "utils":

        # Find wrapper.py
        if os.path.isfile(os.path.join(path, "wrapper.py")):
            try:
                importlib.import_module(f".{item}.wrapper", package=__name__)

            except Exception as e:
                log("REGISTRY", f"Failed to load detector wrapper '{item}': {e}", level="error")
                import traceback

                traceback.print_exc()

        # Fallback: if wrapper.py does not exist inside the detector package, try to import __init__.py
        elif os.path.isfile(os.path.join(path, "__init__.py")):
            try:
                importlib.import_module(f".{item}", package=__name__)
            except Exception as e:
                log("REGISTRY", f"Failed to load detector package '{item}': {e}", level="error")
