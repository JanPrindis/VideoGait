"""
This module initializes the model registry by dynamically importing all model architectures.

It scans the 'architectures' subdirectory and imports every Python file found there.
This ensures that all model classes decorated with @MODELS.register_module() are
automatically added to the registry when this package is imported.
"""
import os
import importlib

from gaitDetectNN.utils.registry import MODELS

arch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architectures")
for file in os.listdir(arch_dir):
    if file.endswith(".py") and not file.startswith("_"):
        module_name = file[:-3]
        importlib.import_module(f".architectures.{module_name}", package=__name__)
