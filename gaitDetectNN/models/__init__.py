"""
This module initializes the model registry by dynamically importing all model architectures.

It scans the directory containing this file and imports all Python modules found.
This ensures that all model classes decorated with @MODELS.register are
automatically added to the registry when this package is imported.
"""
import os
import importlib

from utils.registry import MODELS

arch_dir = os.path.dirname(__file__)

for file in os.listdir(arch_dir):
    if file.endswith(".py") and not file.startswith("_"):
        module_name = file[:-3]
        importlib.import_module(f".{module_name}", package=__name__)
