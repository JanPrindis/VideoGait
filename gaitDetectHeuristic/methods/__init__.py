import os
import importlib

from utils.registry import HEURISTICS

methods_dir = os.path.dirname(__file__)

for file in os.listdir(methods_dir):
    if file.endswith(".py") and not file.startswith("_"):
        module_name = file[:-3]
        importlib.import_module(f".{module_name}", package=__name__)
