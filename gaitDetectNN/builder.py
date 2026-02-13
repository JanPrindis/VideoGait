"""
This module provides utility functions for building model instances from configuration dictionaries.

It leverages the model registry to dynamically instantiate classes based on string names
defined in the configuration.
"""
import torch

from utils.config_models import ModelDefinition
from utils.registry import MODELS
import gaitDetectNN.models


def build_model(cfg: ModelDefinition, input_size: int):
    """
    Instantiates a model from the registry based on the provided configuration.

    Args:
        cfg (dict): A dictionary containing the model configuration.
                    Must contain a "type" key matching a registered model name.
                    May contain a "params" key with keyword arguments for the model constructor.
        input_size (int): The size of the input feature vector (number of input channels).

    Returns:
        torch.nn.Module: The initialized model instance.
    """
    model_type = cfg.type
    model_params = cfg.params
    model_class = MODELS.get(model_type)
    if not model_class:
        raise ValueError(f"Model '{model_type}' not found in registry!")

    model = model_class(input_size=input_size, **model_params)
    return model
