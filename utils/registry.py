"""
This module implements a simple registry pattern for managing model architectures.

It allows decoupling the instantiation of classes from their definition by mapping
string names to class objects. This is used to dynamically build models from configuration files.
"""

class Registry:
    """
    A generic registry to map string names to classes.

    This is typically used to register model architectures or other components
    so they can be instantiated by name from a configuration file.
    """
    def __init__(self, name):
        """
        Initializes the registry.

        Args:
            name (str): The name of the registry (e.g., 'models'). Used for error messages.
        """
        self._name = name
        self._module_dict = {}

    def register(self, cls):
        """
        Decorator to register a class.

        Args:
            cls (type): The class to register.

        Returns:
            type: The same class, allowing this method to be used as a decorator.
        """
        self._module_dict[cls.__name__] = cls
        return cls

    def get(self, name):
        """
        Retrieves a registered class by name.

        Args:
            name (str): The name of the class to retrieve.

        Returns:
            type: The registered class.

        Raises:
            KeyError: If the name is not found in the registry.
        """
        if name not in self._module_dict:
            raise KeyError(f"'{name}' is not registered in {self._name}")
        return self._module_dict[name]


MODELS = Registry("models")
HEURISTICS = Registry("heuristics")
POSE_DETECTORS = Registry("pose_detectors")
