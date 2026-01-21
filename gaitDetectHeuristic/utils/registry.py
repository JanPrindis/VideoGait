class Registry:
    """
    A generic registry to map string names to classes.
    """
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, cls):
        self._module_dict[cls.__name__] = cls
        return cls

    def get(self, name):
        if name not in self._module_dict:
            raise KeyError(f"'{name}' is not registered in {self._name}. Available: {list(self._module_dict.keys())}")
        return self._module_dict[name]

HEURISTICS = Registry("heuristics")
