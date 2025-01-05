import importlib


def get_type(module_name, type_name):
    module = __import__(module_name)
    return getattr(module, type_name)


def load_type_from_name(type_name):
    if "." in type_name:
        return load_type_from_full_name(type_name)

    return get_type("diffusers", type_name)


def load_type_from_full_name(full_name):
    # Split the full name into module path and object name
    module_path, object_name = full_name.rsplit(".", 1)

    # Dynamically import the module
    module = importlib.import_module(module_path)

    # Get the object from the module
    return getattr(module, object_name)


def has_method(o, name):
    return callable(getattr(o, name, None))
