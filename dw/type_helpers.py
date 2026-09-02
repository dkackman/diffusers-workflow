import importlib

from .security import require_trusted_dotted_name


def get_type(module_name, type_name):
    module = __import__(module_name)
    return getattr(module, type_name)


def load_type_from_name(type_name):
    if "." in type_name:
        return load_type_from_full_name(type_name)

    return get_type("diffusers", type_name)


def load_type_from_full_name(full_name):
    # A bare name resolves against diffusers regardless of trust; a dotted
    # name imports whatever module it names, which is the code-execution
    # surface an untrusted workflow is refused unless it stays in-ecosystem
    require_trusted_dotted_name(full_name, "a dotted type reference")

    # Split the full name into module path and object name
    module_path, object_name = full_name.rsplit(".", 1)

    # Dynamically import the module
    module = importlib.import_module(module_path)

    # Get the object from the module
    return getattr(module, object_name)


def has_method(o, name):
    return callable(getattr(o, name, None))


def load_constant_from_name(name):
    """Load a constant declared in python, by its dotted name.

    The leading run of names that imports is the module the constant lives in and
    the rest are read from it, so a constant held in a dataclass is reachable
    ('...utils.GEMMA4_PROMPT_ENHANCEMENT_CONFIG.max_new_tokens') as well as one
    declared at module scope. A bare name is read from diffusers, matching the way
    a bare type reference resolves.

    Args:
        name: Dotted name of the constant

    Returns:
        The value the name refers to

    Raises:
        ImportError: If no leading part of the name names a module
        AttributeError: If the module has no such attribute
    """
    parts = name.split(".")

    module, attributes = None, parts
    for i in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
            attributes = parts[i:]
            break
        except ImportError:
            continue

    if module is None:
        # No dotted module path - a bare name, read from diffusers
        module = importlib.import_module("diffusers")

    value = module
    for attribute in attributes:
        value = getattr(value, attribute)
    return value
