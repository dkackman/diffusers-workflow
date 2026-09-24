import importlib
import inspect
import types

from .security import (
    TRUSTED_TOP_LEVEL_PACKAGES,
    UntrustedWorkflowError,
    require_trusted_dotted_name,
    workflows_are_trusted,
)


def get_type(module_name, type_name):
    module = __import__(module_name)
    return getattr(module, type_name)


def _accepts_dtype(key):
    return key is not None and (key == "dtype" or key.endswith("_dtype"))


def require_loadable_type(name, value, key=None):
    """Refuse a type reference that resolved to something other than a class,
    unless the workflow is trusted.

    A '*_type' value is constructed with the workflow's own arguments, so an
    allowlisted package is only safe if what the name reaches is a class:
    'torch.hub.load' is in 'torch' and runs a GitHub repo's code when called.
    A 'dtype' or '*_dtype' key names a torch.dtype, which is data, not a
    class, and is accepted there.

    Raises:
        UntrustedWorkflowError: If untrusted and `value` is neither a class
            nor, under a dtype key, a torch.dtype
    """
    if workflows_are_trusted() or inspect.isclass(value):
        return value
    if _accepts_dtype(key):
        import torch

        if isinstance(value, torch.dtype):
            return value

    kind = "module" if isinstance(value, types.ModuleType) else type(value).__name__
    allowed = "a class or a torch.dtype" if _accepts_dtype(key) else "a class"
    raise UntrustedWorkflowError(
        f"Refusing to load {key or 'a type reference'} '{name}': it is a {kind}, "
        f"not a class. An untrusted workflow's type reference must name "
        f"{allowed} - anything else could be called with the workflow's "
        f"arguments. Pass --trust-workflows if you trust this workflow's source."
    )


def load_type_from_name(type_name, key=None):
    if "." in type_name:
        return load_type_from_full_name(type_name, key)

    return require_loadable_type(type_name, get_type("diffusers", type_name), key)


def load_type_from_full_name(full_name, key=None):
    # A bare name resolves against diffusers regardless of trust; a dotted
    # name imports whatever module it names, which is the code-execution
    # surface an untrusted workflow is refused unless it stays in-ecosystem
    require_trusted_dotted_name(full_name, "a dotted type reference")

    # Split the full name into module path and object name
    module_path, object_name = full_name.rsplit(".", 1)

    # Dynamically import the module
    module = importlib.import_module(module_path)

    # Get the object from the module
    return require_loadable_type(full_name, getattr(module, object_name), key)


def has_method(o, name):
    return callable(getattr(o, name, None))


def _require_walk_stays_inside(name, parts, value, index):
    """Refuse a constant walk that leaves the allowed packages.

    The allowlist is checked on the name's top-level package, but a module
    re-exports what it imported: 'torch.os.environ' starts in torch and ends
    in the server's environment. So every module the walk passes through must
    itself be in an allowed package, and no segment may be private.
    """
    if parts[index].startswith("_"):
        raise UntrustedWorkflowError(
            f"Refusing the constant '{name}': '{parts[index]}' is a private "
            f"name, and an untrusted workflow may only read public ones. "
            f"Pass --trust-workflows if you trust this workflow's source."
        )
    if isinstance(value, types.ModuleType):
        top_level = value.__name__.split(".", 1)[0]
        if top_level not in TRUSTED_TOP_LEVEL_PACKAGES:
            raise UntrustedWorkflowError(
                f"Refusing the constant '{name}': "
                f"'{'.'.join(parts[: index + 1])}' is the '{value.__name__}' "
                f"module, which is outside the ecosystem "
                f"({', '.join(TRUSTED_TOP_LEVEL_PACKAGES)}) this workflow is "
                f"allowed to reach untrusted. Pass --trust-workflows if you "
                f"trust this workflow's source."
            )


def load_constant_from_name(name):
    """Load a constant declared in python, by its dotted name.

    The leading run of names that imports is the module the constant lives in and
    the rest are read from it, so a constant held in a dataclass is reachable
    ('...utils.GEMMA4_PROMPT_ENHANCEMENT_CONFIG.max_new_tokens') as well as one
    declared at module scope. A bare name is read from diffusers, matching the way
    a bare type reference resolves.

    Untrusted, the walk may not pass through a private name or a module outside
    TRUSTED_TOP_LEVEL_PACKAGES.

    Args:
        name: Dotted name of the constant

    Returns:
        The value the name refers to

    Raises:
        ImportError: If no leading part of the name names a module
        AttributeError: If the module has no such attribute
        UntrustedWorkflowError: If untrusted and the walk leaves the allowed
            packages or reads a private name
    """
    # A dotted constant imports the module it names before anything reads
    # the attribute - the same code-execution surface as a dotted type
    parts = name.split(".")
    guarded = not workflows_are_trusted()
    if "." in name:
        require_trusted_dotted_name(name, "a constant: reference")
    if guarded:
        # Checked before anything imports: a private module's import runs
        # its code whatever the walk would have read from it
        for index, part in enumerate(parts):
            if part.startswith("_"):
                _require_walk_stays_inside(name, parts, None, index)

    module, attributes, start = None, parts, 0
    for i in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
            attributes, start = parts[i:], i
            break
        except ImportError:
            continue

    if module is None:
        # No dotted module path - a bare name, read from diffusers
        module = importlib.import_module("diffusers")

    value = module
    for offset, attribute in enumerate(attributes):
        value = getattr(value, attribute)
        if guarded:
            _require_walk_stays_inside(name, parts, value, start + offset)
    return value
