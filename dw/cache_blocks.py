"""Register transformer block metadata diffusers' cache hooks are missing.

The first_block, mag and layer_skip cache types look a model's transformer block
class up in diffusers.hooks._helpers.TransformerBlockRegistry and raise
ValueError when it is absent, so those caches are unavailable on any model
diffusers has not registered upstream - even though the model itself supports
enable_cache(). The blocks in cache_blocks.json fill that gap; entries become
redundant, not wrong, once diffusers registers the same class itself.
"""

import json
import logging
from pathlib import Path

from .type_helpers import load_type_from_name

logger = logging.getLogger("dw")


_REGISTRY_PATH = Path(__file__).parent / "cache_blocks.json"

# Registration walks every entry, so do it once rather than per pipeline load
_registered = False


def _load_registry():
    """Load the block metadata registry from the JSON file."""
    with open(_REGISTRY_PATH) as f:
        return json.load(f).get("blocks", {})


def register_cache_blocks():
    """Register any known-missing transformer blocks with diffusers.

    Safe to call repeatedly and before any cache type - blocks diffusers already
    knows are left alone, and a block whose class the installed diffusers does
    not have is skipped rather than raising.
    """
    global _registered
    if _registered:
        return

    try:
        from diffusers.hooks._helpers import (
            TransformerBlockMetadata,
            TransformerBlockRegistry,
        )
    except ImportError:
        # A diffusers without the cache hooks has nothing to register against
        logger.debug(
            "diffusers cache hook helpers unavailable, skipping block registration"
        )
        _registered = True
        return

    for class_name, metadata in _load_registry().items():
        try:
            block_class = load_type_from_name(class_name)
        except (ImportError, AttributeError):
            # The installed diffusers predates this model - its blocks are not
            # missing from the registry, they do not exist
            logger.debug(f"Cache block '{class_name}' not in this diffusers, skipping")
            continue

        # TransformerBlockRegistry.get raises rather than returning None, and an
        # upstream registration is the more authoritative one either way
        try:
            TransformerBlockRegistry.get(block_class)
            logger.debug(f"Cache block '{class_name}' already registered, leaving it")
            continue
        except ValueError:
            pass

        logger.debug(f"Registering cache block metadata for '{class_name}'")
        TransformerBlockRegistry.register(
            model_class=block_class,
            metadata=TransformerBlockMetadata(**metadata),
        )

    _registered = True
