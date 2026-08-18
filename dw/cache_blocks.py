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

# The registry field that is dw's own rather than one of TransformerBlockMetadata's -
# see _dual_stream_metadata_class for what it is for
ARGUMENT_REMAP_KEY = "encoder_hidden_states_argument_name"

# One subclass per metadata class, built on first use - the class it derives from
# only exists once diffusers has been imported
_dual_stream_classes = {}


def _load_registry():
    """Load the block metadata registry from the JSON file."""
    with open(_REGISTRY_PATH) as f:
        return json.load(f).get("blocks", {})


def _dual_stream_metadata_class(metadata_class):
    """A metadata class that can name the block's second stream itself.

    A block that returns two streams has the second one read back out of its
    forward arguments when the cache skips it, and diffusers looks that argument
    up under the fixed name 'encoder_hidden_states' - the only two-stream shape it
    registers upstream is text beside image. LTX-2's blocks return video beside
    *audio* while also taking an 'encoder_hidden_states' of their own (the text
    conditioning), so the fixed name silently reads the wrong tensor and feeds the
    text embeddings back as the audio stream. Remapping the identifier is what
    makes first_block caching correct on those blocks rather than merely quiet.

    Args:
        metadata_class: diffusers' TransformerBlockMetadata

    Returns:
        A subclass reading the second stream from the argument the entry names
    """
    if metadata_class not in _dual_stream_classes:

        class DualStreamMetadata(metadata_class):
            # Overridden per instance from the registry entry; the default keeps
            # the class behaving exactly like the one it derives from
            encoder_hidden_states_argument_name = "encoder_hidden_states"

            def _get_parameter_from_args_kwargs(self, identifier, args=(), kwargs=None):
                if identifier == "encoder_hidden_states":
                    identifier = self.encoder_hidden_states_argument_name
                return super()._get_parameter_from_args_kwargs(identifier, args, kwargs)

        _dual_stream_classes[metadata_class] = DualStreamMetadata

    return _dual_stream_classes[metadata_class]


def build_metadata(metadata_class, entry):
    """Build the metadata object one registry entry describes.

    Args:
        metadata_class: diffusers' TransformerBlockMetadata
        entry: One block's fields from cache_blocks.json

    Returns:
        An instance of metadata_class, or of the dual-stream subclass when the
        entry names the argument its second returned stream comes from
    """
    remapped = entry.get(ARGUMENT_REMAP_KEY)
    fields = {k: v for k, v in entry.items() if k != ARGUMENT_REMAP_KEY}

    if remapped is None:
        return metadata_class(**fields)

    metadata = _dual_stream_metadata_class(metadata_class)(**fields)
    metadata.encoder_hidden_states_argument_name = remapped
    return metadata


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
            metadata=build_metadata(TransformerBlockMetadata, metadata),
        )

    _registered = True
