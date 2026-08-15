import logging

logger = logging.getLogger("dw")

# Loaded task models, keyed by whatever identifies a load - typically
# (task, model_name, device). step.py runs a task handler once per cartesian
# product iteration; without this, segmenting 20 images would load the same
# multi-gigabyte checkpoints 20 times over
_cache = {}


def cached_model(key, factory):
    """Return the model for key, loading it with factory() on first use.

    Args:
        key: Hashable identity of the load - include the model name and the
            device, plus anything else that changes what factory() builds
        factory: Zero-argument callable performing the actual load

    Returns:
        The cached or freshly loaded model
    """
    if key not in _cache:
        logger.info(f"Loading task model: {key}")
        _cache[key] = factory()
    else:
        logger.debug(f"Reusing cached task model: {key}")
    return _cache[key]


def clear_model_cache():
    """Release every cached task model.

    Wired into the worker's memory cleanup - dropping the references here is
    what lets gc and the allocator actually reclaim the weights.
    """
    if _cache:
        logger.info(f"Clearing {len(_cache)} cached task models")
    _cache.clear()
