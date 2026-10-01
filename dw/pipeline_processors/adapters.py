import torch
import logging
from ..events import emit_phase

logger = logging.getLogger("dw")


def set_adapter_alpha(pipeline, adapter_name, alpha):
    """Override the network alpha an adapter was loaded with.

    peft scales an adapter by `scale * alpha / rank`, and the alpha comes from
    the checkpoint: a per-module `.alpha` tensor, else a `__metadata__` alpha
    where the loader honors one, else the rank itself. That is the right
    default, and for a file that records the wrong figure (or none) it is
    wrong. Check the header before overriding: the MiniMax-H3 turbo LoRAs each
    record the alpha they were trained at, and upstream's `--lora-alpha 128`
    matches the one 768p file that records 128 - stating it for the 8-step
    768p file, which records 8, ran that one at sixteen times its trained
    strength. The number is a property of the model, so the workflow states
    it rather than the engine guessing.

    Set before the caller's set_adapters(), which is what recomputes each
    layer's scaling from the alpha found here.

    Args:
        pipeline: The loaded pipeline
        adapter_name: Which adapter's alpha to override
        alpha: The network alpha to use

    Raises:
        ValueError: if alpha is not positive, or no loaded layer carries the
            adapter - an alpha that silently applied to nothing is the quality
            failure it exists to prevent
    """
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError(f"A lora 'alpha' must be positive, got {alpha}.")

    touched = 0
    ranks = set()
    for module in _lora_layers(pipeline):
        if adapter_name in getattr(module, "lora_alpha", {}):
            module.lora_alpha[adapter_name] = alpha
            ranks.add(module.r[adapter_name])
            touched += 1

    if not touched:
        raise ValueError(
            f"Cannot set alpha {alpha} on adapter '{adapter_name}' - no loaded "
            "layer carries it. Nothing would have been scaled."
        )

    # The figure upstream's own runner prints, because alpha alone says
    # nothing without the rank it is divided by
    effective = sorted(alpha / rank for rank in ranks)
    logger.info(
        f"Adapter '{adapter_name}': alpha {alpha} over rank(s) "
        f"{sorted(ranks)} - scaling {effective[0]:.6g}"
        + (f" to {effective[-1]:.6g}" if len(effective) > 1 else "")
        + " before the adapter weight"
    )


def _lora_layers(pipeline):
    """Every peft-wrapped module under a pipeline, whatever holds it.

    A pipeline names the components a LoRA can reach; a modular one holds them
    in `components` instead. Walking both and de-duplicating by identity is
    what keeps this working for MiniMax-H3, whose two DiT partitions are
    separate components, without enumerating model names here.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    seen = set()
    holders = []
    for name in getattr(pipeline, "_lora_loadable_modules", []) or []:
        holders.append(getattr(pipeline, name, None))
    if not any(holder is not None for holder in holders):
        components = getattr(pipeline, "components", None)
        if isinstance(components, dict):
            holders.extend(components.values())

    for holder in holders:
        if not isinstance(holder, torch.nn.Module):
            continue
        for module in holder.modules():
            if isinstance(module, BaseTunerLayer) and id(module) not in seen:
                seen.add(id(module))
                yield module


def active_loras(loras):
    """The `loras` entries that will load: a null `model_name` switches one
    off, which is how a caller runs a template's step without its adapter -
    the list itself is fixed JSON, and a variable can null a value but not
    remove an entry."""
    return [
        lora
        for lora in loras or []
        if not isinstance(lora, dict) or lora.get("model_name") is not None
    ]


def load_loras(loras, pipeline):
    """Load and configure LoRA models."""
    adapter_names = []
    adapter_weights = []
    alphas = {}

    for i, lora in enumerate(loras or []):
        if isinstance(lora, dict) and lora.get("model_name") is None:
            # Switched off - said to the caller by warn_adapters before the
            # run started, so only logged here
            logger.info(f"LoRA {i} has a null model_name - not loaded")
            continue
        model_name = lora.pop("model_name", None)
        logger.info(f"Loading LoRA: {model_name}")
        emit_phase("loading", detail=f"LoRA: {model_name}")

        # Use provided adapter_name or generate from index - `or`, because a
        # variable nulled by the caller arrives as a present None
        adapter_name = lora.pop("adapter_name", None) or str(i)
        adapter_names.append(adapter_name)

        # Extract scale for adapter weights - float() because the schema takes a
        # 'variable:' reference here, and a variable declared as a string default
        # substitutes as one
        scale = lora.pop("scale", None)
        scale = 1.0 if scale is None else float(scale)
        adapter_weights.append(scale)

        # Popped before the load: everything left in the dict is a keyword
        # argument to load_lora_weights, and the alpha is applied to the layers
        # afterwards rather than passed to it
        alpha = lora.pop("alpha", None)
        if alpha is not None:
            alphas[adapter_name] = alpha

        # Load the LoRA with the adapter name
        pipeline.load_lora_weights(model_name, adapter_name=adapter_name, **lora)

    for adapter_name, alpha in alphas.items():
        set_adapter_alpha(pipeline, adapter_name, alpha)

    # Set adapter weights for all loaded LoRAs
    if adapter_names:
        logger.info(
            f"Setting adapter weights: {list(zip(adapter_names, adapter_weights))}"
        )
        # Positionally - diffusers' mixin calls the second parameter 'adapter_weights'
        # while custom pipelines that delegate to the model (ostris/Krea2OstrisEdit)
        # call it 'weights'
        pipeline.set_adapters(adapter_names, adapter_weights)


def load_ip_adapter(ip_adapter_definition, pipeline):
    """Load and configure IP-Adapter if specified."""
    if ip_adapter_definition is not None:
        model_name = ip_adapter_definition.pop("model_name")
        logger.info(f"Loading IP-Adapter: {model_name}")
        scale = ip_adapter_definition.pop("scale", None)
        pipeline.load_ip_adapter(model_name, **ip_adapter_definition)
        if scale is not None:
            pipeline.set_ip_adapter_scale(scale)
