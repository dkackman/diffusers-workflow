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


def adapter_settings(loras):
    """(adapter_names, adapter_weights, alphas) for the active entries of
    `loras`. An adapter is named by its own adapter_name or its ORIGINAL
    index - a nulled entry keeps its slot, so the name a hit re-applies to
    is the name the load gave. Reads the entries, never pops them.

    `or` on adapter_name, because a variable nulled by the caller arrives as
    a present None; float() on scale, because the schema takes a 'variable:'
    reference there, and a variable declared as a string default substitutes
    as one."""
    names, weights, alphas = [], [], {}
    for i, lora in enumerate(loras or []):
        if not isinstance(lora, dict) or lora.get("model_name") is None:
            continue
        name = lora.get("adapter_name") or str(i)
        names.append(name)
        scale = lora.get("scale")
        weights.append(1.0 if scale is None else float(scale))
        if lora.get("alpha") is not None:
            alphas[name] = lora["alpha"]
    return names, weights, alphas


def apply_adapter_settings(loras, pipeline):
    """Set each active LoRA's alpha, then every scale, on a pipeline that
    already holds the adapters. Alpha first: set_adapters is what recomputes
    each layer's scaling from it. No-op with no active entry."""
    names, weights, alphas = adapter_settings(loras)
    if not names:
        return
    for name, alpha in alphas.items():
        set_adapter_alpha(pipeline, name, alpha)
    logger.info(f"Setting adapter weights: {list(zip(names, weights))}")
    # Positionally - diffusers' mixin calls the second parameter 'adapter_weights'
    # while custom pipelines that delegate to the model (ostris/Krea2OstrisEdit)
    # call it 'weights'
    pipeline.set_adapters(names, weights)


def load_loras(loras, pipeline):
    """Load and configure LoRA models.

    Names, scales and alphas come from adapter_settings, read before the
    entries are popped, so a load and a later cache hit's
    apply_adapter_settings set the same values on the same names."""
    # A snapshot of each entry's runtime keys, taken before the pops below
    settings = [
        {k: lora.get(k) for k in ("model_name", "adapter_name", "scale", "alpha")}
        if isinstance(lora, dict)
        else lora
        for lora in loras or []
    ]
    names = iter(adapter_settings(settings)[0])

    for i, lora in enumerate(loras or []):
        if isinstance(lora, dict) and lora.get("model_name") is None:
            # Switched off - said to the caller by warn_adapters before the
            # run started, so only logged here
            logger.info(f"LoRA {i} has a null model_name - not loaded")
            continue
        model_name = lora.pop("model_name", None)
        logger.info(f"Loading LoRA: {model_name}")
        emit_phase("loading", detail=f"LoRA: {model_name}")

        # Popped before the load: everything left in the dict is a keyword
        # argument to load_lora_weights; the scale and alpha are applied to
        # the layers afterwards rather than passed to it
        for key in ("adapter_name", "scale", "alpha"):
            lora.pop(key, None)

        pipeline.load_lora_weights(model_name, adapter_name=next(names), **lora)

    apply_adapter_settings(settings, pipeline)


def load_ip_adapter(ip_adapter_definition, pipeline):
    """Load and configure IP-Adapter if specified."""
    if ip_adapter_definition is not None:
        model_name = ip_adapter_definition.pop("model_name")
        logger.info(f"Loading IP-Adapter: {model_name}")
        scale = ip_adapter_definition.pop("scale", None)
        pipeline.load_ip_adapter(model_name, **ip_adapter_definition)
        if scale is not None:
            pipeline.set_ip_adapter_scale(scale)
