"""The cache hooks' denoiser and MagCache calibration capture.

`get_cache_transformer` names the component a cache hook attaches to, and
`captured_mag_calibration` turns diffusers' end-of-run calibration log record
into job events (#891), `_report_unfinished_mag_calibration` the ratios a call
that ran fewer forwards than its schedule left behind (#927). Imported by
dw/pipeline_processors/components.py, which re-exports the two public names for
dw/pipeline_processors/pipeline.py, and never imports it back.
"""

import ast
import contextlib
import logging

from ..events import emit_log, emit_warning


def get_cache_transformer(pipeline):
    """Find the denoiser a cache hook attaches to.

    Most pipelines register theirs as 'transformer', but a modular pipeline names
    it after the workflow it serves - MiniMax-H3's ref2va denoises through
    'transformer_ref'. Looking only for 'transformer' silently skips caching on
    those, so try the alternates diffusers' modular pipelines actually use.

    Args:
        pipeline: The loaded diffusers pipeline

    Returns:
        The transformer component, or None when the pipeline has none
    """
    for name in ("transformer", "transformer_ref"):
        transformer = getattr(pipeline, name, None)
        if transformer is not None:
            return transformer
    return None


_MAG_LOGGER = "diffusers.hooks.mag_cache"
_MAG_RESULT_PREFIX = "MagCache Calibration Results:"


def _active_cache_context_name(transformer):
    """The name of the cache context a stateful hook under the transformer is
    running in, or None. Under CFG that is "cond" or "uncond"."""
    for module in transformer.modules():
        registry = getattr(module, "_diffusers_hook", None)
        for hook in getattr(registry, "hooks", {}).values():
            for attr in vars(hook).values():
                context = getattr(attr, "_context", None)
                if context is not None and hasattr(attr, "get_state"):
                    return context.name
    return None


class _MagCalibrationHandler(logging.Handler):
    """Turns diffusers' end-of-run calibration log record into job events.

    diffusers prints the ratios, logs them at INFO and clears the list in the
    same call (`MagCacheHook._advance_step`), so the record is the only moment
    they can be read, and the worker's stdout reaches no consumer. The record
    is emitted synchronously from inside the forward, so the state manager's
    context is still the one the ratios belong to.
    """

    def __init__(self, transformer):
        super().__init__(level=logging.INFO)
        self.transformer = transformer

    def emit(self, record):
        message = record.getMessage()
        if not message.startswith(_MAG_RESULT_PREFIX):
            return
        try:
            ratios = ast.literal_eval(message[len(_MAG_RESULT_PREFIX) :].strip())
            context = _active_cache_context_name(self.transformer)
        except (ValueError, SyntaxError):
            return
        _report_mag_ratios(ratios, context)


def _report_mag_ratios(ratios, context):
    label = f" [{context}]" if context in ("cond", "uncond") else ""
    text = (
        f"MagCache calibration ({len(ratios)} steps){label}: "
        f"{ratios} \u2014 paste into mag_ratios"
    )
    if context == "cond":
        text += " (classifier-free guidance runs two lists; the cond one is the usual choice)"
    emit_log(text)
    emit_warning(text)


def _report_unfinished_mag_calibration(transformer):
    """Report the ratios a calibrating call left in its mag state (#927).

    diffusers logs a calibration only when its step counter reaches the
    config's num_inference_steps, and a scheduler can run fewer forwards than
    that: MiniMax-H3's turns N sigmas into N - 1 evaluations. Such a call ends
    with its ratios still in the state, which the call's reset then drops. A
    call that did finish has nothing left here - diffusers clears the list
    when it logs - so nothing is reported twice.
    """
    if not getattr(getattr(transformer, "_cache_config", None), "calibrate", False):
        return
    from diffusers.hooks.mag_cache import MagCacheBlockHook

    for module in transformer.modules():
        registry = getattr(module, "_diffusers_hook", None)
        for hook in getattr(registry, "hooks", {}).values():
            if isinstance(hook, MagCacheBlockHook) and hook.is_tail:
                for name, state in hook.state_manager._state_cache.items():
                    if state.calibration_ratios:
                        _report_mag_ratios(list(state.calibration_ratios), name)
                return


@contextlib.contextmanager
def captured_mag_calibration(pipeline, arguments=None):
    """Report a mag cache's calibration ratios as job events (#891).

    A no-op unless the transformer's cache is a MagCacheConfig with calibrate
    on. Under classifier-free guidance dw's pipelines run "cond" and "uncond"
    contexts, each with its own state, so two labeled lists are emitted.

    diffusers only finishes (and logs) a calibration once its step counter
    reaches the config's `num_inference_steps`, which defaults to 28. A run of
    any other length would end without a record, so for the call the config
    counts the steps the call actually runs.
    """
    transformer = get_cache_transformer(pipeline)
    config = getattr(transformer, "_cache_config", None)
    if not getattr(config, "calibrate", False):
        yield
        return

    steps = (arguments or {}).get("num_inference_steps")
    declared_steps = config.num_inference_steps
    if isinstance(steps, int) and steps != declared_steps:
        emit_warning(
            f"MagCache calibration: the cache declares num_inference_steps "
            f"{declared_steps} but this step runs {steps}; calibrating over {steps}"
        )
        config.num_inference_steps = steps

    mag_logger = logging.getLogger(_MAG_LOGGER)
    handler = _MagCalibrationHandler(transformer)
    old_level, old_propagate = mag_logger.level, mag_logger.propagate
    mag_logger.addHandler(handler)
    mag_logger.setLevel(logging.INFO)
    mag_logger.propagate = False
    try:
        yield
    finally:
        config.num_inference_steps = declared_steps
        mag_logger.removeHandler(handler)
        mag_logger.setLevel(old_level)
        mag_logger.propagate = old_propagate
