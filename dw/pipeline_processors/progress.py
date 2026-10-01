import torch
import contextlib
import logging
from ..events import WorkflowCancelled, emit_phase, get_context

logger = logging.getLogger("dw")


class _ReportingProgressBar:
    """A tqdm bar that also reports each advance to the active run.

    Wraps rather than subclasses, because the bar it wraps is whatever the
    block's own progress_bar() built - tqdm, or a notebook bar, or whatever
    a future diffusers uses. Everything it does not intercept falls through
    to the real bar, so the terminal output is unchanged.
    """

    def __init__(self, bar, on_advance, total=None):
        self._bar = bar
        self._on_advance = on_advance
        # Counted here rather than read off the bar: a disabled tqdm - which
        # is what a quiet server or a notebook config leaves you with - keeps
        # its own `n` at zero while still being advanced normally
        self._done = 0
        self._total = total if total is not None else getattr(bar, "total", None)

    def update(self, n=1):
        result = self._bar.update(n)
        self._done += n or 0
        self._on_advance(self._done, self._total)
        return result

    def __iter__(self):
        # Reported after the body of the loop has run, not before it: the
        # step is finished when control comes back here
        for item in self._bar:
            yield item
            self._done += 1
            self._on_advance(self._done, self._total)

    def __enter__(self):
        self._bar.__enter__()
        return self

    def __exit__(self, *exception):
        return self._bar.__exit__(*exception)

    def __getattr__(self, name):
        # Guarded: the wrapped bar is the first thing __init__ sets, and an
        # unguarded lookup of it before then recurses forever
        if name == "_bar":
            raise AttributeError(name)
        return getattr(self._bar, name)


def _progress_bar_holders(pipeline):
    """Every object under a modular pipeline that can open a progress bar.

    The denoise loop is a block, not the pipeline, and it calls its own
    `self.progress_bar(...)` - so the tree is what has to be walked. Uses
    `_blocks`, not the public `blocks`, which hands back a deepcopy: patching
    a copy would report nothing and look like this never worked.
    """
    holders = []
    seen = set()

    def walk(candidate):
        if candidate is None or id(candidate) in seen:
            return
        seen.add(id(candidate))
        # __dict__, because the patch is an instance attribute: an object
        # with none could not be patched and must not be tried
        if callable(getattr(candidate, "progress_bar", None)) and hasattr(
            candidate, "__dict__"
        ):
            holders.append(candidate)
        children = getattr(candidate, "sub_blocks", None)
        if hasattr(children, "values"):
            for child in children.values():
                walk(child)

    walk(pipeline)
    walk(getattr(pipeline, "_blocks", None))
    return holders


@contextlib.contextmanager
def reported_progress_bars(pipeline):
    """Report each denoise step of a pipeline that takes no step callback.

    A ModularPipeline - H3, LTX-2, Qwen-Image and every family diffusers has
    moved over - has no `callback_on_step_end` parameter, so the whole
    denoise loop passed in silence: one 'generating' phase, then nothing for
    however many minutes it took, which reads exactly like a hung run. What
    those blocks do have is a tqdm bar, and every advance of it is a step.

    The patch is per-instance and undone on the way out, so a pipeline this
    process keeps loaded is handed back as it was found.
    """
    holders = _progress_bar_holders(pipeline)
    if not holders:
        yield
        return

    run_context = get_context()

    def on_advance(done, total):
        run_context.emit("pipeline_step", step=done, total_steps=total)
        # Past the last step there is still the decode, which on video is
        # minutes with the bar sitting at 100%
        if done is not None and total is not None and done >= total:
            emit_phase("decoding")
        # The one cancellation checkpoint inside a modular denoise loop:
        # without it a cancel waits out the whole generation
        run_context.check_cancelled()

    patched = []
    for holder in holders:
        original = holder.progress_bar
        # Whether the name was already an attribute of the instance decides
        # how it is put back: restored, or removed so the class method shows
        # through again rather than a bound copy of it being frozen on
        patched.append((holder, original, "progress_bar" in vars(holder)))

        def reporting(iterable=None, total=None, _original=original):
            bar = _original(iterable=iterable, total=total)
            if total is None and iterable is not None:
                # An iterated bar's total is the length of what it iterates,
                # when that can be known at all
                total = getattr(bar, "total", None)
            return _ReportingProgressBar(bar, on_advance, total)

        holder.progress_bar = reporting
    try:
        yield
    finally:
        for holder, original, was_own in patched:
            if was_own:
                holder.progress_bar = original
            else:
                try:
                    del holder.progress_bar
                except AttributeError:
                    holder.progress_bar = original


def _runs_its_blocks_in_sequence(blocks):
    """Whether a modular block container runs every sub-block in order.

    diffusers' own `SequentialPipelineBlocks` is the answer; the import is
    local and forgiving because a pipeline that is not modular at all never
    reaches here, and a diffusers without the class is one with no modular
    pipelines to narrate.
    """
    try:
        from diffusers.modular_pipelines.modular_pipeline import (
            SequentialPipelineBlocks,
        )
    except ImportError:  # pragma: no cover - a diffusers without modular
        return False
    return isinstance(blocks, SequentialPipelineBlocks)


@contextlib.contextmanager
def reported_blocks(pipeline, label):
    """Name each of a modular pipeline's top-level blocks as it starts.

    The denoise loop is only one of them, and on a reference-conditioned
    model it is not the long one: encoding a video reference runs for
    minutes inside `vae_encoder` before a single bar advances, so the whole
    lead-in went by with nothing emitted and a consumer could not tell it
    from a hang (#95). The blocks are named - `before_encode`,
    `text_encoder`, `vae_encoder`, `denoise`, `decode` on H3 - and naming
    each one as it begins turns that silence into "it is encoding the
    reference", plus a `seconds_since_event` that resets at every boundary.

    Reported as `log` events rather than phases: `PHASES` is a closed set a
    consumer switches on, and a block name is a detail, not a new state.

    The patch is on the class because `block(pipeline, state)` resolves
    `__call__` on the type, not the instance - so it is guarded by identity
    (only the pipeline's own top-level blocks report) and undone on the way
    out.
    """
    blocks = getattr(pipeline, "_blocks", None)
    sub_blocks = getattr(blocks, "sub_blocks", None)
    if blocks is None or not hasattr(sub_blocks, "items"):
        yield
        return
    # Only a sequence runs all of its sub-blocks. A conditional container
    # (AutoPipelineBlocks, which is what several of H3's own steps are)
    # *picks* one on its inputs, so narrating it by walking the mapping
    # would run every branch - hence the check for the one dispatch this
    # reproduces rather than a duck-typed `sub_blocks`
    if not _runs_its_blocks_in_sequence(blocks):
        yield
        return

    holder = type(blocks)
    original = holder.__call__
    was_own = "__call__" in vars(holder)
    run_context = get_context()

    @torch.no_grad()
    def reporting(self, pipe, state):
        # A nested SequentialPipelineBlocks shares the class; only the
        # pipeline's own top-level sequence is the one worth narrating
        if self is not blocks:
            return original(self, pipe, state)
        for name, block in self.sub_blocks.items():
            run_context.emit("log", message=f"{label}: {name}")
            # A block boundary is a cancellation checkpoint the lead-in
            # otherwise has none of
            run_context.check_cancelled()
            try:
                pipe, state = block(pipe, state)
            except WorkflowCancelled:
                raise
            except Exception:
                # What diffusers' own dispatch logs, kept because this
                # replaces that loop
                logger.error(f"Error in block: ({name}, {block.__class__.__name__})")
                raise
        return pipe, state

    holder.__call__ = reporting
    try:
        yield
    finally:
        if was_own:
            holder.__call__ = original
        else:
            try:
                del holder.__call__
            except AttributeError:
                holder.__call__ = original
