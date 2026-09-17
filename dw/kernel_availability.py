"""An 'attn_processor_type' backed by a Hub kernel that this box cannot run.

`LTX2VideoVaeNeighborhoodNattenProcessor` (LTX-2.5's diffusion decoder)
fetches its `na3d` kernel from `shi-labs/natten` through the `kernels`
package in `__init__` - no forward pass, no `__call__`, just construction.
When the installed torch is newer than every published natten build variant
(torch 2.14, natten's newest wheel built for 2.13), that `__init__` raises
`FileNotFoundError` - and by the time anything constructs the processor, the
pipeline has already loaded its transformer and text encoder onto the GPU.
`validate_workflow` reported this workflow clean and `run_workflow` paid
~88s of loading to learn what construction alone would have said in under
two seconds (#178).

The processor class is resolved exactly as `dw/pipeline_processors/pipeline.py`
resolves it at run time - `load_type_from_name`, then a zero-argument
construction, since every 'attn_processor_type' site (the top-level
`unet`/`transformer` blocks and the generic per-component
`set_attn_processor`) constructs it that way. Not every attention processor
depends on a Hub kernel, so this only probes ones whose `__init__` calls
`kernels.get_kernel` - a source-level check rather than a natten-specific
name, so a future diffusers processor with the same Hub-fetch shape is
caught without a new entry here. A processor that constructs cleanly, or
whose source can't be inspected, is left alone: this check exists to move a
real, cheap, side-effect-free construction failure earlier, not to guess at
one.
"""

import functools
import inspect

from .for_each import MEMBER_SEPARATOR, render_path
from .type_helpers import load_type_from_name

ATTN_PROCESSOR_KEY = "attn_processor_type"
_UNRESOLVED_PREFIXES = ("variable:", "item:")
KERNEL_FAULT_MARKER = "cannot be used on this machine"


def is_kernel_availability_fault(message):
    """Whether a validation error's text is this check's, not the author's.

    A template naming a Hub-kernel-backed processor is only ever as valid as
    the box running the check - one box's torch build satisfies natten and
    another's doesn't, so a failure here says nothing about the workflow
    itself (dw/kernel_availability.py, #178). Callers that need to tell "this
    box can't satisfy a real dependency" apart from "the workflow is wrong"
    - such as tests/test_examples.py, which otherwise expects every shipped
    template to validate on whatever machine runs the suite - match on this
    rather than re-deriving the message.
    """
    return KERNEL_FAULT_MARKER in message


def _attn_processor_locations(configuration):
    """Yield (path_suffix, value) for every 'attn_processor_type' string in
    one step's pipeline configuration - the top-level 'unet'/'transformer'
    blocks and any 'components' entry, matching every site
    dw/pipeline_processors/pipeline.py constructs one from.
    """
    if not isinstance(configuration, dict):
        return

    for block_name in ("unet", "transformer"):
        block = configuration.get(block_name)
        if isinstance(block, dict) and ATTN_PROCESSOR_KEY in block:
            yield (block_name, ATTN_PROCESSOR_KEY), block[ATTN_PROCESSOR_KEY]

    components = configuration.get("components")
    if isinstance(components, dict):
        for component_name, component_configuration in components.items():
            if (
                isinstance(component_configuration, dict)
                and ATTN_PROCESSOR_KEY in component_configuration
            ):
                yield (
                    ("components", component_name, ATTN_PROCESSOR_KEY),
                    component_configuration[ATTN_PROCESSOR_KEY],
                )


def _requires_remote_kernel(attn_processor_class):
    """Whether constructing this class fetches a kernel from the Hub -
    a source-level check so a future processor with the same shape as
    natten's is caught without naming it here.
    """
    try:
        source = inspect.getsource(attn_processor_class.__init__)
    except (OSError, TypeError):
        return False
    return "get_kernel(" in source


class _KernelFault(Exception):
    """Carries a fault message out of `_fault_for_name` without letting
    `lru_cache` memoize it - see that function's docstring."""


@functools.lru_cache(maxsize=None)
def _fault_for_name(value):
    """The answer for one type name, memoized - except a fault.

    Which class a name resolves to, and whether that class *requires* a
    remote kernel, are per-process constants, so a clean `None` answer is
    cached. Whether *this* construction attempt succeeds is not: the
    construction is a Hub kernel fetch, which can fail transiently (network,
    rate limit, cold cache), and caching that failure would pin every
    validate/submit/rerun in the process to "broken" until the process
    restarts, even once the transient condition clears. `lru_cache` never
    memoizes a call that raises, so a fault is raised as `_KernelFault`
    rather than returned - `kernel_availability_fault` catches it. The cost
    of not caching a fault is one more fetch on the next call, which is
    cheap next to being wrong for a process's whole lifetime.
    """
    try:
        attn_processor_class = load_type_from_name(value)
    except Exception:
        return None

    if not _requires_remote_kernel(attn_processor_class):
        return None

    try:
        attn_processor_class()
    except Exception as e:
        raise _KernelFault(f"'{value}' {KERNEL_FAULT_MARKER}: {e}") from e
    return None


def kernel_availability_fault(value):
    """Why constructing this 'attn_processor_type' value will fail, or None.

    `value` is the dotted or bare type name as written in the workflow.
    A name that fails to resolve at all is not this check's job - that is
    an ordinary '_type' load failure at run time. A processor that doesn't
    depend on a remote kernel is never constructed here, since a processor
    with real side effects in `__init__` should not be instantiated just to
    validate it.

    A clean answer is memoized once per process per name: validate, submit
    and rerun all ask, and a for_each asks once per member (`_fault_for_name`).
    A fault is not memoized, and so is re-probed on every call - see
    `_fault_for_name`.
    """
    if not isinstance(value, str) or value.startswith(_UNRESOLVED_PREFIXES):
        return None
    try:
        return _fault_for_name(value)
    except _KernelFault as fault:
        return str(fault)


def kernel_availability_errors(workflow_definition, source_indices=None):
    """Every 'attn_processor_type' whose Hub kernel this machine can't
    satisfy, as [{path, message}] - reported before any model loads rather
    than discovered inside one (#178).

    The definition handed here has already been substituted and expanded,
    so every value in it is literal; a 'variable:' or 'item:' still spelled
    out is left alone. `source_indices`, when given, is the source step
    index of each step - a 'for_each' group turns one written step into
    several, and the path an error carries has to be one the author can
    find in the file they wrote; the member is named in the message.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        pipeline = step.get("pipeline")
        if not isinstance(pipeline, dict):
            continue
        configuration = pipeline.get("configuration")

        for path_suffix, value in _attn_processor_locations(configuration):
            fault = kernel_availability_fault(value)
            if fault is None:
                continue

            source = (
                source_indices[index]
                if source_indices is not None and index < len(source_indices)
                else index
            )
            name = step.get("name")
            where = (
                f" in member '{name}'"
                if isinstance(name, str) and MEMBER_SEPARATOR in name
                else ""
            )
            errors.append(
                {
                    "path": render_path(
                        ("steps", source, "pipeline", "configuration") + path_suffix
                    ),
                    "message": f"{fault}{where}",
                }
            )
    return errors


__all__ = [
    "is_kernel_availability_fault",
    "kernel_availability_errors",
    "kernel_availability_fault",
]
