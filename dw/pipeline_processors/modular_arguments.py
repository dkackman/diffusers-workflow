"""What a modular pipeline call is handed: its keyword arguments, minus None.

The None rule (#1315, plan Q1): before dw calls a modular pipeline (a
diffusers `ModularPipeline` - MiniMax-H3, FLUX 3 Klein), it drops every
top-level keyword argument whose value is `None`, so `null` means "the
block's default", whatever the source - a `previous_result:` that resolved
to nothing, a literal `null`, or an H3 wrapper's explicit `guides=None`.

That was diffusers 0.42's own meaning: a modular block treated an explicit
None as absent and took its `InputParam` default. 0.43 keeps an explicit None
and hands it to the block, so without this drop the same workflow would
change meaning on a diffusers upgrade.

- A classic `DiffusionPipeline` is untouched: it never had the fallback, and
  None is a real value to many of its signatures.
- Only the top level is filtered. A None inside a list (a null variable in a
  list argument) is left as is.
- A falsy non-None value (0, False, "", []) is a value, and is kept.

Used by dw/pipeline_processors/pipeline.py (the main call) and
dw/pipeline_processors/h3_tiles.py (the tiled refine's text and tile calls).
"""


def is_modular_pipeline(pipeline):
    """Whether `pipeline` is a diffusers ModularPipeline."""
    try:
        from diffusers.modular_pipelines import ModularPipeline
    except ImportError:  # pragma: no cover - every supported diffusers has it
        return False
    return isinstance(pipeline, ModularPipeline)


def without_none(arguments):
    """`arguments` without its top-level None values (a new dict)."""
    return {key: value for key, value in arguments.items() if value is not None}


def call_arguments(pipeline, arguments):
    """The keyword arguments to call `pipeline` with: `arguments` without its
    top-level None values when the pipeline is modular, unchanged otherwise."""
    if is_modular_pipeline(pipeline):
        return without_none(arguments)
    return arguments
