"""MiniMax-H3 audio hold and refine: generating a video to a soundtrack the caller
already has, and re-denoising an upscaled latent from a low sigma.

The one place dw modifies a diffusers modular pipeline's block graph. H3 denoises
video and audio rows in one packed sequence, and the rows it treats as conditioning
are the leading `num_condition_audio_rows` of the audio stream: pinned clean at
`t = 1.0`, never stepped by the scheduler, attended to by every generated row. That
is how a Ref2VA soundtrack reference rides along. Holding a track is the same
mechanism pointed at the target rows instead - the caller's waveform is encoded
through diffusers' own Ref2VA reference encode, written over the target audio rows,
and the count is widened to cover them, so the denoiser generates video that fits a
soundtrack it is not allowed to change.

Three blocks go into each of H3's three core-denoise sequences (`t2va`, `fl2va`,
`ref2va`), always, as no-ops when their argument is not passed:

- `DwH3HoldAudioStep`, before `set_timesteps` - after every `prepare_latents*`
  step, so Ref2VA's reference rows are already in front and its reference-row
  count check has already run against the count it expects;
- `DwH3RefineScheduleStep`, before `denoise` - after `set_timesteps`, whose
  schedule it replaces when `refine_strength` is passed: σ₀ = `refine_strength`,
  then `num_inference_steps` points down to 0 spaced by the scheduler's shift
  (`refine_sigmas`), so `num_inference_steps - 1` evaluations. It re-noises the
  generated video rows of a passed `latents` to σ₀. Every audio row has to be a
  held one (`hold_audio`): the audio scheduler gets the same list only so the
  loop's audio step finds its timesteps, and steps no row;
- `DwH3ReleaseAudioStep`, before `after_denoise` - which slices
  `audio_latents[num_condition_audio_rows:]` as the generated track, so the count
  has to be back to what it was for the held rows to be decoded as the target.

The decoded `audio` is still the VAE round trip of the held rows. The step's result
uses `dw_held_audio` instead - the caller's own waveform fitted to the video's
duration (dw/output_extraction.py), which `HELD_AUDIO_OUTPUT` names for the call
(dw/pipeline_processors/pipeline.py).

This module puts the blocks in and resolves the `hold_audio` argument; the
blocks themselves, and the sigma and row arithmetic they run, are
dw/pipeline_processors/h3_hold_steps.py, imported in `blocks()` only, since it
imports diffusers when it loads (#790). The names and pure rules these blocks
share with validation are dw/pipeline_processors/h3_rules.py; the guide layout,
built on these, is dw/pipeline_processors/h3_guides.py.
"""

import logging

from .h3_rules import (
    CORE_DENOISE_SEQUENCES,
    DENOISE_STEP,
    HOLD_BEFORE,
    HOLD_BLOCK,
    REFINE_BEFORE,
    REFINE_BLOCK,
    RELEASE_BEFORE,
    RELEASE_BLOCK,
)

logger = logging.getLogger("dw")


def blocks():
    """(DwH3HoldAudioStep, DwH3RefineScheduleStep, DwH3ReleaseAudioStep) - the
    classes in dw/pipeline_processors/h3_hold_steps.py, imported here on first use
    because that module imports diffusers when it loads (#790)."""
    from . import h3_hold_steps

    return (
        h3_hold_steps.DwH3HoldAudioStep,
        h3_hold_steps.DwH3RefineScheduleStep,
        h3_hold_steps.DwH3ReleaseAudioStep,
    )


def core_denoise_sequences(pipeline):
    """[(prefix, sequence)] for every H3 core-denoise sequence a pipeline's own
    block graph holds - empty for anything that is not a MiniMax-H3 modular
    pipeline.

    Two shapes. Loaded whole, `denoise` is an auto step choosing between the
    three sequences, each holding `set_timesteps`. Loaded with a
    `from_pretrained` `workflow` (what the catalog does), diffusers prunes the
    graph to that workflow's execution blocks, flattened into the top level
    under dotted names - `denoise.set_timesteps` - so the top level is the one
    sequence and `denoise.` the prefix its block names carry.

    `_blocks`, not `blocks`: the property hands back a deep copy, and an edit to
    that copy never reaches the graph the pipeline runs.
    """
    try:
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline
    except ImportError:
        return []
    if not isinstance(pipeline, MiniMaxH3ModularPipeline):
        return []
    top_blocks = getattr(pipeline, "_blocks", None)
    top = getattr(top_blocks, "sub_blocks", None) or {}
    flat_prefix = f"{DENOISE_STEP}."
    if f"{flat_prefix}{HOLD_BEFORE}" in top:
        return [(flat_prefix, top_blocks)]
    denoise = getattr(top.get(DENOISE_STEP), "sub_blocks", None) or {}
    return [
        ("", denoise[name])
        for name in CORE_DENOISE_SEQUENCES
        if name in denoise and hasattr(denoise[name], "sub_blocks")
    ]


def insert_audio_hold(pipeline):
    """Put the hold, refine and release blocks into every H3 core-denoise sequence.

    Idempotent - a block a sequence already holds is left alone. Returns whether
    the pipeline now holds audio. Hold and release go in as a pair: a sequence
    missing either anchor (a diffusers that renamed `set_timesteps` or
    `after_denoise`) gets neither, and refine needs its own anchor, `denoise`.
    A sequence that cannot take a block is skipped with a warning, and a
    pipeline left with none refuses that block's argument at the call.
    """
    hold_block, refine_block, release_block = blocks()
    inserted = False
    for prefix, sequence in core_denoise_sequences(pipeline):
        sub_blocks = sequence.sub_blocks
        hold_name, release_name = prefix + HOLD_BLOCK, prefix + RELEASE_BLOCK
        hold_before, release_before = prefix + HOLD_BEFORE, prefix + RELEASE_BEFORE
        refine_name, refine_before = prefix + REFINE_BLOCK, prefix + REFINE_BEFORE
        names = list(sub_blocks)
        if hold_name in names:
            inserted = True
        elif hold_before not in names or release_before not in names:
            logger.warning(
                f"MiniMax-H3 has a denoise sequence with no '{hold_before}'/"
                f"'{release_before}' step to anchor the audio hold to, so it "
                f"cannot hold audio"
            )
        else:
            sub_blocks.insert(hold_name, hold_block(), names.index(hold_before))
            names = list(sub_blocks)
            sub_blocks.insert(
                release_name, release_block(), names.index(release_before)
            )
            inserted = True
        names = list(sub_blocks)
        if refine_name in names:
            continue
        if refine_before not in names:
            logger.warning(
                f"MiniMax-H3 has a denoise sequence with no '{refine_before}' "
                f"step to anchor the refine schedule to, so it cannot refine"
            )
            continue
        sub_blocks.insert(refine_name, refine_block(), names.index(refine_before))
    return inserted


def holds_audio(pipeline):
    """Whether a pipeline's block graph carries the audio hold."""
    return any(
        prefix + HOLD_BLOCK in sequence.sub_blocks
        for prefix, sequence in core_denoise_sequences(pipeline)
    )


def refines(pipeline):
    """Whether a pipeline's block graph carries the refine schedule."""
    return any(
        prefix + REFINE_BLOCK in sequence.sub_blocks
        for prefix, sequence in core_denoise_sequences(pipeline)
    )


def hold_audio_reference(value, base_dir=None):
    """A `hold_audio` argument as the MiniMaxH3AudioReference the hold block takes.

    Whatever an argument can hold audio as: a file path (an `asset:` or `output:`
    reference arrives resolved to one), a step's artifact - an AudioTrack, an
    AudioVideo, a bare waveform - or a reference already built. A path or URL goes
    through the location owner (dw/locations.py) before anything opens it, the
    run-time half of the check validation makes on the same argument.

    Args:
        value: The argument's value
        base_dir: Directory a relative path resolves against - the workflow
            file's directory

    Raises:
        ValueError: If the value is not audio
        SecurityError: If the path or URL is outside what the workflow may read
    """
    from ..argument_media import local_media_file
    from ..arguments import media_arguments
    from ..locations import is_http_url, validate_media_path, validate_media_url
    from ..security import (
        ALLOWED_AUDIO_EXTENSIONS,
        InvalidInputError,
        validate_file_extension,
    )

    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
    )

    if isinstance(value, MiniMaxH3AudioReference):
        return value
    if isinstance(value, str):
        try:
            validate_file_extension(value, ALLOWED_AUDIO_EXTENSIONS)
        except InvalidInputError as error:
            raise ValueError(
                f"hold_audio holds a soundtrack, and '{value}' is not an audio file "
                f"({', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))})"
            ) from error
        if is_http_url(value):
            # Checked again, hop by hop, by the fetch below
            location = validate_media_url(value, "hold_audio")
        else:
            location = validate_file_extension(
                validate_media_path(value, base_dir, "hold_audio"),
                ALLOWED_AUDIO_EXTENSIONS,
            )
        # from_file given a URL downloads it with requests itself, outside
        # the host policy; given a path it only decodes
        with local_media_file(location, "hold_audio") as path:
            return MiniMaxH3AudioReference.from_file(path)
    try:
        return MiniMaxH3AudioReference(
            **media_arguments(MiniMaxH3AudioReference, value)
        )
    except ValueError as error:
        raise ValueError(f"hold_audio: {error}") from error
