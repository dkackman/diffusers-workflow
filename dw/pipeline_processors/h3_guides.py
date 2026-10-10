"""MiniMax-H3 guides: earlier clips held as condition rows of a new generation.

Guides (`guides=[{video, frame}]`) hold clips of an existing video as video
condition rows. Unlike the three blocks above they are added only to a pipeline
built to take them (`insert_guides`): `DwH3GuideLayoutStep` wraps the stock layout
step, runs it, then VAE-encodes each clip on the target canvas and splices its rows
after any keyframe rows, timed from the target frame it lands on, widening
`num_condition_video_rows` so `after_denoise` slices them off. On `t2va` two more
blocks run the stock `fl2va` condition-latent steps only when there are guides.

A guide with `audio` also holds its soundtrack over its span (#649). The span starts
at the audio latent at or before the guide's first frame, `floor(frame * 40 / 24)`,
and runs `round(frames * 40 / 24)` latents on from the frame's own time; the
waveform is pre-padded with the silence between that latent and the frame, so the
time origin holds (`guide_audio_span`, `pad_guide_audio`). It is encoded the way a
Ref2VA soundtrack is (the posterior mode), and the layout splices its rows in
front of the target audio, at the target's own rotary times for that span, as
condition audio rows held at `t = 1.0`. `DwH3GuideAudioStep` puts the encoded rows
in front of `audio_latents` before the audio hold, which then writes after them.
Nothing restores them before decode: `after_denoise` slices the condition prefix
off, and the target rows at the same times are generated, as with a video guide.

This module puts the guide blocks in; the blocks themselves, and the row and
position arithmetic they run (`splice_guide_rows`, `guide_audio_span`), are
dw/pipeline_processors/h3_guide_steps.py, imported in `guide_blocks()` only,
since it imports diffusers when it loads (#790). The hold blocks are
dw/pipeline_processors/h3_hold.py and h3_hold_steps.py, which these modules build
on and call through the module (`h3_hold_steps.encode_audio_span`), never the
reverse; the pure guide rules are dw/pipeline_processors/h3_rules.py.
"""

import logging

import torch

from .h3_hold import core_denoise_sequences
from .h3_rules import (
    GUIDE_AUDIO_BLOCK,
    GUIDE_CONDITION_BLOCK,
    GUIDE_LATENTS_BLOCK,
    HOLD_BEFORE,
    HOLD_BLOCK,
    LAYOUT_STEP,
)

logger = logging.getLogger("dw")


# Guides (#611): earlier clips held as multi-frame condition rows.
#
# H3 already conditions on rows it never steps: a `fl2va` keyframe is one encoded
# latent frame in front of the target, noised to `keyframe_noise_aug` and given
# the target's own rotary time at its anchor. A guide is the same thing for a
# clip - every latent frame of its encode, timed from the target latent it lands
# on, appended after the keyframe rows. Only the layout knows one frame per
# keyframe, so the layout is the step dw replaces; the condition noise, the
# denoise loop and `after_denoise` already work on any prefix of condition rows.
#
# `t2va` has no condition steps, so it gets two thin ones that run diffusers' own
# `prepare_condition_latents` / `prepare_latents_fl2va` only when there is
# conditioning to pack - the stock step cannot take an empty list.
LAYOUT_ANCHORS = (
    "MiniMaxH3PrepareLayoutStep",
    "MiniMaxH3PrepareLayoutStep.build_packed_sequence",
    "MiniMaxH3PrepareConditionLatentsStep",
    "MiniMaxH3FL2VAPrepareLatentsStep",
    "_temporal_position_grid",
    "_frame_position_grid",
    "_fill_audio_positions",
)


def default_num_frames():
    """The `num_frames` H3 renders when a step passes none: the stock layout
    step's own default, read off diffusers rather than copied. None when the
    installed diffusers has no such step or default."""
    try:
        from diffusers.modular_pipelines.minimax_h3 import before_denoise
    except ImportError:
        return None
    stock = getattr(before_denoise, "MiniMaxH3PrepareLayoutStep", None)
    if stock is None:
        return None
    try:
        inputs = stock().inputs
    except Exception:  # a diffusers whose step needs arguments to build
        return None
    for param in inputs:
        if param.name == "num_frames" and isinstance(param.default, int):
            return param.default
    return None


def layout_anchor_problem():
    """The first diffusers name the guide layout is built on that the installed
    diffusers lacks, or None."""
    try:
        from diffusers.modular_pipelines.minimax_h3 import before_denoise
    except ImportError:
        return "diffusers.modular_pipelines.minimax_h3.before_denoise"
    for anchor in LAYOUT_ANCHORS:
        owner = before_denoise
        for part in anchor.split("."):
            owner = getattr(owner, part, None)
            if owner is None:
                return anchor
    return None


def guide_audio_waveform(value):
    """(waveform, sample_rate) of a guide video's soundtrack: a `(channels,
    samples)` float32 tensor, from what a guide's `video` loads to with its audio -
    an AudioVideo, a step's Selected pick of one or a one-video list of one.

    Raises:
        ValueError: If the video carries no audio
    """
    # Here, not at the top: h3_hold_steps imports diffusers when it loads (#790)
    from .h3_hold_steps import as_channels_samples

    while True:
        if hasattr(value, "value") and hasattr(value, "position"):
            value = value.value
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        else:
            break
    audio = getattr(value, "audio", None)
    if audio is not None:
        audio = as_channels_samples(audio)
    if audio is None or not audio.numel():
        raise ValueError(
            "'audio' is true, but this guide's video has no audio - pass a video "
            "with a soundtrack, or drop 'audio'"
        )
    return audio, getattr(value, "sample_rate", None)


def guide_frames_array(value):
    """A guide's video as `(n, h, w, 3)` uint8 frames - from a list of PIL images
    or arrays (what a `video` argument loads to), or one array or tensor of
    frames, float in [0, 1] or uint8.

    Raises:
        ValueError: If the value is not frames of a video
    """
    import numpy as np
    from PIL import Image

    # A step's result: a Selected pick, a one-video list, a clip with its audio
    while True:
        if hasattr(value, "value") and hasattr(value, "position"):
            value = value.value
        elif hasattr(value, "frames") and hasattr(value, "audio"):
            value = value.frames
        elif (
            isinstance(value, (list, tuple))
            and len(value) == 1
            and not isinstance(value[0], (str, Image.Image))
            and getattr(value[0], "ndim", 4) != 3
        ):
            value = value[0]
        else:
            break
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("a guide's video has no frames")
        if isinstance(value[0], str):
            raise ValueError(f"a guide's video did not load: {value[0]!r}")
        value = np.stack(
            [
                np.asarray(f.convert("RGB")) if isinstance(f, Image.Image) else f
                for f in value
            ]
        )
    if not isinstance(value, np.ndarray):
        raise ValueError(f"a guide's video must be a video, got {type(value).__name__}")
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4 or value.shape[-1] not in (3, 4):
        raise ValueError(
            f"a guide's video must be (frames, height, width, 3), got {value.shape}"
        )
    value = value[..., :3]
    if value.dtype != np.uint8:
        value = (np.clip(value.astype(np.float32), 0.0, 1.0) * 255.0).round()
        value = value.astype(np.uint8)
    return np.ascontiguousarray(value)


def guide_blocks():
    """(DwH3GuideLayoutStep, the t2va condition step, the t2va latents step,
    DwH3GuideAudioStep) - the classes in
    dw/pipeline_processors/h3_guide_steps.py, imported here on first use because
    that module imports diffusers when it loads (#790)."""
    from . import h3_guide_steps

    return (
        h3_guide_steps.DwH3GuideLayoutStep,
        h3_guide_steps.CONDITION_STEP,
        h3_guide_steps.LATENTS_STEP,
        h3_guide_steps.DwH3GuideAudioStep,
    )


def _layout_sequences(pipeline):
    """[(prefix, sequence)] for the core-denoise sequences that lay out their own
    keyframes - every one but `ref2va`, whose layout is a different step."""
    from diffusers.modular_pipelines.minimax_h3 import before_denoise

    stock = before_denoise.MiniMaxH3PrepareLayoutStep
    layout = guide_blocks()[0]
    found = []
    for prefix, sequence in core_denoise_sequences(pipeline):
        step = sequence.sub_blocks.get(prefix + LAYOUT_STEP)
        if type(step) in (stock, layout):
            found.append((prefix, sequence))
    return found


def insert_guides(pipeline):
    """Swap `prepare_layout` for the guide layout in every `t2va`/`fl2va`
    core-denoise sequence, and give `t2va` the condition steps it lacks.

    Idempotent. Returns whether the pipeline now takes `guides`. A diffusers
    missing a name the layout is built on gets nothing, with a warning naming it,
    and `guides` is refused at the call (`guides_refusal`).
    """
    missing = layout_anchor_problem()
    if missing:
        if core_denoise_sequences(pipeline):
            logger.warning(
                f"MiniMax-H3 guides are built on diffusers' '{missing}', which the "
                f"installed diffusers does not have, so this pipeline cannot take guides"
            )
        return False
    layout_block, condition_block, latents_block, audio_block = guide_blocks()
    inserted = False
    for prefix, sequence in _layout_sequences(pipeline):
        sub_blocks = sequence.sub_blocks
        name = prefix + LAYOUT_STEP
        if not isinstance(sub_blocks[name], layout_block):
            sub_blocks[name] = layout_block()
        names = list(sub_blocks)
        latents_name = prefix + "prepare_latents"
        if prefix + "prepare_condition_latents" not in names and (
            latents_name in names
        ):
            # t2va: condition noise is drawn before the generated rows' noise,
            # and the condition rows go in front of them after - fl2va's order
            if prefix + GUIDE_CONDITION_BLOCK not in names:
                sub_blocks.insert(
                    prefix + GUIDE_CONDITION_BLOCK,
                    condition_block(),
                    names.index(latents_name),
                )
                names = list(sub_blocks)
            if prefix + GUIDE_LATENTS_BLOCK not in names:
                sub_blocks.insert(
                    prefix + GUIDE_LATENTS_BLOCK,
                    latents_block(),
                    names.index(latents_name) + 1,
                )
        names = list(sub_blocks)
        # Guide audio leads the audio rows, so it goes in before the audio hold
        # writes the target rows after it - before `set_timesteps`, or before
        # the hold when that is already in
        audio_before = next(
            (n for n in (prefix + HOLD_BLOCK, prefix + HOLD_BEFORE) if n in names), None
        )
        if prefix + GUIDE_AUDIO_BLOCK not in names and audio_before:
            sub_blocks.insert(
                prefix + GUIDE_AUDIO_BLOCK, audio_block(), names.index(audio_before)
            )
        inserted = True
    return inserted


def takes_guides(pipeline):
    """Whether a pipeline's block graph carries the guide layout."""
    if layout_anchor_problem():
        return False
    layout_block = guide_blocks()[0]
    return any(
        isinstance(sequence.sub_blocks.get(prefix + LAYOUT_STEP), layout_block)
        for prefix, sequence in core_denoise_sequences(pipeline)
    )


def guides_refusal(pipeline):
    """Why this pipeline cannot take `guides`, or None."""
    missing = layout_anchor_problem()
    if missing:
        return (
            f"guides need diffusers' MiniMax-H3 '{missing}', which the installed "
            f"diffusers does not have"
        )
    if not takes_guides(pipeline):
        return (
            f"guides is a MiniMax-H3 t2va/fl2va argument, and "
            f"{type(pipeline).__name__} has no '{LAYOUT_STEP}' step to take them"
        )
    return None
