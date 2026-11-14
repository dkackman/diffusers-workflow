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

This module puts the guide blocks in, and reads each `guides` entry a step
passes into the arrays they take (`prepared_guide`); the blocks themselves, and the row and
position arithmetic they run (`splice_guide_rows`, `guide_audio_span`), are
dw/pipeline_processors/h3_guide_steps.py, imported in `guide_blocks()` only,
since it imports diffusers when it loads (#790). The hold blocks are
dw/pipeline_processors/h3_hold.py and h3_hold_steps.py, which these modules build
on and call through the module (`h3_hold_steps.encode_audio_span`), never the
reverse; the pure guide rules are dw/pipeline_processors/h3_rules.py.
"""

import logging

import torch

from ..events import emit_log, emit_warning
from ..tasks.video_utils import load_audio_video
from .h3_hold import core_denoise_sequences
from .h3_rules import (
    CAPSTONE_GATE,
    CAPSTONE_PAST_END,
    GUIDE_AUDIO_BLOCK,
    GUIDE_CONDITION_BLOCK,
    GUIDE_LATENTS_BLOCK,
    HOLD_BEFORE,
    HOLD_BLOCK,
    KEYSTONE_LATENTS,
    KEYSTONE_RESTORE_BLOCK,
    LAYOUT_STEP,
    RELEASE_BEFORE,
    REFINE_WINDOWS_ENV_VAR,
    GUIDE_FRAMES_PER_CHUNK,
    GUIDE_LATENTS_PER_CHUNK,
    H3_FPS,
    guide_frame_problem,
    guide_latent_frames,
    refine_windows_enabled,
    snap_guide_length,
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

# H3's audio latents run at 40 a second against its 24 fps, so a segment lands
# on the audio grid only at a frame count divisible by 3 - on the 17n+5 render
# grid, these lengths. A held chain off it puts each seam mid-latent
# (`held_audio_warnings` in dw/guides.py).
AUDIO_GRID_FRAMES = 3
AUDIO_GRID_LENGTHS = (141, 192, 243, 294, 345)


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


# A source rate within this fraction of H3_FPS is H3_FPS: NTSC 23.976, or the
# 23.899 an AAC-padded soundtrack's length gives, is not worth a retime (#1232)
H3_FPS_TOLERANCE = 0.005


def grid_rate(fps):
    """`fps` as the audio pass treats it: H3_FPS when it is within
    `H3_FPS_TOLERANCE` of it (no retime), else `fps` itself. None and anything
    not a positive number pass through."""
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        return fps
    return H3_FPS if abs(fps - H3_FPS) <= H3_FPS * H3_FPS_TOLERANCE else fps


def source_frame_rate(value, frame_count):
    """The rate a video argument plays at: the rate it carries at any level a
    step's result wraps it in (a FrameList's, an AudioVideo's, a
    FittedVideo's), else the rate its own soundtrack's length gives its
    `frame_count` frames, else None - what `guide_frames_array` drops."""
    from ..media_types import sample_axis

    seconds = None
    while value is not None:
        fps = getattr(value, "fps", None)
        if isinstance(fps, (int, float)) and not isinstance(fps, bool) and fps > 0:
            return grid_rate(fps)
        audio, rate = getattr(value, "audio", None), getattr(value, "sample_rate", None)
        axis = sample_axis(audio) if hasattr(audio, "ndim") else None
        if seconds is None and axis is not None and rate and audio.shape[axis]:
            seconds = audio.shape[axis] / rate
        if hasattr(value, "value") and hasattr(value, "position"):
            value = value.value
        elif hasattr(value, "frames") and hasattr(value, "audio"):
            value = value.frames
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        else:
            break
    return grid_rate(frame_count / seconds) if seconds else None


def prepared_guide(where, index, guide, base_dir):
    """One `guides` entry checked and read: its clip as uint8 frames cut to
    a whole-latent length, with its soundtrack when it asks for one."""
    if not isinstance(guide, dict):
        raise ValueError(
            f"{where}[{index}] must be {{video, frame}}, got {type(guide).__name__}"
        )
    unknown = sorted(
        set(guide)
        - {"video", "frame", "audio", *CAPSTONE_KEYS, *LATENT_KEYS, KEYSTONE_LATENTS}
    )
    if unknown:
        raise ValueError(
            f"{where}[{index}] has unknown key(s) {unknown} - a guide is "
            f"{{video, frame}}, with an optional 'audio'"
        )
    capstone = _capstone(f"{where}[{index}]", guide)
    if capstone.get(CAPSTONE_PAST_END):
        if "video" not in guide:
            raise ValueError(f"{where}[{index}] needs 'video'")
    elif "video" not in guide or "frame" not in guide:
        raise ValueError(f"{where}[{index}] needs both 'video' and 'frame'")
    problem = (
        None if capstone.get(CAPSTONE_PAST_END) else guide_frame_problem(guide["frame"])
    )
    if problem:
        raise ValueError(f"{where}[{index}]: {problem}")
    with_audio = guide.get("audio", False)
    if not isinstance(with_audio, bool):
        raise ValueError(
            f"{where}[{index}]: 'audio' must be true or false, got "
            f"{type(with_audio).__name__}"
        )
    video = guide["video"]
    if isinstance(video, str) or (isinstance(video, dict) and with_audio):
        # argument_media imports diffusers, which this module waits for (#790)
        from ..argument_media import fetch_video

        # Read with its soundtrack only when the guide holds it
        video = (
            load_audio_video(video, base_dir)
            if with_audio
            else fetch_video(video, base_dir)
        )
    try:
        frames = guide_frames_array(video)
        audio, sample_rate = guide_audio_waveform(video) if with_audio else (None, None)
    except ValueError as error:
        raise ValueError(f"{where}[{index}]: {error}") from error
    if capstone.get(CAPSTONE_PAST_END) and (frames.shape[0] != 1 or with_audio):
        raise ValueError(
            f"{where}[{index}]: a '{CAPSTONE_PAST_END}' guide is one frame with no "
            f"audio - one latent past the render's end - got {frames.shape[0]} "
            f"frame(s)" + (" and audio" if with_audio else "")
        )
    length = snap_guide_length(frames.shape[0])
    if length != frames.shape[0]:
        emit_warning(
            f"{where}[{index}]: a guide clip encodes to whole latents only at "
            f"1, 5 or 17m + 5 frames, so its {frames.shape[0]} frames are cut "
            f"to the first {length}"
        )
        frames = frames[:length]
    carried = _carried_latents(f"{where}[{index}]", guide, frames.shape[0])
    keystone = _keystone_latents(f"{where}[{index}]", guide)
    return {
        "video": frames,
        "frame": guide.get("frame"),
        "audio": audio,
        "sample_rate": sample_rate,
        **capstone,
        **carried,
        **keystone,
    }


# The experimental latent carry's guide keys (#1150; chain_rules
# CHAIN_LATENT_CARRY): the guide's own latents, (1, C, T, h, w) in the space
# the pipeline's encode gives, and its audio latents, (channels, n, C) rows of
# the target audio from the guide's first frame - taken by the layout in place
# of encoding `video` and `audio`
LATENT_KEYS = ("latents", "audio_latents")
# H3's audio latent rate (diffusers' MINIMAX_H3_AUDIO_LATENTS_PER_SECOND)
H3_AUDIO_LATENTS_PER_SECOND = 40


def _carried_latents(where, guide, num_frames):
    """A guide's latent-carry keys, checked: {} for a pixel guide. A scratch
    spike, refused unless REFINE_WINDOWS_ENV_VAR is on."""
    given = {key: guide[key] for key in LATENT_KEYS if guide.get(key) is not None}
    if not given:
        return {}
    if not refine_windows_enabled():
        raise ValueError(
            f"{where}: {', '.join(given)} is an experimental latent carry, off "
            f"unless {REFINE_WINDOWS_ENV_VAR}=1 is set where the pipeline loads"
        )
    if guide.get(CAPSTONE_PAST_END) or guide.get(CAPSTONE_GATE) is not None:
        raise ValueError(f"{where}: a capstone guide carries no latents")
    if guide.get("audio"):
        raise ValueError(
            f"{where}: 'audio' true encodes the guide's soundtrack, and a latent "
            "guide carries its audio as 'audio_latents' - drop one"
        )
    latents = given.get("latents")
    if latents is not None:
        expected = guide_latent_frames(num_frames)
        if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
            raise ValueError(
                f"{where}: 'latents' is a (1, C, T, h, w) tensor, got "
                f"{getattr(latents, 'shape', type(latents).__name__)}"
            )
        if latents.shape[2] != expected:
            raise ValueError(
                f"{where}: a {num_frames}-frame guide encodes to {expected} latent "
                f"frames, and its 'latents' hold {latents.shape[2]}"
            )
    rows = given.get("audio_latents")
    if rows is not None and (not isinstance(rows, torch.Tensor) or rows.ndim != 3):
        raise ValueError(
            f"{where}: 'audio_latents' is a (channels, n, C) tensor, got "
            f"{getattr(rows, 'shape', type(rows).__name__)}"
        )
    return given


def _keystone_latents(where, guide):
    """A guide's latent keystone (chain keystone_slot "latent"), checked: {}
    without one. A scratch spike, refused unless REFINE_WINDOWS_ENV_VAR is on;
    a `(1, C, 1, h, w)` tensor, written over the guide's latent 0 by the
    layout (h3_guide_steps.keystone_latent_0)."""
    keystone = guide.get(KEYSTONE_LATENTS)
    if keystone is None:
        return {}
    if not refine_windows_enabled():
        raise ValueError(
            f"{where}: {KEYSTONE_LATENTS} is an experimental latent keystone, off "
            f"unless {REFINE_WINDOWS_ENV_VAR}=1 is set where the pipeline loads"
        )
    if guide.get(CAPSTONE_PAST_END) or guide.get(CAPSTONE_GATE) is not None:
        raise ValueError(f"{where}: a capstone guide carries no {KEYSTONE_LATENTS}")
    if (
        not isinstance(keystone, torch.Tensor)
        or keystone.ndim != 5
        or keystone.shape[2] != 1
    ):
        raise ValueError(
            f"{where}: '{KEYSTONE_LATENTS}' is one latent frame, a (1, C, 1, h, w) "
            f"tensor, got {getattr(keystone, 'shape', type(keystone).__name__)}"
        )
    return {KEYSTONE_LATENTS: keystone}


def latent_guide_tail(latents, audio_latents, num_frames, count, back, fps=H3_FPS):
    """(video, audio): the latents of a finished segment's last `count`
    frames before its last `back`, the span a guide chain's pixel guide is cut
    from (chain.GuideContinuity) - `latents[:, :, a : a + guide_latent_frames
    (count)]`, and the audio latents of the same span as `(channels, n, C)`
    rows (None without `audio_latents`).

    `latents` is the segment's `(1, C, T, h, w)` output, T = 5m + 2 for its
    `num_frames` = 17m + 5. The VAE encodes 17-frame clips each on its own
    (clip i is latents 5i..5i+4, the encoder causal within a clip only), so
    a span starting on a clip boundary, frame 17j, is latents from 5j on -
    the same layout the pixel guide's own encode gives. A start off a clip
    is refused. `audio_latents` is the segment's `(channels, C, N)` output,
    N = round(num_frames * 40 / 24); the span is the round(count * 40 / 24)
    latents that end at the span's last frame (the pixel guide's audio is
    cut back from that end too), so its first latent can sit up to two thirds
    of a latent (about 16.7 ms) off the span's first frame (the two roundings)."""
    start = num_frames - back - count
    if start < 0 or start % GUIDE_FRAMES_PER_CHUNK:
        raise ValueError(
            f"A latent carry cuts a {count}-frame tail {back} frames before the "
            f"end of {num_frames}, starting at frame {start}, which is not on a "
            f"{GUIDE_FRAMES_PER_CHUNK}-frame VAE clip"
        )
    if latents.shape[2] != guide_latent_frames(num_frames):
        raise ValueError(
            f"A latent carry needs the segment's {guide_latent_frames(num_frames)} "
            f"latent frames for its {num_frames} frames, and got {latents.shape[2]}"
        )
    first = start // GUIDE_FRAMES_PER_CHUNK * GUIDE_LATENTS_PER_CHUNK
    video = latents[:, :, first : first + guide_latent_frames(count)].clone()
    if audio_latents is None:
        return video, None
    # h3_guide_steps.guide_audio_latents, which imports diffusers to load
    rows = int(round(count * H3_AUDIO_LATENTS_PER_SECOND / fps))
    end = int(round((num_frames - back) * H3_AUDIO_LATENTS_PER_SECOND / fps))
    audio = audio_latents[:, :, max(end - rows, 0) : end].clone().permute(0, 2, 1)
    return video, audio


def latent_level(latents):
    """(mean, spread) of a `(1, C, T, h, w)` latent per channel, each
    `(1, C, 1, 1, 1)`: what `levelled` pulls a later tail back to."""
    values = latents.float()
    return (
        values.mean(dim=(2, 3, 4), keepdim=True),
        values.std(dim=(2, 3, 4), keepdim=True),
    )


def levelled(latents, reference):
    """`latents` with each channel's mean and spread set to `reference`'s (a
    `latent_level`), in the latents' own dtype (#1322): a carried tail pulled
    back to the tone of the first one. A channel with no spread keeps its
    values and takes the mean only."""
    mean, spread = latent_level(latents)
    target_mean, target_spread = reference
    gain = torch.where(spread > 1e-6, target_spread / spread.clamp_min(1e-6), 1.0)
    return ((latents.float() - mean) * gain + target_mean).to(latents.dtype)


# The experimental late capstone's guide keys (#1187; h3_windows.py)
CAPSTONE_KEYS = (CAPSTONE_PAST_END, CAPSTONE_GATE)


def _capstone(where, guide):
    """A guide's capstone keys, checked: {} for a plain guide. Both keys are
    a scratch spike, refused unless REFINE_WINDOWS_ENV_VAR is on, since the
    gate is run by the windowed denoiser that flag puts in."""
    given = {key: guide[key] for key in CAPSTONE_KEYS if key in guide}
    if not given:
        return {}
    if not refine_windows_enabled():
        raise ValueError(
            f"{where}: {', '.join(given)} is an experimental capstone, off unless "
            f"{REFINE_WINDOWS_ENV_VAR}=1 is set where the pipeline loads"
        )
    past_end = given.get(CAPSTONE_PAST_END, False)
    if not isinstance(past_end, bool):
        raise ValueError(f"{where}: '{CAPSTONE_PAST_END}' must be true or false")
    gate = given.get(CAPSTONE_GATE)
    if gate is not None and (
        isinstance(gate, bool)
        or not isinstance(gate, (int, float))
        or not 0 < gate <= 1
    ):
        raise ValueError(
            f"{where}: '{CAPSTONE_GATE}' is a sigma in (0, 1] - the row enters the "
            f"forward once the step's sigma falls below it - got {gate!r}"
        )
    return {
        CAPSTONE_PAST_END: past_end,
        CAPSTONE_GATE: None if gate is None else float(gate),
    }


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


def insert_keystone_restore(pipeline):
    """Put `DwH3KeystoneRestoreStep` in front of `after_denoise` in every
    `t2va`/`fl2va` core-denoise sequence, for the experimental latent keystone;
    it does nothing unless a guide carries `keystone_latents`. The loader puts
    it in behind REFINE_WINDOWS_ENV_VAR only. Idempotent; returns whether any
    sequence has it."""
    from .h3_guide_steps import DwH3KeystoneRestoreStep

    inserted = False
    for prefix, sequence in _layout_sequences(pipeline):
        names = list(sequence.sub_blocks)
        name, before = prefix + KEYSTONE_RESTORE_BLOCK, prefix + RELEASE_BEFORE
        if name not in names and before in names:
            sequence.sub_blocks.insert(
                name, DwH3KeystoneRestoreStep(), names.index(before)
            )
        inserted = inserted or name in sequence.sub_blocks
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


def levelled_tail(carry, level, index):
    """The level a levelled carry holds (#1322): the first tail this run
    carries sets it and goes on as rendered; every later tail is pulled back
    to it before the next segment reads it. The finished frames are untouched."""
    if level is None:
        emit_log(f"Levelled carry: segment {index + 1}'s tail sets the level")
        return latent_level(carry.latents)
    mean, spread = latent_level(carry.latents)
    carry.latents = levelled(carry.latents, level)
    emit_log(
        f"Levelled carry: segment {index + 1}'s tail pulled back - mean off by "
        f"{float((mean - level[0]).abs().mean()):.4f}, spread x"
        f"{float((spread / level[1].clamp_min(1e-6)).mean()):.3f} of the level"
    )
    return level
