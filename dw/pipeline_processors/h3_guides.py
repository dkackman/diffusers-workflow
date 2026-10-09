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

The hold blocks are dw/pipeline_processors/h3_hold.py, which this module builds
on and calls through the module (`h3_hold.encode_audio_span`), never the reverse;
the pure guide rules are dw/pipeline_processors/h3_rules.py.
"""

import logging

import torch

from . import h3_hold
from .h3_hold import as_channels_samples, core_denoise_sequences
from .h3_rules import (
    GUIDE_AUDIO_BLOCK,
    GUIDE_AUDIO_ROWS,
    GUIDE_CONDITION_BLOCK,
    GUIDE_FRAMES_PER_CHUNK,
    GUIDE_LATENTS_BLOCK,
    GUIDE_LATENTS_PER_CHUNK,
    GUIDES_INPUT,
    HOLD_BEFORE,
    HOLD_BLOCK,
    LAYOUT_STEP,
    guide_end_problem,
    guide_frame_problem,
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


def guide_position_ids(target_time, frame_grid, latent_index, num_latent_frames):
    """`(num_latent_frames * rows_per_frame, 3)` rotary positions of one guide.

    Time runs from the target's own time at `latent_index`, with the target's
    spacing (`_temporal_position_grid`); every frame carries the target's (h, w)
    grid. Rows are frame-major, the order `patchify_video_latents` packs them in.
    """
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        _temporal_position_grid,
    )

    times = _temporal_position_grid(num_latent_frames, float(target_time[latent_index]))
    rows_per_frame = frame_grid.shape[0]
    positions = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    positions[:, :, 0] = times[:, None]
    positions[:, :, 1:] = frame_grid[None]
    return positions.reshape(-1, 3)


def splice_guide_rows(layout, num_text_tokens, guide_positions, video_tag):
    """The packed layout with guide rows inserted after the existing condition rows.

    `layout` is `(position_ids, token_tags, video_indices, audio_indices,
    text_indices, num_condition_video_rows)` as the stock layout built it, on the
    CPU. Returns the same six, widened by `len(guide_positions)` video rows that
    sit between the keyframe rows and the target audio.
    """
    position_ids, token_tags, video_indices, audio_indices, text_indices, rows = layout
    count = guide_positions.shape[0]
    if not count:
        return layout
    at = num_text_tokens + rows
    sequence_length = position_ids.shape[0] + count
    position_ids = torch.cat(
        [position_ids[:at], guide_positions.to(position_ids.dtype), position_ids[at:]]
    )
    token_tags = torch.cat(
        [
            token_tags[:at],
            torch.full((count,), video_tag, dtype=token_tags.dtype),
            token_tags[at:],
        ]
    )
    audio_start = int(audio_indices[0]) if audio_indices.numel() else at
    video_start = audio_start + audio_indices.numel()
    video_indices = torch.cat(
        [
            torch.arange(num_text_tokens, audio_start + count),
            torch.arange(video_start + count, sequence_length),
        ]
    )
    audio_indices = audio_indices + count
    return (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        rows + count,
    )


def guide_audio_latents(num_frames, fps=24, latents_per_second=40):
    """M, the audio latents a guide of `num_frames` frames holds at frame 0:
    round(num_frames * 40 / 24). 22 -> 37, 39 -> 65."""
    return int(round(num_frames * latents_per_second / fps))


def guide_audio_span(frame, num_frames, fps=24, latents_per_second=40):
    """(start, pad, count) of a guide's audio on the target audio grid.

    `start` is the target audio latent the guide's audio lands on, the one at or
    before its first frame: floor(frame * 40 / 24). `pad` is the seconds of
    silence that go in front of the guide's audio so its first sample keeps its
    time - frame / 24 - once the encode starts at that latent. `count` is the
    latents per channel it covers, to round((frame + num_frames) * 40 / 24); M at
    frame 0 (`guide_audio_latents`)."""
    start = frame * latents_per_second // fps
    pad = frame / fps - start / latents_per_second
    end = guide_audio_latents(frame + num_frames, fps, latents_per_second)
    return start, pad, end - start


def pad_guide_audio(waveform, frame, sample_rate, fps=24, latents_per_second=40):
    """A guide's `(channels, samples)` waveform, at `sample_rate`, with the
    silence in front that puts its first sample at `frame / fps` when sample 0
    is the guide's start latent (`guide_audio_span`) - the audio VAE's hop."""
    start = frame * latents_per_second // fps
    origin = int(round(frame * sample_rate / fps))
    pad = origin - start * (sample_rate // latents_per_second)
    return torch.nn.functional.pad(waveform, (pad, 0)) if pad > 0 else waveform


def guide_audio_positions(num_text_tokens, width_grid, start, count, audio_channels):
    """`(count * audio_channels, 3)` rotary positions of one guide's audio rows:
    the target soundtrack's own, from its latent `start` - time num_text + start
    on, channel-major on the two extremes of the width grid."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        _fill_audio_positions,
    )

    positions = torch.zeros(count * audio_channels, 3, dtype=torch.float64)
    _fill_audio_positions(
        positions,
        slice(0, positions.shape[0]),
        count,
        float(num_text_tokens + start),
        width_grid,
        audio_channels,
    )
    return positions


def splice_guide_audio_rows(layout, audio_positions, audio_tag):
    """The packed layout with guide audio rows in front of the target audio.

    `layout` is `(position_ids, token_tags, video_indices, audio_indices,
    text_indices)` on the CPU. Returns the same five, widened by
    `len(audio_positions)` audio rows that lead `audio_indices` - the
    condition audio prefix - with every video row after them moved along.
    """
    position_ids, token_tags, video_indices, audio_indices, text_indices = layout
    count = audio_positions.shape[0]
    if not count:
        return layout
    at = int(audio_indices[0])
    position_ids = torch.cat(
        [position_ids[:at], audio_positions.to(position_ids.dtype), position_ids[at:]]
    )
    token_tags = torch.cat(
        [
            token_tags[:at],
            torch.full((count,), audio_tag, dtype=token_tags.dtype),
            token_tags[at:],
        ]
    )
    video_indices = torch.where(
        video_indices >= at, video_indices + count, video_indices
    )
    audio_indices = torch.cat([torch.arange(at, at + count), audio_indices + count])
    return position_ids, token_tags, video_indices, audio_indices, text_indices


def fit_guide_frames(frames, height, width):
    """`(n, height, width, 3)` uint8 frames covering the canvas: scaled to cover
    it and centre-cropped, with the arithmetic H3 fits a follower keyframe with
    (diffusers' `MiniMaxH3BeforeEncodeStep`). A clip already at the canvas
    passes through untouched."""
    import numpy as np
    from PIL import Image

    if frames.shape[1:3] == (height, width):
        return frames
    source_height, source_width = frames.shape[1:3]
    scale = max(width / source_width, height / source_height)
    size = (
        max(width, round(source_width * scale)),
        max(height, round(source_height * scale)),
    )
    left = max(0, (size[0] - width) // 2)
    top = max(0, (size[1] - height) // 2)
    box = (left, top, left + width, top + height)
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS).crop(box)
            )
            for frame in frames
        ]
    )


def guide_audio_waveform(value):
    """(waveform, sample_rate) of a guide video's soundtrack: a `(channels,
    samples)` float32 tensor, from what a guide's `video` loads to with its audio -
    an AudioVideo, a step's Selected pick of one or a one-video list of one.

    Raises:
        ValueError: If the video carries no audio
    """
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


def _make_guide_blocks():
    """The guide layout and the two t2va condition steps, defined against the
    installed diffusers."""
    import numpy as np
    from diffusers.models import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    from diffusers.modular_pipelines.minimax_h3 import before_denoise
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
    from diffusers.modular_pipelines.modular_pipeline_utils import (
        ComponentSpec,
        InputParam,
        OutputParam,
    )

    setup_step, reference_encoder, _, latents_per_second = h3_hold._diffusers()
    stock_layout = before_denoise.MiniMaxH3PrepareLayoutStep
    stock_condition = before_denoise.MiniMaxH3PrepareConditionLatentsStep
    stock_latents = before_denoise.MiniMaxH3FL2VAPrepareLatentsStep

    class DwH3GuideLayoutStep(ModularPipelineBlocks):
        model_name = "minimax-h3"

        def __init__(self):
            super().__init__()
            # Called, not copied: the stock layout resolves the canvas and builds
            # the sequence, and the guides are spliced into what it built
            self._layout = stock_layout()
            # ...and the encode a Ref2VA soundtrack reference takes, for a
            # guide's audio
            self._audio_encoder = reference_encoder()

        @property
        def description(self):
            return (
                "dw: the stock layout, plus `guides` - each guide clip VAE-encoded on the canvas and appended as "
                "condition rows timed from the target frame it lands on. Without `guides` it is the stock layout."
            )

        @property
        def expected_components(self):
            return list(self._layout.expected_components) + [
                ComponentSpec("vae", AutoencoderKLMiniMaxH3),
                ComponentSpec("audio_vae", AutoencoderKLMiniMaxH3Audio),
            ]

        @property
        def expected_configs(self):
            return self._layout.expected_configs

        @property
        def inputs(self):
            return list(self._layout.inputs) + [
                InputParam(
                    name=GUIDES_INPUT,
                    type_hint=list,
                    default=None,
                    description=(
                        "Clips to hold the generated video to: a list of {'video': (n, h, w, 3) uint8 frames, "
                        "'frame': pixel frame, a multiple of 17}, n one of 1, 5 or 17m + 5. A guide that also "
                        "carries 'audio': (channels, samples) and 'sample_rate' holds that soundtrack over its span."
                    ),
                ),
                InputParam(name="condition_latents", type_hint=list, default=None),
            ]

        @property
        def intermediate_outputs(self):
            return list(self._layout.intermediate_outputs) + [
                OutputParam("condition_latents", type_hint=list),
                OutputParam(GUIDE_AUDIO_ROWS, type_hint=torch.Tensor),
            ]

        @torch.no_grad()
        def __call__(self, components, state):
            components, state = self._layout(components, state)
            guides = state.get(GUIDES_INPUT) or []
            if not guides:
                return components, state
            # What the stock layout set is its output, not this block's input,
            # so it is read off the state rather than a block state
            laid = state.get(
                [
                    "height",
                    "width",
                    "num_frames",
                    "text_token_tags",
                    "latent_height",
                    "latent_width",
                    "num_latent_frames",
                    "position_ids",
                    "token_tags",
                    "video_indices",
                    "audio_indices",
                    "text_indices",
                    "num_condition_video_rows",
                    "condition_latents",
                    "num_audio_latents",
                ]
            )

            device = components._execution_device
            num_text = laid["text_token_tags"].shape[0]
            _, patch_h, patch_w = components.patch_size
            frame_grid, width_grid = before_denoise._frame_position_grid(
                laid["latent_height"], laid["latent_width"], patch_h, patch_w
            )
            target_time = before_denoise._temporal_position_grid(
                laid["num_latent_frames"], float(num_text)
            )

            latents, positions = [], []
            for index, guide in enumerate(guides):
                frame, frames = guide["frame"], guide["video"]
                problem = guide_frame_problem(frame) or guide_end_problem(
                    frame, frames.shape[0], laid["num_frames"]
                )
                if problem:
                    raise ValueError(f"guides[{index}]: {problem}")
                pixels = fit_guide_frames(frames, laid["height"], laid["width"])
                pixels = torch.from_numpy(np.ascontiguousarray(pixels))
                encoded = encode_vae_condition(
                    components.vae,
                    pixels.to(device).permute(3, 0, 1, 2)[None],
                    components.pixel_mean,
                    components.pixel_std,
                    components.keyframe_encode_seed,
                )
                latents.append(encoded)
                positions.append(
                    guide_position_ids(
                        target_time,
                        frame_grid,
                        frame // GUIDE_FRAMES_PER_CHUNK * GUIDE_LATENTS_PER_CHUNK,
                        encoded.shape[2],
                    )
                )
                logger.info(
                    f"Guide {index}: {frames.shape[0]} frames at frame {frame} "
                    f"({encoded.shape[2]} latent frames)"
                )

            names = (
                "position_ids",
                "token_tags",
                "video_indices",
                "audio_indices",
                "text_indices",
            )
            layout = splice_guide_rows(
                tuple(laid[name].cpu() for name in names)
                + (laid["num_condition_video_rows"],),
                num_text,
                torch.cat(positions),
                components.video_tag,
            )

            # Each guide's audio, encoded on the target audio grid from the latent
            # at or before its first frame, as condition audio rows timed there
            audio_rows, audio_positions = [], []
            for index, guide in enumerate(guides):
                if guide.get("audio") is None:
                    continue
                rows, start, count = self._guide_audio(
                    components, guide, laid["num_audio_latents"]
                )
                audio_rows.append(rows)
                audio_positions.append(
                    guide_audio_positions(
                        num_text, width_grid, start, count, components.audio_channels
                    )
                )
                logger.info(
                    f"Guide {index}: audio held over audio latents {start} to "
                    f"{start + count} ({rows.shape[0]} rows)"
                )
            num_audio_rows = state.get("num_condition_audio_rows") or 0
            if audio_rows:
                layout = (
                    splice_guide_audio_rows(
                        layout[:5], torch.cat(audio_positions), components.audio_tag
                    )
                    + layout[5:]
                )
                state.set(GUIDE_AUDIO_ROWS, torch.cat(audio_rows))
                num_audio_rows += sum(rows.shape[0] for rows in audio_rows)

            for name, value in zip(names, layout):
                state.set(name, value.to(device))
            state.set("num_condition_video_rows", layout[5])
            state.set("num_condition_audio_rows", num_audio_rows)
            # Keyframes first, then guides: the order the rows were laid out in
            state.set(
                "condition_latents", list(laid["condition_latents"] or []) + latents
            )
            return components, state

        def _guide_audio(self, components, guide, num_audio_latents):
            """(rows, start, count): a guide's audio as `(channels * count, C)`
            channel-major rows from target audio latent `start`."""
            frame, length = guide["frame"], guide["video"].shape[0]
            fps = components.fps
            vae_rate = components.audio_sampling_rate
            start, _, count = guide_audio_span(frame, length, fps, latents_per_second)
            count = min(count, num_audio_latents - start)
            original = as_channels_samples(guide["audio"])
            rate = guide.get("sample_rate") or vae_rate
            # The guide's span only, on the VAE's rate and channels the way a
            # Ref2VA soundtrack is, then pre-padded to the VAE's hop
            waveform = setup_step._normalize_audio_condition(
                original, rate, vae_rate, max_duration=length / fps
            )
            waveform = pad_guide_audio(
                waveform, frame, vae_rate, fps, latents_per_second
            )
            latents = h3_hold.encode_audio_span(
                self._audio_encoder, components, waveform, count
            )
            return latents.reshape(-1, components.audio_latent_channels), start, count

    def _when_conditioned(stock, description):
        """A t2va step that runs the stock `fl2va` one only with conditioning
        to pack - the stock step refuses an empty list."""

        class DwH3WhenConditionedStep(ModularPipelineBlocks):
            model_name = "minimax-h3"

            def __init__(self):
                super().__init__()
                self._stock = stock()

            @property
            def description(self):
                return description

            @property
            def expected_components(self):
                return self._stock.expected_components

            @property
            def inputs(self):
                inputs = []
                for param in self._stock.inputs:
                    if param.name == "condition_latents":
                        param = InputParam(
                            name="condition_latents", type_hint=list, default=None
                        )
                    elif param.name == "condition_rows":
                        param = InputParam(
                            name="condition_rows",
                            type_hint=torch.Tensor,
                            default=None,
                        )
                    inputs.append(param)
                return inputs

            @property
            def intermediate_outputs(self):
                return self._stock.intermediate_outputs

            def __call__(self, components, state):
                # Read off the state: the latents step does not declare it
                if not state.get("condition_latents"):
                    return components, state
                return self._stock(components, state)

        return DwH3WhenConditionedStep

    class DwH3GuideAudioStep(ModularPipelineBlocks):
        model_name = "minimax-h3"

        @property
        def description(self):
            return (
                "dw: puts the guides' encoded audio in front of `audio_latents` as the condition audio prefix the "
                "guide layout reserved. Does nothing when no guide carries audio."
            )

        @property
        def inputs(self):
            return [
                InputParam(name="audio_latents", type_hint=torch.Tensor, required=True),
                InputParam(name="audio_indices", type_hint=torch.Tensor, required=True),
                InputParam(name=GUIDE_AUDIO_ROWS, type_hint=torch.Tensor, default=None),
            ]

        @property
        def intermediate_outputs(self):
            return [OutputParam("audio_latents", type_hint=torch.Tensor)]

        def __call__(self, components, state):
            block_state = self.get_block_state(state)
            rows = getattr(block_state, GUIDE_AUDIO_ROWS)
            audio_latents = block_state.audio_latents
            # Once: the layout counted the rows into `audio_indices` already
            if (
                rows is None
                or audio_latents.shape[0] >= block_state.audio_indices.numel()
            ):
                return components, state
            block_state.audio_latents = torch.cat(
                [rows.to(audio_latents.device, audio_latents.dtype), audio_latents]
            )
            self.set_block_state(state, block_state)
            return components, state

    condition_step = _when_conditioned(
        stock_condition,
        "dw: diffusers' `prepare_condition_latents`, run in `t2va` only when `guides` gave it condition latents.",
    )
    latents_step = _when_conditioned(
        stock_latents,
        "dw: diffusers' `prepare_latents_fl2va`, run in `t2va` only when `guides` gave it condition rows.",
    )
    return DwH3GuideLayoutStep, condition_step, latents_step, DwH3GuideAudioStep


_GUIDE_BLOCKS = None


def guide_blocks():
    """(DwH3GuideLayoutStep, the t2va condition step, the t2va latents step,
    DwH3GuideAudioStep), built once."""
    global _GUIDE_BLOCKS
    if _GUIDE_BLOCKS is None:
        _GUIDE_BLOCKS = _make_guide_blocks()
    return _GUIDE_BLOCKS


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
