"""MiniMax-H3 guides: the four blocks dw puts into the denoise graph for a
`guides` call, and the row and position arithmetic they run.

What a guide is, and where the blocks go, is dw/pipeline_processors/h3_guides.py,
which puts them in. They live here, at module level, rather than in a factory
inside h3_guides (#790): a class defined in a function body is counted into that
function, and the factory read as one 313-line function to the ratchet. This
module imports diffusers when it loads, so nothing imports it at module level -
`h3_guides.guide_blocks()` imports it on first use, and validation, which imports
h3_guides for `default_num_frames`, still never loads a modular pipeline.

The splicing helpers are here too, not in h3_guides, because the layout block
calls them and h3_guides imports this module: a helper left behind would make
the import a cycle. A guide's audio is encoded through the hold blocks' own
`h3_hold_steps.encode_audio_span`, called through the module so a test can stand
in for the encode; h3_hold_steps never imports this module back.

The private diffusers helpers the layout is built on (`_frame_position_grid`,
`_temporal_position_grid`, `_fill_audio_positions`) are read off
`before_denoise` when they run, not imported by name: they are among
`LAYOUT_ANCHORS` in h3_guides, and a diffusers missing one refuses `guides`
there, with the name, rather than failing this import.
"""

import logging

import numpy as np
import torch
from diffusers.models import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from diffusers.modular_pipelines.minimax_h3 import before_denoise, encoders
from diffusers.modular_pipelines.minimax_h3.before_encoder import (
    MiniMaxH3Ref2VASetupStep,
)
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3Ref2VAReferenceEncoderStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
)
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)
from PIL import Image

from . import h3_hold_steps
from .h3_rules import (
    GUIDE_AUDIO_ROWS,
    GUIDE_FRAMES_PER_CHUNK,
    GUIDE_LATENTS_PER_CHUNK,
    GUIDES_INPUT,
    guide_end_problem,
    guide_frame_problem,
)

logger = logging.getLogger("dw")

# What the stock layout sets, read back off the state by the guide layout
_LAID_OUT = (
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
)

# The packed-sequence tensors a splice widens, in `splice_guide_rows` order
_PACKED = (
    "position_ids",
    "token_tags",
    "video_indices",
    "audio_indices",
    "text_indices",
)


def guide_position_ids(target_time, frame_grid, latent_index, num_latent_frames):
    """`(num_latent_frames * rows_per_frame, 3)` rotary positions of one guide.

    Time runs from the target's own time at `latent_index`, with the target's
    spacing (`_temporal_position_grid`); every frame carries the target's (h, w)
    grid. Rows are frame-major, the order `patchify_video_latents` packs them in.
    """
    times = before_denoise._temporal_position_grid(
        num_latent_frames, float(target_time[latent_index])
    )
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
    positions = torch.zeros(count * audio_channels, 3, dtype=torch.float64)
    before_denoise._fill_audio_positions(
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


class DwH3GuideLayoutStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    def __init__(self):
        super().__init__()
        # Called, not copied: the stock layout resolves the canvas and builds
        # the sequence, and the guides are spliced into what it built
        self._layout = before_denoise.MiniMaxH3PrepareLayoutStep()
        # ...and the encode a Ref2VA soundtrack reference takes, for a
        # guide's audio
        self._audio_encoder = MiniMaxH3Ref2VAReferenceEncoderStep()

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
        laid = state.get(list(_LAID_OUT))

        device = components._execution_device
        num_text = laid["text_token_tags"].shape[0]
        _, patch_h, patch_w = components.patch_size
        frame_grid, width_grid = before_denoise._frame_position_grid(
            laid["latent_height"], laid["latent_width"], patch_h, patch_w
        )
        latents, positions = self._encode_guides(
            components, guides, laid, frame_grid, num_text
        )
        layout = splice_guide_rows(
            tuple(laid[name].cpu() for name in _PACKED)
            + (laid["num_condition_video_rows"],),
            num_text,
            torch.cat(positions),
            components.video_tag,
        )
        layout, num_audio_rows = self._hold_guide_audio(
            components, state, guides, layout, laid, width_grid, num_text
        )

        for name, value in zip(_PACKED, layout):
            state.set(name, value.to(device))
        state.set("num_condition_video_rows", layout[5])
        state.set("num_condition_audio_rows", num_audio_rows)
        # Keyframes first, then guides: the order the rows were laid out in
        state.set("condition_latents", list(laid["condition_latents"] or []) + latents)
        return components, state

    def _encode_guides(self, components, guides, laid, frame_grid, num_text):
        """([latents], [positions]): each guide VAE-encoded on the canvas, and
        its rows' rotary positions, timed from the target frame it lands on."""
        device = components._execution_device
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
            # Through the module, so the encode is the one installed when it runs
            encoded = encoders.encode_vae_condition(
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
        return latents, positions

    def _hold_guide_audio(
        self, components, state, guides, layout, laid, width_grid, num_text
    ):
        """(layout, num_condition_audio_rows): each guide's audio, encoded on the
        target audio grid from the latent at or before its first frame, spliced
        in as condition audio rows timed there, and left in the state for
        `DwH3GuideAudioStep`."""
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
        return layout, num_audio_rows

    def _guide_audio(self, components, guide, num_audio_latents):
        """(rows, start, count): a guide's audio as `(channels * count, C)`
        channel-major rows from target audio latent `start`."""
        frame, length = guide["frame"], guide["video"].shape[0]
        fps = components.fps
        vae_rate = components.audio_sampling_rate
        latents_per_second = MINIMAX_H3_AUDIO_LATENTS_PER_SECOND
        start, _, count = guide_audio_span(frame, length, fps, latents_per_second)
        count = min(count, num_audio_latents - start)
        original = h3_hold_steps.as_channels_samples(guide["audio"])
        rate = guide.get("sample_rate") or vae_rate
        # The guide's span only, on the VAE's rate and channels the way a
        # Ref2VA soundtrack is, then pre-padded to the VAE's hop
        waveform = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
            original, rate, vae_rate, max_duration=length / fps
        )
        waveform = pad_guide_audio(waveform, frame, vae_rate, fps, latents_per_second)
        latents = h3_hold_steps.encode_audio_span(
            self._audio_encoder, components, waveform, count
        )
        return latents.reshape(-1, components.audio_latent_channels), start, count


# A t2va step that runs the stock `fl2va` one only with conditioning to pack -
# the stock step refuses an empty list. `_when_conditioned` makes one per stock
# step, each under this one name, as the factory before #790 did
class DwH3WhenConditionedStep(ModularPipelineBlocks):
    model_name = "minimax-h3"
    stock = None
    stock_description = None

    def __init__(self):
        super().__init__()
        self._stock = self.stock()

    @property
    def description(self):
        return self.stock_description

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


def _when_conditioned(stock, description):
    """A `DwH3WhenConditionedStep` running the stock step `stock`."""
    return type(
        DwH3WhenConditionedStep.__name__,
        (DwH3WhenConditionedStep,),
        {"stock": stock, "stock_description": description},
    )


CONDITION_STEP = _when_conditioned(
    before_denoise.MiniMaxH3PrepareConditionLatentsStep,
    "dw: diffusers' `prepare_condition_latents`, run in `t2va` only when `guides` gave it condition latents.",
)
LATENTS_STEP = _when_conditioned(
    before_denoise.MiniMaxH3FL2VAPrepareLatentsStep,
    "dw: diffusers' `prepare_latents_fl2va`, run in `t2va` only when `guides` gave it condition rows.",
)


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
        if rows is None or audio_latents.shape[0] >= block_state.audio_indices.numel():
            return components, state
        block_state.audio_latents = torch.cat(
            [rows.to(audio_latents.device, audio_latents.dtype), audio_latents]
        )
        self.set_block_state(state, block_state)
        return components, state
