"""MiniMax-H3 time-windowed refine (#1199 Task 17 Step 4, #1188): EXPERIMENTAL.

A spike, not a public surface. `experimental_refine_windows` on a
`refine_strength` call (`latents`, or an unmasked `source_video` - a finished
take as pixels, encoded by the regional edit's block - plus `hold_audio`)
re-denoises the take in
overlapping windows of latent frames instead of one full-sequence forward,
and fuses the windows' predictions into the full-sequence update at every
step. It is the refine tiled in time instead of space (`refine_tiles`,
dw/pipeline_processors/h3_tiles.py), with one difference that decides where it
lives: the tiles are whole pipeline calls blended once at the end, and the
windows are fused at *every* step, so the fusion has to be inside the denoise
loop.

Off unless `DW_EXPERIMENTAL_REFINE_WINDOWS=1` is set where the pipeline loads.
Without it the graph is stock and a step passing the argument is refused at the
call (`windows_refusal`).

Where it sits. H3's denoise loop is two sub-blocks, `denoiser` (one
transformer forward over the whole packed sequence) and `update` (one
scheduler step of the generated rows). `insert_refine_windows` replaces
`denoiser` with `DwH3WindowedLoopDenoiser`, a subclass that runs the stock
forward when the argument is absent and otherwise:

- for each window `(start, stop)` of `window_plan`, gathers a sub-sequence of
  the packed one - every text row, the window's generated video rows (frame-
  major, so a contiguous run), and the held audio rows whose rotary time lies
  in the window's span on the shared clock (`window_rows`). The sub-sequence
  keeps the global row order, so the target video is still the trailing run
  dw's sparse attention lays out. Each row keeps its own `position_ids`,
  `token_tags` and `timestep_indices` (`positions: "true"`), or has its media
  time shifted so the window starts where the take does (`"rebased"`);
- runs the transformer on that sub-sequence;
- accumulates its video velocity weighted by `fuse_weights` (normalized to 1
  at every latent) into a full-size `noise_pred`.

`update` then steps the whole sequence once, unchanged. The flow-matching
step is linear in the velocity, so fusing the predictions is fusing the
updates. The audio stays held, as in any refine: every audio row is a
conditioning row at t = 1, `update` steps none of them, so the audio
prediction is zeros.

With one window covering the whole take the sub-sequence is the sequence
itself, every weight is 1, and the result is bit-identical to the stock step
(tests/test_h3_windows.py).

Imports diffusers when it loads, so nothing imports it at module level: the
loader (dw/pipeline_processors/components.py) and the run-time check
(dw/pipeline_processors/pipeline_h3.py) import it on first use.
"""

import inspect
import logging
import math
import os

import torch
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam

from .h3_hold import core_denoise_sequences
from .h3_rules import (
    HELD_ROWS,
    HOLD_AUDIO_INPUT,
    REFINE_BEFORE,
    REFINE_STRENGTH_INPUT,
    REFINE_TILES_INPUT,
    MASK_INPUTS,
    SOURCE_VIDEO_INPUT,
)

logger = logging.getLogger("dw")

REFINE_WINDOWS_INPUT = "experimental_refine_windows"
ENABLE_ENV_VAR = "DW_EXPERIMENTAL_REFINE_WINDOWS"
# The denoise loop's forward sub-block, which the windowed one replaces
DENOISER = "denoiser"
BLENDS = ("pyramid", "cosine")
POSITIONS = ("true", "rebased")
_KEYS = ("frames", "overlap", "blend", "positions")


def enabled():
    """Whether the scratch flag is on in this process."""
    return os.environ.get(ENABLE_ENV_VAR) == "1"


# 1. The plan and the weights - pure, on the latent time grid


def window_plan(num_latent_frames, window, overlap):
    """`[(start, stop)]` windows of `window` latent frames over a take of
    `num_latent_frames`, each starting `window - overlap` after the one before
    on a whole-latent boundary. The last window is pulled back to end on the
    last latent rather than padded, so it overlaps the one before by `overlap`
    or more. A take no longer than one window is one window.

    Raises:
        ValueError: If the sizes are not whole numbers with
            1 <= window and 0 <= overlap < window, or the take is empty
    """
    for name, value in (("window", window), ("overlap", overlap)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{name} is a whole number of latent frames, got {value!r}"
            )
    if num_latent_frames < 1:
        raise ValueError(f"the take has no latent frames ({num_latent_frames})")
    if window < 1 or not 0 <= overlap < window:
        raise ValueError(
            f"a window is 1 or more latent frames and its overlap is 0 or more "
            f"and under the window, got window {window} and overlap {overlap}"
        )
    if num_latent_frames <= window:
        return [(0, num_latent_frames)]
    plan, start = [], 0
    while start + window < num_latent_frames:
        plan.append((start, start + window))
        start += window - overlap
    last = (num_latent_frames - window, num_latent_frames)
    if last != plan[-1]:
        plan.append(last)
    return plan


def _ramp(length, blend):
    """`length` rising weights in (0, 1): j/(n+1) for a pyramid,
    (1 - cos(pi j/(n+1)))/2 for a cosine, j = 1..n. Either way the rise of one
    window and the fall of its neighbour over the same frames sum to 1."""
    steps = torch.arange(1, length + 1, dtype=torch.float64) / (length + 1)
    if blend == "cosine":
        return 0.5 - 0.5 * torch.cos(math.pi * steps)
    return steps


def fuse_weights(plan, num_latent_frames, blend="pyramid"):
    """Per window of `plan`, a float64 weight per latent frame it covers,
    normalized so the weights of every window covering a latent sum to 1.

    A window ramps up over the frames it shares with the window before and
    down over the frames it shares with the window after, and is 1 elsewhere
    and at the take's own ends. The ramp never reaches 0, so a frame only one
    window covers keeps that window's prediction whole.

    Raises:
        ValueError: If `blend` is not one of BLENDS, or a latent is uncovered
    """
    if blend not in BLENDS:
        raise ValueError(f"blend is one of {', '.join(BLENDS)}, got {blend!r}")
    raw = []
    for index, (start, stop) in enumerate(plan):
        weights = torch.ones(stop - start, dtype=torch.float64)
        if index > 0:
            before = max(0, min(plan[index - 1][1], stop) - start)
            weights[:before] *= _ramp(before, blend)
        if index + 1 < len(plan):
            after = max(0, stop - max(plan[index + 1][0], start))
            if after:
                weights[stop - start - after :] *= _ramp(after, blend).flip(0)
        raw.append(weights)
    total = torch.zeros(num_latent_frames, dtype=torch.float64)
    for (start, stop), weights in zip(plan, raw):
        total[start:stop] += weights
    if bool((total <= 0).any()):
        raise ValueError("the windows leave a latent frame uncovered")
    return [weights / total[start:stop] for (start, stop), weights in zip(plan, raw)]


# 2. The rules - what the call needs, checked before it runs


def window_problems(arguments, has_cache=False):
    """Why this step's `experimental_refine_windows` cannot run - a list,
    empty when it can. Pure: the settings, and the refine it rides on."""
    settings = arguments.get(REFINE_WINDOWS_INPUT)
    if settings is None:
        return []
    name = REFINE_WINDOWS_INPUT
    problems = []
    if not isinstance(settings, dict):
        return [f"{name} is {{frames, overlap}} in latent frames, got {settings!r}"]
    extra = sorted(set(settings) - set(_KEYS))
    if extra:
        problems.append(f"{name} takes {', '.join(_KEYS)}, not {', '.join(extra)}")
    frames, overlap = settings.get("frames"), settings.get("overlap", 0)
    try:
        window_plan(max(1, frames if isinstance(frames, int) else 1), frames, overlap)
    except (TypeError, ValueError) as error:
        problems.append(f"{name}: {error}")
    if settings.get("blend", "pyramid") not in BLENDS:
        problems.append(f"{name} blend is one of {', '.join(BLENDS)}")
    if settings.get("positions", "true") not in POSITIONS:
        problems.append(f"{name} positions is one of {', '.join(POSITIONS)}")
    if arguments.get(REFINE_STRENGTH_INPUT) is None:
        problems.append(f"{name} windows a refine_strength pass - add one")
    # A take as latents, or as pixels through the regional edit's encode
    # with no mask - the whole source re-denoised (h3_mask.py)
    take = any(arguments.get(n) is not None for n in ("latents", SOURCE_VIDEO_INPUT))
    if not take or arguments.get(HOLD_AUDIO_INPUT) is None:
        problems.append(
            f"{name} refines 'latents' or an unmasked 'source_video' with "
            f"'hold_audio' held"
        )
    for other in (REFINE_TILES_INPUT, *MASK_INPUTS):
        if arguments.get(other) is not None:
            problems.append(f"{name} does not take {other} yet")
    if has_cache:
        problems.append(
            f"{name} runs the transformer once per window per step, and a step "
            f"cache would reuse one window's result for another - drop 'cache'"
        )
    return problems


def windows_refusal(pipeline, arguments, has_cache=False):
    """The first reason this call cannot window its refine, or None."""
    if arguments.get(REFINE_WINDOWS_INPUT) is None:
        return None
    if not windows_refine(pipeline):
        return (
            f"{REFINE_WINDOWS_INPUT} is experimental and off in this process - "
            f"start the server with {ENABLE_ENV_VAR}=1"
        )
    problems = window_problems(arguments, has_cache)
    return problems[0] if problems else None


# 3. The graph change


def _loops(pipeline):
    for prefix, sequence in core_denoise_sequences(pipeline):
        loop = sequence.sub_blocks.get(prefix + REFINE_BEFORE)
        if hasattr(loop, "sub_blocks") and DENOISER in loop.sub_blocks:
            yield loop


def insert_refine_windows(pipeline):
    """Replace every H3 denoise loop's `denoiser` with the windowed one, in
    place and keeping its transformer. Idempotent. Returns whether the
    pipeline now windows a refine."""
    inserted = False
    for loop in _loops(pipeline):
        current = loop.sub_blocks[DENOISER]
        if not isinstance(current, DwH3WindowedLoopDenoiser):
            # OrderedDict assignment keeps the key's place before `update`
            loop.sub_blocks[DENOISER] = DwH3WindowedLoopDenoiser(
                current.transformer_name
            )
        inserted = True
    return inserted


def windows_refine(pipeline):
    """Whether a pipeline's denoise loops carry the windowed denoiser."""
    loops = list(_loops(pipeline))
    return bool(loops) and all(
        isinstance(loop.sub_blocks[DENOISER], DwH3WindowedLoopDenoiser)
        for loop in loops
    )


# 4. One window of the packed sequence


def window_rows(
    fields, video_rows, num_condition_video_rows, num_latent_frames, window
):
    """`(video, audio, sequence)` for one window `(start, stop)`: the local
    rows of `latents` and `audio_latents` it covers, and their packed sequence
    positions with every text row, sorted - the global order, kept.

    The video rows are frame-major after the conditioning ones. The audio rows
    are the ones whose rotary time lies in the window's span of the shared
    clock: from its first frame's time to the next window frame's, or to the
    end for the last frame of the take."""
    start, stop = window
    position_ids = fields["position_ids"]
    video_indices = fields["video_indices"]
    audio_indices = fields["audio_indices"]
    generated = video_rows - num_condition_video_rows
    if generated % num_latent_frames:
        raise ValueError(
            f"{generated} generated video rows do not split into "
            f"{num_latent_frames} latent frames"
        )
    per_frame = generated // num_latent_frames
    first = num_condition_video_rows
    video = torch.arange(
        first + start * per_frame, first + stop * per_frame, device=video_indices.device
    )
    times = position_ids[:, 0]
    begin = times[video_indices[video[0]]]
    audio_times = times[audio_indices]
    inside = audio_times >= begin
    if stop < num_latent_frames:
        end = times[video_indices[first + stop * per_frame]]
        inside &= audio_times < end
    audio = inside.nonzero().squeeze(1)
    sequence = (
        torch.cat([fields["text_indices"], audio_indices[audio], video_indices[video]])
        .sort()
        .values
    )
    return video, audio, sequence


def window_inputs(fields, plan_entry, window_rows_, rebase_from=None):
    """The transformer's layout kwargs for one window's sub-sequence:
    `position_ids`, `token_tags`, `timestep_indices` gathered at its packed
    positions, and the three index tensors renumbered into it. With
    `rebase_from` (the take's first generated video time), every media row's
    time is shifted so the window's first frame lands there."""
    video, audio, sequence = window_rows_
    unique_timesteps, timestep_indices = plan_entry
    position_ids = fields["position_ids"][sequence]
    local = {
        name: torch.searchsorted(sequence, indices)
        for name, indices in (
            ("video_indices", fields["video_indices"][video]),
            ("audio_indices", fields["audio_indices"][audio]),
            ("text_indices", fields["text_indices"]),
        )
    }
    if rebase_from is not None:
        offset = position_ids[local["video_indices"][0], 0] - rebase_from
        media = torch.cat([local["audio_indices"], local["video_indices"]])
        position_ids = position_ids.clone()
        position_ids[media, 0] -= offset
    return {
        "position_ids": position_ids,
        "token_tags": fields["token_tags"][sequence],
        **local,
    }, (unique_timesteps, timestep_indices[sequence])


class DwH3WindowedLoopDenoiser(MiniMaxH3LoopDenoiser):
    model_name = "minimax-h3"

    @property
    def description(self):
        return (
            "dw (experimental): with `experimental_refine_windows`, runs the forward on overlapping windows of "
            "latent frames - each with the text, its video rows and the held audio rows of its span - and fuses "
            "the video velocities into the full-sequence prediction. The stock forward otherwise."
        )

    @property
    def inputs(self):
        return super().inputs + [
            InputParam(
                name=REFINE_WINDOWS_INPUT,
                type_hint=dict,
                default=None,
                description=(
                    "Experimental: {frames, overlap} in latent frames (blend pyramid|cosine, positions "
                    "true|rebased) - refine in fused time windows. Needs refine_strength and hold_audio."
                ),
            ),
            InputParam(name="num_latent_frames", type_hint=int, default=None),
            InputParam(name="num_condition_video_rows", type_hint=int, default=0),
            InputParam(name="num_condition_audio_rows", type_hint=int, default=0),
            InputParam(name=HELD_ROWS, type_hint=int, default=0),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        settings = getattr(block_state, REFINE_WINDOWS_INPUT, None)
        if settings is None:
            return super().__call__(components, block_state, i, t)
        return self._windowed(components, block_state, i, settings)

    def _check_rows(self, block_state):
        audio_rows = block_state.audio_latents.shape[0]
        held = getattr(block_state, HELD_ROWS, 0) or 0
        if block_state.num_condition_video_rows or held != audio_rows:
            raise ValueError(
                f"{REFINE_WINDOWS_INPUT} windows a t2va refine whose every audio row is "
                f"the held track: {block_state.num_condition_video_rows or 0} conditioning "
                f"video rows and {held} of {audio_rows} audio rows held"
            )

    def _windowed(self, components, block_state, i, settings):
        self._check_rows(block_state)
        transformer = getattr(components, self.transformer_name)
        accepted = inspect.signature(transformer.forward).parameters
        fields = block_state.denoiser_input_fields
        num_latent_frames = block_state.num_latent_frames
        plan = window_plan(
            num_latent_frames, settings["frames"], settings.get("overlap", 0)
        )
        weights = fuse_weights(
            plan, num_latent_frames, settings.get("blend", "pyramid")
        )
        latents, audio_latents = block_state.latents, block_state.audio_latents
        rebase_from = None
        if settings.get("positions", "true") == "rebased":
            first = fields["video_indices"][0]
            rebase_from = fields["position_ids"][first, 0]
        if i == 0:
            logger.info(f"Refining in {len(plan)} time windows: {plan}")

        fused, total = None, None
        for (start, stop), window_weights in zip(plan, weights):
            rows = window_rows(
                fields,
                latents.shape[0],
                block_state.num_condition_video_rows or 0,
                num_latent_frames,
                (start, stop),
            )
            layout, (timestep, timestep_indices) = window_inputs(
                fields, block_state.row_timestep_plan[i], rows, rebase_from
            )
            video, audio, _ = rows
            noise_pred, _ = transformer(
                hidden_states=latents[video][None],
                audio_hidden_states=audio_latents[audio][None],
                encoder_hidden_states=block_state.prompt_embeds,
                timestep=timestep,
                timestep_indices=timestep_indices,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **{k: v for k, v in layout.items() if k in accepted},
            )
            if fused is None:
                fused = torch.zeros(
                    (1, latents.shape[0], noise_pred.shape[-1]),
                    dtype=torch.float32,
                    device=noise_pred.device,
                )
                total = torch.zeros(
                    latents.shape[0], dtype=torch.float32, device=fused.device
                )
            per_row = window_weights.to(fused.device, torch.float32).repeat_interleave(
                video.numel() // (stop - start)
            )
            fused[0, video] += noise_pred[0].float() * per_row[:, None]
            total[video] += per_row
        # The weights sum to 1 per latent already: this only re-normalizes
        # float32 rounding, and is exactly 1 for a single window
        block_state.noise_pred = fused / total[None, :, None]
        block_state.audio_noise_pred = torch.zeros(
            (1, *audio_latents.shape), dtype=torch.float32, device=fused.device
        )
        return components, block_state
