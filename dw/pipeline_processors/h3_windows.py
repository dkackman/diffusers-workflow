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

Past 15 s. diffusers' layout step refuses a take over
`MiniMaxH3ModularPipeline.max_duration` (15 s, a read-only property).
`insert_refine_windows` also swaps each t2va/fl2va `prepare_layout` for
`DwH3LongRefineLayoutStep`, which runs the stock step unchanged unless the
call carries both `experimental_refine_windows` and `refine_strength`; then it
hands the stock step a view of `components` whose `max_duration` is the take's
own (`_LongTakeComponents`). Nothing is patched: every other call, and every
other job in the worker, keeps the ceiling, because a generation over 15 s is
another model regime. The upstream change that would replace it is a
`ConfigSpec("max_duration", 15.0)` on the layout step, read as
`components.config.max_duration` like `canvas_max_pixels`.

The decode of a long take. The stock video decode keeps every decoded clip
on the device and converts the whole take to float32, about 36 GB at 120 s
and 960x544. `insert_refine_windows` swaps it for
`DwH3StreamedVideoDecodeStep`, which decodes a piece of the take at a time and
gives the same frames (`streamed_decode`); `decode_h3_latents` decodes with it
too.

Imports diffusers when it loads, so nothing imports it at module level: the
loader (dw/pipeline_processors/components.py) and the run-time check
(dw/pipeline_processors/pipeline_h3.py) import it on first use.
"""

import inspect
import logging
import math

import torch
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    align_num_frames,
)
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam

from .h3_hold import core_denoise_sequences
from .h3_rules import (
    GATED_GUIDE_ROWS,
    GUIDE_LATENTS_PER_CHUNK,
    GUIDES_INPUT,
    HELD_ROWS,
    HOLD_AUDIO_INPUT,
    LATENT_ANCHOR_INPUT,
    MASK_INPUTS,
    REFINE_BEFORE,
    REFINE_STRENGTH_INPUT,
    REFINE_TILES_INPUT,
    REFINE_WINDOWS_ENV_VAR as ENABLE_ENV_VAR,
    REFINE_WINDOWS_INPUT,
    H3_FPS,
    RENDER_GRID,
    SOURCE_VIDEO_INPUT,
    guide_latent_frames,
)

logger = logging.getLogger("dw")

# The denoise loop's forward sub-block, which the windowed one replaces
DENOISER = "denoiser"
# The core sequence's layout block, whose duration ceiling a windowed refine lifts
LAYOUT = "prepare_layout"
# diffusers' ceiling, read off its own property rather than copied
H3_MAX_SECONDS = MiniMaxH3ModularPipeline.max_duration.fget(None)
BLENDS = ("pyramid", "cosine")
POSITIONS = ("true", "rebased")
_KEYS = ("frames", "overlap", "blend", "positions")
# The arguments that put rows into the sequence the windows do not hold - a
# keyframe's or a reference's conditioning rows, a guide's - or that act on
# the take after the loop. The loop refuses the rows at step 0; these are
# refused before the call, by name
CONFLICTS = (
    "image",
    "last_image",
    "keyframes",
    "references",
    GUIDES_INPUT,
    LATENT_ANCHOR_INPUT,
)
# H3's clip grid: a 17-frame clip encodes to 5 latents, the first of them one
# frame. A rebased window's first latent sits at the take's origin, the slot
# of a one-frame latent, so it must be one - a start on this grid
CLIP_LATENTS = GUIDE_LATENTS_PER_CHUNK


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


def rebase_problem(plan):
    """Why `positions: "rebased"` cannot move this plan's windows to the
    take's origin, or None: a window that starts off the clip grid starts on a
    four-frame latent, and the origin slot is a one-frame latent's."""
    off = [start for start, _ in plan if start % CLIP_LATENTS]
    if not off:
        return None
    return (
        f"{REFINE_WINDOWS_INPUT} positions 'rebased' moves each window's first "
        f"latent to the take's origin, which is a one-frame latent's slot, so "
        f"every window must start on a multiple of {CLIP_LATENTS} latents (H3's "
        f"clip grid) - windows start at {', '.join(map(str, off))}; pick frames "
        f"and overlap so frames - overlap is a multiple of {CLIP_LATENTS} and the "
        f"last window lands on it, or use positions 'true'"
    )


def _take_latent_frames(arguments):
    """The take's latent frame count when the call already says it - a 5-d
    `latents` tensor, or a `num_frames` on the 17m + 5 render grid - or None."""
    latents = arguments.get("latents")
    shape = getattr(latents, "shape", None)
    if shape is not None and len(shape) == 5:
        return int(shape[2])
    num_frames = arguments.get("num_frames")
    if (
        isinstance(num_frames, int)
        and not isinstance(num_frames, bool)
        and num_frames >= RENDER_GRID["remainder"]
        and (num_frames - RENDER_GRID["remainder"]) % RENDER_GRID["modulus"] == 0
    ):
        return guide_latent_frames(num_frames)
    return None


# 2. The rules - what the call needs, checked before it runs


def _long_source_problem(arguments):
    """Why a `source_video` is too long to refine through its own encode, or
    None. Past the ceiling the whole take would go through the VAE as one
    float32 tensor (about 18 GB at 60 s, 768p); `encode_h3_latents` encodes it
    a clip at a time into the `latents` the refine takes instead."""
    if arguments.get(SOURCE_VIDEO_INPUT) is None:
        return None
    num_frames = arguments.get("num_frames")
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        # Frames or an array have a length; a path (all validation sees) is
        # characters, not frames
        source = arguments[SOURCE_VIDEO_INPUT]
        num_frames = (
            len(source)
            if hasattr(source, "__len__") and not isinstance(source, (str, bytes, dict))
            else None
        )
    if num_frames is None or num_frames / H3_FPS <= H3_MAX_SECONDS:
        return None
    return (
        f"{REFINE_WINDOWS_INPUT}: a source_video of {num_frames} frames is "
        f"over {H3_MAX_SECONDS:g} s, and its encode holds the whole take as "
        f"float32 - encode it with an encode_h3_latents step and pass that "
        f"step's latents and num_frames instead"
    )


def window_problems(arguments, has_cache=False):
    """Why this step's `experimental_refine_windows` cannot run - a list,
    empty when it can. Pure: the settings, and the refine it rides on."""
    settings = arguments.get(REFINE_WINDOWS_INPUT)
    if settings is None:
        return []
    if not isinstance(settings, dict):
        return [
            f"{REFINE_WINDOWS_INPUT} is {{frames, overlap}} in latent frames, "
            f"got {settings!r}"
        ]
    return _settings_problems(settings, arguments) + _refine_problems(
        arguments, has_cache
    )


def _settings_problems(settings, arguments):
    """The window settings' own problems: their keys, the plan they make, the
    blend and the positions."""
    name = REFINE_WINDOWS_INPUT
    problems = []
    extra = sorted(set(settings) - set(_KEYS))
    if extra:
        problems.append(f"{name} takes {', '.join(_KEYS)}, not {', '.join(extra)}")
    frames, overlap = settings.get("frames"), settings.get("overlap", 0)
    planned = False
    try:
        window_plan(max(1, frames if isinstance(frames, int) else 1), frames, overlap)
        planned = True
    except (TypeError, ValueError) as error:
        problems.append(f"{name}: {error}")
    if settings.get("blend", "pyramid") not in BLENDS:
        problems.append(f"{name} blend is one of {', '.join(BLENDS)}")
    positions = settings.get("positions", "true")
    if positions not in POSITIONS:
        problems.append(f"{name} positions is one of {', '.join(POSITIONS)}")
    elif positions == "rebased" and planned:
        problem = _rebased_plan_problem(arguments, frames, overlap)
        if problem:
            problems.append(problem)
    return problems


def _rebased_plan_problem(arguments, frames, overlap):
    """Why rebased positions cannot hold for this plan, or None."""
    # The stride decides every start but the pulled-back last; that one
    # needs the take's length, when the call already says it. The loop
    # checks the plan it runs either way
    take = _take_latent_frames(arguments)
    if take is not None:
        plan = window_plan(take, frames, overlap)
    else:
        plan = [(0, frames), (frames - overlap, 2 * frames - overlap)]
    return rebase_problem(plan)


def _refine_problems(arguments, has_cache):
    """The problems of the refine the windows ride on: its strength, its
    take, its length, and the inputs and cache the windows cannot split."""
    name = REFINE_WINDOWS_INPUT
    problems = []
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
    long_source = _long_source_problem(arguments)
    if long_source:
        problems.append(long_source)
    for other in (REFINE_TILES_INPUT, *MASK_INPUTS):
        if arguments.get(other) is not None:
            problems.append(f"{name} does not take {other} yet")
    for other in CONFLICTS:
        if arguments.get(other) is not None:
            problems.append(
                f"{name} windows a t2va refine whose only audio is the held "
                f"track and whose only video is the take, and {other} adds rows "
                f"the windows do not split - drop {other}"
            )
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


def _layout_slots(pipeline):
    """[(holder, key)] for every t2va/fl2va layout step that runs: the
    sequence's `prepare_layout`, or, when the guides' layout wraps it
    (h3_guide_steps.DwH3GuideLayoutStep, which calls a stock layout it holds
    as `_layout`), that inner step. Ref2VA's layout is another class and is
    left out: it keeps its own ceiling, and the windows refuse references."""
    slots = []
    for prefix, sequence in core_denoise_sequences(pipeline):
        name = prefix + LAYOUT
        step = sequence.sub_blocks.get(name)
        if isinstance(step, MiniMaxH3PrepareLayoutStep):
            slots.append((sequence.sub_blocks, name))
        elif isinstance(getattr(step, "_layout", None), MiniMaxH3PrepareLayoutStep):
            slots.append((step, "_layout"))
    return slots


def _slot(holder, key):
    return holder[key] if isinstance(holder, dict) else getattr(holder, key)


def _set_slot(holder, key, value):
    if isinstance(holder, dict):
        # OrderedDict assignment keeps the key's place in the sequence
        holder[key] = value
    else:
        setattr(holder, key, value)


def layout_steps(pipeline):
    """The t2va/fl2va layout steps a pipeline runs, wrapped or not."""
    return [_slot(holder, key) for holder, key in _layout_slots(pipeline)]


def insert_refine_windows(pipeline):
    """Replace every H3 denoise loop's `denoiser` with the windowed one, in
    place and keeping its transformer, every t2va/fl2va `prepare_layout`
    with the one that lifts the duration ceiling for a windowed refine, and
    the video decode with the one that decodes a piece of the take at a time.
    Idempotent. Returns whether the pipeline now windows a refine."""
    for holder, key in _layout_slots(pipeline):
        if type(_slot(holder, key)) is MiniMaxH3PrepareLayoutStep:
            _set_slot(holder, key, DwH3LongRefineLayoutStep())
    # A take past 15 s decodes too: its pixels do not fit the device whole
    for holder, key in _video_decode_slots(getattr(pipeline, "_blocks", None)):
        if type(holder[key]) is MiniMaxH3VideoDecodeStep:
            holder[key] = DwH3StreamedVideoDecodeStep()
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


class _LongTakeComponents:
    """`components` as the stock layout step reads them, with `max_duration`
    raised to one call's take. A view, not a change: the pipeline itself, and
    every other call, keep diffusers' ceiling."""

    def __init__(self, components, max_duration):
        self._components = components
        self.max_duration = max_duration

    def __getattr__(self, name):
        return getattr(self._components, name)


class DwH3LongRefineLayoutStep(MiniMaxH3PrepareLayoutStep):
    model_name = "minimax-h3"

    @property
    def description(self):
        return (
            super().description
            + " dw (experimental): a refine with `experimental_refine_windows` lays out a take longer than "
            "15 s; every other call keeps the ceiling."
        )

    @property
    def inputs(self):
        return super().inputs + [
            InputParam(name=REFINE_WINDOWS_INPUT, type_hint=dict, default=None),
            InputParam(name=REFINE_STRENGTH_INPUT, default=None),
        ]

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        windowed = (
            getattr(block_state, REFINE_WINDOWS_INPUT, None) is not None
            and getattr(block_state, REFINE_STRENGTH_INPUT, None) is not None
        )
        if not windowed or block_state.num_frames is None:
            return super().__call__(components, state)
        aligned = align_num_frames(
            block_state.num_frames,
            components.vae_frames_per_chunk,
            components.vae_latents_per_chunk,
        )
        ceiling = max(components.max_duration, aligned / components.fps)
        if ceiling > components.max_duration:
            logger.info(
                f"Windowed refine: laying out {aligned} frames "
                f"({aligned / components.fps:.2f}s), past H3's "
                f"{components.max_duration:g}s ceiling"
            )
        super().__call__(_LongTakeComponents(components, ceiling), state)
        return components, state


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


# 4. The late capstone's gate (#1187, #1199 Task 14 Step 4): EXPERIMENTAL.
# A guide row marked `enter_below_sigma` is held out of the forward while the
# step's sigma is at or above its gate - the sequence is the rest, as a window
# is - and joins it once sigma falls below. H3's timesteps are 1 - sigma.


def gated_out_rows(gates, t, device=None):
    """The `latents` rows a step leaves out of its forward - every gated
    guide's rows while the step's sigma (1 - t) is at or above its gate - or
    None when it leaves none out."""
    sigma = 1.0 - float(t)
    rows = [
        torch.arange(start, start + count, device=device)
        for start, count, gate in gates or ()
        if sigma >= gate
    ]
    return torch.cat(rows) if rows else None


def without_rows(fields, num_video_rows, num_audio_rows, rows):
    """`window_rows`' `(video, audio, sequence)` for the whole packed
    sequence less the video rows `rows`."""
    device = fields["video_indices"].device
    keep = torch.ones(num_video_rows, dtype=torch.bool, device=device)
    keep[rows] = False
    sequence = torch.ones(
        fields["position_ids"].shape[0], dtype=torch.bool, device=device
    )
    sequence[fields["video_indices"][rows]] = False
    return (
        keep.nonzero().squeeze(1),
        torch.arange(num_audio_rows, device=device),
        sequence.nonzero().squeeze(1),
    )


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
            InputParam(name=HELD_ROWS, type_hint=int, default=0),
            InputParam(name=GATED_GUIDE_ROWS, type_hint=list, default=None),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        settings = getattr(block_state, REFINE_WINDOWS_INPUT, None)
        if settings is not None:
            return self._windowed(components, block_state, i, settings)
        rows = gated_out_rows(
            getattr(block_state, GATED_GUIDE_ROWS, None), t, block_state.latents.device
        )
        if rows is None:
            return super().__call__(components, block_state, i, t)
        return self._gated(components, block_state, i, rows)

    def _gated(self, components, block_state, i, rows):
        """The stock forward over every row but the gated-out `rows`, whose
        prediction is zero - condition rows are never stepped."""
        transformer = getattr(components, self.transformer_name)
        accepted = inspect.signature(transformer.forward).parameters
        latents, audio_latents = block_state.latents, block_state.audio_latents
        kept = without_rows(
            block_state.denoiser_input_fields,
            latents.shape[0],
            audio_latents.shape[0],
            rows,
        )
        layout, (timestep, timestep_indices) = window_inputs(
            block_state.denoiser_input_fields, block_state.row_timestep_plan[i], kept
        )
        video = kept[0]
        noise_pred, block_state.audio_noise_pred = transformer(
            hidden_states=latents[video][None],
            audio_hidden_states=audio_latents[None],
            encoder_hidden_states=block_state.prompt_embeds,
            timestep=timestep,
            timestep_indices=timestep_indices,
            attention_kwargs=block_state.attention_kwargs,
            return_dict=False,
            **{k: v for k, v in layout.items() if k in accepted},
        )
        full = noise_pred.new_zeros((1, latents.shape[0], noise_pred.shape[-1]))
        full[0, video] = noise_pred[0]
        block_state.noise_pred = full
        return components, block_state

    def _check_rows(self, block_state):
        audio_rows = block_state.audio_latents.shape[0]
        held = getattr(block_state, HELD_ROWS, 0) or 0
        if getattr(block_state, GATED_GUIDE_ROWS, None):
            # the windowed forward never reads the gates, so a capstone would
            # be silently ignored (the whole-sequence forward is what gates it)
            raise ValueError(
                f"{REFINE_WINDOWS_INPUT} does not hold a gated guide row back "
                f"(the experimental capstone): drop one of the two"
            )
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
            problem = rebase_problem(plan)
            if problem:
                raise ValueError(problem)
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


# 5. The long decode: a piece of the take at a time

# VAE clips per decode piece: 8 clips are 136 frames, about 0.85 GB of float32
# pixels at 960x544 and 1.7 GB at 1344x768
DECODE_PIECE_CLIPS = 8


def streamed_decode(vae, z, finish, piece_clips=DECODE_PIECE_CLIPS):
    """Decode H3 video latents `z` a piece at a time, handing each run of
    finished pixel frames to `finish` in order. The runs, concatenated in
    time, equal `vae.decode(z)` exactly.

    The stock decode keeps every decoded clip on the device and concatenates
    them, then the decode block converts the whole take to float32: about
    9 + 9 + 18 GB for 120 s at 960x544, past a 24 GB card. A take that long
    exists only through the windowed refine.

    Each piece is `vae.decode` of `piece_clips` whole clips' latents plus the
    `token_overlap` latents the last clip also reads, so `decode` decodes
    exactly those clips: the same latents the whole decode gives each clip,
    and through `decode`, so on-demand placement and offload hooks still run.
    `decode` leaves a piece's last clip's overlap frames raw at its end, and
    the whole decode cross-fades them into the next clip's first frames. That
    tail is held back and blended into the next piece with the VAE's own
    `_blend`, which is how `_decode` joins two clips. The last piece runs to
    the end of `z`, so `decode` pads and trims the end as the whole decode
    does: the padding depends only on the latent count modulo the clip, and
    every piece starts on a clip.

    A VAE with `token_drop` 0 has no overlap to join: it decodes whole.

    Raises:
        ValueError: If `piece_clips` is not a whole number of 1 or more
    """
    if isinstance(piece_clips, bool) or not isinstance(piece_clips, int):
        raise ValueError(f"piece_clips is a whole number, got {piece_clips!r}")
    if piece_clips < 1:
        raise ValueError(f"piece_clips is 1 or more, got {piece_clips}")
    clip = vae.tokens_chunk_size
    overlap = vae.token_overlap
    if overlap < 1:
        # No `token_drop`: clips do not overlap, so there is no tail to hold
        # back and the stock whole decode is already the right one
        finish(vae.decode(z, return_dict=False)[0])
        return
    # Frames each clip contributes before the next clip's cross-fade
    clip_frames = clip * vae.temporal_compression_ratio - vae.frame_pre_padding
    total = z.shape[2]
    piece = piece_clips * clip
    tail = None
    start = 0
    while True:
        # A piece is not the last while one more clip reads past it: then the
        # whole decode blends this piece's tail into that clip
        last = start + piece + overlap >= total
        stop = total if last else start + piece + overlap
        pixels = vae.decode(z[:, :, start:stop], return_dict=False)[0]
        if tail is not None:
            pixels = vae._blend(tail, pixels, vae.frame_overlap, dim=-3)
        if last:
            finish(pixels)
            return
        kept = piece_clips * clip_frames
        tail = pixels[:, :, kept:]
        finish(pixels[:, :, :kept])
        del pixels
        start += piece


class DwH3StreamedVideoDecodeStep(MiniMaxH3VideoDecodeStep):
    """H3's video decode, a piece of the take at a time (`streamed_decode`).

    The frames are the stock block's exactly. Each piece is denormalized,
    clamped and post-processed to the output type as it is decoded, so the
    device holds one piece's pixels rather than the whole take's. A `pt`
    output goes through the stock block."""

    piece_clips = DECODE_PIECE_CLIPS

    @property
    def description(self):
        return (
            "dw: decodes the video latents a piece of the take at a time, so a long take's pixels never sit on "
            "the device whole - the frames are the stock decode's. "
            + super().description
        )

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        output_type = block_state.output_type
        if output_type not in ("pil", "np"):
            return super().__call__(components, state)
        device = components._execution_device
        config = components.vae.config
        latents_mean = torch.tensor(config.latents_mean, device=device).view(
            1, -1, 1, 1, 1
        )
        latents_std = torch.tensor(config.latents_std, device=device).view(
            1, -1, 1, 1, 1
        )
        latents = block_state.latents * latents_std + latents_mean
        pixel_mean = torch.tensor(components.pixel_mean, device=device).view(
            1, -1, 1, 1, 1
        )
        pixel_std = torch.tensor(components.pixel_std, device=device).view(
            1, -1, 1, 1, 1
        )
        pieces = []

        def finish(video):
            video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
            pieces.append(
                components.video_processor.postprocess_video(
                    video, output_type=output_type
                )
            )

        streamed_decode(components.vae, latents, finish, self.piece_clips)
        if output_type == "np":
            import numpy

            block_state.videos = numpy.concatenate(pieces, axis=1)
        else:
            block_state.videos = [
                [frame for piece in pieces for frame in piece[batch]]
                for batch in range(latents.shape[0])
            ]
        self.set_block_state(state, block_state)
        return components, state


def _video_decode_slots(block):
    """[(holder, key)] for every MiniMaxH3VideoDecodeStep under `block`."""
    slots = []
    for key, child in (getattr(block, "sub_blocks", None) or {}).items():
        if isinstance(child, MiniMaxH3VideoDecodeStep):
            slots.append((block.sub_blocks, key))
        else:
            slots.extend(_video_decode_slots(child))
    return slots


def video_decode_steps(pipeline):
    """The video decode blocks a pipeline runs."""
    return [
        holder[key]
        for holder, key in _video_decode_slots(getattr(pipeline, "_blocks", None))
    ]
