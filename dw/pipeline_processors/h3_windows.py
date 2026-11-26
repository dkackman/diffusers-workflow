"""MiniMax-H3 time-windowed refine (#1199 Task 17 Step 4, #1188, #1230).

A seam pass over a finished take: it smooths the jumps at a chain's seams, and
does not undo a burn (bench/records/structural.md, *Step 4*). `refine_windows`
on a `refine_strength` call (`latents`, or an unmasked `source_video` - a finished
take as pixels, encoded by the regional edit's block - plus `hold_audio`)
re-denoises the take in
overlapping windows of latent frames instead of one full-sequence forward,
and fuses the windows' predictions into the full-sequence update at every
step. It is the refine tiled in time instead of space (`refine_tiles`,
dw/pipeline_processors/h3_tiles.py), with one difference that decides where it
lives: the tiles are whole pipeline calls blended once at the end, and the
windows are fused at *every* step, so the fusion has to be inside the denoise
loop.

On every H3 load (#1230): the loader (components.py) swaps the blocks in
unconditionally, and each runs the stock path unless a call carries
`refine_windows`. Every key of the argument is optional (`WINDOW_DEFAULTS`, the
benched 120 s setting), so `refine_windows: {}` is the default windows. The
call rules (`window_problems`) are checked at validate time too
(dw/guides.py `windows_errors`) and again before the call (`windows_refusal`).
The long-form spikes' flag, `DW_EXPERIMENTAL_REFINE_WINDOWS`, still gates the
capstone's gate rows this denoiser holds out (`_gated`); the windows no longer
need it.

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
call carries both `refine_windows` and `refine_strength`, or
`experimental_hybrid_windows`; then it hands the stock step a view of
`components` whose `max_duration` is the take's own (`_LongTakeComponents`).
Nothing is patched: every other call, and every other job in the worker,
keeps the ceiling. Past 15 s is a regime only these two reach: a refine
whose every forward sees one window, and the experimental generation below,
whose every forward does too. The upstream change that would replace it is a
`ConfigSpec("max_duration", 15.0)` on the layout step, read as
`components.config.max_duration` like `canvas_max_pixels`.

Windowed generation (#1188, EXPERIMENTAL). `experimental_hybrid_windows`
- `{frames, overlap, blend, positions, switch_step, handoff, anchor}`, the first four
as `refine_windows`'s - generates a t2va take from noise as the same fused
windows, on every step, with a held `hold_audio` track as its only audio. It
is refused unless the process sets `DW_EXPERIMENTAL_HYBRID_WINDOWS=1`, naming
the flag (`hybrid_problems`), and is in no schema or tool text. The take's
full state lives in host RAM (pinned on CUDA): at step 0 the denoiser moves
`latents` and `audio_latents` off the device, each window's rows go to the
device for its forward only, its velocity comes back and is fused on the
host, and `DwH3HostLoopSchedulerStep` - the stock update, which moves its
sigmas to the sample's device - steps the whole take there, then puts it back
on the device after the last step for the release and the streamed decode.

It runs in two phases, split at `switch_step` k (default 3; at most the run's
S = num_inference_steps - 1 denoise steps, refused at validate past that).
The ordered phase (`_ordered`, all of it at step 0, before the first fused
forward): window 1 runs every one of the S steps alone and is kept finished;
each later window runs steps 1..k alone with its overlap pinned to the window
before it - that window's finished frames (`handoff: "finished"`, the
default; the window before runs to the last step for it) or its x0 at step k
(`"x0"`, x + sigma * v) - the pinned rows read at the clean timestep and never
stepped. Every row keeps the first window's value (keep-first): a window
writes back only the rows no earlier window covered, at their step-k state.
The blended phase is steps k+1..S: the fused windows above, with window 1's
rows held finished (`dw_hybrid_held`) - read at the clean timestep, left out
of the fusion and restored after each update - so a window wholly inside it
is not run. Steps 1..k of the outer loop only advance the schedulers. k = 0
is no ordered phase, every step blended; k = S is plain sequential windows,
each finished in order with its overlap pinned to its predecessor's frames.

The keystone anchor (`anchor`, S4): `"none"` (the default) touches none of
the above; `"keystone"` makes window 1's finished first latent frame a
keystone every later forward reads. With an ordered phase it is window 1's
own first frame; with k = 0, `_keystone_prepass` runs window 1 alone to the
last step first, writing nothing back. Every forward but window 1's own then
carries the keystone's rows ahead of the window's video (`keystone_inputs`):
the take's frame-0 positions, token tag 0, held at the clean timestep (H3's
t = 1), and dropped from the prediction before the fuse, so the take never
holds them. On the ordered phase's later windows the keystone attention
(`KEYSTONE_ATTENTION`, h3_keystone_attention's K/C re-split, K the keystone
rows and C the window's pinned overlap) runs: the owner installs it per
forward for the ordered phase (`install(per_forward=True)`), each later
window publishes its K and C to it (`publish_window_rows`, checked by the
owner's `check_window_rows`) and clears them after, and window 1 and the
blended steps keep the plain attention. A block in it the transformer lacks
raises, as the owner's parse does. On a transformer the owner refuses
(`transformer_refusal`: sparse attention, or another re-split installed) the
attention is skipped with a warning and the rows ship alone.
Any other `anchor` is refused at validate, before the flag. Until promotion
no refusal's key list names `anchor` (`_HYBRID_LISTED_KEYS`): only its own
value refusal does, once a caller has written the field.

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
)
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep
from diffusers.modular_pipelines.minimax_h3.denoise import (
    MiniMaxH3LoopDenoiser,
    MiniMaxH3LoopSchedulerStep,
)
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam

from .. import get_device_type, references
from ..events import emit_warning
from . import h3_keystone_attention as keystone_attention
from .h3_guide_steps import splice_guide_rows
from .h3_hold_steps import (  # noqa: F401 - re-exported: the decode's and the window's names live here too
    DECODE_PIECE_CLIPS,
    DwH3LongRefineLayoutStep,
    DwH3StreamedVideoDecodeStep,
    _video_decode_slots,
    clean_timesteps,
    gated_out_rows,
    streamed_decode,
    video_decode_steps,
    window_inputs,
    window_rows,
    without_rows,
)
from .h3_hold import core_denoise_sequences
from .h3_rules import (
    GATED_GUIDE_ROWS,
    GUIDE_LATENTS_PER_CHUNK,
    GUIDES_INPUT,
    HELD_ROWS,
    HOLD_AUDIO_INPUT,
    HYBRID_WINDOWS_ENV_VAR,
    HYBRID_WINDOWS_INPUT,
    LATENT_ANCHOR_INPUT,
    MASK_INPUTS,
    REFINE_BEFORE,
    REFINE_STRENGTH_INPUT,
    REFINE_TILES_INPUT,
    REFINE_WINDOWS_INPUT,
    H3_FPS,
    RENDER_GRID,
    SOURCE_VIDEO_INPUT,
    guide_latent_frames,
    hybrid_windows_enabled,
)

logger = logging.getLogger("dw")

# The denoise loop's forward sub-block, which the windowed one replaces
DENOISER = "denoiser"
# Its update sub-block, which the host-RAM one replaces (#1188)
UPDATE = "update"
# The core sequence's layout block, whose duration ceiling a windowed refine lifts
LAYOUT = "prepare_layout"
# diffusers' ceiling, read off its own property rather than copied
H3_MAX_SECONDS = MiniMaxH3ModularPipeline.max_duration.fget(None)
BLENDS = ("pyramid", "cosine")
POSITIONS = ("true", "rebased")
_KEYS = ("frames", "overlap", "blend", "positions")
# What an absent key means (#1230 Q2): the setting benched at 120 s, which won
# seam jump on every chain shot type (bench/records/structural.md, *Step 4*).
# 42 - 12 = 30 latents is on the rebased clip grid
WINDOW_DEFAULTS = {
    "frames": 42,
    "overlap": 12,
    "blend": "pyramid",
    "positions": "rebased",
}
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
# The windowed generation's own keys (#1188), past the window's four: the step
# the ordered windows hand over to blended ones at, what they hand over, and
# the anchor every window's forward reads (#1188 S4)
_HYBRID_KEYS = _KEYS + ("switch_step", "handoff", "anchor")
# The keys a refusal lists. `anchor` is accepted but unpromoted (#1188 plan
# v3: no MCP text documents it), so a key list never names it; only its own
# value refusal does, once a caller has written it (#1572 bounce, M-F495)
_HYBRID_LISTED_KEYS = tuple(key for key in _HYBRID_KEYS if key != "anchor")
HYBRID_DEFAULTS = {
    **WINDOW_DEFAULTS,
    "switch_step": 3,
    "handoff": "finished",
    "anchor": "none",
}
HANDOFFS = ("finished", "x0")
ANCHORS = ("none", "keystone")
# What the generation refuses beyond CONFLICTS: a refine's inputs (the take is
# noise), and the refine windows, which are the other branch of the denoiser
HYBRID_REFINE_INPUTS = (
    REFINE_WINDOWS_INPUT,
    REFINE_STRENGTH_INPUT,
    REFINE_TILES_INPUT,
    "latents",
    SOURCE_VIDEO_INPUT,
    *MASK_INPUTS,
)
# The loop's note of the device the take came from, set while it is on the host
HYBRID_DEVICE = "dw_hybrid_device"
# The ordered phase's notes for the loop (#1188 S2): the step the blended
# finish starts at, and the rows the first window left finished, which the
# blended finish holds where they are
HYBRID_SWITCH = "dw_hybrid_switch"
HYBRID_HELD = "dw_hybrid_held"
# The keystone (#1188 S4, `anchor: "keystone"`): window 1's finished first
# latent frame, a host tensor of its rows, set at step 0 while the take is on
# the host
HYBRID_KEYSTONE = "dw_hybrid_keystone"
# The keystone attention on the ordered phase's later windows (geiru 1.1): the
# re-split's balance and blocks, the long-form spike's defaults. Blocks the
# transformer does not have are left out
KEYSTONE_ATTENTION = keystone_attention.KeystoneAttentionSettings()
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


def rebase_problem(plan, name=REFINE_WINDOWS_INPUT):
    """Why `positions: "rebased"` cannot move this plan's windows to the
    take's origin, or None: a window that starts off the clip grid starts on a
    four-frame latent, and the origin slot is a one-frame latent's."""
    off = [start for start, _ in plan if start % CLIP_LATENTS]
    if not off:
        return None
    return (
        f"{name} positions 'rebased' moves each window's first "
        f"latent to the take's origin, which is a one-frame latent's slot, so "
        f"every window must start on a multiple of {CLIP_LATENTS} latents (H3's "
        f"clip grid) - windows start at {', '.join(map(str, off))}; pick frames "
        f"and overlap so frames - overlap is a multiple of {CLIP_LATENTS} and the "
        f"last window lands on it, or use positions 'true'"
    )


def window_settings(settings):
    """A `refine_windows` value with every absent key at its default
    (`WINDOW_DEFAULTS`). The settings are not checked here."""
    return {**WINDOW_DEFAULTS, **settings}


def _unresolved(value):
    return isinstance(value, str) and references.is_ref(references.UNRESOLVED, value)


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
    """Why this step's `refine_windows` cannot run - a list, empty when it
    can or cannot be told yet. Pure: the settings, and the refine it rides on.
    A setting that is still an unresolved reference (validate time) is left
    to the run, which checks it again."""
    settings = arguments.get(REFINE_WINDOWS_INPUT)
    if settings is None or _unresolved(settings):
        return []
    if not isinstance(settings, dict):
        return [
            f"{REFINE_WINDOWS_INPUT} is {{frames, overlap, blend, positions}}, "
            f"every key optional, got {settings!r}"
        ]
    return _settings_problems(settings, arguments) + _refine_problems(
        arguments, has_cache
    )


def _settings_problems(
    settings, arguments, name=REFINE_WINDOWS_INPUT, keys=_KEYS, listed=None
):
    """The window settings' own problems: their keys, the plan they make, the
    blend and the positions. `listed` is the keys an unknown key's refusal
    names - `keys` unless some are accepted but left unlisted."""
    problems = []
    extra = sorted(set(settings) - set(keys))
    if extra:
        named = ", ".join(listed if listed is not None else keys)
        problems.append(f"{name} takes {named}, not {', '.join(extra)}")
    resolved = window_settings(settings)
    frames, overlap = resolved["frames"], resolved["overlap"]
    planned = False
    if not (_unresolved(frames) or _unresolved(overlap)):
        try:
            window_plan(
                max(1, frames if isinstance(frames, int) else 1), frames, overlap
            )
            planned = True
        except (TypeError, ValueError) as error:
            problems.append(f"{name}: {error}")
    blend = resolved["blend"]
    if not _unresolved(blend) and blend not in BLENDS:
        problems.append(f"{name} blend is one of {', '.join(BLENDS)}")
    positions = resolved["positions"]
    if _unresolved(positions):
        return problems
    if positions not in POSITIONS:
        problems.append(f"{name} positions is one of {', '.join(POSITIONS)}")
    elif positions == "rebased" and planned:
        problem = _rebased_plan_problem(arguments, frames, overlap, name)
        if problem:
            problems.append(problem)
    return problems


def _rebased_plan_problem(arguments, frames, overlap, name=REFINE_WINDOWS_INPUT):
    """Why rebased positions cannot hold for this plan, or None."""
    # The stride decides every start but the pulled-back last; that one
    # needs the take's length, when the call already says it. The loop
    # checks the plan it runs either way
    take = _take_latent_frames(arguments)
    if take is not None:
        plan = window_plan(take, frames, overlap)
    else:
        plan = [(0, frames), (frames - overlap, 2 * frames - overlap)]
    return rebase_problem(plan, name)


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


def hybrid_settings(settings):
    """An `experimental_hybrid_windows` value with every absent key at its
    default (`HYBRID_DEFAULTS`). The settings are not checked here."""
    return {**HYBRID_DEFAULTS, **settings}


def _hybrid_key_problems(settings, arguments=None):
    """`switch_step`, `handoff` and `anchor`'s rules. `switch_step` is at most the
    run's denoise steps - H3 runs `num_inference_steps - 1` - when the call
    names a whole `num_inference_steps`; the run checks it again against the
    steps it has."""
    name = HYBRID_WINDOWS_INPUT
    resolved = hybrid_settings(settings)
    problems = []
    switch = resolved["switch_step"]
    if not _unresolved(switch) and (
        isinstance(switch, bool) or not isinstance(switch, int) or switch < 0
    ):
        problems.append(
            f"{name} switch_step is a whole number of steps, 0 or more, got {switch!r}"
        )
    else:
        steps = (arguments or {}).get("num_inference_steps")
        if (
            not _unresolved(switch)
            and isinstance(steps, int)
            and not isinstance(steps, bool)
            and switch > steps - 1
        ):
            problems.append(
                f"{name} switch_step is at most the run's denoise steps - "
                f"num_inference_steps {steps} runs {max(steps - 1, 0)} - got {switch}"
            )
    handoff = resolved["handoff"]
    if not _unresolved(handoff) and handoff not in HANDOFFS:
        problems.append(f"{name} handoff is one of {', '.join(HANDOFFS)}")
    anchor = resolved["anchor"]
    if not _unresolved(anchor) and anchor not in ANCHORS:
        problems.append(f"{name} anchor is one of {', '.join(ANCHORS)}, got {anchor!r}")
    return problems


def hybrid_problems(arguments, has_cache=False):
    """Why this step's `experimental_hybrid_windows` cannot run - a list,
    empty when it can or cannot be told yet. Pure. The settings' own rules
    come first, so a caller sees the key's rules before the flag's; then the
    flag, then the call: a held track, no rows the windows do not split, no
    refine and no step cache. A setting still an unresolved reference is
    left to the run, which checks it again."""
    name = HYBRID_WINDOWS_INPUT
    settings = arguments.get(name)
    if settings is None or _unresolved(settings):
        return []
    if not isinstance(settings, dict):
        return [
            f"{name} is {{{', '.join(_HYBRID_LISTED_KEYS)}}}, every key optional, "
            f"got {settings!r}"
        ]
    problems = _settings_problems(
        settings, arguments, name, _HYBRID_KEYS, _HYBRID_LISTED_KEYS
    ) + _hybrid_key_problems(settings, arguments)
    if not hybrid_windows_enabled():
        problems.append(
            f"{name} is experimental: it runs only on a server started with "
            f"{HYBRID_WINDOWS_ENV_VAR}=1"
        )
    if arguments.get(HOLD_AUDIO_INPUT) is None:
        problems.append(
            f"{name} generates the video of a take whose audio is held - add "
            f"'hold_audio' with the track"
        )
    for other in CONFLICTS:
        if arguments.get(other) is not None:
            problems.append(
                f"{name} generates a t2va take whose only audio is the held "
                f"track, and {other} adds rows the windows do not split - "
                f"drop {other}"
            )
    for other in HYBRID_REFINE_INPUTS:
        if arguments.get(other) is not None:
            problems.append(
                f"{name} generates a take from noise and does not take {other}"
            )
    if has_cache:
        problems.append(
            f"{name} runs the transformer once per window per step, and a step "
            f"cache would reuse one window's result for another - drop 'cache'"
        )
    return problems


def hybrid_refusal(pipeline, arguments, has_cache=False):
    """The first reason this call cannot generate in windows, or None."""
    if arguments.get(HYBRID_WINDOWS_INPUT) is None:
        return None
    if not windows_refine(pipeline):
        return (
            f"{HYBRID_WINDOWS_INPUT} is a MiniMax-H3 t2va argument, and "
            f"{type(pipeline).__name__} has no denoise loop that windows"
        )
    problems = hybrid_problems(arguments, has_cache)
    return problems[0] if problems else None


def windows_refusal(pipeline, arguments, has_cache=False):
    """The first reason this call cannot window its refine, or None. The
    loader puts the windowed denoiser in every H3 load, so a pipeline without
    it is one that has no core denoise loop to window."""
    if arguments.get(REFINE_WINDOWS_INPUT) is None:
        return None
    if not windows_refine(pipeline):
        return (
            f"{REFINE_WINDOWS_INPUT} is a MiniMax-H3 t2va/fl2va refine argument, "
            f"and {type(pipeline).__name__} has no denoise loop that windows"
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
    Idempotent. Returns whether the pipeline now windows a refine. The
    loop's `update` becomes `DwH3HostLoopSchedulerStep`, the stock step for
    every call but a windowed generation, whose take it brings back to the
    device after the last step."""
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
        if type(loop.sub_blocks.get(UPDATE)) is MiniMaxH3LoopSchedulerStep:
            loop.sub_blocks[UPDATE] = DwH3HostLoopSchedulerStep()
        inserted = True
    return inserted


def windows_refine(pipeline):
    """Whether a pipeline's denoise loops carry the windowed denoiser."""
    loops = list(_loops(pipeline))
    return bool(loops) and all(
        isinstance(loop.sub_blocks[DENOISER], DwH3WindowedLoopDenoiser)
        for loop in loops
    )


class DwH3WindowedLoopDenoiser(MiniMaxH3LoopDenoiser):
    model_name = "minimax-h3"

    @property
    def description(self):
        return (
            "dw: with `refine_windows`, runs the forward on overlapping windows of "
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
                    "{frames 42, overlap 12} in latent frames (blend pyramid|cosine, positions "
                    "rebased|true), every key optional - refine in fused time windows. Needs "
                    "refine_strength and hold_audio."
                ),
            ),
            InputParam(
                name=HYBRID_WINDOWS_INPUT,
                type_hint=dict,
                default=None,
                description=(
                    "EXPERIMENTAL, behind DW_EXPERIMENTAL_HYBRID_WINDOWS: generate in fused "
                    "time windows with the take's state in host RAM. Needs hold_audio."
                ),
            ),
            InputParam(name="num_latent_frames", type_hint=int, default=None),
            InputParam(name="num_condition_video_rows", type_hint=int, default=0),
            InputParam(name=HELD_ROWS, type_hint=int, default=0),
            InputParam(name=GATED_GUIDE_ROWS, type_hint=list, default=None),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        hybrid = getattr(block_state, HYBRID_WINDOWS_INPUT, None)
        if hybrid is not None:
            return self._hybrid(components, block_state, i, hybrid)
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

    def _check_rows(self, block_state, name=REFINE_WINDOWS_INPUT):
        audio_rows = block_state.audio_latents.shape[0]
        held = getattr(block_state, HELD_ROWS, 0) or 0
        if getattr(block_state, GATED_GUIDE_ROWS, None):
            # the windowed forward never reads the gates, so a capstone would
            # be silently ignored (the whole-sequence forward is what gates it)
            raise ValueError(
                f"{name} does not hold a gated guide row back "
                f"(the experimental capstone): drop one of the two"
            )
        if block_state.num_condition_video_rows or held != audio_rows:
            raise ValueError(
                f"{name} windows a t2va take whose every audio row is "
                f"the held track: {block_state.num_condition_video_rows or 0} conditioning "
                f"video rows and {held} of {audio_rows} audio rows held"
            )

    def _hybrid(self, components, block_state, i, settings):
        """The windowed generation (#1188): the take's state in host RAM,
        one window's rows on the device per forward. With `switch_step` k > 0
        step 0 runs the ordered phase (`_ordered`) - every window's steps
        1..k, the first window's to the end - and steps 1..k-1 have nothing
        left to run. Steps k+1 onward are the blended finish: the fused
        velocity on the host for `DwH3HostLoopSchedulerStep` to step, with the
        first window's finished rows held."""
        name = HYBRID_WINDOWS_INPUT
        if not hybrid_windows_enabled():
            # validate and the run's check refuse it first; this is the floor
            raise ValueError(
                f"{name} is experimental: it runs only on a server started "
                f"with {HYBRID_WINDOWS_ENV_VAR}=1"
            )
        settings = hybrid_settings(settings)
        switch = settings["switch_step"]
        if i == 0:
            steps = len(block_state.timesteps)
            if switch > steps:
                raise ValueError(
                    f"{name} switch_step is at most the run's {steps} denoise "
                    f"steps, got {switch}"
                )
            self._to_host(block_state)
            setattr(block_state, HYBRID_SWITCH, switch)
            setattr(block_state, HYBRID_HELD, None)
            setattr(block_state, HYBRID_KEYSTONE, None)
            if switch:
                self._ordered(components, block_state, settings, switch)
            elif settings["anchor"] == "keystone":
                setattr(
                    block_state,
                    HYBRID_KEYSTONE,
                    self._keystone_prepass(components, block_state, settings),
                )
        if i < switch:
            # the ordered phase has run this step for every window
            block_state.noise_pred = None
            block_state.audio_noise_pred = None
            return components, block_state
        return self._windowed(
            components,
            block_state,
            i,
            settings,
            HYBRID_WINDOWS_INPUT,
            host=True,
            held=getattr(block_state, HYBRID_HELD, None),
            keystone=getattr(block_state, HYBRID_KEYSTONE, None),
        )

    @staticmethod
    def _hybrid_rebase_from(fields, plan, positions):
        """The take's first generated video time when the windows are
        rebased, else None; a rebased plan off the clip grid raises."""
        if positions != "rebased":
            return None
        problem = rebase_problem(plan, HYBRID_WINDOWS_INPUT)
        if problem:
            raise ValueError(problem)
        return fields["position_ids"][fields["video_indices"][0], 0]

    def _keystone_prepass(self, components, block_state, settings):
        """The keystone with no ordered phase (`switch_step` 0, design 7):
        window 1 run alone to the last step on the ordered phase's
        single-window path, writing nothing back. Returns its finished first
        latent frame, a host tensor of that frame's rows."""
        self._check_rows(block_state, HYBRID_WINDOWS_INPUT)
        fields = block_state.denoiser_input_fields
        num_latent_frames = block_state.num_latent_frames
        window = window_settings(settings)
        plan = window_plan(num_latent_frames, window["frames"], window["overlap"])
        rebase_from = self._hybrid_rebase_from(fields, plan, window["positions"])
        latents, audio_latents = block_state.latents, block_state.audio_latents
        timesteps = block_state.timesteps
        scheduler = components.scheduler
        num_rows = latents.shape[0]
        rows = window_rows(fields, num_rows, 0, num_latent_frames, plan[0])
        video = rows[0].cpu()
        x = latents[video].clone()
        none = torch.zeros(video.numel(), dtype=torch.bool)
        free = ~none
        logger.info(
            f"Running window {plan[0]} alone for the keystone, {len(timesteps)} steps"
        )
        restore = scheduler._step_index
        try:
            for j in range(len(timesteps)):
                velocity = self._window_velocity(
                    components,
                    block_state,
                    j,
                    rows,
                    rebase_from,
                    x,
                    audio_latents[rows[1].cpu()],
                    none,
                )
                scheduler._step_index = j
                x[free] = scheduler.step(
                    velocity[free], timesteps[j], x[free], return_dict=False
                )[0]
        finally:
            scheduler._step_index = restore
        return x[: num_rows // num_latent_frames].clone()

    def _ordered_attention(self, components, settings, plan):
        """The keystone attention for the ordered phase's later windows
        (design 9): the owner's re-split (h3_keystone_attention.install,
        `per_forward`) on `KEYSTONE_ATTENTION`'s blocks, which runs the
        original attention on every forward that publishes no rows. Returns
        the transformer it went on, or None when it does not run - no
        keystone, one window, or a transformer the owner refuses (sparse
        attention, another re-split), which a warning names; the keystone
        rows still go in. A block the transformer lacks raises, as the owner's
        parse does."""
        if settings["anchor"] != "keystone" or len(plan) < 2:
            return None
        transformer = getattr(components, self.transformer_name)
        refusal = keystone_attention.transformer_refusal(transformer)
        if refusal is not None:
            emit_warning(
                f"{HYBRID_WINDOWS_INPUT} anchor keystone: the windows read the "
                f"keystone rows without the keystone attention - {refusal}"
            )
            return None
        try:
            keystone_attention.install(
                transformer, KEYSTONE_ATTENTION, per_forward=True
            )
        except ValueError as error:
            keystone_attention.uninstall(transformer)
            raise ValueError(
                f"{HYBRID_WINDOWS_INPUT} anchor keystone: {error}"
            ) from error
        return transformer

    def _ordered(self, components, block_state, settings, switch):
        """The ordered phase, in host RAM. Window 1 runs every step and is
        kept finished. Each later window runs steps 1..`switch` from its own
        slice of the start noise, with the rows an earlier window covers
        pinned to that window's handoff - its finished rows (`finished`), or
        its x0 estimate at step `switch` (`x0`) - and read at the clean
        timestep; only its other rows are stepped, by the stock scheduler
        step. Every row keeps the first window's state that covers it:
        window 1's finished, the later windows' at step `switch`. A window
        whose handoff a later one needs (`finished`, not the last) runs on to
        the end for it."""
        name = HYBRID_WINDOWS_INPUT
        self._check_rows(block_state, name)
        fields = block_state.denoiser_input_fields
        num_latent_frames = block_state.num_latent_frames
        window = window_settings(settings)
        plan = window_plan(num_latent_frames, window["frames"], window["overlap"])
        steps = len(block_state.timesteps)
        scheduler = components.scheduler
        rebase_from = self._hybrid_rebase_from(fields, plan, window["positions"])
        logger.info(
            f"Generating in {len(plan)} ordered time windows to step {switch} of "
            f"{steps} ({settings['handoff']} handoff), blended after: {plan}"
        )
        restore = scheduler._step_index
        attention = self._ordered_attention(components, settings, plan)
        try:
            self._ordered_windows(
                components,
                block_state,
                settings,
                switch,
                plan,
                rebase_from,
                attention,
            )
        finally:
            scheduler._step_index = restore
            if attention is not None:
                keystone_attention.uninstall(attention)

    def _ordered_windows(
        self, components, block_state, settings, switch, plan, rebase_from, attention
    ):
        """`_ordered`'s loop over the windows, with the keystone - window 1's
        finished first frame, under `anchor: "keystone"` - on every later
        window's forward, and the keystone attention on them when it runs."""
        fields = block_state.denoiser_input_fields
        num_latent_frames = block_state.num_latent_frames
        latents, audio_latents = block_state.latents, block_state.audio_latents
        timesteps = block_state.timesteps
        steps = len(timesteps)
        scheduler = components.scheduler
        handoff = settings["handoff"]
        num_rows = latents.shape[0]
        covered = torch.zeros(num_rows, dtype=torch.bool)
        pins = torch.empty_like(latents)
        keystone = None
        for w, span in enumerate(plan):
            rows = window_rows(fields, num_rows, 0, num_latent_frames, span)
            video = rows[0].cpu()
            pinned = covered[video]
            free = ~pinned
            x = latents[video].clone()
            x[pinned] = pins[video][pinned]
            last = w == len(plan) - 1
            run_to = steps if w == 0 or (handoff == "finished" and not last) else switch
            pin, kept = None, None
            for j in range(run_to):
                velocity = self._window_velocity(
                    components,
                    block_state,
                    j,
                    rows,
                    rebase_from,
                    x,
                    audio_latents[rows[1].cpu()],
                    pinned,
                    keystone,
                    attention is not None,
                )
                if j == switch - 1 and handoff == "x0":
                    sigma = float(scheduler.sigmas[j])
                    pin = (x.float() + sigma * velocity).to(x.dtype)
                    pin[pinned] = x[pinned]
                if free.any():
                    scheduler._step_index = j
                    x[free] = scheduler.step(
                        velocity[free], timesteps[j], x[free], return_dict=False
                    )[0]
                if j == switch - 1:
                    kept = x.clone()
            if w == 0:
                kept = x
            if handoff == "finished":
                pin = x
            latents[video[free]] = kept[free]
            if pin is not None and not last:
                pins[video] = pin
            covered[video] = True
            if w == 0:
                setattr(block_state, HYBRID_HELD, video)
                if settings["anchor"] == "keystone":
                    keystone = x[: num_rows // num_latent_frames].clone()
                    setattr(block_state, HYBRID_KEYSTONE, keystone)

    def _window_velocity(
        self,
        components,
        block_state,
        i,
        rows,
        rebase_from,
        hidden,
        audio_hidden,
        clean,
        keystone=None,
        attention=False,
    ):
        """One window's forward at step `i` from host `hidden` rows - the
        rows `clean` (a mask over them) at the clean timestep - its video
        velocity as float32 on the host. A `keystone` goes in ahead of the
        window's rows (`keystone_inputs`) and its velocity is dropped; under
        `attention` (the ordered phase's per-forward re-split is installed)
        the keystone and the clean rows are published to it as this forward's
        K and C (h3_keystone_attention.publish_window_rows) when `clean` is a
        leading run of the window, and cleared after."""
        transformer = getattr(components, self.transformer_name)
        accepted = inspect.signature(transformer.forward).parameters
        fields = block_state.denoiser_input_fields
        device = getattr(block_state, HYBRID_DEVICE)
        layout, entry = window_inputs(
            fields, block_state.row_timestep_plan[i], rows, rebase_from
        )
        if clean.any():
            local = layout["video_indices"][clean.to(layout["video_indices"].device)]
            entry = clean_timesteps(entry, local)
        hidden = hidden.to(device)
        count = 0
        if keystone is not None:
            layout, entry = keystone_inputs(fields, layout, entry, keystone.shape[0])
            count = keystone.shape[0]
            hidden = torch.cat([keystone.to(device), hidden])
        pinned = int(clean.sum())
        if attention and count and pinned and bool(clean[:pinned].all()):
            keystone_attention.publish_window_rows(
                transformer,
                (int(layout["video_indices"][0]), count),
                (int(layout["video_indices"][count]), pinned),
                frame_rows=count,
            )
        try:
            noise_pred, _ = transformer(
                hidden_states=hidden[None],
                audio_hidden_states=audio_hidden.to(device)[None],
                encoder_hidden_states=block_state.prompt_embeds,
                timestep=entry[0],
                timestep_indices=entry[1],
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **{k: v for k, v in layout.items() if k in accepted},
            )
        finally:
            if attention:
                keystone_attention.clear_rows(transformer)
        return noise_pred[0, count:].float().cpu()

    @staticmethod
    def _to_host(block_state):
        """Move the take's latents and audio rows to host RAM, pinned on
        CUDA, and note the device they came from."""
        device = block_state.latents.device
        setattr(block_state, HYBRID_DEVICE, device)
        pin = get_device_type(str(device)) == "cuda"
        for key in ("latents", "audio_latents"):
            host = getattr(block_state, key).cpu()
            setattr(block_state, key, host.pin_memory() if pin else host)

    def _windowed(
        self,
        components,
        block_state,
        i,
        settings,
        name=REFINE_WINDOWS_INPUT,
        host=False,
        held=None,
        keystone=None,
    ):
        """The fused windowed forward. With `host`, `latents` and
        `audio_latents` are host tensors: each window's rows go to the
        device for its forward, and the fused velocity stays on the host.
        `held`, the host rows the ordered phase left finished, are read at the
        clean timestep, get zero velocity, and a window of only held rows is
        not run. A host `keystone` goes in ahead of every window's rows that
        runs (`keystone_inputs`), and its velocity is dropped before the
        fuse."""
        self._check_rows(block_state, name)
        transformer = getattr(components, self.transformer_name)
        accepted = inspect.signature(transformer.forward).parameters
        fields = block_state.denoiser_input_fields
        num_latent_frames = block_state.num_latent_frames
        settings = window_settings(settings)
        plan = window_plan(num_latent_frames, settings["frames"], settings["overlap"])
        weights = fuse_weights(plan, num_latent_frames, settings["blend"])
        latents, audio_latents = block_state.latents, block_state.audio_latents
        device = getattr(block_state, HYBRID_DEVICE, None) if host else None
        rebase_from = None
        if settings["positions"] == "rebased":
            problem = rebase_problem(plan, name)
            if problem:
                raise ValueError(problem)
            first = fields["video_indices"][0]
            rebase_from = fields["position_ids"][first, 0]
        if i == 0:
            verb = "Generating" if host else "Refining"
            logger.info(f"{verb} in {len(plan)} time windows: {plan}")

        fused, total = None, None
        is_held = None
        if held is not None:
            is_held = torch.zeros(latents.shape[0], dtype=torch.bool)
            is_held[held] = True
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
            if is_held is not None:
                local_held = is_held[video.cpu()]
                if local_held.all():
                    continue
                if local_held.any():
                    positions = layout["video_indices"]
                    timestep, timestep_indices = clean_timesteps(
                        (timestep, timestep_indices),
                        positions[local_held.to(positions.device)],
                    )
            if host:
                # the take is on the host: index it there, ship the window
                video, audio = video.cpu(), audio.cpu()
                hidden = latents[video].to(device)
                audio_hidden = audio_latents[audio].to(device)
            else:
                hidden, audio_hidden = latents[video], audio_latents[audio]
            count = 0
            if keystone is not None:
                layout, (timestep, timestep_indices) = keystone_inputs(
                    fields, layout, (timestep, timestep_indices), keystone.shape[0]
                )
                count = keystone.shape[0]
                hidden = torch.cat([keystone.to(hidden.device), hidden])
            noise_pred, _ = transformer(
                hidden_states=hidden[None],
                audio_hidden_states=audio_hidden[None],
                encoder_hidden_states=block_state.prompt_embeds,
                timestep=timestep,
                timestep_indices=timestep_indices,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **{k: v for k, v in layout.items() if k in accepted},
            )
            if count:
                noise_pred = noise_pred[:, count:]
            if host:
                noise_pred = noise_pred.float().cpu()
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
        # float32 rounding, and is exactly 1 for a single window. A held row
        # only the skipped windows cover has no weight, and no velocity
        if is_held is not None:
            total[is_held] = 1.0
            fused[0, is_held] = 0.0
        block_state.noise_pred = fused / total[None, :, None]
        block_state.audio_noise_pred = torch.zeros(
            (1, *audio_latents.shape), dtype=torch.float32, device=fused.device
        )
        return components, block_state


class DwH3HostLoopSchedulerStep(MiniMaxH3LoopSchedulerStep):
    """The stock update, unchanged for every call. For a windowed generation
    (#1188) the take is on the host, and so are the fused velocities: the
    scheduler moves its sigmas to the sample's device, so the step runs
    there, and after the last step the take goes back to the device it came
    from for the release and the decode. A step before `switch_step` was run
    by the ordered phase, so it only moves the schedulers on a step; after
    it, the rows the first window left finished are put back as they were."""

    model_name = "minimax-h3"

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        device = getattr(block_state, HYBRID_DEVICE, None)
        if device is not None and i < (getattr(block_state, HYBRID_SWITCH, 0) or 0):
            components.scheduler._step_index = i + 1
            components.audio_scheduler._step_index = i + 1
        else:
            held = (
                getattr(block_state, HYBRID_HELD, None) if device is not None else None
            )
            kept = block_state.latents[held].clone() if held is not None else None
            components, block_state = super().__call__(components, block_state, i, t)
            if held is not None:
                block_state.latents[held] = kept
        timesteps = getattr(block_state, "timesteps", None)
        if device is not None and timesteps is not None and i == len(timesteps) - 1:
            block_state.latents = block_state.latents.to(device)
            block_state.audio_latents = block_state.audio_latents.to(device)
            setattr(block_state, HYBRID_DEVICE, None)
            setattr(block_state, HYBRID_HELD, None)
            setattr(block_state, HYBRID_KEYSTONE, None)
        return components, block_state


def keystone_inputs(fields, layout, plan_entry, count):
    """A window's layout and `(unique_timesteps, timestep_indices)` with the
    keystone's `count` rows (#1188 S4, design 8) spliced in ahead of its media,
    where a keyframe's condition rows sit: the take's first latent frame's
    positions and video tag, unshifted - under rebased positions they land on
    the window's own first frame, as a frame-0 keyframe would - and the clean
    timestep. They are the first `count` of the new `video_indices`, so the
    forward's video prediction leads with theirs.

    The rows go in by the guide's splice (h3_guide_steps.splice_guide_rows): a
    window is [text][audio][window video], a guide layout with no condition
    rows whose text ends where its media starts, so the keystone lands as a
    guide would. The splice carries no timesteps; the keystone's go in here, at
    the clean timestep."""
    first = fields["video_indices"][:count]
    media = torch.cat([layout["audio_indices"], layout["video_indices"]])
    at = int(media.min())
    position_ids, token_tags, video_indices, audio_indices, _, _ = splice_guide_rows(
        (
            layout["position_ids"],
            layout["token_tags"],
            layout["video_indices"],
            layout["audio_indices"],
            layout.get("text_indices"),
            0,
        ),
        at,
        fields["position_ids"][first],
        int(fields["token_tags"][first[0]]),
    )
    unique, timestep_indices = plan_entry
    timestep_indices = torch.cat(
        [
            timestep_indices[:at],
            timestep_indices.new_zeros(count),
            timestep_indices[at:],
        ]
    )
    spliced = {
        **layout,
        "position_ids": position_ids,
        "token_tags": token_tags,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
    }
    return spliced, clean_timesteps(
        (unique, timestep_indices), video_indices[:count].to(timestep_indices.device)
    )
