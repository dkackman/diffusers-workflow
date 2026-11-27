"""MiniMax-H3 time windows: the plan, the fuse weights and the call rules
(#1199 Task 17 Step 4, #1230, #1188) - the half of the windowed denoiser
(dw/pipeline_processors/h3_windows.py) that touches no pipeline.

Owns, for `refine_windows` and for `experimental_hybrid_windows` (the windowed
generation, refused unless the process sets `DW_EXPERIMENTAL_HYBRID_WINDOWS=1`):

- the settings and what an absent key means (`WINDOW_DEFAULTS`,
  `HYBRID_DEFAULTS`, `window_settings`, `hybrid_settings`);
- the plan on the latent time grid and its weights (`window_plan`,
  `fuse_weights`), and the clip-grid rule rebased positions need
  (`rebase_problem`);
- the rules of the call, as lists of problems (`window_problems`,
  `hybrid_problems`), asked at validate time (dw/guides.py `windows_errors`)
  and again before the call by h3_windows's `windows_refusal` and
  `hybrid_refusal`, which add the one thing that needs the pipeline.

The denoiser reads the plan, the weights and the settings from here; nothing
here reads the denoiser, so the import runs one way.

Imports torch (the weights are tensors) and diffusers (`H3_MAX_SECONDS` is
read off its pipeline) when it loads, so nothing imports it at module level:
h3_windows.py does, itself imported on first use, and dw/guides.py imports it
inside `windows_errors`.
"""

import math

import torch
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline

from .. import references
from .h3_rules import (
    GUIDE_LATENTS_PER_CHUNK,
    GUIDES_INPUT,
    HOLD_AUDIO_INPUT,
    HYBRID_WINDOWS_ENV_VAR,
    HYBRID_WINDOWS_INPUT,
    LATENT_ANCHOR_INPUT,
    MASK_INPUTS,
    REFINE_STRENGTH_INPUT,
    REFINE_TILES_INPUT,
    REFINE_WINDOWS_INPUT,
    H3_FPS,
    RENDER_GRID,
    SOURCE_VIDEO_INPUT,
    guide_latent_frames,
    hybrid_windows_enabled,
)

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
