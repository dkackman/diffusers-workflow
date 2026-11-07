"""MiniMax-H3's pure rules: the argument and block names dw's H3 blocks use,
the guide grid, and the checks validation and the run both ask before anything
loads - the rule half of the H3 adapter.

Torch- and diffusers-free, so validation reads these without the block factories:
the hold, refine and release blocks are dw/pipeline_processors/h3_hold.py, the
guide layout dw/pipeline_processors/h3_guides.py. tests/test_h3_rules.py pins
that it imports neither.
"""

import math
import os
from array import array

from .. import references
from ..adapter_compatibility import FROM_PRETRAINED_KEY, H3_WORKFLOWS, WORKFLOW_KEY
from ..variable_constraints import aligned, aligned_down

MODULAR_PIPELINE = "ModularPipeline"

# The call argument, and the names the hold block leaves in the pipeline state
HOLD_AUDIO_INPUT = "hold_audio"
HELD_ROWS = "dw_held_audio_rows"
HELD_AUDIO_OUTPUT = "dw_held_audio"
HELD_AUDIO_RATE_OUTPUT = "dw_held_audio_sampling_rate"
HELD_LATENTS_OUTPUT = "dw_held_audio_latents"

# The refine call arguments: the sigma to start from, and an explicit schedule
# from it down to 0 in place of the shift-spaced one (#1082)
REFINE_STRENGTH_INPUT = "refine_strength"
REFINE_SIGMAS_INPUT = "refine_sigmas"
# How far a `refine_sigmas` first point may sit from `refine_strength`
SIGMA_TOLERANCE = 1e-6

# The block names dw inserts, and the diffusers block each goes in front of
HOLD_BLOCK = "dw_hold_audio"
REFINE_BLOCK = "dw_refine_schedule"
RELEASE_BLOCK = "dw_release_audio"
HOLD_BEFORE = "set_timesteps"
REFINE_BEFORE = "denoise"
RELEASE_BEFORE = "after_denoise"

# MiniMaxH3Blocks: the top-level step holding the auto denoise step, and the three
# core-denoise sequences that step chooses between
DENOISE_STEP = "denoise"
CORE_DENOISE_SEQUENCES = ("t2va", "fl2va", "ref2va")


def refine_problems(arguments, canvas=None):
    """Why this step's `refine_strength`, `refine_sigmas` and `refine_tiles`
    cannot run, as a list - empty when they can, or cannot be told yet. The
    one owner of the per-argument rules: the static check (dw/hold_audio.py)
    and the run-time check before the call (`Pipeline._check_refine`) both ask
    it. The sigma is in (0, 1), or (0, 1] with a `source_video`: a regional
    edit at 1.0 regenerates its region from pure noise. A tile problem's
    message starts with `refine_tiles`, a schedule problem's with
    `refine_sigmas`. `canvas` is the (width, height) the static check read
    from the upscale step `latents` names, when it could; at run time the
    `latents` tensor gives it."""
    if arguments.get(REFINE_STRENGTH_INPUT) is None:
        return [
            f"{name} {does} a refine_strength pass, and this step passes no "
            f"refine_strength - add one, about 0.2"
            for name, does in (
                (REFINE_SIGMAS_INPUT, "schedules"),
                (REFINE_TILES_INPUT, "tiles"),
            )
            if arguments.get(name) is not None
        ]
    problems = []
    strength = arguments[REFINE_STRENGTH_INPUT]
    if isinstance(strength, str) and references.is_ref(references.UNRESOLVED, strength):
        pass
    elif isinstance(strength, bool) or not isinstance(strength, (int, float)):
        problems.append(
            f"refine_strength is a number in (0, 1), and {strength!r} is not a number"
        )
    elif arguments.get(SOURCE_VIDEO_INPUT) is not None:
        # A regional edit may start from pure noise: 1.0 regenerates the region
        if not 0 < strength <= 1:
            problems.append(
                f"refine_strength is a sigma in (0, 1] with a source_video - 1.0 "
                f"regenerates the masked region from noise, about 0.9 replaces an "
                f"object in it - and {strength} is outside it"
            )
    elif not 0 < strength < 1:
        problems.append(
            f"refine_strength is a sigma in (0, 1) - about 0.2 refines an "
            f"upscaled take - and {strength} is outside it"
        )
    if arguments.get("latents") is None and arguments.get(SOURCE_VIDEO_INPUT) is None:
        problems.append(
            "refine_strength re-denoises the 'latents' it is passed - pass the "
            "upscaled latents, e.g. 'previous_result:up'"
        )
    # An audio pass refuses refine_strength itself (h3_audio_pass.py)
    if arguments.get(HOLD_AUDIO_INPUT) is None and not is_audio_pass(arguments):
        problems.append(
            "refine_strength re-denoises the video only, so it needs 'hold_audio' "
            "to keep a soundtrack - e.g. the base pass's 'previous_result:base.audio'"
        )
    steps = arguments.get("num_inference_steps")
    if isinstance(steps, (int, float)) and not isinstance(steps, bool) and steps < 2:
        problems.append(
            f"refine_strength runs num_inference_steps - 1 denoise steps, so "
            f"num_inference_steps must be 2 or more, not {steps}"
        )
    if arguments.get(REFINE_SIGMAS_INPUT) is not None:
        problems.extend(_sigma_problems(arguments))
    if arguments.get(REFINE_TILES_INPUT) is not None:
        problems.extend(_tile_problems(arguments, canvas))
    elif canvas is not None:
        problems.extend(_full_frame_problems(arguments, canvas))
    return problems


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sigma_problems(arguments):
    """Why this step's `refine_sigmas` cannot run - a list, empty when it can.
    The list is the whole schedule, sigma points down to 0, standing in for
    the shift-spaced one: at least 2 points, strictly decreasing, from
    `refine_strength` (within SIGMA_TOLERANCE) to 0, one point per
    `num_inference_steps` - which is required beside it and must agree."""
    sigmas = arguments[REFINE_SIGMAS_INPUT]
    if _deferred(sigmas):
        return []
    if not isinstance(sigmas, (list, tuple)) or not all(map(_number, sigmas)):
        return [
            f"refine_sigmas is a list of sigma points from refine_strength down "
            f"to 0, and {sigmas!r} is not a list of numbers"
        ]
    problems = []
    if not all(math.isfinite(s) for s in sigmas):
        return [f"refine_sigmas must be finite numbers, and {list(sigmas)} is not"]
    # The scheduler compares the points as float32: two that differ only past
    # that precision collapse there and are refused after the load, so they
    # are refused here first
    as_stored = [float(array("f", [s])[0]) for s in sigmas]
    if len(sigmas) < 2:
        problems.append(
            f"refine_sigmas needs at least 2 points, refine_strength and 0, and "
            f"has {len(sigmas)}"
        )
    elif any(later >= earlier for earlier, later in zip(as_stored, as_stored[1:])):
        problems.append(
            f"refine_sigmas must be strictly decreasing (as float32, the "
            f"scheduler's precision), and {list(sigmas)} is not"
        )
    strength = arguments[REFINE_STRENGTH_INPUT]
    if sigmas and _number(strength) and abs(sigmas[0] - strength) > SIGMA_TOLERANCE:
        problems.append(
            f"refine_sigmas starts at refine_strength, {strength}, and its first "
            f"point is {sigmas[0]}"
        )
    if sigmas and sigmas[-1] != 0:
        problems.append(f"refine_sigmas ends at 0, and its last point is {sigmas[-1]}")
    steps = arguments.get("num_inference_steps")
    if steps is None:
        problems.append(
            f"refine_sigmas has {len(sigmas)} points, and this step sets no "
            f"num_inference_steps - set it to {len(sigmas)}, one per point "
            f"({len(sigmas) - 1} evaluations)"
        )
    elif _number(steps) and steps != len(sigmas):
        problems.append(
            f"refine_sigmas has {len(sigmas)} points, {len(sigmas) - 1} "
            f"evaluations, and num_inference_steps {steps} runs {steps - 1} - "
            f"set num_inference_steps to {len(sigmas)}"
        )
    return problems


def _full_frame_problems(arguments, canvas):
    """A full-frame refine's own width and height against the `latents` canvas
    the static check read: an omitted size falls to the layout default
    (TILE_LIMIT), and any size but the canvas's fails at the denoise."""
    problems = []
    for key, size, default in zip(("width", "height"), canvas, TILE_LIMIT):
        value = arguments.get(key)
        if value is None:
            if size != default:
                problems.append(
                    f"refine_strength re-denoises the 'latents' canvas, "
                    f"{canvas[0]}x{canvas[1]}, and this step sets no {key}, so its "
                    f"layout falls to {TILE_LIMIT[0]}x{TILE_LIMIT[1]} - set width "
                    f"{canvas[0]} and height {canvas[1]}"
                )
                break
        elif _is_int(value) and value != size:
            problems.append(
                f"refine_strength re-denoises the 'latents' canvas, "
                f"{canvas[0]}x{canvas[1]}, and the step's {key} {value} disagrees "
                f"- set it to {size}"
            )
    return problems


# refine_tiles (#1043): a refine above H3's 768p canvas, run as overlapping
# tiles of at most TILE_LIMIT, blended in latent space and decoded once
REFINE_TILES_INPUT = "refine_tiles"
# The canvas noise drawn once and cropped per tile, handed to the refine block
REFINE_NOISE = "dw_refine_noise"
TILE_LIMIT = (1344, 768)
TILE_GRID = 32
TILE_OVERLAP_FLOOR = 192
# Pixels per latent cell: a `latents` canvas is its latent height x width x 16
LATENT_SCALE = 16
# The rate H3 generates video at
H3_FPS = 24
# The experimental time-windowed refine (#1199 Task 17 Step 4,
# dw/pipeline_processors/h3_windows.py): its call argument, and the scratch flag
# the loader reads before it puts the windowed denoiser in. Here, not in
# h3_windows.py, because that module imports diffusers and the loader reads the
# flag on every H3 load
REFINE_WINDOWS_INPUT = "experimental_refine_windows"
REFINE_WINDOWS_ENV_VAR = "DW_EXPERIMENTAL_REFINE_WINDOWS"


def refine_windows_enabled():
    """Whether the windowed refine's scratch flag is on in this process."""
    return os.environ.get(REFINE_WINDOWS_ENV_VAR) == "1"


# The experimental SelfLift (#1195, #1199 Task 19; h3_selflift.py): the early
# steps of one schedule on a smaller canvas, lifted to the call's canvas mid
# schedule. Behind REFINE_WINDOWS_ENV_VAR, the long-form spikes' one flag
SELFLIFT_INPUT = "experimental_selflift"


# The latent statistics anchor's call argument (#1166; see ANCHOR_BLOCK)
LATENT_ANCHOR_INPUT = "latent_anchor"
# How far the match goes (#1190): 1 (default) is the full per-channel match, 0
# leaves the latent alone; a blend in between
LATENT_ANCHOR_STRENGTH_INPUT = "latent_anchor_strength"

# The arguments that condition a step on a reference, a keyframe or a guide: the
# tile plan has no place to put one, and each tile would need its own crop
TILE_CONFLICTS = (
    "image",
    "last_image",
    "references",
    "guides",
    "source_video",
    "mask_video",
    "mask_image",
    LATENT_ANCHOR_INPUT,
)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _deferred(value):
    return isinstance(value, str) and references.is_ref(references.UNRESOLVED, value)


def _latent_canvas(arguments, canvas):
    """The (width, height) of the `latents` canvas, or None when not yet known:
    the tensor's own at run time, else the one the static check read from the
    upscale step."""
    shape = getattr(arguments.get("latents"), "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1]) * LATENT_SCALE, int(shape[-2]) * LATENT_SCALE
    return canvas


_TILE_KEYS = ("width", "height", "overlap")


def _tile_problems(arguments, canvas):
    """Why this step's `refine_tiles` cannot run - a list, empty when it can."""
    tiles = arguments[REFINE_TILES_INPUT]
    if _deferred(tiles):
        return []
    if not isinstance(tiles, dict):
        return [
            f"refine_tiles is {{width, height, overlap}} in pixels, and "
            f"{tiles!r} is not a mapping"
        ]
    problems, known = _tile_mapping_problems(tiles)
    width, height, overlap = (known.get(key) for key in _TILE_KEYS)
    problems += _tile_size_problems(width, height, overlap)
    problems += _tile_conflict_problems(arguments)
    problems += _tile_canvas_problems(arguments, canvas, width, height)
    return problems


def _tile_mapping_problems(tiles):
    """The key and per-value rules of the `refine_tiles` mapping: the problems,
    and the values that passed, by key."""
    problems = []
    missing = [key for key in _TILE_KEYS if key not in tiles]
    extra = sorted(str(key) for key in tiles if key not in _TILE_KEYS)
    if missing:
        problems.append(
            f"refine_tiles needs width, height and overlap, and is missing "
            f"{', '.join(missing)}"
        )
    if extra:
        problems.append(
            f"refine_tiles takes only width, height and overlap, not {', '.join(extra)}"
        )
    known = {}
    for key in _TILE_KEYS:
        value = tiles.get(key)
        if key not in tiles or _deferred(value):
            continue
        if not _is_int(value):
            problems.append(
                f"refine_tiles {key} is an integer number of pixels, and "
                f"{value!r} is not an integer"
            )
        else:
            known[key] = value
            if value % TILE_GRID:
                problems.append(
                    f"refine_tiles {key} is a multiple of {TILE_GRID} px, and "
                    f"{value} is not"
                )
    return problems, known


def _tile_size_problems(width, height, overlap):
    """The tile against H3's trained canvas, and the overlap against the tile:
    each value None when not known."""
    problems = []
    if width is not None and height is not None:
        longer, shorter = max(TILE_LIMIT), min(TILE_LIMIT)
        if not (width <= longer and height <= shorter) and not (
            width <= shorter and height <= longer
        ):
            problems.append(
                f"refine_tiles tiles are at most {longer}x{shorter} (or "
                f"{shorter}x{longer}) - H3's trained canvas - and {width}x{height} "
                f"is larger"
            )
        if overlap is not None and overlap >= min(width, height) // 2:
            problems.append(
                f"refine_tiles overlap must be under half the tile's shorter side "
                f"({min(width, height) // 2} px for {width}x{height}), and {overlap} "
                f"is not"
            )
    if overlap is not None and overlap < TILE_OVERLAP_FLOOR:
        problems.append(
            f"refine_tiles overlap is at least {TILE_OVERLAP_FLOOR} px, or the seams "
            f"show, and {overlap} is below it"
        )
    return problems


def _tile_conflict_problems(arguments):
    """The arguments a tiled refine cannot be combined with."""
    problems = []
    for key in TILE_CONFLICTS:
        if arguments.get(key) is not None:
            problems.append(
                f"refine_tiles refines the 'latents' canvas alone and cannot be "
                f"combined with '{key}'"
            )
    prompt = arguments.get("prompt")
    if isinstance(prompt, str) and "<Picture" in prompt:
        problems.append(
            "refine_tiles refines the 'latents' canvas alone, and the prompt "
            "names a <Picture N> reference"
        )
    return problems


def _tile_canvas_problems(arguments, canvas, width, height):
    """The tile against the canvas it refines, and the step's own width and
    height against that canvas when the `latents` canvas is known."""
    problems = []
    derived = _latent_canvas(arguments, canvas)
    # With no latents canvas known, the step's own explicit size stands in
    sizes = derived or tuple(
        arguments[key] if _is_int(arguments.get(key)) else None
        for key in ("width", "height")
    )
    if any(
        size is not None and tile is not None and tile > size
        for size, tile in zip(sizes, (width, height))
    ):
        problems.append(
            f"refine_tiles {width}x{height} is larger than the {sizes[0]}x"
            f"{sizes[1]} canvas - use a tile inside it"
        )
    if derived is not None:
        for key, size in zip(("width", "height"), derived):
            value = arguments.get(key)
            if _is_int(value) and value != size:
                problems.append(
                    f"refine_tiles refines the 'latents' canvas, {derived[0]}x"
                    f"{derived[1]}, and the step's {key} {value} disagrees - drop "
                    f"it or match the canvas"
                )
    return problems


def _axis_starts(length, tile, overlap):
    """The tile origins along one axis: the fewest tiles whose every overlap is
    at least `overlap`, spread evenly on the TILE_GRID, the last flush to the
    edge."""
    if tile >= length:
        return [0]
    span = length - tile
    count = 2 + max(0, -(-span // (tile - overlap)) - 1)
    while True:
        starts = [
            (span * index // (count - 1)) // TILE_GRID * TILE_GRID
            for index in range(count - 1)
        ] + [span]
        if all(
            tile - (later - earlier) >= overlap
            for earlier, later in zip(starts, starts[1:])
        ):
            return starts
        count += 1


def tile_plan(width, height, tile_width, tile_height, overlap):
    """The tiles covering a `width` x `height` canvas, as (x, y, w, h) in
    pixels, row-major: per axis the fewest tiles whose overlap is at least
    `overlap`, spread evenly on the 32 px grid, the last flush to the edge.
    1920x1088 at 1344x768/256 is 2x2. A pure function - the run and its
    tests both ask it."""
    tile_width, tile_height = min(tile_width, width), min(tile_height, height)
    return [
        (x, y, tile_width, tile_height)
        for y in _axis_starts(height, tile_height, overlap)
        for x in _axis_starts(width, tile_width, overlap)
    ]


def not_h3(pipeline, argument=HOLD_AUDIO_INPUT, taken_by=H3_WORKFLOWS):
    """Why this step's pipeline cannot take `argument` - one of the H3 block
    arguments - or None when it can or cannot be told before the load.
    `taken_by` is the H3 workflows the message names as taking it: an
    argument ref2va refuses (guides, latent_anchor) passes t2va/fl2va only,
    so the refusal does not send the author to a workflow that refuses it
    too. It only words the message - an H3 workflow outside it is left to
    the caller's own rule, which says why."""
    configuration = pipeline.get("configuration")
    component_type = (
        configuration.get("component_type") if isinstance(configuration, dict) else None
    )
    from_pretrained = pipeline.get(FROM_PRETRAINED_KEY)
    workflow = (
        from_pretrained.get(WORKFLOW_KEY) if isinstance(from_pretrained, dict) else None
    )
    holds = (
        f"{argument} is a MiniMax-H3 argument, taken by its "
        f"{', '.join(sorted(taken_by))} workflows"
    )
    if (
        isinstance(component_type, str)
        and not references.is_ref(references.UNRESOLVED, component_type)
        and component_type.rsplit(".", 1)[-1] != MODULAR_PIPELINE
    ):
        return f"{holds}, and this step loads {component_type}"
    if isinstance(workflow, str) and workflow not in H3_WORKFLOWS:
        return f"{holds}, and this step loads the '{workflow}' workflow"
    return None


# The call argument, and the most guides one call may carry - a VRAM guard: every
# guide frame is another frame of rows for attention to cover (#648)
GUIDES_INPUT = "guides"
GUIDE_LIMIT = 4

# The diffusers step the layout replaces, and the ones t2va borrows from fl2va
LAYOUT_STEP = "prepare_layout"
GUIDE_CONDITION_BLOCK = "dw_guide_condition_latents"
GUIDE_LATENTS_BLOCK = "dw_guide_latents"
# The block that puts the guides' audio rows in front of the target audio, and
# the state key the layout leaves them under (#649)
GUIDE_AUDIO_BLOCK = "dw_guide_audio"
GUIDE_AUDIO_ROWS = "dw_guide_audio_latents"
# The experimental late capstone (#1187, #1199 Task 14 Step 4; h3_windows.py):
# a guide's keys for a one-frame row timed one latent past the render's end,
# held out of the forward until sigma falls below its gate, and the state key
# the layout leaves the gated rows under. Behind REFINE_WINDOWS_ENV_VAR
CAPSTONE_PAST_END = "past_end"
CAPSTONE_GATE = "enter_below_sigma"
GATED_GUIDE_ROWS = "dw_gated_guide_rows"

# Fixed guide noise (#1166, after PrefixStream): pixel noise on every video
# guide, tapered by `guide_noise_taper` to clean at the guide's last frame, put
# into its latent by encoding the noised pixels (h3_guide_steps.DwH3GuideLayoutStep). The
# rows stay held condition rows; audio is never noised.
GUIDE_NOISE_INPUT = "guide_noise"


def guide_noise_problem(value):
    """Why `value` cannot be a `guide_noise`, naming the range, or None. An
    unresolved reference is left to the run."""
    if isinstance(value, str) and references.is_ref(references.UNRESOLVED, value):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return (
            f"guide_noise is a number in [0, 1) - about 0.4 - and {value!r} is "
            f"not a number"
        )
    if not 0 <= value < 1:  # NaN fails it too
        return (
            f"guide_noise is a number in [0, 1) - about 0.4; 0 adds none - and "
            f"{value} is outside it"
        )
    return None


def guide_noise_taper(num_frames, strength):
    """The noise amplitude on each of a guide's `num_frames` frames: a cosine
    taper from `strength` at the first frame to 0 at the last, the frame next
    to the join. A one-frame guide is all join, so it is clean."""
    if num_frames <= 1:
        return [0.0] * num_frames
    return [
        strength * 0.5 * (1.0 + math.cos(math.pi * index / (num_frames - 1)))
        for index in range(num_frames)
    ]


# Pixel frames per VAE chunk and latent frames per chunk - a clip encodes to whole
# latents at 1, 5 or 17m + 5 frames, and lines up with the target's latent grid
# only at a chunk boundary, frame 17j (latent 5j). Copies of the H3 video VAE's
# `clip_length` and its latents per clip, pinned by tests/test_h3_guides.py
GUIDE_FRAMES_PER_CHUNK = 17
GUIDE_LATENTS_PER_CHUNK = 5
# The 17n + 5 frame grid a render and a long clip sit on, as a constraint grid for
# `variable_constraints.aligned` / `aligned_down` - the owner of that arithmetic
RENDER_GRID = {"modulus": GUIDE_FRAMES_PER_CHUNK, "remainder": 5}


def snap_guide_length(num_frames):
    """The longest whole-latent clip length - 1, 5 or 17m + 5 - not over
    `num_frames`. 23 -> 22, 40 -> 39, 3 -> 1."""
    if num_frames < 5:
        return 1
    if num_frames < 17 + 5:
        return 5
    return aligned_down(num_frames, RENDER_GRID)


def guide_latent_frames(num_frames):
    """Latent frames an aligned clip encodes to: 1, 2, or 5m + 2."""
    if num_frames == 1:
        return 1
    return (num_frames - 5) // GUIDE_FRAMES_PER_CHUNK * GUIDE_LATENTS_PER_CHUNK + 2


def guide_frame_problem(frame):
    """Why `frame` cannot place a guide, or None - a whole, non-negative pixel
    frame on a chunk boundary (17j)."""
    if isinstance(frame, bool) or not isinstance(frame, int):
        return f"'frame' must be a whole pixel frame, got {frame!r}"
    if frame < 0:
        return f"'frame' cannot be negative, got {frame}"
    if frame % GUIDE_FRAMES_PER_CHUNK:
        below = frame // GUIDE_FRAMES_PER_CHUNK * GUIDE_FRAMES_PER_CHUNK
        return (
            f"'frame' must be a multiple of {GUIDE_FRAMES_PER_CHUNK} (a VAE chunk "
            f"boundary, where a guide lines up with the generated frames), got "
            f"{frame} - use {below} or {below + GUIDE_FRAMES_PER_CHUNK}"
        )
    return None


# A chain's `continuity: "guide"` (dw/pipeline_processors/chain.py) lays the
# previous segment's last `guide_frames` frames in at frame 0 of the next. Only
# these lengths are whole-latent guides (17m + 5) short enough to leave a
# segment most of its frames
GUIDE_CONTINUITY = "guide"
GUIDE_CHAIN_FRAMES = (22, 39)
GUIDE_CHAIN_DEFAULT = 22
# guide_backoff (#1189) moves the carry back whole 17-frame blocks
GUIDE_BACKOFF_BLOCK = 17
# Where a guide chain runs is where `guides` runs (dw/guides.py); the run and
# validate lead their refusal with this rule
GUIDE_CHAIN_RULE = (
    "continuity 'guide' runs on MiniMax-H3 t2va or fl2va only - guides stay off ref2va"
)

# Every chain continuity mode, in the order chain_continuity.py registers its classes
# (`CONTINUITY_MODES`, zipped strictly against these). Kept here so validation
# (dw/guides.py) reads the same names without importing chain_continuity.py
CHAIN_CONTINUITY_MODES = ("last_frame", "last_segment", GUIDE_CONTINUITY)

# The latent statistics anchor (#1166): `latent_anchor` is video frames whose
# per-channel latent mean/std the generated video latent is rescaled to, after
# `after_denoise` and before decode, by the ANCHOR_BLOCK step
# (dw/pipeline_processors/h3_anchor.py). A chain's `anchor: "segment_1"` sets it
# on every segment after the first to segment 1's decoded frames.
ANCHOR_BLOCK = "dw_latent_anchor"
ANCHOR_AFTER = "after_denoise"
ANCHOR_NONE = "none"
ANCHOR_SEGMENT_1 = "segment_1"
CHAIN_ANCHOR_MODES = (ANCHOR_NONE, ANCHOR_SEGMENT_1)
# The continuities whose segments are fresh t2va/fl2va generations the anchor
# block runs in; `last_segment` carries a ref2va reference, which has no block
ANCHOR_CONTINUITIES = ("last_frame", GUIDE_CONTINUITY)


# The keystone (#1187): segment 1's frame 0 as every later segment's fl2va
# keyframe, instead of the guide's first frame. Guide continuity only
KEYSTONE_NONE = "none"
KEYSTONE_SEGMENT_1 = "segment_1"
CHAIN_KEYSTONE_MODES = (KEYSTONE_NONE, KEYSTONE_SEGMENT_1)


def anchor_strength_problem(value):
    """Why `value` cannot be a `latent_anchor_strength` / chain `anchor_strength`,
    naming the range, or None. An unresolved reference is left to the run."""
    if isinstance(value, str) and references.is_ref(references.UNRESOLVED, value):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return (
            f"anchor strength is a number in [0, 1] - 1 is the full match, 0 "
            f"none - and {value!r} is not a number"
        )
    if not 0 <= value <= 1:  # NaN fails it too
        return (
            f"anchor strength is a number in [0, 1] - 1 is the full match, 0 "
            f"none - and {value} is outside it"
        )
    return None


def chain_keystone_problem(chain):
    """The reason a chain block's `keystone` is refused, or None: an unknown
    value (naming the values), or `segment_1` without `continuity: "guide"`.
    Unresolved references are left to the run."""
    keystone = chain.get("keystone", KEYSTONE_NONE)
    if isinstance(keystone, str) and references.is_ref(references.UNRESOLVED, keystone):
        return None
    if keystone not in CHAIN_KEYSTONE_MODES:
        return (
            f"unknown chain keystone {keystone!r} - expected one of "
            f"{', '.join(CHAIN_KEYSTONE_MODES)}"
        )
    if keystone == KEYSTONE_NONE:
        return None
    continuity = chain.get("continuity", "last_frame")
    if isinstance(continuity, str) and references.is_ref(
        references.UNRESOLVED, continuity
    ):
        return None
    if continuity != GUIDE_CONTINUITY:
        return (
            f"keystone 'segment_1' needs continuity 'guide' - the keystone "
            f"replaces the keyframe, and on {continuity!r} the keyframe is the "
            f"only carry; got {continuity!r}"
        )
    return None


# The carry pick (#1032): the sharpest of the last N frames as the next
# keyframe, the segment trimmed after it. Family-independent, `last_frame` only
CARRY_PICK_MIN = 1
CARRY_PICK_MAX = 8


def _set_for_plan(chain, *keys):
    """Whether a key is set; an unsubstituted reference is not (#1232)."""
    values = [chain.get(k) for k in keys]
    values = [v for v in values if not references.is_ref(references.UNRESOLVED, v)]
    return any(v is not None and v is not False for v in values)


def chain_carry_pick_problem(chain, num_frames=None):
    """The reason a chain block's `carry_pick` is refused, or None: a shape
    other than {"sharpest": N}, N outside 1..8, a continuity other than
    `last_frame`, `match_audio` (the plan is fixed from the track's length and
    the pick drops frames per seam), or - when the step's `num_frames` is known
    - an N that could empty a segment. Unresolved references are left to the
    run."""
    if "carry_pick" not in chain:
        return None
    pick = chain["carry_pick"]
    if isinstance(pick, str) and references.is_ref(references.UNRESOLVED, pick):
        return None
    expected = (
        f"expected {{'sharpest': N}} with N from {CARRY_PICK_MIN} to {CARRY_PICK_MAX}"
    )
    if not isinstance(pick, dict) or set(pick) != {"sharpest"}:
        return f"invalid carry_pick {pick!r} - {expected}"
    window = pick["sharpest"]
    if isinstance(window, str) and references.is_ref(references.UNRESOLVED, window):
        return None
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or not CARRY_PICK_MIN <= window <= CARRY_PICK_MAX
    ):
        return (
            f"carry_pick sharpest must be a whole number between {CARRY_PICK_MIN} "
            f"and {CARRY_PICK_MAX}, got {window!r}"
        )
    continuity = chain.get("continuity", "last_frame")
    if isinstance(continuity, str) and references.is_ref(
        references.UNRESOLVED, continuity
    ):
        return None
    if continuity != "last_frame":
        return (
            f"carry_pick needs continuity 'last_frame' - the other modes carry "
            f"more than one frame; got {continuity!r}"
        )
    if _set_for_plan(chain, "match_audio", "duration_seconds"):
        return (
            "carry_pick does not combine with match_audio or duration_seconds: "
            "the plan covers the length up front and the pick drops up to "
            f"{CARRY_PICK_MAX - 1} more frames per seam, leaving the video "
            "shorter than asked - use 'segments'"
        )
    trim = chain.get("trim_frames", 1)
    if isinstance(num_frames, int) and isinstance(chain.get("frame_snap"), dict):
        num_frames = aligned(num_frames, chain["frame_snap"]) or num_frames
    if (
        isinstance(num_frames, int)
        and not isinstance(num_frames, bool)
        and isinstance(trim, int)
        and not isinstance(trim, bool)
        and window >= num_frames - trim
    ):
        return (
            f"carry_pick sharpest {window} must be less than num_frames "
            f"{num_frames} minus trim_frames {trim} ({num_frames - trim}), or "
            "the pick could empty a segment"
        )
    return None


def chain_anchor_problem(chain):
    """The reason a chain block's `anchor` is refused for its own settings, or
    None: an unknown value (naming the values), or `segment_1` with a continuity
    other than guide or last_frame, or carrying into `references`. Unresolved
    references are left to the run;
    whether the step is H3 is the caller's to ask (`not_h3`)."""
    anchor = chain.get("anchor", ANCHOR_NONE)
    if isinstance(anchor, str) and references.is_ref(references.UNRESOLVED, anchor):
        return None
    if anchor not in CHAIN_ANCHOR_MODES:
        return (
            f"unknown chain anchor {anchor!r} - expected one of "
            f"{', '.join(CHAIN_ANCHOR_MODES)}"
        )
    if anchor == ANCHOR_NONE:
        if chain.get("anchor_strength") is not None:
            return (
                "anchor_strength needs anchor 'segment_1' - it sets how far "
                "the anchor's latent match goes, and anchor is 'none'"
            )
        return None
    continuity = chain.get("continuity", "last_frame")
    if isinstance(continuity, str) and references.is_ref(
        references.UNRESOLVED, continuity
    ):
        return None
    if continuity not in ANCHOR_CONTINUITIES:
        return (
            f"anchor 'segment_1' needs continuity "
            f"{' or '.join(repr(c) for c in ANCHOR_CONTINUITIES)} - its segments "
            f"are MiniMax-H3 t2va/fl2va generations; got {continuity!r}"
        )
    # The carry lands in `segment_argument` on every later segment, and a call
    # that passes references runs ref2va, which has no anchor step - the
    # anchor would be dropped without a word
    if chain.get("segment_argument") == "references":
        return (
            "anchor 'segment_1' cannot carry into 'references' - a segment that "
            "passes references runs ref2va, which has no anchor step; use "
            "segment_argument 'image' (t2va/fl2va)"
        )
    return None


def guide_chain_problems(chain):
    """[(key, message)] for a guide-continuity chain block's own settings: a
    `guide_frames` other than 22 or 39 (refused, not snapped), and a
    `carry_frames`, which a guide chain does not read. An unresolved reference
    is left to the run."""
    problems = []
    lengths = " or ".join(str(length) for length in GUIDE_CHAIN_FRAMES)
    frames = chain.get("guide_frames", GUIDE_CHAIN_DEFAULT)
    unresolved = isinstance(frames, str) and references.is_ref(
        references.UNRESOLVED, frames
    )
    if not unresolved and (
        isinstance(frames, bool)
        or not isinstance(frames, int)
        or frames not in GUIDE_CHAIN_FRAMES
    ):
        problems.append(
            (
                "guide_frames",
                f"guide_frames must be {lengths} (the whole-latent guide lengths "
                f"a chain carries), got {frames!r}",
            )
        )
    if chain.get("carry_frames") is not None:
        problems.append(
            (
                "carry_frames",
                "carry_frames is a last_segment setting - a guide chain carries "
                "the last guide_frames frames; drop carry_frames",
            )
        )
    return problems


def guide_backoff_problems(chain):
    """[(key, message)] for a chain block's `guide_backoff` (#1189): a whole
    number of 17-frame blocks, on a guide-continuity chain, planned from a
    segment count - a `match_audio` or `duration_seconds` plan counts every segment's new frames from
    its head trim alone. An unresolved reference is left to the run."""
    if "guide_backoff" not in chain:
        return []
    backoff = chain["guide_backoff"]
    if isinstance(backoff, str) and references.is_ref(references.UNRESOLVED, backoff):
        return []
    block = GUIDE_BACKOFF_BLOCK
    if (
        isinstance(backoff, bool)
        or not isinstance(backoff, int)
        or backoff < 0
        or backoff % block
    ):
        return [
            (
                "guide_backoff",
                f"guide_backoff must be a multiple of {block} frames (whole "
                f"latent blocks), got {backoff!r}",
            )
        ]
    if backoff == 0:
        return []
    if chain.get("continuity", "last_frame") != GUIDE_CONTINUITY:
        return [
            (
                "guide_backoff",
                "guide_backoff is a guide continuity setting - set continuity "
                "'guide' or drop guide_backoff",
            )
        ]
    if _set_for_plan(chain, "match_audio", "duration_seconds"):
        return [
            (
                "guide_backoff",
                "guide_backoff does not combine with match_audio or "
                "duration_seconds: the plan covers the length from each "
                "segment's head trim alone, and backing off drops more frames "
                "per segment - use 'segments'",
            )
        ]
    return []


def guide_end_problem(frame, length, num_frames):
    """Why a guide of `length` frames at `frame` runs past a `num_frames` render,
    or None. Ending exactly at `num_frames` fits."""
    if frame + length > num_frames:
        return (
            f"a {length}-frame guide at frame {frame} runs to frame "
            f"{frame + length}, past the end of the {num_frames}-frame render"
        )
    return None


# Regional edit (#1062): re-denoise a `source_video` from `refine_strength`,
# holding every row outside a mask to the source at each step's sigma
# (dw/pipeline_processors/h3_mask.py)
SOURCE_VIDEO_INPUT = "source_video"
MASK_VIDEO_INPUT = "mask_video"
MASK_IMAGE_INPUT = "mask_image"
MASK_GROW_INPUT = "mask_grow"
MASK_INPUTS = (MASK_VIDEO_INPUT, MASK_IMAGE_INPUT)
# The audio pass (#1192): a `source_video` with no mask and no `hold_audio`
# keeps every video row and generates only the audio; `listen_scale` runs the
# kept rows on a smaller canvas (dw/pipeline_processors/h3_mask.py)
LISTEN_SCALE_INPUT = "listen_scale"
LISTEN_SCALES = (1, 0.5, 0.25)
REGIONAL_EDIT_INPUTS = (
    SOURCE_VIDEO_INPUT,
    *MASK_INPUTS,
    MASK_GROW_INPUT,
    LISTEN_SCALE_INPUT,
)
# Patch cells a mask is widened by past what it covers, so the edit's seam
# lands in regenerated rows rather than against a kept one
MASK_GROW_DEFAULT = 1
# A mask pixel at or above this reads as white - regenerate (remove_object's)
MASK_THRESHOLD = 128
# The mask as the source block takes it: (frames, height, width) bool, True to
# regenerate - what Pipeline._with_mask turns either mask argument into
REGENERATE_MASK = "dw_regenerate_mask"
SOURCE_BLOCK = "dw_source_latents"
SOURCE_BEFORE = HOLD_BEFORE
BLEND_BLOCK = "dw_mask_blend"
# The loop sub-block the blend follows, inside the core sequence's `denoise`
BLEND_AFTER = "update"
SOURCE_ROWS = "dw_source_rows"
KEEP_ROWS = "dw_keep_rows"
# The refine block's re-noise of the generated rows, packed - the noise the
# blend re-noises the kept rows with, so they ride the same trajectory
REFINE_ROW_NOISE = "dw_refine_row_noise"
# The audio pass's blocks: one before `denoise` widens the conditioning rows
# over every video row, one before `after_denoise` puts the count back
AUDIO_PASS_BLOCK = "dw_audio_pass"
AUDIO_PASS_BEFORE = REFINE_BEFORE
RELEASE_VIDEO_BLOCK = "dw_release_video"
RELEASE_VIDEO_BEFORE = RELEASE_BEFORE
AUDIO_PASS_ROWS = "dw_audio_pass_rows"
# The source's own frames at full size, uint8 - what the result plays in
# place of the decode (dw/output_extraction.py), as dw_held_audio is for audio
HELD_VIDEO_OUTPUT = "dw_held_video"
# The rate those frames play at: the source's own, never H3's 24 (R25)
HELD_VIDEO_RATE_OUTPUT = "dw_held_video_fps"
# The refine arguments an audio pass has no use for: no video row is re-denoised
AUDIO_PASS_REFUSES = (REFINE_STRENGTH_INPUT, REFINE_SIGMAS_INPUT, REFINE_TILES_INPUT)


def is_h3_step(pipeline):
    """Whether a step's pipeline is known to be H3 before the load - its
    `from_pretrained` `workflow` is one of H3's three."""
    from_pretrained = pipeline.get(FROM_PRETRAINED_KEY)
    workflow = (
        from_pretrained.get(WORKFLOW_KEY) if isinstance(from_pretrained, dict) else None
    )
    return isinstance(workflow, str) and workflow in H3_WORKFLOWS


def edits_region(arguments, h3):
    """Whether a step's arguments ask for a regional edit. `mask_image` and
    `mask_video` are diffusers inpaint arguments too, so on a pipeline not
    known to be H3 (`h3` false) only `source_video` does."""
    if _given(arguments, SOURCE_VIDEO_INPUT):
        return True
    return h3 and any(_given(arguments, name) for name in REGIONAL_EDIT_INPUTS)


def _given(arguments, name):
    return arguments.get(name) is not None


def is_audio_pass(arguments):
    """A `source_video` with no mask and no `hold_audio`: the audio pass, which
    keeps every video row and generates the audio (#1192)."""
    names = (SOURCE_VIDEO_INPUT, *MASK_INPUTS, HOLD_AUDIO_INPUT)
    given = [_given(arguments, name) for name in names]
    return given[0] and not any(given[1:])


def mask_problems(arguments, chained=False):
    """Why this step's regional edit - `source_video`, `mask_video` or
    `mask_image`, `mask_grow` - cannot run, as a list; empty when it can or
    cannot be told yet. The one owner of the argument rules: the static check
    (dw/hold_audio.py `mask_errors`) and the run-time check before the call
    (`Pipeline._with_mask`) both ask it. What needs the media itself - its size
    and frame count against the canvas, an empty mask - is `media_problems`
    and `Pipeline._with_mask`. `chained` is whether the step is a chain."""
    masks = [name for name in MASK_INPUTS if _given(arguments, name)]
    source = _given(arguments, SOURCE_VIDEO_INPUT)
    grow = arguments.get(MASK_GROW_INPUT)
    audio_pass = is_audio_pass(arguments)
    problems = [] if audio_pass else _listen_scale_problems(arguments)
    if not masks and not source and grow is None:
        return problems
    if len(masks) > 1:
        problems.append(
            "pass mask_video or mask_image, not both - mask_image holds one mask "
            "for every frame"
        )
    if masks and not source:
        problems.append(
            f"{masks[0]} marks what to regenerate in a 'source_video' - pass the "
            f"video to edit"
        )
    if grow is not None and not masks:
        problems.append(
            "mask_grow widens a mask_video or mask_image, and this step passes neither"
        )
    elif grow is not None and not _deferred(grow) and (not _is_int(grow) or grow < 0):
        problems.append(
            f"mask_grow is a whole number of patch cells, 0 or more, not {grow!r}"
        )
    if audio_pass:
        problems.extend(_audio_pass_problems(arguments))
    elif source and not _given(arguments, REFINE_STRENGTH_INPUT):
        problems.append(
            "source_video is re-denoised from 'refine_strength' - pass one, near "
            "1.0 to replace what the mask covers, lower to keep its layout"
        )
    if source and _given(arguments, "latents"):
        problems.append("source_video stands in for 'latents' - pass one or the other")
    if source and chained:
        what = "an audio pass scores" if audio_pass else "a regional edit re-denoises"
        problems.append(
            f"{what} one source_video, and a chained step renders several "
            f"segments - use an unchained step"
        )
    return problems


def _listen_scale_problems(arguments):
    """A `listen_scale` on a step that is no audio pass: what makes it one is
    said, since a mask or `hold_audio` is the usual reason it is not."""
    if not _given(arguments, LISTEN_SCALE_INPUT):
        return []
    return [
        "listen_scale sets the canvas of an audio pass - a 'source_video' with "
        "no mask_video, mask_image or hold_audio - and this step is not one"
    ]


def _audio_pass_problems(arguments):
    """Why this audio pass's own arguments cannot run: a refine argument (no
    video row is re-denoised) or a `listen_scale` not in LISTEN_SCALES."""
    problems = [
        f"{name} re-denoises video, and an audio pass (a source_video with no "
        f"mask and no hold_audio) keeps every video row - drop it, or pass "
        f"hold_audio to re-denoise the source under its own soundtrack"
        for name in AUDIO_PASS_REFUSES
        if _given(arguments, name)
    ][:1]
    scale = arguments.get(LISTEN_SCALE_INPUT)
    if scale is not None and not _deferred(scale):
        if isinstance(scale, bool) or scale not in LISTEN_SCALES:
            problems.append(
                f"listen_scale is one of {', '.join(map(str, LISTEN_SCALES))} - "
                f"the audio pass's canvas against the source's - not {scale!r}"
            )
    return problems


def listen_canvas(width, height, scale):
    """The (width, height) an audio pass at `scale` runs its kept rows on: the
    shorter side scaled and snapped to the nearest TILE_GRID multiple, ties
    up, the longer derived from the source's aspect and snapped the same way,
    neither below one cell of the grid. Snapping each side on its own would
    squeeze 960x544 at 0.25 to 256x128, 13% off its aspect."""

    def snapped(side):
        return max(TILE_GRID, int(side / TILE_GRID + 0.5) * TILE_GRID)

    short = snapped(min(width, height) * scale)
    long = snapped(short * max(width, height) / min(width, height))
    return (long, short) if width >= height else (short, long)


H3_MAX_FRAMES = 345  # mirrors the templates' num_frames max_frames


def retimed_indices(frame_count, fps):
    """The source frame each conditioning frame of an audio pass shows: the
    source's `frame_count` frames at `fps` resampled in time, nearest frame,
    to H3_FPS over the same duration, the count snapped up onto RENDER_GRID
    as `num_frames` is (the frames past the end repeat the last). The video
    rows then encode a 24 fps picture of the same moments, and the audio
    generated against them covers the source's real duration (R25)."""
    count = aligned(max(1, round(frame_count / fps * H3_FPS)), RENDER_GRID)
    return [min(frame_count - 1, int(i * fps / H3_FPS + 0.5)) for i in range(count)]


def media_problems(name, shape, width, height, num_frames=None, fps=None):
    """Why a `source_video` or mask of `shape` - (frames, height, width) -
    does not fit a `width` x `height` canvas, as a list. `num_frames` is the
    step's frame count, when it is known; a `mask_image` (one frame) fits any.
    `fps` is an audio pass's source rate: at any rate but H3_FPS the source is
    retimed onto the grid (`retimed_indices`), so its own count need not be."""
    problems = []
    if (width, height) != (shape[2], shape[1]):
        problems.append(
            f"{name} is {shape[2]}x{shape[1]}, and the step renders "
            f"{width}x{height} - they must match"
        )
    # An audio pass at another rate conditions on the source retimed to H3_FPS,
    # so it is that count, not the source's own, the step must render
    expected = shape[0]
    retimed = name == SOURCE_VIDEO_INPUT and fps not in (None, H3_FPS)
    if retimed:
        expected = len(retimed_indices(shape[0], fps))
    if num_frames is not None and name != MASK_IMAGE_INPUT and expected != num_frames:
        if retimed:
            have = (
                f"{name} has {shape[0]} frames at {fps:g} fps ({expected} once "
                f"retimed to {H3_FPS} fps)"
            )
        else:
            have = f"{name} has {shape[0]} frames"
        problem = (
            f"{have}, and the step renders num_frames={num_frames} - they must match"
        )
        if name == SOURCE_VIDEO_INPUT and expected > H3_MAX_FRAMES:
            problem += (
                f". MiniMax-H3 renders at most {H3_MAX_FRAMES} frames "
                f"({H3_MAX_FRAMES / H3_FPS:.0f} seconds at {H3_FPS} fps), so this "
                f"clip is too long - trim it to {aligned_down(H3_MAX_FRAMES, RENDER_GRID)} "
                f"frames or fewer"
            )
        problems.append(problem)
    elif (
        num_frames is not None and name == SOURCE_VIDEO_INPUT and fps in (None, H3_FPS)
    ):
        # Off the 17n + 5 grid the pipeline snaps num_frames up and the source
        # is padded to fit, so the mask drifts in time and the output is longer
        # than the source: refused here, where the count can still be changed
        on_grid = aligned(num_frames, RENDER_GRID)
        if on_grid != num_frames:
            below = aligned_down(num_frames, RENDER_GRID)
            problems.append(
                f"{name} has {shape[0]} frames, off the 17n + 5 grid the step "
                f"renders on: the pipeline would render {on_grid} and pad the "
                f"source, drifting the mask - trim the clip to {below} frames or "
                f"pad it to {on_grid}"
            )
    return problems
