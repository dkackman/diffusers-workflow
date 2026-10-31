"""MiniMax-H3's pure rules: the argument and block names dw's H3 blocks use,
the guide grid, and the checks validation and the run both ask before anything
loads - the rule half of the H3 adapter.

Torch- and diffusers-free, so validation reads these without the block factories:
the hold, refine and release blocks are dw/pipeline_processors/h3_hold.py, the
guide layout dw/pipeline_processors/h3_guides.py. tests/test_h3_rules.py pins
that it imports neither.
"""

from .. import references
from ..adapter_compatibility import FROM_PRETRAINED_KEY, H3_WORKFLOWS, WORKFLOW_KEY
from ..variable_constraints import aligned_down

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
    if arguments.get(HOLD_AUDIO_INPUT) is None:
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
    if len(sigmas) < 2:
        problems.append(
            f"refine_sigmas needs at least 2 points, refine_strength and 0, and "
            f"has {len(sigmas)}"
        )
    elif any(later >= earlier for earlier, later in zip(sigmas, sigmas[1:])):
        problems.append(
            f"refine_sigmas must be strictly decreasing, and {list(sigmas)} is not"
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


def not_h3(pipeline, argument=HOLD_AUDIO_INPUT):
    """Why this step's pipeline cannot take `argument` - one of the H3 block
    arguments - or None when it can or cannot be told before the load."""
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
        f"{', '.join(sorted(H3_WORKFLOWS))} workflows"
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
# Where a guide chain runs is where `guides` runs (dw/guides.py); the run and
# validate lead their refusal with this rule
GUIDE_CHAIN_RULE = (
    "continuity 'guide' runs on MiniMax-H3 t2va or fl2va only - guides stay off ref2va"
)

# Every chain continuity mode, in the order chain.py registers its classes
# (`CONTINUITY_MODES`, zipped strictly against these). Kept here so validation
# (dw/guides.py) reads the same names without importing chain.py
CHAIN_CONTINUITY_MODES = ("last_frame", "last_segment", GUIDE_CONTINUITY)


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
REGIONAL_EDIT_INPUTS = (SOURCE_VIDEO_INPUT, *MASK_INPUTS, MASK_GROW_INPUT)
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
    if not masks and not source and grow is None:
        return []
    problems = []
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
    if source and arguments.get(REFINE_STRENGTH_INPUT) is None:
        problems.append(
            "source_video is re-denoised from 'refine_strength' - pass one, near "
            "1.0 to replace what the mask covers, lower to keep its layout"
        )
    if source and _given(arguments, "latents"):
        problems.append("source_video stands in for 'latents' - pass one or the other")
    if source and chained:
        problems.append(
            "a regional edit re-denoises one source_video, and a chained step "
            "renders several segments - use an unchained step"
        )
    return problems


def media_problems(name, shape, width, height, num_frames=None):
    """Why a `source_video` or mask of `shape` - (frames, height, width) -
    does not fit a `width` x `height` canvas, as a list. `num_frames` is the
    step's frame count, when it is known; a `mask_image` (one frame) fits any."""
    problems = []
    if (width, height) != (shape[2], shape[1]):
        problems.append(
            f"{name} is {shape[2]}x{shape[1]}, and the step renders "
            f"{width}x{height} - they must match"
        )
    if num_frames is not None and name != MASK_IMAGE_INPUT and shape[0] != num_frames:
        problems.append(
            f"{name} has {shape[0]} frames, and the step renders num_frames="
            f"{num_frames} - they must match"
        )
    return problems
