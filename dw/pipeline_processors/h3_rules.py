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

# The refine call argument
REFINE_STRENGTH_INPUT = "refine_strength"

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
    """Why this step's `refine_strength` and `refine_tiles` cannot run, as a
    list - empty when they can, or cannot be told yet. The one owner of the
    per-argument rules: the static check (dw/hold_audio.py) and the run-time
    check before the call (`Pipeline._check_refine`) both ask it. A tile
    problem's message starts with `refine_tiles`. `canvas` is the (width,
    height) the static check read from the upscale step `latents` names, when
    it could; at run time the `latents` tensor gives it."""
    if arguments.get(REFINE_STRENGTH_INPUT) is None:
        return [
            "refine_tiles tiles a refine_strength pass, and this step passes no "
            "refine_strength - add one, about 0.2"
        ]
    problems = []
    strength = arguments[REFINE_STRENGTH_INPUT]
    if isinstance(strength, str) and references.is_ref(references.UNRESOLVED, strength):
        pass
    elif isinstance(strength, bool) or not isinstance(strength, (int, float)):
        problems.append(
            f"refine_strength is a number in (0, 1), and {strength!r} is not a number"
        )
    elif not 0 < strength < 1:
        problems.append(
            f"refine_strength is a sigma in (0, 1) - about 0.2 refines an "
            f"upscaled take - and {strength} is outside it"
        )
    if arguments.get("latents") is None:
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
    if arguments.get(REFINE_TILES_INPUT) is not None:
        problems.extend(_tile_problems(arguments, canvas))
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
TILE_CONFLICTS = ("image", "last_image", "references", "guides")


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
