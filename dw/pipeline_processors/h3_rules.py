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


def refine_problems(arguments):
    """Why this step's `refine_strength` cannot run, as a list - empty when it
    can, or cannot be told yet. The one owner of the per-argument rules: the
    static check (dw/hold_audio.py) and the run-time check before the call
    (`Pipeline._check_refine`) both ask it."""
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
    return problems


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
