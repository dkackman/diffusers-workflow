"""MiniMax-H3 SelfLift (#1195, #1199 Task 19): EXPERIMENTAL.

A spike, not a public surface. `experimental_selflift: {lift_step,
start_scale, upsample?}` on a t2va/fl2va call runs the first `lift_step`
evaluations of the call's own schedule on a canvas `start_scale` the size
(0.5: a quarter of the video rows), lifts the partly denoised latents to the
call's canvas, and finishes the schedule there. Same schedule, same number of
transformer evaluations, no second pass. dw's upscale + refine
(`h3_latent_upscale`, then `refine_strength`) is a different thing: a whole
low-res generation, decoded or carried as latents, then a second schedule of
its own from a refine sigma.

The method is the SelfLift paper's SelfLift-zero (arXiv 2609.02036) for
rectified-flow models; the idea reached dw through leonardofrito/Directors-Cut
(MIT), which wires facok/comfyui-SelfLift (no licence: concepts only) as an
H3 sampler. Nothing here is adapted from either's code - only the concept, as
the issue describes it.

Off unless `DW_EXPERIMENTAL_REFINE_WINDOWS=1` (the long-form spikes' one
flag) is set where the pipeline loads; refused at validate time
(dw/guides.py `selflift_errors`) and at the call (`selflift_refusal`).

Feasibility, read off diffusers' H3 blocks (no diffusers change needed):

- Where. H3's loop is `denoiser` (one forward over the packed sequence) then
  `update` (one Euler step of the generated rows). A loop sub-block in front
  of `denoiser` (h3_selflift_steps.DwH3SelfLiftStep) sees every step: at
  step 0 it swaps the call's full-canvas state for a half-canvas one, and at
  `lift_step` it swaps back with the lifted latents. Everything around the
  loop - layout, noise draw, keyframe and guide encodes, timesteps, decode -
  runs stock at the call's canvas, so what leaves the loop is the call's own
  shape. The scheduler is shape-agnostic (its step index and sigma grid do
  not see the rows), so one schedule spans both canvases.
- The schedule. `num_inference_steps` N gives N - 1 evaluations at sigmas
  shifted towards 1 (`shift` 6 on the bench: 1, 0.947, 0.857, 0.667 at N = 5,
  the 4-step turbo). A lift at evaluation k replaces k of the N - 1 full
  forwards with quarter-row ones. At the 4-step turbo that is k = 2 (half)
  or 1; the 8-step schedule (N = 9) gives k = 4 with two-thirds of the
  sigma range still to run after it, which is where the paper's numbers come
  from. The sigma at the lift is still high (0.86 at k = 2 of 4, shift 6), so
  the full canvas has most of the noise level left to add detail in.
- The space. The loop's latents are the VAE's normalized latents. Area
  down and bilinear/nearest up are linear with weights summing to 1 per
  output cell, so they commute with the per-channel affine normalization:
  resizing in normalized or raw latent space is the same thing.
- The noise, sigma-consistent. H3's velocity is data-ward: x0 = x + sigma v.
  After Euler steps with eta 0, x_k = (1 - sigma_k) x0 + sigma_k eps exactly,
  with x0 = x_k + sigma_k v_{k-1} and eps = x0 - v_{k-1} from the last
  prediction (the update is an x/x0 blend). The lift is
  `(1 - sigma_k) up(x0) + sigma_k E`, where E keeps the trajectory's own
  noise as its low band and takes its high band from the call's own
  full-canvas noise draw: E = nearest(eps) / f + (N - nearest(pool(N))), for
  f x f blocks. Both terms are Gaussian; the first has variance 1/f^2 and is
  constant on a block, the second has variance 1 - 1/f^2 and sums to zero on
  a block, so E is unit-variance i.i.d. noise again and f pool(E) = eps.
  "Unit i.i.d." holds only as far as the model's implied eps (denoised minus
  the last velocity) is itself unit noise: an approximation, close at the
  lift's sigma of about 0.86, not exact.
  The half canvas starts from that same draw, f pool(N) (unit variance), so
  with no denoising a lift at sigma 1 gives N back exactly. No extra random
  draw, so a seed reproduces.
- RoPE and the packed sequence. The spatial rotary axes are aspect-
  normalized (`dim / sqrt(area)`), so a canvas of the same aspect spans the
  same coordinate range, coarser. `shrink_layout` derives the half-canvas
  layout from the full one: every video frame of rows (a generated latent
  frame, a keyframe's frame, a guide's frame) becomes a frame of the half grid
  at the same rotary time, text rows are unchanged, and audio rows keep their
  time and move their width pin to the half grid's edge. The order of rows,
  so the trailing target video dw's sparse attention needs, is kept, and the
  result equals diffusers' own layout built at the half canvas
  (tests/test_h3_selflift.py). The row-timestep plan is mapped the same way,
  so held rows keep whatever timestep their own blocks gave them. At the lift
  the full layout and plan go back in unchanged: the remaining forwards are
  the stock ones' layout exactly.
- Audio rows are canvas-independent: the same `audio_latents` tensor rides
  through both phases, stepped by its own scheduler, never touched by the
  lift.
- Condition rows. A keyframe's or a guide's frames are clean latents
  (noised to t = 0.999 at most). The half phase sees them area-downsampled,
  which approximates encoding the half-size pixels (one VAE pass less, not
  bit-equal to it). At the lift the call's own full-canvas condition rows go
  back. The late capstone's gated rows (row ranges in the full layout) and
  every refine (`latents`, `source_video`, a regional edit, tiles, windows)
  are refused: a refine starts from a take, not from noise, and the regional
  edit pins rows by the full layout's index.

The canvas: the call's patch grid must divide by 1/start_scale, so at 0.5 the
width and height are multiples of 64 (1024x576 is; 960x544 is not, 544 is 17
patches). The half canvas is then a multiple of 32, a canvas H3 lays out.

Pure torch, no diffusers at import: validation reads the rules, and the
loop block (h3_selflift_steps.py) imports diffusers on first use.
"""

import torch
import torch.nn.functional as F

from .. import references
from .h3_hold import core_denoise_sequences
from .h3_rules import (
    MASK_INPUTS,
    REFINE_SIGMAS_INPUT,
    REFINE_STRENGTH_INPUT,
    REFINE_TILES_INPUT,
    REFINE_WINDOWS_ENV_VAR as ENABLE_ENV_VAR,
    REFINE_WINDOWS_INPUT,
    REFINE_BEFORE,
    SELFLIFT_INPUT,
    SOURCE_VIDEO_INPUT,
    refine_windows_enabled,
)

# The loop sub-block's name, in front of the denoiser
SELFLIFT_BLOCK = "dw_selflift"
DENOISER = "denoiser"
KEYS = ("lift_step", "start_scale", "upsample")
# 0.5 only: a quarter of the rows, and the paper's and Directors-Cut's scale
START_SCALES = (0.5,)
UPSAMPLES = ("bilinear", "nearest")
# H3's canvas grid: the VAE's 16 px times the transformer's 2-cell patch
CANVAS_GRID = 32
# Arguments that start the loop from a take rather than from noise, or pin
# rows by the full layout's index
CONFLICTS = (
    "latents",
    "references",
    REFINE_STRENGTH_INPUT,
    REFINE_SIGMAS_INPUT,
    REFINE_TILES_INPUT,
    REFINE_WINDOWS_INPUT,
    SOURCE_VIDEO_INPUT,
    *MASK_INPUTS,
)


# 1. The rules - what the call needs, checked before it runs


def _unresolved(value):
    return isinstance(value, str) and references.is_ref(references.UNRESOLVED, value)


def _whole(value):
    return isinstance(value, int) and not isinstance(value, bool)


def scale_factor(settings):
    """The whole number of full-canvas cells per half-canvas cell, per axis."""
    return round(1 / settings.get("start_scale", START_SCALES[0]))


def selflift_problems(arguments, has_cache=False):
    """Why this call's `experimental_selflift` cannot run - a list, empty
    when it can or cannot be told yet. Pure: the settings and the call's
    literal arguments. Unresolved references are left to the run."""
    settings = arguments.get(SELFLIFT_INPUT)
    if settings is None or _unresolved(settings):
        return []
    name = SELFLIFT_INPUT
    if not isinstance(settings, dict):
        return [f"{name} is {{lift_step, start_scale}}, got {settings!r}"]
    problems = []
    extra = sorted(set(settings) - set(KEYS))
    if extra:
        problems.append(f"{name} takes {', '.join(KEYS)}, not {', '.join(extra)}")
    lift_step = settings.get("lift_step")
    if not _unresolved(lift_step) and (not _whole(lift_step) or lift_step < 1):
        problems.append(
            f"{name} lift_step is the whole number of evaluations run on the "
            f"small canvas, 1 or more, got {lift_step!r}"
        )
        lift_step = None
    scale = settings.get("start_scale", START_SCALES[0])
    if not _unresolved(scale) and scale not in START_SCALES:
        problems.append(
            f"{name} start_scale is {', '.join(map(str, START_SCALES))}, got {scale!r}"
        )
        scale = None
    upsample = settings.get("upsample", UPSAMPLES[0])
    if not _unresolved(upsample) and upsample not in UPSAMPLES:
        problems.append(f"{name} upsample is one of {', '.join(UPSAMPLES)}")
    steps = arguments.get("num_inference_steps")
    if _whole(lift_step) and _whole(steps) and lift_step >= steps - 1:
        problems.append(
            f"{name} lift_step {lift_step} leaves no evaluation on the full "
            f"canvas: num_inference_steps {steps} runs {steps - 1}, so lift_step "
            f"is at most {steps - 2}"
        )
    if scale in START_SCALES:
        grid = round(CANVAS_GRID / scale)
        for axis in ("width", "height"):
            value = arguments.get(axis)
            if _whole(value) and value % grid:
                problems.append(
                    f"{name} at start_scale {scale} needs a {axis} that is a "
                    f"multiple of {grid}, so the small canvas is still on H3's "
                    f"{CANVAS_GRID} px grid - got {value}"
                )
    for other in CONFLICTS:
        if arguments.get(other) is not None:
            problems.append(
                f"{name} lifts a generation from noise, and {other} starts it "
                f"from a take or pins rows by the full canvas's layout - drop "
                f"{other}"
            )
    if arguments.get("guide_noise") is not None:
        problems.append(
            f"{name} does not combine with guide_noise: pooling a noised "
            f"guide row breaks its noise level - drop guide_noise"
        )
    if has_cache:
        problems.append(
            f"{name} changes the sequence mid schedule, and a step cache would "
            f"reuse a small-canvas result on the full canvas - drop 'cache'"
        )
    return problems


def selflift_refusal(pipeline, arguments, has_cache=False):
    """The first reason this call cannot lift, or None."""
    if arguments.get(SELFLIFT_INPUT) is None:
        return None
    if not refine_windows_enabled():
        return (
            f"{SELFLIFT_INPUT} is experimental and off in this process - start "
            f"the server with {ENABLE_ENV_VAR}=1"
        )
    if not lifts(pipeline):
        return (
            f"{SELFLIFT_INPUT} is a MiniMax-H3 t2va/fl2va argument, and "
            f"{type(pipeline).__name__} has no loop that lifts"
        )
    problems = selflift_problems(arguments, has_cache)
    return problems[0] if problems else None


# 2. The latents - packing, the half canvas and the lift


def patchify(latents, patch):
    """(1, C, F, H, W) latents as packed rows, frame-major then row-major:
    diffusers' `patchify_video_latents`, for a batch of one."""
    pt, ph, pw = patch
    _, channels, frames, height, width = latents.shape
    rows = latents.reshape(
        1, channels, frames // pt, pt, height // ph, ph, width // pw, pw
    )
    rows = rows.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return rows.reshape(-1, channels * pt * ph * pw).contiguous()


def unpatchify(rows, frames, height, width, patch):
    """`patchify`'s inverse: packed rows as (1, C, frames, height, width)."""
    pt, ph, pw = patch
    channels = rows.shape[1] // (pt * ph * pw)
    latents = rows.reshape(
        frames // pt, height // ph, width // pw, channels, pt, ph, pw
    )
    latents = latents.permute(3, 0, 4, 1, 5, 2, 6)
    return latents.reshape(1, channels, frames, height, width)


def _per_frame(latents, operation):
    """Apply a 2-d `operation` to every frame of (1, C, F, H, W)."""
    _, channels, frames, height, width = latents.shape
    flat = latents.permute(0, 2, 1, 3, 4).reshape(frames, channels, height, width)
    out = operation(flat)
    return out.reshape(1, frames, channels, *out.shape[-2:]).permute(0, 2, 1, 3, 4)


def shrink_clean(latents, factor):
    """Clean latents on the small canvas: the mean of each f x f block."""
    return _per_frame(latents, lambda x: F.avg_pool2d(x, factor))


def shrink_noise(noise, factor):
    """Unit noise on the small canvas: f x the mean of each f x f block, so
    i.i.d. unit-variance noise stays i.i.d. unit-variance."""
    return shrink_clean(noise, factor) * factor


def _up_nearest(latents, factor):
    return _per_frame(
        latents, lambda x: F.interpolate(x, scale_factor=factor, mode="nearest")
    )


def detail_noise(noise, factor):
    """The full-canvas noise less its block means: the high band the small
    canvas never saw, variance 1 - 1/f^2, summing to zero on every block."""
    return noise - _up_nearest(shrink_clean(noise, factor), factor)


def upsample(latents, factor, mode="bilinear"):
    """Small-canvas latents at the full canvas, per frame."""
    if mode == "nearest":
        return _up_nearest(latents, factor)
    return _per_frame(
        latents,
        lambda x: F.interpolate(
            x, scale_factor=factor, mode="bilinear", align_corners=False
        ),
    )


def lift(latents, velocity, sigma, noise, factor, mode="bilinear"):
    """The small canvas's latents at `sigma`, lifted to the full canvas at
    the same sigma: `(1 - sigma) up(x0) + sigma E`. `velocity` is the last
    (data-ward) prediction, `noise` the call's full-canvas draw. All
    (1, C, F, h, w) small or (1, C, F, H, W) full; float32.

    With `mode` bilinear, pool(up(x0)) is not x0 (the interpolation mixes
    neighbouring blocks), so the low band shifts slightly; nearest is exact.
    E is unit i.i.d. only as far as the model's implied eps is unit noise:
    an approximation, close at sigma about 0.86."""
    latents, velocity = latents.float(), velocity.float()
    denoised = latents + sigma * velocity
    trajectory_noise = denoised - velocity
    lifted_noise = _up_nearest(trajectory_noise, factor) / factor + detail_noise(
        noise.float(), factor
    )
    return (1.0 - sigma) * upsample(denoised, factor, mode) + sigma * lifted_noise


# 3. The layout - the full canvas's packed sequence on the small canvas


def shrink_layout(fields, plan, rows_per_frame, full_grid, half_grid):
    """The packed layout of the small canvas, derived from the full one.

    `fields` holds `position_ids` (S, 3), `token_tags` and the three index
    tensors; `plan` is the per-step `(timesteps, timestep_indices)` list;
    `full_grid` and `half_grid` are one frame's (h, w) rotary coordinates on
    each canvas, in row order. Every run of `rows_per_frame` video rows -
    which must sit together in the sequence on the full grid - becomes one
    half-grid frame at the same time, tag and timestep; text rows are kept;
    audio rows keep their time and move a width pin on the full grid's edge to
    the half grid's. Returns `(fields, plan)`, other `fields` entries passed
    through.

    Raises:
        ValueError: If the video rows are not whole frames of the full grid
    """
    positions = fields["position_ids"]
    tags = fields["token_tags"]
    video = fields["video_indices"]
    audio = fields["audio_indices"]
    text = fields["text_indices"]
    device = positions.device
    full_grid = full_grid.to(device, positions.dtype)
    half_grid = half_grid.to(device, positions.dtype)
    half_rows = half_grid.shape[0]
    if video.numel() % rows_per_frame:
        raise ValueError(
            f"{SELFLIFT_INPUT}: {video.numel()} video rows are not whole frames "
            f"of {rows_per_frame} rows"
        )
    frames = video.reshape(-1, rows_per_frame)
    if not bool(
        (frames == frames[:, :1] + torch.arange(rows_per_frame, device=device)).all()
    ):
        raise ValueError(
            f"{SELFLIFT_INPUT}: a frame's video rows do not sit together in the "
            f"sequence"
        )
    if not torch.equal(
        positions[frames][:, :, 1:], full_grid.expand(len(frames), -1, -1)
    ):
        raise ValueError(
            f"{SELFLIFT_INPUT}: a video frame is not on the call's canvas grid "
            f"(a reference or a guide at another size) - it cannot be shrunk"
        )
    length = positions.shape[0]
    counts = torch.ones(length, dtype=torch.long, device=device)
    counts[video] = 0
    counts[frames[:, 0]] = half_rows
    starts = counts.cumsum(0) - counts
    new_length = int(counts.sum())
    singles = torch.cat([text, audio])

    new_positions = positions.new_zeros((new_length, 3))
    new_positions[starts[singles]] = positions[singles]
    width = positions[audio, 2]
    moved = new_positions[starts[audio]]
    moved[width == full_grid[:, 1].min(), 2] = half_grid[:, 1].min()
    moved[width == full_grid[:, 1].max(), 2] = half_grid[:, 1].max()
    new_positions[starts[audio]] = moved
    new_video = (
        starts[frames[:, 0]][:, None] + torch.arange(half_rows, device=device)
    ).reshape(-1)
    new_positions[new_video, 0] = positions[frames[:, 0], 0].repeat_interleave(
        half_rows
    )
    new_positions[new_video, 1:] = half_grid.repeat(len(frames), 1)

    def per_row(values):
        out = values.new_empty(new_length)
        out[starts[singles]] = values[singles]
        out[new_video] = values[frames[:, 0]].repeat_interleave(half_rows)
        return out

    new_fields = dict(fields)
    new_fields.update(
        position_ids=new_positions,
        token_tags=per_row(tags),
        video_indices=new_video,
        audio_indices=starts[audio],
        text_indices=starts[text],
    )
    new_plan = []
    for timesteps, indices in plan:
        if not bool((indices[frames] == indices[frames[:, :1]]).all()):
            raise ValueError(
                f"{SELFLIFT_INPUT}: a video frame's rows run at different timesteps"
            )
        new_plan.append((timesteps, per_row(indices)))
    return new_fields, new_plan


# 4. The graph change


def _loops(pipeline):
    """Every t2va/fl2va denoise loop (ref2va's runs the reference partition)."""
    for prefix, sequence in core_denoise_sequences(pipeline):
        loop = sequence.sub_blocks.get(prefix + REFINE_BEFORE)
        if not hasattr(loop, "sub_blocks") or DENOISER not in loop.sub_blocks:
            continue
        if (
            getattr(loop.sub_blocks[DENOISER], "transformer_name", None)
            != "transformer"
        ):
            continue
        yield loop


def insert_selflift(pipeline):
    """Put the SelfLift sub-block in front of the denoiser of every t2va/fl2va
    loop. Idempotent. Returns whether the pipeline now lifts."""
    from .h3_selflift_steps import DwH3SelfLiftStep

    inserted = False
    for loop in _loops(pipeline):
        if SELFLIFT_BLOCK not in loop.sub_blocks:
            names = list(loop.sub_blocks)
            loop.sub_blocks.insert(
                SELFLIFT_BLOCK, DwH3SelfLiftStep(), names.index(DENOISER)
            )
        inserted = True
    return inserted


def lifts(pipeline):
    """Whether a pipeline has a loop carrying the SelfLift sub-block."""
    return any(SELFLIFT_BLOCK in loop.sub_blocks for loop in _loops(pipeline))


def saved_fraction(lift_step, evaluations, scale=START_SCALES[0]):
    """The share of the transformer's video-row work a lift saves, counting
    a small-canvas forward as `scale^2` of a full one (the linear layers;
    the attention saves more, the text and audio rows less)."""
    return lift_step * (1 - scale**2) / evaluations


__all__ = [
    "ENABLE_ENV_VAR",
    "SELFLIFT_BLOCK",
    "SELFLIFT_INPUT",
    "insert_selflift",
    "lift",
    "lifts",
    "selflift_problems",
    "selflift_refusal",
    "shrink_layout",
]
