"""MiniMax-H3 audio hold and refine: the three blocks dw puts into the denoise
graph, and the arithmetic they run.

What the blocks do, and where they go, is dw/pipeline_processors/h3_hold.py,
which puts them in. They live here, at module level, rather than in a factory
inside h3_hold (#790): a class defined in a function body is counted into that
function, and the factory read as one 261-line function to the ratchet. This
module imports diffusers when it loads, so nothing imports it at module level -
`h3_hold.blocks()` imports it on first use, and validation, which imports
h3_hold and h3_guides, still never loads a modular pipeline.

The rows and sigmas are here too, not in h3_hold, because the blocks call them
and h3_hold imports this module: a helper left behind would make the import a
cycle. The guide blocks (dw/pipeline_processors/h3_guide_steps.py) encode a
guide's audio through `encode_audio_span` here, which this module never
imports back.
"""

import logging

import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3SetTimestepsStep,
)
from diffusers.modular_pipelines.minimax_h3.before_encoder import (
    MiniMaxH3Ref2VASetupStep,
)
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3Ref2VAReferenceEncoderStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
)
from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3AudioReference
from diffusers.modular_pipelines.modular_pipeline import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam
from diffusers.utils.torch_utils import randn_tensor

from .h3_rules import (
    HELD_AUDIO_OUTPUT,
    HELD_AUDIO_RATE_OUTPUT,
    HELD_ROWS,
    HOLD_AUDIO_INPUT,
    REFINE_STRENGTH_INPUT,
)

logger = logging.getLogger("dw")

# Encoded silence pads a track shorter than the video: this many latents of zeros
# are encoded and the middle one taken, clear of any edge effect of the VAE's
SILENCE_LATENTS = 4


def refine_sigmas(strength, num_points, shift):
    """The refine schedule: σ₀ = `strength`, then `num_points` points in all down
    to 0, linear in the unshifted coordinate u and mapped through the scheduler's
    shift σ = shift·u / (1 + (shift−1)·u) - the spacing `set_timesteps` gives a
    full grid, which `set_timesteps(sigmas=...)` does not apply itself.

    Raises:
        ValueError: If `strength` is not in (0, 1) or `num_points` is below 2
    """
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise ValueError(f"refine_strength must be a number, got {strength!r}")
    if not 0 < strength < 1:
        raise ValueError(f"refine_strength must be in (0, 1), got {strength}")
    if num_points is None or num_points < 2:
        raise ValueError(
            f"refine_strength needs num_inference_steps of 2 or more, got {num_points}"
        )
    shift = float(shift)
    start = strength / (shift - (shift - 1) * strength)
    sigmas = shifted_sigma_grid(num_points, shift, start)
    sigmas[0], sigmas[-1] = strength, 0.0
    return sigmas.float()


def shifted_sigma_grid(num_points, shift, start=1.0):
    """`num_points` sigmas, float64, linear in the unshifted coordinate u from
    `start` down to 0 and mapped through σ = shift·u / (1 + (shift−1)·u). At
    `start` 1 this is the grid stock `set_timesteps(num_inference_steps)` builds -
    the copy `refine_sigmas` rests on, pinned against it by
    tests/test_h3_refine.py."""
    shift = float(shift)
    u = torch.linspace(start, 0.0, int(num_points), dtype=torch.float64)
    return shift * u / (1 + (shift - 1) * u)


def as_channels_samples(waveform):
    """A waveform as a float32 `(channels, samples)` tensor - a bare mono
    `(samples,)` one gains its channel axis."""
    waveform = torch.as_tensor(waveform).float()
    if waveform.ndim == 1:
        waveform = waveform[None]
    return waveform


def fit_samples(waveform, num_samples):
    """Crop a `(channels, samples)` waveform to `num_samples`, or pad it with
    silence to that length."""
    missing = num_samples - waveform.shape[-1]
    if missing <= 0:
        return waveform[..., :num_samples]
    return torch.nn.functional.pad(waveform, (0, missing))


def fit_latents(latents, num_latents, silence):
    """Crop `(channels, n, C)` audio latents to `num_latents` per channel, or pad
    them with `silence` - one `(channels, 1, C)` latent of encoded silence,
    repeated."""
    missing = num_latents - latents.shape[1]
    if missing <= 0:
        return latents[:, :num_latents]
    padding = silence.to(latents.dtype).expand(-1, missing, -1)
    return torch.cat([latents, padding], dim=1)


def encode_audio_latents(encoder, components, waveform):
    """`(channels, n, C)` normalized audio latents of a VAE-rate stereo waveform,
    through `encoder` - a MiniMaxH3Ref2VAReferenceEncoderStep, whose soundtrack
    encode takes the posterior mode."""
    sub_state = PipelineState()
    sub_state.set(
        "normalized_references",
        [
            MiniMaxH3AudioReference(
                audio=waveform, sample_rate=components.audio_sampling_rate
            )
        ],
    )
    _, sub_state = encoder(components, sub_state)
    rows = sub_state.get("audio_condition_latents")[0]
    return rows.reshape(components.audio_channels, -1, components.audio_latent_channels)


def encode_audio_span(encoder, components, waveform, num_latents):
    """`(channels, num_latents, C)` latents of a VAE-rate stereo waveform: cut or
    padded with silence to `num_latents` latents of samples, encoded, and any
    latent the encode falls short by filled with encoded silence."""
    samples_per_latent = (
        components.audio_sampling_rate // MINIMAX_H3_AUDIO_LATENTS_PER_SECOND
    )
    waveform = fit_samples(waveform, num_latents * samples_per_latent)
    latents = encode_audio_latents(encoder, components, waveform)
    if latents.shape[1] >= num_latents:
        return fit_latents(latents, num_latents, None)
    silence = encode_audio_latents(
        encoder, components, torch.zeros(2, SILENCE_LATENTS * samples_per_latent)
    )
    middle = silence.shape[1] // 2
    return fit_latents(latents, num_latents, silence[:, middle : middle + 1])


class DwH3HoldAudioStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    def __init__(self):
        super().__init__()
        # Called, not copied: the encode a Ref2VA soundtrack reference takes
        self._encoder = MiniMaxH3Ref2VAReferenceEncoderStep()

    @property
    def description(self):
        return (
            "dw: holds `hold_audio` as the target soundtrack - encodes it like a Ref2VA soundtrack reference, "
            "writes it over the target audio rows and widens `num_condition_audio_rows` over them, so the "
            "denoiser treats it as clean conditioning. Does nothing without `hold_audio`."
        )

    @property
    def inputs(self):
        return [
            InputParam(
                name=HOLD_AUDIO_INPUT,
                type_hint=MiniMaxH3AudioReference,
                default=None,
                description="The soundtrack to generate the video to, as a MiniMaxH3AudioReference.",
            ),
            InputParam(name="audio_latents", type_hint=torch.Tensor, required=True),
            InputParam(name="num_audio_latents", type_hint=int, required=True),
            InputParam(name="num_condition_audio_rows", type_hint=int, default=0),
            InputParam(name="num_frames", type_hint=int, required=True),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("audio_latents", type_hint=torch.Tensor),
            OutputParam("num_condition_audio_rows", type_hint=int),
            OutputParam(HELD_ROWS, type_hint=int),
            OutputParam(HELD_AUDIO_OUTPUT, type_hint=torch.Tensor),
            OutputParam(HELD_AUDIO_RATE_OUTPUT, type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        block_state.dw_held_audio_rows = 0
        block_state.dw_held_audio = None
        block_state.dw_held_audio_sampling_rate = None
        held = block_state.hold_audio
        if held is None:
            self.set_block_state(state, block_state)
            return components, state

        vae_rate = components.audio_sampling_rate
        duration = block_state.num_frames / components.fps
        num_latents = block_state.num_audio_latents
        original = as_channels_samples(held.audio)
        rate = held.sample_rate or vae_rate

        # Onto the VAE's rate and channels the way a Ref2VA reference is, then
        # padded with silence to the whole target - a short track encodes
        # with its silence rather than having it appended in latent space
        waveform = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
            original, rate, vae_rate, max_duration=duration
        )
        latents = encode_audio_span(self._encoder, components, waveform, num_latents)

        # Channel-major target rows, after any Ref2VA reference rows
        audio_latents = block_state.audio_latents
        start = block_state.num_condition_audio_rows or 0
        rows = latents.reshape(-1, components.audio_latent_channels)
        audio_latents[start : start + rows.shape[0]] = rows.to(
            audio_latents.device, audio_latents.dtype
        )
        block_state.audio_latents = audio_latents
        block_state.num_condition_audio_rows = start + rows.shape[0]
        block_state.dw_held_audio_rows = rows.shape[0]

        # What the step's result plays: the caller's own track, not the VAE
        # round trip of it, fitted to the video at the rate it came at
        fitted = fit_samples(original, int(round(duration * rate)))
        block_state.dw_held_audio = fitted[None]
        block_state.dw_held_audio_sampling_rate = int(rate)

        logger.info(
            f"Holding {original.shape[-1] / rate:.2f}s of audio as the "
            f"{duration:.2f}s target soundtrack ({rows.shape[0]} rows)"
        )
        self.set_block_state(state, block_state)
        return components, state


class DwH3RefineScheduleStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self):
        return (
            "dw: with `refine_strength`, re-denoises the passed `latents` from that sigma - replaces both "
            "schedules with `num_inference_steps` shift-spaced points from it down to 0 and re-noises the "
            "generated video rows to it. Needs every audio row held (`hold_audio`). Does nothing without "
            "`refine_strength`."
        )

    @property
    def inputs(self):
        return [
            InputParam(
                name=REFINE_STRENGTH_INPUT,
                type_hint=float,
                default=None,
                description=(
                    "The sigma in (0, 1) to re-denoise a passed `latents` from - an upscaled take refined at "
                    "about 0.2. Runs num_inference_steps - 1 evaluations."
                ),
            ),
            InputParam(name="num_inference_steps", type_hint=int, default=50),
            InputParam.template("generator"),
            InputParam(name="latents", type_hint=torch.Tensor, required=True),
            InputParam(name="audio_latents", type_hint=torch.Tensor, required=True),
            InputParam(name="video_indices", type_hint=torch.Tensor, required=True),
            InputParam(name="audio_indices", type_hint=torch.Tensor, required=True),
            InputParam(name="text_indices", type_hint=torch.Tensor, required=True),
            InputParam(name="num_condition_video_rows", type_hint=int, default=0),
            InputParam(name="num_condition_audio_rows", type_hint=int, default=0),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("audio_timesteps", type_hint=torch.Tensor),
            OutputParam("row_timestep_plan", type_hint=list),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        strength = block_state.refine_strength
        if strength is None:
            # Leaves the state as set_timesteps left it: the timesteps are
            # outputs here, not inputs, so set_block_state would refuse them
            return components, state

        audio_rows = block_state.audio_latents.shape[0]
        held = block_state.num_condition_audio_rows or 0
        if held != audio_rows:
            raise ValueError(
                f"refine_strength re-denoises the video only, and {audio_rows - held} of "
                f"{audio_rows} audio rows are not held - pass hold_audio with it"
            )

        device = components._execution_device
        sigmas = refine_sigmas(
            strength, block_state.num_inference_steps, components.scheduler.shift
        )
        # Verbatim: set_timesteps(sigmas=...) applies no shift. The audio
        # schedule only has to match the video's for the loop's audio step,
        # which steps no row when every one is held
        components.scheduler.set_timesteps(sigmas=sigmas, device=device)
        components.audio_scheduler.set_timesteps(sigmas=sigmas, device=device)
        block_state.timesteps = components.scheduler.timesteps
        block_state.audio_timesteps = components.audio_scheduler.timesteps
        block_state.row_timestep_plan = [
            tuple(
                tensor.to(device)
                for tensor in MiniMaxH3SetTimestepsStep.build_row_timesteps(
                    block_state.video_indices,
                    block_state.audio_indices,
                    block_state.num_condition_video_rows,
                    block_state.num_condition_audio_rows,
                    block_state.text_indices.numel(),
                    float(timestep),
                    float(audio_timestep),
                    max(float(timestep), components.keyframe_noise_aug),
                    1.0,
                )
            )
            for timestep, audio_timestep in zip(
                block_state.timesteps, block_state.audio_timesteps
            )
        ]

        # Only the generated rows: FL2VA and Ref2VA conditioning rows lead
        latents = block_state.latents
        start = block_state.num_condition_video_rows or 0
        clean = latents[start:]
        noise = randn_tensor(
            clean.shape,
            generator=block_state.generator,
            device=clean.device,
            dtype=clean.dtype,
        )
        # The scheduler's own forward process, in H3's t = 1 - σ convention
        block_state.latents = torch.cat(
            [
                latents[:start],
                components.scheduler.scale_noise(clean, 1 - strength, noise),
            ]
        )

        logger.info(
            f"Refining from sigma {strength} over {len(block_state.timesteps)} steps"
        )
        self.set_block_state(state, block_state)
        return components, state


class DwH3ReleaseAudioStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self):
        return (
            "dw: restores `num_condition_audio_rows` to what it was before the hold, so the held rows decode "
            "as the generated soundtrack."
        )

    @property
    def inputs(self):
        return [
            InputParam(name="num_condition_audio_rows", type_hint=int, default=0),
            InputParam(name=HELD_ROWS, type_hint=int, default=0),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("num_condition_audio_rows", type_hint=int),
            OutputParam(HELD_ROWS, type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        held = block_state.dw_held_audio_rows or 0
        if held:
            block_state.num_condition_audio_rows -= held
            block_state.dw_held_audio_rows = 0
        self.set_block_state(state, block_state)
        return components, state
