"""MiniMax-H3 audio hold and refine: generating a video to a soundtrack the caller
already has, and re-denoising an upscaled latent from a low sigma.

The one place dw modifies a diffusers modular pipeline's block graph. H3 denoises
video and audio rows in one packed sequence, and the rows it treats as conditioning
are the leading `num_condition_audio_rows` of the audio stream: pinned clean at
`t = 1.0`, never stepped by the scheduler, attended to by every generated row. That
is how a Ref2VA soundtrack reference rides along. Holding a track is the same
mechanism pointed at the target rows instead - the caller's waveform is encoded
through diffusers' own Ref2VA reference encode, written over the target audio rows,
and the count is widened to cover them, so the denoiser generates video that fits a
soundtrack it is not allowed to change.

Three blocks go into each of H3's three core-denoise sequences (`t2va`, `fl2va`,
`ref2va`), always, as no-ops when their argument is not passed:

- `DwH3HoldAudioStep`, before `set_timesteps` - after every `prepare_latents*`
  step, so Ref2VA's reference rows are already in front and its reference-row
  count check has already run against the count it expects;
- `DwH3RefineScheduleStep`, before `denoise` - after `set_timesteps`, whose
  schedule it replaces when `refine_strength` is passed: σ₀ = `refine_strength`,
  then `num_inference_steps` points down to 0 spaced by the scheduler's shift
  (`refine_sigmas`), so `num_inference_steps - 1` evaluations. It re-noises the
  generated video rows of a passed `latents` to σ₀. Every audio row has to be a
  held one (`hold_audio`): the audio scheduler gets the same list only so the
  loop's audio step finds its timesteps, and steps no row;
- `DwH3ReleaseAudioStep`, before `after_denoise` - which slices
  `audio_latents[num_condition_audio_rows:]` as the generated track, so the count
  has to be back to what it was for the held rows to be decoded as the target.

The decoded `audio` is still the VAE round trip of the held rows. The step's result
uses `dw_held_audio` instead - the caller's own waveform fitted to the video's
duration (dw/output_extraction.py), which `HELD_AUDIO_OUTPUT` names for the call
(dw/pipeline_processors/pipeline.py).

Guides (`guides=[{video, frame}]`) hold clips of an existing video as video
condition rows. Unlike the three blocks above they are added only to a pipeline
built to take them (`insert_guides`): `DwH3GuideLayoutStep` wraps the stock layout
step, runs it, then VAE-encodes each clip on the target canvas and splices its rows
after any keyframe rows, timed from the target frame it lands on, widening
`num_condition_video_rows` so `after_denoise` slices them off. On `t2va` two more
blocks run the stock `fl2va` condition-latent steps only when there are guides.

A guide with `audio` also holds its soundtrack over its span (#649). The span starts
at the audio latent at or before the guide's first frame, `floor(frame * 40 / 24)`,
and runs `round(frames * 40 / 24)` latents on from the frame's own time; the
waveform is pre-padded with the silence between that latent and the frame, so the
time origin holds (`guide_audio_span`, `pad_guide_audio`). It is encoded the way a
Ref2VA soundtrack is (the posterior mode), and the layout splices its rows in
front of the target audio, at the target's own rotary times for that span, as
condition audio rows held at `t = 1.0`. `DwH3GuideAudioStep` puts the encoded rows
in front of `audio_latents` before the audio hold, which then writes after them.
Nothing restores them before decode: `after_denoise` slices the condition prefix
off, and the target rows at the same times are generated, as with a video guide.
"""

import logging

import torch

from .. import references
from ..variable_constraints import aligned_down

logger = logging.getLogger("dw")

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
    u = torch.linspace(start, 0.0, int(num_points), dtype=torch.float64)
    sigmas = shift * u / (1 + (shift - 1) * u)
    sigmas[0], sigmas[-1] = strength, 0.0
    return sigmas.float()


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


def _diffusers():
    """The diffusers pieces the blocks are built from - imported on first use, so
    importing this module does not import a modular pipeline."""
    from diffusers.modular_pipelines.minimax_h3 import before_encoder, encoders
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    )
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
    )

    return (
        before_encoder.MiniMaxH3Ref2VASetupStep,
        encoders.MiniMaxH3Ref2VAReferenceEncoderStep,
        MiniMaxH3AudioReference,
        MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    )


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
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    audio_reference = _diffusers()[2]
    sub_state = PipelineState()
    sub_state.set(
        "normalized_references",
        [audio_reference(audio=waveform, sample_rate=components.audio_sampling_rate)],
    )
    _, sub_state = encoder(components, sub_state)
    rows = sub_state.get("audio_condition_latents")[0]
    return rows.reshape(components.audio_channels, -1, components.audio_latent_channels)


def encode_audio_span(encoder, components, waveform, num_latents):
    """`(channels, num_latents, C)` latents of a VAE-rate stereo waveform: cut or
    padded with silence to `num_latents` latents of samples, encoded, and any
    latent the encode falls short by filled with encoded silence."""
    samples_per_latent = components.audio_sampling_rate // _diffusers()[3]
    waveform = fit_samples(waveform, num_latents * samples_per_latent)
    latents = encode_audio_latents(encoder, components, waveform)
    if latents.shape[1] >= num_latents:
        return fit_latents(latents, num_latents, None)
    silence = encode_audio_latents(
        encoder, components, torch.zeros(2, SILENCE_LATENTS * samples_per_latent)
    )
    middle = silence.shape[1] // 2
    return fit_latents(latents, num_latents, silence[:, middle : middle + 1])


def _make_blocks():
    """The three block classes, defined against the installed diffusers."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3SetTimestepsStep,
    )
    from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
    from diffusers.modular_pipelines.modular_pipeline_utils import (
        InputParam,
        OutputParam,
    )
    from diffusers.utils.torch_utils import randn_tensor

    (
        setup_step,
        reference_encoder,
        audio_reference,
        latents_per_second,
    ) = _diffusers()

    class DwH3HoldAudioStep(ModularPipelineBlocks):
        model_name = "minimax-h3"

        def __init__(self):
            super().__init__()
            # Called, not copied: the encode a Ref2VA soundtrack reference takes
            self._encoder = reference_encoder()

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
                    type_hint=audio_reference,
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
            waveform = setup_step._normalize_audio_condition(
                original, rate, vae_rate, max_duration=duration
            )
            latents = encode_audio_span(
                self._encoder, components, waveform, num_latents
            )

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

    return DwH3HoldAudioStep, DwH3RefineScheduleStep, DwH3ReleaseAudioStep


_BLOCKS = None


def blocks():
    """(DwH3HoldAudioStep, DwH3RefineScheduleStep, DwH3ReleaseAudioStep), built
    once."""
    global _BLOCKS
    if _BLOCKS is None:
        _BLOCKS = _make_blocks()
    return _BLOCKS


def core_denoise_sequences(pipeline):
    """[(prefix, sequence)] for every H3 core-denoise sequence a pipeline's own
    block graph holds - empty for anything that is not a MiniMax-H3 modular
    pipeline.

    Two shapes. Loaded whole, `denoise` is an auto step choosing between the
    three sequences, each holding `set_timesteps`. Loaded with a
    `from_pretrained` `workflow` (what the catalog does), diffusers prunes the
    graph to that workflow's execution blocks, flattened into the top level
    under dotted names - `denoise.set_timesteps` - so the top level is the one
    sequence and `denoise.` the prefix its block names carry.

    `_blocks`, not `blocks`: the property hands back a deep copy, and an edit to
    that copy never reaches the graph the pipeline runs.
    """
    try:
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline
    except ImportError:
        return []
    if not isinstance(pipeline, MiniMaxH3ModularPipeline):
        return []
    top_blocks = getattr(pipeline, "_blocks", None)
    top = getattr(top_blocks, "sub_blocks", None) or {}
    flat_prefix = f"{DENOISE_STEP}."
    if f"{flat_prefix}{HOLD_BEFORE}" in top:
        return [(flat_prefix, top_blocks)]
    denoise = getattr(top.get(DENOISE_STEP), "sub_blocks", None) or {}
    return [
        ("", denoise[name])
        for name in CORE_DENOISE_SEQUENCES
        if name in denoise and hasattr(denoise[name], "sub_blocks")
    ]


def insert_audio_hold(pipeline):
    """Put the hold, refine and release blocks into every H3 core-denoise sequence.

    Idempotent - a block a sequence already holds is left alone. Returns whether
    the pipeline now holds audio. Hold and release go in as a pair: a sequence
    missing either anchor (a diffusers that renamed `set_timesteps` or
    `after_denoise`) gets neither, and refine needs its own anchor, `denoise`.
    A sequence that cannot take a block is skipped with a warning, and a
    pipeline left with none refuses that block's argument at the call.
    """
    hold_block, refine_block, release_block = blocks()
    inserted = False
    for prefix, sequence in core_denoise_sequences(pipeline):
        sub_blocks = sequence.sub_blocks
        hold_name, release_name = prefix + HOLD_BLOCK, prefix + RELEASE_BLOCK
        hold_before, release_before = prefix + HOLD_BEFORE, prefix + RELEASE_BEFORE
        refine_name, refine_before = prefix + REFINE_BLOCK, prefix + REFINE_BEFORE
        names = list(sub_blocks)
        if hold_name in names:
            inserted = True
        elif hold_before not in names or release_before not in names:
            logger.warning(
                f"MiniMax-H3 has a denoise sequence with no '{hold_before}'/"
                f"'{release_before}' step to anchor the audio hold to, so it "
                f"cannot hold audio"
            )
        else:
            sub_blocks.insert(hold_name, hold_block(), names.index(hold_before))
            names = list(sub_blocks)
            sub_blocks.insert(
                release_name, release_block(), names.index(release_before)
            )
            inserted = True
        names = list(sub_blocks)
        if refine_name in names:
            continue
        if refine_before not in names:
            logger.warning(
                f"MiniMax-H3 has a denoise sequence with no '{refine_before}' "
                f"step to anchor the refine schedule to, so it cannot refine"
            )
            continue
        sub_blocks.insert(refine_name, refine_block(), names.index(refine_before))
    return inserted


def holds_audio(pipeline):
    """Whether a pipeline's block graph carries the audio hold."""
    return any(
        prefix + HOLD_BLOCK in sequence.sub_blocks
        for prefix, sequence in core_denoise_sequences(pipeline)
    )


def refines(pipeline):
    """Whether a pipeline's block graph carries the refine schedule."""
    return any(
        prefix + REFINE_BLOCK in sequence.sub_blocks
        for prefix, sequence in core_denoise_sequences(pipeline)
    )


def hold_audio_reference(value, base_dir=None):
    """A `hold_audio` argument as the MiniMaxH3AudioReference the hold block takes.

    Whatever an argument can hold audio as: a file path (an `asset:` or `output:`
    reference arrives resolved to one), a step's artifact - an AudioTrack, an
    AudioVideo, a bare waveform - or a reference already built. A path or URL goes
    through the location owner (dw/locations.py) before anything opens it, the
    run-time half of the check validation makes on the same argument.

    Args:
        value: The argument's value
        base_dir: Directory a relative path resolves against - the workflow
            file's directory

    Raises:
        ValueError: If the value is not audio
        SecurityError: If the path or URL is outside what the workflow may read
    """
    from ..arguments import media_arguments
    from ..locations import is_http_url, validate_media_path, validate_media_url
    from ..security import (
        ALLOWED_AUDIO_EXTENSIONS,
        InvalidInputError,
        validate_file_extension,
    )

    audio_reference = _diffusers()[2]
    if isinstance(value, audio_reference):
        return value
    if isinstance(value, str):
        try:
            validate_file_extension(value, ALLOWED_AUDIO_EXTENSIONS)
        except InvalidInputError as error:
            raise ValueError(
                f"hold_audio holds a soundtrack, and '{value}' is not an audio file "
                f"({', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))})"
            ) from error
        if is_http_url(value):
            location = validate_media_url(value, "hold_audio")
        else:
            location = validate_file_extension(
                validate_media_path(value, base_dir, "hold_audio"),
                ALLOWED_AUDIO_EXTENSIONS,
            )
        return audio_reference.from_file(location)
    try:
        return audio_reference(**media_arguments(audio_reference, value))
    except ValueError as error:
        raise ValueError(f"hold_audio: {error}") from error


# Guides (#611): earlier clips held as multi-frame condition rows.
#
# H3 already conditions on rows it never steps: a `fl2va` keyframe is one encoded
# latent frame in front of the target, noised to `keyframe_noise_aug` and given
# the target's own rotary time at its anchor. A guide is the same thing for a
# clip - every latent frame of its encode, timed from the target latent it lands
# on, appended after the keyframe rows. Only the layout knows one frame per
# keyframe, so the layout is the step dw replaces; the condition noise, the
# denoise loop and `after_denoise` already work on any prefix of condition rows.
#
# `t2va` has no condition steps, so it gets two thin ones that run diffusers' own
# `prepare_condition_latents` / `prepare_latents_fl2va` only when there is
# conditioning to pack - the stock step cannot take an empty list.

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
LAYOUT_ANCHORS = (
    "MiniMaxH3PrepareLayoutStep",
    "MiniMaxH3PrepareLayoutStep.build_packed_sequence",
    "MiniMaxH3PrepareConditionLatentsStep",
    "MiniMaxH3FL2VAPrepareLatentsStep",
    "_temporal_position_grid",
    "_frame_position_grid",
)

# Pixel frames per VAE chunk and latent frames per chunk - a clip encodes to whole
# latents at 1, 5 or 17m + 5 frames, and lines up with the target's latent grid
# only at a chunk boundary, frame 17j (latent 5j)
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
                f"guide_frames must be 22 or 39 (the whole-latent guide lengths a "
                f"chain carries), got {frames!r}",
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


def default_num_frames():
    """The `num_frames` H3 renders when a step passes none: the stock layout
    step's own default, read off diffusers rather than copied. None when the
    installed diffusers has no such step or default."""
    try:
        from diffusers.modular_pipelines.minimax_h3 import before_denoise
    except ImportError:
        return None
    stock = getattr(before_denoise, "MiniMaxH3PrepareLayoutStep", None)
    if stock is None:
        return None
    try:
        inputs = stock().inputs
    except Exception:  # a diffusers whose step needs arguments to build
        return None
    for param in inputs:
        if param.name == "num_frames" and isinstance(param.default, int):
            return param.default
    return None


def layout_anchor_problem():
    """The first diffusers name the guide layout is built on that the installed
    diffusers lacks, or None."""
    try:
        from diffusers.modular_pipelines.minimax_h3 import before_denoise
    except ImportError:
        return "diffusers.modular_pipelines.minimax_h3.before_denoise"
    for anchor in LAYOUT_ANCHORS:
        owner = before_denoise
        for part in anchor.split("."):
            owner = getattr(owner, part, None)
            if owner is None:
                return anchor
    return None


def guide_position_ids(target_time, frame_grid, latent_index, num_latent_frames):
    """`(num_latent_frames * rows_per_frame, 3)` rotary positions of one guide.

    Time runs from the target's own time at `latent_index`, with the target's
    spacing (`_temporal_position_grid`); every frame carries the target's (h, w)
    grid. Rows are frame-major, the order `patchify_video_latents` packs them in.
    """
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        _temporal_position_grid,
    )

    times = _temporal_position_grid(num_latent_frames, float(target_time[latent_index]))
    rows_per_frame = frame_grid.shape[0]
    positions = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    positions[:, :, 0] = times[:, None]
    positions[:, :, 1:] = frame_grid[None]
    return positions.reshape(-1, 3)


def splice_guide_rows(layout, num_text_tokens, guide_positions, video_tag):
    """The packed layout with guide rows inserted after the existing condition rows.

    `layout` is `(position_ids, token_tags, video_indices, audio_indices,
    text_indices, num_condition_video_rows)` as the stock layout built it, on the
    CPU. Returns the same six, widened by `len(guide_positions)` video rows that
    sit between the keyframe rows and the target audio.
    """
    position_ids, token_tags, video_indices, audio_indices, text_indices, rows = layout
    count = guide_positions.shape[0]
    if not count:
        return layout
    at = num_text_tokens + rows
    sequence_length = position_ids.shape[0] + count
    position_ids = torch.cat(
        [position_ids[:at], guide_positions.to(position_ids.dtype), position_ids[at:]]
    )
    token_tags = torch.cat(
        [
            token_tags[:at],
            torch.full((count,), video_tag, dtype=token_tags.dtype),
            token_tags[at:],
        ]
    )
    audio_start = int(audio_indices[0]) if audio_indices.numel() else at
    video_start = audio_start + audio_indices.numel()
    video_indices = torch.cat(
        [
            torch.arange(num_text_tokens, audio_start + count),
            torch.arange(video_start + count, sequence_length),
        ]
    )
    audio_indices = audio_indices + count
    return (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        rows + count,
    )


def guide_audio_latents(num_frames, fps=24, latents_per_second=40):
    """M, the audio latents a guide of `num_frames` frames holds at frame 0:
    round(num_frames * 40 / 24). 22 -> 37, 39 -> 65."""
    return int(round(num_frames * latents_per_second / fps))


def guide_audio_span(frame, num_frames, fps=24, latents_per_second=40):
    """(start, pad, count) of a guide's audio on the target audio grid.

    `start` is the target audio latent the guide's audio lands on, the one at or
    before its first frame: floor(frame * 40 / 24). `pad` is the seconds of
    silence that go in front of the guide's audio so its first sample keeps its
    time - frame / 24 - once the encode starts at that latent. `count` is the
    latents per channel it covers, to round((frame + num_frames) * 40 / 24); M at
    frame 0 (`guide_audio_latents`)."""
    start = frame * latents_per_second // fps
    pad = frame / fps - start / latents_per_second
    end = guide_audio_latents(frame + num_frames, fps, latents_per_second)
    return start, pad, end - start


def pad_guide_audio(waveform, frame, sample_rate, fps=24, latents_per_second=40):
    """A guide's `(channels, samples)` waveform, at `sample_rate`, with the
    silence in front that puts its first sample at `frame / fps` when sample 0
    is the guide's start latent (`guide_audio_span`) - the audio VAE's hop."""
    start = frame * latents_per_second // fps
    origin = int(round(frame * sample_rate / fps))
    pad = origin - start * (sample_rate // latents_per_second)
    return torch.nn.functional.pad(waveform, (pad, 0)) if pad > 0 else waveform


def guide_audio_positions(num_text_tokens, width_grid, start, count, audio_channels):
    """`(count * audio_channels, 3)` rotary positions of one guide's audio rows:
    the target soundtrack's own, from its latent `start` - time num_text + start
    on, channel-major on the two extremes of the width grid."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        _fill_audio_positions,
    )

    positions = torch.zeros(count * audio_channels, 3, dtype=torch.float64)
    _fill_audio_positions(
        positions,
        slice(0, positions.shape[0]),
        count,
        float(num_text_tokens + start),
        width_grid,
        audio_channels,
    )
    return positions


def splice_guide_audio_rows(layout, audio_positions, audio_tag):
    """The packed layout with guide audio rows in front of the target audio.

    `layout` is `(position_ids, token_tags, video_indices, audio_indices,
    text_indices)` on the CPU. Returns the same five, widened by
    `len(audio_positions)` audio rows that lead `audio_indices` - the
    condition audio prefix - with every video row after them moved along.
    """
    position_ids, token_tags, video_indices, audio_indices, text_indices = layout
    count = audio_positions.shape[0]
    if not count:
        return layout
    at = int(audio_indices[0])
    position_ids = torch.cat(
        [position_ids[:at], audio_positions.to(position_ids.dtype), position_ids[at:]]
    )
    token_tags = torch.cat(
        [
            token_tags[:at],
            torch.full((count,), audio_tag, dtype=token_tags.dtype),
            token_tags[at:],
        ]
    )
    video_indices = torch.where(
        video_indices >= at, video_indices + count, video_indices
    )
    audio_indices = torch.cat([torch.arange(at, at + count), audio_indices + count])
    return position_ids, token_tags, video_indices, audio_indices, text_indices


def fit_guide_frames(frames, height, width):
    """`(n, height, width, 3)` uint8 frames covering the canvas: scaled to cover
    it and centre-cropped, with the arithmetic H3 fits a follower keyframe with
    (diffusers' `MiniMaxH3BeforeEncodeStep`). A clip already at the canvas
    passes through untouched."""
    import numpy as np
    from PIL import Image

    if frames.shape[1:3] == (height, width):
        return frames
    source_height, source_width = frames.shape[1:3]
    scale = max(width / source_width, height / source_height)
    size = (
        max(width, round(source_width * scale)),
        max(height, round(source_height * scale)),
    )
    left = max(0, (size[0] - width) // 2)
    top = max(0, (size[1] - height) // 2)
    box = (left, top, left + width, top + height)
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS).crop(box)
            )
            for frame in frames
        ]
    )


def guide_audio_waveform(value):
    """(waveform, sample_rate) of a guide video's soundtrack: a `(channels,
    samples)` float32 tensor, from what a guide's `video` loads to with its audio -
    an AudioVideo, a step's Selected pick of one or a one-video list of one.

    Raises:
        ValueError: If the video carries no audio
    """
    while True:
        if hasattr(value, "value") and hasattr(value, "position"):
            value = value.value
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        else:
            break
    audio = getattr(value, "audio", None)
    if audio is not None:
        audio = as_channels_samples(audio)
    if audio is None or not audio.numel():
        raise ValueError(
            "'audio' is true, but this guide's video has no audio - pass a video "
            "with a soundtrack, or drop 'audio'"
        )
    return audio, getattr(value, "sample_rate", None)


def guide_frames_array(value):
    """A guide's video as `(n, h, w, 3)` uint8 frames - from a list of PIL images
    or arrays (what a `video` argument loads to), or one array or tensor of
    frames, float in [0, 1] or uint8.

    Raises:
        ValueError: If the value is not frames of a video
    """
    import numpy as np
    from PIL import Image

    # A step's result: a Selected pick, a one-video list, a clip with its audio
    while True:
        if hasattr(value, "value") and hasattr(value, "position"):
            value = value.value
        elif hasattr(value, "frames") and hasattr(value, "audio"):
            value = value.frames
        elif (
            isinstance(value, (list, tuple))
            and len(value) == 1
            and not isinstance(value[0], (str, Image.Image))
            and getattr(value[0], "ndim", 4) != 3
        ):
            value = value[0]
        else:
            break
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("a guide's video has no frames")
        if isinstance(value[0], str):
            raise ValueError(f"a guide's video did not load: {value[0]!r}")
        value = np.stack(
            [
                np.asarray(f.convert("RGB")) if isinstance(f, Image.Image) else f
                for f in value
            ]
        )
    if not isinstance(value, np.ndarray):
        raise ValueError(f"a guide's video must be a video, got {type(value).__name__}")
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4 or value.shape[-1] not in (3, 4):
        raise ValueError(
            f"a guide's video must be (frames, height, width, 3), got {value.shape}"
        )
    value = value[..., :3]
    if value.dtype != np.uint8:
        value = (np.clip(value.astype(np.float32), 0.0, 1.0) * 255.0).round()
        value = value.astype(np.uint8)
    return np.ascontiguousarray(value)


def _make_guide_blocks():
    """The guide layout and the two t2va condition steps, defined against the
    installed diffusers."""
    import numpy as np
    from diffusers.models import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    from diffusers.modular_pipelines.minimax_h3 import before_denoise
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
    from diffusers.modular_pipelines.modular_pipeline_utils import (
        ComponentSpec,
        InputParam,
        OutputParam,
    )

    setup_step, reference_encoder, _, latents_per_second = _diffusers()
    stock_layout = before_denoise.MiniMaxH3PrepareLayoutStep
    stock_condition = before_denoise.MiniMaxH3PrepareConditionLatentsStep
    stock_latents = before_denoise.MiniMaxH3FL2VAPrepareLatentsStep

    class DwH3GuideLayoutStep(ModularPipelineBlocks):
        model_name = "minimax-h3"

        def __init__(self):
            super().__init__()
            # Called, not copied: the stock layout resolves the canvas and builds
            # the sequence, and the guides are spliced into what it built
            self._layout = stock_layout()
            # ...and the encode a Ref2VA soundtrack reference takes, for a
            # guide's audio
            self._audio_encoder = reference_encoder()

        @property
        def description(self):
            return (
                "dw: the stock layout, plus `guides` - each guide clip VAE-encoded on the canvas and appended as "
                "condition rows timed from the target frame it lands on. Without `guides` it is the stock layout."
            )

        @property
        def expected_components(self):
            return list(self._layout.expected_components) + [
                ComponentSpec("vae", AutoencoderKLMiniMaxH3),
                ComponentSpec("audio_vae", AutoencoderKLMiniMaxH3Audio),
            ]

        @property
        def expected_configs(self):
            return self._layout.expected_configs

        @property
        def inputs(self):
            return list(self._layout.inputs) + [
                InputParam(
                    name=GUIDES_INPUT,
                    type_hint=list,
                    default=None,
                    description=(
                        "Clips to hold the generated video to: a list of {'video': (n, h, w, 3) uint8 frames, "
                        "'frame': pixel frame, a multiple of 17}, n one of 1, 5 or 17m + 5. A guide that also "
                        "carries 'audio': (channels, samples) and 'sample_rate' holds that soundtrack over its span."
                    ),
                ),
                InputParam(name="condition_latents", type_hint=list, default=None),
            ]

        @property
        def intermediate_outputs(self):
            return list(self._layout.intermediate_outputs) + [
                OutputParam("condition_latents", type_hint=list),
                OutputParam(GUIDE_AUDIO_ROWS, type_hint=torch.Tensor),
            ]

        @torch.no_grad()
        def __call__(self, components, state):
            components, state = self._layout(components, state)
            guides = state.get(GUIDES_INPUT) or []
            if not guides:
                return components, state
            # What the stock layout set is its output, not this block's input,
            # so it is read off the state rather than a block state
            laid = state.get(
                [
                    "height",
                    "width",
                    "num_frames",
                    "text_token_tags",
                    "latent_height",
                    "latent_width",
                    "num_latent_frames",
                    "position_ids",
                    "token_tags",
                    "video_indices",
                    "audio_indices",
                    "text_indices",
                    "num_condition_video_rows",
                    "condition_latents",
                    "num_audio_latents",
                ]
            )

            device = components._execution_device
            num_text = laid["text_token_tags"].shape[0]
            _, patch_h, patch_w = components.patch_size
            frame_grid, width_grid = before_denoise._frame_position_grid(
                laid["latent_height"], laid["latent_width"], patch_h, patch_w
            )
            target_time = before_denoise._temporal_position_grid(
                laid["num_latent_frames"], float(num_text)
            )

            latents, positions = [], []
            for index, guide in enumerate(guides):
                frame, frames = guide["frame"], guide["video"]
                problem = guide_frame_problem(frame) or guide_end_problem(
                    frame, frames.shape[0], laid["num_frames"]
                )
                if problem:
                    raise ValueError(f"guides[{index}]: {problem}")
                pixels = fit_guide_frames(frames, laid["height"], laid["width"])
                pixels = torch.from_numpy(np.ascontiguousarray(pixels))
                encoded = encode_vae_condition(
                    components.vae,
                    pixels.to(device).permute(3, 0, 1, 2)[None],
                    components.pixel_mean,
                    components.pixel_std,
                    components.keyframe_encode_seed,
                )
                latents.append(encoded)
                positions.append(
                    guide_position_ids(
                        target_time,
                        frame_grid,
                        frame // GUIDE_FRAMES_PER_CHUNK * GUIDE_LATENTS_PER_CHUNK,
                        encoded.shape[2],
                    )
                )
                logger.info(
                    f"Guide {index}: {frames.shape[0]} frames at frame {frame} "
                    f"({encoded.shape[2]} latent frames)"
                )

            names = (
                "position_ids",
                "token_tags",
                "video_indices",
                "audio_indices",
                "text_indices",
            )
            layout = splice_guide_rows(
                tuple(laid[name].cpu() for name in names)
                + (laid["num_condition_video_rows"],),
                num_text,
                torch.cat(positions),
                components.video_tag,
            )

            # Each guide's audio, encoded on the target audio grid from the latent
            # at or before its first frame, as condition audio rows timed there
            audio_rows, audio_positions = [], []
            for index, guide in enumerate(guides):
                if guide.get("audio") is None:
                    continue
                rows, start, count = self._guide_audio(
                    components, guide, laid["num_audio_latents"]
                )
                audio_rows.append(rows)
                audio_positions.append(
                    guide_audio_positions(
                        num_text, width_grid, start, count, components.audio_channels
                    )
                )
                logger.info(
                    f"Guide {index}: audio held over audio latents {start} to "
                    f"{start + count} ({rows.shape[0]} rows)"
                )
            num_audio_rows = state.get("num_condition_audio_rows") or 0
            if audio_rows:
                layout = (
                    splice_guide_audio_rows(
                        layout[:5], torch.cat(audio_positions), components.audio_tag
                    )
                    + layout[5:]
                )
                state.set(GUIDE_AUDIO_ROWS, torch.cat(audio_rows))
                num_audio_rows += sum(rows.shape[0] for rows in audio_rows)

            for name, value in zip(names, layout):
                state.set(name, value.to(device))
            state.set("num_condition_video_rows", layout[5])
            state.set("num_condition_audio_rows", num_audio_rows)
            # Keyframes first, then guides: the order the rows were laid out in
            state.set(
                "condition_latents", list(laid["condition_latents"] or []) + latents
            )
            return components, state

        def _guide_audio(self, components, guide, num_audio_latents):
            """(rows, start, count): a guide's audio as `(channels * count, C)`
            channel-major rows from target audio latent `start`."""
            frame, length = guide["frame"], guide["video"].shape[0]
            fps = components.fps
            vae_rate = components.audio_sampling_rate
            start, _, count = guide_audio_span(frame, length, fps, latents_per_second)
            count = min(count, num_audio_latents - start)
            original = as_channels_samples(guide["audio"])
            rate = guide.get("sample_rate") or vae_rate
            # The guide's span only, on the VAE's rate and channels the way a
            # Ref2VA soundtrack is, then pre-padded to the VAE's hop
            waveform = setup_step._normalize_audio_condition(
                original, rate, vae_rate, max_duration=length / fps
            )
            waveform = pad_guide_audio(
                waveform, frame, vae_rate, fps, latents_per_second
            )
            latents = encode_audio_span(
                self._audio_encoder, components, waveform, count
            )
            return latents.reshape(-1, components.audio_latent_channels), start, count

    def _when_conditioned(stock, description):
        """A t2va step that runs the stock `fl2va` one only with conditioning
        to pack - the stock step refuses an empty list."""

        class DwH3WhenConditionedStep(ModularPipelineBlocks):
            model_name = "minimax-h3"

            def __init__(self):
                super().__init__()
                self._stock = stock()

            @property
            def description(self):
                return description

            @property
            def expected_components(self):
                return self._stock.expected_components

            @property
            def inputs(self):
                inputs = []
                for param in self._stock.inputs:
                    if param.name == "condition_latents":
                        param = InputParam(
                            name="condition_latents", type_hint=list, default=None
                        )
                    elif param.name == "condition_rows":
                        param = InputParam(
                            name="condition_rows",
                            type_hint=torch.Tensor,
                            default=None,
                        )
                    inputs.append(param)
                return inputs

            @property
            def intermediate_outputs(self):
                return self._stock.intermediate_outputs

            def __call__(self, components, state):
                # Read off the state: the latents step does not declare it
                if not state.get("condition_latents"):
                    return components, state
                return self._stock(components, state)

        return DwH3WhenConditionedStep

    class DwH3GuideAudioStep(ModularPipelineBlocks):
        model_name = "minimax-h3"

        @property
        def description(self):
            return (
                "dw: puts the guides' encoded audio in front of `audio_latents` as the condition audio prefix the "
                "guide layout reserved. Does nothing when no guide carries audio."
            )

        @property
        def inputs(self):
            return [
                InputParam(name="audio_latents", type_hint=torch.Tensor, required=True),
                InputParam(name="audio_indices", type_hint=torch.Tensor, required=True),
                InputParam(name=GUIDE_AUDIO_ROWS, type_hint=torch.Tensor, default=None),
            ]

        @property
        def intermediate_outputs(self):
            return [OutputParam("audio_latents", type_hint=torch.Tensor)]

        def __call__(self, components, state):
            block_state = self.get_block_state(state)
            rows = getattr(block_state, GUIDE_AUDIO_ROWS)
            audio_latents = block_state.audio_latents
            # Once: the layout counted the rows into `audio_indices` already
            if (
                rows is None
                or audio_latents.shape[0] >= block_state.audio_indices.numel()
            ):
                return components, state
            block_state.audio_latents = torch.cat(
                [rows.to(audio_latents.device, audio_latents.dtype), audio_latents]
            )
            self.set_block_state(state, block_state)
            return components, state

    condition_step = _when_conditioned(
        stock_condition,
        "dw: diffusers' `prepare_condition_latents`, run in `t2va` only when `guides` gave it condition latents.",
    )
    latents_step = _when_conditioned(
        stock_latents,
        "dw: diffusers' `prepare_latents_fl2va`, run in `t2va` only when `guides` gave it condition rows.",
    )
    return DwH3GuideLayoutStep, condition_step, latents_step, DwH3GuideAudioStep


_GUIDE_BLOCKS = None


def guide_blocks():
    """(DwH3GuideLayoutStep, the t2va condition step, the t2va latents step,
    DwH3GuideAudioStep), built once."""
    global _GUIDE_BLOCKS
    if _GUIDE_BLOCKS is None:
        _GUIDE_BLOCKS = _make_guide_blocks()
    return _GUIDE_BLOCKS


def _layout_sequences(pipeline):
    """[(prefix, sequence)] for the core-denoise sequences that lay out their own
    keyframes - every one but `ref2va`, whose layout is a different step."""
    from diffusers.modular_pipelines.minimax_h3 import before_denoise

    stock = before_denoise.MiniMaxH3PrepareLayoutStep
    layout = guide_blocks()[0]
    found = []
    for prefix, sequence in core_denoise_sequences(pipeline):
        step = sequence.sub_blocks.get(prefix + LAYOUT_STEP)
        if type(step) in (stock, layout):
            found.append((prefix, sequence))
    return found


def insert_guides(pipeline):
    """Swap `prepare_layout` for the guide layout in every `t2va`/`fl2va`
    core-denoise sequence, and give `t2va` the condition steps it lacks.

    Idempotent. Returns whether the pipeline now takes `guides`. A diffusers
    missing a name the layout is built on gets nothing, with a warning naming it,
    and `guides` is refused at the call (`guides_refusal`).
    """
    missing = layout_anchor_problem()
    if missing:
        if core_denoise_sequences(pipeline):
            logger.warning(
                f"MiniMax-H3 guides are built on diffusers' '{missing}', which the "
                f"installed diffusers does not have, so this pipeline cannot take guides"
            )
        return False
    layout_block, condition_block, latents_block, audio_block = guide_blocks()
    inserted = False
    for prefix, sequence in _layout_sequences(pipeline):
        sub_blocks = sequence.sub_blocks
        name = prefix + LAYOUT_STEP
        if not isinstance(sub_blocks[name], layout_block):
            sub_blocks[name] = layout_block()
        names = list(sub_blocks)
        latents_name = prefix + "prepare_latents"
        if prefix + "prepare_condition_latents" not in names and (
            latents_name in names
        ):
            # t2va: condition noise is drawn before the generated rows' noise,
            # and the condition rows go in front of them after - fl2va's order
            if prefix + GUIDE_CONDITION_BLOCK not in names:
                sub_blocks.insert(
                    prefix + GUIDE_CONDITION_BLOCK,
                    condition_block(),
                    names.index(latents_name),
                )
                names = list(sub_blocks)
            if prefix + GUIDE_LATENTS_BLOCK not in names:
                sub_blocks.insert(
                    prefix + GUIDE_LATENTS_BLOCK,
                    latents_block(),
                    names.index(latents_name) + 1,
                )
        names = list(sub_blocks)
        # Guide audio leads the audio rows, so it goes in before the audio hold
        # writes the target rows after it - before `set_timesteps`, or before
        # the hold when that is already in
        audio_before = next(
            (n for n in (prefix + HOLD_BLOCK, prefix + HOLD_BEFORE) if n in names), None
        )
        if prefix + GUIDE_AUDIO_BLOCK not in names and audio_before:
            sub_blocks.insert(
                prefix + GUIDE_AUDIO_BLOCK, audio_block(), names.index(audio_before)
            )
        inserted = True
    return inserted


def takes_guides(pipeline):
    """Whether a pipeline's block graph carries the guide layout."""
    if layout_anchor_problem():
        return False
    layout_block = guide_blocks()[0]
    return any(
        isinstance(sequence.sub_blocks.get(prefix + LAYOUT_STEP), layout_block)
        for prefix, sequence in core_denoise_sequences(pipeline)
    )


def guides_refusal(pipeline):
    """Why this pipeline cannot take `guides`, or None."""
    missing = layout_anchor_problem()
    if missing:
        return (
            f"guides need diffusers' MiniMax-H3 '{missing}', which the installed "
            f"diffusers does not have"
        )
    if not takes_guides(pipeline):
        return (
            f"guides is a MiniMax-H3 t2va/fl2va argument, and "
            f"{type(pipeline).__name__} has no '{LAYOUT_STEP}' step to take them"
        )
    return None
