"""MiniMax-H3 audio hold: generating a video to a soundtrack the caller already has.

The one place dw modifies a diffusers modular pipeline's block graph. H3 denoises
video and audio rows in one packed sequence, and the rows it treats as conditioning
are the leading `num_condition_audio_rows` of the audio stream: pinned clean at
`t = 1.0`, never stepped by the scheduler, attended to by every generated row. That
is how a Ref2VA soundtrack reference rides along. Holding a track is the same
mechanism pointed at the target rows instead - the caller's waveform is encoded
through diffusers' own Ref2VA reference encode, written over the target audio rows,
and the count is widened to cover them, so the denoiser generates video that fits a
soundtrack it is not allowed to change.

Two blocks go into each of H3's three core-denoise sequences (`t2va`, `fl2va`,
`ref2va`), always, as no-ops when `hold_audio` is not passed:

- `DwH3HoldAudioStep`, before `set_timesteps` - after every `prepare_latents*`
  step, so Ref2VA's reference rows are already in front and its reference-row
  count check has already run against the count it expects;
- `DwH3ReleaseAudioStep`, before `after_denoise` - which slices
  `audio_latents[num_condition_audio_rows:]` as the generated track, so the count
  has to be back to what it was for the held rows to be decoded as the target.

The decoded `audio` is still the VAE round trip of the held rows. The step's result
uses `dw_held_audio` instead - the caller's own waveform fitted to the video's
duration (dw/output_extraction.py), which `HELD_AUDIO_OUTPUT` names for the call
(dw/pipeline_processors/pipeline.py).
"""

import logging

import torch

logger = logging.getLogger("dw")

# The call argument, and the names the hold block leaves in the pipeline state
HOLD_AUDIO_INPUT = "hold_audio"
HELD_ROWS = "dw_held_audio_rows"
HELD_AUDIO_OUTPUT = "dw_held_audio"
HELD_AUDIO_RATE_OUTPUT = "dw_held_audio_sampling_rate"

# The block names dw inserts, and the diffusers block each goes in front of
HOLD_BLOCK = "dw_hold_audio"
RELEASE_BLOCK = "dw_release_audio"
HOLD_BEFORE = "set_timesteps"
RELEASE_BEFORE = "after_denoise"

# MiniMaxH3Blocks: the top-level step holding the auto denoise step, and the three
# core-denoise sequences that step chooses between
DENOISE_STEP = "denoise"
CORE_DENOISE_SEQUENCES = ("t2va", "fl2va", "ref2va")

# Encoded silence pads a track shorter than the video: this many latents of zeros
# are encoded and the middle one taken, clear of any edge effect of the VAE's
SILENCE_LATENTS = 4


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


def _make_blocks():
    """The two block classes, defined against the installed diffusers."""
    from diffusers.modular_pipelines.modular_pipeline import (
        ModularPipelineBlocks,
        PipelineState,
    )
    from diffusers.modular_pipelines.modular_pipeline_utils import (
        InputParam,
        OutputParam,
    )

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

        def _encode(self, components, waveform):
            """`(channels, n, C)` normalized audio latents of a VAE-rate stereo waveform."""
            sub_state = PipelineState()
            sub_state.set(
                "normalized_references",
                [
                    audio_reference(
                        audio=waveform, sample_rate=components.audio_sampling_rate
                    )
                ],
            )
            _, sub_state = self._encoder(components, sub_state)
            rows = sub_state.get("audio_condition_latents")[0]
            return rows.reshape(
                components.audio_channels, -1, components.audio_latent_channels
            )

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
            samples_per_latent = vae_rate // latents_per_second
            original = as_channels_samples(held.audio)
            rate = held.sample_rate or vae_rate

            # Onto the VAE's rate and channels the way a Ref2VA reference is, then
            # padded with silence to the whole target - a short track encodes
            # with its silence rather than having it appended in latent space
            waveform = setup_step._normalize_audio_condition(
                original, rate, vae_rate, max_duration=duration
            )
            waveform = fit_samples(waveform, num_latents * samples_per_latent)
            latents = self._encode(components, waveform)
            if latents.shape[1] < num_latents:
                silence = self._encode(
                    components,
                    torch.zeros(2, SILENCE_LATENTS * samples_per_latent),
                )
                middle = silence.shape[1] // 2
                latents = fit_latents(
                    latents, num_latents, silence[:, middle : middle + 1]
                )
            else:
                latents = fit_latents(latents, num_latents, None)

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

    return DwH3HoldAudioStep, DwH3ReleaseAudioStep


_BLOCKS = None


def blocks():
    """(DwH3HoldAudioStep, DwH3ReleaseAudioStep), built once."""
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
    """Put the hold and release blocks into every H3 core-denoise sequence.

    Idempotent - a sequence already holding them is left alone. Returns whether
    the pipeline now holds audio; a sequence missing an anchor (a diffusers that
    renamed `set_timesteps` or `after_denoise`) is skipped with a warning, and a
    pipeline left with none refuses `hold_audio` at the call.
    """
    hold_block, release_block = blocks()
    inserted = False
    for prefix, sequence in core_denoise_sequences(pipeline):
        sub_blocks = sequence.sub_blocks
        hold_name, release_name = prefix + HOLD_BLOCK, prefix + RELEASE_BLOCK
        hold_before, release_before = prefix + HOLD_BEFORE, prefix + RELEASE_BEFORE
        if hold_name in sub_blocks:
            inserted = True
            continue
        names = list(sub_blocks)
        if hold_before not in names or release_before not in names:
            logger.warning(
                f"MiniMax-H3 has a denoise sequence with no '{hold_before}'/"
                f"'{release_before}' step to anchor the audio hold to, so it "
                f"cannot hold audio"
            )
            continue
        sub_blocks.insert(hold_name, hold_block(), names.index(hold_before))
        names = list(sub_blocks)
        sub_blocks.insert(release_name, release_block(), names.index(release_before))
        inserted = True
    return inserted


def holds_audio(pipeline):
    """Whether a pipeline's block graph carries the audio hold."""
    return any(
        prefix + HOLD_BLOCK in sequence.sub_blocks
        for prefix, sequence in core_denoise_sequences(pipeline)
    )


def hold_audio_reference(value):
    """A `hold_audio` argument as the MiniMaxH3AudioReference the hold block takes.

    Whatever an argument can hold audio as: a file path (an `asset:` or `output:`
    reference arrives resolved to one), a step's artifact - an AudioTrack, an
    AudioVideo, a bare waveform - or a reference already built.

    Raises:
        ValueError: If the value is not audio
    """
    from ..arguments import media_arguments
    from ..security import ALLOWED_AUDIO_EXTENSIONS

    audio_reference = _diffusers()[2]
    if isinstance(value, audio_reference):
        return value
    if isinstance(value, str):
        extension = "." + value.rsplit(".", 1)[-1].lower() if "." in value else ""
        if extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"hold_audio holds a soundtrack, and '{value}' is not an audio file "
                f"({', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))})"
            )
        return audio_reference.from_file(value)
    try:
        return audio_reference(**media_arguments(audio_reference, value))
    except ValueError as error:
        raise ValueError(f"hold_audio: {error}") from error
