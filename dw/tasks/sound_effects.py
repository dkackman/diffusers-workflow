"""
generate_sound_effects - MMAudio (large 44.1 kHz v2) sound effects for a clip,
a prompt, or both (#1124 stage D).

The pieces are stages A-C's: the flow network (`mmaudio_model`), the DFN5B
CLIP towers (`mmaudio_clip`), the Synchformer encoder (`mmaudio_sync_model`)
and the VAE + BigVGAN decoder read from Kandinsky 6's `audio_vae/` and
`vocoder/` folders (`mmaudio_decoder`). This module downloads each one's
pinned files with `hf_hub_download(revision=...)`, caches it with
`cached_model`, and runs them in upstream's order: frames at 8 fps (CLIP,
384) and 25 fps (Synchformer, 224), the prompt through CLIP's text tower,
Euler flow matching with classifier-free guidance, then the decode.

The network's `unnormalize` returns raw VAE latents, so they go to the
decoder as they are - no division by the VAE's scaling_factor, which K6's
own pipeline applies to latents of its own scaled space.

Every refusal runs before any load: the trust gate on a remote-code
argument, an unknown argument (a kwargs task would otherwise drop it
silently), no `video` and no `prompt`, and a `duration` over 20 s or
longer than the clip. Two notes go on the result through
`dw.events.emit_warning`: a duration outside the 4-12 s the model was
trained around, and an output that measures noise-like (spectral flatness
>= 0.3 and centroid >= 8 kHz).

Licences: the MMAudio weights (network, Synchformer, VAE tensors) are
CC-BY-NC 4.0; DFN5B is under the Apple Machine Learning Research Model
License (research only) - ATTRIBUTIONS.md.
"""

import logging
import math
import os
import tempfile

import numpy
import torch

from .. import get_device_type
from ..media_types import AudioTrack
from .weights import file_entries

logger = logging.getLogger("dw")

COMMAND = "generate_sound_effects"

DEFAULT_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 4.5
TEXT_ONLY_DURATION = 8.0
MAX_DURATION = 20.0
# The span the large 44k v2 network was trained and evaluated around (8 s);
# outside it the output drifts, so a duration there is noted, not refused
TRAINED_DURATION = (4.0, 12.0)
# The noise-like output note: both must hold (dw/dsp.py spectral_centroid)
NOISE_FLATNESS = 0.3
NOISE_CENTROID_HZ = 8000.0

CLIP_FPS = 8
SYNC_FPS = 25
SYNC_SIZE = 224
# Synchformer reads 16-frame segments: the shortest clip it can encode
MIN_SYNC_FRAMES = 16
# A bare frame list carries no rate; an AudioVideo or a file does
ASSUMED_FPS = 24

ARGUMENTS = frozenset(
    {"video", "prompt", "negative_prompt", "duration", "steps", "guidance_scale"}
)

LICENCE_NOTE = (
    "MMAudio weights (network, Synchformer, VAE tensors): CC-BY-NC 4.0, "
    "non-commercial. DFN5B CLIP: Apple Machine Learning Research Model License "
    "(research only) - Apple Machine Learning Research Model is licensed under "
    "the Apple Machine Learning Research Model License Agreement."
)


def required_weights(arguments=None):
    """Every file the task reads, each pinned (dw/tasks/weights.py): the
    MMAudio network and Synchformer, K6's audio_vae/ and vocoder/ config and
    safetensors, and DFN5B's open_clip weights and tokenizer. Never MMAudio's
    own v1-44.pth VAE or bigvgan_generator.pt: the decoder is K6's copy."""
    from .mmaudio_clip import (
        CLIP_REPO_ID,
        CLIP_REVISION,
        CLIP_TOKENIZER_FILE,
        CLIP_WEIGHTS_FILE,
    )
    from .mmaudio_decoder import K6_REPO_ID, K6_REVISION
    from .mmaudio_model import NETWORK_FILE, REPO_ID, REVISION
    from .mmaudio_sync_model import SYNC_FILE

    entries = []
    for filename in (NETWORK_FILE, SYNC_FILE):
        entries += file_entries(REPO_ID, filename, REVISION)
    for filename in _decoder_files():
        entries += file_entries(K6_REPO_ID, filename, K6_REVISION)
    for filename in (CLIP_WEIGHTS_FILE, CLIP_TOKENIZER_FILE):
        entries += file_entries(CLIP_REPO_ID, filename, CLIP_REVISION)
    return entries


def _decoder_files():
    from .mmaudio_decoder import CONFIG_FILE, VAE_FOLDER, VOCODER_FOLDER, WEIGHTS_FILE

    return [
        f"{folder}/{name}"
        for folder in (VAE_FOLDER, VOCODER_FOLDER)
        for name in (CONFIG_FILE, WEIGHTS_FILE)
    ]


# ---------------------------------------------------------------- refusals


def _refuse_remote_code(kwargs):
    """trust_remote_code / custom_pipeline / custom_generate as a step
    argument: the trust gate's refusal when workflows are untrusted, and a
    plain error otherwise - this task loads no Hub code either way."""
    from ..security import UntrustedWorkflowError
    from ..trust import TASK_REMOTE_CODE_ARGUMENTS, workflows_are_trusted

    given = [name for name in TASK_REMOTE_CODE_ARGUMENTS if kwargs.get(name)]
    if not given:
        return
    names = " or ".join(map(repr, given))
    if not workflows_are_trusted():
        raise UntrustedWorkflowError(
            f"Refusing to run {COMMAND}: its arguments set {names}, which "
            "downloads and executes Python from a Hub repository, so it needs "
            "trusted workflows - see docs/SECURITY.md. generate_sound_effects "
            "loads no Hub code from its arguments in any case; drop it"
        )
    raise ValueError(
        f"{COMMAND} takes no {names} argument - it loads no Hub code from its "
        "arguments, so the setting would have no effect. Drop it"
    )


def _refuse_unknown(kwargs):
    from ..trust import TASK_REMOTE_CODE_ARGUMENTS

    unknown = sorted(set(kwargs) - ARGUMENTS - set(TASK_REMOTE_CODE_ARGUMENTS))
    if unknown:
        raise ValueError(
            f"{COMMAND} takes no {', '.join(map(repr, unknown))} argument - it "
            f"takes {', '.join(sorted(ARGUMENTS))}"
        )


def _number(name, value, whole=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{COMMAND}: '{name}' must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{COMMAND}: '{name}' must be positive, got {value!r}")
    if whole and int(value) != value:
        raise ValueError(f"{COMMAND}: '{name}' must be a whole number, got {value!r}")
    return int(value) if whole else float(value)


def _text(name, value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{COMMAND}: '{name}' must be text, got {value!r}")
    return value.strip() or None


def _check_duration(duration, clip_seconds):
    """The duration the run uses: the given one, refused over MAX_DURATION
    or past the clip's end, else the clip's length (capped), else 8 s."""
    if duration is None:
        if clip_seconds is None:
            return TEXT_ONLY_DURATION
        return min(clip_seconds, MAX_DURATION)
    duration = _number("duration", duration)
    if duration > MAX_DURATION:
        raise ValueError(
            f"{COMMAND}: 'duration' {duration:g} s is over the {MAX_DURATION:g} s "
            "this task generates; ask for less, or generate the clip in pieces"
        )
    if clip_seconds is not None and duration > clip_seconds + 1e-3:
        raise ValueError(
            f"{COMMAND}: 'duration' {duration:g} s is longer than the "
            f"{clip_seconds:.2f} s clip in 'video' - leave it unset for the "
            "clip's whole length"
        )
    return duration


# ------------------------------------------------------------------- video


def _video_file(video):
    """(path, is_temporary) for a 'video': the file a stored video names, or
    an in-memory clip (an AudioVideo or a frame list) written to a temporary
    mp4 at its own frame rate (ASSUMED_FPS for bare frames)."""
    from ..locations import validate_media_path
    from .video_utils import VideoFileReference

    if isinstance(video, VideoFileReference):
        video = video.path
    if isinstance(video, str):
        return validate_media_path(video, None, "a video argument"), False
    frames = getattr(video, "frames", video)
    fps = getattr(video, "fps", None) or ASSUMED_FPS
    if frames is None or isinstance(frames, (dict, bytes)):
        raise ValueError(
            f"{COMMAND}: 'video' must be a video (asset:, output:, a path or an "
            f"earlier step's video), not {type(video).__name__}"
        )
    from diffusers.utils import export_to_video

    from ..result import _frames_for_export_to_video

    handle, path = tempfile.mkstemp(suffix=".mp4", prefix="dw-sfx-")
    os.close(handle)
    export_to_video(_frames_for_export_to_video(frames), path, fps=fps)
    return path, True


def _clip_seconds(path):
    from ..media import video_shape

    shape = video_shape(path)
    if shape.get("fps") and shape.get("frame_count"):
        return shape["frame_count"] / shape["fps"]
    from ..media import media_duration

    return media_duration(path)


def _frame_tensors(path, duration):
    """The clip's CLIP frames (1, T, 3, 384, 384) in [0, 1] and Synchformer
    frames (1, T, 3, 224, 224) in [-1, 1], and the duration both cover
    (upstream trims to the shorter of the two)."""
    from ..media import sample_frames
    from .mmaudio_clip import ClipConfig

    clip = sample_frames(path, CLIP_FPS, duration, ClipConfig().image_size)
    sync = sample_frames(path, SYNC_FPS, duration, SYNC_SIZE, crop=True)
    if len(sync) < MIN_SYNC_FRAMES or len(clip) < 1:
        raise ValueError(
            f"{COMMAND}: the clip in 'video' is too short - it needs at least "
            f"{MIN_SYNC_FRAMES / SYNC_FPS:.2f} s of frames"
        )
    covered = min(len(clip) / CLIP_FPS, len(sync) / SYNC_FPS, duration)
    clip_t = torch.from_numpy(numpy.ascontiguousarray(clip)).float().div(255.0)
    sync_t = torch.from_numpy(numpy.ascontiguousarray(sync)).float()
    return (
        clip_t.permute(0, 3, 1, 2).unsqueeze(0),
        sync_t.permute(0, 3, 1, 2).unsqueeze(0),
        covered,
    )


# ------------------------------------------------------------------ models


def _dtype(device):
    return torch.bfloat16 if get_device_type(device) == "cuda" else torch.float32


def _download(repo, filename, revision):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, revision=revision)


def _load_models(device, dtype):
    """The network, CLIP + tokenizer, Synchformer and decoder, each cached
    under (command, repo, file, revision, device, dtype)."""
    from . import mmaudio_clip, mmaudio_decoder, mmaudio_model, mmaudio_sync_model
    from .model_cache import cached_model

    repo, rev = mmaudio_model.REPO_ID, mmaudio_model.REVISION
    where = (str(device), str(dtype))

    def network():
        path = _download(repo, mmaudio_model.NETWORK_FILE, rev)
        logger.info(f"Loading MMAudio network to {device}")
        return mmaudio_model.load_network(path, device=device, dtype=dtype)

    def sync():
        path = _download(repo, mmaudio_sync_model.SYNC_FILE, rev)
        logger.info(f"Loading MMAudio Synchformer to {device}")
        return mmaudio_sync_model.load_synchformer(path, device=device, dtype=dtype)

    clip_repo, clip_rev = mmaudio_clip.CLIP_REPO_ID, mmaudio_clip.CLIP_REVISION

    def clip():
        weights = _download(clip_repo, mmaudio_clip.CLIP_WEIGHTS_FILE, clip_rev)
        tokens = _download(clip_repo, mmaudio_clip.CLIP_TOKENIZER_FILE, clip_rev)
        logger.info(f"Loading DFN5B CLIP to {device}")
        model = mmaudio_clip.load_clip(weights, device=device, dtype=dtype)
        return model, mmaudio_clip.load_clip_tokenizer(tokens)

    k6_repo, k6_rev = mmaudio_decoder.K6_REPO_ID, mmaudio_decoder.K6_REVISION

    def decoder():
        paths = [_download(k6_repo, name, k6_rev) for name in _decoder_files()]
        vae_folder, vocoder_folder = (
            os.path.dirname(paths[0]),
            os.path.dirname(paths[2]),
        )
        logger.info(f"Loading MMAudio VAE + vocoder (K6) to {device}")
        return mmaudio_decoder.load_decoder(vae_folder, vocoder_folder, device=device)

    return (
        cached_model((COMMAND, repo, mmaudio_model.NETWORK_FILE, rev, *where), network),
        cached_model(
            (COMMAND, clip_repo, mmaudio_clip.CLIP_WEIGHTS_FILE, clip_rev, *where), clip
        ),
        cached_model((COMMAND, repo, mmaudio_sync_model.SYNC_FILE, rev, *where), sync),
        cached_model(
            (COMMAND, k6_repo, "audio_vae+vocoder", k6_rev, str(device)), decoder
        ),
    )


# --------------------------------------------------------------- warnings


def _warn(waveform, sample_rate, duration):
    from .. import dsp
    from ..events import emit_warning

    low, high = TRAINED_DURATION
    if not low <= duration <= high:
        emit_warning(
            f"{COMMAND}: duration {duration:g} s is outside the {low:g}-{high:g} s "
            "MMAudio was trained around; the effects may drift or thin out - "
            "generate 8 s pieces for a longer clip",
            command=COMMAND,
            duration=duration,
        )
    if not numpy.any(waveform):
        return
    flatness = dsp.spectral_flatness(waveform)
    centroid = dsp.spectral_centroid(waveform, sample_rate)
    if flatness >= NOISE_FLATNESS and centroid >= NOISE_CENTROID_HZ:
        emit_warning(
            f"{COMMAND}: the output measures noise-like (spectral flatness "
            f"{flatness:.2f}, centroid {centroid / 1000:.1f} kHz) - often hiss "
            "rather than effects. Name the on-screen sound sources in 'prompt' "
            "(video-only is the weaker mode), or try another seed",
            command=COMMAND,
            flatness=flatness,
            centroid_hz=centroid,
        )


# ------------------------------------------------------------------- task


def generate_sound_effects(
    video=None,
    prompt=None,
    negative_prompt=None,
    duration=None,
    steps=DEFAULT_STEPS,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    device="cpu",
    seed=None,
    **kwargs,
):
    """Sound effects for a clip, a prompt, or both, as a 44.1 kHz mono
    AudioTrack.

    Args:
        video: The clip to score (a stored video or an earlier step's).
            Video-only is the weaker mode; a prompt naming the on-screen
            sound sources is recommended alongside it
        prompt: What should be heard
        negative_prompt: What should not (replaces the empty condition)
        duration: Seconds of audio, at most 20 and at most the clip's
            length (default: the clip's length, or 8 s text-only)
        steps: Euler flow-matching steps (default 25)
        guidance_scale: Classifier-free guidance strength (default 4.5)
        device: Where to run
        seed: The step/workflow seed, for a reproducible take
    """
    _refuse_remote_code(kwargs)
    _refuse_unknown(kwargs)
    prompt = _text("prompt", prompt)
    negative_prompt = _text("negative_prompt", negative_prompt)
    steps = _number("steps", steps, whole=True)
    guidance_scale = float(_number("guidance_scale", guidance_scale))
    if video is None and prompt is None:
        raise ValueError(
            f"{COMMAND} needs 'video' or 'prompt' (or both): there is nothing "
            "to make sound effects for"
        )

    path, temporary = (None, False) if video is None else _video_file(video)
    try:
        clip_seconds = None if path is None else _clip_seconds(path)
        duration = _check_duration(duration, clip_seconds)
        frames = None if path is None else _frame_tensors(path, duration)
    finally:
        if temporary:
            os.unlink(path)
    if frames is not None:
        duration = frames[2]
    return _generate(
        frames, prompt, negative_prompt, duration, steps, guidance_scale, device, seed
    )


def _generate(
    frames, prompt, negative_prompt, duration, steps, guidance_scale, device, seed
):
    from .mmaudio_clip import tokenize
    from .mmaudio_model import sample_latent
    from .mmaudio_sync_model import encode_sync_frames

    dtype = _dtype(device)
    net, (clip, tokenizer), sync, decoder = _load_models(device, dtype)
    seq = net.set_duration(duration)

    clip_f = sync_f = text_f = negative_f = None
    if frames is not None:
        clip_frames, sync_frames, _ = frames
        clip_f = clip.encode_frames(clip_frames)[:, : net.clip_seq_len]
        param = next(sync.parameters())
        sync_frames = sync_frames.to(param.device, param.dtype)
        sync_f = encode_sync_frames(sync, sync_frames)
        sync_f = sync_f[:, : net.sync_seq_len]
    if prompt is not None:
        text_f = clip.encode_text(tokenize(tokenizer, [prompt]))
    if negative_prompt is not None:
        negative_f = clip.encode_text(tokenize(tokenizer, [negative_prompt]))

    generator = None
    if seed is not None:
        generator = torch.Generator(device=net.device).manual_seed(int(seed))
    logger.info(
        f"{COMMAND}: {duration:.2f} s, {steps} steps, guidance {guidance_scale:g}"
        f"{', video' if frames is not None else ''}{', prompt' if prompt else ''}"
    )
    latent = sample_latent(
        net,
        clip_features=clip_f,
        sync_features=sync_f,
        text_features=text_f,
        negative_text_features=negative_f,
        steps=steps,
        cfg_strength=guidance_scale,
        generator=generator,
    )
    waveform = decoder.decode(latent.transpose(1, 2).float())
    samples = min(seq.num_audio_frames, round(duration * decoder.sample_rate))
    audio = waveform[0, :, :samples].float().cpu().numpy()
    _warn(audio, decoder.sample_rate, duration)
    return AudioTrack(audio, decoder.sample_rate)
