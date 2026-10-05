"""Turning what a pipeline or task returned into the list of artifacts a
`Result` saves: diffusers output objects, modular pipeline dicts, and the
pairing of generated frames with the audio generated beside them.
"""

import logging

import numpy
import torch

from .media_types import AudioTrack, AudioVideo
from .pipeline_processors.h3_blocks import HELD_AUDIO_OUTPUT, HELD_AUDIO_RATE_OUTPUT

logger = logging.getLogger("dw")


# The names a modular pipeline's outputs go by. Asked for more than one output it returns
# them in a dict rather than on a pipeline output object, so its videos and the soundtrack
# generated alongside them arrive keyed instead of as attributes. Every diffusers modular
# pipeline (minimax_h3, ltx2, ...) names these "videos"/"audio"/"sampling_rate" - kept as
# tuples, rather than plain strings, only so the lookup goes through first_item like the
# other key sets. "audio_sample_rate" is real too: it is the name dw's own
# attach_audio_sample_rate (pipeline_processors/pipeline.py) gives the rate when it
# attaches it to a non-modular output - a modular result carrying it under that name is
# tested (TestModularOutputs.test_audio_sample_rate_names_the_rate_too) and kept for it.
MODULAR_VIDEO_KEYS = ("videos",)


MODULAR_AUDIO_KEYS = ("audio",)


MODULAR_SAMPLE_RATE_KEYS = ("sampling_rate", "audio_sample_rate")


# A MiniMax-H3 step that held a soundtrack: the caller's own track, fitted to the
# video, which plays in place of the decoded audio (the VAE round trip of it)
HELD_AUDIO_KEYS = (HELD_AUDIO_OUTPUT,)
HELD_AUDIO_RATE_KEYS = (HELD_AUDIO_RATE_OUTPUT,)


def _frames_from_attributes(result):
    """The frames extractor for a pipeline output that carries `.frames` directly.

    Some video pipelines (LTX-2) generate an audio track along with the frames, exposed
    as `.audio` and `.audio_sample_rate` attributes alongside `.frames`.
    """
    return frames_with_audio(
        result.frames,
        getattr(result, "audio", None),
        getattr(result, "audio_sample_rate", None),
    )


def _audios_from_attribute(result):
    """Each `.audios` item as an artifact.

    With the rate attach_audio_sample_rate recorded, each item becomes an
    AudioTrack carrying it, shaped (channels, samples); without one, the bare
    (samples, channels) array it always was, and the workflow's 'sample_rate'
    (or the default) applies at save.
    """
    sample_rate = getattr(result, "audio_sample_rate", None)
    if sample_rate is None:
        return [as_waveform_array(audio) for audio in result.audios]
    return [
        AudioTrack(as_waveform_array(audio).T, int(sample_rate))
        for audio in result.audios
    ]


# Diffusers output fields get_artifact_list knows how to turn into artifacts, tried in
# this order. A result is dispatched to the first field it has - images wins over frames
# if a result somehow has both, matching the fixed hasattr chain this replaced. Supporting
# a new diffusers output field (e.g. a standalone "depth" attribute) is one more entry
# here, instead of another branch threaded through the chain.
OUTPUT_FIELD_EXTRACTORS = [
    ("images", lambda result: result.images),
    ("image_embeds", lambda result: result.image_embeds),
    ("image_embeddings", lambda result: result.image_embeddings),
    ("frames", _frames_from_attributes),
    ("audios", _audios_from_attribute),
]


def get_artifact_list(result):
    """Extract list of artifacts from a result object.

    Handles various result types including images, embeddings, frames, and audio.

    Args:
        result: Result object to extract artifacts from

    Returns:
        List of artifacts
    """
    # Already a paired video and audio track - it has a .frames attribute of its own,
    # but the pair is one artifact, not something to run back through frame extraction
    if isinstance(result, AudioVideo):
        return [result]

    for field_name, extract in OUTPUT_FIELD_EXTRACTORS:
        if hasattr(result, field_name):
            return extract(result)

    if isinstance(result, dict):
        # A modular pipeline asked for several outputs returns them keyed
        artifacts = modular_artifacts(result)
        if artifacts is not None:
            return artifacts

    if isinstance(result, list):
        return result

    if hasattr(result, "to_tuple") or hasattr(result, "__dataclass_fields__"):
        # A diffusers output (BaseOutput subclasses have to_tuple; plain dataclasses
        # have __dataclass_fields__) whose fields matched none of the extractors above -
        # log what it actually looks like so the resulting content-type mismatch on save
        # is diagnosable instead of a bare "does not match result type" surprise.
        logger.warning(
            f"Don't know how to extract artifacts from a {type(result).__name__} - "
            f"treating it as a single artifact. Its fields are: {output_field_names(result)}"
        )

    return [result]


def output_field_names(result):
    """Best-effort list of field names on a diffusers-style output object, for logging."""
    if hasattr(result, "keys"):
        return list(result.keys())
    if hasattr(result, "__dataclass_fields__"):
        return list(result.__dataclass_fields__.keys())
    return []


def modular_artifacts(result):
    """Extract the artifacts from the outputs a modular pipeline returns together.

    Asked for several outputs - `"output": ["videos", "audio", "sampling_rate"]` - a
    modular pipeline returns them in a dict instead of on one output object. Pairing the
    videos with the audio generated alongside them here saves them the same way a video
    pipeline's own output is saved, muxed into a single file. Any other requested output
    - "images" or "latents", say - is not part of that pairing, so it is carried along as
    one extra dict artifact, saved key by key the same way any other dictionary result is.

    Args:
        result: Dict of outputs returned by a modular pipeline

    Returns:
        List of artifacts, or None when the outputs hold no video - those are saved one
        output at a time instead
    """
    video_key, videos = first_item(result, MODULAR_VIDEO_KEYS)
    if videos is None:
        return None

    consumed_keys = {video_key}

    audio_key, audio = first_item(result, MODULAR_AUDIO_KEYS)
    sample_rate = None
    if audio is not None:
        consumed_keys.add(audio_key)
        rate_key, sample_rate = first_item(result, MODULAR_SAMPLE_RATE_KEYS)
        consumed_keys.add(rate_key)

    held_key, held = first_item(result, HELD_AUDIO_KEYS)
    if held is not None:
        rate_key, held_rate = first_item(result, HELD_AUDIO_RATE_KEYS)
        consumed_keys.update({held_key, rate_key})
        # One track, held under every video the call generated
        audio, sample_rate = [held[0]] * len(videos), held_rate
    consumed_keys.update(HELD_AUDIO_KEYS + HELD_AUDIO_RATE_KEYS)

    artifacts = frames_with_audio(videos, audio, sample_rate)

    # Keys the video/audio pairing above did not consume still need to be saved, not
    # dropped - carry them along as one extra artifact, saved key by key like any other
    # dictionary result
    leftovers = {
        key: value
        for key, value in result.items()
        if key not in consumed_keys and value is not None
    }
    if leftovers:
        artifacts = list(artifacts) + [leftovers]

    return artifacts


def first_item(values, keys):
    """The key and value of the first of `keys` present in `values`.

    Returns (None, None) when none of them are.
    """
    for key in keys:
        value = values.get(key, None)
        if value is not None:
            return key, value

    return None, None


def frames_with_audio(frames, audio, sample_rate):
    """Pair frames with the audio track generated alongside them, if there is one.

    The one place that decides whether frames need pairing with audio at all - used by
    both routes a pipeline's frames-plus-audio output can take: attributes on a pipeline
    output object (`_frames_from_attributes`), and keys in the dict a modular pipeline
    returns (`modular_artifacts`). Frames without audio are returned unchanged; actually
    pairing them is `pair_audio_with_frames`'s job.

    Args:
        frames: The generated video(s), one list of frames per generation
        audio: The generated waveform(s), or None if the pipeline produced no audio
        sample_rate: Sample rate of the waveform(s), or None if unknown

    Returns:
        `frames` unchanged if `audio` is None, otherwise the list of AudioVideo pairs
        `pair_audio_with_frames` produces
    """
    if audio is None:
        return frames
    return pair_audio_with_frames(frames, audio, sample_rate)


def pair_audio_with_frames(videos, audio, sample_rate):
    """Pair each generated video with its own audio track.

    Both are batched - videos[i] and audio[i] belong to the same generation. Only the
    pipeline knows the sample rate its vocoder produced the audio at, so it comes along
    rather than being guessed at here.

    Args:
        videos: The generated videos, one list of frames per generation
        audio: The generated waveforms, one per generation
        sample_rate: Sample rate of the waveforms, or None if the pipeline did not report one

    Returns:
        List of AudioVideo artifacts, one per generated video
    """
    return [
        AudioVideo(frames, audio[i] if i < len(audio) else None, sample_rate)
        for i, frames in enumerate(videos)
    ]


def as_waveform_array(audio):
    """Transpose one batch item of a pipeline's `.audios` output to (samples, channels).

    `.audios` is shaped (batch, channels, samples). Under the diffusers default
    `output_type='np'`, pipelines such as AudioLDM2 and StableAudio already call
    `.numpy()` before returning, so each item here is a numpy ndarray rather than a
    torch tensor - it has no `.float()`/`.cpu()` methods, only `.T`/`.astype()`.

    Args:
        audio: One batch item, shaped (channels, samples), as a torch tensor or numpy array

    Returns:
        Numpy float32 array shaped (samples, channels)
    """
    if isinstance(audio, torch.Tensor):
        return audio.T.float().cpu().numpy()

    return numpy.asarray(audio).T.astype(numpy.float32, copy=False)
