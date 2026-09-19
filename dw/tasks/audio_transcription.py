"""
Speech-to-text via HuggingFace transformers.

Runs a local automatic-speech-recognition model, so a word-correctness
defect in a TTS deliverable (a dropped line, a mid-sentence truncation) can
be checked against the text it was supposed to speak instead of inferred
from duration and timing arithmetic (#188).

A dedicated module, like text_generation.py, rather than living in
audio_utils.py alongside the other audio tasks: those tasks share that
module and lazily import transformers only when a task that needs it runs,
and a module-level import here would force that heavy import on every one
of them.
"""

import logging

import numpy
from transformers import pipeline as hf_pipeline

from .. import preferred_task_dtype
from .model_cache import cached_model, hf_pipeline_placement
from .audio_utils import _waveform_and_rate, resample_waveform

logger = logging.getLogger("dw")

_DEFAULT_ASR_MODEL = "openai/whisper-base"
_ASR_SAMPLE_RATE = 16000


def _downmixed_mono(waveform):
    """Average a (channels, samples) waveform down to one channel.

    Whisper-class models are trained on mono; every other audio task keeps
    multi-channel audio as far as it can, so this stays local to
    transcribe_audio rather than becoming a general helper.
    """
    if waveform.shape[0] == 1:
        return waveform[0]
    return waveform.mean(axis=0)


def transcribe_audio(audio, device="cpu", sample_rate=None, **kwargs):
    """Task command: transcribe spoken audio to text.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a soundtrack, or a
            waveform (which needs sample_rate alongside it)
        device: Target device ("cuda", "mps", "cpu").
        sample_rate: Sample rate of a waveform passed directly.
        **kwargs:
            model_name: HuggingFace model ID of a Whisper-class ASR model
                (default: openai/whisper-base).

    Returns:
        Transcribed text string.
    """
    waveform, waveform_rate = _waveform_and_rate(audio, sample_rate, "transcribe_audio")
    mono = _downmixed_mono(waveform)
    if waveform_rate != _ASR_SAMPLE_RATE:
        mono = resample_waveform(mono.reshape(1, -1), waveform_rate, _ASR_SAMPLE_RATE)[
            0
        ]

    model_name = kwargs.get("model_name", _DEFAULT_ASR_MODEL)
    dtype = preferred_task_dtype(device)

    def load_pipe():
        logger.info(f"Transcribing audio with {model_name} on {device}")
        placement = hf_pipeline_placement(device)
        return hf_pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=dtype,
            **placement,
        )

    pipe = cached_model(
        ("transcribe_audio", model_name, str(device), str(dtype)), load_pipe
    )

    result = pipe(
        {"raw": mono.astype(numpy.float32), "sampling_rate": _ASR_SAMPLE_RATE}
    )
    text = result["text"].strip()
    logger.info(f"Transcript: {text[:100]}{'...' if len(text) > 100 else ''}")
    return text
