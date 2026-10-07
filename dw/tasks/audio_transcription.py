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
from ..events import emit_warning
from .model_cache import cached_model, hf_pipeline_placement
from ..dsp import resample_waveform
from .audio_utils import waveform_and_rate

logger = logging.getLogger("dw")

_DEFAULT_ASR_MODEL = "openai/whisper-base"
_ASR_SAMPLE_RATE = 16000
# Public: dw/scalar_result_validation.py checks a literal `timestamps`
# argument against this same tuple to catch a `result.content_type` that
# does not match the {text, chunks} dict shape timestamps switches the
# return value to (#498).
TIMESTAMP_KINDS = ("segment", "word")


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
            timestamps: Unset (default) returns plain text. "segment" or
                "word" instead returns a dict of {text, chunks}, chunks
                being a list of {start, end, text}, every time a number of
                seconds (a chunk the clip ends inside ends at the clip's
                duration). Word bounds are Whisper's attention timestamps
                tightened to the waveform's energy, so a word does not
                absorb the silence before, after or inside it (#661); a
                word with no audible energy keeps Whisper's bounds - the step's result
                content_type must then be "application/json" rather than
                "text/plain", since the return shape follows the argument.

    Returns:
        Transcribed text string, or (with timestamps set) a
        {text, chunks} dict of {start, end, text} chunks.
    """
    timestamps = kwargs.get("timestamps")
    if timestamps is not None and timestamps not in TIMESTAMP_KINDS:
        raise ValueError(
            f"timestamps must be one of {TIMESTAMP_KINDS}, got {timestamps!r}"
        )
    waveform, waveform_rate = waveform_and_rate(audio, sample_rate, "transcribe_audio")
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

    # Whisper takes 30 s per window and refuses a longer clip ("more than 3000
    # mel input features") unless it predicts timestamps, which is how its
    # long-form mode stitches windows. Without timestamps, Whisper's decoder
    # can also emit an early end-of-text after a pause between lines - a
    # multi-line clip with gaps between shots then transcribes only the first
    # line even well under 30 s (#559). Forcing return_timestamps on for every
    # Whisper call, plain-text included, avoids both: only "word"/"segment"
    # callers see the chunks, plain mode still returns just the full text.
    # Asked for only of Whisper: transformers raises for a CTC model unless
    # the value is "char" or "word", and for any other seq2seq model at all
    is_whisper = getattr(pipe, "type", None) == "seq2seq_whisper"
    options = {}
    if timestamps == "word":
        # "word" is accepted by both a seq2seq Whisper model and a CTC model
        options["return_timestamps"] = "word"
    elif timestamps == "segment":
        options["return_timestamps"] = True
    elif is_whisper:
        options["return_timestamps"] = True
    samples = mono.astype(numpy.float32)
    result = pipe({"raw": samples, "sampling_rate": _ASR_SAMPLE_RATE}, **options)
    result = _with_resumed_tail(pipe, samples, result, options)
    text = result["text"].strip()
    logger.info(f"Transcript: {text[:100]}{'...' if len(text) > 100 else ''}")
    if timestamps is None:
        return text

    duration = len(mono) / _ASR_SAMPLE_RATE
    chunks = _numeric_chunks(result.get("chunks", []), duration)
    if timestamps == "word":
        chunks = _trim_to_speech(chunks, mono)
    return {"text": text, "chunks": chunks}


def _numeric_chunks(raw_chunks, duration):
    """The {start, end, text} chunks, with every time a number of seconds.

    Whisper leaves a chunk's end None when the audio stops inside it - a song
    cut mid-line - and a None there failed every reader that does arithmetic
    on the span: attribute_voices refused the whole transcript (#488). Such a
    chunk ends at the clip's own duration, and a None start takes the
    previous chunk's end (0 for the first).
    """
    chunks = []
    previous_end = 0.0
    for chunk in raw_chunks:
        start, end = chunk.get("timestamp", (None, None))
        if start is None:
            start = previous_end
        if end is None:
            end = max(duration, start)
        chunks.append(
            {"start": start, "end": end, "text": chunk.get("text", "").strip()}
        )
        previous_end = end
    return chunks


_FRAME_SECONDS = 0.02
# Whisper's long-form decoding can stop at a mid-clip silence and drop every
# line after it with no error (#672). A last chunk ending more than this
# short of audible speech is treated as a stop, and the rest is decoded again.
_TAIL_GAP_SECONDS = 1.0
_MAX_TAIL_RESUMES = 3
# A frame is speech when its RMS is above this fraction of the clip's loud
# (95th percentile) frames, about -26 dB: clear of room tone, under soft speech.
_SPEECH_RELATIVE_LEVEL = 0.05


def _speech_runs(mono):
    """(first, last_exclusive) frame runs of audible speech."""
    frame = int(_FRAME_SECONDS * _ASR_SAMPLE_RATE)
    count = len(mono) // frame
    if count == 0:
        return []
    frames = mono[: count * frame].astype(numpy.float64).reshape(count, frame)
    rms = numpy.sqrt((frames**2).mean(axis=1))
    loud = float(numpy.percentile(rms, 95))
    if loud <= 0.0:
        return []
    active = rms > loud * _SPEECH_RELATIVE_LEVEL
    edges = numpy.diff(numpy.concatenate(([0], active.astype(numpy.int8), [0])))
    runs = zip(numpy.flatnonzero(edges == 1), numpy.flatnonzero(edges == -1))
    return [(int(lo), int(hi)) for lo, hi in runs]


def _chunk_ends(raw_chunks):
    """Each chunk's end; an open-ended one (audio stopped inside it) ends at infinity."""
    spans = (c.get("timestamp") or (None, None) for c in raw_chunks)
    return [float("inf") if end is None else end for _, end in spans]


def _with_resumed_tail(pipe, samples, result, options):
    """The transcription, with speech Whisper stopped short of decoded too.

    When the last chunk ends well before the last audible speech, decoding is
    restarted at the start of the speech run holding (or following) that end
    and its chunks appended, shifted to the clip's timeline. If speech is
    still left after the resumes, a warning says where the transcript stops.
    """
    raw_chunks = list(result.get("chunks") or [])
    runs = _speech_runs(samples)
    if not raw_chunks or not runs:
        return result
    texts = [result["text"].strip()]
    speech_end = runs[-1][1] * _FRAME_SECONDS
    for _ in range(_MAX_TAIL_RESUMES):
        ends = _chunk_ends(raw_chunks)
        if not ends or speech_end - max(ends) <= _TAIL_GAP_SECONDS:
            break
        end_frame = int(max(ends) / _FRAME_SECONDS)
        resume = next((lo for lo, hi in runs if hi > end_frame), None)
        if resume is None:
            break
        origin = resume * _FRAME_SECONDS
        tail = pipe(
            {
                "raw": samples[int(origin * _ASR_SAMPLE_RATE) :],
                "sampling_rate": _ASR_SAMPLE_RATE,
            },
            **options,
        )
        tail_text = tail["text"].strip()
        tail_chunks = tail.get("chunks") or []
        if not tail_text or not tail_chunks:
            break
        texts.append(tail_text)
        for chunk in tail_chunks:
            start, end = chunk.get("timestamp") or (None, None)
            raw_chunks.append(
                {
                    **chunk,
                    "timestamp": (
                        None if start is None else start + origin,
                        None if end is None else end + origin,
                    ),
                }
            )
    ends = _chunk_ends(raw_chunks)
    if ends and speech_end - max(ends) > _TAIL_GAP_SECONDS:
        emit_warning(
            f"transcribe_audio: transcript ends at {max(ends):.1f} s but audible "
            f"speech continues to {speech_end:.1f} s; the last line(s) may be "
            "missing from the text"
        )
    return {**result, "text": " ".join(texts), "chunks": raw_chunks}


def _trim_to_speech(chunks, mono):
    """Chunks with each span narrowed to the audible part of the waveform.

    Whisper's word timestamps come from cross-attention and absorb silence:
    the first word starts at 0 and the last ends at the clip's end however
    long the lead-in or tail, and a word after a pause swallows the pause
    (#661). Each chunk's start moves up to its first speech frame and its end
    back to its last, never outside its original span, so order is kept. A
    speech run lying mostly outside the span is a neighbour's, bled in; it
    is ignored unless the span has no run of its own, so a word whose span
    covers its neighbour's onset across a pause shrinks to its own sound. A
    chunk with no speech frame in it is left as Whisper gave it.
    """
    count = len(mono) // int(_FRAME_SECONDS * _ASR_SAMPLE_RATE)
    runs = _speech_runs(mono)
    if not runs:
        return chunks

    trimmed = []
    for chunk in chunks:
        first = int(chunk["start"] / _FRAME_SECONDS)
        last = min(count, int(numpy.ceil(chunk["end"] / _FRAME_SECONDS)))
        # Speech runs touching the span, clipped to it. A run mostly outside the
        # span belongs to a neighbouring word whose edge bled in; drop it when
        # the span holds a run of its own (the pause-then-word case).
        touching = [(lo, hi) for lo, hi in runs if lo < last and hi > first]
        own = [
            (max(lo, first), min(hi, last))
            for lo, hi in touching
            if (min(hi, last) - max(lo, first)) * 2 >= hi - lo
        ]
        spans = own or [(max(lo, first), min(hi, last)) for lo, hi in touching]
        if not spans:
            trimmed.append(chunk)
            continue
        start = max(chunk["start"], spans[0][0] * _FRAME_SECONDS)
        end = min(chunk["end"], spans[-1][1] * _FRAME_SECONDS)
        trimmed.append(
            {**chunk, "start": round(float(start), 3), "end": round(float(end), 3)}
        )
    return trimmed
