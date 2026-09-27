"""Which reference voice sings each line of a song, by timbre (#485).

A generated song's section-to-singer map is unknown - MiniMax Music3 ignores
per-section singer directions - and staging lip-sync shots needs it. Pitch
cannot tell a tenor from a mezzo in the range they share: a pitch heuristic
called a tenor female and six chorus shots were re-rendered for it
(2026-09-25). What worked was a speaker embedding of each line's isolated
vocal compared against a reference span per singer, and that is this task:

1. the vocal stem is separated from the accompaniment with htdemucs (the
   accompaniment otherwise dominates a full mix's embedding) - `separate:
   false` skips it for a stem that is already dry;
2. every line, and every voice's reference, is reduced to its voiced frames
   and embedded with speechbrain's ECAPA speaker encoder;
3. each line scores the cosine against each voice, and lines roll up into
   named windows (shots, say) by the voiced seconds they overlap.

It decides nothing. Like the assessment probes it answers measurements - the
scores, the margin, how much of a line was voiced - and names the thresholds
it compared them to, so a caller can see a weak answer as weak. Every
threshold is a constant below, named in docs/TASKS.md.

Both models are fixed: nothing the caller writes reaches a model loader.
"""

import logging
import math

import numpy
import torch

from ..events import emit_warning
from ..for_each import MEMBER_SEPARATOR, render_path
from ..security import InvalidInputError, validate_variable_name
from ..task_domains import check_arguments
from .audio_utils import _waveform_and_rate, load_audio, resample_waveform
from .model_cache import cached_model

logger = logging.getLogger("dw")

COMMAND = "attribute_voices"

# The separator: Hybrid Transformer Demucs, four stems of which only
# 'vocals' is kept - the other three sum to the accompaniment it discards
_SEPARATOR_MODEL = "htdemucs"
# The speaker encoder: ECAPA-TDNN trained on VoxCeleb. speechbrain is already
# a dependency (generate_speech's speaker_embedding), so this adds none
_EMBEDDER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
# What the encoder was trained on - resampling to it is part of the embedding
_EMBEDDER_SAMPLE_RATE = 16000

# Voiced-frame detection on the (separated) vocal stem: 20 ms frames, voiced
# when the frame's rms reaches a floor set from the stem's own level - its
# VOICED_LEVEL_PERCENTILE frame rms - less VOICED_FLOOR_BELOW_LEVEL_DB, and
# never below VOICED_FLOOR_MIN_DBFS. Relative, because a song's sections sit
# at very different levels: an absolute -40 dBFS floor threw away a quiet
# verse that was clearly sung while its loud chorus sat 20 dB above it
# (#494). Separation leaves a residue under an instrumental passage well
# below the floor either way
FRAME_SECONDS = 0.02
VOICED_LEVEL_PERCENTILE = 95.0
VOICED_FLOOR_BELOW_LEVEL_DB = 35.0
VOICED_FLOOR_MIN_DBFS = -60.0
# Below this many voiced seconds a line has too little voice to embed, and
# its `voice` is null with the reason; a window below it has no voice either
MIN_VOICED_SECONDS = 0.5
# Default least total length of a voice's reference, in seconds
MIN_REFERENCE_SECONDS = 3.0
# A line whose best score beats the runner-up by less than this is
# `uncertain` - the argmax is still reported, but it is a coin toss
UNCERTAIN_MARGIN = 0.05
# A window whose leading voice's share beats the runner-up's by less than
# this is `uncertain` - two singers split it (a duet line, or a window
# straddling a hand-over)
UNCERTAIN_SHARE_MARGIN = 0.2
# Two references more alike than this make every answer between them weak,
# whatever the scores say: `voices_too_similar`
VOICES_TOO_SIMILAR = 0.8
# How far past the end of the audio a line or window may reach and be
# clipped rather than refused - a transcript's last chunk commonly ends a
# hair past the file
END_TOLERANCE_SECONDS = 0.1
# Default length of the fixed windows the song is cut into without `lines`
WINDOW_SECONDS = 2.0

THRESHOLDS = {
    "voiced_level_percentile": VOICED_LEVEL_PERCENTILE,
    "voiced_floor_below_level_db": VOICED_FLOOR_BELOW_LEVEL_DB,
    "voiced_floor_min_dbfs": VOICED_FLOOR_MIN_DBFS,
    "min_voiced_seconds": MIN_VOICED_SECONDS,
    "uncertain_margin": UNCERTAIN_MARGIN,
    "uncertain_share_margin": UNCERTAIN_SHARE_MARGIN,
    "voices_too_similar": VOICES_TOO_SIMILAR,
}


def _missing(package, error):
    return RuntimeError(
        f"{COMMAND} needs the '{package}' package, which is not installed "
        f"({error}). It is a dependency of diffusers-workflow: reinstall with "
        f"`pip install -e .` or `pip install {package}`"
    )


def _round(value, places=4):
    return None if value is None else round(float(value), places)


# --- arguments -------------------------------------------------------------


def _span(entry, where, duration):
    """A {start, end} span in seconds from either accepted shape.

    `{start, end}` is transcribe_audio's chunk shape (#483), so its output
    drops in as `lines`; `{start_seconds, duration_seconds}` is slice_audio's.
    """
    if not isinstance(entry, dict):
        raise ValueError(
            f"{COMMAND}: {where} must be an object with 'start'/'end' or "
            f"'start_seconds'/'duration_seconds', got {entry!r}"
        )
    has_end = "start" in entry or "end" in entry
    has_duration = "start_seconds" in entry or "duration_seconds" in entry
    if has_end == has_duration:
        raise ValueError(
            f"{COMMAND}: {where} needs exactly one of 'start'/'end' or "
            f"'start_seconds'/'duration_seconds', got keys {sorted(entry)}"
        )
    try:
        if has_end:
            start, end = float(entry["start"]), float(entry["end"])
        else:
            start = float(entry["start_seconds"])
            end = start + float(entry["duration_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{COMMAND}: {where} needs both numbers of its shape ({error})"
        ) from error
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError(f"{COMMAND}: {where} is not a finite span")
    if start < 0:
        raise ValueError(f"{COMMAND}: {where} starts before the audio ({start} s)")
    if end <= start:
        raise ValueError(f"{COMMAND}: {where} ends at {end} s, not after its start")
    if duration is not None and start >= duration:
        raise ValueError(
            f"{COMMAND}: {where} starts at {start:.3f} s, past the end of the "
            f"{duration:.3f} s audio"
        )
    if duration is not None and end > duration + END_TOLERANCE_SECONDS:
        raise ValueError(
            f"{COMMAND}: {where} ends at {end:.3f} s, past the end of the "
            f"{duration:.3f} s audio"
        )
    return start, min(end, duration) if duration is not None else end


def parse_voices(voices, duration, min_reference_seconds, clip_duration=None):
    """The voices argument as {name: spans | clip path}, checked.

    A reference is a list of spans into the audio itself (one span may be
    given bare), or a path to a separate clip. Refuses fewer than two
    voices, a name outside validate_variable_name's pattern, a span outside
    the audio, and a reference shorter than min_reference_seconds - naming
    the voice. clip_duration(path) measures a clip; it is only called for one.
    """
    if not isinstance(voices, dict):
        raise ValueError(
            f"{COMMAND}: 'voices' must map each voice's name to its reference, "
            f"got {type(voices).__name__}"
        )
    if len(voices) < 2:
        raise ValueError(
            f"{COMMAND} needs at least 2 voices to choose between, got "
            f"{len(voices)} ({', '.join(voices) or 'none'})"
        )
    parsed = {}
    for name, reference in voices.items():
        try:
            validate_variable_name(name)
        except InvalidInputError as error:
            raise ValueError(
                f"{COMMAND}: voice name {name!r} is not allowed - letters, "
                f"digits, '_' and '-', starting with a letter or '_' ({error})"
            ) from error
        if isinstance(reference, str):
            length = clip_duration(reference) if clip_duration else None
            parsed[name] = reference
        else:
            spans = [reference] if isinstance(reference, dict) else reference
            if not isinstance(spans, list) or not spans:
                raise ValueError(
                    f"{COMMAND}: voice '{name}' needs a clip path or a list of "
                    f"spans into the audio, got {reference!r}"
                )
            spans = [
                _span(span, f"voice '{name}' span {index}", duration)
                for index, span in enumerate(spans)
            ]
            length = sum(end - start for start, end in spans)
            parsed[name] = spans
        if length is not None and length < min_reference_seconds:
            raise ValueError(
                f"{COMMAND}: voice '{name}' has {length:.2f} s of reference, "
                f"under the {min_reference_seconds:g} s minimum "
                "(min_reference_seconds) - give it a longer span or more of them"
            )
    return parsed


def voices_errors(workflow_definition, source_indices=None):
    """Every attribute_voices step whose literal `voices` parse_voices would
    refuse, as [{path, message}] - fewer than two, a bad name, a malformed
    span, a reference under min_reference_seconds.

    The definition is substituted and expanded, so a `voices` a caller
    passed is checked as it will run. What needs the audio - a span past its
    end, a clip's length - stays the run's: the duration is not known here.
    A `voices` still spelled as a reference, or with a span holding one, is
    left to the run too.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []
    errors = []
    for index, step in enumerate(steps):
        task = step.get("task") if isinstance(step, dict) else None
        if not isinstance(task, dict) or task.get("command") != COMMAND:
            continue
        arguments = task.get("arguments")
        if not isinstance(arguments, dict) or "voices" not in arguments:
            continue
        voices = arguments["voices"]
        if isinstance(voices, str) or _holds_reference(voices):
            continue
        minimum = arguments.get("min_reference_seconds", MIN_REFERENCE_SECONDS)
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
            minimum = MIN_REFERENCE_SECONDS
        try:
            parse_voices(voices, None, minimum)
        except ValueError as error:
            source = (
                source_indices[index]
                if source_indices is not None and index < len(source_indices)
                else index
            )
            name = step.get("name")
            where = (
                f" in member '{name}'"
                if isinstance(name, str) and MEMBER_SEPARATOR in name
                else ""
            )
            path = ("steps", source, "task", "arguments", "voices")
            errors.append({"path": render_path(path), "message": f"{error}{where}"})
    return errors


def _holds_reference(voices):
    """Whether any span in `voices` carries a string - a reference the run
    resolves - so its numbers are not known yet. A clip path in place of a
    voice's spans is not a span and does not count."""
    if not isinstance(voices, dict):
        return False
    for reference in voices.values():
        spans = [reference] if isinstance(reference, dict) else reference
        if not isinstance(spans, list):
            continue
        for span in spans:
            values = span.values() if isinstance(span, dict) else [span]
            if any(isinstance(value, str) for value in values):
                return True
    return False


def parse_lines(lines, duration, window_seconds):
    """The lines to attribute, as [{start, end, text}].

    Without `lines`, the song is cut into fixed windows of window_seconds,
    the last one whatever is left.
    """
    if lines is None:
        count = max(1, math.ceil(duration / window_seconds - 1e-9))
        return [
            {
                "start": index * window_seconds,
                "end": min((index + 1) * window_seconds, duration),
                "text": None,
            }
            for index in range(count)
        ]
    if isinstance(lines, dict) and isinstance(lines.get("chunks"), list):
        # A whole transcript rather than its chunk list
        lines = lines["chunks"]
    if not isinstance(lines, list) or not lines:
        raise ValueError(f"{COMMAND}: 'lines' must be a non-empty list of spans")
    parsed = []
    for index, line in enumerate(lines):
        start, end = _span(line, f"lines[{index}]", duration)
        parsed.append({"start": start, "end": end, "text": line.get("text")})
    return parsed


def parse_windows(windows, duration):
    """The named windows lines roll up into, as [{name, start, end}]."""
    if not isinstance(windows, list) or not windows:
        raise ValueError(f"{COMMAND}: 'windows' must be a non-empty list")
    parsed = []
    for index, window in enumerate(windows):
        start, end = _span(window, f"windows[{index}]", duration)
        name = window.get("name")
        parsed.append(
            {
                "name": str(name) if name is not None else str(index),
                "start": start,
                "end": end,
            }
        )
    return parsed


# --- measurement -----------------------------------------------------------


def _dbfs(amplitude):
    return 20 * math.log10(amplitude) if amplitude > 0 else -math.inf


def voiced_floor_dbfs(rms):
    """The voiced floor for a stem whose frames have these rms values: its
    VOICED_LEVEL_PERCENTILE level less VOICED_FLOOR_BELOW_LEVEL_DB, never
    below VOICED_FLOOR_MIN_DBFS."""
    if len(rms) == 0:
        return VOICED_FLOOR_MIN_DBFS
    level = _dbfs(float(numpy.percentile(rms, VOICED_LEVEL_PERCENTILE)))
    return max(VOICED_FLOOR_MIN_DBFS, level - VOICED_FLOOR_BELOW_LEVEL_DB)


def voiced_mask(waveform, sample_rate):
    """Per-frame voiced flags for a mono waveform, the frame length in
    samples and the floor used: FRAME_SECONDS frames whose rms reaches
    voiced_floor_dbfs."""
    frame = max(1, int(round(FRAME_SECONDS * sample_rate)))
    count = len(waveform) // frame
    if count == 0:
        return numpy.zeros(0, dtype=bool), frame, VOICED_FLOOR_MIN_DBFS
    frames = waveform[: count * frame].reshape(count, frame)
    rms = numpy.sqrt(numpy.mean(frames.astype(numpy.float64) ** 2, axis=1))
    floor_dbfs = voiced_floor_dbfs(rms)
    return rms >= 10 ** (floor_dbfs / 20), frame, floor_dbfs


class _Voicing:
    """The voiced frames of the vocal stem, answering spans in seconds."""

    def __init__(self, waveform, sample_rate):
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.mask, self.frame, self.floor_dbfs = voiced_mask(waveform, sample_rate)
        self.frame_seconds = self.frame / sample_rate

    def _frames(self, start, end):
        first = int(math.floor(start / self.frame_seconds + 1e-9))
        last = int(math.ceil(end / self.frame_seconds - 1e-9))
        return max(0, first), min(len(self.mask), last)

    def seconds(self, start, end):
        first, last = self._frames(start, end)
        return float(self.mask[first:last].sum()) * self.frame_seconds

    def samples(self, spans):
        """The voiced samples of every span, concatenated."""
        pieces = []
        for start, end in spans:
            first, last = self._frames(start, end)
            for index in numpy.nonzero(self.mask[first:last])[0] + first:
                pieces.append(
                    self.waveform[index * self.frame : (index + 1) * self.frame]
                )
        if not pieces:
            return numpy.zeros(0, dtype=numpy.float32)
        return numpy.concatenate(pieces)


def cosine(a, b):
    a = numpy.asarray(a, dtype=numpy.float64)
    b = numpy.asarray(b, dtype=numpy.float64)
    denominator = numpy.linalg.norm(a) * numpy.linalg.norm(b)
    return float(a @ b / denominator) if denominator else 0.0


def reference_similarity(embeddings):
    """Pairwise cosine between the voices' references, and the pairs above
    VOICES_TOO_SIMILAR."""
    names = list(embeddings)
    pairs = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            value = cosine(embeddings[first], embeddings[second])
            pairs.append(
                {
                    "voices": [first, second],
                    "cosine": _round(value),
                    "too_similar": value > VOICES_TOO_SIMILAR,
                }
            )
    return pairs


def score_line(
    embedding, references, voiced_seconds, too_similar_pairs=(), separated=True
):
    """One line's scores against every reference, its argmax and margin.

    A line under MIN_VOICED_SECONDS is not embedded (embedding None) and has
    no voice. A line whose best two voices are a too-similar pair is
    uncertain whatever its margin. `separated` only words the reason: a
    stem that was never separated is not "after separation".
    """
    if embedding is None or voiced_seconds < MIN_VOICED_SECONDS:
        return {
            "scores": {},
            "voice": None,
            "margin": None,
            "voiced_seconds": _round(voiced_seconds, 3),
            "uncertain": True,
            "reason": (
                f"only {voiced_seconds:.2f} s voiced"
                f"{' after separation' if separated else ''} "
                f"(needs {MIN_VOICED_SECONDS} s) - instrumental, or too short "
                "to embed"
            ),
        }
    scores = {name: cosine(embedding, ref) for name, ref in references.items()}
    ranked = sorted(scores, key=scores.get, reverse=True)
    margin = scores[ranked[0]] - scores[ranked[1]]
    reason = None
    if frozenset(ranked[:2]) in {frozenset(pair) for pair in too_similar_pairs}:
        reason = (
            f"'{ranked[0]}' and '{ranked[1]}' have references too similar to "
            "tell apart (voices_too_similar)"
        )
    elif margin < UNCERTAIN_MARGIN:
        reason = f"margin {margin:.3f} is under {UNCERTAIN_MARGIN}"
    return {
        "scores": {name: _round(value) for name, value in scores.items()},
        "voice": ranked[0],
        "margin": _round(margin),
        "voiced_seconds": _round(voiced_seconds, 3),
        "uncertain": reason is not None,
        "reason": reason,
    }


def roll_up(windows, lines, voiced_seconds, voices):
    """Each window's share of voiced time by voice.

    A line counts toward a window by the voiced seconds of the stretch the
    two overlap, credited to the line's voice; a line with no voice counts
    for nothing. voiced_seconds(start, end) measures a stretch.
    """
    rolled = []
    for window in windows:
        weights = dict.fromkeys(voices, 0.0)
        for line in lines:
            if line["voice"] is None:
                continue
            start = max(window["start"], line["start"])
            end = min(window["end"], line["end"])
            if end > start:
                weights[line["voice"]] += voiced_seconds(start, end)
        total = sum(weights.values())
        entry = {
            "name": window["name"],
            "start": _round(window["start"], 3),
            "end": _round(window["end"], 3),
            "voiced_seconds": _round(total, 3),
        }
        if total < MIN_VOICED_SECONDS:
            entry.update(
                share={},
                voice=None,
                uncertain=True,
                reason=(
                    f"only {total:.2f} s of attributed voice in the window "
                    f"(needs {MIN_VOICED_SECONDS} s)"
                ),
            )
        else:
            share = {name: weight / total for name, weight in weights.items()}
            ranked = sorted(share, key=share.get, reverse=True)
            lead = share[ranked[0]] - share[ranked[1]]
            uncertain = lead < UNCERTAIN_SHARE_MARGIN
            entry.update(
                share={name: _round(value) for name, value in share.items()},
                voice=ranked[0],
                uncertain=uncertain,
                reason=(
                    f"'{ranked[0]}' leads '{ranked[1]}' by {lead:.2f} of the "
                    f"window, under {UNCERTAIN_SHARE_MARGIN}"
                    if uncertain
                    else None
                ),
            )
        rolled.append(entry)
    return rolled


# --- models ----------------------------------------------------------------


def _load_separator(device, dtype):
    try:
        from demucs.pretrained import get_model
    except ImportError as error:
        raise _missing("demucs", error) from error

    def load():
        model = get_model(_SEPARATOR_MODEL)
        model.eval()
        return model.to(device=device, dtype=dtype)

    return cached_model((COMMAND, _SEPARATOR_MODEL, str(device), str(dtype)), load)


def _load_embedder(device, dtype):
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as error:
        raise _missing("speechbrain", error) from error

    def load():
        return EncoderClassifier.from_hparams(
            source=_EMBEDDER_MODEL, run_opts={"device": str(device)}
        )

    return cached_model((COMMAND, _EMBEDDER_MODEL, str(device), str(dtype)), load)


def _run_separator(waveform, sample_rate, device, dtype):
    from demucs.apply import apply_model

    model = _load_separator(device, dtype)
    rate = model.samplerate
    mix = resample_waveform(waveform, sample_rate, rate)
    if mix.shape[0] == 1:
        mix = numpy.repeat(mix, model.audio_channels, axis=0)
    mix = torch.as_tensor(mix[: model.audio_channels], dtype=dtype)
    # demucs' own separate.py normalizes the mix and undoes it on the stems
    reference = mix.mean(0)
    mean, std = reference.mean(), reference.std()
    std = std if float(std) > 0 else torch.tensor(1.0)
    with torch.no_grad():
        stems = apply_model(
            model, ((mix - mean) / std)[None], device=device, split=True
        )[0]
    vocals = stems[model.sources.index("vocals")] * std + mean
    return vocals.float().cpu().numpy(), rate


def separate_vocals(waveform, sample_rate, device, dtype):
    """The vocal stem of a (channels, samples) mix, as a mono 16 kHz array.

    demucs on MPS is unverified; if it fails there, separation is retried on
    the CPU with a warning rather than failing the step.
    """
    from .. import get_device_type

    try:
        vocals, rate = _run_separator(waveform, sample_rate, device, dtype)
    except Exception as error:
        if get_device_type(device) != "mps":
            raise
        emit_warning(
            f"{COMMAND}: htdemucs failed on {device} ({error}); separating on "
            "the CPU instead, which is slower",
            kind="separation_cpu_fallback",
            command=COMMAND,
        )
        vocals, rate = _run_separator(waveform, sample_rate, "cpu", torch.float32)
    return _mono_16k(vocals, rate)


def _mono_16k(waveform, sample_rate):
    waveform = resample_waveform(
        numpy.ascontiguousarray(waveform, dtype=numpy.float32),
        sample_rate,
        _EMBEDDER_SAMPLE_RATE,
    )
    return waveform.mean(axis=0).astype(numpy.float32)


def embed(encoder, samples):
    """An ECAPA embedding of mono 16 kHz samples, as a unit numpy vector."""
    with torch.no_grad():
        embedding = encoder.encode_batch(
            torch.as_tensor(samples, dtype=torch.float32).unsqueeze(0)
        )
    vector = embedding.detach().float().cpu().numpy().reshape(-1)
    norm = numpy.linalg.norm(vector)
    return vector / norm if norm else vector


# --- the task --------------------------------------------------------------


def attribute_voices(
    audio,
    voices,
    lines=None,
    windows=None,
    window_seconds=WINDOW_SECONDS,
    min_reference_seconds=MIN_REFERENCE_SECONDS,
    separate=True,
    device="cpu",
):
    """Say which reference voice sings each line of a song, by timbre.

    Separates the vocal stem (htdemucs), embeds each line's voiced audio and
    each voice's reference with speechbrain's ECAPA speaker encoder, and
    scores every line against every voice by cosine. Decides nothing: it
    reports scores, the margin between the best two, and how much of each
    line was voiced, and flags a weak answer as `uncertain`. Pitch is not
    used - it cannot separate a tenor from a mezzo in their shared range.

    Args:
        audio: The song - a path, 'asset:'/'output:' reference, or an
            earlier step's audio or video.
        voices: Each voice's name mapped to its reference: a list of
            {start_seconds, duration_seconds} (or {start, end}) spans into
            'audio' itself, or a path/'asset:' of a separate clip of that
            voice. At least 2 voices; names follow the variable-name pattern
            (letters, digits, '_', '-').
        lines: The spans to attribute, in seconds - a list of {start, end,
            text?} (transcribe_audio's chunk shape) or {start_seconds,
            duration_seconds, text?}. Omitted: the song is cut into fixed
            windows of window_seconds.
        windows: Named spans to roll the lines up into, e.g. shots - a list
            of {name, start, end}. Each reports every voice's share of its
            voiced time. Omitted: the fixed windows when 'lines' is omitted
            too, else none.
        window_seconds: Length of the fixed windows used without 'lines'.
            Default 2.0.
        min_reference_seconds: Least total reference length per voice; a
            voice under it is refused by name. Default 3.0.
        separate: Isolate the vocal stem with htdemucs before embedding
            (default true). False for audio that is already a dry vocal.
        device: Where to run the models.

    Returns:
        A JSON document: 'voices', 'separated', 'lines' (start, end, text,
        scores, voice, margin, voiced_seconds, uncertain, reason), 'windows'
        (name, start, end, share, voice, uncertain, reason),
        'reference_similarity' (pairwise cosine between references),
        'warnings' ('voices_too_similar') and the 'thresholds' used.
    """
    check_arguments(
        COMMAND,
        window_seconds=window_seconds,
        min_reference_seconds=min_reference_seconds,
    )
    waveform, sample_rate = _waveform_and_rate(audio, None, COMMAND)
    duration = waveform.shape[1] / sample_rate

    clips = {}

    def clip_duration(path):
        clip, rate = load_audio(path)
        clips[path] = (clip, rate)
        return clip.shape[1] / rate

    references = parse_voices(voices, duration, min_reference_seconds, clip_duration)
    parsed_lines = parse_lines(lines, duration, window_seconds)
    parsed_windows = None
    if windows is not None:
        parsed_windows = parse_windows(windows, duration)
    elif lines is None:
        parsed_windows = [
            {"name": f"window-{index + 1}", "start": line["start"], "end": line["end"]}
            for index, line in enumerate(parsed_lines)
        ]

    # Both models' STFT front ends are fp32-only in practice (cuFFT's half
    # path needs a power-of-two size, and ECAPA's 400-sample window is not),
    # and both are small, so they run in fp32 on every device
    dtype = torch.float32
    separate = bool(separate)

    def stem(mix, rate):
        if separate:
            return separate_vocals(mix, rate, device, dtype)
        return _mono_16k(mix, rate)

    song = _Voicing(stem(waveform, sample_rate), _EMBEDDER_SAMPLE_RATE)
    encoder = _load_embedder(device, dtype)

    embeddings = {}
    for name, reference in references.items():
        if isinstance(reference, str):
            clip = _Voicing(stem(*clips[reference]), _EMBEDDER_SAMPLE_RATE)
            samples = clip.samples([(0.0, len(clip.waveform) / clip.sample_rate)])
        else:
            samples = song.samples(reference)
        voiced = len(samples) / _EMBEDDER_SAMPLE_RATE
        if voiced < MIN_VOICED_SECONDS:
            raise ValueError(
                f"{COMMAND}: voice '{name}''s reference holds only {voiced:.2f} s "
                f"of voice{' after separation' if separate else ''} (needs "
                f"{MIN_VOICED_SECONDS} s) - point it at a stretch where that "
                "voice sings"
            )
        embeddings[name] = embed(encoder, samples)

    similarity = reference_similarity(embeddings)
    too_similar = [pair["voices"] for pair in similarity if pair["too_similar"]]
    warnings = []
    for pair in similarity:
        if pair["too_similar"]:
            message = (
                f"{COMMAND}: the references for '{pair['voices'][0]}' and "
                f"'{pair['voices'][1]}' score {pair['cosine']} against each "
                f"other, above {VOICES_TOO_SIMILAR} - any line between them "
                "is a weak answer whatever its scores say"
            )
            warnings.append(
                {
                    "kind": "voices_too_similar",
                    "voices": pair["voices"],
                    "cosine": pair["cosine"],
                    "threshold": VOICES_TOO_SIMILAR,
                    "message": message,
                }
            )
            emit_warning(
                message,
                kind="voices_too_similar",
                command=COMMAND,
                voices=pair["voices"],
                cosine=pair["cosine"],
            )

    attributed = []
    for line in parsed_lines:
        voiced = song.seconds(line["start"], line["end"])
        embedding = None
        if voiced >= MIN_VOICED_SECONDS:
            embedding = embed(encoder, song.samples([(line["start"], line["end"])]))
        attributed.append(
            {
                "start": _round(line["start"], 3),
                "end": _round(line["end"], 3),
                "text": line["text"],
                **score_line(
                    embedding, embeddings, voiced, too_similar, separated=separate
                ),
            }
        )

    rolled = []
    if parsed_windows is not None:
        rolled = roll_up(
            parsed_windows,
            [
                {"start": line["start"], "end": line["end"], "voice": result["voice"]}
                for line, result in zip(parsed_lines, attributed)
            ],
            song.seconds,
            list(references),
        )

    return {
        "voices": list(references),
        "separated": separate,
        "duration_seconds": _round(duration, 3),
        "voiced_floor_dbfs": _round(song.floor_dbfs, 2),
        "lines": attributed,
        "windows": rolled,
        "reference_similarity": similarity,
        "warnings": warnings,
        "thresholds": dict(THRESHOLDS),
    }
