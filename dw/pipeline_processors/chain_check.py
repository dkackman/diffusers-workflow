"""The per-segment check a chain runs on each segment before keeping it, and
the policy of its bounded re-roll.

After segment k renders (k > 1), its kept frames are measured against
segment k - 1's - the seam's picture jump and look step, by the arithmetic
`analyze_seams` uses (`dw/tasks/assess.py` `seam_measures`) - and against
segment 1's, for fine-band texture drift (`dw/texture_bands.py`) - and its
generated audio over the kept span is held to a level floor (#1221: a draw
can come out silent). A segment over any ceiling, or under the floor, fails,
naming the measures; `run_chain` re-renders it from the same carry with
`reroll_seed`'s seed up to the chain's `retries`, and keeps `best_attempt`
of what it rendered. Segment 1 has no seam before it, so `check_audio` holds
it to the audio floor alone. `check_segment` can skip a seam marked
`hard_cut`, but `run_chain` never passes it: a chain's own seams are never
marked cuts.

`check_words` is the dialogue measure (#1262): a segment's kept audio
transcribed and held to its line with check_script's comparison, after the
seam-head rule drops the previous line's carried words at its head. A
checked chain whose `check` sets `words_similarity_min` runs it on every
segment with a line, segment 1 included (`chain_config.segment_line`), and
`with_words` folds its answer into the segment's CheckResult as `words`, a
re-rolling measure. The default minimum is None, the measure off:
bench/records/words.md found no clean separation of good takes from bad.

`with_face_drop` folds in the identity measure, warn-only: `face_drop`,
segment 1's median SFace similarity to its own frame 0 (`face_base`, measured
once, from frame 1 on) minus this segment's to that same frame
(`face_drift.face_similarities`). Its ceiling `face_drop_max` defaults to
None: measured and recorded on the shot's `check`, never failed, until the
bench sets it. A model that cannot load leaves the measure None rather than
failing a chain. Lip sync is not measured here: no threshold separated one
segment's good audio from bad (ruling R9, bench/records/narrated.md), so
`measure_lip_sync` is a take-level task only.

Measurement and choice only: nothing here renders, and nothing here touches
the timeline. The chain keys are validated by `chain_rules.chain_check_problem`.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy

logger = logging.getLogger("dw")

# The default ceilings, set from the 99th percentile of each seam measure
# across the noise-floor runs of S1-S3 (bench/records/noise-floor.md, "Chain
# check thresholds"; 28 seams of unchecked chains, measured with this
# module's own arithmetic). The jump ceiling is about twice its P99 (2.02),
# so 4.0; the look ceiling is doubled and rounded up to half a point:
# - jump_ratio: P99 2.02 -> 4.0. Well under `seam_frame_jump`'s own threshold
#   (25, dw/assessment_rules.py) and the 9.9-20.3 the stage D runs measured
#   for an intended cut, above a dissolve's 1.8 - a chain seam over it is a
#   cut nobody asked for
# - look_ratio: P99 2.67 -> 5.5. The calibration
#   (docs/proposals/complete/seam-look-measures-complete.md) read 0.64 for a
#   clip joined to itself and 19.2 for a saturation-1.6 grade, so 5.5 is
#   well under a visible grade
# - texture_ratio: still PROVISIONAL, 2.0, the segment's fine-band luma
#   against segment 1's, either way - halved (blur) or doubled (crust). The
#   noise runs cannot set it: the two same-seed runs of S1 and S2 are
#   identical (no spread), and the unchecked chains' own ratio is the drift
#   the check exists to catch (P99 3.0x, up to 2.4x in S1 and 0.32x in S2
#   by segment 10), so a percentile of it would raise the ceiling to the
#   very defect. It is set from a checked chain's runs, not these
JUMP_RATIO_MAX = 4.0
LOOK_RATIO_MAX = 5.5
TEXTURE_RATIO_MAX = 2.0
# - cut_ratio: the largest frame step inside the segment over its own typical
#   step - a seam nobody marked. The seam measures above miss a segment that
#   cuts away inside itself (bench/records/check.md: an S3 re-roll cut to
#   another scene 26 frames past its seam and passed). 282 segments of 47 bench
#   takes read P50 1.22, P99 2.62, max 2.98 (bench/cuts.py); the two cuts in
#   that take read 4.16 and 17.2. The seam's own ceiling, 4.0, sits between
# Issue #1205: the cut read 4.15 on one card and under 4.0 on another, so a 4.0
# ceiling sat on the boundary. 3.5 sits between the good max (2.98) and the cut
CUT_RATIO_MAX = 3.5
# - audio_dbfs: the floor, in dBFS rms, of a segment's generated audio over
#   its kept span (#1221). The S5/extend dropouts on mini-ai read -87.8 to
#   -89.2; the quietest real segment of the bench read -57 (the 36 s seed-2
#   take). -70 sits between. Family-independent: generated audio is taken as
#   wanted, and a step that wants silence sets the floor to null
AUDIO_DBFS_MIN = -70.0

# - words_similarity: the least similarity of what a dialogue segment is heard
#   to say to its line (#1262), check_script's measure on one line.
#   0.80, the ceiling bench/records/words.md measured on the constitution-walk-2
#   takes: no good take scored under 0.90, but a bad take (a dropped head
#   word, 0.9565) scored above good ones, so the measure does not separate
#   good from bad and a chain does not re-roll on it by default
#   (WORDS_SIMILARITY_DEFAULT, None: the separation gate of #1262). 0.80 is the
#   most a chain should set; its own constant, apart from check_script's
#   DEFAULT_SIMILARITY (0.85), because a chain re-rolls on it
WORDS_SIMILARITY_MIN = 0.80
WORDS_SIMILARITY_DEFAULT = None
# The check_script findings that fail a segment's words whatever its
# similarity: speech cut off by the segment's end, Whisper stuck in a loop, a
# markup word spoken aloud, words matched to no line (an ad-lib or a repeat)
WORDS_FINDINGS = (
    "line_clipped_at_end",
    "transcript_loop",
    "tag_spoken",
    "unscripted_speech",
)
# The words of the line before a segment's that the seam-head rule looks for
# at the head of its kept span: the carry's last word, cut part-way by
# head_trim, and the one or two before it
SEAM_TAIL_WORDS = 3

# face_drop: how far a segment's faces have moved from segment 1's frame 0,
# below segment 1's own median similarity to it. None by default - measured
# and recorded, never failed - until the bench sets it
FACE_DROP_DEFAULT = None
# Frames of a segment, evenly spaced, its face similarity is the median of
FACE_SAMPLES = 8

# The measures a check can fail, in the order a failure names them
MEASURES = (
    "jump_ratio",
    "look_ratio",
    "texture_ratio",
    "cut_ratio",
    "audio_dbfs",
    "words",
    "face_drop",
)

# Measures that warn and record but never re-roll a segment (ruling R38):
# texture drift is in the carry, so a fresh seed from the same carry inherits
# it (bench/records/check.md); the identity measure is not yet known to be
# one a new seed fixes. The others a new seed can fix
WARN_ONLY = ("texture_ratio", "face_drop")

# The measures with_face_drop records on a checked segment's shot `check`
FACE_RECORD = ("face_drop",)

# The failed measures that read the `words` answer: a failed one, and a passed
# one whose opening is missing (#1308)
WORDS_MEASURES = frozenset({"words", "words_opening"})

# Frames of a segment, evenly spaced, its fine-band texture is the mean of
TEXTURE_SAMPLES = 4

# Attempt n of a segment (0 is the first render) runs at the step's seed
# plus n times this prime, so a re-roll is reproducible from the step's seed
REROLL_PRIME = 1000003


@dataclass(frozen=True)
class SegmentCheck:
    """The ceilings one segment is held to. `texture_ratio_max` bounds the
    fine-band ratio both ways (None skips it); `audio_dbfs_min` is the floor
    of its generated audio (None skips it); `words_similarity_min` the least
    similarity of its heard words to its line (None, the default, skips the
    words measure); `face_drop_max` the most a segment's face similarity
    may fall below segment 1's (None, the default, measures and records
    without failing)."""

    jump_ratio_max: float = JUMP_RATIO_MAX
    look_ratio_max: float = LOOK_RATIO_MAX
    texture_ratio_max: float | None = TEXTURE_RATIO_MAX
    cut_ratio_max: float | None = CUT_RATIO_MAX
    audio_dbfs_min: float | None = AUDIO_DBFS_MIN
    words_similarity_min: float | None = WORDS_SIMILARITY_DEFAULT
    face_drop_max: float | None = FACE_DROP_DEFAULT


@dataclass
class CheckResult:
    passed: bool
    measures: dict
    failed: list = field(default_factory=list)
    # The segment's own (thumbs, looks), so the next segment's check measures
    # its seam against them without holding the segment's frames (#1241)
    frame_data: tuple | None = field(default=None, repr=False, compare=False)

    @property
    def rerolls(self):
        """Whether a re-roll could help: a failed measure that is not warn-only."""
        return any(name not in WARN_ONLY for name in self.failed)


def segment_check(value):
    """A chain block's `check` as a SegmentCheck, or None when off: true is
    the defaults, an object overrides the fields it names. The value has
    passed `chain_rules.chain_check_problem`."""
    if value is None or value is False:
        return None
    if value is True:
        return SegmentCheck()
    fields = {
        name: float(value[name])
        for name in ("jump_ratio_max", "look_ratio_max")
        if name in value
    }
    for name in (
        "texture_ratio_max",
        "cut_ratio_max",
        "audio_dbfs_min",
        "words_similarity_min",
        "face_drop_max",
    ):
        if name in value:
            fields[name] = None if value[name] is None else float(value[name])
    return SegmentCheck(**fields)


def texture_samples(frames):
    """The TEXTURE_SAMPLES evenly spaced frames the texture measure reads -
    all a chain needs to keep of segment 1."""
    if len(frames) <= TEXTURE_SAMPLES:
        return list(frames)
    picks = numpy.linspace(0, len(frames) - 1, TEXTURE_SAMPLES).round().astype(int)
    return [frames[i] for i in picks]


def measure_frames(frames):
    """A segment's (thumbs, looks) - what a seam check reads of the segment
    before it, small enough to keep in place of its frames."""
    from ..tasks.assess import frame_measures

    return frame_measures(frames)


def audio_dbfs(audio):
    """The rms level in dBFS of a segment's kept audio, (channels, samples),
    rounded to 0.1 dB - digital silence reads SILENCE_DBFS - or None when
    there is no audio to measure (none generated, or the chain holds it)."""
    from ..dsp import SILENCE_DBFS, dbfs, rms

    if audio is None or getattr(audio, "size", 0) == 0:
        return None
    return round(dbfs(rms(numpy.asarray(audio)), floor=SILENCE_DBFS), 1)


def check_audio(audio, check):
    """Hold a segment's kept audio to the check's floor alone, as a
    CheckResult - segment 1's check, which has no seam to measure."""
    level = audio_dbfs(audio)
    failed = ["audio_dbfs"] if _under(level, check.audio_dbfs_min) else []
    return CheckResult(not failed, {"audio_dbfs": level}, failed)


def check_segment(
    previous, frames, segment_1_texture, check, hard_cut=False, audio=None
):
    """Measure a segment's kept frames against the segment before it and
    segment 1, and its kept audio against the floor, as a CheckResult.
    `previous` is that segment's frames, or the (thumbs, looks) pair
    `measure_frames` returned for them; `segment_1_texture` is the base
    `texture_base` measured from segment 1's frames once, or None when it had
    none to measure (no texture ratio is then read). `audio` is the
    segment's kept generated audio, None when it has none or the chain holds
    it. `hard_cut` marks a seam meant as a cut, which is never checked."""
    if hard_cut:
        return CheckResult(True, {"skipped": "hard_cut"}, [])
    from ..tasks.assess import seam_measures

    level = audio_dbfs(audio)
    audio_failed = ["audio_dbfs"] if _under(level, check.audio_dbfs_min) else []
    before_thumbs, before_looks = (
        previous if isinstance(previous, tuple) else measure_frames(previous)
    )
    after_data = measure_frames(frames)
    after_thumbs, after_looks = after_data
    if before_thumbs is None or after_thumbs is None:
        return CheckResult(
            not audio_failed,
            {"skipped": "no frames", "audio_dbfs": level},
            audio_failed,
            after_data,
        )
    seam = seam_measures(
        numpy.concatenate([before_thumbs, after_thumbs]),
        numpy.concatenate([before_looks, after_looks]),
        {"start_frame": 0, "num_frames": len(before_thumbs)},
        {"start_frame": len(before_thumbs), "num_frames": len(after_thumbs)},
    )
    measures = {
        "jump_ratio": seam["jump_ratio"],
        "frame_delta": seam["frame_delta"],
        "typical_delta": seam["typical_delta"],
        "look_ratio": seam["look_ratio"],
        "look_ratio_measure": seam["look_ratio_measure"],
        "texture_ratio": _texture_ratio(frames, segment_1_texture),
        "cut_ratio": _cut_ratio(after_thumbs),
        "audio_dbfs": level,
    }
    failed = []
    if _over(measures["jump_ratio"], check.jump_ratio_max):
        failed.append("jump_ratio")
    if _over(measures["look_ratio"], check.look_ratio_max):
        failed.append("look_ratio")
    texture = measures["texture_ratio"]
    if (
        check.texture_ratio_max is not None
        and texture is not None
        and max(texture, 1 / texture if texture else math.inf) > check.texture_ratio_max
    ):
        failed.append("texture_ratio")
    if check.cut_ratio_max is not None and _over(
        measures["cut_ratio"], check.cut_ratio_max
    ):
        failed.append("cut_ratio")
    failed.extend(audio_failed)
    return CheckResult(not failed, measures, failed, after_data)


def with_words(result, words):
    """`result` with a segment's words answer (`check_words`, or None for a
    segment with no line) recorded as its `words` measure, failing it when
    the answer did not pass. A segment with no line is never failed for it:
    nothing says what it should speak (#1262)."""
    result.measures["words"] = words
    if words is not None and not words["passed"]:
        result.failed.append("words")
        result.passed = False
    elif words is not None and any(
        f.get("rule") == "line_opening_missing" for f in words["findings"]
    ):
        # Passed, but its opening is gone (#1308): re-rolls like a failure
        result.failed.append("words_opening")
        result.passed = False
    return result


def _face_scores(frames, reference, samples):
    """The seam the face models sit behind: `face_similarities`."""
    from ..tasks.face_drift import face_similarities

    return face_similarities(frames, reference, samples)


def _measured(name, measure, *args):
    """`measure(*args)`, or None when it raises: a measure that only records
    must never fail the chain it is measuring (a reduced cv2, no weights)."""
    try:
        return measure(*args)
    except Exception as error:
        logger.warning(f"Chain check: {name} not measured ({error})")
        return None


def face_base(frames):
    """(reference, median): segment 1's frame 0 and the median similarity of
    segment 1's sampled frames from index 1 on to it - frame 0 against itself
    would match at 1.0 and lift the base - which every later segment's
    face_drop is measured from, once. None when frame 0 holds no face (one
    log line says so) or the models cannot load: face_drop is then never
    measured."""
    if len(frames) < 2:
        return None
    from ..tasks.face_drift import NoFaceInReference

    reference = frames[0]
    try:
        scores = _face_scores(list(frames[1:]), reference, FACE_SAMPLES)
    except NoFaceInReference:
        # Said once, in the chain's words: the task's own message asks for a
        # reference picture, which a chain's caller never gave
        logger.info(
            "Chain check: face_drop not measured - segment 1's first frame "
            "shows no face"
        )
        return None
    except Exception as error:
        logger.warning(f"Chain check: face_drop not measured ({error})")
        return None
    if not scores:
        return None
    return reference, float(numpy.median(scores))


def with_face_drop(result, frames, base, check):
    """`result` with the segment's `face_drop` (against `face_base`'s `base`,
    None without one or with no face in the segment) recorded, failing it
    only when the check sets `face_drop_max`. WARN_ONLY: a failure is
    recorded and warned, never re-rolled."""
    drop = None
    if base is not None:
        scores = _measured("face_drop", _face_scores, frames, base[0], FACE_SAMPLES)
        if scores:
            drop = round(base[1] - float(numpy.median(scores)), 3)
    result.measures["face_drop"] = drop
    if check.face_drop_max is not None and _over(drop, check.face_drop_max):
        result.failed.append("face_drop")
    result.passed = result.passed and not result.failed
    return result


def face_record(result):
    """The identity measure of a checked segment, for its shot."""
    return {name: result.measures.get(name) for name in FACE_RECORD}


def shown_value(result, name):
    """A failed measure's value as a failure names it: the words measure by
    its similarity, every other by its number."""
    value = result.measures.get(name)
    if name in WORDS_MEASURES:
        value = result.measures.get("words")
    if name in WORDS_MEASURES and isinstance(value, dict):
        return value.get("similarity")
    return value


def _failure_text(result, name):
    """One failed measure as a log or warning names it. A words failure also
    names the findings that failed it and what was heard: a passing
    similarity still fails on any WORDS_FINDINGS finding (#1262), and the
    number alone would not say why."""
    text = f"{name} {shown_value(result, name)}"
    words = result.measures.get("words") if name in WORDS_MEASURES else None
    if not isinstance(words, dict):
        return text
    rules = list(dict.fromkeys(f.get("rule") for f in words.get("findings") or ()))
    if rules:
        text += f" with {', '.join(str(rule) for rule in rules)}"
    return text + f", heard {words.get('heard', '')!r}"


def failures_text(result):
    """Every failed measure of a CheckResult, as the log and warning name them."""
    return "; ".join(_failure_text(result, name) for name in result.failed)


def _words_only(result):
    return result.failed == ["words"]


def best_attempt(results):
    """The index of the attempt to keep: fewest failed measures, then the
    lowest jump ratio, then the earliest. Attempts that failed words alone
    rank after any other one-measure failure and among themselves by the
    higher similarity (#1262): their pictures passed, so the jump says
    nothing about which spoke its line best."""

    def rank(index):
        result = results[index]
        if _words_only(result):
            similarity = (result.measures.get("words") or {}).get("similarity")
            return (1, 1, -(similarity or 0.0), index)
        jump = result.measures.get("jump_ratio")
        return (len(result.failed), 0, math.inf if jump is None else jump, index)

    return min(range(len(results)), key=rank)


def reroll_seed(seed, attempt):
    """The seed attempt `attempt` of a segment runs at (0 is the first)."""
    return seed + attempt * REROLL_PRIME


def _cut_ratio(thumbs):
    """The largest frame step inside a segment over its own typical step,
    floored as a seam's is - `_seam_video`'s arithmetic on a seam inside the
    segment - or None for fewer than three frames. The typical step is the
    90th percentile, linearly interpolated, so at 3-5 frames (2-4 steps) it is
    at least 0.7 of the largest step and the ratio stays under about 1.43: a
    short segment cannot fail the default 3.5 ceiling (a lower `cut_ratio_max`
    can fail one)."""
    from ..tasks.assess import TYPICAL_DELTA_FLOOR, TYPICAL_DELTA_PERCENTILE

    if thumbs.shape[0] < 3:
        return None
    deltas = numpy.abs(numpy.diff(thumbs.astype(numpy.int16), axis=0)).mean(axis=(1, 2))
    typical = max(
        TYPICAL_DELTA_FLOOR, float(numpy.percentile(deltas, TYPICAL_DELTA_PERCENTILE))
    )
    return round(float(deltas.max()) / typical, 2)


def _over(value, ceiling):
    return value is not None and value > ceiling


def _under(value, floor):
    return value is not None and floor is not None and value < floor


def _fine_luma(frames):
    """The mean fine-band luma of `frames`' texture samples, or None when a
    frame is too small to hold the band."""
    from ..texture_bands import MIN_SIDE, texture_bands

    values = []
    for frame in texture_samples(frames):
        # A chain's frames are PIL images; an array is taken as RGB already
        rgb = numpy.asarray(
            frame.convert("RGB") if hasattr(frame, "convert") else frame
        )
        if min(rgb.shape[:2]) < MIN_SIDE:
            return None
        values.append(texture_bands(rgb)["luma"]["fine"])
    return float(numpy.mean(values)) if values else None


def texture_base(frames):
    """Segment 1's fine-band luma, the base every later segment's
    texture_ratio divides by - measured once, after segment 1, and passed to
    `check_segment` in place of its frames; None when it has none to measure."""
    return _fine_luma(frames)


def _texture_ratio(frames, segment_1_texture):
    """The segment's fine-band luma over segment 1's (`texture_base`), or
    None where either has none to measure."""
    value = _fine_luma(frames)
    if not segment_1_texture or value is None:
        return None
    return round(value / segment_1_texture, 4)


def _whisper_chunks(waveform, sample_rate, device="cpu"):
    """transcribe_audio's word chunks for a waveform, levelled as check_script
    levels a take - the same model cache, so a chain loads Whisper once."""
    from ..tasks.audio_transcription import transcribe_audio
    from ..tasks.script_take import level_for_transcription

    leveled, _ = level_for_transcription(waveform)
    return transcribe_audio(
        leveled, device=device, sample_rate=sample_rate, timestamps="word"
    )["chunks"]


def _seam_head(chunks, expected, previous_line):
    """(kept, dropped): `chunks` with the seam-head rule applied. A chunk
    heard before the first word that aligns to the line (`expected`, its
    normalized words) is dropped when every word of it is one of the last
    SEAM_TAIL_WORDS words of `previous_line` - a carried word the head trim
    cut part-way. A chunk there that matches nothing is kept: an invented
    word counts.

    The heard words are the owners' (Script check row): each chunk's
    words unfolded (`normalize_words(numbers=False)`), then folded across
    chunks by `fold_numbers`, as `guard_words` builds them, so 'twenty' and
    'one' heard as two chunks are the one word '21' here as in `check`. The
    head is `script_check.align`'s: the heard words it leaves unassigned
    before the line's first assigned word. A head word folded across chunks
    drops every chunk it spans. The guard's discards (below the floor, a
    loop) are not applied here: `check` applies them to the chunks kept."""
    from ..script_lines import fold_numbers, normalize_words, strip_markup
    from ..tasks.script_check import align

    if not previous_line or not expected:
        return list(chunks), []
    tail = set(normalize_words(strip_markup(previous_line)[0])[-SEAM_TAIL_WORDS:])
    if not tail:
        return list(chunks), []
    owners = [
        index
        for index, chunk in enumerate(chunks)
        for _ in normalize_words(chunk.get("text", ""), numbers=False)
    ]
    unfolded = [
        word
        for chunk in chunks
        for word in normalize_words(chunk.get("text", ""), numbers=False)
    ]
    heard, spans, k = [], [], 0
    for word, count in fold_numbers(unfolded):
        heard.append({"word": word})
        spans.append(set(owners[k : k + count]))
        k += count
    assigned, unassigned = align([expected], heard)
    if not assigned[0]:
        return list(chunks), []
    first = min(assigned[0])
    head = {j for j, _ in unassigned if j < first}
    # A chunk is dropped when every heard word touching it is a head word
    # from the tail; one shared with the line's first word stays
    touching = {}
    for j, chunk_indices in enumerate(spans):
        for index in chunk_indices:
            touching.setdefault(index, []).append(j)
    dropped = {
        index
        for index, words in touching.items()
        if all(j in head and heard[j]["word"] in tail for j in words)
    }
    kept = [chunk for index, chunk in enumerate(chunks) if index not in dropped]
    return kept, [chunks[index] for index in sorted(dropped)]


def _clipped_at_head(chunks, expected, previous_line, head_seconds):
    """line_clipped_at_head (#1300) as a finding, or None: `chunks` is the
    transcript of a segment's whole audio, head trim included. The line's
    opening is cut when the first word heard after the seam-head rule starts
    inside the first `head_seconds` - the span the chain trims off the head -
    so what the kept span opens on is already the line's middle."""
    kept, _ = _seam_head(chunks, expected, previous_line)
    for chunk in kept:
        if not chunk.get("text", "").strip():
            continue
        start = chunk.get("start")
        if start is None or start >= head_seconds:
            return None
        return {
            "rule": "line_clipped_at_head",
            "severity": "warn",
            "at": {"seconds": round(float(start), 3), "word": chunk["text"].strip()},
            "value": round(float(start), 3),
            "threshold": round(float(head_seconds), 3),
            "says": (
                "the line starts inside the frames the chain trims off the head of"
                " the segment, so its opening words are cut from the delivered"
                " video; a fresh seed starts it at the same place - move the line"
                " later in the prompt (an 'at N seconds' cue past the guide span)"
            ),
        }
    return None


def _folded_heard(chunks):
    """(heard, first_chunk): `chunks`' words as `_seam_head` builds them -
    unfolded per chunk, then folded across chunks by `fold_numbers` - as
    [{"word": ...}], with the index of the chunk each begins in."""
    from ..script_lines import fold_numbers, normalize_words

    owners = [
        index
        for index, chunk in enumerate(chunks)
        for _ in normalize_words(chunk.get("text", ""), numbers=False)
    ]
    unfolded = [
        word
        for chunk in chunks
        for word in normalize_words(chunk.get("text", ""), numbers=False)
    ]
    heard, firsts, k = [], [], 0
    for word, count in fold_numbers(unfolded):
        heard.append({"word": word})
        firsts.append(owners[k])
        k += count
    return heard, firsts


def _opens_mid_line(chunks, expected, head_seconds, previous_line=None):
    """line_opening_missing (#1308) for a kept span that passed, or None: the
    line's words are aligned to the span's heard words (`script_check.align`,
    heard words folded as `_seam_head` folds them) and the words of the line
    before the first one heard are its `missing` opening. Fires on a carried
    segment (`previous_line`) or one with a head trim (`head_seconds`): the
    trim or the carry cut the line's start. Reported, and the segment
    re-rolls (`with_words`): the similarity cleared the floor with the
    opening gone."""
    from difflib import SequenceMatcher

    from ..tasks.script_check import align

    if not expected or not (previous_line or head_seconds > 0):
        return None
    heard, firsts = _folded_heard(chunks)
    assigned, _ = align([expected], heard)
    if not assigned[0]:
        return None
    blocks = SequenceMatcher(
        None, expected, [h["word"] for h in heard], autojunk=False
    ).get_matching_blocks()
    first = next((block for block in blocks if block.size), None)
    if first is None or first.a == 0:
        return None
    missing = expected[: first.a]
    chunk = chunks[firsts[first.b]]
    start = chunk.get("start")
    return {
        "rule": "line_opening_missing",
        "severity": "warn",
        "at": {
            "seconds": None if start is None else round(float(start), 3),
            "word": heard[first.b]["word"],
        },
        "value": first.b,
        "threshold": 0,
        "missing": missing,
        "says": (
            f"the segment's audio opens past the line's first words ({' '.join(missing)}):"
            " its opening was cut by the head trim or never spoken; the"
            " similarity passed, but the delivered video lacks them - the"
            " segment re-rolls once, and a fresh seed may start the line"
            " earlier; for a long monologue use a narrated take"
        ),
    }


def _first_word_start(chunks):
    """The start of the first chunk with text, or None."""
    for chunk in chunks:
        if chunk.get("text", "").strip():
            start = chunk.get("start")
            return None if start is None else round(float(start), 3)
    return None


def check_words(
    audio,
    sample_rate,
    line,
    previous_line,
    minimum,
    transcribe=None,
    device="cpu",
    full_audio=None,
    head_seconds=0.0,
):
    """Whether a dialogue segment's kept audio speaks its line (#1262), as
    {similarity, heard, findings, passed, seam_head_dropped, first_word_start}.

    `audio` is the kept span ((channels, samples) or 1-D), `line` the text
    the segment is meant to speak (markup kept: `script_lines.dialogue_text`
    or the chain's `lines`), `previous_line` the line of the segment before
    it (None for segment 1), `minimum` the least similarity that passes
    (None holds only the findings). `transcribe(waveform, sample_rate)` gives
    word chunks; by default Whisper's, on `device`.

    The comparison is `script_check.check` unchanged, on one line and no
    shots, after the seam-head rule (`_seam_head`) drops carried words of
    `previous_line` at the head. `findings` are its WORDS_FINDINGS, each of
    which fails the segment; so does a similarity under `minimum`, or none
    (the line looped and was never transcribed). A passed segment may also
    carry `line_opening_missing` (#1308): it stays `passed`, but `with_words`
    fails the segment's `words_opening` measure so it re-rolls, and the
    chain reports it when every attempt carries it. `first_word_start` is
    the start of the first kept chunk. `heard` is the transcript
    after the seam-head rule, joined; `seam_head_dropped` the chunks it
    dropped."""
    from ..script_lines import normalize_words, parse_lines
    from ..tasks.script_check import check
    from ..tasks.script_findings import DEFAULT_SIMILARITY

    parsed = parse_lines([line])
    expected = normalize_words(parsed[0]["text"])
    waveform = numpy.asarray(audio, dtype=numpy.float32)
    if waveform.ndim == 1:
        waveform = waveform[numpy.newaxis, :]
    mono = waveform[0] if waveform.shape[0] == 1 else waveform.mean(axis=0)
    if transcribe is None:

        def transcribe(wave, rate):
            return _whisper_chunks(wave, rate, device)

    chunks, dropped = _seam_head(
        transcribe(waveform, sample_rate), expected, previous_line
    )
    threshold = DEFAULT_SIMILARITY if minimum is None else float(minimum)
    answer = check(parsed, chunks, mono, sample_rate, threshold)
    similarity = answer["lines"][0]["similarity"]
    findings = [f for f in answer["findings"] if f["rule"] in WORDS_FINDINGS]
    low = similarity is None or (minimum is not None and similarity < minimum)
    failed = low or bool(findings)
    if failed and full_audio is not None and head_seconds > 0:
        # The kept span fails its line: look at the audio the head trim cut
        # off, where the line may have opened (#1300). Only a failure pays
        # for the second transcription
        whole = numpy.asarray(full_audio, dtype=numpy.float32)
        if whole.ndim == 1:
            whole = whole[numpy.newaxis, :]
        clipped = _clipped_at_head(
            transcribe(whole, sample_rate), expected, previous_line, head_seconds
        )
        if clipped is not None:
            findings.append(clipped)
    passed = not failed
    if passed:
        # A pass can hide a missing opening (#1308): the similarity cleared
        # the floor without the line's first words. The kept span's own
        # alignment shows it without a second pass
        opening = _opens_mid_line(chunks, expected, head_seconds, previous_line)
        if opening is not None:
            findings.append(opening)
    heard = " ".join(
        text for text in (chunk.get("text", "").strip() for chunk in chunks) if text
    )
    return {
        "similarity": similarity,
        "heard": heard,
        "findings": findings,
        "passed": passed,
        "seam_head_dropped": [chunk.get("text", "").strip() for chunk in dropped],
        "first_word_start": _first_word_start(chunks),
    }
