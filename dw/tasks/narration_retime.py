"""retime_narration and narration_lines: a narration laid out for a chain.

A narrated long take is spoken first and pictured after: a chain renders the
picture in fixed-length segments, so the speech has to fit the chain's frame
grid rather than the other way round. `retime_narration` sets every pause by
the class of punctuation it follows - a comma, a sentence end, a section break
- and, on the chain's frame grid, brings a word junction onto each segment
boundary so the boundary falls between words instead of through one. `narration_lines` then
reads the words each planned segment holds, so a chain step gets one spoken
line (and, from a caller's template, one prompt) per segment.

Word times come from one word-timestamp transcription (`transcribe_audio`,
through the `_word_chunks` seam the tests replace). Given the script as
`text`, the punctuation and the spelling come from the script - aligned to
the heard words with `script_check.align` - and a blank line, or a line that
opens with SECTION_LINE_PREFIX, is a section break; without it they come
from Whisper's own chunk text, which knows no sections.

A pause is lengthened by inserting the source's own room tone - the best
`find_loop_bed` candidate, else its quietest QUIET_WINDOW_SECONDS window,
looped with `loop_audio` - and shortened by cutting out its middle; every cut
is an equal-power crossfade of SEAM_SECONDS, so the result is never digital
zero where the source was not.
"""

import bisect
import logging
import re


from .audio_transcription import (
    WHISPER_DEFAULT_MODEL,
    model_source,
    required_weights,  # noqa: F401 - re-exported for the registry hook
    transcribe_audio,
)
from .audio_utils import as_track, loop_audio, waveform_and_rate
from .joins import crossfade_concat
from .script_check import align
from .script_take import level_for_transcription
from ..dsp import rms
from ..events import emit_log
from ..media_types import JsonRecord
from ..script_lines import normalize_words
from ..task_domains import check_arguments, whole_number

logger = logging.getLogger("dw")

RETIME = "retime_narration"
LINES = "narration_lines"

# Weakest first: a gap that two rules classify takes the stronger
PAUSE_CLASSES = ("comma", "sentence", "section")
_COMMA_MARKS = ",;:"
_SENTENCE_MARKS = ".?!"
# Closing quotes and brackets after a mark do not hide it: 'said."' ends a
# sentence
_CLOSERS = "\"')]}”’»"
# A script line opening with this word starts a section, as a blank line does
SECTION_LINE_PREFIX = "Amendment"
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")

# The equal-power crossfade at every cut and every inserted bed's edges
SEAM_SECONDS = 0.01
# A boundary placed in an eligible gap sits at least this far from the words
# either side of it; in a narrower junction it may sit anywhere
BOUNDARY_MARGIN_SECONDS = 0.02
# Moves aim this far past the margin; margin tests allow _TOLERANCE of float noise.
_AIM_EXTRA = 1e-3
_TOLERANCE = 1e-6
# The window looped for room tone when find_loop_bed finds no candidate
# narration_lines judges a boundary mid-word only when it lies inside a word
# by more than this on both sides: Whisper's word times on edited audio jitter
# by tens of ms, so a strict start < t < end reads junction placements as cuts.
WORD_JITTER_SECONDS = 0.08
QUIET_WINDOW_SECONDS = 0.25
_QUIET_HOP_SECONDS = 0.05


def _word_chunks(waveform, rate, device, model_name):
    """The word chunks heard in a waveform: `transcribe_audio` with word
    timestamps, the waveform first levelled as check_script levels it (#925).

    Tests replace this module attribute; it is not a task argument."""
    leveled, _gain = level_for_transcription(waveform)
    transcript = transcribe_audio(
        leveled,
        device=device,
        sample_rate=rate,
        model_name=model_name,
        timestamps="word",
    )
    return transcript["chunks"]


def punctuation_class(token):
    """The pause class the punctuation ending `token` asks for, or None."""
    stripped = token.rstrip(_CLOSERS)
    if not stripped:
        return None
    if stripped[-1] in _SENTENCE_MARKS:
        return "sentence"
    if stripped[-1] in _COMMA_MARKS:
        return "comma"
    return None


def _stronger(first, second):
    if first is None or second is None:
        return first or second
    return max(first, second, key=PAUSE_CLASSES.index)


def script_tokens(text):
    """The script's words as [{text, words, class}]: each whitespace token as
    written, its normalized words, and the class of the pause after it. A
    token with no words ('-') hands its punctuation to the token before."""
    tokens = []

    def mark(pause):
        if tokens:
            tokens[-1]["class"] = _stronger(tokens[-1]["class"], pause)

    for paragraph in _PARAGRAPH_BREAK.split(text.strip()):
        for line in paragraph.splitlines():
            if line.lstrip().startswith(SECTION_LINE_PREFIX):
                mark("section")
            for raw in line.split():
                words = normalize_words(raw, numbers=False)
                if not words:
                    mark(punctuation_class(raw))
                    continue
                tokens.append(
                    {"text": raw, "words": words, "class": punctuation_class(raw)}
                )
        mark("section")
    return tokens


def _heard_words(chunks):
    """The transcription's word chunks as word entries, in time order. A
    chunk with no text, or one `transcribe_audio` marks as silent, is not a
    word: the span it claims is part of a gap."""
    words = []
    for chunk in sorted(chunks, key=lambda c: float(c["start"])):
        text = str(chunk.get("text") or "").strip()
        if not text or chunk.get("suspect") == "silent":
            continue
        words.append(
            {
                "word": text,
                "say": text,
                "start": float(chunk["start"]),
                "end": float(chunk["end"]),
                "class": punctuation_class(text),
            }
        )
    return words


def _apply_script(words, tokens):
    """Take punctuation classes and spelling from the script. Each script
    token is aligned to the heard words; the heard word holding its last
    word carries its class and says its text, and the other heard words it
    spans say nothing (so a line does not repeat them). A token no heard
    word matched ('dollars,' when Whisper heard '$20,' for 'twenty
    dollars,') is said, with its class, by the heard word of the token
    before it - by the first token's word when it opens the script. A heard
    word no token claims keeps Whisper's text."""
    heard, chunk_of = [], []
    for k, word in enumerate(words):
        for normalized in normalize_words(word["word"], numbers=False):
            heard.append({"word": normalized})
            chunk_of.append(k)
    assigned, _unassigned = align([t["words"] for t in tokens], heard)
    said = [[] for _ in words]
    absorbed = set()
    for word in words:
        word["class"] = None
    last, opening = None, []
    for token, owned in zip(tokens, assigned):
        if not owned and last is None:
            opening.append(token["text"])
            continue
        if owned:
            chunks = [chunk_of[j] for j in owned]
            last = chunks[-1]
            absorbed.update(k for k in chunks if k != last)
        words[last]["class"] = _stronger(words[last]["class"], token["class"])
        said[last].extend([*opening, token["text"]])
        opening = []
    for k, word in enumerate(words):
        if said[k]:
            word["say"] = word["word"] = " ".join(said[k])
        elif k in absorbed:
            word["say"] = ""


def word_map(waveform, rate, text, device, model_name):
    """The words heard in a waveform as [{word, say, start, end, class}]:
    `class` is the pause class of the gap after the word (None for none),
    from the script when `text` is given and from Whisper's text otherwise."""
    words = _heard_words(_word_chunks(waveform, rate, device, model_name))
    # A script with no words in it is no script: Whisper's own punctuation
    # stands, rather than every pause class being cleared
    tokens = script_tokens(str(text)) if text is not None else []
    if tokens and words:
        _apply_script(words, tokens)
    return words


# The longest pause a class may be set to: the bed under it is looped room
# tone, built a lap at a time, so an unbounded length is unbounded work
MAX_PAUSE_SECONDS = 60.0


def _frames(seconds, fps):
    """The frames a chain matched to `seconds` of audio plans at `fps` - the
    count ChainConfig takes from a match_audio track, so the boundaries and
    the lines are the chain's own."""
    return max(1, round(seconds * fps))


def kept_span_ends(total_frames, segment_frames, guide_frames, fps):
    """The seconds where each planned segment's kept span ends, but the last:
    the chain's segment boundaries. Segment 1 ends at segment_frames, and
    every later one segment_frames - guide_frames after the one before
    (`chain_config.plan_segments` with guide_frames as the head trim)."""
    from ..pipeline_processors.chain_config import plan_segments

    plan = plan_segments(total_frames, segment_frames, guide_frames)
    return [(s.audio_start_frame + s.num_frames) / fps for s in plan[:-1]]


def _retimed(words, gaps, targets):
    """The words at their times once every gap i lasts targets[i]."""
    out, shift = [], 0.0
    for i, word in enumerate(words):
        if i:
            shift += targets[i - 1] - gaps[i - 1]
        out.append(
            {
                "word": word["word"],
                "start": word["start"] + shift,
                "end": word["end"] + shift,
            }
        )
    return out


def _junctions(words, out, eligible):
    """Every junction between two words as (gap, start, end, eligible): from
    one word's end to the next's start, zero-length when they touch. The
    heard chunks of one script token (all but its last say nothing) meet
    inside a word, so they have no junction."""
    return [
        (i, out[i]["end"], max(out[i]["end"], out[i + 1]["start"]), i in eligible)
        for i in range(len(out) - 1)
        if words[i]["say"] != ""
    ]


def _margin(start, end, eligible):
    """The clearance a placed boundary keeps from the words: the margin in
    an eligible gap it fits, else none - anywhere in the junction will do."""
    fits = end - start >= 2 * BOUNDARY_MARGIN_SECONDS - _TOLERANCE
    return BOUNDARY_MARGIN_SECONDS if eligible and fits else 0.0


def _holds(junction, seconds, margin=True):
    """Whether `seconds` sits in the junction (clear of the margin with
    `margin`; anywhere between the two words without)."""
    _i, start, end, eligible = junction
    m = _margin(start, end, eligible) if margin else 0.0
    return start + m - _TOLERANCE <= seconds <= end - m + _TOLERANCE


def _shifted(junction, gap, change):
    """The junction once eligible gap `gap` lasts `change` longer."""
    i, start, end, eligible = junction
    if i < gap:
        return junction
    if i == gap:
        return (i, start, end + change, eligible)
    return (i, start + change, end + change, eligible)


def _nearest_change(low, high):
    """The change in [low, high] nearest zero, aimed _AIM_EXTRA inside the
    range so float noise cannot land it on an edge."""
    aim = min(_AIM_EXTRA, (high - low) / 2)
    return low + aim if low > 0 else high - aim


def _junction_moves(junction, boundary, eligible_order):
    """The (gap, change) pairs that bring `boundary` into the junction: the
    nearest eligible gap before it lengthened or shortened, carrying the
    whole junction, and an eligible junction's own gap lengthened."""
    j, start, end, eligible = junction
    moves = []
    before = bisect.bisect_left(eligible_order, j) - 1
    if before >= 0:
        m = _margin(start, end, eligible)
        low, high = boundary - (end - m), boundary - (start + m)
        if low <= high:
            moves.append((eligible_order[before], _nearest_change(low, high)))
    clear = start + BOUNDARY_MARGIN_SECONDS - _TOLERANCE <= boundary
    if eligible and clear and boundary > end - _margin(start, end, eligible):
        moves.append((j, boundary + BOUNDARY_MARGIN_SECONDS + _AIM_EXTRA - end))
    return moves


def _keeps_placed(junctions, gap, change, placed):
    """Whether no boundary already between two words leaves its junction
    when `gap` changes."""
    for seconds in placed:
        held = [j for j in junctions if _holds(j, seconds, margin=False)]
        if held and not any(
            _holds(_shifted(j, gap, change), seconds, margin=False) for j in held
        ):
            return False
    return True


def _best_move(junctions, boundary, layout, placed):
    """The (gap, change) of the smallest shift that brings the nearest
    junction onto `boundary`, or None. `layout` is (eligible_order, targets,
    floors, slack); a gap never shrinks below its floor."""
    eligible_order, targets, floors, slack = layout
    reach = slack + 2 * BOUNDARY_MARGIN_SECONDS
    best = None
    for junction in junctions:
        if junction[2] < boundary - reach or junction[1] > boundary + reach:
            continue
        for gap, change in _junction_moves(junction, boundary, eligible_order):
            if abs(change) > slack + _TOLERANCE:
                continue
            if targets[gap] + change < floors[gap] - _TOLERANCE:
                continue
            if best is not None and abs(change) >= abs(best[1]):
                continue
            if _keeps_placed(junctions, gap, change, placed):
                best = (gap, change)
    return best


def _place_boundaries(words, gaps, targets, eligible, grid, slack, source):
    """Bring a word junction onto each chain boundary, in order: a boundary
    already between two words stays; otherwise the nearest junction is
    moved onto it by lengthening or shortening the nearest eligible gap
    before that junction, by at most `slack`, never below half its class
    length nor below twice the margin. A boundary no move reaches stays put.
    `grid` is (segment_frames, guide_frames, fps) and `source` the source's
    seconds; targets change in place."""
    segment_frames, guide_frames, fps = grid
    order = sorted(eligible)
    floors = {i: max(targets[i] / 2, 2 * BOUNDARY_MARGIN_SECONDS) for i in order}
    layout = (order, targets, floors, slack)
    placed = []
    while True:
        out = _retimed(words, gaps, targets)
        duration = source + sum(t - g for t, g in zip(targets, gaps))
        ends = kept_span_ends(_frames(duration, fps), segment_frames, guide_frames, fps)
        if len(placed) >= len(ends):
            return
        boundary = ends[len(placed)]
        junctions = _junctions(words, out, set(eligible))
        if not any(_holds(j, boundary) for j in junctions):
            move = _best_move(junctions, boundary, layout, placed)
            if move is not None:
                targets[move[0]] += move[1]
        placed.append(boundary)


def _quietest_window(waveform, rate):
    """(start, length) in samples of the quietest QUIET_WINDOW_SECONDS window
    that is not digital zero (the whole waveform when shorter)."""
    length = min(waveform.shape[1], max(1, int(round(QUIET_WINDOW_SECONDS * rate))))
    hop = max(1, int(round(_QUIET_HOP_SECONDS * rate)))
    best, best_level = 0, None
    for start in range(0, waveform.shape[1] - length + 1, hop):
        level = rms(waveform[:, start : start + length])
        if level and (best_level is None or level < best_level):
            best, best_level = start, level
    return best, length


def room_tone(waveform, rate, samples):
    """`samples` of the source's own room tone, (channels, samples): the best
    find_loop_bed candidate looped with loop_audio, else the quietest window
    looped (ruling R4) - never synthesized silence."""
    from .loop_bed import find_loop_bed

    candidates = find_loop_bed(as_track(waveform, rate, RETIME), max_candidates=1)[
        "candidates"
    ]
    if candidates:
        start = int(round(candidates[0]["start_seconds"] * rate))
        length = int(round(candidates[0]["duration_seconds"] * rate))
        source = "loop_bed"
    else:
        (start, length), source = _quietest_window(waveform, rate), "quietest_window"
    emit_log(
        f"{RETIME}: room tone from the {source} at {start / rate:.2f} s "
        f"({length / rate:.2f} s)",
        command=RETIME,
        room_tone_source=source,
    )
    piece = as_track(waveform[:, start : start + length], rate, RETIME)
    bed = loop_audio(piece, duration_seconds=samples / rate + 0.01)
    return bed.audio[:, :samples]


def _edits(words, gaps, targets, rate):
    """Each changed gap as (cut, resume, fill) in samples: the source plays
    to `cut`, `fill` samples of room tone follow, and it resumes at
    `resume`. A gap is shortened by cutting out its middle and lengthened by
    a bed inserted at its middle."""
    edits = []
    for i, (gap, target) in enumerate(zip(gaps, targets)):
        if abs(target - gap) * rate < 1:
            continue
        end, start = words[i]["end"], words[i + 1]["start"]
        if target < gap:
            cut = int(round((end + target / 2) * rate))
            resume = cut + int(round((gap - target) * rate))
            edits.append((cut, resume, 0))
        else:
            middle = int(round((end + start) / 2 * rate))
            edits.append((middle, middle, int(round((target - gap) * rate))))
    return edits


def _render(waveform, rate, edits):
    """The source with every edit applied, each seam an equal-power
    crossfade: each piece reaches half a seam past its cut so the overlap
    leaves the edit's length exact."""
    if not edits:
        return waveform
    half = max(1, int(round(SEAM_SECONDS / 2 * rate)))
    largest = max(fill for _cut, _resume, fill in edits)
    bed = room_tone(waveform, rate, largest + 2 * half) if largest else None
    pieces, cursor = [], 0
    for cut, resume, fill in edits:
        pieces.append(waveform[:, cursor : cut + half])
        if fill:
            pieces.append(bed[:, : fill + 2 * half])
        cursor = max(0, resume - half)
    pieces.append(waveform[:, cursor:])
    return crossfade_concat(pieces, rate, 2 * half * 1000.0 / rate)


def _no_words(command):
    return ValueError(
        f"{command} heard no words in the audio, so there are no pauses to "
        f"set or lines to read - check the audio is the narration and is "
        f"audible"
    )


def _boundary_word(out, seconds):
    """The word `seconds` falls in, else the nearest word."""
    return min(
        out,
        key=lambda w: (
            0.0
            if w["start"] <= seconds <= w["end"]
            else min(abs(seconds - w["start"]), abs(seconds - w["end"]))
        ),
    )["word"]


def _boundaries(out, junctions, duration, grid):
    """The chain's boundaries over the final timeline as (boundaries,
    mid_word_boundaries). A boundary anywhere between two words is in a
    gap: the take's narration plays unbroken, so only a cut through a word
    is reported."""
    segment_frames, guide_frames, fps = grid
    ends = kept_span_ends(_frames(duration, fps), segment_frames, guide_frames, fps)
    boundaries, stuck = [], []
    for index, seconds in enumerate(ends):
        # past the last word is the take's tail, not a cut through a word
        in_gap = seconds >= out[-1]["end"] or any(
            _holds(j, seconds, margin=False) for j in junctions
        )
        boundaries.append(
            {"segment": index, "seconds": round(seconds, 4), "in_gap": in_gap}
        )
        if not in_gap:
            stuck.append({"segment": index, "word": _boundary_word(out, seconds)})
    return boundaries, stuck


def _report(words, gaps, targets, eligible, duration, grid):
    out = _retimed(words, gaps, targets)
    pauses = [
        {
            "after": words[i]["word"],
            "class": words[i]["class"],
            "before_seconds": round(gaps[i], 4),
            "after_seconds": round(targets[i], 4),
        }
        for i in eligible
        if words[i]["class"] is not None or abs(targets[i] - gaps[i]) > 1e-6
    ]
    report = {
        "words": [
            {
                "word": w["word"],
                "start": round(w["start"], 4),
                "end": round(w["end"], 4),
            }
            for w in out
        ],
        "pauses": pauses,
        "duration_seconds": round(duration, 4),
        "mid_word_boundaries": [],
    }
    if grid is not None:
        junctions = _junctions(words, out, set(eligible))
        report["boundaries"], report["mid_word_boundaries"] = _boundaries(
            out, junctions, duration, grid
        )
    return report


def _checked_grid(command, segment_frames, guide_frames, fps):
    """(segment_frames, guide_frames, fps) read as whole frames, or None
    when there is no chain grid."""
    segment_frames = whole_number(segment_frames, "segment_frames", command)
    guide_frames = whole_number(guide_frames, "guide_frames", command) or 0
    check_arguments(command, fps=fps, guide_frames=guide_frames)
    if segment_frames is None:
        return None
    check_arguments(command, segment_frames=segment_frames)
    return segment_frames, guide_frames, fps


def retime_narration(
    audio,
    text=None,
    comma_seconds=0.3,
    sentence_seconds=0.6,
    section_seconds=1.0,
    min_gap_seconds=0.12,
    segment_frames=None,
    guide_frames=0,
    fps=24,
    boundary_slack_seconds=0.25,
    model_name=WHISPER_DEFAULT_MODEL,
    sample_rate=None,
    device="cpu",
):
    """Task command: set a narration's pauses by punctuation class and, on a
    chain's frame grid, bring a word junction onto each segment boundary so
    the boundary falls between words.

    Args:
        audio: Path or URL of the narration (or of a video whose soundtrack
            is taken), a track from an earlier step, or a waveform (which
            needs sample_rate)
        text: The script; its punctuation, spelling and section breaks
            (a blank line, or a line opening "Amendment") win over Whisper's
        comma_seconds: Pause after a word ending , ; or :
        sentence_seconds: Pause after a word ending . ? or !
        section_seconds: Pause at a section break in `text`
        min_gap_seconds: A gap shorter than this is left alone, and is not
            moved to bring a junction onto a boundary
        segment_frames: Frames a chain segment generates; with it, the
            boundaries are placed and reported
        guide_frames: The chain's guide frames - the head every segment
            after the first trims
        fps: Frame rate the frames are counted at
        boundary_slack_seconds: Most one gap may be lengthened or shortened
            to bring a word junction onto a boundary
        model_name: A Whisper-class ASR model for the word times
        sample_rate: Sample rate of a waveform passed directly
        device: Target device ("cuda", "mps", "cpu")

    Returns:
        {audio, report}: the retimed track, and {words, pauses,
        duration_seconds, boundaries (with segment_frames),
        mid_word_boundaries}

    Raises:
        ValueError: If no words are heard, or a pause class is set past
            MAX_PAUSE_SECONDS (60)
    """
    lengths = dict(
        zip(PAUSE_CLASSES, (comma_seconds, sentence_seconds, section_seconds))
    )
    check_arguments(
        RETIME,
        **{f"{name}_seconds": value for name, value in lengths.items()},
        min_gap_seconds=min_gap_seconds,
        boundary_slack_seconds=boundary_slack_seconds,
    )
    for name, value in lengths.items():
        if float(value) > MAX_PAUSE_SECONDS:
            raise ValueError(
                f"{RETIME}: '{name}_seconds' is {value}, and a pause is at most "
                f"{MAX_PAUSE_SECONDS:g} seconds"
            )
    grid = _checked_grid(RETIME, segment_frames, guide_frames, fps)
    waveform, rate = waveform_and_rate(audio, sample_rate, RETIME)
    model = model_source({"model_name": model_name})
    words = word_map(waveform, rate, text, device, model)
    if not words:
        raise _no_words(RETIME)
    gaps = [words[i + 1]["start"] - words[i]["end"] for i in range(len(words) - 1)]
    eligible = [i for i, g in enumerate(gaps) if g > 0 and g >= min_gap_seconds]
    targets = list(gaps)
    for i in eligible:
        if words[i]["class"] is not None:
            targets[i] = float(lengths[words[i]["class"]])
    if grid is not None:
        _place_boundaries(
            words,
            gaps,
            targets,
            eligible,
            grid,
            boundary_slack_seconds,
            waveform.shape[1] / rate,
        )
    retimed = _render(waveform, rate, _edits(words, gaps, targets, rate))
    report = _report(words, gaps, targets, eligible, retimed.shape[1] / rate, grid)
    logger.info(
        f"{RETIME}: {len(report['pauses'])} pauses set, "
        f"{len(report['mid_word_boundaries'])} boundaries inside a word"
    )
    return {"audio": as_track(retimed, rate, RETIME), "report": JsonRecord(report)}


_PLACEHOLDER = re.compile(r"\{\{|\}\}|\{([^{}]*)\}|[{}]")
MAX_PROMPT_CHARS = 20_000


def filled_prompts(prompt_template, fill, lines):
    """The template once per line, `{line}` and every `fill` key substituted
    by plain text replacement. `{{` and `}}` are literal braces; any other
    brace - a format spec, a conversion, an attribute or index, a name `fill`
    does not give - is refused, naming it. A filled prompt over
    MAX_PROMPT_CHARS is refused, naming its segment."""
    if fill is not None and not isinstance(fill, dict):
        raise ValueError(f"{LINES}: 'fill' must be an object of placeholder values")
    values = {str(key): str(value) for key, value in (fill or {}).items()}
    allowed = {**values, "line": None}

    def check(match):
        token = match.group(0)
        if token in ("{{", "}}"):
            return
        name = match.group(1)
        if name not in allowed:
            raise ValueError(
                f"{LINES}: prompt_template has the placeholder {token!r}, which "
                f"is not a bare name 'fill' gives (it gives "
                f"{sorted(values) or 'nothing'}; {{line}} is the segment's line; "
                f"write {{{{ and }}}} for literal braces)"
            )

    if len(prompt_template) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"{LINES}: prompt_template is {len(prompt_template)} characters, "
            f"over the {MAX_PROMPT_CHARS} cap"
        )
    for name, value in (*values.items(), *(("line", line) for line in lines)):
        if len(value) > MAX_PROMPT_CHARS:
            raise ValueError(
                f"{LINES}: the value for {name!r} is {len(value)} characters, "
                f"over the {MAX_PROMPT_CHARS} cap"
            )
    matches = list(_PLACEHOLDER.finditer(prompt_template))
    for match in matches:
        check(match)
    prompts = []
    for index, line in enumerate(lines):
        pieces = {**values, "line": line}
        size = 0
        end = 0
        for match in matches:
            size += match.start() - end
            end = match.end()
            token = match.group(0)
            size += 1 if token in ("{{", "}}") else len(pieces[match.group(1)])
            if size > MAX_PROMPT_CHARS:
                break
        else:
            size += len(prompt_template) - end
        if size > MAX_PROMPT_CHARS:
            raise ValueError(
                f"{LINES}: the prompt for segment {index} is over the "
                f"{MAX_PROMPT_CHARS} character cap"
            )

        def put(match):
            token = match.group(0)
            if token in ("{{", "}}"):
                return token[0]
            return pieces[match.group(1)]

        prompts.append(_PLACEHOLDER.sub(put, prompt_template))
    return prompts


def _segment_lines(words, plan, fps, end_seconds):
    """Each planned segment's kept span and the words whose midpoint falls in
    it (the last segment keeps every word after its start)."""
    segments = []
    for s in plan:
        start = (s.audio_start_frame + s.head_trim) / fps
        end = min((s.audio_start_frame + s.num_frames) / fps, end_seconds)
        last = s is plan[-1]
        said = [
            w["say"]
            for w in words
            if w["say"]
            and start <= (w["start"] + w["end"]) / 2
            and (last or (w["start"] + w["end"]) / 2 < end)
        ]
        segments.append(
            {
                "index": s.index,
                "start_seconds": round(start, 4),
                "end_seconds": round(end, 4),
                "line": " ".join(said),
            }
        )
    return segments


def narration_lines(
    audio,
    segment_frames,
    guide_frames=0,
    fps=24,
    text=None,
    prompt_template=None,
    fill=None,
    model_name=WHISPER_DEFAULT_MODEL,
    sample_rate=None,
    device="cpu",
):
    """Task command: the line a narration speaks in each planned chain
    segment, and optionally a prompt per segment from a template.

    Args:
        audio: The narration (usually `retime_narration`'s audio): a path or
            URL, a track from an earlier step, or a waveform (with
            sample_rate)
        segment_frames: Frames a chain segment generates
        guide_frames: The chain's guide frames - the head every segment
            after the first trims
        fps: Frame rate the frames are counted at
        text: The script, whose spelling and punctuation the lines take
        prompt_template: A prompt with `{line}` and `fill`'s placeholders
        fill: {placeholder: value} for the template's other placeholders
        model_name: A Whisper-class ASR model for the word times
        sample_rate: Sample rate of a waveform passed directly
        device: Target device ("cuda", "mps", "cpu")

    Returns:
        {lines, prompts (with prompt_template), segments,
        mid_word_boundaries, total_frames}

    Raises:
        ValueError: If no words are heard, or the template names a
            placeholder `fill` does not give
    """
    from ..pipeline_processors.chain_config import plan_segments

    if segment_frames is None:
        raise ValueError(f"{LINES} needs 'segment_frames', the chain's segment length")
    segment_frames, guide_frames, fps = _checked_grid(
        LINES, segment_frames, guide_frames, fps
    )
    waveform, rate = waveform_and_rate(audio, sample_rate, LINES)
    words = word_map(
        waveform, rate, text, device, model_source({"model_name": model_name})
    )
    if not words:
        raise _no_words(LINES)
    duration = waveform.shape[1] / rate
    total = _frames(duration, fps)
    plan = plan_segments(total, segment_frames, guide_frames)
    segments = _segment_lines(words, plan, fps, total / fps)
    lines = [s["line"] for s in segments]
    answer = {"lines": lines}
    if prompt_template is not None:
        answer["prompts"] = filled_prompts(str(prompt_template), fill, lines)
    answer["segments"] = segments
    answer["mid_word_boundaries"] = [
        {"segment": s["index"], "word": w["word"]}
        for s in segments[:-1]
        for w in words
        if w["start"] + WORD_JITTER_SECONDS
        < s["end_seconds"]
        < w["end"] - WORD_JITTER_SECONDS
    ]
    answer["total_frames"] = total
    return answer


def _register():
    from .registry import register_command
    from ..task_domains import NON_NEGATIVE, POSITIVE

    frames = {
        "fps": POSITIVE,
        "segment_frames": POSITIVE,
        "guide_frames": NON_NEGATIVE,
        "sample_rate": POSITIVE,
    }
    seconds = (
        "comma_seconds",
        "sentence_seconds",
        "section_seconds",
        "min_gap_seconds",
        "boundary_slack_seconds",
    )

    @register_command(
        RETIME,
        implementation="dw.tasks.narration_retime.retime_narration",
        required_weights="dw.tasks.narration_retime.required_weights",
        consumes_device=True,
        domains={**frames, **{name: NON_NEGATIVE for name in seconds}},
        whole_numbers=("segment_frames", "guide_frames"),
    )
    def _handle_retime_narration(task, arguments, previous_pipelines):
        """Set a narration's pauses by punctuation, boundaries on a chain's grid"""
        return retime_narration(device=task.device_for(arguments), **arguments)

    @register_command(
        LINES,
        implementation="dw.tasks.narration_retime.narration_lines",
        required_weights="dw.tasks.narration_retime.required_weights",
        returns="json",
        consumes_device=True,
        domains=frames,
        whole_numbers=("segment_frames", "guide_frames"),
    )
    def _handle_narration_lines(task, arguments, previous_pipelines):
        """Read the line a narration speaks in each planned chain segment"""
        return narration_lines(device=task.device_for(arguments), **arguments)


_register()
