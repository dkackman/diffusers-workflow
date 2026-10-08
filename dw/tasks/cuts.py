"""plan_cuts: a music video's cut list from a song's lyrics and beats (#600).

A music video cuts where a line starts, holds while it is sung, and lands
the cut on the beat. `transcribe_audio` (with timestamps) says when each
line is sung and `analyze_beats` says where the beats fall; this turns the
two into shots - each a start frame and a length - that tile the song
exactly, so the agent writes one prompt per shot and renders the list.

The shape of it:
- Lines. Each transcript chunk is a sung line. Given the song's own lyrics,
  those are the lines instead - their text and order exactly - and the
  transcript only lends them timings, aligned word by word, since Whisper
  mishears sung words and splits lines where it likes.
- Scenes. A shot per line, per stanza, or per few beats (`segment_by`).
  The silence between lines goes to the shot before it, or becomes its own
  instrumental shot when it is long enough and asked for.
- Lengths. A shot over `max_scene_s` splits (on a beat when there are
  beats); one under `min_scene_s` merges into its shorter neighbour. A
  shot still outside the range is warned about by name.
- Frames. Every boundary is rounded once, from its absolute time, so the
  shots' frame counts sum to the song's and no rounding drifts.
- Render grid. A model renders only some lengths (a modulus and remainder),
  never fewer than `min_frames` or more than `max_frames`, and a shot may
  start `lead_s` early to give the model a run-up to trim off. All of it
  comes in as arguments: a shot's `num_frames` is its lead plus its cut,
  raised to the grid, and a shot too long for `max_frames` splits.

It answers JSON - a dict, since a list result would become one artifact per
shot - and builds nothing.
"""

import difflib
import logging
import math
import re

from ..variable_constraints import aligned, aligned_down
from .registry import register_command

logger = logging.getLogger("dw")

COMMAND = "plan_cuts"

SEGMENT_BY = ("line", "stanza", "beat")
# A lyric line that is only a section tag ("[Chorus]") is structure, not a
# sung line: it breaks a stanza, as a blank line does
SECTION_TAG = re.compile(r"^\s*\[[^\]]*\]\s*$")
# Two words are the same word sung, misheard by at most this much
WORD_SIMILARITY = 0.75
TIME_PLACES = 4


def _number(value, name, required=False):
    """A float from a number or numeric string; None stays None unless
    required."""
    if value is None:
        if required:
            raise ValueError(f"{COMMAND} needs '{name}'")
        return None
    if isinstance(value, bool):
        raise ValueError(f"{COMMAND} needs a number for '{name}', got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{COMMAND} needs a number for '{name}', got {value!r}"
        ) from error
    if not math.isfinite(number):
        raise ValueError(f"{COMMAND} needs a finite number for '{name}'")
    return number


def _integer(value, name):
    """An int from a whole number or numeric string; None stays None."""
    number = _number(value, name)
    if number is None:
        return None
    if number != int(number):
        raise ValueError(f"{COMMAND} needs a whole number for '{name}', got {value!r}")
    return int(number)


def _grid_up(frames, grid):
    """The smallest render length on the grid at or above `frames`; `frames`
    itself when there is no modulus. The arithmetic is the template
    constraint's own (`variable_constraints.aligned`), so a planned
    `num_frames` is one the template's constraint accepts."""
    target = aligned(frames, grid)
    return frames if target is None else target


def _grid_down(frames, grid):
    """The largest render length on the grid at or below `frames` (`frames`
    itself when there is no modulus), or None when none fits."""
    if not grid["modulus"]:
        return frames
    return aligned_down(frames, grid)


def transcript_problem(transcript):
    """Why this transcript can't be planned from, or None. A bare string is
    text without timings - the commonest mistake, so it is named."""
    if isinstance(transcript, str):
        return (
            f"{COMMAND} needs a timestamped transcript - {{text, chunks: "
            "[{start, end, text}]}, not plain text: run transcribe_audio with "
            "'timestamps': \"segment\" (Whisper's return_timestamps), its "
            "result's content_type application/json"
        )
    if isinstance(transcript, dict):
        chunks = transcript.get("chunks")
        if not isinstance(chunks, list):
            return (
                f"{COMMAND}'s 'transcript' has no 'chunks' list - run "
                "transcribe_audio with 'timestamps' set (Whisper's "
                "return_timestamps) for the {text, chunks} shape"
            )
        return None
    if isinstance(transcript, list):
        return None
    return (
        f"{COMMAND}'s 'transcript' is a {{text, chunks}} dict from "
        "transcribe_audio with 'timestamps' set (Whisper's return_timestamps), "
        f"not {type(transcript).__name__}"
    )


def _chunks(transcript, duration):
    """[{start, end, text}] in seconds, ascending. A null end takes the next
    chunk's start, or the song's end (None when the song's end is not yet
    known)."""
    problem = transcript_problem(transcript)
    if problem:
        raise ValueError(problem)
    raw = transcript["chunks"] if isinstance(transcript, dict) else transcript
    parsed = []
    for index, chunk in enumerate(raw):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"{COMMAND}: transcript chunk {index} is not a "
                "{start, end, text} object"
            )
        start = chunk.get("start")
        if start is None and isinstance(chunk.get("timestamp"), (list, tuple)):
            # Whisper's raw pipeline shape: timestamp: [start, end]
            start, end = (list(chunk["timestamp"]) + [None, None])[:2]
        else:
            end = chunk.get("end")
        start = _number(start, f"transcript chunks[{index}].start", required=True)
        end = _number(end, f"transcript chunks[{index}].end")
        parsed.append(
            {"start": start, "end": end, "text": str(chunk.get("text") or "").strip()}
        )
    parsed.sort(key=lambda chunk: chunk["start"])
    for index, chunk in enumerate(parsed):
        if chunk["end"] is None:
            following = parsed[index + 1]["start"] if index + 1 < len(parsed) else None
            chunk["end"] = following if following is not None else duration
        if chunk["end"] is not None and chunk["end"] < chunk["start"]:
            chunk["end"] = chunk["start"]
    return parsed


def _beats(beats):
    """(beat seconds ascending, bpm or None, song duration or None) from an
    analyze_beats result or a bare list of seconds."""
    if beats is None:
        return [], None, None
    bpm = duration = None
    times = beats
    if isinstance(beats, dict):
        times = beats.get("beats") or []
        bpm = _number(beats.get("bpm"), "beats.bpm")
        duration = _number(beats.get("duration_seconds"), "beats.duration_seconds")
    if not isinstance(times, (list, tuple)):
        raise ValueError(
            f"{COMMAND}'s 'beats' is an analyze_beats result or a list of "
            f"seconds, not {type(times).__name__}"
        )
    times = sorted(_number(time, f"beats[{index}]") for index, time in enumerate(times))
    if bpm is None and len(times) >= 2:
        gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
        if gaps:
            bpm = 60.0 / gaps[len(gaps) // 2]
    return times, bpm, duration


def _lyric_lines(lyrics):
    """[(text, stanza index)] for the sung lines of the lyrics. A blank line
    or a section tag on its own line starts a new stanza; neither is a line."""
    if isinstance(lyrics, str):
        rows = lyrics.splitlines()
    elif isinstance(lyrics, (list, tuple)):
        rows = []
        for row in lyrics:
            rows.extend(str(row).splitlines() or [""])
    else:
        raise ValueError(
            f"{COMMAND}'s 'lyrics' is the song's text, one line per line, or a "
            f"list of lines - not {type(lyrics).__name__}"
        )
    lines, stanza, broke = [], 0, False
    for row in rows:
        text = row.strip()
        if not text or SECTION_TAG.match(text):
            broke = bool(lines)
            continue
        if broke:
            stanza += 1
            broke = False
        lines.append((text, stanza))
    if not lines:
        raise ValueError(f"{COMMAND}'s 'lyrics' has no sung lines")
    return lines


def _words(text):
    """The text's words, lower-cased with punctuation dropped."""
    return [word for word in re.findall(r"[\w']+", text.lower().replace("'", ""))]


def _timed_words(chunks):
    """[(word, start, end)] across the transcript: a chunk's span shared out
    over its words by length, since a segment chunk carries no word times."""
    timed = []
    for chunk in chunks:
        words = _words(chunk["text"])
        if not words:
            continue
        span = chunk["end"] - chunk["start"]
        total = sum(len(word) for word in words)
        at = chunk["start"]
        for word in words:
            length = span * len(word) / total
            timed.append((word, at, at + length))
            at += length
    return timed


def _similar(a, b):
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= WORD_SIMILARITY


def _align_words(lyric_words, heard_words):
    """{lyric word index: heard word index} - the longest in-order run of
    words the two share, a word matching one misheard by a little."""
    n, m = len(lyric_words), len(heard_words)
    # Longest common subsequence by similarity, filled from the end
    table = [[0] * (m + 1) for _ in range(n + 1)]
    same = {}
    for i in range(n - 1, -1, -1):
        row, below = table[i], table[i + 1]
        for j in range(m - 1, -1, -1):
            if _similar(lyric_words[i], heard_words[j]):
                same[i, j] = True
                row[j] = below[j + 1] + 1
            else:
                row[j] = max(below[j], row[j + 1])
    pairs, i, j = {}, 0, 0
    while i < n and j < m:
        if same.get((i, j)) and table[i][j] == table[i + 1][j + 1] + 1:
            pairs[i] = j
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _align_lyrics(lyric_lines, chunks, duration, warnings, min_line_s):
    """[{start, end, text, stanza}] - the lyrics' lines in their order, timed
    from the transcript. A line none of whose words were heard is placed
    between its neighbours, and warned about; when they touch it takes
    min_line_s from them, so it is still a shot of its own."""
    heard = _timed_words(chunks)
    lyric_words, owner = [], []
    for index, (text, _) in enumerate(lyric_lines):
        for word in _words(text):
            lyric_words.append(word)
            owner.append(index)
    pairs = _align_words(lyric_words, [word for word, _, _ in heard])

    timed = [None] * len(lyric_lines)
    for lyric_index, heard_index in pairs.items():
        line = owner[lyric_index]
        _, start, end = heard[heard_index]
        if timed[line] is None:
            timed[line] = [start, end]
        else:
            timed[line][0] = min(timed[line][0], start)
            timed[line][1] = max(timed[line][1], end)

    # A line's span runs no further than the next timed line's start
    known = [index for index, span in enumerate(timed) if span is not None]
    for a, b in zip(known, known[1:]):
        timed[a][1] = min(timed[a][1], timed[b][0])

    if not known:
        warnings.append(
            f"{COMMAND}: none of the lyrics' words were heard in the transcript "
            "- the lines are spread evenly over the sung part of the song; "
            "check the transcript is of this song"
        )
    vocal_start = chunks[0]["start"] if chunks else 0.0
    vocal_end = chunks[-1]["end"] if chunks else duration
    index = 0
    while index < len(timed):
        if timed[index] is not None:
            index += 1
            continue
        run_end = index
        while run_end < len(timed) and timed[run_end] is None:
            run_end += 1
        following = timed[run_end][0] if run_end < len(timed) else None
        if index > 0:
            low = timed[index - 1][1]
        else:
            low = vocal_start if following is None else min(vocal_start, following)
        high = following if following is not None else max(vocal_end, low)
        count = run_end - index
        # The neighbours took the unheard line's time with its misheard words:
        # give it back from each side, never more than half a neighbour
        short = count * min_line_s - (high - low)
        if short > 1e-9:
            before = timed[index - 1] if index > 0 else None
            after = timed[run_end] if run_end < len(timed) else None
            spare_before = (before[1] - before[0]) / 2 if before else 0.0
            spare_after = (after[1] - after[0]) / 2 if after else 0.0
            take_before = min(spare_before, max(short / 2, short - spare_after))
            take_after = min(spare_after, short - take_before)
            low -= take_before
            high += take_after
            if before:
                before[1] = low
            if after:
                after[0] = high
        step = (high - low) / count
        for offset in range(count):
            timed[index + offset] = [low + offset * step, low + (offset + 1) * step]
            if known:
                warnings.append(
                    f"{COMMAND}: lyric line {index + offset + 1} "
                    f"({lyric_lines[index + offset][0]!r}) was not heard in the "
                    f"transcript - placed between its neighbours, at "
                    f"{timed[index + offset][0]:.2f}-{timed[index + offset][1]:.2f} s"
                )
        index = run_end
    return [
        {"start": span[0], "end": span[1], "text": text, "stanza": stanza}
        for span, (text, stanza) in zip(timed, lyric_lines)
    ]


def _transcript_lines(chunks, stanza_gap):
    """[{start, end, text, stanza}] - a line per chunk; a silence of at least
    stanza_gap starts a new stanza."""
    lines, stanza = [], 0
    for index, chunk in enumerate(chunks):
        if not chunk["text"]:
            continue
        if lines and chunk["start"] - lines[-1]["end"] >= stanza_gap:
            stanza += 1
        lines.append(dict(chunk, stanza=stanza))
    return lines


def _units(lines, segment_by):
    """The vocal spans scenes are cut around: a line each, or a stanza."""
    units = []
    for line in lines:
        if segment_by == "stanza" and units and units[-1]["stanza"] == line["stanza"]:
            units[-1]["end"] = max(units[-1]["end"], line["end"])
            units[-1]["lyrics"].append(line["text"])
        else:
            units.append(
                {
                    "start": line["start"],
                    "end": line["end"],
                    "lyrics": [line["text"]],
                    "stanza": line["stanza"],
                }
            )
    return units


def _scene(start, end, kind, lyrics):
    return {"start": start, "end": end, "kind": kind, "lyrics": list(lyrics)}


def _scenes_by_units(units, duration, args, warnings):
    """Scenes tiling [0, duration): a vocal scene per unit, its tail and the
    silence after it held until the next unit starts - unless that silence
    is long enough to be its own instrumental scene and gaps were asked for."""
    if not units:
        return [_scene(0.0, duration, "instrumental", [])]
    gaps = args.include_instrumental_gaps
    scenes = []
    first = max(0.0, min(units[0]["start"], duration))
    if gaps and first >= args.min_gap_seconds:
        scenes.append(_scene(0.0, first, "instrumental", []))
        cursor = first
    else:
        cursor = 0.0
    for index, unit in enumerate(units):
        following = units[index + 1]["start"] if index + 1 < len(units) else duration
        following = min(max(following, cursor), duration)
        sung_to = min(max(unit["end"] + args.vocal_tail_s, cursor), following)
        if gaps and following - sung_to >= args.min_gap_seconds:
            scenes.append(_scene(cursor, sung_to, "vocal", unit["lyrics"]))
            scenes.append(_scene(sung_to, following, "instrumental", []))
        else:
            scenes.append(_scene(cursor, following, "vocal", unit["lyrics"]))
        cursor = following
    # A line squeezed to no time between touching neighbours keeps its place
    # in the next scene rather than leaving the lyrics
    return _merge_empty(scenes, warnings, "placing the lines")


def _scenes_by_beats(lines, beats, duration, args):
    """Scenes cut on beats, each the first beat at least min_scene_s after
    the last cut; a scene's lyric is the lines that start in it."""
    cuts, last = [0.0], 0.0
    for beat in beats:
        if beat >= last + max(args.min_scene_s, 1e-9) and beat < duration:
            cuts.append(beat)
            last = beat
    cuts.append(duration)
    scenes = []
    for start, end in zip(cuts, cuts[1:]):
        if end - start < max(args.min_scene_s, 1e-9) and scenes:
            # The song ends less than a scene after the last beat cut
            scenes[-1]["end"] = end
            continue
        scenes.append(_scene(start, end, "instrumental", []))
    for line in lines:
        for scene in scenes:
            if scene["start"] <= line["start"] < scene["end"] or (
                scene is scenes[-1] and line["start"] >= scene["end"]
            ):
                scene["lyrics"].append(line["text"])
                break
    for scene in scenes:
        if any(
            line["start"] < scene["end"] and line["end"] > scene["start"]
            for line in lines
        ):
            scene["kind"] = "vocal"
    return scenes


def _split_long(scenes, beats, max_scene_s, warnings):
    """Each scene over max_scene_s cut into the fewest even pieces under it,
    each cut moved to the nearest beat inside the scene when there are beats.
    A piece of a sung scene keeps its lyric, and the split is warned about,
    since the same words then carry several shots."""
    if max_scene_s is None:
        return scenes
    result = []
    for scene in scenes:
        length = scene["end"] - scene["start"]
        if length <= max_scene_s + 1e-9:
            result.append(scene)
            continue
        pieces = math.ceil(length / max_scene_s - 1e-9)
        cuts, previous = [], scene["start"]
        for k in range(1, pieces):
            target = scene["start"] + k * length / pieces
            inside = [b for b in beats if previous < b < scene["end"]]
            if inside:
                target = min(inside, key=lambda beat: abs(beat - target))
            if previous < target < scene["end"]:
                cuts.append(target)
                previous = target
        edges = [scene["start"], *cuts, scene["end"]]
        if scene["lyrics"] and len(edges) > 2:
            warnings.append(
                f"{COMMAND}: {' / '.join(scene['lyrics'])!r} lasts {length:.2f} s, "
                f"over max_scene_s ({max_scene_s:g} s) - split into "
                f"{len(edges) - 1} shots, each carrying its lyric"
            )
        for start, end in zip(edges, edges[1:]):
            result.append(_scene(start, end, scene["kind"], scene["lyrics"]))
    return result


def _snap(scenes, beats, warnings):
    """Every cut between scenes moved to its nearest beat. Two cuts landing
    on one beat leave a scene with no length, which merges into the scene
    after it."""
    if not beats:
        warnings.append(
            f"{COMMAND}: snap_to_beats was asked for and there are no beats - "
            "the cuts stay where the lines put them"
        )
        return scenes
    for before, after in zip(scenes, scenes[1:]):
        cut = after["start"]
        nearest = min(beats, key=lambda beat: abs(beat - cut))
        before["end"] = after["start"] = nearest
    return _merge_empty(scenes, warnings, "snapping to the beat")


def _merge_into(scenes, index, other):
    """Scene `index` merged into its neighbour `other`, in place."""
    low, high = sorted((index, other))
    a, b = scenes[low], scenes[high]
    merged = _scene(
        a["start"],
        b["end"],
        "vocal" if "vocal" in (a["kind"], b["kind"]) else "instrumental",
        _joined(a["lyrics"], b["lyrics"]),
    )
    scenes[low : high + 1] = [merged]


def _joined(first, second):
    """Two scenes' lines as one; a line split across both is kept once."""
    if first and second and first[-1] == second[0]:
        return first + second[1:]
    return first + second


def _merge_empty(scenes, warnings, why):
    index = 0
    while index < len(scenes) and len(scenes) > 1:
        scene = scenes[index]
        if scene["end"] - scene["start"] > 1e-9:
            index += 1
            continue
        if scene["lyrics"]:
            warnings.append(
                f"{COMMAND}: {why} left no time for "
                f"{' / '.join(scene['lyrics'])!r}; it joins the next scene"
            )
        _merge_into(scenes, index, index + 1 if index + 1 < len(scenes) else index - 1)
    return scenes


def _merge_short(scenes, min_scene_s):
    """Each scene under min_scene_s merged into its shorter neighbour,
    shortest first, until none is short or one scene is left."""
    while len(scenes) > 1:
        short = [
            index
            for index, scene in enumerate(scenes)
            if scene["end"] - scene["start"] < min_scene_s - 1e-9
        ]
        if not short:
            break
        index = min(short, key=lambda i: scenes[i]["end"] - scenes[i]["start"])
        neighbours = [i for i in (index - 1, index + 1) if 0 <= i < len(scenes)]
        other = min(neighbours, key=lambda i: scenes[i]["end"] - scenes[i]["start"])
        _merge_into(scenes, index, other)
    return scenes


def _frames(scenes, fps, total_frames, warnings):
    """(start frame, cut frames) per scene, each boundary rounded once from
    its absolute time. A scene rounding to no frames joins its neighbour."""
    starts = [int(round(scene["start"] * fps)) for scene in scenes] + [total_frames]
    starts[0] = 0
    framed, carried = [], []
    for scene, start, end in zip(scenes, starts, starts[1:]):
        end = min(end, total_frames)
        if end <= start:
            if scene["lyrics"]:
                warnings.append(
                    f"{COMMAND}: {' / '.join(scene['lyrics'])!r} is shorter than "
                    "a frame and joins a neighbouring shot"
                )
                if framed:
                    framed[-1][0]["lyrics"] = _joined(
                        framed[-1][0]["lyrics"], scene["lyrics"]
                    )
                else:
                    carried = _joined(carried, scene["lyrics"])
            continue
        if carried:
            scene["lyrics"] = _joined(carried, scene["lyrics"])
            scene["kind"], carried = "vocal", []
        framed.append((scene, start, end - start))
    return framed


def _render_length(grid, lead, cut):
    """Frames to render for a cut with this lead: lead + cut, no fewer than
    min_frames, raised to the grid."""
    return _grid_up(max(grid["min_frames"] or 0, lead + cut), grid)


def _split_over_max(framed, grid, beat_frames, warnings):
    """Each (scene, start, cut) whose render would pass max_frames cut into
    pieces that fit, each cut on the beat nearest an even split when one lies
    in reach. The longest cut a piece may have is the longest grid length at
    or under max_frames, less the piece's own lead (a piece starting early in
    the song has less). A piece of a sung scene keeps its lyric, and the split
    is warned about, since the same words then carry several shots."""
    if grid["max_frames"] is None:
        return framed
    ceiling = _grid_down(grid["max_frames"], grid)

    def reach(start):
        return ceiling - min(grid["lead_frames"], start)

    result = []
    for scene, start, cut in framed:
        end = start + cut
        if cut <= reach(start):
            result.append((scene, start, cut))
            continue
        needed = _render_length(grid, min(grid["lead_frames"], start), cut)
        pieces, at = [], start
        while end - at > reach(at):
            room = reach(at)
            count = math.ceil((end - at) / room)
            target = at + (end - at) / count
            inside = [b for b in beat_frames if at < b <= at + room]
            if inside:
                cut_at = min(inside, key=lambda beat: abs(beat - target))
            else:
                cut_at = min(at + room, max(at + 1, int(round(target))))
            pieces.append((at, cut_at - at))
            at = cut_at
        pieces.append((at, end - at))
        label = " / ".join(scene["lyrics"]) or f"frames {start}-{end}"
        warnings.append(
            f"{COMMAND}: {label!r} needs {needed} rendered frames, over "
            f"max_frames ({grid['max_frames']}) - split into {len(pieces)} shots"
        )
        for piece_start, piece_cut in pieces:
            piece = _scene(scene["start"], scene["end"], scene["kind"], scene["lyrics"])
            result.append((piece, piece_start, piece_cut))
    return result


def plan_cuts(
    transcript,
    lyrics=None,
    beats=None,
    segment_by="line",
    fps=24,
    duration_s=None,
    min_scene_s=1.0,
    max_scene_s=None,
    vocal_tail_s=0.0,
    include_instrumental_gaps=True,
    min_gap_seconds=2.0,
    snap_to_beats=False,
    modulus=None,
    remainder=None,
    min_frames=None,
    max_frames=None,
    lead_s=None,
):
    """Plan a music video's cuts from a song's transcript, lyrics and beats.

    Args:
        transcript: transcribe_audio's result with 'timestamps' set - a
            {text, chunks} dict of {start, end, text} chunks in seconds. A
            null 'end' runs to the next chunk's start, or the song's end
        lyrics: The song's own lyrics - one line per line, a blank line or a
            lone section tag ("[Chorus]") between stanzas - or a list of
            lines. The shots carry these lines verbatim and in order; the
            transcript only times them
        beats: analyze_beats' result, or a list of beat times in seconds
        segment_by: "line" (a shot per sung line), "stanza" (per stanza:
            blank-line groups of the lyrics, or lines without a silence of
            min_gap_seconds between them) or "beat" (a cut on the first beat
            at least min_scene_s after the last; needs beats)
        fps: Frames per second the shots are counted in
        duration_s: The song's length; analyze_beats' duration_seconds when
            omitted, else the transcript's last end
        min_scene_s: The shortest a shot may be; a shorter one merges into
            its shorter neighbour
        max_scene_s: The longest a shot may be; a longer one splits evenly,
            each cut on its nearest beat when there are beats
        vocal_tail_s: Seconds a sung shot holds past its last word, for the
            breath and the held note's tail
        include_instrumental_gaps: Whether a silence between lines (or
            before the first, or after the last) of at least min_gap_seconds
            becomes its own instrumental shot, rather than the shot before
            it holding over it
        min_gap_seconds: The shortest silence that becomes its own shot
        snap_to_beats: Whether every cut moves to its nearest beat
        modulus: The render grid's step: a shot renders 'modulus * n +
            remainder' frames. Omitted, any length renders
        remainder: The grid's offset, a whole number from 0 below modulus
            (0 when omitted); needs modulus
        min_frames: The fewest frames a shot renders. A shot shorter than its
            lead plus cut renders this many, then the grid raises it, so
            num_frames is always on the grid even when min_frames is not
        max_frames: The most frames a shot renders; a shot needing more
            splits (on a beat when one is in reach) and is warned about. It
            must allow at least the smallest grid length at or above
            min_frames
        lead_s: Seconds a shot starts before its cut, as a run-up the render
            is trimmed of; the first shot has none, and no shot leads past
            the song's start. Omitted, none

    Returns:
        {shots, bpm, fps, duration_s, total_frames, render_frames, warnings}:
        `shots` tile the song in order, each {name, start_frame, num_frames,
        cut_frames, lead_frames, lyric, kind} - `start_frame` and
        `cut_frames` the cut on the timeline, `lead_frames` the run-up
        before it (min(round(lead_s * fps), start_frame)) and `num_frames`
        the length to render: lead_frames + cut_frames, no less than
        min_frames, raised to the grid. With no grid arguments that is
        cut_frames and lead_frames is 0. `lyric` is the lines sung in it
        joined by newlines or null, `kind` "vocal" or "instrumental".
        `render_frames` is the sum of the shots' num_frames
    """
    import types

    from ..events import emit_warning
    from ..task_domains import check_arguments, cuts_problems

    args = types.SimpleNamespace(
        fps=_number(fps, "fps", required=True),
        duration_s=_number(duration_s, "duration_s"),
        min_scene_s=_number(min_scene_s, "min_scene_s") or 0.0,
        max_scene_s=_number(max_scene_s, "max_scene_s"),
        vocal_tail_s=_number(vocal_tail_s, "vocal_tail_s") or 0.0,
        min_gap_seconds=_number(min_gap_seconds, "min_gap_seconds") or 0.0,
        include_instrumental_gaps=bool(include_instrumental_gaps),
        modulus=_integer(modulus, "modulus"),
        remainder=_integer(remainder, "remainder"),
        min_frames=_integer(min_frames, "min_frames"),
        max_frames=_integer(max_frames, "max_frames"),
        lead_s=_number(lead_s, "lead_s"),
    )
    check_arguments(
        COMMAND,
        fps=args.fps,
        duration_s=args.duration_s,
        min_scene_s=args.min_scene_s,
        max_scene_s=args.max_scene_s,
        vocal_tail_s=args.vocal_tail_s,
        min_gap_seconds=args.min_gap_seconds,
        modulus=args.modulus,
        remainder=args.remainder,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        lead_s=args.lead_s,
    )
    if segment_by not in SEGMENT_BY:
        raise ValueError(
            f"{COMMAND}'s 'segment_by' is one of {', '.join(SEGMENT_BY)}, "
            f"not {segment_by!r}"
        )
    problems = cuts_problems(
        transcript=transcript,
        segment_by=segment_by,
        min_scene_s=args.min_scene_s,
        max_scene_s=args.max_scene_s,
        beats=beats,
        modulus=args.modulus,
        remainder=args.remainder,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )
    if problems:
        raise ValueError("; ".join(message for _, message in problems))

    beat_times, bpm, beats_duration = _beats(beats)
    duration = args.duration_s or beats_duration
    chunks = _chunks(transcript, duration)
    warnings = []
    if duration is None:
        # Only the transcript says how long: to its last word, which is no
        # end at all when that word's end is null
        if not chunks or chunks[-1]["end"] is None:
            raise ValueError(
                f"{COMMAND} can't tell how long the song is - pass 'duration_s', "
                "or the analyze_beats result as 'beats'"
            )
        duration = max(chunk["end"] for chunk in chunks)
        if beats is None:
            why = "no 'duration_s' or beats"
        else:
            # 'previous_result:<step>' into 'beats' hands over only the
            # analyze_beats result's 'beats' list, the key the argument names
            why = (
                "no 'duration_s', and 'beats' is a bare list of times with no "
                "duration_seconds (a previous_result: into 'beats' passes only "
                "the list - pass 'previous_result:<step>.duration_seconds' as "
                "'duration_s')"
            )
        warnings.append(
            f"{COMMAND}: {why} - the plan ends with the transcript's last line, "
            f"at {duration:.2f} s"
        )
    beat_times = [beat for beat in beat_times if 0.0 < beat < duration]

    if lyrics is not None and not (isinstance(lyrics, str) and not lyrics.strip()):
        lines = _align_lyrics(
            _lyric_lines(lyrics), chunks, duration, warnings, max(args.min_scene_s, 0.5)
        )
    else:
        gap = 2.0 if args.min_gap_seconds is None else args.min_gap_seconds
        lines = _transcript_lines(chunks, gap)
        if not lines:
            warnings.append(
                f"{COMMAND}: the transcript has no sung lines - the song is "
                "planned as instrumental"
            )
    for line in lines:
        line["start"] = min(max(line["start"], 0.0), duration)
        line["end"] = min(max(line["end"], line["start"]), duration)

    if segment_by == "beat":
        scenes = _scenes_by_beats(lines, beat_times, duration, args)
    else:
        scenes = _scenes_by_units(_units(lines, segment_by), duration, args, warnings)
    scenes = _split_long(scenes, beat_times, args.max_scene_s, warnings)
    if snap_to_beats:
        scenes = _snap(scenes, beat_times, warnings)
    scenes = _merge_short(scenes, args.min_scene_s)

    total_frames = int(round(duration * args.fps))
    framed = _frames(scenes, args.fps, total_frames, warnings)
    grid = {
        "modulus": args.modulus,
        "remainder": args.remainder or 0,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "lead_frames": int(round((args.lead_s or 0.0) * args.fps)),
    }
    if args.max_frames is not None and grid["lead_frames"] >= _grid_down(
        args.max_frames, grid
    ):
        raise ValueError(
            f"{COMMAND}'s 'lead_s' ({args.lead_s:g} s, {grid['lead_frames']} "
            f"frames) leaves no room under 'max_frames' ({args.max_frames}) "
            "for a shot's own frames"
        )
    beat_frames = [int(round(beat * args.fps)) for beat in beat_times]
    framed = _split_over_max(framed, grid, beat_frames, warnings)
    width = max(2, len(str(len(framed))))
    shots = []
    for number, (scene, start, cut) in enumerate(framed, start=1):
        name = f"shot_{number:0{width}d}"
        seconds = cut / args.fps
        if seconds < args.min_scene_s - 0.5 / args.fps:
            warnings.append(
                f"{COMMAND}: {name} is {seconds:.2f} s, under min_scene_s "
                f"({args.min_scene_s:g} s) - the song has no room to lengthen it"
            )
        if args.max_scene_s is not None and seconds > args.max_scene_s + 0.5 / args.fps:
            warnings.append(
                f"{COMMAND}: {name} is {seconds:.2f} s, over max_scene_s "
                f"({args.max_scene_s:g} s) - merging a shorter neighbour into "
                "it left it long"
            )
        lead = min(grid["lead_frames"], start)
        shots.append(
            {
                "name": name,
                "start_frame": start,
                "num_frames": _render_length(grid, lead, cut),
                "cut_frames": cut,
                "lead_frames": lead,
                "lyric": "\n".join(scene["lyrics"]) or None,
                "kind": scene["kind"],
            }
        )

    for message in warnings:
        emit_warning(message, kind="plan_cuts", command=COMMAND)
    return {
        "shots": shots,
        "bpm": None if bpm is None else round(float(bpm), 2),
        "fps": args.fps,
        "duration_s": round(duration, TIME_PLACES),
        "total_frames": total_frames,
        "render_frames": sum(shot["num_frames"] for shot in shots),
        "warnings": warnings,
    }


@register_command(COMMAND, implementation="dw.tasks.cuts.plan_cuts", returns="json")
def _handle_plan_cuts(task, arguments, previous_pipelines):
    """Plan a music video's cuts from a song's transcript, lyrics and beats"""
    logger.debug("Planning cuts")
    return plan_cuts(**arguments)
