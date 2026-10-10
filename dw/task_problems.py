"""The cross-argument rules of the task families that need more than a domain.

A domain (dw/task_domains.py) is one number against one range. What lives here
is everything a single range cannot say: crop_face_track's crop size on the
video model's frame grid, gate_zero above gate_full; analyze_beats' tempo range
and anchors that are one kind and in time order; plan_cuts' transcript shape
and render grid; apply_lut's one-of-two source and its palette; check_script's
`lines` and `shots`; ingredients_grid's choices and image count. Each family's
`*_problems` function is the rule, and both sides call it - its `*_errors`
static check (registered as the command's `static_check`, run by
`task_argument_errors` at validate) for the literal values a workflow carries,
and the task itself at run time for the values it was actually handed - so a
value is refused at both or at neither, in the same words (#692).

They were split out of task_domains.py when it passed the architecture
ratchet's size ceiling (#790). The direction is one way: these rules read the
generic helpers there (`as_number`, `choice_errors`), and task_domains never
imports this module - `task_argument_errors` reaches a rule only through the
registry's `static_check` table - so the `import_cycles` ratchet stays at 0.
The media-timing rules (window, slice, fit, dissolve, frame size, select) stay
in task_domains.py, beside the comment that explains why they have one home.
"""

import re

from .references import DEFERRED, GATHER, is_ref
from .script_lines import parse_lines, parse_shots, shot_names_error
from .task_domains import as_number, choice_errors

# ingredients_grid's literal choices, read by its registration and by
# ingredients_grid_errors below, and its default image cap
INGREDIENTS_LAYOUTS = ("auto", "rows", "panels")
INGREDIENTS_FITS = ("contain", "cover")
INGREDIENTS_DEFAULT_MAX_IMAGES = 12


def ingredients_background(background):
    """The RGB tuple for an ingredients_grid `background`, or ValueError worded
    as the command's refusal. The one parse, shared by the static check and
    the run."""
    from PIL import ImageColor

    try:
        return ImageColor.getrgb(background)
    except (ValueError, AttributeError):
        raise ValueError(
            "ingredients_grid needs 'background' as a colour name or "
            f"#hex, got {background!r}"
        )


def ingredients_grid_errors(arguments):
    """[(argument, message)] for the ingredients_grid rules a literal
    workflow can break before it runs: a bad `layout`/`fit`/`background`, and
    a literal `images` list longer than a literal `max_images`."""
    errors = choice_errors("ingredients_grid", arguments)
    background = arguments.get("background")
    if isinstance(background, str) and not is_ref(DEFERRED, background):
        try:
            ingredients_background(background)
        except ValueError as error:
            errors.append(("background", str(error)))
    images = arguments.get("images")
    limit = arguments.get("max_images", INGREDIENTS_DEFAULT_MAX_IMAGES)
    if isinstance(images, list) and not any(is_ref(GATHER, item) for item in images):
        number = as_number(limit)
        if number is not None and number > 0 and len(images) > number:
            errors.append(
                (
                    "images",
                    f"ingredients_grid was given {len(images)} images but "
                    f"'max_images' is {limit} - raise it, or pass fewer images",
                )
            )
    return errors


# crop_face_track's rules past a plain domain. The crop feeds a video model
# whose frame size and count move on a grid the workflow declares (modulus,
# remainder and multiple; the defaults are LTX's 8n+1 and 32 pixels), the
# padding is a multiple of the face's own size added on each side, and the gate
# is a ramp from gate_full to gate_zero, so the two must be in that order.
# Owned here so the static pass and the command's run-time refusal read one rule
FACE_CROP_MODULUS = 8
FACE_CROP_REMAINDER = 1
FACE_CROP_MULTIPLE = 32
FACE_PADDING_MAX = 3.0
FACE_DETECTOR_SUFFIX = ".onnx"


def face_track_problems(
    crop_size=None,
    padding=None,
    gate_full=None,
    gate_zero=None,
    min_confidence=None,
    modulus=FACE_CROP_MODULUS,
    remainder=FACE_CROP_REMAINDER,
    multiple=FACE_CROP_MULTIPLE,
):
    """[(argument, message)] for each crop_face_track rule these values break.

    A value that is None or not a number is skipped - the domain check, or
    another pass, owns that complaint.
    """
    problems = []
    size = as_number(crop_size)
    step = as_number(multiple)
    if step is not None and (step <= 0 or step != int(step)):
        step = None
    if size is not None and size > 0 and step is not None:
        step = int(step)
        if size != int(size) or int(size) % step:
            problems.append(
                (
                    "crop_size",
                    f"crop_face_track needs 'crop_size' as a multiple of "
                    f"{step}, got {crop_size!r} - the crops feed a "
                    f"video model whose frame size moves in steps of "
                    f"{step}",
                )
            )
    mod, rem = as_number(modulus), as_number(remainder)
    if (
        mod is not None
        and rem is not None
        and mod == int(mod)
        and rem == int(rem)
        and 0 < mod <= rem
    ):
        problems.append(
            (
                "remainder",
                f"crop_face_track needs 'remainder' ({int(rem)}) below "
                f"'modulus' ({int(mod)})",
            )
        )
    pad = as_number(padding)
    if pad is not None and pad > FACE_PADDING_MAX:
        problems.append(
            (
                "padding",
                f"crop_face_track needs 'padding' from 0 to {FACE_PADDING_MAX} "
                f"(the face's size added on each side), got {padding!r}",
            )
        )
    full, zero = as_number(gate_full), as_number(gate_zero)
    for name, value, number in (
        ("gate_full", gate_full, full),
        ("gate_zero", gate_zero, zero),
    ):
        if number is not None and number > 1:
            problems.append(
                (
                    name,
                    f"crop_face_track needs '{name}' as a fraction of the frame "
                    f"width, at most 1.0, got {value!r}",
                )
            )
    if full is not None and zero is not None and zero <= full:
        problems.append(
            (
                "gate_zero",
                f"crop_face_track needs 'gate_zero' above 'gate_full' - "
                f"strength ramps from full at gate_full down to 0 at gate_zero "
                f"- got gate_full {gate_full!r} and gate_zero {gate_zero!r}",
            )
        )
    confidence = as_number(min_confidence)
    if confidence is not None and confidence >= 1:
        problems.append(
            (
                "min_confidence",
                f"crop_face_track needs 'min_confidence' below 1.0, got "
                f"{min_confidence!r}",
            )
        )
    return problems


def paste_feather_problems(feather=None):
    """[(argument, message)] when paste_face_track's feather is past 1 - the
    fraction of the mask's radius that fades, so more than all of it is none."""
    number = as_number(feather)
    if number is not None and number > 1:
        return [
            (
                "feather",
                f"paste_face_track needs 'feather' from 0 to 1 (the fraction "
                f"of the paste's radius that fades out), got {feather!r}",
            )
        ]
    return []


def paste_face_track_errors(arguments):
    """[(argument, message)] for the paste_face_track rules a literal
    workflow can break before it runs."""
    feather = arguments.get("feather")
    if is_ref(DEFERRED, feather):
        return []
    return paste_feather_problems(feather)


def check_face_detector_source(repo, filename):
    """Refuse a detector source that is not a Hub repo id and a bare .onnx
    file name in it - before anything is downloaded."""
    from .locations import validate_hub_repo_id, validate_weight_name

    validate_hub_repo_id(repo, what="detector_repo")
    validate_weight_name(
        filename, (FACE_DETECTOR_SUFFIX,), what="detector_file", subfolders=False
    )


def face_track_errors(arguments):
    """[(argument, message)] for the crop_face_track rules a literal workflow
    can break before it runs: the crop, padding, gate and frame-grid rules, and
    a detector source that is not a Hub repo and an .onnx file."""
    # A deferred value is passed through, not dropped: as_number reads it as
    # no number, so its rules are skipped instead of run against a default
    literal = {
        name: arguments[name]
        for name in (
            "crop_size",
            "padding",
            "gate_full",
            "gate_zero",
            "min_confidence",
            "modulus",
            "remainder",
            "multiple",
        )
        if name in arguments
    }
    errors = face_track_problems(**literal)
    from .locations import validate_hub_repo_id, validate_weight_name
    from .security import SecurityError

    for name, check in (
        ("detector_repo", lambda v: validate_hub_repo_id(v, what=name)),
        (
            "detector_file",
            lambda v: validate_weight_name(
                v, (FACE_DETECTOR_SUFFIX,), what=name, subfolders=False
            ),
        ),
    ):
        value = arguments.get(name)
        if name not in arguments or is_ref(DEFERRED, value):
            continue
        try:
            check(value)
        except (SecurityError, ValueError) as error:
            errors.append((name, str(error).rstrip(".")))
    return errors


def beats_problems(min_bpm=None, max_bpm=None, anchors=None):
    """[(argument, message)] for each analyze_beats rule these values break:
    a tempo range that is empty, and anchors that are not one kind of mark
    in time order.

    An anchor is a time in seconds, or {beat_index, seconds} placing the
    beat at that index; a list holds one kind or the other. A value that is
    a reference, or not a number, is skipped - the domain check or the run
    owns that complaint. An anchor past the song's end is the run's to find:
    only it has the song.
    """
    problems = []
    low, high = as_number(min_bpm), as_number(max_bpm)
    if low is not None and high is not None and low >= high:
        problems.append(
            (
                "min_bpm",
                f"analyze_beats needs 'min_bpm' ({low:g}) below 'max_bpm' "
                f"({high:g}) - the range is where the tempo is searched for",
            )
        )
    if anchors is None or is_ref(DEFERRED, anchors):
        return problems
    if not isinstance(anchors, (list, tuple)):
        return problems + [
            (
                "anchors",
                "analyze_beats' 'anchors' is a list of times in seconds or of "
                f"{{beat_index, seconds}} marks, not {type(anchors).__name__}",
            )
        ]
    kinds, marks = set(), []
    for index, anchor in enumerate(anchors):
        if is_ref(DEFERRED, anchor):
            marks.append(None)
            continue
        if isinstance(anchor, dict):
            kinds.add("dict")
            beat, seconds = anchor.get("beat_index"), anchor.get("seconds")
            if (
                isinstance(beat, bool)
                or not isinstance(beat, (int, float, str))
                or as_number(beat) is None
                or as_number(beat) != int(as_number(beat))
                or as_number(beat) < 0
            ):
                if not is_ref(DEFERRED, beat):
                    problems.append(
                        (
                            "anchors",
                            f"analyze_beats' anchors[{index}] needs a whole "
                            f"'beat_index' at or above zero, not {beat!r}",
                        )
                    )
                beat = None
            else:
                beat = as_number(beat)
        else:
            kinds.add("seconds")
            beat, seconds = None, anchor
        number = as_number(seconds)
        if number is None or number < 0:
            if not is_ref(DEFERRED, seconds):
                problems.append(
                    (
                        "anchors",
                        f"analyze_beats' anchors[{index}] needs a time in "
                        f"seconds at or above zero, not {seconds!r}",
                    )
                )
            number = None
        marks.append((index, beat, number))
    if len(kinds) > 1:
        problems.append(
            (
                "anchors",
                "analyze_beats' 'anchors' mixes bare times with "
                "{beat_index, seconds} marks - give one kind or the other",
            )
        )
        return problems
    known = [mark for mark in marks if mark is not None]
    for (_, beat_a, time_a), (index, beat_b, time_b) in zip(known, known[1:]):
        if time_a is not None and time_b is not None and time_b <= time_a:
            problems.append(
                (
                    "anchors",
                    f"analyze_beats' anchors[{index}] ({time_b:g} s) is not "
                    f"after the anchor before it ({time_a:g} s) - anchors are "
                    "given in time order, each at a different time",
                )
            )
        elif beat_a is not None and beat_b is not None and beat_b <= beat_a:
            problems.append(
                (
                    "anchors",
                    f"analyze_beats' anchors[{index}] places beat {beat_b:g} "
                    f"after beat {beat_a:g} - beat indexes rise with time",
                )
            )
    return problems


def beats_errors(arguments):
    """[(argument, message)] for the analyze_beats rules a literal workflow
    can break before it runs (`beats_problems`)."""
    literal = {
        name: arguments.get(name)
        for name in ("min_bpm", "max_bpm", "anchors")
        if not is_ref(DEFERRED, arguments.get(name))
    }
    return beats_problems(**literal)


def transcript_problem(transcript):
    """Why this transcript can't be planned from, or None. A bare string is
    text without timings - the commonest mistake, so it is named."""
    if isinstance(transcript, str):
        return (
            "plan_cuts needs a timestamped transcript - {text, chunks: "
            "[{start, end, text}]}, not plain text: run transcribe_audio with "
            "'timestamps': \"segment\" (Whisper's return_timestamps), its "
            "result's content_type application/json"
        )
    if isinstance(transcript, dict):
        chunks = transcript.get("chunks")
        if not isinstance(chunks, list):
            return (
                "plan_cuts's 'transcript' has no 'chunks' list - run "
                "transcribe_audio with 'timestamps' set (Whisper's "
                "return_timestamps) for the {text, chunks} shape"
            )
        return None
    if isinstance(transcript, list):
        return None
    return (
        "plan_cuts's 'transcript' is a {text, chunks} dict from "
        "transcribe_audio with 'timestamps' set (Whisper's return_timestamps), "
        f"not {type(transcript).__name__}"
    )


def cuts_problems(
    transcript=None,
    segment_by=None,
    min_scene_s=None,
    max_scene_s=None,
    beats=None,
    modulus=None,
    remainder=None,
    min_frames=None,
    max_frames=None,
):
    """[(argument, message)] for each plan_cuts rule these values break: a
    transcript that is plain text (no timings), a scene range that is empty,
    cutting by beat with no beats, and a render grid (modulus, remainder,
    min_frames, max_frames) that has a remainder with no modulus or past it, or leaves no length a shot could render at. A
    reference is skipped - only the run has its value."""
    problems = []
    if transcript is not None and not is_ref(DEFERRED, transcript):
        problem = transcript_problem(transcript)
        if problem:
            problems.append(("transcript", problem))
    low, high = as_number(min_scene_s), as_number(max_scene_s)
    if low is not None and high is not None and low > high:
        problems.append(
            (
                "min_scene_s",
                f"plan_cuts needs 'min_scene_s' ({low:g}) at or below "
                f"'max_scene_s' ({high:g}) - no shot could be both",
            )
        )
    if segment_by == "beat" and beats is None:
        problems.append(
            (
                "beats",
                "plan_cuts with 'segment_by': \"beat\" needs 'beats' - "
                "analyze_beats' result or a list of beat times",
            )
        )
    return problems + grid_problems(modulus, remainder, min_frames, max_frames)


def grid_problems(modulus, remainder, min_frames, max_frames):
    """[(argument, message)] for the plan_cuts render grid rules these values
    break. A value that is absent or not a whole number is skipped -
    `domain_violation` refuses the second."""
    problems = []
    whole = {}
    modulus_given = modulus is not None
    for name, value in (
        ("modulus", modulus),
        ("remainder", remainder),
        ("min_frames", min_frames),
        ("max_frames", max_frames),
    ):
        # A fractional value is the shared whole-number check's to report
        number = as_number(value)
        if number is not None and number == int(number):
            whole[name] = int(number)
    modulus, remainder = whole.get("modulus"), whole.get("remainder")
    low, high = whole.get("min_frames"), whole.get("max_frames")
    if remainder is not None and not modulus_given:
        problems.append(
            (
                "remainder",
                "plan_cuts needs 'modulus' with 'remainder' - a remainder is "
                "of a modulus",
            )
        )
    if modulus is not None and remainder is not None and remainder >= modulus > 0:
        problems.append(
            (
                "remainder",
                f"plan_cuts needs 'remainder' ({remainder}) below 'modulus' "
                f"({modulus})",
            )
        )
    if low is not None and high is not None and low > high:
        problems.append(
            (
                "min_frames",
                f"plan_cuts needs 'min_frames' ({low}) at or below "
                f"'max_frames' ({high}) - no shot could be both",
            )
        )
    elif (
        high is not None
        and not problems
        and (modulus is None or modulus > 0)
        and (remainder or 0) >= 0
        and (low or 1) > 0
    ):
        from .variable_constraints import aligned

        floor = max(low or 0, 1)
        smallest = aligned(floor, {"modulus": modulus, "remainder": remainder or 0})
        if smallest is None:
            smallest = floor
        if smallest > high:
            problems.append(
                (
                    "max_frames",
                    f"plan_cuts needs 'max_frames' ({high}) at or above the "
                    f"smallest render length on the grid ({smallest}) - no "
                    "shot could fit",
                )
            )
    return problems


def cuts_errors(arguments):
    """[(argument, message)] for the plan_cuts rules a literal workflow can
    break before it runs (`cuts_problems`). An absent 'beats' counts - a
    reference to one does not."""
    literal = {
        name: arguments.get(name)
        for name in ("transcript", "segment_by", "min_scene_s", "max_scene_s")
        if not is_ref(DEFERRED, arguments.get(name))
    }
    beats = arguments.get("beats")
    literal["beats"] = "deferred" if is_ref(DEFERRED, beats) else beats
    grid = ("modulus", "remainder", "min_frames", "max_frames")
    if not is_ref(DEFERRED, arguments.get("modulus")):
        # A grid with a value yet to come can't be judged; a deferred modulus
        # hides what a remainder or the frame bounds are measured against
        for name in grid:
            if not is_ref(DEFERRED, arguments.get(name)):
                literal[name] = arguments.get(name)
    return choice_errors("plan_cuts", arguments) + cuts_problems(**literal)


# apply_lut's palette: 2..16 colours, each #rrggbb (#603 stage D)
MIN_PALETTE_COLOURS = 2
MAX_PALETTE_COLOURS = 16
_HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{6}\Z")


def lut_source_problem(lut=None, palette=None):
    """The refusal when apply_lut is given both or neither of 'lut' and
    'palette', else None. A null value counts as not given."""
    if (lut is None) == (palette is None):
        given = "both" if lut is not None else "neither"
        return (
            f"apply_lut takes exactly one of 'lut' (a .cube file) or "
            f"'palette' (a list of #rrggbb colours) - {given} given"
        )
    return None


def palette_problem(palette):
    """(entry index or None, message) for the first thing wrong with an
    apply_lut palette, else None. The index points at the bad entry; None
    means the palette as a whole (not a list, or too few colours)."""
    if not isinstance(palette, (list, tuple)):
        return None, (
            f"apply_lut needs 'palette' as a list of {MIN_PALETTE_COLOURS} "
            f"to {MAX_PALETTE_COLOURS} #rrggbb colours, dark to light, not "
            f"{type(palette).__name__}"
        )
    for index, colour in enumerate(palette):
        if not isinstance(colour, str) or not _HEX_COLOUR.match(colour):
            return index, (
                f"apply_lut 'palette' entry {index} ({str(colour)[:40]!r}) "
                f"is not a #rrggbb colour"
            )
    if len(palette) < MIN_PALETTE_COLOURS:
        listed = f" ({palette[0]!r})" if palette else ""
        return None, (
            f"apply_lut 'palette' has {len(palette)} colour{listed}; it "
            f"takes {MIN_PALETTE_COLOURS} to {MAX_PALETTE_COLOURS}, dark to light"
        )
    if len(palette) > MAX_PALETTE_COLOURS:
        index = MAX_PALETTE_COLOURS
        return index, (
            f"apply_lut 'palette' has {len(palette)} colours; it takes at "
            f"most {MAX_PALETTE_COLOURS}, so entry {index} "
            f"({palette[index]!r}) onward is too many"
        )
    return None


def check_lut_source(lut=None, palette=None):
    """lut_source_problem and palette_problem at run time, as ValueError."""
    problem = lut_source_problem(lut, palette)
    if problem is not None:
        raise ValueError(problem)
    if palette is not None:
        found = palette_problem(palette)
        if found is not None:
            raise ValueError(found[1])


def lut_errors(arguments):
    """[(argument, message)] for the apply_lut source rules a literal
    workflow can break before it runs. A reference counts as given - which
    one was named is known now - but a palette's colours are judged only
    when they are literal."""
    lut = arguments.get("lut")
    palette = arguments.get("palette")
    problem = lut_source_problem(lut, palette)
    if problem is not None:
        return [("palette" if palette is not None else "lut", problem)]
    if palette is None or is_ref(DEFERRED, palette):
        return []
    found = palette_problem(palette)
    return [] if found is None else [("palette", found[1])]


def script_lines_errors(arguments):
    """[(argument, message)] for a literal check_script `lines` or `shots`
    the task's `parse_lines`/`parse_shots` would refuse, and for a literal
    line naming a shot a literal `shots` lacks. A reference is left to the
    run, which refuses one that resolves to no list, and so is a shot map the
    take carries: only the run can read it."""
    errors = []
    parsed_lines = parsed_shots = None
    if "lines" in arguments and not is_ref(DEFERRED, arguments["lines"]):
        try:
            parsed_lines = parse_lines(arguments["lines"])
        except ValueError as error:
            errors.append(("lines", str(error)))
    shots = arguments.get("shots")
    if shots is not None and not is_ref(DEFERRED, shots):
        try:
            parsed_shots = parse_shots(shots)
        except ValueError as error:
            errors.append(("shots", str(error)))
    if parsed_lines and parsed_shots:
        refusal = shot_names_error(parsed_lines, parsed_shots)
        if refusal:
            errors.append(("lines", refusal))
    return errors
