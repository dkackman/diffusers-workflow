"""What a task command's numeric arguments are allowed to be.

A task command's arguments are whatever its implementation's signature takes
(`describe_task` in dw/introspection.py reads them off it), and the signature
carries no domain: nothing in `slice_audio(num_frames=...)` says a frame count
cannot be negative. So `validate_workflow` had nothing to check against, and a
number outside its domain reached the command - where, if the command happened
to guard it, the run failed on that step, and if it did not, Python's own
semantics answered instead. `num_frames: -10` became the Python slice
`audio[0:-13333]` and returned the track minus its last ten frames, reported
`succeeded` with no warnings (#139); `target_sample_rate: 0` fell through to a
44100 Hz default and re-headered the samples unresampled, 38% off in duration
(#140).

The domains are declared here, once, and checked in two places for two
reasons: statically in `validation_errors`, so a literal out-of-domain number
is a free pre-flight error at the JSON path the author wrote it at rather than
a spent run; and at run time by the command itself, since a value that arrives
from a `variable:`, a `previous_result:` or a computation is not a literal and
this pass can never see it. The static pass therefore only ever refuses what
the command would refuse anyway - it is the earlier of two answers, not a
second opinion.

Only numbers whose domain is not a judgement call are listed. Whether a level
in dBFS or a gain is the *right* one for a mix is the command's business, and
a command that wants to refuse something subtler does it in its own body -
but a value's documented range is not a judgement call either: grade's
temperature and tint are only defined from -1.0 to 1.0, and a value outside
it is not a bolder version of the effect, just an unmodelled one (#349).
"""

import logging
import numbers

from .references import (
    DEFERRED,
    GATHER,
    MEMBER_SEPARATOR,
    author_index,
    is_ref,
    render_path,
)

logger = logging.getLogger("dw")

# Above zero, and zero-or-above. A count of frames to cut is the first kind -
# a zero-length slice is not a slice - and an offset to start at is the
# second, since the head of a track is a legitimate place to begin
POSITIVE = "positive"
NON_NEGATIVE = "non_negative"
# A level that cannot exceed full scale - peak_dbfs's own long-standing rule
# (0 is full scale, positive is not a level any of these commands can reach),
# shared here with normalize_audio's target_lufs
NON_POSITIVE = "non_positive"
# A value documented as a scale from -1.0 to 1.0 - grade's temperature and
# tint, whose linear interpolation is only defined inside that range; outside
# it the same formula still runs and produces a value, just not the one the
# documented scale promised
CLOSED_UNIT = "closed_unit"

_DOMAIN_TEXT = {
    POSITIVE: "above zero",
    NON_NEGATIVE: "zero or above",
    NON_POSITIVE: "at or below full scale (0)",
    CLOSED_UNIT: "between -1.0 and 1.0",
}

_DOMAIN_REASON = {
    CLOSED_UNIT: (
        "A value outside that range is refused rather than extrapolated - "
        "the documented scale is only defined inside it"
    ),
}
# Shared by POSITIVE, NON_NEGATIVE and NON_POSITIVE, which span both counts/
# rates (audio, frame) and multipliers (grade's contrast, saturation) - kept
# neutral rather than naming either, since a wording specific to one reads as
# nonsense on the other (#383)
_DEFAULT_REASON = (
    "A value outside that range is refused rather than interpreted - it "
    "would otherwise produce a plausible-looking result outside the "
    "documented range"
)

# command -> argument -> domain. Every entry here is pinned to a real command
# and a real parameter of it by tests/test_task_domains.py, so a renamed
# argument cannot leave a domain checking nothing
TASK_ARGUMENT_DOMAINS = {
    "slice_audio": {
        "start_seconds": NON_NEGATIVE,
        "duration_seconds": POSITIVE,
        "start_frame": NON_NEGATIVE,
        "num_frames": POSITIVE,
        "fps": POSITIVE,
        "sample_rate": POSITIVE,
    },
    "gain_audio": {
        "start_seconds": NON_NEGATIVE,
        "duration_seconds": POSITIVE,
        "start_frame": NON_NEGATIVE,
        "num_frames": POSITIVE,
        "fps": POSITIVE,
        "sample_rate": POSITIVE,
    },
    "resample_audio": {
        "target_sample_rate": POSITIVE,
        "sample_rate": POSITIVE,
    },
    "loop_audio": {
        "duration_seconds": POSITIVE,
        "target_frames": POSITIVE,
        "fps": POSITIVE,
        "crossfade_ms": NON_NEGATIVE,
        "sample_rate": POSITIVE,
    },
    "find_loop_bed": {
        "start_seconds": NON_NEGATIVE,
        "end_seconds": POSITIVE,
        "min_seconds": POSITIVE,
        "max_seconds": POSITIVE,
        "max_bin_dbfs": NON_POSITIVE,
        "max_mean_dbfs": NON_POSITIVE,
        "max_spike_db": NON_NEGATIVE,
        "crossfade_ms": NON_NEGATIVE,
        "loop_seconds": POSITIVE,
        "target_bed_dbfs": NON_POSITIVE,
        "max_candidates": POSITIVE,
        "fps": POSITIVE,
    },
    "fade_audio": {
        "fade_in_ms": NON_NEGATIVE,
        "fade_out_ms": NON_NEGATIVE,
        "sample_rate": POSITIVE,
    },
    "normalize_audio": {
        "sample_rate": POSITIVE,
        "target_lufs": NON_POSITIVE,
    },
    "crossfade_audio": {
        "crossfade_ms": NON_NEGATIVE,
        "sample_rate": POSITIVE,
    },
    "mix_audio": {"gains": NON_NEGATIVE, "sample_rate": POSITIVE},
    "pair_audio": {"sample_rate": POSITIVE},
    "concat_videos": {
        "trim_frames": NON_NEGATIVE,
        "crossfade_ms": NON_NEGATIVE,
        "audio_bleed_ms": NON_NEGATIVE,
        "seam_fade_ms": NON_NEGATIVE,
        "fps": POSITIVE,
        "sample_rate": POSITIVE,
    },
    "join_into_song": {
        "cue_seconds": NON_NEGATIVE,
        "duck_delay_ms": NON_NEGATIVE,
        "duck_db": NON_POSITIVE,
        "duck_ramp_ms": NON_NEGATIVE,
        "fps": POSITIVE,
    },
    "loop_frames": {"num_frames": POSITIVE},
    "upscale_h3_latents": {"width": POSITIVE, "height": POSITIVE},
    "frame_grid": {"count": POSITIVE, "columns": POSITIVE, "tile_width": POSITIVE},
    "ingredients_grid": {
        "width": POSITIVE,
        "height": POSITIVE,
        "gap": NON_NEGATIVE,
        "max_images": POSITIVE,
    },
    "dissolve_videos": {
        "dissolve_frames": NON_NEGATIVE,
        "fade_in_frames": NON_NEGATIVE,
        "fade_out_frames": NON_NEGATIVE,
        "fps": POSITIVE,
    },
    "compress_audio": {
        "ratio": POSITIVE,
        "attack_ms": NON_NEGATIVE,
        "release_ms": NON_NEGATIVE,
        "sample_rate": POSITIVE,
    },
    "filter_audio": {
        "cutoff_hz": POSITIVE,
        "sample_rate": POSITIVE,
    },
    "analyze_audio": {"sample_rate": POSITIVE},
    "attribute_voices": {
        "window_seconds": POSITIVE,
        "min_reference_seconds": POSITIVE,
    },
    "grade": {
        "contrast": NON_NEGATIVE,
        "saturation": NON_NEGATIVE,
        "temperature": CLOSED_UNIT,
        "tint": CLOSED_UNIT,
    },
    "crop_face_track": {
        "crop_size": POSITIVE,
        "padding": NON_NEGATIVE,
        "gate_full": POSITIVE,
        "gate_zero": POSITIVE,
        "min_confidence": POSITIVE,
    },
    "paste_face_track": {
        "feather": NON_NEGATIVE,
    },
    "analyze_beats": {
        "sample_rate": POSITIVE,
        "tempo_bpm": POSITIVE,
        "min_bpm": POSITIVE,
        "max_bpm": POSITIVE,
    },
    "plan_cuts": {
        "fps": POSITIVE,
        "duration_s": POSITIVE,
        "min_scene_s": NON_NEGATIVE,
        "max_scene_s": POSITIVE,
        "vocal_tail_s": NON_NEGATIVE,
        "min_gap_seconds": NON_NEGATIVE,
    },
}


# command -> argument -> the literal values it accepts. Owned here so the
# command's run-time refusal and the static pass read one list
INGREDIENTS_LAYOUTS = ("auto", "rows", "panels")
INGREDIENTS_FITS = ("contain", "cover")
TASK_ARGUMENT_CHOICES = {
    "ingredients_grid": {
        "layout": INGREDIENTS_LAYOUTS,
        "fit": INGREDIENTS_FITS,
    },
    "plan_cuts": {
        "segment_by": ("line", "stanza", "beat"),
    },
}
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


def choice_errors(command, arguments):
    """[(argument, message)] for each literal argument outside its accepted
    values, worded as the command's own refusal."""
    errors = []
    for name, choices in TASK_ARGUMENT_CHOICES.get(command, {}).items():
        value = arguments.get(name)
        if name not in arguments or is_ref(DEFERRED, value):
            continue
        if value not in choices:
            errors.append(
                (
                    name,
                    f"{command} needs '{name}' as one of {list(choices)}, got {value!r}",
                )
            )
    return errors


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
# whose latent grid is 32 pixels, the padding is a multiple of the face's own
# size added on each side, and the gate is a ramp from gate_full to gate_zero,
# so the two must be in that order. Owned here so the static pass and the
# command's run-time refusal read one rule
FACE_CROP_MULTIPLE = 32
FACE_PADDING_MAX = 3.0
FACE_DETECTOR_SUFFIX = ".onnx"


def face_track_problems(
    crop_size=None, padding=None, gate_full=None, gate_zero=None, min_confidence=None
):
    """[(argument, message)] for each crop_face_track rule these values break.

    A value that is None or not a number is skipped - the domain check, or
    another pass, owns that complaint.
    """
    problems = []
    size = as_number(crop_size)
    if size is not None and size > 0:
        if size != int(size) or int(size) % FACE_CROP_MULTIPLE:
            problems.append(
                (
                    "crop_size",
                    f"crop_face_track needs 'crop_size' as a multiple of "
                    f"{FACE_CROP_MULTIPLE}, got {crop_size!r} - the crops feed a "
                    f"video model whose frame size moves in steps of "
                    f"{FACE_CROP_MULTIPLE}",
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
    can break before it runs: the crop, padding and gate rules, and a detector
    source that is not a Hub repo and an .onnx file."""
    literal = {
        name: arguments.get(name)
        for name in ("crop_size", "padding", "gate_full", "gate_zero", "min_confidence")
        if not is_ref(DEFERRED, arguments.get(name))
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


def cuts_problems(
    transcript=None, segment_by=None, min_scene_s=None, max_scene_s=None, beats=None
):
    """[(argument, message)] for each plan_cuts rule these values break: a
    transcript that is plain text (no timings), a scene range that is empty,
    and cutting by beat with no beats. A reference is skipped - only the run
    has its value."""
    from .tasks.cuts import transcript_problem

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
    return cuts_problems(**literal)


def in_domain(value, domain):
    """Whether a number satisfies a domain. Anything unmeasurable is True -
    a value this cannot read is not this check's to refuse."""
    number = as_number(value)
    if number is None:
        return True
    if domain == POSITIVE:
        return number > 0
    if domain == NON_POSITIVE:
        return number <= 0
    if domain == CLOSED_UNIT:
        return -1.0 <= number <= 1.0
    return number >= 0


def as_number(value):
    """A value as a float if it is one, else None.

    A workflow variable declared null carries no type, so a number supplied
    for it on the command line arrives as a string - the tasks coerce those
    (`coerce_number` in audio_utils), so this reads them too. Booleans are not
    numbers here whatever Python thinks, and a `variable:`/`item:`/
    `previous_result:` string is somebody else's complaint.
    """
    if isinstance(value, bool) or value is None:
        return None
    # numbers.Real rather than (int, float): an fps is coerced to a Fraction
    # before it is checked, so an exact 24000/1001 stays exact
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _domain_candidates(value):
    """(index, item) pairs to check - one per element of a list argument
    (mix_audio's 'gains', one multiplier per track), else the value itself
    paired with no index."""
    if isinstance(value, list):
        return list(enumerate(value))
    return [(None, value)]


def domain_violation(command, name, value, domain):
    """(index, message) for the first out-of-domain element, or None if all
    are fine. index is None when the argument itself is the scalar checked
    rather than one entry of a list.

    Shared by the static pass and the commands' own run-time guards so the
    two cannot word the same refusal differently.
    """
    for index, item in _domain_candidates(value):
        if in_domain(item, domain):
            continue
        label = f"{name}[{index}]" if index is not None else name
        reason = _DOMAIN_REASON.get(domain, _DEFAULT_REASON)
        message = (
            f"{command} needs '{label}' {_DOMAIN_TEXT[domain]}, got {item!r}. {reason}"
        )
        return index, message
    return None


def domain_error(command, name, value, domain):
    """The message for one out-of-domain argument, or None if it is fine."""
    violation = domain_violation(command, name, value, domain)
    return None if violation is None else violation[1]


def check_argument(command, name, value):
    """Raise ValueError if a task argument is outside its declared domain.

    What a command calls on the value it was actually handed, after its own
    coercion - the value may have come from a variable or an earlier step,
    which the static pass never sees.
    """
    domain = TASK_ARGUMENT_DOMAINS.get(command, {}).get(name)
    if domain is None:
        return
    message = domain_error(command, name, value, domain)
    if message is not None:
        raise ValueError(message)


def check_arguments(command, **values):
    """check_argument over several of a command's arguments, in order."""
    for name, value in values.items():
        check_argument(command, name, value)


def task_argument_errors(workflow_definition, source_indices=None):
    """Every task argument outside its declared domain, as [{path, message}].

    The definition handed here has already been substituted and expanded, so
    a `for_each` member's own arguments are checked as they will run;
    `source_indices` maps each expanded step back to the step the author
    wrote, and the member is named in the message - the convention
    subfolder_errors and reference_limit_errors both follow.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        task = step.get("task")
        if not isinstance(task, dict):
            continue
        command = task.get("command")
        arguments = task.get("arguments")
        if not isinstance(command, str) or not isinstance(arguments, dict):
            continue
        domains = TASK_ARGUMENT_DOMAINS.get(command)
        if not domains:
            continue
        source = author_index(source_indices, index)
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )
        for key, domain in domains.items():
            if key not in arguments:
                continue
            violation = domain_violation(command, key, arguments[key], domain)
            if violation is None:
                continue
            element_index, message = violation
            path = ("steps", source, "task", "arguments", key)
            if element_index is not None:
                path = path + (element_index,)
            errors.append(
                {
                    "path": render_path(path),
                    "message": f"{message}{where}.",
                }
            )
        extra = {
            "ingredients_grid": ingredients_grid_errors,
            "crop_face_track": face_track_errors,
            "paste_face_track": paste_face_track_errors,
            "analyze_beats": beats_errors,
            "plan_cuts": cuts_errors,
        }.get(command)
        if extra is not None:
            for key, message in extra(arguments):
                path = ("steps", source, "task", "arguments", key)
                errors.append(
                    {"path": render_path(path), "message": f"{message}{where}."}
                )
    return errors


# The four rules a checker and a task both apply. Each was written twice -
# once where validate refuses it for free and once where the task refuses it
# at run time - and two of the copies drifted (the run's frame-size refusal
# stopped at the first mismatch and called its reference "video 0"; the
# checker's slice arithmetic lacked #557's end rounding). Each now has one
# home here, and both sides call it: the checker for the inputs it can know
# before the run, the task for the ones it is actually handed.


def dissolve_shortfalls(frame_counts, dissolve_frames):
    """One sentence per video too short for its share of the overlaps.

    `frame_counts` is one entry per video in join order, None where the count
    is not known (validation cannot probe a `previous_result:`) - such an
    entry is skipped but still counts as a neighbour, since a seam is a seam
    whether or not its other side has been measured. An inner video carries
    two dissolves, an end one carries one.
    """
    last = len(frame_counts) - 1
    shortfalls = []
    for index, frame_count in enumerate(frame_counts):
        if frame_count is None:
            continue
        seams = (index > 0) + (index < last)
        if frame_count < seams * dissolve_frames:
            shortfalls.append(
                f"video {index} has {frame_count} frames, too few for its "
                f"{seams} dissolve(s) of {dissolve_frames} frames"
            )
    return shortfalls


def frame_size_error(command, sizes):
    """The refusal sentence for videos of different frame sizes, or None when
    they all agree: `"{command} needs every video at one size: ..."`, naming
    every video whose size disagrees with the first known one.

    The one producer - `check_same_frame_size` raises it at run time and
    `video_size_errors` leads its validation message with it.

    `sizes` maps a video's index in the join to its (width, height); a video
    whose size is not known is absent. The reference is named by its real
    index, and every mismatch is listed, because each one is a fix the caller
    has to make - a report stopping at the first sends them back for the
    next.
    """
    mismatches = _frame_size_mismatches(sizes)
    if mismatches is None:
        return None
    return f"{command} needs every video at one size: {mismatches}"


def _frame_size_mismatches(sizes):
    """The mismatches half of `frame_size_error`'s sentence, or None."""
    if not sizes:
        return None
    first_index = next(iter(sizes))
    first_size = sizes[first_index]
    parts = [f"video {first_index} is {first_size[0]}x{first_size[1]}"]
    for index, size in sizes.items():
        if index != first_index and size != first_size:
            parts.append(f"video {index} is {size[0]}x{size[1]}")
    return ", ".join(parts) if len(parts) > 1 else None


# Padding shorter than this at the end of a slice is the rounding that
# frame-aligned slicing produces, not a slice that overran its source
SLICE_PAD_WARN_MS = 10.0


def frames_to_samples(frames, fps, sample_rate):
    """The number of audio samples spanning a run of video frames."""
    return int(round(frames / fps * sample_rate))


def slice_region(
    sample_rate,
    start_seconds=None,
    duration_seconds=None,
    start_frame=None,
    num_frames=None,
    fps=None,
    total=None,
):
    """The region a `slice_audio` call asks for, as `(start, length)` in
    samples, or None when it cannot be worked out.

    Seconds win over frames, as in the run. A frame-addressed end is rounded
    once, not as two rounded halves (#557), so a slice meant to reach the
    source's exact end does. A slice with no duration (or no `num_frames`)
    runs to the source's end, which takes `total`, the source's length in
    samples: validation does not know it and gets None, the run does. None
    is also an unusable shape - frames with no `fps`, or nothing addressed.
    Arguments are already numbers (the caller coerces them).
    """
    if start_seconds is not None or duration_seconds is not None:
        start = int(round((start_seconds or 0) * sample_rate))
        if duration_seconds is not None:
            return start, int(round(duration_seconds * sample_rate))
    elif start_frame is not None or num_frames is not None:
        if not fps:
            return None
        start = frames_to_samples(start_frame or 0, fps, sample_rate)
        if num_frames is not None:
            end = frames_to_samples((start_frame or 0) + num_frames, fps, sample_rate)
            return start, end - start
    else:
        return None
    if total is None:
        return None
    return start, max(total - start, 0)


def slice_padding(total_samples, start, length, sample_rate):
    """The seconds of silence a slice pads past its source's end, or None
    when there is none worth saying - under `SLICE_PAD_WARN_MS`, which is
    frame-aligned rounding rather than a slice that overran.

    `start` and `length` are in samples, computed the way `slice_audio`
    computes them (a frame-addressed end rounded directly, #557), so the
    checker and the run agree on the same figure for the same arguments.
    """
    if not sample_rate:
        return None
    available = max(0, min(total_samples - start, length))
    padded = length - available
    if padded <= 0:
        return None
    padded_seconds = padded / float(sample_rate)
    if padded_seconds * 1000.0 < SLICE_PAD_WARN_MS:
        return None
    return padded_seconds


SELECT_RULES = frozenset({"argmax", "argmin", "first_above", "first_below", "index"})
SELECT_THRESHOLD_RULES = frozenset({"first_above", "first_below"})


def select_rule_problems(rule, threshold, index):
    """Why `select` cannot run this rule with these arguments, one sentence
    each: an unknown rule, a threshold rule with no threshold, or the index
    rule with no index. Empty when the rule has what it needs.

    Only what the run itself refuses - an argument a rule does not use is
    validation's own complaint, since the run ignores it.
    """
    if rule not in SELECT_RULES:
        return [f"select: unknown rule: {rule!r}"]
    if rule in SELECT_THRESHOLD_RULES and threshold is None:
        return [f"select rule '{rule}' requires a threshold"]
    if rule == "index" and index is None:
        return ["select rule 'index' requires an index"]
    return []
