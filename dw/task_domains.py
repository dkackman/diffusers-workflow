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

The cross-argument rules of the families that need more than a domain -
face tracks, beats, cuts, LUTs, script lines, ingredients grids - live in
dw/task_problems.py, which reads the helpers here; this module never imports
it, so the dependency runs one way (#790). The media-timing rules below
(window, slice, fit, dissolve, frame size, select) stay here.
"""

import logging
import math
import numbers
from fractions import Fraction

from .references import (
    DEFERRED,
    is_ref,
    iter_steps,
    render_path,
)
from .tasks.registry import RegistryTable

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
# A strength or mix documented from 0.0 to 1.0 - grade's fade, where 0 is no
# effect and 1 the strongest the documented scale defines (#603)
UNIT = "unit"
# A size of at least one whole unit - film_grain's size, where a grain
# smaller than a pixel is not drawable (#603)
AT_LEAST_ONE = "at_least_one"
# An 8-bit channel level - sharpen's threshold, a difference between 0 and
# full scale in 0..255 levels (#603)
CHANNEL_LEVEL = "channel_level"
# A random generator's seed - film_grain's seed, which numpy only takes as a
# whole number at or above zero, so anything else would fail the run after
# it was queued (#634)
SEED = "seed"
# Any finite number - grade's exposure, a gain in dB. No range is the
# command's documented rule, but the value is still a number, so it is read
# and refused by the same rule as every other numeric argument (#774)
FINITE = "finite"

_DOMAIN_TEXT = {
    POSITIVE: "above zero",
    NON_NEGATIVE: "zero or above",
    NON_POSITIVE: "at or below full scale (0)",
    CLOSED_UNIT: "between -1.0 and 1.0",
    UNIT: "between 0.0 and 1.0",
    AT_LEAST_ONE: "1 or above",
    CHANNEL_LEVEL: "between 0 and 255",
    SEED: "a whole number, 0 or above",
    FINITE: "a finite number",
}

_SCALE_REASON = (
    "A value outside that range is refused rather than extrapolated - "
    "the documented scale is only defined inside it"
)
_DOMAIN_REASON = {CLOSED_UNIT: _SCALE_REASON, UNIT: _SCALE_REASON}
# Shared by POSITIVE, NON_NEGATIVE and NON_POSITIVE, which span both counts/
# rates (audio, frame) and multipliers (grade's contrast, saturation) - kept
# neutral rather than naming either, since a wording specific to one reads as
# nonsense on the other (#383)
_DEFAULT_REASON = (
    "A value outside that range is refused rather than interpreted - it "
    "would otherwise produce a plausible-looking result outside the "
    "documented range"
)

# command -> argument -> domain, declared on each command's registration
# (`register_command(domains=...)`, dw/tasks/registry.py) and read here as a
# view. Every entry is pinned to a real command and a real parameter of it by
# tests/test_task_domains.py, so a renamed argument cannot leave a domain
# checking nothing
TASK_ARGUMENT_DOMAINS = RegistryTable("domains")


# The literal values an argument accepts, for the commands whose run-time
# refusal and static pass read one list (ingredients_grid's live beside its
# rules, in dw/task_problems.py). FIT_MODES is how fit_to_model puts a source
# into the model's frame (#602)
FIT_MODES = ("letterbox", "stretch", "crop")
JOIN_WINDOWS_CURVES = ("cosine", "smoothstep", "linear")
# Declared on each registration (`register_command(choices=...)`)
TASK_ARGUMENT_CHOICES = RegistryTable("choices")
# command -> its cross-argument check, `arguments -> [(argument, message)]`
# (`register_command(static_check=...)`) - the rules below and the
# per-family ones in dw/task_problems.py, which task_argument_errors runs on
# each step of the command
TASK_STATIC_CHECKS = RegistryTable("static_check")
# command -> the arguments it reads as whole numbers
# (`register_command(whole_numbers=...)`); every other argument with a domain
# is a real number. `whole_number` and `real_number` below read each kind at
# run time, and the static pass refuses what they would
TASK_WHOLE_NUMBER_ARGUMENTS = RegistryTable("whole_numbers")


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
    if domain == UNIT:
        return 0.0 <= number <= 1.0
    if domain == AT_LEAST_ONE:
        return number >= 1.0
    if domain == CHANNEL_LEVEL:
        return 0.0 <= number <= 255.0
    if domain == SEED:
        return number >= 0 and number.is_integer()
    if domain == FINITE:
        return True
    return number >= 0


def as_number(value):
    """A value as a float if it is a finite number, else None.

    A workflow variable declared null carries no type, so a number supplied
    for it on the command line arrives as a string - the tasks coerce those
    (`whole_number` and `real_number` below), so this reads them too.
    Booleans are not numbers here whatever Python thinks, inf and nan are not
    measurable, and a `variable:`/`item:`/`previous_result:` string is
    somebody else's complaint.
    """
    if isinstance(value, bool) or value is None:
        return None
    # numbers.Real rather than (int, float): an fps is coerced to a Fraction
    # before it is checked, so an exact 24000/1001 stays exact
    if isinstance(value, numbers.Real):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


# The one numeric coercion in dw/tasks/ (#692). Before it there were five
# idioms that disagreed: plan_cuts took "3.0" for a frame count where
# window_video, trim_video and fit_to_model refused 3.0, slice_audio's let
# True through as 1, and analyze_beats truncated a sample_rate of "3.5" to 3.
# `number_problem` is the rule; the two readers raise it at run time, and the
# static pass (`domain_violation`) reports it, so the two cannot disagree.


def number_problem(value, name, command, whole=False):
    """The refusal sentence for a value that is not a number of its kind, or
    None - None for a value that is one, for None itself (an argument left
    out), and for a reference string, which is resolved before the run.

    A number is an int, a float or a numeric string, finite, and not a bool.
    A whole number is one with an integral value: 3, 3.0, "3" and "3.0", but
    not 3.5, "3.5" or True.
    """
    if value is None or (isinstance(value, str) and is_ref(DEFERRED, value)):
        return None
    kind = "a whole number" if whole else "a number"
    refusal = f"{command} needs {kind} for '{name}', got {value!r}"
    if isinstance(value, bool):
        return refusal
    if isinstance(value, numbers.Real):
        number = value
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return refusal
    else:
        return refusal
    if not math.isfinite(number):
        return (
            f"{command} needs a finite whole number for '{name}', got {value!r}"
            if whole
            else f"{command} needs a finite number for '{name}', got {value!r}"
        )
    if whole and number != int(number):
        return refusal
    return None


def _required(value, name, command, kind):
    if value is None:
        raise ValueError(f"{command} needs {kind} for '{name}', got None")


def whole_number(value, name, command, required=False):
    """A whole-number task argument as an int; None stays None unless
    `required`. Raises ValueError, naming the argument, for anything
    `number_problem` refuses."""
    if required:
        _required(value, name, command, "a whole number")
    if value is None:
        return None
    problem = number_problem(value, name, command, whole=True)
    if problem is not None:
        raise ValueError(problem)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return int(float(value))
    return int(value)


def real_number(value, name, command, required=False, exact=False):
    """A real-number task argument; None stays None unless `required`.

    A number is handed back as it came - an int stays an int - and a numeric
    string as a float, or with `exact` as a Fraction, so a frame rate written
    "23.976" keeps its decimal value. Raises ValueError, naming the argument,
    for anything `number_problem` refuses."""
    if required:
        _required(value, name, command, "a number")
    if value is None:
        return None
    problem = number_problem(value, name, command)
    if problem is not None:
        raise ValueError(problem)
    if isinstance(value, str):
        return Fraction(value) if exact else float(value)
    return value


def _is_literal_text(value):
    """A string that is no number and no reference - "abc" - which no
    command can coerce, so refusing it names the real problem instead of
    letting in_domain pass it as unmeasurable."""
    return (
        isinstance(value, str)
        and not is_ref(DEFERRED, value)
        and as_number(value) is None
    )


def _domain_candidates(value):
    """(index, item) pairs to check - one per element of a list argument
    (mix_audio's 'gains', one multiplier per track), else the value itself
    paired with no index."""
    if isinstance(value, list):
        return list(enumerate(value))
    return [(None, value)]


def domain_text(domain):
    """A domain in words - "between 0.0 and 1.0" - as its refusal says it."""
    return _DOMAIN_TEXT[domain]


def domain_violation(command, name, value, domain, whole=False):
    """(index, message) for the first element that is not a number of its
    kind (`number_problem`, a whole one when `whole`) or is out of domain, or
    None if all are fine. index is None when the argument itself is the
    scalar checked rather than one entry of a list.

    Shared by the static pass and the commands' own run-time guards so the
    two cannot word the same refusal differently.
    """
    for index, item in _domain_candidates(value):
        label = f"{name}[{index}]" if index is not None else name
        # FINITE has no range to name, so number_problem's sentence says it
        if (
            domain != FINITE
            and _is_literal_text(item)
            and not _is_non_finite_text(item)
        ):
            return index, (
                f"{command} needs '{label}' to be a number "
                f"({_DOMAIN_TEXT[domain]}), got {item!r}."
            )
        problem = number_problem(item, label, command, whole)
        if problem is not None:
            return index, problem
        if in_domain(item, domain):
            continue
        reason = _DOMAIN_REASON.get(domain, _DEFAULT_REASON)
        message = (
            f"{command} needs '{label}' {_DOMAIN_TEXT[domain]}, got {item!r}. {reason}"
        )
        return index, message
    return None


def _is_non_finite_text(value):
    """ "inf" or "nan" - a number to float(), and refused as not a finite one
    rather than as not a number at all."""
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def domain_error(command, name, value, domain, whole=False):
    """The message for one out-of-domain argument, or None if it is fine."""
    violation = domain_violation(command, name, value, domain, whole)
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
    whole = name in TASK_WHOLE_NUMBER_ARGUMENTS.get(command, ())
    message = domain_error(command, name, value, domain, whole)
    if message is not None:
        raise ValueError(message)


def _read_number(value, name, command, whole):
    """One argument (or one list entry) read as its kind: what Task.run
    hands the command instead of a numeric string."""
    if isinstance(value, str) and is_ref(DEFERRED, value):
        return value
    if whole:
        return whole_number(value, name, command)
    if not isinstance(value, str):
        return real_number(value, name, command)
    # A string reads as the number it spells: "24" is the int 24 (a
    # sample_rate some readers need as an int), and a frame rate stays
    # exact - "23.976" is 2997/125, not the float nearest it
    number = real_number(value, name, command, exact=True)
    if number.denominator == 1:
        return int(number)
    return number if name == "fps" else float(number)


def coerce_arguments(command, arguments):
    """The arguments with every one that has a declared domain read as its
    kind - `whole_number` for the command's whole-number arguments,
    `real_number` for the rest, a list one entry at a time - so a numeric
    string from an untyped variable reaches the command as a number.

    Run by Task.run before the command's handler, so no handler has to
    remember to coerce; it raises the static pass's own refusal for a value
    that is no number of its kind. A reference string is left for the
    command (it is resolved before the run). The range is not checked here:
    that stays with each command's `check_arguments`.
    """
    domains = TASK_ARGUMENT_DOMAINS.get(command, {})
    if not domains:
        return arguments
    wholes = TASK_WHOLE_NUMBER_ARGUMENTS.get(command, ())
    coerced = dict(arguments)
    for name, value in arguments.items():
        if name not in domains or value is None:
            continue
        whole = name in wholes
        if isinstance(value, list):
            coerced[name] = [
                _read_number(item, f"{name}[{index}]", command, whole)
                for index, item in enumerate(value)
            ]
        else:
            coerced[name] = _read_number(value, name, command, whole)
    return coerced


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
    errors = []
    for _, _, task, source, where in iter_steps(
        workflow_definition.get("steps"), source_indices, "task"
    ):
        command = task.get("command")
        arguments = task.get("arguments")
        if not isinstance(command, str) or not isinstance(arguments, dict):
            continue
        domains = TASK_ARGUMENT_DOMAINS.get(command, {})
        wholes = TASK_WHOLE_NUMBER_ARGUMENTS.get(command, ())
        static_check = TASK_STATIC_CHECKS.get(command)
        if not domains and static_check is None:
            continue
        for key, domain in domains.items():
            if key not in arguments:
                continue
            violation = domain_violation(
                command, key, arguments[key], domain, key in wholes
            )
            if violation is None:
                continue
            element_index, message = violation
            path = ("steps", source, "task", "arguments", key)
            if element_index is not None:
                path = path + (element_index,)
            errors.append(
                {
                    "path": render_path(path),
                    "message": f"{message.rstrip('.')}{where}.",
                }
            )
        if static_check is not None:
            for key, message in static_check(arguments):
                path = ("steps", source, "task", "arguments", key)
                errors.append(
                    {
                        "path": render_path(path),
                        "message": f"{message.rstrip('.')}{where}.",
                    }
                )
    return errors


# The four rules a checker and a task both apply. Each was written twice -
# once where validate refuses it for free and once where the task refuses it
# at run time - and two of the copies drifted (the run's frame-size refusal
# stopped at the first mismatch and called its reference "video 0"; the
# checker's slice arithmetic lacked #557's end rounding). Each now has one
# home here, and both sides call it: the checker for the inputs it can know
# before the run, the task for the ones it is actually handed.


def window_overlap_problem(num_frames, overlap, command="window_video"):
    """The refusal sentence for a window that is all overlap, or None.

    A window of `num_frames` frames advances by `num_frames - overlap`, so an
    overlap of `num_frames` or more never advances at all. Both numbers are
    whole numbers already inside their own domains. `command` is whichever of
    window_video and join_windows is refusing.
    """
    if overlap >= num_frames:
        return (
            f"{command} needs 'overlap' below 'num_frames' - got overlap "
            f"{overlap} with num_frames {num_frames}, which leaves a window "
            f"no frames of its own to advance by"
        )
    return None


def slice_lead_problem(start_frame, lead_frames, start_seconds, duration_seconds):
    """The refusal sentence for a `slice_audio` lead-in that cannot apply, or
    None: `lead_frames` given with the seconds form (it is extra audio before
    a frame-addressed cut), or one that reaches before the head of the track
    (`start_frame - lead_frames` below zero, `start_frame` defaulting to 0).
    Arguments are already numbers, None where not given.
    """
    if not lead_frames:
        return None
    if start_seconds is not None or duration_seconds is not None:
        return (
            "slice_audio takes 'lead_frames' only with the frame form "
            "('start_frame'/'num_frames'/'fps'), not with 'start_seconds' "
            "or 'duration_seconds'"
        )
    start = start_frame or 0
    if start - lead_frames < 0:
        return (
            f"slice_audio 'lead_frames' ({lead_frames}) reaches before the "
            f"head of the track: 'start_frame' ({start}) minus 'lead_frames' "
            f"would start at frame {start - lead_frames}"
        )
    return None


def slice_audio_errors(arguments):
    """[(argument, message)] for the slice_audio lead rule a literal workflow
    can break before it runs. A value that is not a literal number (a
    reference, a string) is unknown and says nothing."""

    def literal(name):
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return None
        return value

    lead_frames = literal("lead_frames")
    if lead_frames is None or lead_frames < 0 or lead_frames != int(lead_frames):
        return []
    for name in ("start_frame", "start_seconds", "duration_seconds"):
        if arguments.get(name) is not None and literal(name) is None:
            return []
    problem = slice_lead_problem(
        literal("start_frame"),
        lead_frames,
        literal("start_seconds"),
        literal("duration_seconds"),
    )
    return [] if problem is None else [("lead_frames", problem)]


def window_video_errors(arguments, command="window_video"):
    """[(argument, message)] for the window_video rule a literal workflow can
    break before it runs: an overlap that is not below the window length.
    Whether `index` falls inside the source needs the source's frame count,
    which is the run's to measure."""
    counts = []
    for name in ("num_frames", "overlap"):
        value = arguments.get(name)
        if number_problem(value, name, command, whole=True) or is_ref(DEFERRED, value):
            return []
        number = as_number(value)
        if number is None or number < 0:
            return []
        counts.append(int(number))
    problem = window_overlap_problem(*counts, command)
    return [] if problem is None else [("overlap", problem)]


def join_windows_errors(arguments):
    """[(argument, message)] for the join_windows rules a literal workflow
    can break before it runs: an unknown `curve`, and the overlap rule it
    shares with window_video. The window count needs the source's frame
    count, which the run measures."""
    return choice_errors("join_windows", arguments) + window_video_errors(
        arguments, "join_windows"
    )


def window_count(source_frames, num_frames, overlap):
    """How many windows of `num_frames` frames, each sharing `overlap` with
    the one before, cover a source of `source_frames` frames:
    `ceil(source_frames / (num_frames - overlap))`.

    The one home of the rule (#601): window_video's last valid index is one
    below it, join_windows refuses any other count at run time, and its
    static check refuses one at validate time.
    """
    stride = num_frames - overlap
    return -(-source_frames // stride)


def window_count_problem(given, source_frames, num_frames, overlap):
    """The refusal sentence for a join_windows list of `given` windows, or
    None when it is the count `window_count` requires. It names both numbers
    and which list entries to add or drop, by position from 0 - the same
    number as each entry's window_video `index`."""
    needed = window_count(source_frames, num_frames, overlap)
    if given == needed:
        return None
    low, high = sorted((given, needed))
    count = high - low
    entries = f"{count} entr{'y' if count == 1 else 'ies'}"
    span = str(low) if count == 1 else f"{low}..{high - 1}"
    fix = f"{'add' if given < needed else 'drop'} {entries} (index {span})"
    return (
        f"join_windows needs {needed} windows for a {source_frames}-frame "
        f"source with num_frames {num_frames} and overlap {overlap} (stride "
        f"{num_frames - overlap}), got {given} - {fix}"
    )


def fit_mode_problem(mode):
    """The refusal sentence for a fit_to_model `mode` it does not know, or
    None."""
    if isinstance(mode, str) and mode in FIT_MODES:
        return None
    return f"fit_to_model needs 'mode' as one of {list(FIT_MODES)}, got {mode!r}"


def fit_downscale_problem(width, height, downscale):
    """The refusal sentence for a size `downscale` does not divide, or None.

    fit_to_model fits into width/downscale x height/downscale, so a template
    can keep `width`/`height` as the output size of a 2x model and still fit
    the source to the model's input (#602). All three are whole numbers
    already inside their domains. It names only the side or sides that fail,
    so the caller knows which to change."""
    failing = [
        f"'{name}' {value}"
        for name, value in (("width", width), ("height", height))
        if value % downscale
    ]
    if failing:
        return (
            f"fit_to_model needs {' and '.join(failing)} to be divisible "
            f"by 'downscale' {downscale}"
        )
    return None


def fit_to_model_errors(arguments):
    """[(argument, message)] for the fit_to_model rules a literal workflow
    can break before it runs: an unknown `mode` and a size the downscale does
    not divide. A size, frame count or downscale that is not a whole number
    is the shared whole-number check's (`domain_violation`) to refuse."""
    errors = choice_errors("fit_to_model", arguments)
    sizes = [as_number(arguments.get(name)) for name in ("width", "height")]
    downscale = as_number(arguments.get("downscale", 1))
    if not errors and all(
        n is not None and n > 0 and n.is_integer() for n in [*sizes, downscale]
    ):
        problem = fit_downscale_problem(int(sizes[0]), int(sizes[1]), int(downscale))
        if problem is not None:
            errors.append(("downscale", problem))
    return errors


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
    lead_frames=0,
):
    """The region a `slice_audio` call asks for, as `(start, length)` in
    samples, or None when it cannot be worked out.

    Seconds win over frames, as in the run. A frame-addressed end is rounded
    once, not as two rounded halves (#557), so a slice meant to reach the
    source's exact end does. A slice with no duration (or no `num_frames`)
    runs to the source's end, which takes `total`, the source's length in
    samples: validation does not know it and gets None, the run does. None
    is also an unusable shape - frames with no `fps`, or nothing addressed.
    `lead_frames` is extra audio before the cut in the frame form: the slice
    starts at `start_frame - lead_frames` and still runs `num_frames`.
    Arguments are already numbers (the caller coerces them).
    """
    if start_seconds is not None or duration_seconds is not None:
        start = int(round((start_seconds or 0) * sample_rate))
        if duration_seconds is not None:
            return start, int(round(duration_seconds * sample_rate))
    elif start_frame is not None or num_frames is not None:
        if not fps:
            return None
        first = (start_frame or 0) - (lead_frames or 0)
        start = frames_to_samples(first, fps, sample_rate)
        if num_frames is not None:
            end = frames_to_samples(first + num_frames, fps, sample_rate)
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


def select_index_problem(index, count=None):
    """The refusal sentence for a select `index` that is not a whole number
    or is out of range, or None. A whole number may arrive as a float or a
    numeric string (2.0, "2"): a variable declared null carries a
    command-line value as a string. A deferred reference is the run's to
    resolve.

    `count` is how many candidates there are, when the caller knows: the run
    always does, and validation does when `candidates` is a list. With it,
    one sentence names both ends of the range, whichever end was missed;
    without it only a negative index can be refused."""
    if index is None or is_ref(DEFERRED, index):
        return None
    number = as_number(index)
    if number is None or not number.is_integer():
        return f"select: index must be a whole number, got {index!r}"
    if count is not None and not 0 <= number < count:
        return (
            f"select: index {index} is out of range for {count} candidates "
            f"(0 to {count - 1})"
        )
    if number < 0:
        return f"select: index {index} is out of range: it must be zero or above"
    return None


def select_rule_problems(rule, threshold, index, count=None):
    """Why `select` cannot run this rule with these arguments, one sentence
    each: an unknown rule, a threshold rule with no threshold, or the index
    rule with no index or one that is not a whole number in range
    (select_index_problem; `count` is the number of candidates, when known).
    Empty when the rule has what it needs.

    Only what the run itself refuses - an argument a rule does not use is
    validation's own complaint, since the run ignores it.
    """
    if rule not in SELECT_RULES:
        return [f"select: unknown rule: {rule!r}"]
    if rule in SELECT_THRESHOLD_RULES and threshold is None:
        return [f"select rule '{rule}' requires a threshold"]
    if rule == "index":
        if index is None:
            return ["select rule 'index' requires an index"]
        problem = select_index_problem(index, count)
        if problem is not None:
            return [problem]
    return []
