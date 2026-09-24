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

from .for_each import MEMBER_SEPARATOR, render_path

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
    "loop_frames": {"num_frames": POSITIVE},
    "frame_grid": {"count": POSITIVE, "columns": POSITIVE, "tile_width": POSITIVE},
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
    "grade": {
        "contrast": NON_NEGATIVE,
        "saturation": NON_NEGATIVE,
        "temperature": CLOSED_UNIT,
        "tint": CLOSED_UNIT,
    },
}


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
    (`_as_number` in audio_utils), so this reads them too. Booleans are not
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
    return errors
