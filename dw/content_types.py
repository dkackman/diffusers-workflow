"""A step's result 'content_type': the MIME type its writer is chosen by.

`content_type: "video"` validated clean and then died deep inside a writer,
in a traceback naming neither the field nor the value (#168) - the same
shape as #162 in a different field. The writer's own dispatch matches on
`content_type.startswith("video")`, so the bare word "video" took the video
branch anyway; having no real MIME type it also had no extension to write,
and the closest writer imageio could guess from an empty one was not a video
writer at all.

Audio and video each go through exactly one container this engine writes -
`AUDIO_FORMATS` in `dw/result.py`, and the single `video/mp4` mux - so
anything else in either family is refused here rather than accepted only to
mismatch its writer later. image/*, text/* and *.json values stay
permissive beyond the MIME-shape check: their writer dispatch is a generic
prefix/suffix match (PIL's own format inference, a literal text or JSON
write) with no narrower container to enforce a whitelist against.

The one text/* exception is active content. `/outputs` serves a written
file on the UI's own origin, without a token, so an .html or .xml output is
a page whose script reads the API token the UI keeps in localStorage (#407).
`text/html` and `text/xml` are the two active types the text writer can
produce, so they are refused outright; the server also serves every active
type it finds on disk under a `Content-Security-Policy: sandbox`, since a
planted file never passes through here.
"""

from .for_each import MEMBER_SEPARATOR, render_path
from .result import AUDIO_FORMATS, MUXED_VIDEO_CONTENT_TYPE
from .security import InvalidInputError, validate_content_type

CONTENT_TYPE_KEY = "content_type"

# Reference prefixes substitution resolves before this pass runs. One still
# spelled out here is one nothing resolved, and that is the undeclared-
# variable pass's complaint rather than a shape error
_UNRESOLVED_PREFIXES = ("variable:", "item:")

# Result types a browser would run as a document on the UI origin
REFUSED_ACTIVE_CONTENT_TYPES = frozenset({"text/html", "text/xml"})


def _active_content_fault(value):
    # compared without parameters or case: 'Text/HTML; charset=utf-8' is
    # the same document type
    if (
        isinstance(value, str)
        and value.split(";", 1)[0].strip().lower() in REFUSED_ACTIVE_CONTENT_TYPES
    ):
        return (
            f"Invalid content_type: {value!r} - active content is not written: "
            f"a browser would run it as a page on the server's origin. Use "
            f"'text/plain' or 'application/json'"
        )
    return None


def content_type_fault(value):
    """Why this result 'content_type' is invalid, or None.

    Raises nothing - callers that already have an InvalidInputError-raising
    check (validate_content_type) can call that directly; this is the
    string-message form `content_type_errors` collects.
    """
    try:
        validate_content_type(value)
    except InvalidInputError as e:
        return str(e)

    active = _active_content_fault(value)
    if active is not None:
        return active
    main_type = value.split("/", 1)[0]
    if main_type == "audio" and value not in AUDIO_FORMATS:
        return (
            f"Invalid content_type: {value!r} - audio can be written as "
            f"{', '.join(sorted(AUDIO_FORMATS))}"
        )
    if main_type == "video" and value != MUXED_VIDEO_CONTENT_TYPE:
        return (
            f"Invalid content_type: {value!r} - video is only written as "
            f"'{MUXED_VIDEO_CONTENT_TYPE}'"
        )
    return None


def content_type_errors(workflow_definition, source_indices=None):
    """Every result 'content_type' that no writer will accept, as
    [{path, message}].

    The definition handed here has already been substituted and expanded,
    so every value in it is literal; a 'variable:' or 'item:' still spelled
    out is left alone. `source_indices`, when given, is the source step
    index of each step - a 'for_each' group turns one written step into
    several, and the path an error carries has to be one the author can
    find in the file they wrote; the member is named in the message.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        result = step.get("result")
        if not isinstance(result, dict) or CONTENT_TYPE_KEY not in result:
            continue
        value = result[CONTENT_TYPE_KEY]
        if isinstance(value, str) and value.startswith(_UNRESOLVED_PREFIXES):
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
        fault = content_type_fault(value)
        if fault is not None:
            errors.append(
                {
                    "path": render_path(("steps", source, "result", CONTENT_TYPE_KEY)),
                    "message": f"{fault}{where}",
                }
            )
    return errors


def refuse_active_content_type(value):
    """Raise InvalidInputError for an active result type - the writer's
    run-time half of the refusal `content_type_errors` makes, for a
    definition that reached it without validation. Only this refusal: the
    writer's own dispatch still answers every other value as it did."""
    fault = _active_content_fault(value)
    if fault is not None:
        raise InvalidInputError(fault)
    return value


__all__ = [
    "REFUSED_ACTIVE_CONTENT_TYPES",
    "content_type_errors",
    "content_type_fault",
    "refuse_active_content_type",
]
