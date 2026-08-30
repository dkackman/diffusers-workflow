import os
import copy
import logging
from inspect import Parameter, signature
from .type_helpers import load_type_from_name, load_constant_from_name, has_method
from .prompts import PROMPT_PREFIX, fetch_prompt
from diffusers.utils import load_image, load_video
from .security import (
    validate_path,
    validate_constant_name,
    validate_url,
    validate_file_extension,
    SecurityError,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    ALLOWED_AUDIO_EXTENSIONS,
)

logger = logging.getLogger("dw")

# Keys that end in '_type' but name a category rather than a python type. Any other such
# key can be escaped where it is used, by wrapping its value in braces
NON_TYPE_KEYS = {"content_type", "offload_type"}


class EscapedString(str):
    """A string whose {} escape has already been consumed.

    Arguments are realized twice - once for the workflow's variables, and again for
    the steps the variables are substituted into. A variable's own name can look
    like a type reference ("weights_dtype": "{int4}"), so the first pass strips the
    braces and the second would load the bare name as a type. Marking the stripped
    value keeps the second pass from touching it, while it stays an ordinary string
    everywhere else.
    """


def is_escaped(value):
    """Whether a value is a type reference escaped with {} braces"""
    return isinstance(value, str) and value.startswith("{") and value.endswith("}")


# The key naming the file an argument object is constructed from
FROM_FILE_KEY = "from_file"

# The key naming the step whose output an argument object is constructed from
FROM_PREVIOUS_RESULT_KEY = "from_previous_result"

# The key holding the arguments an argument object is constructed from, for a type
# that takes its contents as plain fields rather than opening media itself
FROM_ARGUMENTS_KEY = "from_arguments"

# The prefix marking a value as a reference to an earlier step's output. Those are
# substituted once that step has run, so an object whose arguments hold one is
# constructed then rather than at load time
PREVIOUS_RESULT_PREFIX = "previous_result:"

# The prefix marking a value as a reference to a constant declared in python - the
# schedule a distilled model was trained on, the negative prompt a model family ships.
# Copying those into a workflow is how they go stale when the library moves on
CONSTANT_PREFIX = "constant:"

# The media such an object may be built from - it opens the file itself, so the
# extension is all that is checked here
ALLOWED_FROM_FILE_EXTENSIONS = (
    ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS
)

# The media kinds an object built from a previous step's output can carry. A type
# declares which one it is with a 'kind' attribute, the way MiniMax-H3's reference
# classes do, and that is what says which of its fields the media goes into
MEDIA_KINDS = ("image", "video", "audio")


# Helper functions for processing and loading workflow arguments
def realize_args(arg, base_dir=None):
    """
    Recursively processes workflow arguments to:
    1. Convert type references into actual Python types
    2. Load images from file paths/URLs
    3. Load videos from file paths/URLs
    4. Construct objects that name a type and the file to build it from

    Args:
        arg: The arguments to process, modified in place
        base_dir: Directory relative file paths are resolved against - the
            workflow file's directory. Defaults to the process working directory
    """
    if isinstance(arg, dict):
        logger.debug(f"Processing dictionary arguments: {list(arg.keys())}")
        for k, v in arg.items():
            # A constant resolves under any argument name, and before the
            # conventions below - what it holds is the value, not a file to load
            if is_constant_reference(v):
                arg[k] = fetch_constant(v)
            # A stored prompt resolves the same way - to its text, under any
            # name, before the media conventions could mistake it for a file
            elif is_prompt_reference(v):
                arg[k] = fetch_prompt(v)
            # An explicit media reference loads under any argument name - the
            # key conventions below only cover arguments named like their media
            elif is_media_reference(v):
                arg[k] = fetch_media(v, base_dir)
            # Handle image loading for keys ending in '_image' or exactly 'image'
            elif k.endswith("_image") or k == "image":
                logger.debug(f"Loading image for key: {k}")
                arg[k] = fetch_image(v, base_dir)
            # Handle video loading for keys ending in '_video' or exactly 'video'
            elif k.endswith("_video") or k == "video":
                logger.debug(f"Loading video for key: {k}")
                arg[k] = fetch_video(v, base_dir)
            # Handle type references, and the keys that only look like one
            elif k.endswith("_type") or k.endswith("_dtype") or k == "dtype":
                if isinstance(v, EscapedString):
                    # An earlier pass already consumed this value's escape
                    continue
                if k in NON_TYPE_KEYS:
                    # The value stays a string, but the {} escape is still honored
                    # so both the escaped and the bare spelling name the category
                    if is_escaped(v):
                        arg[k] = EscapedString(v.strip("{}"))
                    continue
                logger.debug(f"Processing type reference for key: {k}")
                # Allow escaping type references using {} brackets
                # this is for instances when the argument name is "something_type" but it is
                # not a reference to a python type, but rather a category or something else
                if isinstance(v, str):
                    if is_escaped(v):
                        arg[k] = EscapedString(v.strip("{}"))
                    else:
                        arg[k] = load_type_from_name(v)
                elif isinstance(v, type):
                    # the value already a type
                    arg[k] = v
            # Recursively process nested dictionaries, then build any object they
            # describe - the type reference it names is realized by the recursion
            else:
                realize_args(v, base_dir)
                arg[k] = realize_object(v, base_dir)

    # Recursively process lists
    elif isinstance(arg, list):
        logger.debug("Processing list arguments")
        for i, item in enumerate(arg):
            if is_constant_reference(item):
                arg[i] = fetch_constant(item)
                continue
            if is_prompt_reference(item):
                arg[i] = fetch_prompt(item)
                continue
            if is_media_reference(item):
                arg[i] = fetch_media(item, base_dir)
                continue
            realize_args(item, base_dir)
            arg[i] = realize_object(item, base_dir)


def is_constant_reference(value):
    """Whether a value references a constant declared in python."""
    return isinstance(value, str) and value.startswith(CONSTANT_PREFIX)


def is_prompt_reference(value):
    """Whether a value references a stored prompt in the prompt library."""
    return isinstance(value, str) and value.startswith(PROMPT_PREFIX)


def fetch_constant(reference):
    """Read the value a 'constant:' reference names.

    A constant is data - a schedule, a default prompt, a token budget - so what the
    name resolves to has to be data too. Anything callable is refused: a type is
    named with a '_type' key and constructed there, and a workflow that could reach
    a function through this would be evaluating python rather than referencing it.

    Mutable values are copied. The workflow holds the module's own object otherwise,
    and a pipeline that consumes its sigmas in place would edit the library's
    constant for every later run in the process - the REPL keeps one alive for a
    whole session.

    Args:
        reference: The 'constant:dotted.NAME' string

    Returns:
        The value the name refers to

    Raises:
        ValueError: If the name resolves to nothing, or to something callable
        InvalidInputError: If the name is not a dotted python name
    """
    name = validate_constant_name(reference.removeprefix(CONSTANT_PREFIX).strip())

    try:
        value = load_constant_from_name(name)
    except (ImportError, AttributeError) as error:
        raise ValueError(
            f"No constant named '{name}' - a constant reference names the module "
            f"it is declared in and the attribute to read from it ({error})"
        ) from error

    if callable(value):
        raise ValueError(
            f"'{name}' is a {type(value).__name__}, not a constant - "
            f"'{CONSTANT_PREFIX}' reads a value, and a type is named with a "
            f"'_type' argument instead"
        )

    logger.info(f"Reading constant {name}")
    return copy.deepcopy(value) if isinstance(value, (list, dict, set)) else value


def realize_constants(arg):
    """Resolve every constant reference in a structure, and nothing else.

    Variables are declared before they are set: a value passed in is converted to
    the type of the declared default, so a default that is still the string naming
    a constant would type a schedule as text. Resolving them first makes the
    constant the declared value, which is what it is meant to be.

    Args:
        arg: The structure to resolve, modified in place
    """
    if isinstance(arg, dict):
        for k, v in arg.items():
            if is_constant_reference(v):
                arg[k] = fetch_constant(v)
            else:
                realize_constants(v)
    elif isinstance(arg, list):
        for i, item in enumerate(arg):
            if is_constant_reference(item):
                arg[i] = fetch_constant(item)
            else:
                realize_constants(item)


def is_media_reference(value):
    """Whether a value is an explicit media reference.

    The form { "media_type": "image", "location": "subject.png" } says what the
    media is instead of relying on what its argument is called, so a "mask" or
    "depth_map" argument can load a file too. A bare {"location": ...} dict is
    NOT treated as one - it stays whatever its consumer expects.
    """
    return isinstance(value, dict) and "media_type" in value and "location" in value


def fetch_media(spec, base_dir=None):
    """Load the media an explicit reference names.

    Args:
        spec: Dict with 'media_type' ('image' or 'video') and 'location'
        base_dir: Directory relative paths are resolved against

    Returns:
        The loaded media - or the location string unchanged when it is a
        deferred variable/previous_result reference

    Raises:
        ValueError: If media_type names neither image nor video
        SecurityError: If the location fails validation
    """
    media_type = spec["media_type"]
    location = {"location": spec["location"]}
    if media_type == "image":
        return fetch_image(location, base_dir)
    if media_type == "video":
        return fetch_video(location, base_dir)
    raise ValueError(f"Unknown media_type {media_type!r} - use 'image' or 'video'")


def object_type_key(value, from_key):
    """The '*_type' key of an object description, or None if it is not one.

    The type to construct is named by a '*_type' key, matching the convention
    realize_args resolves. Without one the dict is not an object description - it
    is left for whatever consumes it, exactly as it was before this feature.

    Args:
        value: The dict to inspect
        from_key: The key that marked it as an object description, named in errors

    Returns:
        The name of the key holding the type, or None

    Raises:
        ValueError: If the dict names more than one type
    """
    type_keys = [k for k in value if k.endswith("_type") and k not in NON_TYPE_KEYS]
    if not type_keys:
        return None
    if len(type_keys) > 1:
        raise ValueError(
            f"'{from_key}' needs exactly one '_type' argument naming the type to "
            f"construct, got {type_keys}"
        )
    return type_keys[0]


def realize_object(value, base_dir=None):
    """Construct an argument that names a type and the media to build it from.

    Some pipelines take arguments that are objects rather than plain media - MiniMax-H3's
    references, which carry the frame rate or the sample rate of the media they hold.
    Those are written as a type and where the media comes from, either a file:

        { "reference_type": "...MiniMaxH3ImageReference", "from_file": "subject.png" }

    built by the type's own from_file(), since only it knows what to bring along with
    the media, or an earlier step of the workflow:

        { "reference_type": "...MiniMaxH3ImageReference", "from_previous_result": "draw_subject" }

    which is built by build_objects() instead, once the step it names has run. Only the
    from_file form is constructed here - the other names media that does not exist yet.
    Any other keys are arguments to from_file() where it takes them, and fields set on
    the object it returns where it does not.

    A type that has no from_file() of its own - LTX-2's conditions, which are plain
    dataclasses holding frames the caller already loaded - is written as the arguments
    to construct it with instead:

        { "condition_type": "...LTX2VideoCondition",
          "from_arguments": { "frames": { "media_type": "image", "location": "last.png" },
                              "index": -1, "strength": 1.0 } }

    Those arguments are ordinary arguments: the media in them is loaded by the recursion
    that reaches this, and one naming an earlier step ("previous_result:draw_subject")
    waits for build_objects() the same way the from_previous_result form does.

    Args:
        value: An already realized argument - anything but a dict naming a '_type'
            and a 'from_file' is returned unchanged
        base_dir: Directory a relative 'from_file' path is resolved against

    Returns:
        The constructed object, or value unchanged

    Raises:
        ValueError: If the dict names more than one type, a type that cannot be built
            from a file, or a file location that cannot be resolved
        SecurityError: If the file it names fails validation
    """
    if isinstance(value, dict) and FROM_PREVIOUS_RESULT_KEY in value:
        # Validated now and built later - a type that cannot hold the step's output
        # is a workflow error worth raising before any of it runs
        validate_deferred_object(value)
        return value

    if isinstance(value, dict) and FROM_ARGUMENTS_KEY in value:
        object_type, arguments = validate_constructed_object(value)
        if names_a_previous_result(arguments):
            # One of its arguments is a step's output, which does not exist yet -
            # build_objects constructs it once that step has run
            return value
        return construct_object(object_type, arguments)

    if not isinstance(value, dict) or FROM_FILE_KEY not in value:
        return value

    type_key = object_type_key(value, FROM_FILE_KEY)
    if type_key is None:
        return value
    type_keys = [type_key]

    object_type = value[type_keys[0]]
    if not isinstance(object_type, type):
        raise ValueError(
            f"'{type_keys[0]}' must name a type to construct, got {object_type!r}"
        )
    if not has_method(object_type, FROM_FILE_KEY):
        raise ValueError(
            f"{object_type.__name__} cannot be constructed from a file - "
            f"it has no {FROM_FILE_KEY}()"
        )

    location = value[FROM_FILE_KEY]
    if isinstance(location, str):
        # These resolve per step iteration, after objects are already built -
        # a clear error here beats a path-validation failure naming the wrong cause
        if location.startswith("previous_result:"):
            raise ValueError(
                f"'{FROM_FILE_KEY}' cannot reference a previous step's result - "
                f"it names a file the object is constructed from. Use "
                f"'{FROM_PREVIOUS_RESULT_KEY}' to build it from what a step "
                f"generated instead"
            )
        if location.startswith("variable:"):
            raise ValueError(
                f"'{FROM_FILE_KEY}' references {location!r} but no such "
                f"variable is defined"
            )

    location = validate_media_location(location, base_dir)
    logger.info(f"Constructing {object_type.__name__} from {location}")

    arguments = {
        k: v for k, v in value.items() if k not in (FROM_FILE_KEY, type_keys[0])
    }
    accepted, overrides = split_from_file_arguments(object_type, arguments)
    return apply_field_overrides(
        object_type.from_file(location, **accepted), overrides, object_type
    )


def split_from_file_arguments(object_type, arguments):
    """Split the keys of a from_file description by where the type can take them.

    A from_file() that takes keyword arguments is handed all of them - that is the
    generic case, where the type decodes the media and the arguments say how. One that
    takes the media and nothing else, which is what MiniMax-H3's references do, gets
    only what its signature names, and the rest are set on the object it returns. That
    is the way diffusers documents correcting a reference whose container lied about
    its frame rate, and without it there is no way to say so from a workflow.

    Args:
        object_type: The type being constructed
        arguments: The description's keys, minus the type and the location

    Returns:
        (arguments for from_file, fields to set on the result)
    """
    parameters = signature(getattr(object_type, FROM_FILE_KEY)).parameters
    if any(
        parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return arguments, {}

    named = {
        name
        for name, parameter in parameters.items()
        if parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }
    accepted = {k: v for k, v in arguments.items() if k in named}
    return accepted, {k: v for k, v in arguments.items() if k not in named}


def apply_field_overrides(constructed, overrides, object_type):
    """Set the fields a description names on the object that was built from its media.

    Args:
        constructed: The object from_file() returned
        overrides: The keys from_file() could not take
        object_type: The type, named in errors

    Returns:
        The object, with the fields set

    Raises:
        ValueError: If it has no field by one of those names
    """
    for name, value in overrides.items():
        if not hasattr(constructed, name):
            fields = getattr(object_type, "__dataclass_fields__", None)
            takes = f" - it holds {', '.join(fields)}" if fields else ""
            raise ValueError(
                f"{object_type.__name__} has no field '{name}', and its "
                f"{FROM_FILE_KEY}() does not take it either{takes}"
            )
        logger.debug(f"Setting {object_type.__name__}.{name} from the workflow")
        setattr(constructed, name, value)

    return constructed


def validate_deferred_object(description):
    """Check an object description built from a previous step, at load time.

    The media it names does not exist until the step it references has run, so the
    construction itself waits. What can be checked now is checked now: a workflow
    that names a type it cannot build should say so before the first model loads.

    Args:
        description: The dict naming a '_type' and a 'from_previous_result'

    Raises:
        ValueError: If it names no type, more than one, something that is not a type,
            or a type that declares no media kind
    """
    type_key = object_type_key(description, FROM_PREVIOUS_RESULT_KEY)
    if type_key is None:
        raise ValueError(
            f"'{FROM_PREVIOUS_RESULT_KEY}' needs a '_type' argument naming the type "
            f"to construct from the step's output"
        )

    object_type = description[type_key]
    if not isinstance(object_type, type):
        raise ValueError(
            f"'{type_key}' must name a type to construct, got {object_type!r}"
        )

    kind = getattr(object_type, "kind", None)
    if kind not in MEDIA_KINDS:
        raise ValueError(
            f"{object_type.__name__} cannot be constructed from a step's output - "
            f"a type built this way declares which media it holds with a 'kind' of "
            f"{', '.join(MEDIA_KINDS)}, and this one declares {kind!r}"
        )


def validate_constructed_object(description):
    """Check an object description built from the arguments it names, at load time.

    Args:
        description: The dict naming a '_type' and a 'from_arguments'

    Returns:
        (the type to construct, the arguments to construct it with)

    Raises:
        ValueError: If it names no type, more than one, something that is not a type,
            carries keys beside the two, or arguments that are not a dict
    """
    type_key = object_type_key(description, FROM_ARGUMENTS_KEY)
    if type_key is None:
        raise ValueError(
            f"'{FROM_ARGUMENTS_KEY}' needs a '_type' argument naming the type to "
            f"construct from the arguments it holds"
        )

    object_type = description[type_key]
    if not isinstance(object_type, type):
        raise ValueError(
            f"'{type_key}' must name a type to construct, got {object_type!r}"
        )

    arguments = description[FROM_ARGUMENTS_KEY]
    if not isinstance(arguments, dict):
        raise ValueError(
            f"'{FROM_ARGUMENTS_KEY}' must hold the arguments {object_type.__name__} "
            f"is constructed with, got {type(arguments).__name__}"
        )

    extra = set(description) - {type_key, FROM_ARGUMENTS_KEY}
    if extra:
        raise ValueError(
            f"'{FROM_ARGUMENTS_KEY}' holds every argument {object_type.__name__} is "
            f"constructed with - move {', '.join(sorted(extra))} inside it"
        )

    return object_type, arguments


def names_a_previous_result(value):
    """Whether anything in an argument structure references an earlier step."""
    if isinstance(value, dict):
        return any(names_a_previous_result(item) for item in value.values())
    if isinstance(value, list):
        return any(names_a_previous_result(item) for item in value)
    return isinstance(value, str) and value.startswith(PREVIOUS_RESULT_PREFIX)


def construct_object(object_type, arguments):
    """Build one object from the arguments its description named.

    Args:
        object_type: The type to construct
        arguments: The arguments to construct it with, with any reference to an
            earlier step already substituted for what it named

    Returns:
        The constructed object

    Raises:
        ValueError: If the type cannot be constructed from those arguments
    """
    logger.info(f"Constructing {object_type.__name__} from {', '.join(arguments)}")
    try:
        return object_type(**arguments)
    except TypeError as error:
        fields = getattr(object_type, "__dataclass_fields__", None)
        takes = f" - it takes {', '.join(fields)}" if fields else ""
        raise ValueError(
            f"Cannot construct {object_type.__name__} from "
            f"{', '.join(arguments) or 'no arguments'}{takes}: {error}"
        ) from error


def build_objects(arguments):
    """Construct the objects whose media came from an earlier step.

    Runs after previous_results has substituted each 'from_previous_result' with the
    artifact it named, which is the earliest the media exists. The same goes for a
    'from_arguments' description one of whose arguments named a step - realize_object
    left it standing, and by now the reference inside it holds what the step produced.
    Containers are rebuilt rather than mutated, and only where something below them
    changed - the arguments of one iteration share their nested values with every
    other iteration, so an in-place edit here would reach into all of them.

    Args:
        arguments: One iteration's arguments, with previous results substituted

    Returns:
        The arguments with every object description replaced by the built object -
        the same object where there was nothing to build
    """
    if isinstance(arguments, dict):
        if FROM_PREVIOUS_RESULT_KEY in arguments:
            return object_from_result(arguments)
        if FROM_ARGUMENTS_KEY in arguments:
            # Its own arguments may hold descriptions too - a condition built from a
            # reference built from a step - so they are built before it is
            object_type, described = validate_constructed_object(arguments)
            return construct_object(object_type, build_objects(described))
        built = {k: build_objects(v) for k, v in arguments.items()}
        return (
            built if any(built[k] is not v for k, v in arguments.items()) else arguments
        )

    if isinstance(arguments, list):
        built = [build_objects(item) for item in arguments]
        return (
            built
            if any(new is not old for new, old in zip(built, arguments))
            else arguments
        )

    return arguments


def object_from_result(description):
    """Build one object from the step output substituted into its description.

    The media is already in memory and already at the rates the step produced it at,
    so it is handed to the constructor field by field rather than through from_file().
    Which field it lands in comes from the type's own 'kind' - the convention
    MiniMax-H3's reference classes follow, and the same one the segment chain reads
    when it carries a generated segment back in as a reference.

    Args:
        description: The dict naming a '_type', with 'from_previous_result' now
            holding the artifact rather than the step name

    Returns:
        The constructed object

    Raises:
        ValueError: If the artifact is not the media the type's kind calls for
    """
    validate_deferred_object(description)
    type_key = object_type_key(description, FROM_PREVIOUS_RESULT_KEY)
    object_type = description[type_key]
    artifact = description[FROM_PREVIOUS_RESULT_KEY]

    # A workflow can still name a field itself - the frame rate of a step that
    # generated at something other than the consuming pipeline's own, say - and
    # what it names wins over what the artifact carried
    overrides = {
        k: v
        for k, v in description.items()
        if k not in (FROM_PREVIOUS_RESULT_KEY, type_key)
    }
    arguments = media_arguments(object_type, artifact)
    arguments.update(overrides)

    logger.info(
        f"Constructing {object_type.__name__} from a previous result "
        f"({', '.join(arguments)})"
    )
    return object_type(**arguments)


def media_arguments(object_type, artifact):
    """The constructor arguments a step's artifact makes, for a type's media kind.

    Imported here rather than at module scope - these pull in the result and task
    helpers, which is a heavier import than an argument file needs for the path
    that never builds one of these.

    Args:
        object_type: The type being constructed, declaring its 'kind'
        artifact: One artifact of the step the description named

    Returns:
        Dict of constructor arguments

    Raises:
        ValueError: If the artifact is not the media the kind calls for
    """
    import torch
    from PIL import Image

    from .result import AudioVideo
    from .tasks.audio_utils import as_channels_samples
    from .tasks.video_utils import frames_as_pil_list

    kind = object_type.kind

    if kind == "image":
        if not isinstance(artifact, Image.Image):
            raise ValueError(
                f"{object_type.__name__} holds an image, but the step it names "
                f"produced a {type(artifact).__name__} - reference a step that "
                f"generates images, or its 'images' property"
            )
        return {"image": artifact}

    # A generated soundtrack arrives paired with the frames it was generated with;
    # anything else the step produced carries none
    audio, sample_rate = None, None
    if isinstance(artifact, AudioVideo) and artifact.audio is not None:
        audio = torch.as_tensor(as_channels_samples(artifact.audio))
        sample_rate = artifact.sample_rate

    if kind == "video":
        frames = frames_as_pil_list(artifact)
        if not frames:
            raise ValueError(
                f"{object_type.__name__} holds a video, but the step it names "
                f"produced no frames"
            )
        arguments = {"frames": frames}
        if audio is not None:
            arguments["audio"] = audio
            arguments["sample_rate"] = sample_rate
        return arguments

    if audio is None and hasattr(artifact, "ndim") and artifact.ndim <= 3:
        # A step that generates audio alone - a music pipeline, a slice_audio
        # task - produces the waveform itself rather than an AudioVideo. It
        # carries no rate of its own, so the reference declares 'sample_rate'
        # alongside 'from_previous_result'
        audio = torch.as_tensor(as_channels_samples(artifact))

    if audio is None:
        raise ValueError(
            f"{object_type.__name__} holds audio, but the step it names produced "
            f"none - reference a step that generates a soundtrack"
        )
    arguments = {"audio": audio}
    if sample_rate is not None:
        arguments["sample_rate"] = sample_rate
    return arguments


def validate_media_location(location, base_dir=None):
    """Validate the media file an argument object is constructed from.

    The object decodes the file itself, so only where it comes from is checked here.

    Args:
        location: Path or URL of the media file
        base_dir: Directory a relative path is resolved against - the workflow
            file's directory. Defaults to the process working directory

    Returns:
        The validated path or URL

    Raises:
        ValueError: If the location is not a string
        SecurityError: If the path, URL or file extension is not allowed
    """
    if not isinstance(location, str):
        raise ValueError(
            f"'{FROM_FILE_KEY}' must be a path or a URL, got {type(location)}"
        )

    if location.startswith("http://") or location.startswith("https://"):
        return validate_url(location)

    validated_path = validate_path(
        resolve_relative_path(location, base_dir), allow_create=False
    )
    return validate_file_extension(validated_path, ALLOWED_FROM_FILE_EXTENSIONS)


def resolve_relative_path(path, base_dir):
    """Resolve a relative file path against the workflow file's directory.

    Workflow files name their media relative to themselves; absolute paths and
    callers with no base_dir keep the path as given (process working directory).
    """
    if base_dir and not os.path.isabs(os.path.expanduser(path)):
        return os.path.join(base_dir, path)
    return path


def fetch_image(img_spec, base_dir=None):
    """
    Load image from file path or URL with security validation.

    Args:
        img_spec: Image specification (file path, URL, dict with 'location' key, PIL Image, or list of any of these)
        base_dir: Directory relative file paths are resolved against - the
            workflow file's directory. Defaults to the process working directory

    Returns:
        Loaded PIL Image, list of PIL Images, or None if img_spec is None

    Raises:
        SecurityError: If validation fails
        ValueError: If img_spec is invalid type
    """
    if img_spec is None:
        return None

    # Handle lists of images (recursively process each)
    if isinstance(img_spec, list):
        logger.debug(f"Loading list of {len(img_spec)} images")
        return [fetch_image(img, base_dir) for img in img_spec]

    # If already a PIL Image, return as-is (allows multiple realize_args calls)
    if hasattr(img_spec, "mode") and hasattr(img_spec, "size"):
        logger.debug(f"Image already loaded, returning as-is")
        return img_spec

    # Handle dict format: {"location": "url_or_path"}
    if isinstance(img_spec, dict):
        if "location" not in img_spec:
            raise ValueError(
                f"Image dict must have 'location' key, got keys: {list(img_spec.keys())}"
            )
        img_spec = img_spec["location"]

    if not isinstance(img_spec, str):
        raise ValueError(f"Image specification must be a string, got {type(img_spec)}")

    # Skip cross-step and variable references — these are resolved later during execution
    if img_spec.startswith("previous_result:") or img_spec.startswith("variable:"):
        logger.debug(f"Skipping deferred reference: {img_spec}")
        return img_spec

    logger.debug(f"Loading image from: {img_spec}")

    try:
        # Check if it's a URL
        if isinstance(img_spec, str) and (
            img_spec.startswith("http://") or img_spec.startswith("https://")
        ):
            validated_url = validate_url(img_spec)
            return load_image(validated_url)
        else:
            # Treat as file path, relative to the workflow file
            validated_path = validate_path(
                resolve_relative_path(str(img_spec), base_dir), allow_create=False
            )
            # Validate file extension
            ext = os.path.splitext(validated_path)[1].lower()
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                raise SecurityError(f"Image file extension not allowed: {ext}")
            return load_image(validated_path)

    except SecurityError:
        raise
    except Exception as e:
        logger.error(f"Failed to load image {img_spec}: {e}")
        raise


def fetch_video(video_spec, base_dir=None):
    """
    Load video from file path or URL with security validation.

    Args:
        video_spec: Video specification (file path, URL, dict with 'location' key, loaded frames, or list of any of these)
        base_dir: Directory relative file paths are resolved against - the
            workflow file's directory. Defaults to the process working directory

    Returns:
        Loaded video frames, list of video frames, or None if video_spec is None

    Raises:
        SecurityError: If validation fails
        ValueError: If video_spec is invalid type
    """
    if video_spec is None:
        return None

    # Handle lists of videos (need to distinguish from video frames)
    # Check if it's a list of specifications (dicts/strings) rather than video frames
    if isinstance(video_spec, list) and len(video_spec) > 0:
        # If first element is a dict with 'location' or a string, treat as list of video specs
        if isinstance(video_spec[0], (dict, str)):
            logger.debug(f"Loading list of {len(video_spec)} videos")
            return [fetch_video(vid, base_dir) for vid in video_spec]
        # Otherwise assume it's already loaded video frames
        else:
            logger.debug(f"Video frames already loaded, returning as-is")
            return video_spec

    # If already loaded video frames (tuple), return as-is
    if isinstance(video_spec, tuple):
        logger.debug(f"Video frames already loaded, returning as-is")
        return video_spec

    # Handle dict format: {"location": "url_or_path"}
    if isinstance(video_spec, dict):
        if "location" not in video_spec:
            raise ValueError(
                f"Video dict must have 'location' key, got keys: {list(video_spec.keys())}"
            )
        video_spec = video_spec["location"]

    if not isinstance(video_spec, str):
        raise ValueError(
            f"Video specification must be a string, got {type(video_spec)}"
        )

    # Skip cross-step and variable references — these are resolved later during execution
    if video_spec.startswith("previous_result:") or video_spec.startswith("variable:"):
        logger.debug(f"Skipping deferred reference: {video_spec}")
        return video_spec

    logger.debug(f"Loading video from: {video_spec}")

    try:
        # Check if it's a URL
        if isinstance(video_spec, str) and (
            video_spec.startswith("http://") or video_spec.startswith("https://")
        ):
            validated_url = validate_url(video_spec)
            return load_video(validated_url)
        else:
            # Treat as file path, relative to the workflow file
            validated_path = validate_path(
                resolve_relative_path(str(video_spec), base_dir), allow_create=False
            )
            # Validate file extension
            ext = os.path.splitext(validated_path)[1].lower()
            if ext not in ALLOWED_VIDEO_EXTENSIONS:
                raise SecurityError(f"Video file extension not allowed: {ext}")
            return load_video(validated_path)

    except SecurityError:
        raise
    except Exception as e:
        logger.error(f"Failed to load video {video_spec}: {e}")
        raise
