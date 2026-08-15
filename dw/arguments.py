import os
import logging
from .type_helpers import load_type_from_name, has_method
from diffusers.utils import load_image, load_video
from .security import (
    validate_path,
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

# The key naming the file an argument object is constructed from
FROM_FILE_KEY = "from_file"

# The media such an object may be built from - it opens the file itself, so the
# extension is all that is checked here
ALLOWED_FROM_FILE_EXTENSIONS = (
    ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS
)


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
            # An explicit media reference loads under any argument name - the
            # key conventions below only cover arguments named like their media
            if is_media_reference(v):
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
                if k in NON_TYPE_KEYS:
                    # The value stays a string, but the {} escape is still honored
                    # so both the escaped and the bare spelling name the category
                    if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                        arg[k] = v.strip("{}")
                    continue
                logger.debug(f"Processing type reference for key: {k}")
                # Allow escaping type references using {} brackets
                # this is for instances when the argument name is "something_type" but it is
                # not a reference to a python type, but rather a category or something else
                if isinstance(v, str):
                    if v.startswith("{") and v.endswith("}"):
                        arg[k] = v.strip("{}")
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
            if is_media_reference(item):
                arg[i] = fetch_media(item, base_dir)
                continue
            realize_args(item, base_dir)
            arg[i] = realize_object(item, base_dir)


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


def realize_object(value, base_dir=None):
    """Construct an argument that names a type and a file to build it from.

    Some pipelines take arguments that are objects rather than plain media - MiniMax-H3's
    references, which carry the frame rate or the sample rate of the media they hold.
    Those are written as a type and a file:

        { "reference_type": "...MiniMaxH3ImageReference", "from_file": "subject.png" }

    and built by the type's own from_file(), since only it knows what to bring along with
    the media. Any other keys are passed to from_file() as keyword arguments.

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
    if not isinstance(value, dict) or FROM_FILE_KEY not in value:
        return value

    # The type to construct is named by a '*_type' key, matching the convention
    # realize_args resolves. Without one the dict is not an object description -
    # it is left for whatever consumes it, exactly as it was before this feature
    type_keys = [k for k in value if k.endswith("_type") and k not in NON_TYPE_KEYS]
    if not type_keys:
        return value
    if len(type_keys) > 1:
        raise ValueError(
            f"'{FROM_FILE_KEY}' needs exactly one '_type' argument naming the type to "
            f"construct, got {type_keys}"
        )

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
                f"it names a file the object is constructed from. Reference a "
                f"saved file's path or a URL instead"
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
    return object_type.from_file(location, **arguments)


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
