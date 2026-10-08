"""Loading the media a workflow argument names: an explicit
{media_type, location} reference, an `image`/`video` argument's path or URL, and
the frame rate and shots a loaded video carries."""

import contextlib
import io
import os
import logging
import tempfile
from urllib.parse import unquote, urlparse
from diffusers.utils import load_image, load_video
from PIL import Image
from . import references
from .runs import shots_beside
from .security import (
    SecurityError,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
)
from .locations import is_http_url, safe_get, validate_media_path

logger = logging.getLogger("dw")


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


def _describe_value_source(value):
    """A short, human phrase for what a mistyped value already is - the
    'source' half of an argument-mismatch error, since the type name alone
    (PIL.Image.Image) doesn't say *how* it got there."""
    if hasattr(value, "mode") and hasattr(value, "size"):
        return "an already-loaded image"
    if isinstance(value, tuple) and value and hasattr(value[0], "size"):
        return "already-loaded video frames"
    return f"a {type(value).__name__}"


def fetch_image_with_context(v, base_dir, key):
    """fetch_image, with the argument key folded into a type-mismatch error -
    a bare 'got <class ...>' names neither the argument nor what the value
    already was (#365)."""
    try:
        return fetch_image(v, base_dir)
    except ValueError as error:
        raise ValueError(
            f"{error} (argument '{key}' expected an image, got "
            f"{_describe_value_source(v)} - check what variable or previous "
            f"result feeds it)"
        ) from error


def fetch_video_with_context(v, base_dir, key, with_audio=False):
    """fetch_video, with the same argument-key context as
    fetch_image_with_context. `with_audio` reads a file's soundtrack along with
    its frames, as an AudioVideo - an H3 guide's `"audio": true` beside its
    `video` (#649), where frames alone leave the guide no audio to hold."""
    try:
        if with_audio:
            return _fetch_audio_video(v, base_dir)
        return fetch_video(v, base_dir)
    except ValueError as error:
        raise ValueError(
            f"{error} (argument '{key}' expected a video, got "
            f"{_describe_value_source(v)} - check what variable or previous "
            f"result feeds it)"
        ) from error


def _fetch_audio_video(video_spec, base_dir=None):
    """A video file - a path, URL or {"location": ...} - loaded with its audio
    as an AudioVideo. A deferred reference stays for later resolution, and a
    value already loaded passes through, so a second realization keeps it."""
    from .tasks.video_utils import is_video_location, load_audio_video

    if isinstance(video_spec, str) and references.is_ref(
        references.LAZY_MEDIA, video_spec
    ):
        return video_spec
    if is_video_location(video_spec):
        return load_audio_video(video_spec, base_dir)
    return video_spec


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
        logger.debug("Image already loaded, returning as-is")
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
    if references.is_ref(references.LAZY_MEDIA, img_spec):
        logger.debug(f"Skipping deferred reference: {img_spec}")
        return img_spec

    logger.debug(f"Loading image from: {img_spec}")

    try:
        # Check if it's a URL
        if isinstance(img_spec, str) and (
            img_spec.startswith("http://") or img_spec.startswith("https://")
        ):
            # Fetched here rather than by load_image, which follows redirects
            # without re-checking them; load_image still does the EXIF
            # transpose and RGB conversion on the decoded result
            response = safe_get(img_spec, "an image argument", timeout=60)
            return load_image(Image.open(io.BytesIO(response.content)))
        else:
            # Treat as file path, relative to the workflow file, and confined
            # to the directories this workflow may read (dw/locations.py)
            validated_path = validate_media_path(
                str(img_spec), base_dir, "an image argument"
            )
            # Validate file extension
            ext = os.path.splitext(validated_path)[1].lower()
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                raise SecurityError(
                    f"Image file extension not allowed: {ext} - a video file "
                    "must go through video_frames first"
                )
            return load_image(validated_path)

    except SecurityError:
        raise
    except Exception as e:
        logger.error(f"Failed to load image {img_spec}: {e}")
        raise


def _with_frame_rate(frames, location):
    """The loaded frames carrying the rate their file declares, and the
    shot boundaries its run (or kept-asset sidecar) recorded for it.

    `load_video` reads frames and drops both: a step that paired a 24 fps
    file with a soundtrack wrote it back at 8 - three times long, silently
    (#104) - and a video loaded from an `asset:`/`output:` path had no
    `shots` to hand `pair_audio`, even when the server had them on file
    for that exact video (#398). The rate is read from the container
    without decoding anything; the shots come from `shots_beside`, which
    only looks at a real local path. A file that says neither stays a plain
    list.
    """
    from .tasks.video_utils import FrameList

    if not isinstance(frames, list):
        return frames
    fps = _declared_fps(location)
    shots = shots_beside(location)
    return FrameList(frames, fps, shots) if (fps or shots) else frames


def _declared_fps(path):
    """The rate a video file declares, or None - a container that will not
    open, carries no video stream or states no rate is a rate we do not
    know, never an error: the caller is loading frames it has already read.
    """
    try:
        from .media import container_fps

        return container_fps(path)
    except Exception as e:
        logger.debug(f"No frame rate for {path}: {e}")
        return None


@contextlib.contextmanager
def local_media_file(location, what, default_suffix=""):
    """`location` as a path a loader opens: a path as it is, an http(s) URL
    fetched through `safe_get` into a temporary file that lasts as long as
    the `with` block.

    A loader handed a URL fetches it itself - diffusers' `load_video`,
    `load_image` and its H3 references' `from_file` all call requests
    directly, resolving the host a second time and following redirects
    unchecked, with no cap on the body. Handed a path, they only decode.
    The suffix comes from the URL, as those loaders' own downloads name
    it, since a container is often told apart by it. The loader must have
    read the file before the block ends.
    """
    if not is_http_url(location):
        yield location
        return
    response = safe_get(location, what, timeout=300)
    suffix = os.path.splitext(unquote(urlparse(location).path))[1] or default_suffix
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        with handle:
            handle.write(response.content)
        yield handle.name
    finally:
        os.remove(handle.name)


def _fetch_remote_video(url):
    """A video URL's frames, decoded from `local_media_file`'s copy."""
    from .tasks.video_utils import FrameList

    with local_media_file(url, "a video argument", default_suffix=".mp4") as path:
        frames = load_video(path)
        fps = _declared_fps(path)
    # A URL has no run beside it, so it carries no shots
    return FrameList(frames, fps, None) if fps else frames


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

    # An explicit {"media_type": ..., "location": ...} reference says what the
    # media is regardless of the argument it fills - a still handed to a
    # 'video' argument this way loads as an image rather than hitting the
    # extension gate below (#443). Checked ahead of the list/dict handling so
    # it also applies per-item inside a list of mixed video/image references,
    # which realize_args's own is_media_reference check never sees - a list
    # is not itself a dict, so a 'video'-named list reaches fetch_video whole
    if is_media_reference(video_spec):
        return fetch_media(video_spec, base_dir)

    # Handle lists of videos (need to distinguish from video frames)
    # Check if it's a list of specifications (dicts/strings) rather than video frames
    if isinstance(video_spec, list) and len(video_spec) > 0:
        # If first element is a dict with 'location' or a string, treat as list of video specs
        if isinstance(video_spec[0], (dict, str)):
            logger.debug(f"Loading list of {len(video_spec)} videos")
            return [fetch_video(vid, base_dir) for vid in video_spec]
        # Otherwise assume it's already loaded video frames
        else:
            logger.debug("Video frames already loaded, returning as-is")
            return video_spec

    # If already loaded video frames (tuple), return as-is
    if isinstance(video_spec, tuple):
        logger.debug("Video frames already loaded, returning as-is")
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
    if references.is_ref(references.LAZY_MEDIA, video_spec):
        logger.debug(f"Skipping deferred reference: {video_spec}")
        return video_spec

    logger.debug(f"Loading video from: {video_spec}")

    try:
        # Check if it's a URL
        if isinstance(video_spec, str) and (
            video_spec.startswith("http://") or video_spec.startswith("https://")
        ):
            return _fetch_remote_video(video_spec)
        else:
            # Treat as file path, relative to the workflow file, and confined
            # to the directories this workflow may read (dw/locations.py)
            validated_path = validate_media_path(
                str(video_spec), base_dir, "a video argument"
            )
            # Validate file extension
            ext = os.path.splitext(validated_path)[1].lower()
            if ext not in ALLOWED_VIDEO_EXTENSIONS:
                raise SecurityError(f"Video file extension not allowed: {ext}")
            return _with_frame_rate(load_video(validated_path), validated_path)

    except SecurityError:
        raise
    except Exception as e:
        logger.error(f"Failed to load video {video_spec}: {e}")
        raise
