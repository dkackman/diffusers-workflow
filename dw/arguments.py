import os
import logging
from .type_helpers import load_type_from_name
from diffusers.utils import load_image, load_video
from .security import (
    validate_path,
    validate_url,
    SecurityError,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
)

logger = logging.getLogger("dw")


# Helper functions for processing and loading workflow arguments
def realize_args(arg):
    """
    Recursively processes workflow arguments to:
    1. Convert type references into actual Python types
    2. Load images from file paths/URLs
    3. Load videos from file paths/URLs
    """
    if isinstance(arg, dict):
        logger.debug(f"Processing dictionary arguments: {list(arg.keys())}")
        for k, v in arg.items():
            # Handle image loading for keys ending in '_image' or exactly 'image'
            if k.endswith("_image") or k == "image":
                logger.debug(f"Loading image for key: {k}")
                arg[k] = fetch_image(v)
            # Handle video loading for keys ending in '_video' or exactly 'video'
            elif k.endswith("_video") or k == "video":
                logger.debug(f"Loading video for key: {k}")
                arg[k] = fetch_video(v)
            # Handle type references (except 'content_type')
            elif (k.endswith("_type") or k.endswith("_dtype")) and k != "content_type":
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
            # Recursively process nested dictionaries
            else:
                realize_args(v)

    # Recursively process lists
    elif isinstance(arg, list):
        logger.debug("Processing list arguments")
        for item in arg:
            realize_args(item)


def fetch_image(img_spec):
    """
    Load image from file path or URL with security validation.

    Args:
        img_spec: Image specification (file path, URL, dict with 'location' key, PIL Image, or list of any of these)

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
        return [fetch_image(img) for img in img_spec]

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

    logger.debug(f"Loading image from: {img_spec}")

    try:
        # Check if it's a URL
        if isinstance(img_spec, str) and (
            img_spec.startswith("http://") or img_spec.startswith("https://")
        ):
            validated_url = validate_url(img_spec)
            return load_image(validated_url)
        else:
            # Treat as file path
            validated_path = validate_path(str(img_spec), allow_create=False)
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


def fetch_video(video_spec):
    """
    Load video from file path or URL with security validation.

    Args:
        video_spec: Video specification (file path, URL, dict with 'location' key, loaded frames, or list of any of these)

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
            return [fetch_video(vid) for vid in video_spec]
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

    logger.debug(f"Loading video from: {video_spec}")

    try:
        # Check if it's a URL
        if isinstance(video_spec, str) and (
            video_spec.startswith("http://") or video_spec.startswith("https://")
        ):
            validated_url = validate_url(video_spec)
            return load_video(validated_url)
        else:
            # Treat as file path
            validated_path = validate_path(str(video_spec), allow_create=False)
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
