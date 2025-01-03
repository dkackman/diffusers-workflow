import logging
from .toolbox.type_helpers import load_type_from_name
from diffusers.utils import load_image, load_video

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


def fetch_image(image):
    """
    Loads an image from a file path/URL or processes image configuration
    Args:
        image: Can be:
            - A string (treated as reference)
            - A dict with 'location' key (path/URL to load)
            - A dict with optional 'size' key for resizing
    Returns:
        Loaded and optionally resized image
    """
    # Handle string references (usually for intermediate results)
    if isinstance(image, str):
        logger.debug(f"Using image reference: {image}")
        return image

    # Load image from location and apply optional resizing
    if isinstance(image, dict) and "location" in image:
        logger.info(f"Loading image from: {image['location']}")
        img = load_image(image["location"])

        if "size" in image:
            logger.debug(f"Resizing image to: {image['size']}")
            img = img.resize((image["size"]["width"], image["size"]["height"]))

        return img

    return image


def fetch_video(video):
    """
    Loads a video from a file path/URL
    Args:
        video: Can be:
            - A string (treated as reference)
            - A dict with 'location' key (path/URL to load)
    Returns:
        Loaded video
    """
    # Handle string references (usually for intermediate results)
    if isinstance(video, str):
        logger.debug(f"Using video reference: {video}")
        return video

    # Load video from location
    if isinstance(video, dict) and "location" in video:
        logger.info(f"Loading video from: {video['location']}")
        return load_video(video["location"])

    return video
