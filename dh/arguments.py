from .toolbox.type_helpers import load_type_from_name
from diffusers.utils import load_image, load_video


# Helper functions for processing and loading workflow arguments
def realize_args(arg):
    """
    Recursively processes workflow arguments to:
    1. Convert type references into actual Python types
    2. Load images from file paths/URLs
    3. Load videos from file paths/URLs
    """
    if isinstance(arg, dict):
        for k, v in arg.items():
            # Handle image loading for keys ending in '_image' or exactly 'image'
            if k.endswith("_image") or k == "image":
                arg[k] = fetch_image(v)
            # Handle video loading for keys ending in '_video' or exactly 'video'
            elif k.endswith("_video") or k == "video":
                arg[k] = fetch_video(v)
            # Handle type references (except 'content_type')
            elif (k.endswith("_type") or k.endswith("_dtype")) and k != "content_type":
                # Allow escaping type references using {} brackets
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    arg[k] = v.strip("{}")
                else:
                    arg[k] = load_type_from_name(v)
            # Recursively process nested dictionaries
            else:
                realize_args(v)

    # Recursively process lists
    elif isinstance(arg, list):
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
        return image

    # Load image from location and apply optional resizing
    if isinstance(image, dict) and "location" in image:
        img = load_image(image["location"])

        if "size" in image:
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
        return video

    # Load video from location
    if isinstance(video, dict) and "location" in video:
        return load_video(video["location"])

    return video
