from .toolbox.type_helpers import load_type_from_name
from diffusers.utils import load_image, load_video


#
# This recursvively processes the arguments of a workflow
# replacing type references with the actual types
# plus loading any images and videos from their locations
#
def realize_args(arg):
    if isinstance(arg, dict):
        for k, v in arg.items():
            if k.endswith("_image") or k == "image":
                arg[k] = fetch_image(v)
            elif k.endswith("_video") or k == "video":
                arg[k] = fetch_video(v)
            elif (k.endswith("_type") or k.endswith("_dtype")) and k != "content_type":
                # use {} to escape key value pairs that are not type references
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    arg[k] = v.strip("{}")
                else:
                    arg[k] = load_type_from_name(v)
            else:
                realize_args(v)

    elif isinstance(arg, list):
        for item in arg:
            realize_args(item)


def fetch_image(image):
    # escape indicator for intermediate result references
    if isinstance(image, str):
        return image

    if isinstance(image, dict) and "location" in image:
        img = load_image(image["location"])

        if "size" in image:
            img = img.resize((image["size"]["width"], image["size"]["height"]))

        return img

    return image


def fetch_video(video):
    # escape indicator for intermediate result references
    if isinstance(video, str):
        return video

    return load_video(video["location"])
