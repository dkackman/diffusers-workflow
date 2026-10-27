"""
Prompt-free object removal with LaMa (#956).

`remove_object` erases what a mask covers and fills the hole from the
surrounding texture - no prompt, no diffusion. The network is big-lama, loaded
through spandrel as a `MaskedImageModelDescriptor` from a safetensors state
dict at a pinned Hub revision. Only the masked region comes from the model:
everything outside the dilated mask is the input's own pixels.

LaMa runs in fp32 on every device. It is ~51M parameters, so a lower
precision saves next to nothing, and spandrel marks it `supports_half=False`.
"""

import logging

import torch
from PIL import Image, ImageFilter

from ..events import emit_warning
from ..task_domains import check_argument, whole_number
from .tensor_image import float_tensor_to_pil, pil_to_float_tensor
from .weights import file_entries

logger = logging.getLogger("dw")

DEFAULT_REMOVER_REPO = "michaelgold/big-lama"
DEFAULT_REMOVER_WEIGHTS = "big-lama.safetensors"
# The commit of the default repo the weights are read at. biali/big-lama holds
# a byte-identical big-lama.safetensors (same LFS hash): if michaelgold's repo
# disappears, point DEFAULT_REMOVER_REPO and this pin at it
REMOVER_REVISION = "98ab7cface89a594b381f32fc7a9f2cae3f91ccc"

DEFAULT_DILATE = 8
# A mask pixel at or above this reads as fill
MASK_THRESHOLD = 128
EMPTY_MASK_WARNING = (
    "remove_object: the mask is all black, so there is nothing to remove; "
    "the image is returned unchanged."
)


def binarize_mask(mask_image):
    """The mask as an L image of 0 (keep) and 255 (fill), thresholded at 128."""
    if not isinstance(mask_image, Image.Image):
        raise ValueError(
            f"remove_object needs 'mask_image' as an image, got "
            f"{type(mask_image).__name__}"
        )
    return mask_image.convert("L").point(lambda p: 255 if p >= MASK_THRESHOLD else 0)


def mask_is_empty(mask_image):
    """True when no pixel of the mask reads as fill - nothing to remove."""
    return binarize_mask(mask_image).getbbox() is None


def prepare_mask(mask_image, size, dilate=DEFAULT_DILATE):
    """The fill mask for an image of `size`: binarized at 128, resized
    (nearest) to the image if it differs, then grown by `dilate` pixels.

    LaMa fills cleanly only when the mask overshoots the object's edge, and a
    `segment` mask is tight to it, hence the dilation.
    """
    mask = binarize_mask(mask_image)
    if mask.size != tuple(size):
        mask = mask.resize(tuple(size), Image.NEAREST)
    if dilate:
        mask = mask.filter(ImageFilter.MaxFilter(2 * dilate + 1))
    return mask


def _check_remover_source(model_name, weight_name):
    """Refuse a remover source that is not a Hub repo and a safetensors file
    in it - the run-time half of the SAFETENSORS_ONLY_COMMANDS gate, for a
    value that arrives through a variable or a run that skipped validation."""
    from ..locations import SAFETENSORS_SUFFIX, validate_hub_repo_id
    from ..security import validate_weight_name

    validate_hub_repo_id(model_name)
    validate_weight_name(weight_name, (SAFETENSORS_SUFFIX,), subfolders=False)


def remover_source(arguments):
    """The (repo, file, revision) a remove_object step loads - `model_name`
    and `weight_name` over the defaults. The pin names a commit of the
    default file; another source is read at its main."""
    arguments = arguments or {}
    model_name = arguments.get("model_name") or DEFAULT_REMOVER_REPO
    weight_name = arguments.get("weight_name") or DEFAULT_REMOVER_WEIGHTS
    pinned = (model_name, weight_name) == (
        DEFAULT_REMOVER_REPO,
        DEFAULT_REMOVER_WEIGHTS,
    )
    return model_name, weight_name, REMOVER_REVISION if pinned else None


def required_weights(arguments):
    """The remover's weight file, for the plan (dw/tasks/weights.py)."""
    return file_entries(*remover_source(arguments))


def _descriptor_from_state_dict(state_dict):
    """The spandrel descriptor for a state dict, refused unless it is a masked
    (inpainting) model."""
    from spandrel import MaskedImageModelDescriptor, ModelLoader

    descriptor = ModelLoader().load_from_state_dict(state_dict)
    if not isinstance(descriptor, MaskedImageModelDescriptor):
        raise ValueError(
            f"remove_object needs an inpainting model such as LaMa; these "
            f"weights load as {descriptor.architecture.name} "
            f"({type(descriptor).__name__}), not a masked image model."
        )
    return descriptor


def _load_remover(device, model_name, weight_name):
    from huggingface_hub import hf_hub_download

    from .model_cache import cached_model

    model_name, weight_name, revision = remover_source(
        {"model_name": model_name, "weight_name": weight_name}
    )

    def load_descriptor():
        from safetensors.torch import load_file

        path = hf_hub_download(
            repo_id=model_name, filename=weight_name, revision=revision
        )
        logger.info(f"Loading object remover {model_name}/{weight_name} to {device}")
        descriptor = _descriptor_from_state_dict(load_file(path))
        descriptor.to(torch.device(device))
        descriptor.float()
        descriptor.eval()
        return descriptor

    return cached_model(
        ("remove_object", model_name, weight_name, revision, str(device)),
        load_descriptor,
    )


def remove_object(
    image,
    mask_image,
    dilate=DEFAULT_DILATE,
    device="cpu",
    model_name=DEFAULT_REMOVER_REPO,
    weight_name=DEFAULT_REMOVER_WEIGHTS,
):
    """Erase what a mask covers and fill it from the surroundings, with no prompt.

    Args:
        image: The image (or, through the command, video frames) to clean
        mask_image: White where to erase, black where to keep - the polarity
            `segment` returns, so its mask chains with no `invert`. Converted
            to grayscale, binarized at 128 and resized to the image if it
            differs
        dilate: Pixels to grow the mask by before the fill, a whole number 0
            or above; LaMa fills cleanly only past the object's edge
        device: Device the network runs on
        model_name: Hugging Face repo holding the LaMa weights (default:
            michaelgold/big-lama, read at a pinned revision)
        weight_name: The safetensors file within it, a bare file name
            (default: big-lama.safetensors)

    Returns:
        PIL Image the size of the input; pixels outside the dilated mask are
        the input's own
    """
    dilate = whole_number(dilate, "dilate", "remove_object")
    dilate = DEFAULT_DILATE if dilate is None else dilate
    check_argument("remove_object", "dilate", dilate)
    model_name, weight_name, _ = remover_source(
        {"model_name": model_name, "weight_name": weight_name}
    )
    _check_remover_source(model_name, weight_name)

    if mask_is_empty(mask_image):
        emit_warning(EMPTY_MASK_WARNING)
        return image

    source = image.convert("RGB")
    mask = prepare_mask(mask_image, source.size, dilate)
    descriptor = _load_remover(device, model_name, weight_name)

    image_tensor = pil_to_float_tensor(source, device, dtype=torch.float32)
    mask_tensor = (
        pil_to_float_tensor(mask, device, dtype=torch.float32)[:, :1]
        .gt(0.5)
        .to(torch.float32)
    )
    with torch.inference_mode():
        output = descriptor(image_tensor, mask_tensor)
    filled = float_tensor_to_pil(output)

    logger.info(f"Removed the masked region of a {source.width}x{source.height} image")
    # Only the masked region comes from the model
    return Image.composite(filled, source, mask)
