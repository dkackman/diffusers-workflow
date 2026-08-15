import torch
import numpy as np
from transformers import pipeline
from torchvision import transforms

from .. import get_device_type
from .model_cache import cached_model


def make_hint_tensor(image, device, dtype=None):
    """Estimate depth and return it as a hint tensor for a controlnet pipeline.

    Args:
        image: Image to estimate depth from
        device: Device to run the estimator on and place the hint on
        dtype: Dtype of the hint, defaulting to the one the device works best in.
            The hint has to match the dtype of the pipeline that consumes it.

    Returns:
        Depth hint as a tensor of shape (1, 3, height, width)
    """
    depth_estimator = cached_model(
        ("depth_estimator", str(device)),
        lambda: pipeline("depth-estimation", device=device),
    )

    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    hint = detected_map.permute(2, 0, 1)

    if dtype is None:
        # float16 produces NaN values on MPS and is unsupported by many CPU operations
        dtype = torch.float16 if get_device_type(device) == "cuda" else torch.float32

    return hint.unsqueeze(0).to(device=device, dtype=dtype)


def make_hint_image(image, device, dtype=None):
    """Estimate depth and return it as an image.

    Args:
        image: Image to estimate depth from
        device: Device to run the estimator on
        dtype: Dtype to compute the hint in - see make_hint_tensor

    Returns:
        Depth map as a PIL image
    """
    hint = make_hint_tensor(image, device, dtype)
    # Convert the tensor to a Pillow image
    to_pil = transforms.ToPILImage()
    return to_pil(hint[0].float().cpu())
