"""
Shared PIL <-> float tensor conversions.

Consolidates the PIL-to-tensor round trip that was previously hand-rolled
independently in upscale.py, restore_faces.py, and interpolate_frames.py.
"""

import numpy as np
import torch
from PIL import Image


def pil_to_float_tensor(image, device, dtype=None):
    """Convert a PIL image to a (1, 3, H, W) float tensor in [0, 1] on device.

    The image is coerced to RGB first, so single-channel or RGBA inputs are
    handled consistently. `dtype` defaults to float32 (the numpy source
    precision); pass e.g. `torch.float16` to cast directly to a model's
    working precision.

    Args:
        image: PIL Image
        device: Target device (str or torch.device)
        dtype: Optional torch dtype to cast to (default: float32)

    Returns:
        torch.Tensor of shape (1, 3, H, W)
    """
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def float_tensor_to_pil(tensor):
    """Convert a (1, 3, H, W) or (3, H, W) float tensor in [0, 1] to a PIL RGB image.

    Quantizes to uint8 via rounding (`.round()`) rather than truncation
    (a bare truncating cast), matching diffusers' VaeImageProcessor.numpy_to_pil
    behavior. This matters: with truncation, an exact 8-bit value that
    round-trips through [0, 1] float (e.g. 128/255) can land a hair below
    its integer (127.999...) and get chopped down a level instead of
    landing back on 128. Rounding fixes that, at the cost of at most 1/255
    of drift per channel versus the old truncating behavior for values that
    were never exact to begin with.

    Args:
        tensor: torch.Tensor of shape (1, 3, H, W) or (3, H, W), values in [0, 1]

    Returns:
        PIL.Image.Image in RGB mode
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.permute(1, 2, 0).mul(255).round().clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(arr)
