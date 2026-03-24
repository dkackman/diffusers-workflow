"""
Image upscaling via spandrel.

Supports 40+ super-resolution architectures (ESRGAN, SwinIR, HAT, DAT, etc.)
with automatic model detection from .pth/.safetensors files.

Models can be loaded from local files or HuggingFace Hub repos.
"""

import logging
import torch
import numpy as np
from PIL import Image

logger = logging.getLogger("dw")

# Maximum tile size before we switch to tiled processing
_MAX_PIXELS_NO_TILE = 512 * 512


def upscale_image(image, model_name, device="cpu", **kwargs):
    """Upscale an image using a spandrel-compatible super-resolution model.

    Args:
        image: PIL Image to upscale
        model_name: HuggingFace repo ID (e.g., "user/repo") with optional filename,
                     or local path to .pth/.safetensors file.
        device: Target device ("cuda", "mps", "cpu")
        **kwargs:
            filename: Weight file name within a HF repo (default: auto-detect)
            tile_size: Tile size for large images (default: 512)
            tile_overlap: Overlap between tiles in pixels (default: 32)

    Returns:
        PIL Image (upscaled)
    """
    try:
        from spandrel import ModelLoader, ImageModelDescriptor
    except ImportError:
        raise ImportError(
            "spandrel is required for upscaling. Install with: pip install spandrel"
        )

    filename = kwargs.get("filename", None)
    tile_size = kwargs.get("tile_size", 512)
    tile_overlap = kwargs.get("tile_overlap", 32)

    model_path = _resolve_model_path(model_name, filename)

    logger.info(f"Loading upscale model from {model_path}")
    loader = ModelLoader(device=torch.device(device))
    descriptor = loader.load_from_file(model_path)

    if not isinstance(descriptor, ImageModelDescriptor):
        raise ValueError(
            f"Model is not an image model (got {type(descriptor).__name__}). "
            f"Only image super-resolution models are supported."
        )

    logger.info(
        f"Loaded {descriptor.architecture.name} "
        f"(scale: {descriptor.scale}x, "
        f"input: {descriptor.input_channels}ch, "
        f"output: {descriptor.output_channels}ch)"
    )

    # Use half precision if supported and on GPU
    if device != "cpu" and descriptor.supports_half:
        descriptor.model.half()
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    # Convert PIL to tensor
    img_array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=model_dtype)

    h, w = tensor.shape[2], tensor.shape[3]

    if h * w <= _MAX_PIXELS_NO_TILE:
        logger.debug(f"Upscaling {w}x{h} directly")
        with torch.inference_mode():
            output = descriptor(tensor)
    else:
        logger.debug(f"Upscaling {w}x{h} with {tile_size}px tiles")
        output = _tiled_inference(descriptor, tensor, tile_size, tile_overlap)

    # Convert back to PIL
    output = output.squeeze(0).permute(1, 2, 0)
    output = (output.clamp(0, 1) * 255).byte().cpu().numpy()
    result = Image.fromarray(output)

    logger.info(f"Upscaled {w}x{h} -> {result.width}x{result.height}")
    return result


def _resolve_model_path(model_name, filename=None):
    """Resolve a model name to a local file path.

    Supports:
        - Local file paths: "/path/to/model.pth"
        - HuggingFace Hub: "user/repo" (auto-detects .pth/.safetensors)
        - HuggingFace Hub with filename: model_name="user/repo", filename="model_x4.pth"
    """
    import os

    # Local file
    if os.path.exists(model_name):
        return model_name

    # HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            f"Model '{model_name}' is not a local file. "
            "Install huggingface_hub to download from HF Hub."
        )

    if filename is not None:
        logger.debug(f"Downloading {filename} from {model_name}")
        return hf_hub_download(repo_id=model_name, filename=filename)

    # Auto-detect: list repo files and find a model file
    from huggingface_hub import list_repo_files

    model_extensions = {".pth", ".pt", ".ckpt", ".safetensors"}
    try:
        files = list_repo_files(model_name)
    except Exception as e:
        raise ValueError(f"Could not access HuggingFace repo '{model_name}': {e}")

    model_files = [f for f in files if any(f.endswith(ext) for ext in model_extensions)]
    if not model_files:
        raise ValueError(
            f"No model files ({', '.join(model_extensions)}) found in '{model_name}'. "
            f"Specify 'filename' explicitly."
        )
    if len(model_files) > 1:
        raise ValueError(
            f"Multiple model files found in '{model_name}': {model_files}. "
            f"Specify 'filename' explicitly."
        )

    logger.debug(f"Downloading {model_files[0]} from {model_name}")
    return hf_hub_download(repo_id=model_name, filename=model_files[0])


def _tiled_inference(descriptor, tensor, tile_size, overlap):
    """Run model inference on overlapping tiles and blend results.

    Splits the input into tiles, runs each through the model, then
    blends overlapping regions with linear interpolation.
    """
    scale = descriptor.scale
    _, c, h, w = tensor.shape
    out_h, out_w = h * scale, w * scale
    output = torch.zeros(1, c, out_h, out_w, device=tensor.device, dtype=tensor.dtype)
    weight = torch.zeros(1, 1, out_h, out_w, device=tensor.device, dtype=tensor.dtype)

    # Generate tile positions
    y_positions = list(range(0, h, tile_size - overlap))
    x_positions = list(range(0, w, tile_size - overlap))

    # Clamp last tile to image boundary
    y_positions = [min(y, max(0, h - tile_size)) for y in y_positions]
    x_positions = [min(x, max(0, w - tile_size)) for x in x_positions]

    # Deduplicate
    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))

    total_tiles = len(y_positions) * len(x_positions)
    logger.debug(f"Processing {total_tiles} tiles ({tile_size}px, {overlap}px overlap)")

    tile_num = 0
    for y in y_positions:
        for x in x_positions:
            tile_num += 1
            th = min(tile_size, h - y)
            tw = min(tile_size, w - x)

            tile = tensor[:, :, y : y + th, x : x + tw]

            with torch.inference_mode():
                tile_out = descriptor(tile)

            oy, ox = y * scale, x * scale
            oth, otw = th * scale, tw * scale

            output[:, :, oy : oy + oth, ox : ox + otw] += tile_out
            weight[:, :, oy : oy + oth, ox : ox + otw] += 1

    # Average overlapping regions
    output = output / weight.clamp(min=1)
    return output
