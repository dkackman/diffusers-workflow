"""
Video frame interpolation via RIFE (Real-Time Intermediate Flow Estimation).

Takes a list of video frames and generates intermediate frames to increase
frame rate. Supports 2x, 4x, and 8x multipliers.

Model weights are downloaded from HuggingFace Hub on first use.
"""

import logging
import torch
import numpy as np
from PIL import Image

logger = logging.getLogger("dw")

_VALID_MULTIPLIERS = {2, 4, 8}


def interpolate_frames(video, device="cpu", **kwargs):
    """Interpolate between video frames using RIFE to increase frame rate.

    Args:
        video: List of PIL Images (video frames)
        device: Target device ("cuda", "mps", "cpu")
        **kwargs:
            multiplier: Frame count multiplier — 2, 4, or 8 (default: 2)
            model_name: HuggingFace repo with RIFE weights (default: auto)

    Returns:
        List of PIL Images with interpolated frames inserted.
    """
    multiplier = int(kwargs.get("multiplier", 2))
    model_name = kwargs.get("model_name", None)

    if multiplier not in _VALID_MULTIPLIERS:
        raise ValueError(
            f"multiplier must be one of {sorted(_VALID_MULTIPLIERS)}, got {multiplier}"
        )

    if len(video) < 2:
        raise ValueError(f"Need at least 2 frames to interpolate, got {len(video)}")

    logger.info(
        f"Interpolating {len(video)} frames with {multiplier}x multiplier on {device}"
    )

    model = _load_rife_model(device, model_name)

    passes = {2: 1, 4: 2, 8: 3}[multiplier]
    frames = list(video)

    for pass_num in range(passes):
        logger.debug(
            f"Interpolation pass {pass_num + 1}/{passes}: {len(frames)} frames"
        )
        frames = _interpolate_2x(frames, model)

    logger.info(f"Interpolation complete: {len(video)} -> {len(frames)} frames")
    return frames


def _interpolate_2x(frames, model):
    """Single pass of 2x interpolation — insert one frame between each pair."""
    result = [frames[0]]
    for i in range(len(frames) - 1):
        mid_frame = model(frames[i], frames[i + 1])
        result.append(mid_frame)
        result.append(frames[i + 1])
    return result


_RIFE_WEIGHTS_URL = (
    "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.6.pkl"
)
_RIFE_WEIGHTS_FILENAME = "flownet_v4.6.pkl"


def _load_rife_model(device, model_name=None):
    """Load RIFE model and return a callable that interpolates two frames.

    Args:
        device: Target device string ("cuda", "mps", "cpu").
        model_name: Optional HuggingFace repo ID containing RIFE weights.
            If None, weights are downloaded from the vs-rife GitHub release.

    Returns:
        Callable that takes (frame1: PIL.Image, frame2: PIL.Image) -> PIL.Image
    """
    from .rife_model import IFNet

    model_path = _download_weights(model_name)

    logger.info(f"Loading RIFE IFNet v4.6 to {device}")

    net = IFNet()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # Strip "module." prefix that comes from DataParallel-saved checkpoints
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix("module.")] = v

    net.load_state_dict(cleaned)
    net.to(device)

    def inference(img1, img2):
        """Interpolate a single frame between two input frames."""
        arr1 = np.array(img1.convert("RGB")).astype(np.float32) / 255.0
        arr2 = np.array(img2.convert("RGB")).astype(np.float32) / 255.0
        t1 = torch.from_numpy(arr1).permute(2, 0, 1).unsqueeze(0).to(device)
        t2 = torch.from_numpy(arr2).permute(2, 0, 1).unsqueeze(0).to(device)

        h, w = t1.shape[2], t1.shape[3]
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        t1_padded = torch.nn.functional.pad(t1, padding)
        t2_padded = torch.nn.functional.pad(t2, padding)

        # Precompute warp grid and flow divisors for the padded resolution
        tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], device=device)
        backwarp_tenGrid = torch.cat(
            [
                torch.linspace(-1.0, 1.0, pw, device=device)
                .view(1, 1, 1, pw)
                .expand(-1, -1, ph, -1),
                torch.linspace(-1.0, 1.0, ph, device=device)
                .view(1, 1, ph, 1)
                .expand(-1, -1, -1, pw),
            ],
            1,
        )

        timestep = torch.full((1, 1, ph, pw), 0.5, dtype=torch.float32, device=device)
        scale_list = [8, 4, 2, 1]

        with torch.inference_mode():
            _, _, merged = net(
                t1_padded,
                t2_padded,
                timestep,
                scale_list,
                tenFlow_div,
                backwarp_tenGrid,
            )

        mid = merged[3][:, :, :h, :w]
        mid = mid.squeeze(0).permute(1, 2, 0)
        mid = (mid.clamp(0, 1) * 255).byte().cpu().numpy()
        return Image.fromarray(mid)

    return inference


def _download_weights(model_name=None):
    """Download RIFE weights and return the local file path.

    If model_name is a HuggingFace repo ID, downloads from HF Hub.
    Otherwise downloads the default weights from the vs-rife GitHub release.
    """
    if model_name is not None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for RIFE model download from HF. "
                "Install with: pip install huggingface_hub"
            )
        return hf_hub_download(repo_id=model_name, filename=_RIFE_WEIGHTS_FILENAME)

    # Download from GitHub releases, caching in torch hub dir
    import os
    import urllib.request

    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    cached_path = os.path.join(cache_dir, _RIFE_WEIGHTS_FILENAME)

    if not os.path.exists(cached_path):
        logger.info(f"Downloading RIFE v4.6 weights to {cached_path}")
        urllib.request.urlretrieve(_RIFE_WEIGHTS_URL, cached_path)

    return cached_path
