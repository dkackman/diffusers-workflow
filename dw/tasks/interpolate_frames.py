"""
Video frame interpolation via RIFE (Real-Time Intermediate Flow Estimation).

Takes a list of video frames and generates intermediate frames to increase
frame rate. Supports 2x, 4x, and 8x multipliers.

Model weights are downloaded from HuggingFace Hub on first use.
"""

import logging
import torch

from .tensor_image import pil_to_float_tensor as _pil_to_tensor, float_tensor_to_pil

logger = logging.getLogger("dw")

_VALID_MULTIPLIERS = {2, 4, 8}


def interpolate_frames(video, device="cpu", **kwargs):
    """Interpolate between video frames using RIFE to increase frame rate.

    Args:
        video: List of PIL Images (video frames)
        device: Target device ("cuda", "mps", "cpu")
        **kwargs:
            multiplier: Frame count multiplier — 2, 4, or 8 (default: 2)
            model_name: HuggingFace repo with RIFE v4.13 weights (default: auto)
            filename: Weights filename within the repo (default: auto)

    Returns:
        List of PIL Images with interpolated frames inserted.
    """
    multiplier = int(kwargs.get("multiplier", 2))
    model_name = kwargs.get("model_name", None)
    filename = kwargs.get("filename", None)

    if multiplier not in _VALID_MULTIPLIERS:
        raise ValueError(
            f"multiplier must be one of {sorted(_VALID_MULTIPLIERS)}, got {multiplier}"
        )

    if len(video) < 2:
        raise ValueError(f"Need at least 2 frames to interpolate, got {len(video)}")

    logger.info(
        f"Interpolating {len(video)} frames with {multiplier}x multiplier on {device}"
    )

    model = _load_rife_model(device, model_name, filename)

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


_DEFAULT_RIFE_REPO = "imaginairy/rife-interpolation"
_DEFAULT_RIFE_FILENAME = "rife-flownet-4.13.2.safetensors"


def _pad_tensor(t, ph, pw):
    """Zero-pad a (1, 3, H, W) tensor on the bottom/right to (ph, pw)."""
    h, w = t.shape[2], t.shape[3]
    padding = (0, pw - w, 0, ph - h)
    return torch.nn.functional.pad(t, padding)


def _padded_size(h, w, multiple=32):
    """Round (h, w) up to the next multiple (RIFE requires dims divisible by 32)."""
    ph = ((h - 1) // multiple + 1) * multiple
    pw = ((w - 1) // multiple + 1) * multiple
    return ph, pw


def _make_flow_context(ph, pw, device):
    """Build the warp grid and flow divisors for one padded resolution."""
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
    return tenFlow_div, backwarp_tenGrid, timestep


def _load_rife_model(device, model_name=None, filename=None):
    """Load RIFE model and return a callable that interpolates two frames.

    Args:
        device: Target device string ("cuda", "mps", "cpu").
        model_name: Optional HuggingFace repo ID containing RIFE v4.13 weights.
            Defaults to imaginairy/rife-interpolation.
        filename: Optional weights filename within the repo. Defaults to
            rife-flownet-4.13.2.safetensors. Both .safetensors and torch
            checkpoint formats (.pkl/.pth) are supported.

    Returns:
        Callable that takes (frame1: PIL.Image, frame2: PIL.Image) -> PIL.Image
    """
    from .rife_model import IFNet
    from huggingface_hub import hf_hub_download
    from .model_cache import cached_model

    repo_id = model_name if model_name is not None else _DEFAULT_RIFE_REPO
    weights_file = filename if filename is not None else _DEFAULT_RIFE_FILENAME

    def load_net():
        model_path = hf_hub_download(repo_id=repo_id, filename=weights_file)

        logger.info(f"Loading RIFE IFNet v4.13 to {device}")

        if weights_file.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        # Strip "module." prefix that comes from DataParallel-saved checkpoints
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.removeprefix("module.")] = v

        net = IFNet()
        net.load_state_dict(cleaned)
        net.eval()
        net.to(device)
        return net

    net = cached_model(
        ("interpolate_frames", repo_id, weights_file, str(device)), load_net
    )

    return _build_inference(net, device)


def _build_inference(net, device):
    """Build the (img1, img2) -> mid_frame callable for a loaded RIFE net.

    All frames of a video share one padded resolution, so the flow-warp
    context (tenFlow_div, backwarp grid, timestep) is computed once per
    (padded_h, padded_w) and reused for every pair instead of being rebuilt
    on every call.

    `_interpolate_2x` walks frames pairwise: (f0, f1), (f1, f2), (f2, f3), ...
    — the second frame of one pair is the same PIL object as the first frame
    of the next. This closure carries the padded tensor it produced for a
    pair's second frame forward, so that when that exact frame object shows
    up again as a pair's first frame, its tensor is reused instead of being
    re-derived from the PIL image (re-decoded, re-normalized, re-padded). A
    cache miss (non-matching object, e.g. across separate interpolation
    passes) simply falls back to a fresh conversion, so this is a pure
    optimization with no behavioral effect — the reused tensor is bit-for-bit
    what a fresh conversion of the same frame would produce.
    """
    scale_list = [8, 4, 2, 1]
    flow_context_cache = {}
    carry = {"img": None}

    def get_flow_context(ph, pw):
        key = (ph, pw)
        ctx = flow_context_cache.get(key)
        if ctx is None:
            ctx = _make_flow_context(ph, pw, device)
            flow_context_cache[key] = ctx
        return ctx

    def inference(img1, img2):
        """Interpolate a single frame between two input frames."""
        if img1 is carry["img"]:
            t1_padded = carry["tensor"]
            h, w, ph, pw = carry["h"], carry["w"], carry["ph"], carry["pw"]
        else:
            t1 = _pil_to_tensor(img1, device)
            h, w = t1.shape[2], t1.shape[3]
            ph, pw = _padded_size(h, w)
            t1_padded = _pad_tensor(t1, ph, pw)

        t2 = _pil_to_tensor(img2, device)
        t2_padded = _pad_tensor(t2, ph, pw)

        # Carry img2's padded tensor forward in case it's the next pair's img1.
        carry["img"] = img2
        carry["tensor"] = t2_padded
        carry["h"], carry["w"], carry["ph"], carry["pw"] = h, w, ph, pw

        tenFlow_div, backwarp_tenGrid, timestep = get_flow_context(ph, pw)

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
        return float_tensor_to_pil(mid)

    return inference
