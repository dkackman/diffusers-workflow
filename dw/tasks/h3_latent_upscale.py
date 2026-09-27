"""
MiniMax-H3 latent upscale and decode.

`upscale_h3_latents` resizes the video latents a MiniMax-H3 pipeline returns
(`output: [..., "latents"]`) with a vendored 3D-convolution network
(`h3_latent_upscaler_model`), so a 544p take can be promoted to 1344x768
without denoising it again. `decode_h3_latents` turns latents back into
frames through diffusers' own H3 video decode block, so the VAE's
denormalization, autocast and pixel un-normalization stay diffusers' rather
than a copy that could drift from it.

The latents a diffusers H3 pipeline hands over are already in the normalized
space the upscaler was trained on, so the network is called directly - the
ComfyUI node's `(x - mean) / std ... x * std + mean` wrapper is not carried,
and applying it here would normalize twice.
"""

import logging

import torch

from ..result import AudioVideo

logger = logging.getLogger("dw")

DEFAULT_UPSCALER_REPO = "LBH-123-AI/Minimax_h3_latent_Upscaler"
DEFAULT_UPSCALER_WEIGHTS = (
    "minimax_h3_latent_upscaler_3d_conv_v1/"
    "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
)
# The commit the default weights are read at: the repo holds several
# checkpoints under one name, and a later push could change what the default
# file is without changing its path
UPSCALER_REVISION = "3f941d5d182014dd5c0a5e16330420ee2d4aa0c6"

DEFAULT_H3_REPO = "MiniMaxAI/MiniMax-H3"

# H3's video VAE compresses 16x in each spatial dimension
SPATIAL_FACTOR = 16
LATENT_CHANNELS = 24
# The per-axis scale range the upscaler was trained over
MIN_SCALE = 1.0
MAX_SCALE = 4.0
# H3's canvas: a short edge of at most 768 and a long edge of at most 1344
MAX_SHORT_EDGE = 768
MAX_LONG_EDGE = 1344
# The rate H3 generates video at
H3_FPS = 24


def target_latent_size(latents, width, height):
    """The (T, H, W) latent size a pixel target names, or a directed refusal.

    Args:
        latents: H3 video latents, (B, 24, T, H, W)
        width: Target width in pixels, a multiple of 16
        height: Target height in pixels, a multiple of 16

    Returns:
        ((T, H, W), (height scale, width scale)) - T unchanged

    Raises:
        ValueError: The latents are not H3 video latents, or the target is
            off the latent grid, outside the network's scale range or over
            H3's canvas
    """
    if not isinstance(latents, torch.Tensor):
        raise ValueError(
            f"latents must be MiniMax-H3 video latents (a tensor), not a "
            f"{type(latents).__name__} - pass a pipeline step's latents, e.g. "
            "'previous_result:base.latents' from a step whose output lists "
            "'latents'"
        )
    if latents.ndim != 5 or latents.shape[1] != LATENT_CHANNELS:
        raise ValueError(
            f"latents must be MiniMax-H3 video latents, shaped (batch, "
            f"{LATENT_CHANNELS}, frames, height, width); got shape "
            f"{list(latents.shape)}"
        )
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{name} must be a positive whole number of pixels, got {value!r}"
            )
        if value % SPATIAL_FACTOR:
            raise ValueError(
                f"{name} must be a multiple of {SPATIAL_FACTOR} (the H3 VAE's "
                f"spatial factor), got {value}; the nearest are "
                f"{value // SPATIAL_FACTOR * SPATIAL_FACTOR} and "
                f"{(value // SPATIAL_FACTOR + 1) * SPATIAL_FACTOR}"
            )
    if min(width, height) > MAX_SHORT_EDGE or max(width, height) > MAX_LONG_EDGE:
        raise ValueError(
            f"width x height {width}x{height} is over MiniMax-H3's canvas: "
            f"at most {MAX_LONG_EDGE}x{MAX_SHORT_EDGE} (or "
            f"{MAX_SHORT_EDGE}x{MAX_LONG_EDGE} portrait)"
        )

    frames, in_h, in_w = latents.shape[-3:]
    out_h, out_w = height // SPATIAL_FACTOR, width // SPATIAL_FACTOR
    scales = (out_h / in_h, out_w / in_w)
    for name, scale, source in (
        ("height", scales[0], in_h * SPATIAL_FACTOR),
        ("width", scales[1], in_w * SPATIAL_FACTOR),
    ):
        if not MIN_SCALE <= scale <= MAX_SCALE:
            raise ValueError(
                f"{name} {source * scale:.0f} is a {scale:.2f}x scale of the "
                f"latents' {source}; the upscaler takes {MIN_SCALE:g}x to "
                f"{MAX_SCALE:g}x per axis, so {name} must be "
                f"{source} to {int(source * MAX_SCALE)}"
            )
    return (frames, out_h, out_w), scales


def _network_dtype(device):
    # The weights are bf16; a CPU run widens them, where bf16 conv3d is slow
    return torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16


def _load_upscaler(device, model_name, weight_name):
    from huggingface_hub import hf_hub_download

    from .h3_latent_upscaler_model import build_upscaler
    from .model_cache import cached_model

    # The pin names a commit of the default repo; a caller naming another
    # repo gets that repo's main
    revision = UPSCALER_REVISION if model_name == DEFAULT_UPSCALER_REPO else None

    def load_net():
        from safetensors.torch import load_file

        path = hf_hub_download(
            repo_id=model_name, filename=weight_name, revision=revision
        )
        logger.info(f"Loading MiniMax-H3 latent upscaler {weight_name} to {device}")
        net = build_upscaler(load_file(path))
        return net.to(device=device, dtype=_network_dtype(device))

    return cached_model(
        ("upscale_h3_latents", model_name, weight_name, revision, str(device)),
        load_net,
    )


def upscale_h3_latents(
    latents,
    width,
    height,
    device="cpu",
    model_name=None,
    weight_name=None,
):
    """Resize MiniMax-H3 video latents to a larger canvas.

    Args:
        latents: H3 video latents, (B, 24, T, H, W), as a pipeline step
            returns them
        width: Target width in pixels, a multiple of 16
        height: Target height in pixels, a multiple of 16
        device: Device the network runs on
        model_name: Hugging Face repo holding the upscaler (default:
            LBH-123-AI/Minimax_h3_latent_Upscaler, read at a pinned revision)
        weight_name: The safetensors file within it (default: the v1 bf16
            checkpoint)

    Returns:
        The upscaled latents, (B, 24, T, height / 16, width / 16), in the
        input's dtype and on the input's device
    """
    size, (scale_h, scale_w) = target_latent_size(latents, width, height)
    model_name = model_name or DEFAULT_UPSCALER_REPO
    weight_name = weight_name or DEFAULT_UPSCALER_WEIGHTS

    net = _load_upscaler(device, model_name, weight_name)
    # Upstream conditions the network on the mean of the two spatial scales
    scale = (scale_h + scale_w) / 2
    logger.info(
        f"Upscaling H3 latents {list(latents.shape[-3:])} -> {list(size)} "
        f"(scale {scale:.3f}) on {device}"
    )
    with torch.no_grad():
        x = latents.to(device=device, dtype=_network_dtype(device))
        upscaled = net(x, scale=scale, target_size=size)
    return upscaled.to(device=latents.device, dtype=latents.dtype)


def _load_decoder(device, model_name):
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep

    from .model_cache import cached_model

    def load_decoder():
        # The block's own pipeline, holding only the components it expects:
        # the VAE from the repo's `vae` subfolder, and the video processor
        # the block configures itself
        pipeline = MiniMaxH3VideoDecodeStep().init_pipeline(model_name)
        pipeline.load_components(names=["vae"], torch_dtype=torch.float32)
        logger.info(f"Loading MiniMax-H3 video VAE from {model_name} to {device}")
        return pipeline.to(device)

    return cached_model(("decode_h3_latents", model_name, str(device)), load_decoder)


def decode_h3_latents(latents, device="cpu", model_name=None):
    """Decode MiniMax-H3 video latents into frames.

    Runs diffusers' `MiniMaxH3VideoDecodeStep` - latent denormalization, the
    VAE decode under its autocast, and ImageNet pixel un-normalization - so
    the colours match what the pipeline itself would have decoded.

    Args:
        latents: H3 video latents, (B, 24, T, H, W)
        device: Device the VAE runs on
        model_name: The MiniMax-H3 repo whose `vae` subfolder decodes
            (default: MiniMaxAI/MiniMax-H3)

    Returns:
        One AudioVideo holding the first video's frames at 24 fps and no
        audio - pair_audio puts the base pass's soundtrack back under it
    """
    if (
        not isinstance(latents, torch.Tensor)
        or latents.ndim != 5
        or latents.shape[1] != LATENT_CHANNELS
    ):
        shape = (
            list(latents.shape)
            if isinstance(latents, torch.Tensor)
            else type(latents).__name__
        )
        raise ValueError(
            f"latents must be MiniMax-H3 video latents, shaped (batch, "
            f"{LATENT_CHANNELS}, frames, height, width); got {shape}"
        )
    pipeline = _load_decoder(device, model_name or DEFAULT_H3_REPO)
    x = latents.to(device=pipeline._execution_device, dtype=torch.float32)
    videos = pipeline(latents=x, output_type="pil", output="videos")
    frames = list(videos[0])
    logger.info(
        f"Decoded {len(frames)} frames at {frames[0].size[0]}x{frames[0].size[1]}"
    )
    return AudioVideo(frames, None, None, fps=H3_FPS)
