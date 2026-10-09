"""
MiniMax-H3 latent upscale and decode.

`upscale_h3_latents` resizes the video latents a MiniMax-H3 pipeline returns
(`output: [..., "latents"]`) with a vendored 3D-convolution network
(`h3_latent_upscaler_model`), so a 544p take can be promoted to 1344x768
without denoising it again. `decode_h3_latents` turns latents back into
frames through diffusers' own H3 video decode block, so the VAE's
denormalization, autocast and pixel un-normalization stay diffusers' rather
than a copy that could drift from it.

The network was trained behind the ComfyUI node's `(x - mean) / std ...
x * std + mean` wrapper, and the wrapper is carried here. ComfyUI's H3 VAE
encodes to latents already normalized by the VAE's `latents_mean` /
`latents_std` - the space a diffusers H3 pipeline hands over too - and the
node normalizes them a second time with the same statistics before the
network sees them. Calling the network on the pipeline's latents directly
feeds it a space it never saw, and the decode is garbage (#499, M-F041).
"""

import logging

import torch

from ..result import AudioVideo

logger = logging.getLogger("dw")

DEFAULT_UPSCALER_REPO = "LBH-123-AI/Minimax_h3_latent_Upscaler"
# weight_name is a bare file name (SE-F039: a '/' is refused), so the
# default repo's checkpoints are read from the folder they sit in there
DEFAULT_UPSCALER_SUBFOLDER = "minimax_h3_latent_upscaler_3d_conv_v1"
DEFAULT_UPSCALER_WEIGHTS = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
# The commit the default weights are read at: the repo holds several
# checkpoints under one name, and a later push could change what the default
# file is without changing its path
UPSCALER_REVISION = "3f941d5d182014dd5c0a5e16330420ee2d4aa0c6"

# Other upscaler checkpoints known to be the v1 architecture, by (repo, file):
# the folder each sits in and the commit it is read at. A source not named
# here is read from its repo root at main
LMS_UPSCALER_REPO = "Alissonerdx/Minimax-H3-ComfyUI"
KNOWN_UPSCALER_SOURCES = {
    (DEFAULT_UPSCALER_REPO, None): (DEFAULT_UPSCALER_SUBFOLDER, UPSCALER_REVISION),
    # Sharpness-tuned from the default, same keys and shapes (#612)
    (LMS_UPSCALER_REPO, "h3_upscaler_lms_v0.1.safetensors"): (
        "latent_upscaler",
        "849fb3d4f434f2e9b4a4d4f7253471bf5e6719f9",
    ),
}

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

# The per-channel statistics the upscaler's training normalized its latents
# by - the ComfyUI node's LATENTS_MEAN / LATENTS_STD, equal to the H3 VAE
# config's latents_mean / latents_std
LATENTS_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
)


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


def _check_upscaler_source(model_name, weight_name):
    """Refuse an upscaler source that is not a Hub repo and a safetensors file in it.

    hf_hub_download takes a repo id and a file inside it, and the weights are
    read with safetensors, so anything else - a local path, a name climbing
    out of the repo, a pickle format - is refused rather than handed on.
    """
    from ..locations import (
        SAFETENSORS_SUFFIX,
        validate_hub_repo_id,
        validate_weight_name,
    )

    validate_hub_repo_id(model_name)
    validate_weight_name(weight_name, (SAFETENSORS_SUFFIX,), subfolders=False)


def _load_upscaler(device, model_name, weight_name):
    from huggingface_hub import hf_hub_download

    from .h3_latent_upscaler_model import build_upscaler
    from .model_cache import cached_model

    # The pin names a commit of the default repo; a caller naming another
    # repo gets that repo's main
    source = KNOWN_UPSCALER_SOURCES.get(
        (model_name, weight_name)
    ) or KNOWN_UPSCALER_SOURCES.get((model_name, None))
    subfolder, revision = source if source else (None, None)

    def load_net():
        from safetensors.torch import load_file

        path = hf_hub_download(
            repo_id=model_name,
            filename=weight_name,
            subfolder=subfolder,
            revision=revision,
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
        weight_name: The safetensors file within it, a bare file name (default:
            the v1 bf16 checkpoint). In the default repo it is read from the
            v1 checkpoint folder, so the v1 fp16 file is named the same way.
            `Alissonerdx/Minimax-H3-ComfyUI` with `h3_upscaler_lms_v0.1.safetensors`
            is the sharpness-tuned LMS upscaler, read at a pinned commit

    Returns:
        The upscaled latents, (B, 24, T, height / 16, width / 16), in the
        input's dtype and on the input's device
    """
    size, (scale_h, scale_w) = target_latent_size(latents, width, height)
    model_name = model_name or DEFAULT_UPSCALER_REPO
    weight_name = weight_name or DEFAULT_UPSCALER_WEIGHTS
    # Before anything downloads: a value arriving through a variable or an
    # earlier step was never in the document validation checked
    _check_upscaler_source(model_name, weight_name)

    net = _load_upscaler(device, model_name, weight_name)
    # Upstream conditions the network on the mean of the two spatial scales
    scale = (scale_h + scale_w) / 2
    logger.info(
        f"Upscaling H3 latents {list(latents.shape[-3:])} -> {list(size)} "
        f"(scale {scale:.3f}) on {device}"
    )
    dtype = _network_dtype(device)
    mean, std = _latent_stats(device)
    with torch.no_grad():
        # The node's wrapper, in float32 so bf16 does not round the statistics
        x = (latents.to(device=device, dtype=torch.float32) - mean) / std
        upscaled = net(x.to(dtype), scale=scale, target_size=size)
        upscaled = upscaled.to(torch.float32) * std + mean
    return upscaled.to(device=latents.device, dtype=latents.dtype)


def _latent_stats(device):
    """The training normalization's mean and std, shaped (1, 24, 1, 1, 1)."""
    shape = (1, LATENT_CHANNELS, 1, 1, 1)
    mean = torch.tensor(LATENTS_MEAN, dtype=torch.float32, device=device)
    std = torch.tensor(LATENTS_STD, dtype=torch.float32, device=device)
    return mean.view(shape), std.view(shape)


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
