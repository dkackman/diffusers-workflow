"""
MiniMax-H3 latent upscale and decode.

`upscale_h3_latents` resizes the video latents a MiniMax-H3 pipeline returns
(`output: [..., "latents"]`) with a vendored 3D-convolution network
(`h3_latent_upscaler_model`), so a 544p take can be promoted to a plate of up to
1920x1088 (a `refine_tiles` refine, #1043, then denoises it) without denoising
it again here. `decode_h3_latents` turns latents back into
frames through diffusers' own H3 video decode block, so the VAE's
denormalization, autocast and pixel un-normalization stay diffusers' rather
than a copy that could drift from it. `encode_h3_latents` goes the other way:
a finished take's pixels into one H3 latent, a VAE clip at a time, for a long
windowed refine (#1199 Task 17 Step 4).

The network was trained behind the ComfyUI node's `(x - mean) / std ...
x * std + mean` wrapper, and the wrapper is carried here. ComfyUI's H3 VAE
encodes to latents already normalized by the VAE's `latents_mean` /
`latents_std` - the space a diffusers H3 pipeline hands over too - and the
node normalizes them a second time with the same statistics before the
network sees them. Calling the network on the pipeline's latents directly
feeds it a space it never saw, and the decode is garbage (#499, M-F041).
"""

import logging
import math

import numpy
import torch

from ..media import media_duration
from ..media_readers import (
    ShortFrameRange,
    count_video_frames,
    read_frame_range,
    video_shape,
)
from ..pipeline_processors.h3_rules import H3_FPS, RENDER_GRID
from ..result import AudioVideo
from ..task_domains import whole_number
from ..variable_constraints import aligned
from .weights import file_entries

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
# The target is a plate for a tiled `refine_tiles` refine (#1043), which
# denoises it in tiles of at most 1344x768. 1920x1088 is 2x a 960x544 take,
# inside the upscaler's trained 1-4x range. An upscale-only plate above 768p
# is soft until it is refined.
MAX_SHORT_EDGE = 1088
MAX_LONG_EDGE = 1920

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


def _whole(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def target_size_problems(width, height):
    """What is wrong with a literal pixel target, as [(argument, message)].

    The one rule validate (the task's `static_check`) and the run
    (`target_latent_size`) both apply, so a size is refused at both or
    neither, in the same words: each side on the 16 px grid, and the pair
    within the 1920x1088 (or 1088x1920 portrait) cap. A side that is not a
    positive whole number - a reference or a variable not yet resolved - is
    left to the run.
    """
    problems = []
    for name, value in (("width", width), ("height", height)):
        if _whole(value) and value % SPATIAL_FACTOR:
            problems.append(
                (
                    name,
                    f"{name} must be a multiple of {SPATIAL_FACTOR} (the H3 "
                    f"VAE's spatial factor), got {value}; the nearest are "
                    f"{value // SPATIAL_FACTOR * SPATIAL_FACTOR} and "
                    f"{(value // SPATIAL_FACTOR + 1) * SPATIAL_FACTOR}",
                )
            )
    if not (_whole(width) and _whole(height)):
        return problems
    long_name, short_name = (
        ("width", "height") if width >= height else ("height", "width")
    )
    over_long = max(width, height) > MAX_LONG_EDGE
    if over_long or min(width, height) > MAX_SHORT_EDGE:
        # Named on the side that breaks the cap
        problems.append(
            (
                long_name if over_long else short_name,
                f"width x height {width}x{height} is over the upscale cap: "
                f"at most {MAX_LONG_EDGE}x{MAX_SHORT_EDGE} (or "
                f"{MAX_SHORT_EDGE}x{MAX_LONG_EDGE} portrait)",
            )
        )
    return problems


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
    problems = target_size_problems(width, height)
    if problems:
        raise ValueError(problems[0][1])

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
    from ..locations import SAFETENSORS_SUFFIX, validate_hub_repo_id
    from ..security import validate_weight_name

    validate_hub_repo_id(model_name)
    validate_weight_name(weight_name, (SAFETENSORS_SUFFIX,), subfolders=False)


def upscaler_source(arguments):
    """The (repo, file, subfolder, revision) an upscale_h3_latents step
    loads - `model_name` and `weight_name` over the defaults. The pin names a
    commit of the default repo; a caller naming another repo gets that
    repo's main."""
    arguments = arguments or {}
    model_name = arguments.get("model_name") or DEFAULT_UPSCALER_REPO
    weight_name = arguments.get("weight_name") or DEFAULT_UPSCALER_WEIGHTS
    source = KNOWN_UPSCALER_SOURCES.get(
        (model_name, weight_name)
    ) or KNOWN_UPSCALER_SOURCES.get((model_name, None))
    subfolder, revision = source if source else (None, None)
    return model_name, weight_name, subfolder, revision


def required_weights(arguments):
    """The upscaler's weight file, for the plan (dw/tasks/weights.py). The
    H3 VAE that decodes its latents is not listed: the H3 pipeline step that
    makes them lists MiniMaxAI/MiniMax-H3 whole (#975 Q4)."""
    model_name, weight_name, subfolder, revision = upscaler_source(arguments)
    if not isinstance(weight_name, str):
        return []
    filename = f"{subfolder}/{weight_name}" if subfolder else weight_name
    return file_entries(model_name, filename, revision)


def _load_upscaler(device, model_name, weight_name):
    from huggingface_hub import hf_hub_download

    from .h3_latent_upscaler_model import build_upscaler
    from .model_cache import cached_model

    model_name, weight_name, subfolder, revision = upscaler_source(
        {"model_name": model_name, "weight_name": weight_name}
    )

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
    model_name, weight_name, _, _ = upscaler_source(
        {"model_name": model_name, "weight_name": weight_name}
    )
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
    # The stock video decode, a piece of the take at a time: the same frames,
    # without the whole take's pixels on the device (a long take's do not fit)
    from ..pipeline_processors.h3_windows import DwH3StreamedVideoDecodeStep
    from .. import quantized_cache
    from .model_cache import cached_model

    def load_decoder():
        # The block's own pipeline, holding only the components it expects:
        # the VAE from the repo's `vae` subfolder, and the video processor
        # the block configures itself
        pipeline = DwH3StreamedVideoDecodeStep().init_pipeline(model_name)
        quantized_cache.load_components(
            pipeline, {"names": ["vae"], "torch_dtype": torch.float32}
        )
        logger.info(f"Loading MiniMax-H3 video VAE from {model_name} to {device}")
        return pipeline.to(device)

    return cached_model(("decode_h3_latents", model_name, str(device)), load_decoder)


def decoded_frame_count(num_latents, clip_length=17, token_drop=3, temporal_ratio=4):
    """How many frames `AutoencoderKLMiniMaxH3._decode` returns for
    `num_latents` latent frames - for any count, not only the 5m + 2 grid.

    `_decode` re-adds `token_drop` tokens, pads them up to whole chunks of
    `ceil(clip_length / ratio)` tokens, decodes all but the last chunk (each
    gives `clip_length` frames) plus the final overlap of
    `token_overlap * ratio - frame_pre_padding` frames, and cuts back the
    pixel frames the padded tokens produced. tests/test_h3_latent_encode.py
    drives the real `_decode` with a stub clip decoder and pins this to it."""
    chunk = math.ceil(clip_length / temporal_ratio)
    pre_padding = (-clip_length) % temporal_ratio
    overlap_tokens = (-token_drop) % chunk
    tokens = num_latents + token_drop
    pad = (-tokens) % chunk
    chunks = (tokens + pad) // chunk - int(token_drop > 0)
    if chunks < 1:
        return 0
    frames = chunks * (chunk * temporal_ratio - pre_padding)
    if token_drop > 0:
        frames += max(overlap_tokens * temporal_ratio - pre_padding, 0)
    intra_tail = clip_length % temporal_ratio
    for k in range(pad):
        at_end = intra_tail and (num_latents + k) % chunk == 0
        frames -= intra_tail if at_end else temporal_ratio
    return frames


def decode_h3_latents(latents, device="cpu", model_name=None, num_frames=None):
    """Decode MiniMax-H3 video latents into frames.

    Runs diffusers' `MiniMaxH3VideoDecodeStep` - latent denormalization, the
    VAE decode under its autocast, and ImageNet pixel un-normalization - so
    the colours match what the pipeline itself would have decoded.

    Args:
        latents: H3 video latents, (B, 24, T, H, W)
        device: Device the VAE runs on
        model_name: The MiniMax-H3 repo whose `vae` subfolder decodes
            (default: MiniMaxAI/MiniMax-H3)
        num_frames: Keep only the first this many frames - an
            `encode_h3_latents` step's `source_frames`, so a take padded onto
            the 17m + 5 grid comes back at its own length

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
    num_frames = whole_number(num_frames, "num_frames", "decode_h3_latents")
    if num_frames is not None:
        decodable = decoded_frame_count(latents.shape[2])
        if num_frames > decodable:
            raise ValueError(
                f"decode_h3_latents: num_frames {num_frames} is more than the "
                f"{decodable} frames the latents decode to"
            )
    pipeline = _load_decoder(device, model_name or DEFAULT_H3_REPO)
    x = latents.to(device=pipeline._execution_device, dtype=torch.float32)
    videos = pipeline(latents=x, output_type="pil", output="videos")
    frames = list(videos[0])
    if num_frames is not None:
        frames = frames[:num_frames]
    logger.info(
        f"Decoded {len(frames)} frames at {frames[0].size[0]}x{frames[0].size[1]}"
    )
    return AudioVideo(frames, None, None, fps=H3_FPS)


# Encode: a finished take into one latent, a VAE clip at a time


def encode_take(vae, read, source_frames, pixel_mean, pixel_std, seed, device):
    """A take of `source_frames` uint8 frames as one normalized H3 video latent,
    encoded a VAE clip at a time - exactly what diffusers' whole-take
    `encode_vae_condition` gives for the take padded onto the 17m + 5 grid.

    `read(start, stop)` returns source frames `[start, stop)` as a
    (n, height, width, 3) uint8 array; it is called once per clip, for at most
    one clip's frames, so neither the take nor its float32 pixels are ever
    whole in memory.

    Why it is exact. `AutoencoderKLMiniMaxH3._encode` already cuts the take
    into `clip_length`-frame clips, encodes each with `_encode_clip` on its
    own (the encoder is causal within a clip only), concatenates the moments
    and drops `token_drop` trailing latents once. The padding is the last
    frame repeated, both onto the grid (`fit_source_frames`) and onto whole
    clips (`_encode`), so clip i is source frames `17i .. 17i + 16` clamped to
    the last. The posterior sample's noise is one draw over the whole latent's
    shape from a CPU generator seeded `seed`, as `DiagonalGaussianDistribution
    .sample` draws it; it is latent-sized (about 165 MB at 60 s 768p) and is
    sliced per clip. Everything after the encoder is elementwise. Calls the
    VAE's private `_encode_clip`, which tests/test_h3_latent_encode.py pins
    against the public encode.

    Returns:
        `(latents, num_frames)`: the (1, 24, 5m + 2, h / 16, w / 16) float32
        CPU latent, and the 17m + 5 frame count it decodes to
    """
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.models.modeling_utils import get_parameter_dtype
    from diffusers.utils.torch_utils import randn_tensor

    clip = vae.config.clip_length
    per_clip = vae.tokens_chunk_size
    num_frames = aligned(source_frames, RENDER_GRID)
    clips = math.ceil(num_frames / clip)
    num_latents = clips * per_clip - vae.config.token_drop
    first = read(0, min(clip, source_frames))
    height, width = first.shape[1:3]
    ratio = vae.spatial_compression_ratio
    shape = (
        1,
        vae.config.latent_channels,
        num_latents,
        height // ratio,
        width // ratio,
    )
    noise = randn_tensor(
        shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32
    )
    mean = torch.tensor(pixel_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(pixel_std, device=device).view(1, -1, 1, 1, 1)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    encoder_dtype = get_parameter_dtype(vae.encoder)
    out = torch.empty(shape, dtype=torch.float32)

    for index in range(clips):
        start = index * clip
        # The clip's source frames, the last repeated past the take's end
        lo = min(start, source_frames - 1)
        hi = min(start + clip, source_frames)
        block = first if index == 0 else read(lo, max(hi, lo + 1))
        picks = numpy.clip(numpy.arange(start, start + clip), 0, source_frames - 1)
        pixels = torch.from_numpy(numpy.ascontiguousarray(block[picks - lo]))
        del block
        pixels = pixels.to(device).permute(3, 0, 1, 2)[None]
        pixels = (pixels.to(torch.float32).div(255.0) - mean) / std
        with torch.no_grad():
            moments = vae._encode_clip(pixels.to(encoder_dtype))
        del pixels
        a = index * per_clip
        b = min(a + per_clip, num_latents)
        posterior = DiagonalGaussianDistribution(moments[:, :, : b - a])
        sample = posterior.mean + posterior.std * noise[:, :, a:b].to(moments.device)
        sample = sample.to(torch.float16).float().cpu()
        out[:, :, a:b] = (sample - latents_mean) / latents_std
    return out, num_frames


def _take_reader(source):
    """`(read, source_frames, fps, path, (height, width))` for a take: a file
    read a clip's range at a time (#695), or frames already in memory sliced."""
    from .video_utils import frames_as_array, local_path_or_video

    path, source = local_path_or_video(source)
    if path is not None:
        shape = video_shape(path)
        return (
            lambda start, stop: read_frame_range(path, start, stop),
            shape["frame_count"],
            shape["fps"],
            path,
            (shape["height"], shape["width"]),
        )
    pixels = frames_as_array(source)
    return (
        lambda start, stop: pixels[start:stop],
        len(pixels),
        getattr(source, "fps", None) or getattr(pixels, "fps", None),
        None,
        tuple(pixels.shape[1:3]) if pixels.ndim == 4 else (0, 0),
    )


# The longest take encode_h3_latents will size an allocation for: 30 minutes at
# 24 fps. A container header's frame count is untrusted input - the noise draw
# and the output latent are sized from it - so nothing larger is allocated for.
MAX_TAKE_FRAMES = 30 * 60 * H3_FPS
# A header count this far over duration x fps is a lie, not a rounding error
_DURATION_SLACK = 2


def bounded_take_frames(claimed, path, fps):
    """The frame count to size allocations by, or a ValueError. The header's
    claim is checked against the file's duration x fps (the smaller wins; a
    count more than twice that is refused as absurd) and against
    MAX_TAKE_FRAMES, before anything is allocated."""
    count = claimed
    if path is not None and fps:
        try:
            seconds = media_duration(path)
        except Exception:
            seconds = None
        if seconds and seconds > 0:
            expected = math.ceil(seconds * fps) + 1
            if claimed > _DURATION_SLACK * expected + fps:
                raise ValueError(
                    f"encode_h3_latents: the file's header claims {claimed} "
                    f"frames but its duration ({seconds:.2f}s at {fps:g} fps) "
                    f"holds about {expected} - refusing a count that does not "
                    f"match the duration"
                )
            count = min(claimed, expected)
    if count > MAX_TAKE_FRAMES:
        raise ValueError(
            f"encode_h3_latents: a take of {count} frames is over the longest "
            f"take it encodes ({MAX_TAKE_FRAMES} frames, "
            f"{MAX_TAKE_FRAMES / H3_FPS / 60:g} minutes at {H3_FPS} fps)"
        )
    return count


def _take_problem(source_frames, fps, height, width, multiple):
    """Why a take cannot be encoded for an H3 refine, or None."""
    if source_frames < 1:
        return "encode_h3_latents was given a take with no frames"
    if fps and abs(fps - H3_FPS) > 0.01:
        return (
            f"encode_h3_latents: the take is {fps:g} fps, and H3 refines "
            f"{H3_FPS} fps - re-time it to {H3_FPS} fps first, or the refine "
            f"and its held audio run at the wrong speed"
        )
    if height % multiple or width % multiple:
        return (
            f"encode_h3_latents: the take is {width}x{height}, and an H3 canvas "
            f"is a multiple of {multiple} on both sides"
        )
    return None


def encode_h3_latents(source, device="cpu", model_name=None):
    """Encode a finished take into one MiniMax-H3 video latent, clip by clip.

    The pixels of a take over 15 s do not fit through a `source_video` refine,
    which encodes the whole take as one float32 tensor (about 18 GB at 60 s,
    768p). This reads and encodes one 17-frame VAE clip at a time
    (`encode_take`), and the result is bit-identical to that whole encode.

    Args:
        source: The take - an `asset:`/`output:` file (read a clip's range
            at a time) or an earlier step's video, at 24 fps
        device: Device the VAE runs on
        model_name: The MiniMax-H3 repo whose `vae` subfolder encodes
            (default: MiniMaxAI/MiniMax-H3)

    Returns:
        A dict: `latents`, (1, 24, 5m + 2, h / 16, w / 16), the refine's
        `latents` argument; `num_frames`, the 17m + 5 count the refine renders
        (the take padded up with its last frame); and `source_frames`, the
        take's own count, for `decode_h3_latents`' `num_frames` to trim back to
    """
    read, source_frames, fps, path, (height, width) = _take_reader(source)
    source_frames = bounded_take_frames(source_frames, path, fps)
    # The canvas multiple of the released H3 (VAE 16x times patch 2)
    problem = _take_problem(source_frames, fps, height, width, 32)
    if problem:
        raise ValueError(problem)

    pipeline = _load_decoder(device, model_name or DEFAULT_H3_REPO)

    def encode(total):
        return encode_take(
            pipeline.vae,
            read,
            total,
            pipeline.pixel_mean,
            pipeline.pixel_std,
            pipeline.keyframe_encode_seed,
            pipeline._execution_device,
        )

    try:
        latents, num_frames = encode(source_frames)
    except ShortFrameRange:
        if path is None:
            raise
        # The header over-counted: encode again on a counted total
        source_frames = bounded_take_frames(count_video_frames(path), path, fps)
        latents, num_frames = encode(source_frames)
    logger.info(
        f"Encoded a {source_frames}-frame take ({source_frames / H3_FPS:.2f}s) "
        f"to {latents.shape[2]} latent frames, padded to {num_frames} frames"
    )
    return {
        "latents": latents,
        "num_frames": num_frames,
        "source_frames": source_frames,
    }
