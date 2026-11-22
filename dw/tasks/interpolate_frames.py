"""
Video frame interpolation via RIFE (Real-Time Intermediate Flow Estimation).

Takes a list of video frames and generates intermediate frames to increase
frame rate. Supports 2x, 4x, and 8x multipliers.

`refill: "held"` (#1527) is the other mode: it keeps the frame count and
rate, and replaces only the frames analyze_held_frames' measure flags as
held - each run of them interpolated between the distinct frames either side
of it - so a take that stutters plays smoothly, with its audio still in sync.

Model weights are downloaded from HuggingFace Hub on first use.
"""

import logging
import torch

from ..media_types import AudioVideo
from ..shots import rescaled_shots
from .tensor_image import pil_to_float_tensor as _pil_to_tensor, float_tensor_to_pil
from .video_utils import frames_as_pil_list
from .weights import argument_or_default, file_entries

logger = logging.getLogger("dw")

_VALID_MULTIPLIERS = {2, 4, 8}

# The values `refill` takes: "held" refills the frames the held-frame measure
# flags (dw/tasks/assess.py held_measure)
REFILL_MODES = ("held",)


def exclusive_argument_errors(arguments):
    """[(argument, message)] for an interpolate_frames step setting both
    `refill` and `multiplier` - the static check validate_workflow runs, and
    the one interpolate_frames repeats at run time. A refill keeps the
    frame count; a multiplier multiplies it, so the two can't both apply."""
    if arguments.get("refill") is not None and arguments.get("multiplier") is not None:
        return [
            (
                "refill",
                "'refill' and 'multiplier' can't both be set: refill keeps the "
                "frame count and rate, multiplier multiplies them. Set one",
            )
        ]
    return []


def interpolate_frames(video, device="cpu", **kwargs):
    """Interpolate between video frames using RIFE.

    Args:
        video: The frames - a list of PIL images, a frame array or tensor, or
            an AudioVideo (a concat or dissolve step's output). A multiplier
            drops its soundtrack (pair_audio puts it back); a refill keeps it
        device: Target device ("cuda", "mps", "cpu")
        **kwargs:
            multiplier: Frame count multiplier - 2, 4, or 8 (default: 2,
                when `refill` is unset)
            refill: "held" - keep the frame count, fps, audio and shots, and
                replace only the held frames (`refill_held`). Exclusive with
                `multiplier`
            model_name: HuggingFace repo with RIFE v4.13 weights (default: auto)
            filename: Weights filename within the repo (default: auto)

    Returns:
        One AudioVideo - one artifact, where a bare frame list would fan a
        later step out over every frame. A multiplier's holds the
        interpolated frames and no audio; a refill's holds the source's
        frame count, audio, fps and shots, with `held_before`/`held_after`.
    """
    errors = exclusive_argument_errors(kwargs)
    if errors:
        raise ValueError(errors[0][1])
    refill = kwargs.get("refill")
    model_name = kwargs.get("model_name", None)
    filename = kwargs.get("filename", None)
    if refill is not None:
        if refill not in REFILL_MODES:
            raise ValueError(
                f"refill must be one of {list(REFILL_MODES)}, got {refill!r}"
            )
        return refill_held(
            video, lambda: _load_rife_model(device, model_name, filename)
        )

    multiplier = int(kwargs.get("multiplier") or 2)

    if multiplier not in _VALID_MULTIPLIERS:
        raise ValueError(
            f"multiplier must be one of {sorted(_VALID_MULTIPLIERS)}, got {multiplier}"
        )

    # An AudioVideo or a frame array unwraps to its frames; a PIL list passes
    # through by identity. The soundtrack does not survive - the frame count
    # changes, so pair_audio is how it comes back
    source_fps = getattr(video, "fps", None)
    source_shots = getattr(video, "shots", None)
    video = frames_as_pil_list(video)
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
    # Multiplied, not carried: interpolation adds frames between the ones it
    # was given, so playing them back at the source rate would run the clip
    # `multiplier` times long. The rate that keeps the source's duration is
    # the source's times the multiplier (#84)
    # The shots stretch with the frames between them; the track is gone, so
    # their sample side goes with it
    return AudioVideo(
        frames,
        None,
        None,
        fps=source_fps * multiplier if source_fps else None,
        shots=rescaled_shots(source_shots, multiplier),
    )


def held_runs(d, held, cuts, frame_count, epsilon=None):
    """The held frames grouped into runs: ([(first, last)] to refill,
    [(first, last)] left held).

    The measure flags a hold only beside a moving frame, so a frame shown
    four times flags its first and last repeat but not the middle one: each
    flagged frame grows over the consecutive frames still under `epsilon`
    either side, and the whole still stretch is the run. Its anchors are
    a = first - 1 and b = last + 1. A run is left held when b is past the
    clip's end or a cut lands in (a, b] - never interpolated across a cut.
    """
    from .assess import HELD_EPSILON

    epsilon = HELD_EPSILON if epsilon is None else epsilon
    runs, seen = [], set()
    for frame in sorted(held):
        if frame in seen:
            continue
        first = frame
        while first - 1 >= 1 and d[first - 1] < epsilon:
            first -= 1
        last = frame
        while last + 1 < frame_count and d[last + 1] < epsilon:
            last += 1
        seen.update(range(first, last + 1))
        runs.append((first, last))
    refill, left = [], []
    for first, last in runs:
        a, b = first - 1, last + 1
        across_cut = any(a < cut <= b for cut in cuts)
        (left if b >= frame_count or across_cut else refill).append((first, last))
    return refill, left


def refill_held(video, load_model):
    """`refill: "held"`: the video with each held run replaced by RIFE
    between its anchors, every other frame passed through as given.

    Run frame first+j of a k-frame run becomes model(a, b, (j+1)/(k+1)). A
    take with no holds comes back with its frames as given and `load_model`
    never called. The audio, sample rate, fps and shots are the source's;
    the result carries `held_before` (the measure on the source) and
    `held_after` (on the refilled frames), and both go out as a `log` event.
    """
    from ..events import emit_log
    from .assess import frames_media, held_measure

    source_fps = getattr(video, "fps", None)
    source_shots = getattr(video, "shots", None)
    audio = getattr(video, "audio", None)
    sample_rate = getattr(video, "sample_rate", None)
    frames = frames_as_pil_list(video)
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames to refill, got {len(frames)}")

    d, held, cuts = held_measure(frames_media(frames))
    refill, left = held_runs(d, held, cuts, len(frames))
    held_before = len(held)

    out = list(frames)
    if refill:
        model = load_model()
        for first, last in refill:
            a, b = frames[first - 1], frames[last + 1]
            k = last - first + 1
            for j in range(k):
                out[first + j] = model(a, b, (j + 1) / (k + 1))
        _, held_after_frames, _ = held_measure(frames_media(out))
        held_after = len(held_after_frames)
    else:
        held_after = held_before

    refilled = sum(last - first + 1 for first, last in refill)
    left_frames = sum(last - first + 1 for first, last in left)
    emit_log(
        f"interpolate_frames: refilled {refilled} held frame(s) in "
        f"{len(refill)} run(s), {left_frames} left held at a cut or the "
        f"clip's end; held frames {held_before} -> {held_after}",
        command="interpolate_frames",
        held_before=held_before,
        held_after=held_after,
        refilled_frames=refilled,
        refilled_runs=len(refill),
        left_held_frames=left_frames,
    )
    result = AudioVideo(out, audio, sample_rate, fps=source_fps, shots=source_shots)
    result.held_before = held_before
    result.held_after = held_after
    return result


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


def rife_source(arguments):
    """The (repo, file) an interpolate_frames step loads RIFE from -
    `model_name` and `filename` over the defaults."""
    return (
        argument_or_default(arguments, "model_name", _DEFAULT_RIFE_REPO),
        argument_or_default(arguments, "filename", _DEFAULT_RIFE_FILENAME),
    )


def required_weights(arguments):
    """The RIFE weight file, for the plan (dw/tasks/weights.py)."""
    return file_entries(*rife_source(arguments))


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


def _make_flow_context(ph, pw, device, timestep=0.5):
    """Build the warp grid, flow divisors and timestep plane for one padded
    resolution. `timestep` is where between the two frames the result
    lands: 0.5 is the midpoint `multiplier` uses."""
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
    timestep = torch.full((1, 1, ph, pw), float(timestep), dtype=torch.float32, device=device)
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
        Callable that takes (frame1: PIL.Image, frame2: PIL.Image,
        timestep=0.5) -> PIL.Image
    """
    from .rife_model import IFNet
    from huggingface_hub import hf_hub_download
    from .model_cache import cached_model

    from ..security import validate_weight_name

    repo_id, weights_file = rife_source(
        {"model_name": model_name, "filename": filename}
    )
    # Joined onto the cache directory by hf_hub_download, so a name that is
    # absolute or climbs is a path on this machine (GHSA-ch56-2p9f-95cg)
    validate_weight_name(weights_file, what="filename")

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
    """Build the (img1, img2, timestep=0.5) -> frame callable for a loaded
    RIFE net.

    All frames of a video share one padded resolution, so the flow-warp
    context (tenFlow_div, backwarp grid, timestep) is computed once per
    (padded_h, padded_w, timestep) and reused for every pair instead of being rebuilt
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

    def get_flow_context(ph, pw, timestep):
        key = (ph, pw, timestep)
        ctx = flow_context_cache.get(key)
        if ctx is None:
            ctx = _make_flow_context(ph, pw, device, timestep)
            flow_context_cache[key] = ctx
        return ctx

    def inference(img1, img2, timestep=0.5):
        """Interpolate a single frame between two input frames, at
        `timestep` between them (0.5, the midpoint, by default)."""
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

        tenFlow_div, backwarp_tenGrid, timestep_plane = get_flow_context(
            ph, pw, timestep
        )

        with torch.inference_mode():
            _, _, merged = net(
                t1_padded,
                t2_padded,
                timestep_plane,
                scale_list,
                tenFlow_div,
                backwarp_tenGrid,
            )

        mid = merged[3][:, :, :h, :w]
        return float_tensor_to_pil(mid)

    return inference
