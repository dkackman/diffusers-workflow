"""Follow one face through a clip and crop a steady square around it.

A face-detail pass (a refiner run on just the face) needs the face large and
still: a small face in a wide shot has too few pixels for a model to restore,
and a crop that jitters with every detection hands the refiner a face that
shakes. This task finds the face, follows it, smooths its box, and cuts the
same padded square around it from every frame - plus a record of where each
crop came from, so a later step can paste the refined face back.

Detection is YuNet (OpenCV's own face detector, `cv2.FaceDetectorYN`), run on
the whole frame and on four overlapping tiles enlarged 2x, so a face only a few
dozen pixels wide is still seen; the tiles' detections are merged by
non-maximum suppression. One track is kept: each frame's detection is matched
to it by overlap, confidence and distance, the box is smoothed by an
exponential moving average, and a frame the detector misses holds the last box
at a decaying strength for a few frames. The track starts again at each shot
boundary the clip records, or - for a clip that records none - where the
picture changes abruptly: its colour histogram drops, or the difference from
the previous frame spikes (a cut between two framings of the same picture keeps
its colours, so the histogram alone misses it).

Each frame also carries a strength from 0 to 1: how much a face-detail pass
should change it. A face already large in the frame has the pixels it needs,
so strength falls from 1 at `gate_full` (face width over frame width) to 0 at
`gate_zero`.

`paste_face_track` is the other half: given the clip, the crops after a
face-detail pass, and the track, it puts each repaired crop back over its
square with a feathered round mask scaled by that frame's strength, so the
change fades out at the edges and is absent where the strength is 0.
"""

import logging

import numpy as np
from PIL import Image

from ..events import emit_log, emit_warning
from ..media_types import AudioVideo, JsonRecord
from ..shots import carried_shots
from ..task_problems import (
    FACE_CROP_MODULUS,
    FACE_CROP_MULTIPLE,
    FACE_CROP_REMAINDER,
    check_face_detector_source,
    face_track_problems,
    paste_feather_problems,
)
from ..variable_constraints import aligned
from .video_utils import frames_as_pil_list, load_audio_video

logger = logging.getLogger("dw")

# OpenCV's own upload of YuNet (MIT), read at a pinned revision so a later
# push to the repo cannot change what an existing workflow runs
DEFAULT_DETECTOR_REPO = "opencv/face_detection_yunet"
DEFAULT_DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
DETECTOR_REVISION = "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310"

# Tiling: four corner tiles, each this fraction of the frame on a side (so
# neighbours overlap), enlarged by TILE_SCALE before detection
TILE_FRACTION = 0.6
TILE_SCALE = 2.0
NMS_IOU = 0.4

# Tracking
EMA_ALPHA = 0.35  # weight of the new detection in the smoothed box
HOLD_DECAY = 0.7  # strength kept per frame the detector misses
MAX_GAP = 6  # missed frames before the track counts as lost
MATCH_DISTANCE = 1.5  # centre distance, in face widths, a match may move
HISTOGRAM_CUT = 0.5  # HSV histogram correlation below which frames are a cut
# A cut that keeps the colours (a zoom between two framings of one picture):
# the mean difference of small thumbnails, as a fraction of full scale, must
# reach CUT_MIN_CHANGE and CUT_SPIKE times the median of the frames around it
THUMBNAIL_SIZE = (32, 18)
CUT_MIN_CHANGE = 0.06
CUT_SPIKE = 4.0
CUT_WINDOW = 8  # frames each side whose differences form that median


def _iou(a, b):
    """Overlap of two [x, y, w, h] boxes, as intersection over union."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def nms(detections, threshold=NMS_IOU):
    """Non-maximum suppression over [x, y, w, h, score] rows (`cv2.dnn.NMSBoxes`).

    The tiles overlap and the whole frame is searched too, so one face is
    usually found more than once; the most confident copy is kept.
    """
    import cv2

    rows = [list(map(float, d[:5])) for d in detections]
    if not rows:
        return []
    keep = cv2.dnn.NMSBoxes([r[:4] for r in rows], [r[4] for r in rows], 0.0, threshold)
    return [rows[i] for i in np.asarray(keep, dtype=int).reshape(-1)]


def _tiles(width, height):
    """The regions searched: the whole frame, then four overlapping corners."""
    tw, th = int(round(width * TILE_FRACTION)), int(round(height * TILE_FRACTION))
    yield 0, 0, width, height, 1.0
    for x0 in (0, width - tw):
        for y0 in (0, height - th):
            yield x0, y0, tw, th, TILE_SCALE


def detect_tiled(detect, rgb):
    """Every face in a frame, searched whole and in enlarged tiles, merged.

    Args:
        detect: Callable taking an RGB uint8 array and returning rows of
            [x, y, w, h, score] in that array's pixels
        rgb: The frame, an RGB uint8 array
    """
    import cv2

    height, width = rgb.shape[:2]
    found = []
    for x0, y0, tw, th, scale in _tiles(width, height):
        region = rgb[y0 : y0 + th, x0 : x0 + tw]
        if scale != 1.0:
            region = cv2.resize(
                region,
                (int(round(tw * scale)), int(round(th * scale))),
                interpolation=cv2.INTER_LINEAR,
            )
        for row in detect(np.ascontiguousarray(region)) or []:
            x, y, w, h, score = (float(v) for v in row[:5])
            found.append([x / scale + x0, y / scale + y0, w / scale, h / scale, score])
    return nms(found)


def _hsv_histogram(rgb):
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram).flatten()


def _thumbnail(rgb):
    import cv2

    return cv2.resize(rgb, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA).astype(
        np.float32
    )


def content_cuts(frames_rgb):
    """Cuts nobody recorded: [(frame index, detail)] where the picture jumps.

    A frame is a cut when its HSV histogram correlates with the previous
    frame's below HISTOGRAM_CUT, or when its thumbnail difference from the
    previous frame is a spike: at least CUT_MIN_CHANGE, and CUT_SPIKE times
    the median difference of the CUT_WINDOW frames either side.
    """
    import cv2

    correlations, changes = [None], [0.0]
    previous = None
    for rgb in frames_rgb:
        current = (_hsv_histogram(rgb), _thumbnail(rgb))
        if previous is not None:
            correlations.append(
                float(cv2.compareHist(previous[0], current[0], cv2.HISTCMP_CORREL))
            )
            changes.append(float(np.mean(np.abs(current[1] - previous[1]))) / 255)
        previous = current

    cuts = []
    for index in range(1, len(changes)):
        change = changes[index]
        around = (
            changes[max(1, index - CUT_WINDOW) : index]
            + changes[index + 1 : index + 1 + CUT_WINDOW]
        )
        baseline = float(np.median(around)) if around else 0.0
        spike = change >= CUT_MIN_CHANGE and change >= CUT_SPIKE * baseline
        if correlations[index] < HISTOGRAM_CUT or spike:
            cuts.append(
                (
                    index,
                    {
                        "histogram_correlation": round(correlations[index], 4),
                        "frame_change": round(change, 4),
                    },
                )
            )
    return cuts


def shot_starts(shots):
    """{frame: shot name} for each recorded shot that starts after frame 0."""
    starts = {}
    for shot in shots or []:
        start = shot.get("start_frame")
        if isinstance(start, int) and start > 0:
            starts[start] = shot.get("name")
    return starts


def gate_strength(face_width, frame_width, gate_full, gate_zero):
    """1 for a small face, 0 for a large one, linear between the two gates."""
    ratio = face_width / frame_width
    if ratio <= gate_full:
        return 1.0
    if ratio >= gate_zero:
        return 0.0
    return (gate_zero - ratio) / (gate_zero - gate_full)


def _centre(box):
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _match(box, detections):
    """The detection that continues a track, or None if none is close enough."""
    best, best_score = None, 0.0
    cx, cy = _centre(box)
    for detection in detections:
        dx, dy = _centre(detection)
        distance = np.hypot(dx - cx, dy - cy) / max(box[2], 1.0)
        if distance > MATCH_DISTANCE:
            continue
        score = (
            _iou(box, detection)
            + 0.5 * detection[4]
            + 0.5 * (1.0 - distance / MATCH_DISTANCE)
        )
        if score > best_score:
            best, best_score = detection, score
    return best


def track_faces(detections_per_frame, frame_width, resets_at, gate_full, gate_zero):
    """Follow one face through per-frame detections.

    Args:
        detections_per_frame: For each frame, the [x, y, w, h, score] rows
            already above the confidence floor
        frame_width: Source width in pixels, for the gate
        resets_at: {frame: (reason, detail)} - where the track must start again
        gate_full, gate_zero: The strength gate (gate_strength)
    Returns:
        (frames, resets): a dict per frame with "box" (smoothed, or None),
        "strength" and "state" ("tracked", "held", "none"), and the list of
        resets that happened, each {frame, reason, detail}
    """
    frames, resets = [], []
    box, hold, missed = None, 1.0, 0
    for index, detections in enumerate(detections_per_frame):
        if index in resets_at:
            reason, detail = resets_at[index]
            resets.append({"frame": index, "reason": reason, "detail": detail})
            box = None

        match = None
        if box is None:
            if detections:
                match = max(detections, key=lambda d: d[4])
        else:
            match = _match(box, detections)

        if match is not None:
            if box is None:
                box = list(match[:4])
            else:
                box = [
                    (1 - EMA_ALPHA) * old + EMA_ALPHA * new
                    for old, new in zip(box, match[:4])
                ]
            hold, missed, state = 1.0, 0, "tracked"
        elif box is not None:
            missed += 1
            if missed > MAX_GAP:
                resets.append({"frame": index, "reason": "lost", "detail": None})
                box, state = None, "none"
            else:
                hold *= HOLD_DECAY
                state = "held"
        else:
            state = "none"

        if box is None:
            frames.append({"box": None, "strength": 0.0, "state": state})
        else:
            strength = hold * gate_strength(box[2], frame_width, gate_full, gate_zero)
            frames.append(
                {
                    "box": [round(v, 2) for v in box],
                    "strength": round(strength, 4),
                    "state": state,
                }
            )
    return frames, resets


def crop_square(box, padding, width, height):
    """The integer square [x, y, side, side] padded around a face box."""
    if box is None:
        side = min(width, height)
        return [(width - side) // 2, (height - side) // 2, side, side]
    side = max(box[2], box[3]) * (1 + 2 * padding)
    side = max(int(round(side)), 1)
    cx, cy = _centre(box)
    return [int(round(cx - side / 2)), int(round(cy - side / 2)), side, side]


def _cut(rgb, square, crop_size):
    """Cut a square from a frame, replicating the border past its edges."""
    import cv2

    x, y, side, _ = square
    height, width = rgb.shape[:2]
    left, top = max(0, -x), max(0, -y)
    right, bottom = max(0, x + side - width), max(0, y + side - height)
    if left or top or right or bottom:
        rgb = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_REPLICATE)
        x, y = x + left, y + top
    region = rgb[y : y + side, x : x + side]
    return Image.fromarray(region).resize((crop_size, crop_size), Image.LANCZOS)


def padding_to_grid(count, modulus, remainder):
    """(before, after): frames to add so count becomes the next
    modulus * n + remainder, split as evenly as it goes. The grid count
    comes from `variable_constraints.aligned`, the one owner of grid
    arithmetic, as `plan_cuts`' render lengths do."""
    pad = aligned(count, {"modulus": modulus, "remainder": remainder}) - count
    return pad // 2, pad - pad // 2


def pad_frames(crops, before, after):
    """Warm-up and cool-down frames, reflected off each end of the crops."""
    indices = np.pad(np.arange(len(crops)), (before, after), mode="reflect")
    return [crops[i] for i in indices]


def _detector(repo, filename, min_confidence, device):
    """A callable running YuNet on an RGB array, loaded once per device."""
    import cv2

    from .. import get_device_type
    from .model_cache import cached_model

    check_face_detector_source(repo, filename)
    revision = DETECTOR_REVISION if repo == DEFAULT_DETECTOR_REPO else None

    use_cuda = (
        get_device_type(device) == "cuda"
        and hasattr(cv2, "cuda")
        and cv2.cuda.getCudaEnabledDeviceCount() > 0
    )

    def load():
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo, filename=filename, revision=revision)
        backend, target = (
            (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
            if use_cuda
            else (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        )
        return cv2.FaceDetectorYN.create(
            path, "", (320, 320), min_confidence, 0.3, 5000, backend, target
        )

    model = cached_model(
        ("crop_face_track", repo, filename, revision, min_confidence, use_cuda), load
    )

    def detect(rgb):
        height, width = rgb.shape[:2]
        model.setInputSize((width, height))
        _, faces = model.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return [] if faces is None else [row[:4].tolist() + [row[-1]] for row in faces]

    return detect


def crop_face_track(
    clip,
    crop_size=512,
    padding=0.6,
    gate_full=0.06,
    gate_zero=0.12,
    min_confidence=0.6,
    modulus=FACE_CROP_MODULUS,
    remainder=FACE_CROP_REMAINDER,
    multiple=FACE_CROP_MULTIPLE,
    detector_repo=DEFAULT_DETECTOR_REPO,
    detector_file=DEFAULT_DETECTOR_FILE,
    device=None,
):
    """Task command: crop a steady square around the one face a clip follows.

    Args:
        clip: The video - frames, an AudioVideo, or the path or URL of a video
            file. Named "clip" rather than "video" so the engine hands it over
            as read, with the shots it records
        crop_size: Side of every crop in pixels, a multiple of `multiple`
        padding: Space added around the face on each side, as a fraction of
            its size - 0.6 gives a square 2.2 face widths across
        gate_full: Face width over frame width at or below which strength is 1
        gate_zero: Face width over frame width at or above which strength is 0
        min_confidence: Detections below this score are ignored
        modulus: The crop count is padded to modulus * n + remainder frames,
            the frame grid of the model the crops feed (8 with remainder 1 is
            LTX's 8n+1)
        remainder: The remainder of the padded count modulo `modulus`, from 0
            to modulus - 1
        multiple: crop_size must be a multiple of this, the model's frame-size
            step (32 for LTX)
        detector_repo: Hugging Face repo holding the YuNet weights
        detector_file: The .onnx file in that repo
        device: Where detection runs; CUDA only when OpenCV was built with it
    Returns:
        {"crops": AudioVideo of crop_size squares, padded to modulus * n + remainder
         frames (8n+1 by default),
         "track": the record of every frame's box, crop and strength}
    """
    problems = face_track_problems(
        crop_size=crop_size,
        padding=padding,
        gate_full=gate_full,
        gate_zero=gate_zero,
        min_confidence=min_confidence,
        modulus=modulus,
        remainder=remainder,
        multiple=multiple,
    )
    if problems:
        raise ValueError("; ".join(message for _, message in problems))
    crop_size = int(crop_size)
    modulus, remainder = int(modulus), int(remainder)

    detector = _detector(detector_repo, detector_file, min_confidence, device)

    if isinstance(clip, str):
        clip = load_audio_video(clip)
    frames = frames_as_pil_list(clip)
    if not frames:
        raise ValueError("crop_face_track needs a clip with at least one frame")
    fps = getattr(clip, "fps", None)
    rgb = [np.asarray(frame.convert("RGB")) for frame in frames]
    height, width = rgb[0].shape[:2]

    shots = getattr(clip, "shots", None)
    starts = shot_starts(shots)
    if starts:
        resets_at = {frame: ("shot", name) for frame, name in starts.items()}
    else:
        resets_at = {frame: ("cut", detail) for frame, detail in content_cuts(rgb)}

    detections = [
        [d for d in detect_tiled(detector, frame) if d[4] >= min_confidence]
        for frame in rgb
    ]
    track, resets = track_faces(detections, width, resets_at, gate_full, gate_zero)
    face_found = any(entry["box"] is not None for entry in track)

    # A frame without a box borrows the nearest one, at strength 0, so the
    # crop does not jump to the frame centre and back across a short miss
    boxes = [entry["box"] for entry in track]
    known = [i for i, b in enumerate(boxes) if b is not None]
    crops = []
    for index, entry in enumerate(track):
        box = entry["box"]
        if box is None and known:
            box = boxes[min(known, key=lambda k: abs(k - index))]
        square = crop_square(box, padding, width, height)
        entry["crop"] = square
        crops.append(_cut(rgb[index], square, crop_size))

    before, after = padding_to_grid(len(crops), modulus, remainder)
    crops = pad_frames(crops, before, after)

    record = JsonRecord(
        {
            "source": {
                "width": width,
                "height": height,
                "frames": len(frames),
                "fps": fps,
            },
            "crop_size": crop_size,
            "padding": padding,
            "gate_full": gate_full,
            "gate_zero": gate_zero,
            "min_confidence": min_confidence,
            "detector": {"repo": detector_repo, "file": detector_file},
            "pad_before": before,
            "pad_after": after,
            "crop_frames": len(crops),
            "face_found": face_found,
            "frames": track,
            "resets": resets,
        }
    )
    if not face_found:
        record["message"] = (
            f"No face was found in {len(frames)} frames (min_confidence "
            f"{min_confidence}); every strength is 0 and each crop is the "
            f"centre square of the frame"
        )
        emit_warning(
            f"crop_face_track: {record['message']}",
            kind="no_face_found",
            command="crop_face_track",
            frames=len(frames),
        )
    tracked = sum(entry["state"] == "tracked" for entry in track)
    emit_log(
        f"crop_face_track: face in {tracked} of {len(frames)} frames, "
        f"{len(resets)} reset(s), {len(crops)} crops of {crop_size}px "
        f"({before} before, {after} after for {modulus}n+{remainder})"
    )
    return {"crops": AudioVideo(crops, None, None, fps=fps), "track": record}


def feather_mask(side, feather):
    """The alpha a pasted square is blended with: radial, 1 at the centre.

    The mask is the circle inscribed in the square. Its outer `feather`
    fraction of the radius falls from 1 to 0 along a smoothstep, so the
    square's corners - and everything past the circle - keep the source.
    """
    radius = side / 2
    coords = np.arange(side, dtype=np.float32) + 0.5 - radius
    distance = np.hypot(coords[None, :], coords[:, None]) / radius
    if feather <= 0:
        return (distance < 1).astype(np.float32)
    t = np.clip((1 - distance) / feather, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def paste_crop(source_rgb, crop, square, strength, feather, color_match):
    """One frame with a repaired crop blended back over its square.

    Args:
        source_rgb: The source frame, an RGB uint8 array
        crop: The repaired crop, a PIL image of any size
        square: [x, y, side, side] the crop was cut from; it may run past the
            frame's edges, where the crop replicated the border
        strength: 0 to 1, scaling the whole mask
        feather: Fraction of the mask's radius that fades (feather_mask)
        color_match: Shift the crop's mean colour, inside the mask, onto the
            source's before blending
    """
    x, y, side, _ = (int(v) for v in square)
    height, width = source_rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + side), min(height, y + side)
    if x1 <= x0 or y1 <= y0:
        return source_rgb

    patch = np.asarray(
        crop.convert("RGB").resize((side, side), Image.LANCZOS), dtype=np.float32
    )
    mask = feather_mask(side, feather)
    inside = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    patch, mask = patch[inside], mask[inside]
    region = source_rgb[y0:y1, x0:x1].astype(np.float32)

    if color_match:
        weight = float(mask.sum())
        if weight > 0:
            patch = patch + ((region - patch) * mask[..., None]).sum((0, 1)) / weight

    alpha = (mask * strength)[..., None]
    out = source_rgb.copy()
    out[y0:y1, x0:x1] = np.clip(
        np.rint(region * (1 - alpha) + patch * alpha), 0, 255
    ).astype(np.uint8)
    return out


def _read_track(track):
    """The track record, from the dict a step handed on or its saved .json."""
    if isinstance(track, str):
        from ..locations import load_json_record

        track = load_json_record(track, None, "a track argument")
    if not (
        isinstance(track, dict)
        and isinstance(track.get("source"), dict)
        and isinstance(track.get("frames"), list)
        and "crop_frames" in track
        and "pad_before" in track
    ):
        raise ValueError(
            "paste_face_track needs 'track' as the track record crop_face_track "
            "returned (previous_result:<step>.track, or its saved .json)"
        )
    return track


def paste_face_track(clip, repaired, track, feather=0.3, color_match=True):
    """Task command: put repaired face crops back into the clip they came from.

    Args:
        clip: The source video crop_face_track read - frames, an AudioVideo,
            or the path or URL of a video file. Its audio, frame rate and
            shots come through unchanged
        repaired: The crops after a face-detail pass, still padded to the
            padded count crop_face_track produced; any frame size
        track: The `track` record crop_face_track returned, or its .json file
        feather: Fraction of the paste's radius that fades out, 0 to 1
        color_match: Match each crop's mean colour to the source's inside the
            mask before blending
    Returns:
        An AudioVideo: the source frames with the face pasted back where the
        strength is above 0, and the source frame itself where it is 0
    """
    problems = paste_feather_problems(feather)
    if problems:
        raise ValueError("; ".join(message for _, message in problems))
    feather = float(feather)
    track = _read_track(track)

    if isinstance(clip, str):
        clip = load_audio_video(clip)
    if isinstance(repaired, str):
        repaired = load_audio_video(repaired)
    frames = frames_as_pil_list(clip)
    crops = frames_as_pil_list(repaired)
    if not frames:
        raise ValueError("paste_face_track needs a clip with at least one frame")

    source = track["source"]
    width, height = frames[0].size
    if source.get("frames") != len(frames) or len(track["frames"]) != len(frames):
        raise ValueError(
            f"paste_face_track: the track was made from a clip of "
            f"{source.get('frames')} frames but this clip has {len(frames)} - "
            f"pass the clip crop_face_track tracked"
        )
    if (source.get("width"), source.get("height")) != (width, height):
        raise ValueError(
            f"paste_face_track: the track was made from a "
            f"{source.get('width')}x{source.get('height')} clip but this clip "
            f"is {width}x{height} - pass the clip crop_face_track tracked"
        )
    needed = int(track["crop_frames"])
    if len(crops) < needed:
        raise ValueError(
            f"paste_face_track: 'repaired' has {len(crops)} frames but the "
            f"track's padded crops number {needed} (pad_before "
            f"{track['pad_before']}, {len(frames)} frames, pad_after "
            f"{track.get('pad_after')}) - a repair pass must keep every crop"
        )

    before = int(track["pad_before"])
    out, pasted = [], 0
    for index, (frame, entry) in enumerate(zip(frames, track["frames"])):
        strength = float(entry.get("strength") or 0.0)
        square = entry.get("crop")
        if strength <= 0 or not square:
            out.append(frame)
            continue
        rgb = np.asarray(frame.convert("RGB"))
        out.append(
            Image.fromarray(
                paste_crop(
                    rgb,
                    crops[before + index],
                    square,
                    min(strength, 1.0),
                    feather,
                    color_match,
                )
            )
        )
        pasted += 1

    emit_log(
        f"paste_face_track: pasted into {pasted} of {len(frames)} frames "
        f"(feather {feather}, color_match {bool(color_match)})"
    )
    # Same frames, one for one, so the clip's sound and shots still hold
    return AudioVideo(
        out,
        getattr(clip, "audio", None),
        getattr(clip, "sample_rate", None),
        fps=getattr(clip, "fps", None),
        shots=carried_shots(clip),
    )
