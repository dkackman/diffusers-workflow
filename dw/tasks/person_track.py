"""Follow each person through a clip as a mask of their own (#1194, stage C).

`segment` masks everything its prompt matches in one picture, so on a
two-person shot 'person' is both of them, and a regional edit meant for one
repaints the other. `track_people` separates them:

1. The clip is split into cuts: frame 0, every shot the clip records
   (`dw/shots.py`), and every cut `detect_cuts`' detector finds
   (`scene_cuts.content_cuts`). A track never crosses a cut, where the
   picture - and who is where - starts over.
2. On each cut's first frame Grounding-DINO finds the prompt's boxes
   (`segment.detect_boxes`, the detector's one owner). Overlapping boxes of
   one person (IoU above BOX_OVERLAP) keep the higher score
   (`torchvision.ops.nms`), and the `max_people` largest are kept.
3. SAM2's video predictor (transformers' `Sam2VideoModel` on the same
   Apache-2.0 `facebook/sam2-hiera-large` weights `segment` loads - the
   Apache claim is this default Grounding-DINO path's) takes
   those boxes as prompts and propagates each person's mask through the
   cut. Where two masks claim a pixel, the person SAM2 scores higher there
   keeps it, so the masks never overlap.
4. In each cut the people are numbered by their mask's area on its first
   frame, largest first: `person-1` is the biggest figure. A cut with fewer
   people leaves the higher numbers black there.

The result is one black-and-white video per person, white where that person
is, with the source's frame count and size, plus `contact_sheet` - each cut's
first frame with every person outlined and numbered as the masks are named -
and `people`, the record of what was found. Numbering is per cut, so on a
multi-cut clip `person-1` is the largest figure in each cut, not one
identity across them; the sheet is how an agent checks.

`backend="sam3"` (#1397, opt-in) replaces steps 2-3 with SAM 3's video model
(`Sam3VideoModel` on the gated `facebook/sam3` weights, under the SAM
License, not Apache - docs/ATTRIBUTIONS.md). It takes the prompt as text and
detects on every frame of the cut, so someone who enters after its first
frame - a late entrant, or a figure coming in through a dissolve - gets a
mask of their own. SAM 3 de-duplicates its own detections, so no NMS runs.
People on the cut's first frame are numbered by area as above; late
entrants follow, in order of first appearance, and `max_people` keeps that
order's first. Each person's `first_frame` in `people` is where, within the
cut, they first appear. The Sam3 classes are resolved only on this path, and
a transformers older than SAM3_MIN_TRANSFORMERS refuses it (`sam3_refusal`).
"""

import logging

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..events import emit_warning
from ..media_types import AudioVideo, JsonRecord
from ..task_domains import NON_NEGATIVE, POSITIVE, check_arguments, whole_number
from .model_cache import cached_model
from .registry import register_command
from .scene_cuts import content_cuts
from .segment import TransformersClass, detect_boxes, model_sources
from .weights import argument_or_default
from .video_utils import VideoFileReference, frames_as_pil_list, load_audio_video
from .weights import repo_entries

logger = logging.getLogger("dw")

COMMAND = "track_people"
PERSON_PREFIX = "person"
SHOT_PREFIX = "shot"

BOX_OVERLAP = 0.5  # IoU above which two detections are one person
SHEET_TILE_WIDTH = 640  # each cut's first frame on the contact sheet
SHEET_COLUMNS = 3
SHEET_FONT_SIZE = 20
# Outline colours by person number, cycling past the sixth
SHEET_COLOURS = [
    (255, 64, 64),
    (64, 160, 255),
    (64, 220, 96),
    (255, 200, 32),
    (220, 96, 255),
    (32, 220, 220),
]

# The SAM2 video predictor, on the weights segment's DEFAULT_SAM_MODEL names.
# Resolved on first use (segment's shim): the plan's required_weights lookup
# imports this module on every validate. Module-level so tests patch them.
Sam2VideoModel = TransformersClass("Sam2VideoModel")
Sam2VideoProcessor = TransformersClass("Sam2VideoProcessor")
# SAM 3's video model, for backend="sam3" only - resolved on first use like
# the SAM2 shims, so the DINO path and every validate never touch them
Sam3VideoModel = TransformersClass("Sam3VideoModel")
Sam3VideoProcessor = TransformersClass("Sam3VideoProcessor")

BACKEND_DINO = "grounding-dino"
BACKEND_SAM3 = "sam3"
BACKENDS = (BACKEND_DINO, BACKEND_SAM3)
DEFAULT_SAM3_MODEL = "facebook/sam3"
# The transformers the sam3 path is refused below (#1397 Q5: no floor raise)
SAM3_MIN_TRANSFORMERS = "5.21"
# Each backend's score floor when the step gives none: the scales differ -
# DINO's 0.3 (#1375) and SAM 3's own score_threshold_detection
DEFAULT_THRESHOLDS = {BACKEND_DINO: 0.3, BACKEND_SAM3: 0.5}
# A logit no real one reaches: an object absent from a frame
ABSENT_LOGIT = -1.0e4


def required_weights(arguments):
    """The repos a track_people step loads, for the plan (dw/tasks/weights.py):
    the detector and SAM2, or with backend "sam3" SAM 3's one repo - none
    when a literal max_people of 0 tracks nobody and loads nothing."""
    max_people = arguments.get("max_people")
    if isinstance(max_people, (int, float)) and not isinstance(max_people, bool):
        if max_people == 0:
            return []
    if arguments.get("backend") == BACKEND_SAM3:
        return repo_entries(
            argument_or_default(arguments, "sam3_model_name", DEFAULT_SAM3_MODEL)
        )
    return repo_entries(*model_sources(arguments))


def sam3_refusal():
    """Why this install cannot run backend "sam3", or None when it can: a
    transformers older than SAM3_MIN_TRANSFORMERS, or one without the Sam3
    video classes."""
    try:
        import transformers
        from packaging.version import InvalidVersion, Version
    except ImportError as error:
        return f"transformers could not be imported ({error})"
    installed = getattr(transformers, "__version__", "unknown")
    try:
        too_old = Version(installed) < Version(SAM3_MIN_TRANSFORMERS)
    except InvalidVersion:
        too_old = True
    if not too_old:
        try:
            getattr(transformers, "Sam3VideoModel")
            getattr(transformers, "Sam3VideoProcessor")
            return None
        except (AttributeError, ImportError):
            pass
    return (
        f"{COMMAND}: backend {BACKEND_SAM3!r} needs transformers >= "
        f"{SAM3_MIN_TRANSFORMERS} (Sam3VideoModel and its session); this "
        f"install has transformers {installed}. Upgrade transformers, or use "
        f"backend {BACKEND_DINO!r}."
    )


# ------------------------------------------------------------------ geometry


def box_area(box):
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def distinct_boxes(boxes, scores):
    """[(box, score)], highest score first, with any box overlapping a kept
    one by more than BOX_OVERLAP dropped - DINO often boxes one person twice.
    torchvision's NMS does the suppression."""
    if len(boxes) == 0:
        return []
    from torchvision.ops import nms

    keep = nms(
        torch.as_tensor(boxes, dtype=torch.float32),
        torch.as_tensor(scores, dtype=torch.float32),
        iou_threshold=BOX_OVERLAP,
    )
    return [(boxes[i], scores[i]) for i in keep.tolist()]


def cut_starts(total, shots, frames_rgb):
    """Sorted first frames of the clip's cuts: 0, each recorded shot start,
    and each content cut, each inside the clip."""
    starts = {0}
    for shot in shots or []:
        start = shot.get("start_frame")
        if isinstance(start, int) and not isinstance(start, bool):
            starts.add(start)
    starts |= {frame for frame, _ in content_cuts(frames_rgb)}
    return sorted(frame for frame in starts if 0 <= frame < total)


# ------------------------------------------------------------------- models


def _load_sam_video(sam_model_name, device):
    def load():
        logger.info(f"Loading SAM2 video predictor from {sam_model_name}")
        processor = Sam2VideoProcessor.from_pretrained(sam_model_name)
        model = Sam2VideoModel.from_pretrained(sam_model_name).to(device)
        return processor, model

    return cached_model(("track_people_sam2_video", sam_model_name, str(device)), load)


def _load_sam3_video(sam3_model_name, device):
    def load():
        logger.info(f"Loading SAM 3 video model from {sam3_model_name}")
        processor = Sam3VideoProcessor.from_pretrained(sam3_model_name)
        # Loaded on the CPU and moved in one call, as SAM2 is: the
        # accommodation hf_pipeline_placement (model_cache.py) names for MPS
        model = Sam3VideoModel.from_pretrained(sam3_model_name).to(device)
        return processor, model

    return cached_model(("track_people_sam3_video", sam3_model_name, str(device)), load)


def detect_people(image, prompt, threshold, model_name, device):
    """([box xyxy], [score]) Grounding-DINO finds for the prompt on image."""
    found = detect_boxes(image, prompt, threshold, model_name, device)
    boxes = [[float(v) for v in box] for box in found["boxes"].cpu().tolist()]
    scores = [float(s) for s in found["scores"].cpu().tolist()]
    return boxes, scores


def propagate_masks(frames, boxes, sam_model_name, device):
    """[frame][person] boolean masks for one cut: SAM2's video predictor
    prompted with each box on the cut's first frame, never overlapping."""
    processor, model = _load_sam_video(sam_model_name, device)
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        video_storage_device="cpu",
        inference_state_device="cpu",
        dtype=torch.float32,
    )
    object_ids = list(range(1, len(boxes) + 1))
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        obj_ids=list(object_ids),  # the processor consumes the list it is given
        input_boxes=[[list(box) for box in boxes]],
    )
    width, height = frames[0].size
    masks = [None] * len(frames)
    with torch.inference_mode():
        model(inference_session=session, frame_idx=0)
        for output in model.propagate_in_video_iterator(session):
            logits = processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[height, width]],
                binarize=False,
            )[0]
            logits = logits.reshape(logits.shape[0], height, width).float().cpu()
            order = [
                object_ids.index(i) for i in getattr(output, "object_ids", object_ids)
            ]
            masks[output.frame_idx] = _exclusive_masks(logits.numpy(), order)
    for index, frame_masks in enumerate(masks):
        if frame_masks is None:  # a frame the predictor did not answer
            masks[index] = [np.zeros((height, width), bool) for _ in boxes]
    return masks


def sam3_track(frames, prompt, threshold, sam3_model_name, device):
    """SAM 3 on one cut: ({object id: [frame] low-res logits or None},
    {object id: score}) - every object the model reports, text-prompted and
    propagated through the cut. The prompt goes in as given, without
    grounding_query's period; SAM 3 de-duplicates itself, so no NMS."""
    processor, model = _load_sam3_video(sam3_model_name, device)
    # The step's threshold is the floor for a detection and for a new object
    model.score_threshold_detection = threshold
    model.new_det_thresh = threshold
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        video_storage_device="cpu",
        inference_state_device="cpu",
        dtype=torch.float32,
    )
    processor.add_text_prompt(session, prompt.strip())
    tracks, scores = {}, {}
    with torch.inference_mode():
        for output in model.propagate_in_video_iterator(session):
            hidden = set(output.suppressed_obj_ids or ())
            for obj_id, low in (output.obj_id_to_mask or {}).items():
                if obj_id in hidden:
                    continue
                track = tracks.setdefault(obj_id, [None] * len(frames))
                track[output.frame_idx] = low.reshape(low.shape[-2:]).float().cpu()
                score = (output.obj_id_to_score or {}).get(obj_id)
                if score is not None and obj_id not in scores:
                    scores[obj_id] = float(score)
    removed = set(getattr(session, "hotstart_removed_obj_ids", ()) or ())
    return (
        {obj_id: track for obj_id, track in tracks.items() if obj_id not in removed},
        scores,
    )


def _full_size(low, height, width):
    """A low-res logit map at the frame's size, bilinear as SAM 3's own
    postprocess_outputs scales it."""
    return torch.nn.functional.interpolate(
        low[None, None], size=(height, width), mode="bilinear", align_corners=False
    )[0, 0].numpy()


def sam3_people(frames, tracks, scores, max_people):
    """([kept object], [frame][person] masks, found) for one cut from
    sam3_track's answer. First-frame people come first, by area, then late
    entrants in order of first appearance; max_people keeps that order's
    first, and the kept masks never overlap (_exclusive_masks)."""
    width, height = frames[0].size
    pixels = width * height
    people = []
    for obj_id, track in tracks.items():
        first = None
        for index, low in enumerate(track):
            if low is None:
                continue
            mask = _full_size(low, height, width) > 0
            if mask.any():
                first = (index, mask)
                break
        if first is None:
            continue  # never covered a pixel
        index, mask = first
        ys, xs = np.nonzero(mask)
        people.append(
            {
                "id": obj_id,
                "first_frame": index,
                "area": int(mask.sum()),
                "box": [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max() + 1),
                    float(ys.max() + 1),
                ],
                "score": scores.get(obj_id, 0.0),
            }
        )
    people.sort(
        key=lambda p: (
            (0, -p["area"], p["id"])
            if p["first_frame"] == 0
            else (1, p["first_frame"], p["id"])
        )
    )
    found = len(people)
    kept = people[:max_people]
    masks = []
    for index in range(len(frames)):
        rows = []
        for person in kept:
            low = tracks[person["id"]][index]
            rows.append(
                np.full((height, width), ABSENT_LOGIT, np.float32)
                if low is None
                else _full_size(low, height, width)
            )
        masks.append(
            _exclusive_masks(np.stack(rows), list(range(len(kept)))) if rows else []
        )
    for person in kept:
        person["area"] = round(person["area"] / pixels, 4)
    return kept, masks, found


def _exclusive_masks(logits, order):
    """Per-person masks from SAM2's logits (one row per object, in `order`):
    a pixel belongs to the person scoring it highest, and only if above 0."""
    count = len(order)
    best = logits.argmax(axis=0)
    positive = logits.max(axis=0) > 0
    by_person = [None] * count
    for row, person in enumerate(order):
        by_person[person] = positive & (best == row)
    return by_person


# ------------------------------------------------------------- contact sheet


def _outline(mask):
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    grown = image.filter(ImageFilter.MaxFilter(7))
    return np.asarray(grown) > np.asarray(image)


def contact_tile(frame, masks, name):
    """A cut's first frame with each person's mask outlined and numbered."""
    tile = frame.convert("RGB").copy()
    pixels = np.asarray(tile).copy()
    labels = []
    for number, mask in enumerate(masks, start=1):
        if mask is None or not mask.any():
            continue
        colour = SHEET_COLOURS[(number - 1) % len(SHEET_COLOURS)]
        pixels[_outline(mask)] = colour
        ys, xs = np.nonzero(mask)
        labels.append((number, colour, int(xs.min()), int(ys.min())))
    tile = Image.fromarray(pixels)
    scale = SHEET_TILE_WIDTH / tile.width
    tile = tile.resize((SHEET_TILE_WIDTH, max(1, round(tile.height * scale))))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default(size=SHEET_FONT_SIZE)
    for number, colour, x, y in labels:
        x, y = int(x * scale), int(y * scale)
        text = f"{PERSON_PREFIX}-{number}"
        left, top, right, bottom = draw.textbbox((x + 4, y + 4), text, font=font)
        draw.rectangle((left - 3, top - 3, right + 3, bottom + 3), fill=colour)
        draw.text((x + 4, y + 4), text, fill=(0, 0, 0), font=font)
    left, top, right, bottom = draw.textbbox((6, 6), name, font=font)
    draw.rectangle((left - 3, top - 3, right + 3, bottom + 3), fill=(0, 0, 0))
    draw.text((6, 6), name, fill=(255, 255, 255), font=font)
    return tile


def contact_sheet(tiles):
    """The tiles in rows of SHEET_COLUMNS, on black."""
    columns = min(SHEET_COLUMNS, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (columns * SHEET_TILE_WIDTH, rows * height))
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet.paste(tile, (column * SHEET_TILE_WIDTH, row * height))
    return sheet


# --------------------------------------------------------------------- task


def _read_video(video):
    if isinstance(video, VideoFileReference):
        video = load_audio_video(video.path)
    elif isinstance(video, str):
        video = load_audio_video(video)
    return video


def track_people(
    video,
    prompt="person",
    max_people=4,
    threshold=None,
    model_name=None,
    sam_model_name=None,
    backend=BACKEND_DINO,
    sam3_model_name=None,
    device="cpu",
):
    """Task command: one mask video per person a clip shows.

    Args:
        video: The clip - a path, an asset:/output: reference, or an earlier
            step's video
        prompt: What to track - Grounding-DINO's query on each cut's first
            frame, or SAM 3's text prompt on every frame
        max_people: The most people tracked, the largest kept (with sam3,
            first-frame people by size, then late entrants by appearance);
            0 tracks nobody, loads no model and answers the `people` record
        threshold: The detection score floor - Grounding-DINO's (default
            0.3) or SAM 3's (default 0.5); the two scales differ
        model_name: Grounding-DINO repo (default IDEA-Research/grounding-dino-base)
        sam_model_name: SAM2 repo (default facebook/sam2-hiera-large)
        backend: "grounding-dino" (the default: DINO boxes + SAM2) or
            "sam3" (SAM 3's video model; gated, transformers >= 5.21)
        sam3_model_name: SAM 3 repo (default facebook/sam3), sam3 only
        device: Where the models run

    Returns:
        {"person-1": video, ..., "contact_sheet": image, "people": record} -
        each video white where that person is, the source's frame count and
        size, numbered by mask area (largest first) on each cut's first frame
    """
    if video is None:
        raise ValueError(f"{COMMAND} needs 'video' - the clip to track people in")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{COMMAND}: 'prompt' must name what to track, e.g. 'person'")
    if backend not in BACKENDS:
        raise ValueError(
            f"{COMMAND} needs 'backend' as one of {list(BACKENDS)}, got {backend!r}"
        )
    max_people = whole_number(max_people, "max_people", COMMAND, required=True)
    threshold = float(DEFAULT_THRESHOLDS[backend] if threshold is None else threshold)
    check_arguments(COMMAND, max_people=max_people, threshold=threshold)
    model_name, sam_model_name = model_sources(
        {"model_name": model_name, "sam_model_name": sam_model_name}
    )
    sam3_model_name = sam3_model_name or DEFAULT_SAM3_MODEL
    if backend == BACKEND_SAM3 and max_people != 0:
        refusal = sam3_refusal()
        if refusal:
            raise ValueError(refusal)

    video = _read_video(video)
    frames = [frame.convert("RGB") for frame in frames_as_pil_list(video)]
    total = len(frames)
    if total == 0:
        raise ValueError(f"{COMMAND}: the video has no frames")
    fps = getattr(video, "fps", None)
    record = {
        "prompt": prompt,
        "backend": backend,
        "threshold": threshold,
        "max_people": max_people,
        "fps": None if fps is None else float(fps),
        "num_frames": total,
        "width": frames[0].width,
        "height": frames[0].height,
    }
    if max_people == 0:
        emit_warning(
            f"{COMMAND}: max_people is 0, so nobody was tracked and no mask "
            "was written - set it to the number of people to separate"
        )
        return {"people": JsonRecord({**record, "tracked": False, "people": 0})}

    starts = cut_starts(
        total,
        getattr(video, "shots", None),
        (np.asarray(frame) for frame in frames),
    )
    bounds = starts[1:] + [total]
    cuts, cut_masks, extra_tiles = [], [], []
    for number, (start, end) in enumerate(zip(starts, bounds), start=1):
        name = f"{SHOT_PREFIX}-{number}"
        if backend == BACKEND_SAM3:
            cut, masks = _sam3_cut(
                frames[start:end],
                name,
                start,
                prompt,
                threshold,
                sam3_model_name,
                max_people,
                device,
            )
            cuts.append(cut)
            cut_masks.append(masks)
            for person_number, person in enumerate(cut["people"], start=1):
                if person["first_frame"] > 0:
                    offset = person["first_frame"]
                    extra_tiles.append(
                        (
                            len(cuts) - 1,
                            contact_tile(
                                frames[start + offset],
                                masks[offset],
                                f"{name} @{offset} {PERSON_PREFIX}-{person_number}",
                            ),
                        )
                    )
            continue
        boxes, scores = detect_people(
            frames[start], prompt, threshold, model_name, device
        )
        found = distinct_boxes(boxes, scores)
        kept = sorted(found, key=lambda pair: -box_area(pair[0]))[:max_people]
        cut = {
            "name": name,
            "start_frame": start,
            "num_frames": end - start,
            "found": len(found),
            "people": [],
        }
        if not kept:
            cuts.append(cut)
            cut_masks.append([])
            continue
        masks = propagate_masks(
            frames[start:end], [box for box, _ in kept], sam_model_name, device
        )
        # Number by the area each mask covers on the cut's first frame
        first = masks[0]
        order = sorted(
            range(len(kept)),
            key=lambda i: (-int(first[i].sum()), -box_area(kept[i][0])),
        )
        masks = [[frame_masks[i] for i in order] for frame_masks in masks]
        pixels = frames[0].width * frames[0].height
        for rank, i in enumerate(order, start=1):
            box, score = kept[i]
            cut["people"].append(
                {
                    "name": f"{PERSON_PREFIX}-{rank}",
                    "box": [round(v, 1) for v in box],
                    "score": round(score, 4),
                    "area": round(int(first[i].sum()) / pixels, 4),
                    "first_frame": 0,
                }
            )
        cuts.append(cut)
        cut_masks.append(masks)

    count = max(len(cut["people"]) for cut in cuts)
    if count == 0:
        if backend == BACKEND_SAM3:
            where = "on any frame"
            detector = f"backend {BACKEND_SAM3!r} (SAM 3)"
        else:
            where = "on the first frame"
            detector = f"backend {BACKEND_DINO!r} (Grounding-DINO)"
        raise ValueError(
            f"{COMMAND}: no person found - {detector} matched nothing for "
            f"{prompt!r} at threshold {threshold} {where} of any of "
            f"the clip's {len(cuts)} cut(s). Lower 'threshold' or change "
            "'prompt' if someone is in frame."
        )

    _warn_about(cuts, max_people, backend)
    blank = Image.new("L", (frames[0].width, frames[0].height), 0)
    result = {}
    for person in range(count):
        mask_frames = []
        for (start, end), masks in zip(zip(starts, bounds), cut_masks):
            for offset in range(end - start):
                if masks and person < len(masks[offset]):
                    mask = masks[offset][person]
                    mask_frames.append(
                        Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
                    )
                else:
                    mask_frames.append(blank)
        result[f"{PERSON_PREFIX}-{person + 1}"] = AudioVideo(
            mask_frames, None, None, fps=fps
        )

    tiles = []
    for index, (cut, masks) in enumerate(zip(cuts, cut_masks)):
        tiles.append(
            contact_tile(
                frames[cut["start_frame"]], masks[0] if masks else [], cut["name"]
            )
        )
        # sam3: each late entrant also on the frame they first appear
        tiles.extend(tile for at, tile in extra_tiles if at == index)
    result["contact_sheet"] = contact_sheet(tiles)
    result["people"] = JsonRecord(
        {**record, "tracked": True, "people": count, "cuts": cuts}
    )
    logger.info(f"{COMMAND}: {count} person(s) over {len(cuts)} cut(s)")
    return result


def _sam3_cut(
    frames, name, start, prompt, threshold, sam3_model_name, max_people, device
):
    """(cut record, [frame][person] masks) for one cut on SAM 3."""
    tracks, scores = sam3_track(frames, prompt, threshold, sam3_model_name, device)
    kept, masks, found = sam3_people(frames, tracks, scores, max_people)
    cut = {
        "name": name,
        "start_frame": start,
        "num_frames": len(frames),
        "found": found,
        "people": [
            {
                "name": f"{PERSON_PREFIX}-{rank}",
                "box": [round(v, 1) for v in person["box"]],
                "score": round(person["score"], 4),
                "area": person["area"],
                "first_frame": person["first_frame"],
            }
            for rank, person in enumerate(kept, start=1)
        ],
    }
    return cut, masks if kept else []


def _warn_about(cuts, max_people, backend=BACKEND_DINO):
    empty = [cut["name"] for cut in cuts if not cut["people"]]
    if empty:
        emit_warning(
            f"{COMMAND}: nobody found on the first frame of {', '.join(empty)} - "
            "every mask is black there"
        )
    capped = [cut["name"] for cut in cuts if cut["found"] > max_people]
    if capped:
        emit_warning(
            f"{COMMAND}: more than max_people ({max_people}) found in "
            f"{', '.join(capped)} - "
            + (
                "the first-frame people (largest first), then the earliest "
                "late entrants, were kept"
                if backend == BACKEND_SAM3
                else "the largest were kept"
            )
        )
    if len(cuts) > 1:
        emit_warning(
            f"{COMMAND}: {len(cuts)} cuts - people are numbered by size within "
            "each cut, so person-1 is the largest figure in each, not one "
            "identity across them; check the contact sheet"
        )


@register_command(
    COMMAND,
    implementation="dw.tasks.person_track.track_people",
    consumes_device=True,
    required_weights="dw.tasks.person_track.required_weights",
    summary=(
        "Follow each person through a clip as their own mask video "
        "(person-1, person-2, ... by size), with a numbered contact sheet."
    ),
    parameter_descriptions={
        "video": (
            "Clip to track - a file path, an asset:/output: reference, or an "
            "earlier step's video. Tracks restart at each recorded shot and "
            "each detected cut."
        ),
        "prompt": (
            "What to track: Grounding-DINO's query on each cut's first frame, "
            "or with backend 'sam3' SAM 3's text prompt on every frame."
        ),
        "max_people": (
            "The most people tracked, the largest kept (sam3: first-frame "
            "people by size, then late entrants by first appearance). 0 tracks "
            "nobody and loads no model."
        ),
        "threshold": (
            "Detection score floor. Its scale depends on the backend: "
            "Grounding-DINO's score (default 0.3) or SAM 3's detection score "
            "(default 0.5) - a value tuned for one is not right for the other."
        ),
        "model_name": "Grounding-DINO repo (backend 'grounding-dino').",
        "sam_model_name": (
            "SAM2 repo whose video predictor propagates the masks "
            "(backend 'grounding-dino')."
        ),
        "backend": (
            "'grounding-dino' (default): Grounding-DINO boxes on each cut's "
            "first frame, propagated by SAM2 - ungated, Apache-2.0, misses "
            "anyone absent from a cut's first frame. 'sam3': SAM 3's video "
            "model detects on every frame, so a late entrant gets its own mask "
            "(first_frame > 0 in people). sam3 is opt-in: gated weights under "
            "the SAM License (not Apache; docs/ATTRIBUTIONS.md), and it needs "
            "transformers >= 5.21."
        ),
        "sam3_model_name": "SAM 3 repo (backend 'sam3').",
    },
    domains={"max_people": NON_NEGATIVE, "threshold": POSITIVE},
    choices={"backend": BACKENDS},
    whole_numbers=("max_people",),
    media_arguments=("video",),
    result_content_type="video/mp4",
)
def _handle_track_people(task, arguments, previous_pipelines):
    """One mask video per person in a clip"""
    logger.debug("Tracking people")
    return track_people(device=task.device_for(arguments), **arguments)
