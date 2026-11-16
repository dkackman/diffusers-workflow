"""Follow each person through a clip as a mask of their own (#1194, stage C).

`segment` masks everything its prompt matches in one picture, so on a
two-person shot 'person' is both of them, and a regional edit meant for one
repaints the other. `track_people` separates them:

1. The clip is split into cuts: frame 0, every shot the clip records
   (`dw/shots.py`), and every cut `detect_cuts`' detector finds
   (`scene_cuts.content_cuts`). A track never crosses a cut, where the
   picture - and who is where - starts over.
2. On each cut's first frame Grounding-DINO finds the prompt's boxes (the
   detector and query form `segment` uses). Overlapping boxes of one person
   (IoU above BOX_OVERLAP) keep the higher score, and the `max_people`
   largest are kept.
3. SAM2's video predictor (transformers' `Sam2VideoModel` on the same
   Apache-2.0 `facebook/sam2-hiera-large` weights `segment` loads) takes
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
from .segment import grounding_query
from .video_utils import VideoFileReference, frames_as_pil_list, load_audio_video
from .weights import argument_or_default, repo_entries

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

# segment's detector and segmenter: the SAM2 repo holds the video predictor too
DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-base"
DEFAULT_SAM_MODEL = "facebook/sam2-hiera-large"


class _TransformersClass:
    """A `transformers` class resolved on first use, as in `segment`: the
    plan's required_weights lookup imports this module on every validate.
    Module-level names so tests patch them as such."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attribute):
        import transformers

        return getattr(getattr(transformers, self._name), attribute)


AutoProcessor = _TransformersClass("AutoProcessor")
AutoModelForZeroShotObjectDetection = _TransformersClass(
    "AutoModelForZeroShotObjectDetection"
)
Sam2VideoModel = _TransformersClass("Sam2VideoModel")
Sam2VideoProcessor = _TransformersClass("Sam2VideoProcessor")


def model_sources(arguments):
    """The (detector, video segmenter) repos a track_people step loads."""
    return (
        argument_or_default(arguments, "model_name", DEFAULT_DINO_MODEL),
        argument_or_default(arguments, "sam_model_name", DEFAULT_SAM_MODEL),
    )


def required_weights(arguments):
    """The detector and SAM2 repos, for the plan (dw/tasks/weights.py) -
    none when a literal max_people of 0 tracks nobody and loads nothing."""
    max_people = arguments.get("max_people")
    if isinstance(max_people, (int, float)) and not isinstance(max_people, bool):
        if max_people == 0:
            return []
    return repo_entries(*model_sources(arguments))


# ------------------------------------------------------------------ geometry


def box_area(box):
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = box_area((ix0, iy0, ix1, iy1))
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def distinct_boxes(boxes, scores):
    """[(box, score)], highest score first, with any box overlapping a kept
    one by more than BOX_OVERLAP dropped - DINO often boxes one person twice."""
    kept = []
    for box, score in sorted(zip(boxes, scores), key=lambda pair: -pair[1]):
        if all(box_iou(box, other) <= BOX_OVERLAP for other, _ in kept):
            kept.append((box, score))
    return kept


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


def _load_dino(model_name, device):
    def load():
        logger.info(f"Loading GroundingDINO from {model_name}")
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(
            device
        )
        return processor, model

    # segment's key, so a workflow running both holds one copy
    return cached_model(("segment_dino", model_name, str(device)), load)


def _load_sam_video(sam_model_name, device):
    def load():
        logger.info(f"Loading SAM2 video predictor from {sam_model_name}")
        processor = Sam2VideoProcessor.from_pretrained(sam_model_name)
        model = Sam2VideoModel.from_pretrained(sam_model_name).to(device)
        return processor, model

    return cached_model(("track_people_sam2_video", sam_model_name, str(device)), load)


def detect_people(image, prompt, threshold, model_name, device):
    """([box xyxy], [score]) Grounding-DINO finds for the prompt on image."""
    processor, model = _load_dino(model_name, device)
    width, height = image.size
    inputs = processor(
        images=image, text=grounding_query(prompt), return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    found = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        target_sizes=[(height, width)],
    )[0]
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
    threshold=0.3,
    model_name=None,
    sam_model_name=None,
    device="cpu",
):
    """Task command: one mask video per person a clip shows.

    Args:
        video: The clip - a path, an asset:/output: reference, or an earlier
            step's video
        prompt: What Grounding-DINO looks for on each cut's first frame
        max_people: The most people tracked, the largest kept; 0 tracks
            nobody, loads no model and answers the `people` record alone
        threshold: Grounding-DINO's detection score floor
        model_name: Grounding-DINO repo (default IDEA-Research/grounding-dino-base)
        sam_model_name: SAM2 repo (default facebook/sam2-hiera-large)
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
    max_people = whole_number(max_people, "max_people", COMMAND, required=True)
    threshold = float(threshold)
    check_arguments(COMMAND, max_people=max_people, threshold=threshold)
    model_name, sam_model_name = model_sources(
        {"model_name": model_name, "sam_model_name": sam_model_name}
    )

    video = _read_video(video)
    frames = [frame.convert("RGB") for frame in frames_as_pil_list(video)]
    total = len(frames)
    if total == 0:
        raise ValueError(f"{COMMAND}: the video has no frames")
    fps = getattr(video, "fps", None)
    record = {
        "prompt": prompt,
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
    cuts, cut_masks = [], []
    for number, (start, end) in enumerate(zip(starts, bounds), start=1):
        name = f"{SHOT_PREFIX}-{number}"
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
                }
            )
        cuts.append(cut)
        cut_masks.append(masks)

    count = max(len(cut["people"]) for cut in cuts)
    if count == 0:
        raise ValueError(
            f"{COMMAND}: no person found - Grounding-DINO matched nothing for "
            f"{prompt!r} at threshold {threshold} on the first frame of any of "
            f"the clip's {len(cuts)} cut(s). Lower 'threshold' or change "
            "'prompt' if someone is in frame."
        )

    _warn_about(cuts, max_people)
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

    result["contact_sheet"] = contact_sheet(
        [
            contact_tile(
                frames[cut["start_frame"]], masks[0] if masks else [], cut["name"]
            )
            for cut, masks in zip(cuts, cut_masks)
        ]
    )
    result["people"] = JsonRecord(
        {**record, "tracked": True, "people": count, "cuts": cuts}
    )
    logger.info(f"{COMMAND}: {count} person(s) over {len(cuts)} cut(s)")
    return result


def _warn_about(cuts, max_people):
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
            f"{', '.join(capped)} - the largest were kept"
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
        "prompt": "What Grounding-DINO finds on each cut's first frame.",
        "max_people": (
            "The most people tracked, the largest kept. 0 tracks nobody and "
            "loads no model."
        ),
        "threshold": "Grounding-DINO's detection score floor.",
        "model_name": "Grounding-DINO repo.",
        "sam_model_name": "SAM2 repo whose video predictor propagates the masks.",
    },
    domains={"max_people": NON_NEGATIVE, "threshold": POSITIVE},
    whole_numbers=("max_people",),
    media_arguments=("video",),
    result_content_type="video/mp4",
)
def _handle_track_people(task, arguments, previous_pipelines):
    """One mask video per person in a clip"""
    logger.debug("Tracking people")
    return track_people(device=task.device_for(arguments), **arguments)
