"""
Image segmentation via GroundingDINO + SAM2.

Takes an image and a text prompt, detects objects matching the prompt,
and returns a binary mask image (white = detected object).
"""

import logging
import torch
import numpy as np
from PIL import Image, ImageOps
from .model_cache import cached_model
from .weights import argument_or_default, repo_entries

logger = logging.getLogger("dw")


class _TransformersClass:
    """A `transformers` class resolved on its first attribute read rather
    than at module top: the plan's `required_weights` lookup imports this
    module on every validate of a segment step, and transformers costs a
    second-plus the server need not pay. Module-level names so the tests
    patch them as such (`patch("dw.tasks.segment.Sam2Model")`)."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attribute):
        import transformers

        return getattr(getattr(transformers, self._name), attribute)


AutoProcessor = _TransformersClass("AutoProcessor")
AutoModelForZeroShotObjectDetection = _TransformersClass(
    "AutoModelForZeroShotObjectDetection"
)
Sam2Model = _TransformersClass("Sam2Model")
Sam2Processor = _TransformersClass("Sam2Processor")

_DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-base"
_DEFAULT_SAM_MODEL = "facebook/sam2-hiera-large"


def model_sources(arguments):
    """The (detector, segmenter) repos a segment step loads - `model_name`
    and `sam_model_name` over the defaults."""
    return (
        argument_or_default(arguments, "model_name", _DEFAULT_DINO_MODEL),
        argument_or_default(arguments, "sam_model_name", _DEFAULT_SAM_MODEL),
    )


def required_weights(arguments):
    """The detector and segmenter repos, for the plan (dw/tasks/weights.py)."""
    return repo_entries(*model_sources(arguments))


def grounding_query(prompt):
    """A prompt in the form GroundingDINO scores: lowercase, each phrase ending
    in a period. 'cat' scored 0.20 against a photo of a cat - under the 0.3
    threshold, so segment returned an empty mask - where 'cat.' scored 0.40."""
    query = prompt.strip().lower()
    return query if query.endswith(".") else f"{query}."


def segment_image(image, prompt, device="cpu", **kwargs):
    """Segment objects matching a text prompt, returning a binary mask.

    Args:
        image: PIL Image to segment
        prompt: Text description of object(s) to detect (e.g., "dog")
        device: Target device ("cuda", "mps", "cpu")
        **kwargs:
            model_name: GroundingDINO model ID
            sam_model_name: SAM2 model ID
            threshold: Detection confidence threshold (default: 0.3)
            invert: Invert the output mask (default: False)

    Returns:
        PIL Image in mode "L" — white (255) for detected objects, black (0) for background.
    """
    model_name, sam_model_name = model_sources(kwargs)
    threshold = kwargs.get("threshold", 0.3)
    invert = kwargs.get("invert", False)

    width, height = image.size

    def load_dino():
        logger.info(f"Loading GroundingDINO from {model_name}")
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(
            device
        )
        return processor, model

    dino_processor, dino_model = cached_model(
        ("segment_dino", model_name, str(device)), load_dino
    )

    inputs = dino_processor(
        images=image, text=grounding_query(prompt), return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        target_sizes=[(height, width)],
    )

    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]

    logger.info(f"Detected {len(boxes)} objects: {labels} (scores: {scores.tolist()})")

    if len(boxes) == 0:
        mask_image = Image.new("L", (width, height), 0)
        if invert:
            mask_image = ImageOps.invert(mask_image)
        return mask_image

    def load_sam():
        logger.info(f"Loading SAM2 from {sam_model_name}")
        processor = Sam2Processor.from_pretrained(sam_model_name)
        model = Sam2Model.from_pretrained(sam_model_name).to(device)
        return processor, model

    sam_processor, sam_model = cached_model(
        ("segment_sam", sam_model_name, str(device)), load_sam
    )

    input_boxes = [boxes.cpu().tolist()]

    sam_inputs = sam_processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        sam_outputs = sam_model(**sam_inputs)

    # SAM2's processor takes the original sizes alone - the third positional
    # argument is mask_threshold, and the SAM v1 'reshaped_input_sizes' key this
    # used to pass does not exist in SAM2's inputs at all
    masks = sam_processor.post_process_masks(
        sam_outputs.pred_masks, sam_inputs["original_sizes"]
    )

    # SAM2 answers each box with three candidate masks and rates each in
    # iou_scores. Take each box's best-rated one: the first is often a part of
    # the object - on the docs cup it was the body alone, missing the lid and
    # with a hole where the logo is, so remove_object left the cup (#974)
    best = sam_outputs.iou_scores[0].argmax(dim=-1).cpu()
    combined = masks[0][torch.arange(len(best)), best].sum(dim=0).clamp(0, 1)
    mask_array = (combined.cpu().numpy() * 255).astype(np.uint8)

    mask_image = Image.fromarray(mask_array, mode="L")

    if invert:
        mask_image = ImageOps.invert(mask_image)

    logger.info(f"Generated segmentation mask {width}x{height}")
    return mask_image
