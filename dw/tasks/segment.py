"""
Image segmentation via GroundingDINO + SAM2.

Takes an image and a text prompt, detects objects matching the prompt,
and returns a binary mask image (white = detected object).
"""

import logging
import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Sam2Model,
    Sam2Processor,
)

logger = logging.getLogger("dw")

_DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-base"
_DEFAULT_SAM_MODEL = "facebook/sam2-hiera-large"


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
    model_name = kwargs.get("model_name", _DEFAULT_DINO_MODEL)
    sam_model_name = kwargs.get("sam_model_name", _DEFAULT_SAM_MODEL)
    threshold = kwargs.get("threshold", 0.3)
    invert = kwargs.get("invert", False)

    width, height = image.size

    logger.info(f"Loading GroundingDINO from {model_name}")
    dino_processor = AutoProcessor.from_pretrained(model_name)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(
        device
    )

    inputs = dino_processor(images=image, text=prompt, return_tensors="pt").to(device)

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

    logger.info(f"Loading SAM2 from {sam_model_name}")
    sam_processor = Sam2Processor.from_pretrained(sam_model_name)
    sam_model = Sam2Model.from_pretrained(sam_model_name).to(device)

    input_boxes = [boxes.cpu().tolist()]

    sam_inputs = sam_processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        sam_outputs = sam_model(**sam_inputs)

    masks = sam_processor.post_process_masks(
        sam_outputs.pred_masks,
        sam_inputs["original_sizes"],
        sam_inputs["reshaped_input_sizes"],
    )

    combined = masks[0][:, 0].sum(dim=0).clamp(0, 1)
    mask_array = (combined.cpu().numpy() * 255).astype(np.uint8)

    mask_image = Image.fromarray(mask_array, mode="L")

    if invert:
        mask_image = ImageOps.invert(mask_image)

    logger.info(f"Generated segmentation mask {width}x{height}")
    return mask_image
