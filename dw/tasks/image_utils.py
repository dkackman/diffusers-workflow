from PIL import Image
import numpy as np
from .borders import add_border_and_mask, add_border_and_mask_with_size
from .model_cache import cached_model
import torch

# cv2, controlnet_aux, transformers and the model-backed task modules are imported
# inside the functions that use them - at module scope they add seconds to every
# startup (including the REPL worker spawn and dw.validate) for workflows that
# never touch an image-processing task


def _import_controlnet_aux():
    import controlnet_aux

    return controlnet_aux


# ---------------------------------------------------------------------------
# controlnet_aux detector dispatch
#
# Most controlnet_aux detectors follow one shape:
#   getattr(controlnet_aux, attr).from_pretrained(*repo_args, **repo_kwargs).to(device)(image, **call_kwargs, **kwargs)
# The table below drives a single generic loader for that shape instead of a
# hand-written branch per detector. Loaded detectors are cached by
# (attr, device) via cached_model so repeated process_image calls - e.g. once
# per cartesian-product iteration in step.py - reuse the loaded weights
# instead of reloading them from disk every time.
# ---------------------------------------------------------------------------

# name -> (controlnet_aux attribute, from_pretrained positional args,
#          from_pretrained kwargs, fixed call() kwargs)
_PRETRAINED_DETECTOR_SPECS = {
    "mlsd": ("MLSDdetector", ("lllyasviel/Annotators",), {}, {}),
    "normal_bae": ("NormalBaeDetector", ("lllyasviel/Annotators",), {}, {}),
    "lineart": ("LineartDetector", ("lllyasviel/Annotators",), {}, {"coarse": True}),
    "openpose": (
        "OpenposeDetector",
        ("lllyasviel/Annotators",),
        {},
        {"hand_and_face": True},
    ),
    "hed": ("HEDdetector", ("lllyasviel/Annotators",), {}, {"scribble": False}),
    "scribble": ("HEDdetector", ("lllyasviel/Annotators",), {}, {"scribble": True}),
    "pidi": ("PidiNetDetector", ("lllyasviel/Annotators",), {}, {"safe": True}),
    "midas": ("MidasDetector", ("lllyasviel/Annotators",), {}, {}),
    "zoe": ("ZoeDetector", ("lllyasviel/Annotators",), {}, {}),
    "teed": ("TEEDdetector", ("fal-ai/teed",), {"filename": "5_model.pth"}, {}),
    "anyline": (
        "AnylineDetector",
        ("TheMistoAI/MistoLine",),
        {"filename": "MTEED.pth", "subfolder": "Anyline"},
        {},
    ),
    "leres": ("LeresDetector", ("lllyasviel/Annotators",), {}, {}),
    # sam is the one from_pretrained detector that is never moved to device -
    # see _PROCESSORS registration below.
}

# Detectors constructed with no from_pretrained call at all.
_ZERO_ARG_DETECTOR_SPECS = {
    "shuffle": "ContentShuffleDetector",
    # controlnet_aux CannyDetector: plain cv2 Canny internally, but resizes
    # the input to 512px first. Kept distinct from "canny_cv" below, which
    # runs cv2 directly at the image's native resolution - same algorithm,
    # different output size, so both names are intentional, not duplicates.
    "canny": "CannyDetector",
    "lineart_standard": "LineartStandardDetector",
}


def _build_pretrained_detector(attr, repo_args, repo_kwargs, to_device, device):
    controlnet_aux = _import_controlnet_aux()
    detector = getattr(controlnet_aux, attr).from_pretrained(*repo_args, **repo_kwargs)
    if to_device:
        detector = detector.to(device)
    return detector


def _build_zero_arg_detector(attr):
    controlnet_aux = _import_controlnet_aux()
    return getattr(controlnet_aux, attr)()


def _make_pretrained_handler(attr, repo_args, repo_kwargs, call_kwargs, to_device=True):
    def handler(image, device, kwargs):
        detector = cached_model(
            ("image_processor", attr, str(device)),
            lambda: _build_pretrained_detector(
                attr, repo_args, repo_kwargs, to_device, device
            ),
        )
        return detector(image, **call_kwargs, **kwargs)

    return handler


def _make_zero_arg_handler(attr):
    def handler(image, device, kwargs):
        detector = cached_model(
            ("image_processor", attr, str(device)),
            lambda: _build_zero_arg_detector(attr),
        )
        return detector(image, **kwargs)

    return handler


def _dw_pose_handler(image, device, kwargs):
    detector = cached_model(
        ("image_processor", "DWposeDetector", str(device)),
        lambda: _import_controlnet_aux().DWposeDetector(device=device),
    )
    return detector(image, **kwargs)


def _remove_background_handler(image, device, kwargs):
    from .background_remover import remove_background

    return remove_background(image, device, **kwargs)


def _depth_estimator_tensor_handler(image, device, kwargs):
    from .depth_estimator import make_hint_tensor

    return make_hint_tensor(image, device, **kwargs)


def _depth_estimator_handler(image, device, kwargs):
    from .depth_estimator import make_hint_image

    return make_hint_image(image, device, **kwargs)


def get_zoe_depth_map(image, device):
    from .zoe_depth import colorize, load_zoe

    model_zoe_n = load_zoe(device)
    # MPS doesn't support autocast, so use 'cpu' for autocast when on MPS
    from dw import get_autocast_device_type

    autocast_device = get_autocast_device_type()
    if autocast_device == "cuda":
        with torch.autocast(autocast_device, enabled=True):
            depth = model_zoe_n.infer_pil(image)
    else:
        # For MPS/CPU, don't use autocast
        depth = model_zoe_n.infer_pil(image)
    return colorize(depth, cmap="gray_r")


def image_to_canny(image, low_threshold=100, high_threshold=200):
    # Raw cv2.Canny at the image's native resolution - intentionally kept
    # separate from the "canny" controlnet_aux CannyDetector above, which
    # resizes to 512px first. See comment on _ZERO_ARG_DETECTOR_SPECS["canny"].
    import cv2

    image = np.array(image)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def image_to_depth(image, device, height=1024, width=1024):
    from transformers import DPTForDepthEstimation, DPTImageProcessor

    size = (width, height)
    depth_estimator = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas"
    ).to(device)
    feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    # MPS doesn't support autocast, so use 'cpu' for autocast when on MPS
    from dw import get_autocast_device_type

    autocast_device = get_autocast_device_type()
    if autocast_device == "cuda":
        with torch.no_grad(), torch.autocast(autocast_device):
            depth_map = depth_estimator(image).predicted_depth
    else:
        # For MPS/CPU, don't use autocast
        with torch.no_grad():
            depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=size,
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


def image_to_segmentation(image):
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

    image_processor = AutoImageProcessor.from_pretrained(
        "openmmlab/upernet-convnext-small"
    )
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-convnext-small"
    )
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3
    for label, color in enumerate(ada_palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    return Image.fromarray(color_seg)


def get_image_size(image):
    return {"width": image.width, "height": image.height}


def crop_square(img: Image) -> Image:
    # Determine the shortest side
    min_side = min(img.width, img.height)

    # Calculate the left and right crop positions for centering
    left = (img.width - min_side) // 2
    right = left + min_side

    # Calculate the top and bottom crop positions for centering
    top = (img.height - min_side) // 2
    bottom = top + min_side

    # Crop the image
    img_cropped = img.crop((left, top, right, bottom))

    return img_cropped


def resize_center_crop(img, height=768, width=768):
    output_size = (width, height)
    W, H = img.size

    # Calculate dimensions to crop to the center
    new_dimension = min(W, H)
    left = (W - new_dimension) / 2
    top = (H - new_dimension) / 2
    right = (W + new_dimension) / 2
    bottom = (H + new_dimension) / 2

    # Crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)

    return img


def resize_rescale(image, height=768, width=768):
    input_image = image.convert("RGB")
    return input_image.resize((width, height))


def resize_resample(image, resolution=1024):
    input_image = image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64

    return input_image.resize((W, H), resample=Image.LANCZOS)


# Standard aspect ratios used by SDXL, Flux, and similar models.
# Each entry is (width_ratio, height_ratio).
_DEFAULT_RATIOS = [
    (1, 1),
    (4, 3),
    (3, 4),
    (3, 2),
    (2, 3),
    (16, 9),
    (9, 16),
    (21, 9),
    (9, 21),
]


def resize_bucket(image, resolution=1024, ratios=None, alignment=64):
    """Resize image to the closest model-native aspect ratio bucket.

    Picks the standard ratio closest to the input image's natural aspect
    ratio, then scales to fit within the target resolution (based on the
    short side) with dimensions aligned to `alignment` pixels.

    Args:
        image: PIL Image to resize.
        resolution: Target size for the short side in pixels (default: 1024).
        ratios: Optional list of [w, h] ratio pairs. Defaults to standard
            ratios used by SDXL/Flux (1:1, 4:3, 3:2, 16:9, etc.).
        alignment: Round dimensions to this multiple (default: 64).

    Returns:
        PIL Image resized to the bucketed dimensions.
    """
    input_image = image.convert("RGB")
    W, H = input_image.size
    input_ratio = W / H

    bucket_ratios = ratios if ratios is not None else _DEFAULT_RATIOS

    # Find the closest aspect ratio
    best_ratio = min(
        bucket_ratios,
        key=lambda r: abs((r[0] / r[1]) - input_ratio),
    )

    wr, hr = best_ratio
    bucket_ratio = wr / hr

    # Scale so the short side matches resolution, then align
    if bucket_ratio >= 1.0:
        # Landscape or square: height is the short side
        out_h = int(round(resolution / alignment)) * alignment
        out_w = int(round((out_h * bucket_ratio) / alignment)) * alignment
    else:
        # Portrait: width is the short side
        out_w = int(round(resolution / alignment)) * alignment
        out_h = int(round((out_w / bucket_ratio) / alignment)) * alignment

    return input_image.resize((out_w, out_h), resample=Image.LANCZOS)


def strip_exif(image):
    """Remove all EXIF and metadata from an image.

    Creates a clean copy with pixel data only — no GPS coordinates,
    camera info, timestamps, or other embedded metadata.

    Args:
        image: PIL Image to strip.

    Returns:
        PIL Image with all metadata removed.
    """
    clean = Image.new(image.mode, image.size)
    clean.paste(image)
    return clean


def add_watermark(
    image,
    text="AI Generated",
    position="bottom-right",
    opacity=128,
    font_size=0,
    margin=10,
    color=None,
):
    """Add a visible text watermark to an image.

    Args:
        image: PIL Image to watermark.
        text: Watermark text (default: "AI Generated").
        position: Placement — "bottom-right", "bottom-left", "top-right",
            "top-left", or "center" (default: "bottom-right").
        opacity: Text opacity 0-255 (default: 128).
        font_size: Font size in pixels. 0 = auto-scale to ~3% of image height.
        margin: Pixel margin from edges (default: 10).
        color: RGB tuple for text color (default: white).

    Returns:
        PIL Image with watermark applied.
    """
    from PIL import ImageDraw, ImageFont

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if color is None:
        color = (255, 255, 255)
    fill = (*color, int(opacity))

    if font_size <= 0:
        font_size = max(12, base.height // 30)

    try:
        font = ImageFont.truetype("Arial", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default(size=font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    positions = {
        "bottom-right": (base.width - text_w - margin, base.height - text_h - margin),
        "bottom-left": (margin, base.height - text_h - margin),
        "top-right": (base.width - text_w - margin, margin),
        "top-left": (margin, margin),
        "center": ((base.width - text_w) // 2, (base.height - text_h) // 2),
    }
    xy = positions.get(position, positions["bottom-right"])

    draw.text(xy, text, font=font, fill=fill)

    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


# ---------------------------------------------------------------------------
# process_image dispatch table
#
# Every handler has the uniform signature (image, device, kwargs) -> result,
# so process_image is just a lookup + call. Built once at import time from
# the detector spec tables above plus direct entries for the plain PIL/task
# functions.
# ---------------------------------------------------------------------------

_PROCESSORS = {
    "get_image_size": lambda image, device, kwargs: get_image_size(image),
    "add_border_and_mask": lambda image, device, kwargs: add_border_and_mask(
        image, **kwargs
    ),
    "add_border_and_mask_with_size": lambda image, device, kwargs: add_border_and_mask_with_size(
        image, **kwargs
    ),
    "remove_background": _remove_background_handler,
    # Raw cv2 Canny at native resolution - see image_to_canny() docstring
    # comment for how this differs from "canny" below.
    "canny_cv": lambda image, device, kwargs: image_to_canny(image, **kwargs),
    "segmentation": lambda image, device, kwargs: image_to_segmentation(image),
    "zoe_depth": lambda image, device, kwargs: get_zoe_depth_map(image, device),
    "depth": lambda image, device, kwargs: image_to_depth(image, device, **kwargs),
    "depth_estimator_tensor": _depth_estimator_tensor_handler,
    "depth_estimator": _depth_estimator_handler,
    "resize_center_crop": lambda image, device, kwargs: resize_center_crop(
        image, **kwargs
    ),
    "resize_resample": lambda image, device, kwargs: resize_resample(image, **kwargs),
    "crop_square": lambda image, device, kwargs: crop_square(image, **kwargs),
    "resize_rescale": lambda image, device, kwargs: resize_rescale(image, **kwargs),
    "resize_bucket": lambda image, device, kwargs: resize_bucket(image, **kwargs),
    "strip_exif": lambda image, device, kwargs: strip_exif(image),
    "add_watermark": lambda image, device, kwargs: add_watermark(image, **kwargs),
}

for _name, (
    _attr,
    _repo_args,
    _repo_kwargs,
    _call_kwargs,
) in _PRETRAINED_DETECTOR_SPECS.items():
    _PROCESSORS[_name] = _make_pretrained_handler(
        _attr, _repo_args, _repo_kwargs, _call_kwargs
    )

# sam is the one from_pretrained detector never moved to device - matches
# the pre-refactor behavior, which called it straight off from_pretrained().
_PROCESSORS["sam"] = _make_pretrained_handler(
    "SamDetector",
    ("ybelkada/segment-anything",),
    {"subfolder": "checkpoints"},
    {},
    to_device=False,
)

for _name, _attr in _ZERO_ARG_DETECTOR_SPECS.items():
    _PROCESSORS[_name] = _make_zero_arg_handler(_attr)

_PROCESSORS["dw_pose"] = _dw_pose_handler

del _name, _attr, _repo_args, _repo_kwargs, _call_kwargs


def available_processors():
    """Return the sorted list of processor names process_image accepts.

    Used by command registration to enumerate supported image processors
    without duplicating this dispatch table.
    """
    return sorted(_PROCESSORS)


def process_image(image, processor, device, kwargs):
    processor = processor.lower()

    handler = _PROCESSORS.get(processor)
    if handler is None:
        raise Exception(f"Unknown image processor type: {processor}")

    return handler(image, device, kwargs)


ada_palette = np.asarray(
    [
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]
)
