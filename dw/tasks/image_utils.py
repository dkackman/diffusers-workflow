from PIL import Image
import cv2
import numpy as np
from controlnet_aux import (
    MLSDdetector,
    NormalBaeDetector,
    LineartDetector,
    OpenposeDetector,
    HEDdetector,
    PidiNetDetector,
    ContentShuffleDetector,
    MidasDetector,
    ZoeDetector,
    SamDetector,
    LeresDetector,
    TEEDdetector,
    AnylineDetector,
    MediapipeFaceDetector,
    CannyDetector,
    LineartStandardDetector,
    DWposeDetector,
)
from transformers import (
    AutoImageProcessor,
    UperNetForSemanticSegmentation,
    DPTForDepthEstimation,
    DPTFeatureExtractor,
)
from .zoe_depth import colorize, load_zoe
from .background_remover import remove_background
from .depth_estimator import make_hint_image, make_hint_tensor
from .borders import add_border_and_mask, add_border_and_mask_with_size
import torch


def process_image(image, processor, device_identifier, kwargs):
    processor = processor.lower()

    if processor == "get_image_size":
        return get_image_size(image)

    if processor == "add_border_and_mask":
        return add_border_and_mask(image, **kwargs)    
        
    if processor == "add_border_and_mask_with_size":
        return add_border_and_mask_with_size(image, **kwargs)

    if processor == "remove_background":
        return remove_background(image, device_identifier, **kwargs)

    if processor == "canny_cv":
        return image_to_canny(image, **kwargs)

    if processor == "mlsd":
        return MLSDdetector.from_pretrained("lllyasviel/Annotators").to(
            device_identifier
        )(image, **kwargs)

    if processor == "normal_bae":
        return NormalBaeDetector.from_pretrained("lllyasviel/Annotators").to(
            device_identifier
        )(image, **kwargs)

    if processor == "segmentation":
        return image_to_segmentation(image)

    if processor == "lineart":
        return LineartDetector.from_pretrained("lllyasviel/Annotators").to(
            device_identifier
        )(image, coarse=True, **kwargs)

    if processor == "openpose":
        return OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(
            device_identifier
        )(image, hand_and_face=True, **kwargs)

    if processor == "hed":
        return HEDdetector.from_pretrained("lllyasviel/Annotators").to(
            device_identifier
        )(image, scribble=False, **kwargs)

    if processor == "scribble":
        return HEDdetector.from_pretrained("lllyasviel/Annotators").to(
            device_identifier
        )(image, scribble=True, **kwargs)

    if processor == "pidi":
        return PidiNetDetector.from_pretrained("lllyasviel/Annotators").to(
            device_identifier
        )(image, safe=True, **kwargs)

    if processor == "midas":
        return MidasDetector.from_pretrained("lllyasviel/Annotators").to(
            device_identifier
        )(image, **kwargs)

    if processor == "shuffle":
        processor = ContentShuffleDetector()
        return processor(image, **kwargs)

    if processor == "face_detector":
        processor = MediapipeFaceDetector()
        return processor(image, **kwargs)

    if processor == "canny":
        processor = CannyDetector()
        return processor(image, **kwargs)

    if processor == "lineart_standard":
        processor = LineartStandardDetector()
        return processor(image, **kwargs)

    if processor == "dw_pose":
        processor = DWposeDetector(device=device_identifier)
        return processor(image, **kwargs)

    if processor == "zoe_depth":
        return get_zoe_depth_map(image, device_identifier)

    if processor == "zoe":
        return ZoeDetector.from_pretrained("lllyasviel/Annotators").to(
            device_identifier
        )(image, **kwargs)

    if processor == "sam":
        return SamDetector.from_pretrained(
            "ybelkada/segment-anything", subfolder="checkpoints"
        )(image, **kwargs)

    if processor == "teed":
        return TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth").to(
            device_identifier
        )(image, **kwargs)

    if processor == "anyline":
        return AnylineDetector.from_pretrained(
            "TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline"
        ).to(device_identifier)(image, **kwargs)

    if processor == "leres":
        return LeresDetector.from_pretrained("lllyasviel/Annotators").to(
            device_identifier
        )(image, **kwargs)

    if processor == "depth":
        return image_to_depth(image, device_identifier, **kwargs)

    if processor == "depth_estimator_tensor":
        return make_hint_tensor(image, device_identifier)

    if processor == "depth_estimator":
        return make_hint_image(image, device_identifier)

    if processor == "resize_center_crop":
        return resize_center_crop(image, **kwargs)

    if processor == "resize_resample":
        return resize_resample(image, **kwargs)

    if processor == "crop_square":
        return crop_square(image, **kwargs)

    if processor == "resize_rescale":
        return resize_rescale(image, **kwargs)

    raise Exception(f"Unknown image processor type: {processor}")


def get_zoe_depth_map(image, device_identifier):
    model_zoe_n = load_zoe()
    with torch.autocast(device_identifier, enabled=True):
        depth = model_zoe_n.infer_pil(image)
    return colorize(depth, cmap="gray_r")


def image_to_canny(image, low_threshold=100, high_threshold=200):
    image = np.array(image)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def image_to_depth(image, device_identifier, height=1024, width=1024):
    size = (width, height)
    depth_estimator = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas"
    ).to(device_identifier)
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(
        device_identifier
    )
    with torch.no_grad(), torch.autocast(device_identifier):
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
