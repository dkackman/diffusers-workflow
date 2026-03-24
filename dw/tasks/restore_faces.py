"""
Face restoration via spandrel + facexlib.

Uses facexlib for face detection/alignment/pasting and spandrel for
neural network inference on cropped faces. Supports GFPGAN, RestoreFormer,
and CodeFormer (via spandrel-extra-arches) model weights.
"""

import logging
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("dw")


def restore_faces(image, model_name, device="cpu", **kwargs):
    """Restore faces in an image using a spandrel-compatible face restoration model.

    Args:
        image: PIL Image containing faces to restore
        model_name: HuggingFace repo ID or local path to model weights
        device: Target device ("cuda", "mps", "cpu")
        **kwargs:
            filename: Weight file name within a HF repo (default: auto-detect)
            upscale_factor: Background upscale factor (default: 1, no upscaling)
            face_size: Cropped face size in pixels (default: 512)
            use_parse: Use face parsing for better blending (default: True)
            only_center_face: Only restore the largest/center face (default: False)
            detection_resize: Resize shorter side for detection speed (default: 640)
            eye_dist_threshold: Skip faces with eye distance below this (default: 5)
            upsample_img: Pre-upscaled background PIL Image (default: None)

    Returns:
        PIL Image with restored faces
    """
    try:
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    except ImportError:
        raise ImportError(
            "facexlib is required for face restoration. Install with: pip install facexlib"
        )

    from .upscale import _resolve_model_path

    filename = kwargs.get("filename", None)
    upscale_factor = kwargs.get("upscale_factor", 1)
    face_size = kwargs.get("face_size", 512)
    use_parse = kwargs.get("use_parse", True)
    only_center_face = kwargs.get("only_center_face", False)
    detection_resize = kwargs.get("detection_resize", 640)
    eye_dist_threshold = kwargs.get("eye_dist_threshold", 5)
    upsample_img = kwargs.get("upsample_img", None)

    # Load the face restoration model via spandrel
    model_path = _resolve_model_path(model_name, filename)
    descriptor = _load_face_model(model_path, device)

    # Set up facexlib helper
    face_helper = FaceRestoreHelper(
        upscale_factor=upscale_factor,
        face_size=face_size,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        use_parse=use_parse,
        device=torch.device(device),
    )

    # Convert PIL to BGR numpy (facexlib format)
    input_bgr = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    face_helper.read_image(input_bgr)

    # Detect faces
    num_faces = face_helper.get_face_landmarks_5(
        only_center_face=only_center_face,
        resize=detection_resize,
        eye_dist_threshold=eye_dist_threshold,
    )
    logger.info(f"Detected {num_faces} face(s)")

    if num_faces == 0:
        logger.warning("No faces detected, returning original image")
        return image

    # Align and warp faces to face_size x face_size
    face_helper.align_warp_face()

    # Use half precision on GPU if supported
    use_half = device != "cpu" and descriptor.supports_half
    if use_half:
        descriptor.model.half()
    model_dtype = torch.float16 if use_half else torch.float32

    # Restore each face
    for i, cropped_face in enumerate(face_helper.cropped_faces):
        logger.debug(f"Restoring face {i + 1}/{num_faces}")

        # BGR uint8 numpy -> float32 tensor [1, 3, H, W]
        face_tensor = (
            torch.from_numpy(cropped_face.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        face_tensor = face_tensor.to(device=device, dtype=model_dtype)

        with torch.inference_mode():
            restored_tensor = descriptor(face_tensor)

        # Tensor -> BGR uint8 numpy
        restored = restored_tensor.squeeze(0).permute(1, 2, 0)
        restored = (restored.clamp(0, 1) * 255).byte().cpu().numpy()

        # Resize to expected face_size if model output differs
        if restored.shape[:2] != (face_size, face_size):
            import cv2

            restored = cv2.resize(
                restored, (face_size, face_size), interpolation=cv2.INTER_LANCZOS4
            )

        face_helper.add_restored_face(restored)

    # Prepare inverse affine transforms
    face_helper.get_inverse_affine()

    # Paste faces back onto the image
    upsample_bgr = None
    if upsample_img is not None:
        upsample_bgr = np.array(upsample_img.convert("RGB"))[:, :, ::-1].copy()

    result_bgr = face_helper.paste_faces_to_input_image(upsample_img=upsample_bgr)

    # BGR numpy -> PIL RGB
    result = Image.fromarray(result_bgr[:, :, ::-1])
    logger.info(
        f"Face restoration complete ({num_faces} face(s), {result.width}x{result.height})"
    )

    face_helper.clean_all()
    return result


def _load_face_model(model_path, device):
    """Load a face restoration model via spandrel."""
    try:
        from spandrel import ModelLoader, ImageModelDescriptor
    except ImportError:
        raise ImportError(
            "spandrel is required for face restoration. Install with: pip install spandrel"
        )

    logger.info(f"Loading face restoration model from {model_path}")
    loader = ModelLoader(device=torch.device(device))
    descriptor = loader.load_from_file(model_path)

    if not isinstance(descriptor, ImageModelDescriptor):
        raise ValueError(
            f"Model is not an image model (got {type(descriptor).__name__}). "
            f"Expected a face restoration model (GFPGAN, CodeFormer, RestoreFormer)."
        )

    logger.info(f"Loaded {descriptor.architecture.name}")
    return descriptor
