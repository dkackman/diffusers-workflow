from PIL import Image


def add_border_and_mask(
    image, zoom_all=1.0, zoom_left=0, zoom_right=0, zoom_up=0, zoom_down=0, overlap=0
):
    """Adds a black border around the image with individual side control and mask overlap"""
    orig_width, orig_height = image.size

    # Calculate padding for each side (in pixels)
    left_pad = int(orig_width * zoom_left)
    right_pad = int(orig_width * zoom_right)
    top_pad = int(orig_height * zoom_up)
    bottom_pad = int(orig_height * zoom_down)

    # Calculate overlap in pixels
    overlap_left = int(orig_width * overlap)
    overlap_right = int(orig_width * overlap)
    overlap_top = int(orig_height * overlap)
    overlap_bottom = int(orig_height * overlap)

    # If using the all-sides zoom, add it to each side
    if zoom_all > 1.0:
        extra_each_side = (zoom_all - 1.0) / 2
        left_pad += int(orig_width * extra_each_side)
        right_pad += int(orig_width * extra_each_side)
        top_pad += int(orig_height * extra_each_side)
        bottom_pad += int(orig_height * extra_each_side)

    # Calculate new dimensions (ensure they're multiples of 32)
    new_width = 32 * round((orig_width + left_pad + right_pad) / 32)
    new_height = 32 * round((orig_height + top_pad + bottom_pad) / 32)

    # Create new image with black border
    bordered_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    # Paste original image in position
    paste_x = left_pad
    paste_y = top_pad
    bordered_image.paste(image, (paste_x, paste_y))

    # Create mask (white where the border is, black where the original image was)
    mask = Image.new("L", (new_width, new_height), 255)  # White background
    # Paste black rectangle with overlap adjustment
    mask.paste(
        0,
        (
            paste_x + overlap_left,  # Left edge moves right
            paste_y + overlap_top,  # Top edge moves down
            paste_x + orig_width - overlap_right,  # Right edge moves left
            paste_y + orig_height - overlap_bottom,  # Bottom edge moves up
        ),
    )

    return {"bordered_image": bordered_image, "mask": mask}


def add_border_and_mask_with_size(image, width, height, overlap=0):
    """
    Resizes the original image to fit within the target dimensions while maintaining
    its aspect ratio, then adds borders as needed to reach the exact target size.

    Args:
        image: PIL Image object
        width: Target width in pixels
        height: Target height in pixels
        overlap: Mask overlap parameter (0-1 range)

    Returns:
        Dictionary with 'bordered_image' and 'mask'
    """
    # Ensure width and height are multiples of 32
    width = 32 * round(width / 32)
    height = 32 * round(height / 32)

    # Get original dimensions
    orig_width, orig_height = image.size
    orig_aspect = orig_width / orig_height
    target_aspect = width / height

    # Resize image to fit within target dimensions while maintaining aspect ratio
    if orig_aspect > target_aspect:
        # Original is wider than target - fit width
        new_width = width
        new_height = int(width / orig_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        # Original is taller than target - fit height
        new_height = height
        new_width = int(height * orig_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Now calculate padding to reach target dimensions
    left_pad = (width - new_width) // 2
    right_pad = width - new_width - left_pad
    top_pad = (height - new_height) // 2
    bottom_pad = height - new_height - top_pad

    # Convert padding to zoom factors (relative to resized dimensions)
    zoom_left = left_pad / new_width if new_width > 0 else 0
    zoom_right = right_pad / new_width if new_width > 0 else 0
    zoom_up = top_pad / new_height if new_height > 0 else 0
    zoom_down = bottom_pad / new_height if new_height > 0 else 0

    # Call the original function with calculated zoom parameters
    return add_border_and_mask(
        resized_image,
        zoom_all=1.0,
        zoom_left=zoom_left,
        zoom_right=zoom_right,
        zoom_up=zoom_up,
        zoom_down=zoom_down,
        overlap=overlap,
    )
