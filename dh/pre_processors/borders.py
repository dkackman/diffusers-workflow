from PIL import Image

def add_border_and_mask(image, zoom_all=1.0, zoom_left=0, zoom_right=0, zoom_up=0, zoom_down=0, overlap=0):
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

    return  {
        "bordered_image": bordered_image, 
        "mask": mask
    }
