from PIL import Image


def crop_sqaure(img: Image) -> Image:
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
