import qrcode
from .image_utils import resize_for_condition_image


def get_qrcode_image(qr_code_contents, height=768, width=768):
    # base the resolution off of size - defaulting to 768
    resolution = max(height, width)

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_code_contents)
    qr.make(fit=True)

    qrcode_image = qr.make_image(fill_color="black", back_color="white")
    return resize_for_condition_image(qrcode_image, resolution)
