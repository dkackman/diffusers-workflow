import qrcode
from ..toolbox.image_utils import resize_for_condition_image


def get_qrcode_image(qr_code_contents, size = (768, 768)):
    # base the resolution of of size - defaulting to 768
    W, H = size if size is not None else (768, 768)
    resolution = max(H, W)

    # user passed a qrcode - generate image
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
