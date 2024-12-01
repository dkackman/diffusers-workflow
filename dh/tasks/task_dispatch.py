from .qr_code import get_qrcode_image
from .image_processor_dispatch import process_image


def dispatch_task(command, kwargs):
    command = command.lower()

    if command == "qr_code":
        return get_qrcode_image(**kwargs)
    
    if "image" in kwargs:
        return process_image(kwargs.pop("image"), command, kwargs.pop("device_identifier", "cuda"), kwargs)
                
    raise ValueError(f"Unknown task {command}")