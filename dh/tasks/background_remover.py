from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def remove_background(image: Image, device_identifier) -> Image:
    # Model settings
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0", trust_remote_code=True
    )
    model.to(device_identifier)
    model.eval()

    # Data settings
    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    working_copy = image.copy()
    input_images = transform_image(working_copy).unsqueeze(0).to(device_identifier)

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(working_copy.size)
    working_copy.putalpha(mask)

    return working_copy
