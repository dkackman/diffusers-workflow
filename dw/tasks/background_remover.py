from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from .model_cache import cached_model

_MODEL_NAME = "briaai/RMBG-2.0"


def remove_background(image: Image, device) -> Image:
    # Model settings
    def load_model():
        model = AutoModelForImageSegmentation.from_pretrained(
            _MODEL_NAME, trust_remote_code=True
        )
        model.to(device)
        model.eval()
        return model

    model = cached_model(("background_remover", _MODEL_NAME, str(device)), load_model)

    # Data settings
    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    working_copy = image.copy()
    input_images = transform_image(working_copy).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(working_copy.size)
    working_copy.putalpha(mask)

    return working_copy
