from PIL import Image
import torch
from torchvision import transforms

from .model_cache import cached_model
from .weights import repo_entries

_MODEL_NAME = "briaai/RMBG-2.0"


def required_weights(arguments):
    """The one fixed repo remove_background loads, for the plan
    (dw/tasks/weights.py) - it takes no model argument."""
    return repo_entries(_MODEL_NAME)


def remove_background(image: Image, device) -> Image:
    # Model settings
    def load_model():
        # Imported here, not at module top: the plan's `required_weights`
        # lookup imports this module on every validate of a remove_background
        # step, and transformers is a second-plus import the server need not pay
        from transformers import AutoModelForImageSegmentation

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
