import torch
from huggingface_hub import get_token
import requests
import io


def remote_text_encoder(prompts, url, device):
    response = requests.post(
        url,
        json={"prompt": prompts},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json",
        },
    )
    prompt_embeds = torch.load(io.BytesIO(response.content))

    return prompt_embeds.to(device)
