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
    content_type = response.headers.get("Content-Type", "")
    # An endpoint that has moved or been retired answers with an HTML page,
    # and torch.load's unpickling error about it names nothing a reader
    # could act on
    if not response.ok or "text/html" in content_type:
        raise RuntimeError(
            f"The remote text encoder at {url} did not return embeddings "
            f"(HTTP {response.status_code}, {content_type or 'no content type'}). "
            "The endpoint may have moved or been retired; drop "
            "'remote_text_encoder' to load the text encoder locally."
        )
    prompt_embeds = torch.load(io.BytesIO(response.content))

    return prompt_embeds.to(device)
