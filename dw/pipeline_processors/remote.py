import io
import logging
from urllib.parse import urlparse

import requests
import torch
from huggingface_hub import get_token

from ..locations import (
    HF_TOKEN_HOST_SUFFIXES,
    token_host_allowed,
    validate_remote_encoder_url,
)
from ..security import workflows_are_trusted

logger = logging.getLogger("dw")


def remote_text_encoder(prompts, url, device):
    url = validate_remote_encoder_url(url)
    headers = {"Content-Type": "application/json"}
    host = urlparse(url).hostname
    if token_host_allowed(host) or workflows_are_trusted():
        headers["Authorization"] = f"Bearer {get_token()}"
    else:
        logger.warning(
            f"Not sending the HuggingFace token to {host}: it is outside "
            f"{', '.join(HF_TOKEN_HOST_SUFFIXES)}. If the endpoint needs the "
            f"token, run with --trust-workflows."
        )

    response = requests.post(url, json={"prompt": prompts}, headers=headers)
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
