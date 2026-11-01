"""
MMAudio - the decoder that turns the flow network's latents into a waveform:
diffusers' MMAudioVAE (latent -> mel spectrogram) and MMAudioVocoder (the
BigVGAN vocoder, mel -> 44.1 kHz waveform), v1.

The tensors are Kandinsky 6's: K6_REPO_ID's `audio_vae/` and `vocoder/`
folders at the full pinned K6_REVISION, each a config.json and a safetensors
file, all float32. They are MMAudio's VAE and BigVGAN, already in diffusers'
key layout, so there is no key map and no fuse here (safetensors only) - one
filter: K6's audio_vae file also carries an embedded copy of the vocoder
(667 `vocoder.`-prefixed tensors, from the 0.40 dev layout) that the 0.42
MMAudioVAE does not own, so those keys are dropped before the strict load
(the vocoder loads from its own folder). The diffusers classes need
diffusers >= 0.42.0 and are imported when a loader runs, so importing this
module never needs them.

Licences: the BigVGAN vocoder code and design are MIT (NVIDIA); the MMAudio
VAE tensors are CC-BY-NC 4.0 per MMAudio (K6's card states MIT) -
ATTRIBUTIONS.md.

load_decoder() reads only local paths its caller gives: a config.json read
as JSON and a safetensors file read with safetensors.torch.load_file. It
never touches the Hub.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

K6_REPO_ID = "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers"
K6_REVISION = "4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81"
VAE_FOLDER = "audio_vae"
VOCODER_FOLDER = "vocoder"
CONFIG_FILE = "config.json"
WEIGHTS_FILE = "diffusion_pytorch_model.safetensors"

# Prefix of the vocoder copy K6's audio_vae file carries but MMAudioVAE does not own.
_EMBEDDED_VOCODER = "vocoder."

__all__ = [
    "CONFIG_FILE",
    "K6_REPO_ID",
    "K6_REVISION",
    "VAE_FOLDER",
    "VOCODER_FOLDER",
    "WEIGHTS_FILE",
    "MMAudioDecoder",
    "decode_latents",
    "load_decoder",
]


def _init_kwargs(cls, config):
    """The config's entries cls.__init__ takes. K6's audio_vae config also
    names keys the 0.42 class does not (mode, downsample_factor, a nested
    vocoder_config), which are dropped rather than passed through."""
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return {key: value for key, value in config.items() if key in accepted}


def _load_component(cls, folder, drop_prefix=None):
    folder = Path(folder)
    config = json.loads((folder / CONFIG_FILE).read_text(encoding="utf-8"))
    state = load_file(str(folder / WEIGHTS_FILE), device="cpu")
    if drop_prefix:
        state = {k: v for k, v in state.items() if not k.startswith(drop_prefix)}
    with torch.device("meta"):
        model = cls(**_init_kwargs(cls, config))
    model.load_state_dict(state, strict=True, assign=True)
    return model


class MMAudioDecoder:
    """The VAE and vocoder as one latent -> waveform decode."""

    def __init__(self, vae, vocoder):
        self.vae = vae
        self.vocoder = vocoder

    @property
    def samples_per_latent(self):
        """Waveform samples per latent frame: the VAE's two-times-hop mel
        frames per latent, each one vocoder hop long (1024 at 44.1 kHz)."""
        return self.vae.config.hop_length * 2

    @property
    def sample_rate(self):
        return self.vae.config.sample_rate

    def to(self, device=None, dtype=None):
        self.vae.to(device=device, dtype=dtype)
        self.vocoder.to(device=device, dtype=dtype)
        return self

    def decode(self, latents):
        """`latents` (B, latent_channels, frames) as the flow network sampled
        them - no scaling_factor division; MMAudio's VAE takes them raw - to a
        waveform (B, 1, frames * samples_per_latent)."""
        return decode_latents(self.vae, self.vocoder, latents)


def decode_latents(vae, vocoder, latents):
    """Latents (B, C, F) -> mel (B, mel_bins, 2F) -> waveform (B, 1, 1024F)."""
    with torch.inference_mode():
        param = next(vae.parameters())
        latents = latents.to(device=param.device, dtype=param.dtype)
        mel = vae.decode(latents).sample
        waveform = vocoder(mel).sample
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(1)
    return waveform


def load_decoder(vae_folder, vocoder_folder, device="cpu", dtype=torch.float32):
    """Build MMAudioVAE and MMAudioVocoder from local K6 `audio_vae/` and
    `vocoder/` folders (config.json + safetensors each), strictly, in eval."""
    from diffusers import MMAudioVAE, MMAudioVocoder  # needs diffusers >= 0.42.0

    vae = _load_component(MMAudioVAE, vae_folder, drop_prefix=_EMBEDDED_VOCODER)
    vocoder = _load_component(MMAudioVocoder, vocoder_folder)
    for model in (vae, vocoder):
        model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return MMAudioDecoder(vae, vocoder)
