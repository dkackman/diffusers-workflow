"""MMAudio decoder loader (#1132): MMAudioVAE + MMAudioVocoder from K6-shaped
folders, strictly, from safetensors, with no Hub I/O or pickle."""

import json
import re
from pathlib import Path

import pytest
import torch

from dw.tasks import mmaudio_decoder as dec

diffusers = pytest.importorskip("diffusers")
needs_mmaudio = pytest.mark.skipif(
    not hasattr(diffusers, "MMAudioVAE"),
    reason="needs diffusers >= 0.42.0 (MMAudioVAE / MMAudioVocoder)",
)

SOURCE = Path(dec.__file__).read_text(encoding="utf-8")

# A tiny VAE / vocoder pair whose hops agree: the vocoder upsamples
# 4 * 4 * 4 = 64 = the VAE hop_length.
VAE_CONFIG = {
    "_class_name": "MMAudioVAE",
    "mel_bins": 16,
    "latent_channels": 4,
    "hidden_channels": 16,
    "channel_multipliers": [1, 2],
    "layers_per_block": 1,
    "sample_rate": 16000,
    "n_fft": 256,
    "hop_length": 64,
    "scaling_factor": 0.417,
    # keys K6's 0.40-dev config carries that the 0.42 class does not take
    "mode": "44k",
    "downsample_factor": 128,
    "vocoder_config": {"num_mels": 16},
}
VOCODER_CONFIG = {
    "_class_name": "MMAudioVocoder",
    "num_mels": 16,
    "upsample_initial_channel": 16,
    "upsample_rates": [4, 4, 4],
    "upsample_kernel_sizes": [8, 8, 8],
    "resblock_kernel_sizes": [3],
    "resblock_dilation_sizes": [[1, 3]],
}


def _randomize(module, seed):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
    return module


def _write(folder, cls, config, seed, extra=None):
    from safetensors.torch import save_file

    folder.mkdir(parents=True)
    (folder / dec.CONFIG_FILE).write_text(json.dumps(config))
    model = cls(**dec._init_kwargs(cls, config))
    _randomize(model, seed)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.update(extra or {})
    save_file(state, str(folder / dec.WEIGHTS_FILE))
    return model


@pytest.fixture
def k6_folders(tmp_path):
    vocoder = _write(
        tmp_path / dec.VOCODER_FOLDER, diffusers.MMAudioVocoder, VOCODER_CONFIG, 1
    )
    # K6's audio_vae file embeds a vocoder copy under `vocoder.`; the loader drops it.
    embedded = {f"vocoder.{k}": v.clone() for k, v in vocoder.state_dict().items()}
    vae = _write(
        tmp_path / dec.VAE_FOLDER, diffusers.MMAudioVAE, VAE_CONFIG, 2, embedded
    )
    return tmp_path, vae, vocoder


@needs_mmaudio
def test_loader_builds_both_classes_and_decodes_to_the_right_length(k6_folders):
    root, vae_ref, vocoder_ref = k6_folders
    decoder = dec.load_decoder(root / dec.VAE_FOLDER, root / dec.VOCODER_FOLDER)
    assert isinstance(decoder.vae, diffusers.MMAudioVAE)
    assert isinstance(decoder.vocoder, diffusers.MMAudioVocoder)
    assert not decoder.vae.training and not decoder.vocoder.training
    for loaded, ref in ((decoder.vae, vae_ref), (decoder.vocoder, vocoder_ref)):
        for (name, a), (_, b) in zip(
            loaded.state_dict().items(), ref.state_dict().items()
        ):
            assert torch.equal(a, b), name
    assert decoder.samples_per_latent == 2 * VAE_CONFIG["hop_length"]
    assert decoder.sample_rate == VAE_CONFIG["sample_rate"]

    latent_frames = 6
    z = torch.randn(2, VAE_CONFIG["latent_channels"], latent_frames)
    waveform = decoder.decode(z)
    assert waveform.shape == (2, 1, latent_frames * decoder.samples_per_latent)
    assert torch.isfinite(waveform).all()


@needs_mmaudio
def test_the_constructor_runs_off_meta(k6_folders, monkeypatch):
    """MMAudioVAE builds its mel filterbank in __init__. Under a meta device
    context that arithmetic mixed a CPU tensor with meta ones, which a
    long-lived worker's stricter fake kernels refused (#1201, on MPS). Only
    the parameters may be empty; the constructor's own tensors are real."""
    from diffusers.models.autoencoders import autoencoder_mmaudio

    root, _, _ = k6_folders
    devices = []
    original = autoencoder_mmaudio.mel_filterbank

    def recording(*args, **kwargs):
        devices.append(torch.empty(1).device.type)
        return original(*args, **kwargs)

    monkeypatch.setattr(autoencoder_mmaudio, "mel_filterbank", recording)
    decoder = dec.load_decoder(root / dec.VAE_FOLDER, root / dec.VOCODER_FOLDER)
    assert devices == ["cpu"]
    for model in (decoder.vae, decoder.vocoder):
        assert not any(t.is_meta for t in model.parameters())
        assert not any(t.is_meta for t in model.buffers())


@needs_mmaudio
def test_loads_while_transformers_has_torch_linspace_patched(k6_folders):
    """The worker's crash, reproduced: transformers' meta-init context swaps
    the process-global torch.linspace for one that defaults to CPU, and
    MMAudioVAE's filterbank then mixed a CPU linspace with meta tensors -
    'Tensor on device cpu is not on the expected device meta!' (#1201)."""
    initialization = pytest.importorskip("transformers.initialization")
    patch = getattr(initialization, "meta_device_safe_creation_ops", None)
    if patch is None:
        pytest.skip("this transformers does not patch torch.linspace")
    root, _, _ = k6_folders
    with patch():
        decoder = dec.load_decoder(root / dec.VAE_FOLDER, root / dec.VOCODER_FOLDER)
    assert not any(t.is_meta for t in decoder.vae.buffers())


@needs_mmaudio
def test_loader_is_strict(k6_folders, tmp_path):
    root, _, _ = k6_folders
    from safetensors.torch import load_file, save_file

    weights = root / dec.VOCODER_FOLDER / dec.WEIGHTS_FILE
    state = load_file(str(weights))
    state.pop(next(iter(state)))
    save_file(state, str(weights))
    with pytest.raises(RuntimeError, match="Missing key"):
        dec.load_decoder(root / dec.VAE_FOLDER, root / dec.VOCODER_FOLDER)


@needs_mmaudio
def test_init_kwargs_drops_keys_the_class_does_not_take():
    kwargs = dec._init_kwargs(diffusers.MMAudioVAE, VAE_CONFIG)
    assert "mode" not in kwargs and "vocoder_config" not in kwargs
    assert "_class_name" not in kwargs and kwargs["hop_length"] == 64


def test_module_rules():
    for banned in (
        "from_pretrained",
        "hf_hub_download",
        "cached_file",
        "snapshot_download",
        "torch.load(",
        "pickle",
        "save_file",
        "torch.save",
    ):
        assert banned not in SOURCE, banned
    assert not re.search(
        r"^\s*(import|from)\s+(open_clip|timm|einops|hydra|omegaconf)\b",
        SOURCE,
        re.M,
    )
    # diffusers is imported inside the loader, so the module imports without it
    assert not re.search(r"^(import|from)\s+diffusers\b", SOURCE, re.M)
    assert len(SOURCE.splitlines()) <= 1100


def test_weights_are_pinned():
    assert dec.K6_REPO_ID == "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers"
    assert re.fullmatch(r"[0-9a-f]{40}", dec.K6_REVISION)
    assert dec.K6_REVISION == "4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81"
    assert (dec.VAE_FOLDER, dec.VOCODER_FOLDER) == ("audio_vae", "vocoder")
    assert dec.WEIGHTS_FILE.endswith(".safetensors")


def test_module_is_imported_only_by_the_sound_effects_task():
    # stage D (#1133): generate_sound_effects is its one consumer
    dw_root = Path(dec.__file__).resolve().parents[1]
    for path in dw_root.rglob("*.py"):
        if path.name in ("mmaudio_decoder.py", "sound_effects.py"):
            continue
        assert "mmaudio_decoder" not in path.read_text(encoding="utf-8"), path
