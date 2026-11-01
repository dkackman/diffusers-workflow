"""The vendored Audio8-TTS model module (#1120): a tiny random-config model
runs end to end on CPU, the processor builds its prompt inputs, and the
module loads only from local files - codec.pth with weights_only=True."""

import json
import re
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from dw.tasks import audio8_tts_model as a8

MODULE_PATH = Path(a8.__file__)

# semantic ids sit at the top of the vocabulary like the real model's;
# eos and pad are ordinary text ids below them
TINY = dict(
    vocab_size=64,
    dim=32,
    n_layer=2,
    n_head=4,
    n_local_heads=2,
    head_dim=8,
    intermediate_size=64,
    max_seq_len=128,
    codebook_size=16,
    num_codebooks=10,
    semantic_begin_id=40,
    semantic_end_id=55,
    n_fast_layer=1,
    fast_dim=32,
    fast_n_head=4,
    fast_n_local_heads=2,
    fast_head_dim=8,
    fast_intermediate_size=64,
    eos_token_id=2,
    pad_token_id=0,
    codec_encoder_dim=4,
    codec_encoder_transformer_layers=1,
    codec_latent_dim=64,
    codec_decoder_dim=32,
    codec_pre_n_layer=1,
    codec_pre_n_head=1,
    codec_pre_intermediate_size=64,
    codec_post_n_layer=1,
    codec_post_n_head=1,
    codec_post_n_local_heads=1,
    codec_post_intermediate_size=64,
    codec_residual_codebook_size=8,
    codec_codebook_dim=4,
)

LEGACY_PROJECTION = re.compile(r"quantizers\.\d+\.(in_proj|out_proj)\.weight$")


def tiny_config(**overrides):
    return a8.ArkttsConfig(**{**TINY, **overrides})


def legacy_codec_state(codec):
    """The codec's state as the checkpoint stores it: quantizer projections as
    legacy weight_norm g/v pairs, so loading goes through the fold."""
    state = {}
    for key, value in codec.state_dict().items():
        if LEGACY_PROJECTION.search(key):
            norm = value.norm(dim=(1, 2), keepdim=True)
            state[f"{key}_g"] = norm.clone()
            state[f"{key}_v"] = value.clone()
        else:
            state[key] = value.clone()
    return state


@pytest.fixture
def tiny_files(tmp_path):
    torch.manual_seed(0)
    config = tiny_config()
    model = a8.ArkttsModel(config)
    codec = a8.ArkttsCodec(config)
    from safetensors.torch import save_file

    weights = tmp_path / a8.WEIGHTS_FILE
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, weights)
    codec_path = tmp_path / a8.CODEC_FILE
    torch.save(legacy_codec_state(codec), codec_path)
    config_path = tmp_path / a8.CONFIG_FILE
    config_path.write_text(
        json.dumps({**TINY, "architectures": ["ArkttsModel"], "dtype": "bfloat16"})
    )
    generation_path = tmp_path / a8.GENERATION_CONFIG_FILE
    generation_path.write_text(json.dumps({"max_new_tokens": 4, "do_sample": False}))
    return {
        "model": model,
        "codec": codec,
        "config": config_path,
        "weights": weights,
        "codec_path": codec_path,
        "generation": generation_path,
    }


class NoEos:
    """A logits processor that keeps a random model from stopping at once."""

    def __init__(self, eos):
        self.eos = eos

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        scores[:, self.eos] = float("-inf")
        return scores


def text_prompt(config, length=6, batch=1):
    ids = torch.zeros((batch, config.num_codebooks + 1, length), dtype=torch.long)
    ids[:, 0] = torch.arange(3, 3 + length)
    return ids


def test_tiny_model_generates_and_decodes_on_cpu(tiny_files):
    model = a8.load_model(
        tiny_files["config"],
        tiny_files["weights"],
        tiny_files["codec_path"],
        tiny_files["generation"],
        device="cpu",
        dtype=torch.float32,
    )
    config = model.config
    generator = torch.Generator().manual_seed(1)
    codes = model.generate(
        input_ids=text_prompt(config),
        max_new_tokens=5,
        do_sample=True,
        logits_processor=[NoEos(config.eos_token_id)],
        generator=generator,
    )
    assert codes.shape == (1, config.num_codebooks, 5)
    assert int(codes.min()) >= 0
    assert int(codes[:, 0].max()) < config.codebook_size
    assert int(codes[:, 1:].max()) < config.codebook_size

    waveforms, lengths = model.decode_audio(codes)
    assert waveforms.dtype == torch.float32
    assert waveforms.shape == (1, 5 * a8.ArkttsCodec.frame_length)
    assert lengths.tolist() == [5 * a8.ArkttsCodec.frame_length]
    assert config.codec_sample_rate == 44100
    assert torch.isfinite(waveforms).all()
    # the KV caches do not outlive the call on a model kept between jobs
    assert all(layer.attention.kv_cache is None for layer in model.layers)


def test_generation_config_file_sets_the_defaults(tiny_files):
    model = a8.load_model(
        tiny_files["config"],
        tiny_files["weights"],
        tiny_files["codec_path"],
        tiny_files["generation"],
        dtype=torch.float32,
    )
    output = model.generate(
        input_ids=text_prompt(model.config),
        logits_processor=[NoEos(model.config.eos_token_id)],
        return_dict_in_generate=True,
    )
    # generation_config.json's max_new_tokens=4, greedy
    assert output.codes.shape[-1] == 4
    assert output.code_lengths.tolist() == [4]


def test_stopping_criteria_end_generation(tiny_files):
    model = a8.load_model(
        tiny_files["config"],
        tiny_files["weights"],
        tiny_files["codec_path"],
        dtype=torch.float32,
    )
    calls = []

    def stop_after_two(history, scores):
        calls.append(history.shape[-1])
        return len(calls) >= 2

    codes = model.generate(
        input_ids=text_prompt(model.config),
        max_new_tokens=8,
        do_sample=False,
        logits_processor=[NoEos(model.config.eos_token_id)],
        stopping_criteria=[stop_after_two],
    )
    assert codes.shape[-1] == 2


def test_load_model_matches_the_saved_weights(tiny_files):
    source = tiny_files["model"].eval()
    loaded = a8.load_model(
        tiny_files["config"],
        tiny_files["weights"],
        tiny_files["codec_path"],
        dtype=torch.float32,
    )
    ids = text_prompt(source.config)
    with torch.no_grad():
        expected = source(ids).logits
        actual = loaded(ids).logits
    assert torch.allclose(expected, actual, atol=1e-5)
    assert loaded.freqs_cis.device.type == "cpu"
    assert loaded.freqs_cis.shape[0] == source.config.max_seq_len


def test_codec_loads_with_weights_only(tiny_files, monkeypatch):
    seen = []
    real_load = torch.load

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(a8.torch, "load", spy)
    model = a8.ArkttsModel(tiny_config(), codec_path=tiny_files["codec_path"])
    model.load_codec()
    assert seen and all(call.get("weights_only") is True for call in seen)


def test_legacy_weight_norm_folds_to_the_same_projection():
    torch.manual_seed(0)
    legacy = torch.nn.utils.weight_norm(torch.nn.Conv1d(16, 4, kernel_size=1))
    with torch.no_grad():
        legacy.weight_g.mul_(torch.rand_like(legacy.weight_g) + 0.5)
    state = {f"in_proj.{k}": v for k, v in legacy.state_dict().items()}
    folded = a8._fold_legacy_weight_norm(state)
    assert set(folded) == {"in_proj.weight", "in_proj.bias"}
    plain = torch.nn.Conv1d(16, 4, kernel_size=1)
    plain.load_state_dict(
        {k.removeprefix("in_proj."): v for k, v in folded.items()}, strict=True
    )
    x = torch.randn(2, 16, 7)
    with torch.no_grad():
        assert torch.allclose(legacy(x), plain(x), atol=1e-6)


def test_codec_round_trip_keeps_the_frame_grid(tiny_files):
    model = a8.ArkttsModel(tiny_config(), codec_path=tiny_files["codec_path"])
    audio = torch.randn(1, 1, 3 * a8.ArkttsCodec.frame_length - 100) * 0.1
    codes, lengths = model.encode_audio(audio)
    assert codes.shape == (1, 10, 3)
    assert lengths.tolist() == [3]
    waveforms, _ = model.decode_audio(codes)
    assert waveforms.shape[-1] == 3 * a8.ArkttsCodec.frame_length


def test_default_config_is_the_released_checkpoint():
    config = a8.ArkttsConfig()
    with torch.device("meta"):
        codec = a8.ArkttsCodec(config)
    assert config.dim == 896 and config.n_layer == 24 and config.num_codebooks == 10
    assert codec.quantizer.semantic_quantizer.codebook_size == 4096
    assert codec.quantizer.quantizer.n_codebooks == 9
    assert codec.quantizer.quantizer.codebook_size == 1024
    assert codec.decoder.model[0].conv.in_channels == 1024


def test_module_has_no_hub_io():
    source = MODULE_PATH.read_text(encoding="utf-8")
    for name in ("from_pretrained", "hf_hub_download", "cached_file"):
        assert name not in source, name
    assert not re.search(r"^\s*(import|from)\s+(torchaudio|soundfile)\b", source, re.M)
    assert "weights_only=True" in source
    assert a8.REVISION == "1e857568bfd4f02931aca3988ee773a44d2d26d3"


# --- processor ---------------------------------------------------------------


class CharTokenizer:
    """One id per character - enough to see the prompt's layout."""

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return [ord(ch) % 1000 + 1 for ch in text]


def decode(ids):
    return "".join(chr(i - 1) for i in ids.tolist())


def make_processor():
    return a8.ArkttsProcessor.for_model(CharTokenizer(), a8.ArkttsConfig())


def test_processor_builds_a_text_only_prompt():
    data = make_processor()("  hello   world ")
    assert set(data) == {
        "prefix_input_ids",
        "prefix_attention_mask",
        "suffix_input_ids",
        "suffix_attention_mask",
    }
    prompt = decode(data["prefix_input_ids"][0])
    assert prompt == (
        "<|im_start|>system\nconvert the provided text to speech<|im_end|>\n"
        "<|im_start|>user\nhello world<|im_end|>\n<|im_start|>assistant\n<|voice|>"
    )
    assert data["suffix_input_ids"].shape == (1, 0)
    assert data["prefix_attention_mask"].all()


def test_processor_builds_a_reference_prompt_and_resamples():
    processor = make_processor()
    audio = np.sin(np.linspace(0, 200, 22050)).astype(np.float32)
    data = processor(
        "say this",
        reference_text="the reference",
        reference_audio=np.stack([audio, audio]),
        sampling_rate=22050,
    )
    prefix = decode(data["prefix_input_ids"][0])
    assert prefix.endswith("<|speaker:0|>the reference\n\nSpeech:\n")
    assert decode(data["suffix_input_ids"][0]).startswith("<|im_end|>\n")
    values = data["reference_audio_values"]
    assert values.shape[:2] == (1, 1)
    assert abs(values.shape[-1] - 44100) <= 64
    assert data["reference_audio_lengths"].tolist() == [values.shape[-1]]


def test_processor_requires_reference_text_with_audio():
    with pytest.raises(ValueError, match="reference_text"):
        make_processor()("hi", reference_audio=np.ones(100), sampling_rate=44100)


def test_processor_reads_a_local_path_and_refuses_a_url(tmp_path):
    path = tmp_path / "voice.wav"
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x10\x00" * 4410)
    data = make_processor()(
        "hi", reference_text="ref", reference_audio=str(path), base_dir=str(tmp_path)
    )
    assert data["reference_audio_lengths"].tolist() == [4410]
    with pytest.raises(ValueError, match="local file"):
        make_processor()(
            "hi", reference_text="ref", reference_audio="https://example.com/a.wav"
        )


def test_load_tokenizer_reads_a_local_tokenizer_json(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / a8.TOKENIZER_FILE
    tokenizer.save(str(path))
    loaded = a8.load_tokenizer(path)
    assert loaded.encode("hello world", add_special_tokens=False) == [1, 2]
    data = a8.ArkttsProcessor.for_model(loaded, a8.ArkttsConfig())("hello")
    assert data["prefix_input_ids"].dtype == torch.long
