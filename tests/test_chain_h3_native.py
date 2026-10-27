"""
A one-segment H3 chain is bit-identical to a native run (#1046).

tests/test_chain.py drives run_chain through a fake pipeline, so nothing there
checks that the chain path, given one segment, produces what the plain H3
pipeline produces. Here the real MiniMax-H3 modular pipeline runs on CPU from
tiny random-weight components built from diffusers' own classes (no download),
once natively and once through run_chain, on the same prompt and seed.
"""

import numpy
import pytest
import torch

from dw.pipeline_processors.chain import run_chain
from dw.pipeline_processors.pipeline import Pipeline

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

# H3's shortest video: the VAE's 17 * n + 5 frame grid, at least five seconds
NUM_FRAMES = 124
SEED = 1234
PROMPT = "a cat sat"


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen2TokenizerFast

    specials = ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    vocab = {"<unk>": 0}
    vocab.update({token: index + 1 for index, token in enumerate(specials)})
    for character in "abcdefghijklmnopqrstuvwxyz .,":
        vocab[character] = len(vocab)
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    return Qwen2TokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        additional_special_tokens=specials,
    )


def _text_encoder():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config=dict(
            hidden_size=16,
            intermediate_size=32,
            # H3 reads hidden_states[50], so the stack needs more than 50 layers
            num_hidden_layers=51,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=64,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=16,
            num_heads=2,
            out_hidden_size=16,
            deepstack_visual_indexes=[0],
            num_position_embeddings=16,
        ),
    )
    return Qwen3VLForConditionalGeneration(config)


def _tiny_h3():
    """The real H3 modular pipeline over tiny random weights."""
    from diffusers import (
        AutoencoderKLMiniMaxH3,
        AutoencoderKLMiniMaxH3Audio,
        MiniMaxH3Scheduler,
        MiniMaxH3Transformer3DModel,
    )
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
    from transformers import (
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    torch.manual_seed(0)
    tokenizer = _tokenizer()
    pipeline = MiniMaxH3Blocks().init_pipeline()
    pipeline.update_components(
        transformer=MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            hidden_size=32,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=32,
            text_dim=16,
            freq_dim=16,
            time_embed_hidden_dim=16,
            time_embed_dim=16,
            rope_freq_dim=2,
        ),
        vae=AutoencoderKLMiniMaxH3(
            block_out_channels=(8, 8, 8, 8, 8, 8),
            layers_per_block=1,
            norm_num_groups=4,
            decoder_num_layers=1,
            decoder_num_attention_heads=2,
            decoder_attention_head_dim=8,
        ),
        audio_vae=AutoencoderKLMiniMaxH3Audio(
            encoder_dim=4,
            latent_dim=64,
            num_attention_heads=2,
            # the decoder halves its width at each of seven upsamples
            decoder_dim=128,
            latents_mean=[0.0] * 32,
            latents_std=[1.0] * 32,
        ),
        text_encoder=_text_encoder(),
        tokenizer=tokenizer,
        processor=Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(),
            video_processor=Qwen3VLVideoProcessor(),
            tokenizer=tokenizer,
        ),
        scheduler=MiniMaxH3Scheduler(shift=12.0),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
    )
    return pipeline


@pytest.fixture(scope="module")
def stock_h3():
    return _tiny_h3()


@pytest.fixture(scope="module")
def guide_h3():
    """The same pipeline with the guide layout dw swaps in when it loads one."""
    from dw.pipeline_processors.h3_guides import insert_guides

    pipeline = _tiny_h3()
    assert insert_guides(pipeline)
    return pipeline


def _wrapper(tiny_h3, chain=None):
    definition = {"type": "MiniMaxH3Blocks", "configuration": {}}
    if chain is not None:
        definition["chain"] = chain
    return Pipeline(definition, SEED, "cpu", pipeline=tiny_h3)


def _arguments():
    return {
        "prompt": PROMPT,
        "height": 32,
        "width": 32,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": 2,
        "output": ["videos", "audio", "sampling_rate"],
        # a fresh generator per run: the seed is the whole of the randomness
        "generator": torch.Generator("cpu").manual_seed(SEED),
    }


def _frames(frames):
    return numpy.stack([numpy.asarray(frame) for frame in frames])


def _native(tiny_h3):
    output = _wrapper(tiny_h3).run(_arguments())
    return _frames(output["videos"][0]), output["audio"], output["sampling_rate"]


def _chained(tiny_h3, chain):
    output = run_chain(_wrapper(tiny_h3, chain), chain, _arguments())
    return _frames(output.frames), output.audio, output.sample_rate


@pytest.mark.parametrize(
    "chain, fixture",
    [
        pytest.param({"segments": 1, "fps": 24}, "stock_h3", id="default"),
        pytest.param(
            {"segments": 1, "fps": 24, "continuity": "guide"}, "guide_h3", id="guide"
        ),
    ],
)
def test_one_segment_chain_matches_native_run(chain, fixture, request):
    tiny_h3 = request.getfixturevalue(fixture)
    native_frames, native_audio, native_rate = _native(tiny_h3)
    chained_frames, chained_audio, chained_rate = _chained(tiny_h3, chain)

    assert chained_rate == native_rate
    assert chained_frames.shape == native_frames.shape
    assert torch.equal(
        torch.from_numpy(chained_frames), torch.from_numpy(native_frames)
    )

    native_audio = torch.as_tensor(native_audio)[0]
    chained_audio = torch.as_tensor(chained_audio)
    # The chain trims a segment's audio to its frames' own duration (codec
    # padding, #408), so it is the native track's prefix - sample for sample
    kept = round(NUM_FRAMES / 24 * native_rate)
    assert chained_audio.shape == (2, kept)
    assert native_audio.shape[1] >= kept
    assert torch.equal(chained_audio, native_audio[:, :kept])
