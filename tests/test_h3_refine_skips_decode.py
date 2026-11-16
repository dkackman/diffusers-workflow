"""An H3 refine whose `output` asks for no decoded media skips the decode (#1377).

The real H3 graph on tiny random weights (tests/test_chain_h3_native.py),
called through dw's `Pipeline` wrapper so `_call_pipeline` picks the route.
"""

import pytest
import torch

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

from dw.pipeline_processors.h3_hold import insert_audio_hold  # noqa: E402
from dw.pipeline_processors.h3_rules import REFINE_WINDOWS_INPUT, guide_latent_frames
from dw.pipeline_processors.h3_tiles import _is_decode, skips_decode
from dw.pipeline_processors.h3_windows import insert_refine_windows

NUM_FRAMES = 124
WINDOWS = {"frames": 12, "overlap": 4, "blend": "cosine", "positions": "true"}


@pytest.fixture(scope="module")
def tiny_h3():
    from tests.test_chain_h3_native import _tiny_h3

    pipeline = _tiny_h3()
    insert_audio_hold(pipeline)
    assert insert_refine_windows(pipeline)
    return pipeline


@pytest.fixture
def decode_calls(tiny_h3, monkeypatch):
    """The names of the top-level decode blocks as they run, in order."""
    calls = []
    for name, block in tiny_h3._blocks.sub_blocks.items():
        if not _is_decode(name):
            continue
        original = type(block).__call__

        def spy(self, pipe, state, _name=name, _original=original):
            calls.append(_name)
            return _original(self, pipe, state)

        monkeypatch.setattr(type(block), "__call__", spy)
    assert calls == []
    return calls


def refine_arguments(output, windows=WINDOWS):
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
    )

    track = torch.randn(
        2, NUM_FRAMES * 16000 // 24, generator=torch.Generator().manual_seed(4)
    )
    latents = torch.randn(
        1,
        24,
        guide_latent_frames(NUM_FRAMES),
        2,
        2,
        generator=torch.Generator().manual_seed(3),
    )
    arguments = {
        "prompt": "a cat sat",
        "height": 32,
        "width": 32,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": 2,
        "latents": latents,
        "hold_audio": MiniMaxH3AudioReference(audio=track * 0.1, sample_rate=16000),
        "refine_strength": 0.3,
        "generator": torch.Generator("cpu").manual_seed(1234),
        "output": output,
    }
    if windows is not None:
        arguments[REFINE_WINDOWS_INPUT] = windows
    return arguments


def run(tiny_h3, arguments):
    from tests.test_chain_h3_native import _wrapper

    return _wrapper(tiny_h3).run(arguments)


def latents_of(result):
    return result["latents"] if isinstance(result, dict) else result


# 1. The route, on the real graph


def test_latents_only_runs_no_decode_block(tiny_h3, decode_calls):
    blocks = tiny_h3._blocks
    result = run(tiny_h3, refine_arguments(["latents"]))
    assert decode_calls == []
    assert torch.isfinite(latents_of(result)).all()
    # The swap is undone
    assert tiny_h3._blocks is blocks


def test_asking_for_videos_runs_the_decode(tiny_h3, decode_calls):
    result = run(tiny_h3, refine_arguments(["videos", "latents"]))
    assert any(name.startswith("decode") for name in decode_calls)
    assert result["videos"] is not None


def test_the_latents_are_identical_either_way(tiny_h3):
    skipped = latents_of(run(tiny_h3, refine_arguments(["latents"])))
    decoded = latents_of(run(tiny_h3, refine_arguments(["videos", "latents"])))
    assert torch.equal(skipped, decoded)


def test_a_plain_refine_without_windows_skips_it_too(tiny_h3, decode_calls):
    skipped = latents_of(run(tiny_h3, refine_arguments(["latents"], windows=None)))
    assert decode_calls == []
    decoded = latents_of(
        run(tiny_h3, refine_arguments(["videos", "latents"], windows=None))
    )
    assert decode_calls
    assert torch.equal(skipped, decoded)


def test_asking_for_audio_still_decodes(tiny_h3, decode_calls):
    result = run(tiny_h3, refine_arguments(["latents", "audio", "sampling_rate"]))
    assert decode_calls
    assert result["audio"] is not None


# 2. The predicate


@pytest.mark.parametrize(
    "change, skips",
    [
        ({}, True),
        ({"output": "latents"}, True),
        ({"output": ["videos"]}, False),
        ({"output": ["latents", "audio"]}, False),
        ({"output": None}, False),
        ({"output": []}, False),
        ({"refine_strength": None}, False),
        ({"refine_tiles": {"width": 32, "height": 32, "overlap": 8}}, False),
    ],
)
def test_which_calls_skip_the_decode(tiny_h3, change, skips):
    arguments = {**refine_arguments(["latents"]), **change}
    assert skips_decode(tiny_h3, arguments) is skips


def test_a_pipeline_that_is_not_h3_never_skips():
    class NotH3:
        pass

    assert skips_decode(NotH3(), refine_arguments(["latents"])) is False
