"""RoPE asked for in float64 runs in float32 on MPS.

MPS has no float64. diffusers already falls back to float32 RoPE there for
Wan, Lumina2, SkyReels-V2, ChronoEdit and Sana-Video, but LTX-2's transformer
and text connectors build their frequencies in float64 whenever the module's
`double_precision` is set (the default) - so LTX-2.5 on a Mac loaded, then
failed on its first step with "Cannot convert a MPS Tensor to float64".
"""

import logging
from types import SimpleNamespace

import torch

from dw.pipeline_processors.pipeline import apply_mps_rope_precision


class Rope(torch.nn.Module):
    def __init__(self, double_precision=True):
        super().__init__()
        self.double_precision = double_precision


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rope = Rope()
        self.audio_rope = Rope()


def pipeline():
    return SimpleNamespace(
        components={
            "transformer": Transformer(),
            "connectors": Rope(),
            "scheduler": object(),
            "tokenizer": None,
        }
    )


def test_float64_rope_runs_in_float32_on_mps(caplog):
    p = pipeline()

    with caplog.at_level(logging.WARNING, logger="dw"):
        apply_mps_rope_precision(p, "mps")

    assert p.components["transformer"].rope.double_precision is False
    assert p.components["transformer"].audio_rope.double_precision is False
    assert p.components["connectors"].double_precision is False
    assert "float32" in caplog.text


def test_cuda_keeps_float64_rope():
    p = pipeline()

    apply_mps_rope_precision(p, "cuda")

    assert p.components["transformer"].rope.double_precision is True
    assert p.components["connectors"].double_precision is True


def test_a_pipeline_without_components_is_left_alone():
    apply_mps_rope_precision(object(), "mps")
