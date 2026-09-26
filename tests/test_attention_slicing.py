"""Attention slicing is opt-in on every backend.

It used to switch on automatically on MPS. Measured on an M5 Pro under
inference_mode, it made SDXL-sized UNet attention 2.4x slower in bf16
(12.8 vs 5.4 ms) and 1.4x slower in fp32 - PyTorch's SDPA is already the
memory-efficient path, which diffusers' own enable_attention_slicing
docstring warns about. Only UNet/ControlNet models take slicing at all.
"""

from dw.pipeline_processors.pipeline import attention_slicing_requested


def test_off_by_default():
    assert attention_slicing_requested({}) is False


def test_on_when_asked():
    assert attention_slicing_requested({"enable_attention_slicing": True}) is True


def test_the_old_opt_out_is_harmless():
    assert attention_slicing_requested({"disable_attention_slicing": True}) is False
