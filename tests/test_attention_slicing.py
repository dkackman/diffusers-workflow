"""When a pipeline's attention runs sliced.

Automatic on MPS, opt-in elsewhere. Measured on an M5 Pro (torch 2.14), MPS
SDPA is slow at head dims 40, 48 and 160 and fast at 32 and 64-128: sliced
attention made SD 1.5 (head dims 40/80/160) ~20% faster end to end, while it
made SDXL-shaped attention (head dim 64) 2.4x slower. Automatic slicing stays
because turning it off regressed the catalog's quick-start template; a
head-dim-aware policy is the follow-up. Only UNet/ControlNet models take
slicing at all.
"""

from dw.pipeline_processors.pipeline import attention_slicing_requested


def test_automatic_on_mps():
    assert attention_slicing_requested({}, "mps") is True


def test_mps_can_opt_out():
    assert (
        attention_slicing_requested({"disable_attention_slicing": True}, "mps") is False
    )


def test_off_by_default_elsewhere():
    assert attention_slicing_requested({}, "cuda") is False
    assert attention_slicing_requested({}, "cpu") is False


def test_opt_in_anywhere():
    assert attention_slicing_requested({"enable_attention_slicing": True}, "cuda")
