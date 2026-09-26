# Mac (MPS) Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the catalog's CUDA-authored workflows load and run sanely on Apple Silicon, fix the Mac-specific defects the 2026-09-25 review verified, and fix the one cross-platform break it found (`segment`).

**Architecture:** Every fix is an engine-level translation at the point a CUDA-only setting is consumed (`dw/pipeline_processors/config_objects.py`, `dw/vram_estimate.py`, `dw/__init__.py`), in the same spirit as the existing `resolve_device()` and the sequential-to-model offload downgrade: the workflow JSON stays as written and the engine adapts it to the machine, warning when it does. No template is edited to become Mac-specific.

**Tech Stack:** Python 3.14, torch 2.14 (MPS), diffusers 0.41.dev, SDNQ, transformers (SAM2), pytest.

**Spec:** No separate spec. The source is the Mac review of 2026-09-25, summarized in *Background* below. Each claim there was verified on an M5 Pro with 64 GB (code read, or measured with `venv/bin/python`).

## Background (the review findings this plan implements)

| # | Finding (verified) | Task |
|---|---|---|
| 1 | SDNQ `quantization_device: "cuda"` passed through unresolved; SDNQ does `param.to("cuda")` during load → `AssertionError: Torch not compiled with CUDA enabled`. All 31 `ltx2/*` + `minimax/*` files. | 1 |
| 2 | SDNQ `use_quantized_matmul(_conv): true` routes through `torch._int_mm`, measured 1096 ms vs 2.1 ms bf16 on MPS (~500x). | 1 |
| 3 | `group_offload.use_stream` / `record_stream` passed through; diffusers raises `Using streams for data transfer requires a CUDA device` off CUDA/XPU, and `record_stream` without `use_stream` raises too. Same 31 files. | 2 |
| 4 | `dw/tasks/segment.py:109` reads `sam_inputs["reshaped_input_sizes"]`, a SAM v1 key the SAM2 processor never returns → KeyError on every backend; it also lands in the `mask_threshold` positional slot. The unit test mocks the processor. | 3 |
| 5 | `device_memory_stats()` reports zeros on MPS although `torch.mps.current_allocated_memory()`, `driver_allocated_memory()`, `recommended_max_memory()` all work. | 4 |
| 6 | `vram_estimate_errors` compares against every `cost` entry regardless of device; every entry is `cuda` 24 GB, so a 64 GB Mac is refused against an RTX 3090. Filtering to matching entries alone would silently *disable* the guard on a Mac (no `mps` entries exist). | 5 |
| 7 | Attention slicing is auto-enabled on MPS; measured under `inference_mode` it makes UNet attention 2.4x slower in bf16 (12.8 vs 5.4 ms) and 1.4x in fp32. Only UNet/ControlNet models have `set_attention_slice`. The fp16 warning text is missing a space and recommends fp32. | 6 |
| 8 | `dw/__init__.py:29` ignores every `UserWarning`; torch's "not currently supported on the MPS backend and will fall back to run on the CPU" is a `UserWarning`, and `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the `fp4-fp8-for-torch-mps` autoload (and Don's `~/.zprofile`). CPU fallbacks are therefore invisible. | 7 |
| 9 | Template defaults: `upscale-diffusion`/`upscale-spandrel` default input is `https://example.com/image.jpg` (not an image); `image-to-image` pulls its default from a third-party repo named `sage-permission-probe`; `flux-torchao.json:38` spells `Flux.1-Dev` (separate HF cache dir → ~23 GB duplicate download); `attention-processor`/`controlnet-component` use `runwayml/stable-diffusion-v1-5`; `dw/workflows/test.json` (the `dw.test` system test) runs SD 1.5 in float16. | 8 |
| 10 | Stale docs: CLAUDE.md "MPS: no autocast" (`torch.autocast("mps", bfloat16)` works on torch 2.14), CLAUDE.md "`music` keeps -1" (template is -3), `dw.test` described as an import check (it downloads SD 1.5 and generates), ACCELERATION.md MPS notes. | 6, 9 |

## Global Constraints

- Run tests with `venv/bin/python -m pytest` (torch is in the venv). Do **not** contact or deploy to lem, which is busy.
- Work on branch `feat/mac-mps-support` cut from `develop`. Commit after every task.
- The behavior on a CUDA box must be unchanged by every task. Each task has a test that pins this.
- Security rules in CLAUDE.md apply. No `shell=True`. Subprocess arguments are constant lists.
- Compare backends with `get_device_type()`, never `== "cuda"`. `cpu` is never rewritten.
- Every engine adaptation logs a `logger.warning` naming what it changed and why, as the sequential-offload downgrade in `place_component` does. There are no silent rewrites.
- Format touched Python with `venv/bin/ruff format <files>` before committing.
- Match the surrounding comment style: comments explain *why*, in full sentences, and reference the measurement or issue.

## Review Focus

1. **A step pinned to `cpu`:** SDNQ `quantization_device: "cpu"` stays `cpu`, a group offload whose onload device is `cpu` has its streams stripped without error, and quantized matmul is left on under DW_DEVICE=cpu. Tests are in Tasks 1 and 2.
2. **A CUDA box:** streams are kept, quantized matmul is kept, the device strings are unchanged, and the VRAM refusal is unchanged. Tests are in Tasks 1, 2 and 5.
3. **A cached pipeline reloaded, where the builder sees the same configuration dict twice:** stripping streams and resolving devices are idempotent. Tests are in Tasks 1 and 2.
4. **A `torch.mps` memory API that raises or is missing:** `device_memory_stats()` degrades to `None` figures and does not raise. Test is in Task 4.
5. **A Mac whose capacity can't be read:** the VRAM check falls back to the old all-entries behavior, which is conservative, rather than to no check. Test is in Task 5.

---

### Task 1: SDNQ quantization settings adapt to the machine

**Files:**
- Modify: `dw/pipeline_processors/config_objects.py` (imports at top; `create_quantization_config` at ~line 27)
- Test: `tests/test_config_objects.py` (class `TestQuantizationConfiguration`)

**Interfaces:**
- Consumes: `dw.resolve_device(requested)`, `dw.get_device_type(device=None)`
- Produces: `portable_quantization_arguments(arguments: dict) -> dict`, which is module-level in `config_objects.py` and returns a new dict without mutating its input. `create_quantization_config` calls it.

- [ ] **Step 1: Write the failing tests.** Append these to `class TestQuantizationConfiguration` in `tests/test_config_objects.py`:

```python
def test_a_cuda_quantization_device_is_translated_on_a_mac(self, monkeypatch):
    # A catalog template written on the CUDA box names "cuda" here, and SDNQ
    # moves every weight to it inside from_pretrained
    monkeypatch.setattr(dw, "backend_available", lambda backend: backend != "cuda")
    monkeypatch.setattr(dw, "get_device", lambda: "mps")

    config = create_quantization_config(
        quantization_definition(quantization_device="cuda", return_device="cpu")
    )

    assert config.kwargs["quantization_device"] == "mps"
    assert config.kwargs["return_device"] == "cpu"


def test_a_cpu_quantization_device_is_never_rewritten(self, monkeypatch):
    monkeypatch.setattr(dw, "backend_available", lambda backend: backend != "cuda")
    monkeypatch.setattr(dw, "get_device", lambda: "mps")

    config = create_quantization_config(
        quantization_definition(quantization_device="cpu")
    )

    assert config.kwargs["quantization_device"] == "cpu"


def test_quantized_matmul_is_switched_off_on_mps(self, monkeypatch):
    # Without Triton SDNQ's quantized matmul is torch._int_mm, measured
    # ~500x slower than the bf16 matmul it replaces on an M5 Pro
    monkeypatch.setattr(dw, "backend_available", lambda backend: backend != "cuda")
    monkeypatch.setattr(dw, "get_device", lambda: "mps")

    config = create_quantization_config(
        quantization_definition(
            use_quantized_matmul=True, use_quantized_matmul_conv=True
        )
    )

    assert config.kwargs["use_quantized_matmul"] is False
    assert config.kwargs["use_quantized_matmul_conv"] is False


def test_quantized_matmul_is_kept_on_cuda(self, monkeypatch):
    monkeypatch.setattr(dw, "backend_available", lambda backend: True)
    monkeypatch.setattr(dw, "get_device", lambda: "cuda")

    config = create_quantization_config(
        quantization_definition(quantization_device="cuda", use_quantized_matmul=True)
    )

    assert config.kwargs["quantization_device"] == "cuda"
    assert config.kwargs["use_quantized_matmul"] is True


def test_quantized_matmul_is_kept_on_cpu(self, monkeypatch):
    monkeypatch.setattr(dw, "get_device", lambda: "cpu")

    config = create_quantization_config(
        quantization_definition(use_quantized_matmul=True)
    )

    assert config.kwargs["use_quantized_matmul"] is True


def test_adapting_does_not_mutate_the_definition(self, monkeypatch):
    # A cached pipeline builds from the same definition a second time
    monkeypatch.setattr(dw, "backend_available", lambda backend: backend != "cuda")
    monkeypatch.setattr(dw, "get_device", lambda: "mps")
    definition = quantization_definition(
        quantization_device="cuda", use_quantized_matmul=True
    )

    create_quantization_config(definition)
    second = create_quantization_config(definition)

    assert definition["arguments"] == {
        "quantization_device": "cuda",
        "use_quantized_matmul": True,
    }
    assert second.kwargs["quantization_device"] == "mps"
```

- [ ] **Step 2: Run the tests to confirm they fail.**

Run: `venv/bin/python -m pytest tests/test_config_objects.py -k "quantization_device or quantized_matmul or mutate_the_definition" -v`
Expected: the Mac-translation, matmul-off and mutation tests FAIL, because `quantization_device` is still `"cuda"` and the matmul flags are still `True`. The CUDA and CPU keep-tests pass already.

- [ ] **Step 3: Implement.** In `dw/pipeline_processors/config_objects.py`, change the import line and add the helper above `create_quantization_config`:

```python
from .. import get_device_type, resolve_device
```

```python
# Quantization config arguments that name a device. A workflow authored on the
# CUDA box writes "cuda" here, and nothing downstream translates it - SDNQ moves
# every weight to it inside from_pretrained, which fails outright on a Mac
DEVICE_ARGUMENTS = ("quantization_device", "return_device")

# SDNQ's quantized matmul has no Triton on MPS and falls back to torch._int_mm,
# measured at ~500x the bf16 matmul it replaces (1096 ms vs 2.1 ms, 512x3072 by
# 3072x3072 on an M5 Pro). The dequantize path is the fast one there
QUANTIZED_MATMUL_ARGUMENTS = ("use_quantized_matmul", "use_quantized_matmul_conv")


def portable_quantization_arguments(arguments):
    """Adapt a quantization config's arguments to this machine without touching
    the definition they came from - a cached pipeline builds from it again."""
    adapted = dict(arguments)
    for key in DEVICE_ARGUMENTS:
        if isinstance(adapted.get(key), str):
            adapted[key] = resolve_device(adapted[key])
    if get_device_type() == "mps":
        for key in QUANTIZED_MATMUL_ARGUMENTS:
            if adapted.get(key):
                logger.warning(
                    f"Turning off '{key}' on MPS - without Triton it runs through "
                    "torch._int_mm, which is far slower there than dequantizing"
                )
                adapted[key] = False
    return adapted
```

In `create_quantization_config`, replace `quantization_config["arguments"].items()` in the dict comprehension with `portable_quantization_arguments(quantization_config["arguments"]).items()`.

- [ ] **Step 4: Run the tests to confirm they pass.**

Run: `venv/bin/python -m pytest tests/test_config_objects.py -v`
Expected: all PASS, including the pre-existing tests.

- [ ] **Step 5: Commit.**

```bash
git checkout -b feat/mac-mps-support develop   # first task only
venv/bin/ruff format dw/pipeline_processors/config_objects.py tests/test_config_objects.py
git add dw/pipeline_processors/config_objects.py tests/test_config_objects.py
git commit -m "fix(quantization): translate SDNQ devices and drop quantized matmul on MPS"
```

---

### Task 2: Group offload drops CUDA streams off CUDA

**Files:**
- Modify: `dw/pipeline_processors/config_objects.py` (`get_group_offload_configuration`, ~line 91)
- Test: `tests/test_config_objects.py` (class `TestGroupOffloadConfiguration`)

**Interfaces:**
- Consumes: the existing `get_group_offload_configuration(configuration, default_device)`
- Produces: the same signature. The returned dict has no `use_stream`/`record_stream` keys when its `onload_device.type` is not `cuda` or `xpu`.

- [ ] **Step 1: Fix the existing test that assumes CUDA.** On a Mac, `test_other_keys_are_carried_through` would now lose its `use_stream`. Pin CUDA as available by adding `monkeypatch` to its signature and this line as its first statement:

```python
        monkeypatch.setattr(dw, "backend_available", lambda backend: True)
```

- [ ] **Step 2: Write the failing tests.** Append these to `class TestGroupOffloadConfiguration`:

```python
def test_streams_are_dropped_on_mps(self, monkeypatch):
    # diffusers refuses use_stream without CUDA/XPU, and refuses
    # record_stream without use_stream - both have to go together
    monkeypatch.setattr(dw, "backend_available", lambda backend: backend != "cuda")
    monkeypatch.setattr(dw, "get_device", lambda: "mps")

    config = get_group_offload_configuration(
        {
            "group_offload": {
                "onload_device": "cuda",
                "num_blocks_per_group": 1,
                "use_stream": True,
                "record_stream": True,
            }
        },
        "cuda",
    )

    assert config["onload_device"] == torch.device("mps")
    assert "use_stream" not in config
    assert "record_stream" not in config
    assert config["num_blocks_per_group"] == 1


def test_streams_are_dropped_when_onloading_to_cpu(self):
    config = get_group_offload_configuration(
        {"group_offload": {"onload_device": "cpu", "use_stream": True}}, "cpu"
    )

    assert "use_stream" not in config


def test_streams_are_kept_on_cuda(self, monkeypatch):
    monkeypatch.setattr(dw, "backend_available", lambda backend: True)

    config = get_group_offload_configuration(
        {"group_offload": {"use_stream": True, "record_stream": True}}, "cuda"
    )

    assert config["use_stream"] is True
    assert config["record_stream"] is True


def test_dropping_streams_twice_is_stable(self, monkeypatch):
    monkeypatch.setattr(dw, "backend_available", lambda backend: backend != "cuda")
    monkeypatch.setattr(dw, "get_device", lambda: "mps")
    configuration = {"group_offload": {"use_stream": True, "record_stream": True}}

    get_group_offload_configuration(configuration, "cuda")
    second = get_group_offload_configuration(configuration, "cuda")

    assert "use_stream" not in second
    assert second["onload_device"] == torch.device("mps")
```

- [ ] **Step 3: Run the tests to confirm they fail.**

Run: `venv/bin/python -m pytest tests/test_config_objects.py::TestGroupOffloadConfiguration -v`
Expected: the three "dropped" tests FAIL, because `use_stream` is still present.

- [ ] **Step 4: Implement.** In `get_group_offload_configuration`, after the two `resolve_device` assignments and before `return group_offload_config`, add:

```python
        # CUDA streams overlap the next group's transfer with this one's compute.
        # diffusers refuses them without CUDA or XPU, and refuses record_stream
        # without use_stream, so both go together - a catalog template written
        # for the CUDA box would otherwise fail to load on a Mac
        if group_offload_config["onload_device"].type not in ("cuda", "xpu"):
            dropped = [
                key
                for key in ("use_stream", "record_stream")
                if group_offload_config.pop(key, False)
            ]
            if dropped:
                logger.warning(
                    f"Ignoring group offload {', '.join(dropped)} - streams need "
                    f"CUDA or XPU, and this onloads to "
                    f"{group_offload_config['onload_device']}"
                )
```

- [ ] **Step 5: Run the tests to confirm they pass.**

Run: `venv/bin/python -m pytest tests/test_config_objects.py tests/test_device_portability.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit.**

```bash
venv/bin/ruff format dw/pipeline_processors/config_objects.py tests/test_config_objects.py
git add dw/pipeline_processors/config_objects.py tests/test_config_objects.py
git commit -m "fix(offload): drop group-offload CUDA streams when onloading elsewhere"
```

---

### Task 3: `segment` uses SAM2's real post-processing contract

**Files:**
- Modify: `dw/tasks/segment.py:106-110`
- Test: `tests/test_segment.py`

**Interfaces:**
- Consumes: the `transformers.Sam2Processor.post_process_masks(masks, original_sizes, mask_threshold=0.0, binarize=True, ...)` signature, and `Sam2Processor` built offline from `Sam2ImageProcessor()`. Both were verified on the venv's transformers.
- Produces: `segment_image(...)` works with a real `Sam2Processor`.

- [ ] **Step 1: Write the failing contract test.** This test uses a **real** `Sam2Processor`, which needs no download, so a key or signature drift fails here rather than on a GPU box. Append to `tests/test_segment.py`:

```python
class TestSegmentWithRealSam2Processor:
    """The mocked tests above hand segment_image a dict with whatever keys
    they are told to - which is how a SAM v1 key ('reshaped_input_sizes')
    survived in the code while every real run raised KeyError. This one
    builds the real processor offline and fakes only the models."""

    @patch("dw.tasks.segment.Sam2Model")
    @patch("dw.tasks.segment.Sam2Processor")
    @patch("dw.tasks.segment.AutoModelForZeroShotObjectDetection")
    @patch("dw.tasks.segment.AutoProcessor")
    def test_masks_come_back_at_the_image_size(
        self, mock_auto_proc, mock_auto_model, mock_sam_proc, mock_sam_model
    ):
        from transformers import Sam2ImageProcessor
        from transformers import Sam2Processor as RealSam2Processor

        from dw.tasks.segment import segment_image

        dino = MagicMock()
        mock_auto_proc.from_pretrained.return_value = dino
        dino.return_value = _make_batch_encoding({"input_ids": torch.zeros(1, 10)})
        dino.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([[100.0, 100.0, 300.0, 300.0]]),
                "scores": torch.tensor([0.9]),
                "labels": ["dog"],
            }
        ]
        dino_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = dino_model
        dino_model.to.return_value = dino_model

        mock_sam_proc.from_pretrained.return_value = RealSam2Processor(
            image_processor=Sam2ImageProcessor()
        )
        sam_model = MagicMock()
        mock_sam_model.from_pretrained.return_value = sam_model
        sam_model.to.return_value = sam_model
        # SAM2's low-res mask logits: (batch, boxes, masks, 256, 256)
        logits = torch.full((1, 1, 3, 256, 256), -10.0)
        logits[..., 64:128, 64:128] = 10.0
        sam_model.return_value = MagicMock(pred_masks=logits)

        result = segment_image(_make_test_image(), "dog")

        assert result.mode == "L"
        assert result.size == (640, 480)
        assert np.array(result).max() == 255
```

- [ ] **Step 2: Run the test to confirm it fails.**

Run: `venv/bin/python -m pytest tests/test_segment.py::TestSegmentWithRealSam2Processor -v`
Expected: FAIL with `KeyError: 'reshaped_input_sizes'`.

- [ ] **Step 3: Implement.** In `dw/tasks/segment.py`, replace the `post_process_masks` call with:

```python
    # SAM2's processor takes the original sizes alone - the third positional
    # argument is mask_threshold, and the SAM v1 'reshaped_input_sizes' key this
    # used to pass does not exist in SAM2's inputs at all
    masks = sam_processor.post_process_masks(
        sam_outputs.pred_masks, sam_inputs["original_sizes"]
    )
```

- [ ] **Step 4: Update the mocked tests to the real contract.** In `tests/test_segment.py`, delete the `"reshaped_input_sizes": ...` entry from every `_make_batch_encoding({...})` SAM input dict, so the mocks can no longer offer a key the real processor lacks. Then in `test_returns_pil_image_mode_l`, after `result = segment_image(image, "dog")`, add:

```python
        args, kwargs = mock_sam_proc_instance.post_process_masks.call_args
        assert len(args) == 2 and not kwargs
```

- [ ] **Step 5: Run the tests to confirm they pass.**

Run: `venv/bin/python -m pytest tests/test_segment.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit.**

```bash
venv/bin/ruff format dw/tasks/segment.py tests/test_segment.py
git add dw/tasks/segment.py tests/test_segment.py
git commit -m "fix(segment): SAM2 post_process_masks takes original sizes only

Every real segment run raised KeyError on the SAM v1 'reshaped_input_sizes'
key; the mocked tests supplied it. Adds a test on the real processor."
```

---

### Task 4: Real memory figures on MPS, and a capacity helper

**Files:**
- Modify: `dw/__init__.py` (`device_memory_stats` ~line 290; add `_apple_chip_name` and `device_capacity_gb` after it; add `import functools` and `import subprocess` to the imports)
- Test: `tests/test_device_helpers.py` (replace `test_mps_reports_zeroed_known_stats`; add tests)

**Interfaces:**
- Produces:
  - `dw.device_capacity_gb(device=None) -> float | None`. On MPS this is `torch.mps.recommended_max_memory()` in GiB; on CUDA it is the card's total memory in GiB; otherwise it is `None`. It never raises.
  - On MPS, `dw.device_memory_stats()` returns `allocated_mb = current_allocated_memory`, `reserved_mb = driver_allocated_memory`, `total_mb = recommended_max_memory`, `free_mb = max(total - reserved, 0)`, and `device_name = "<chip> (MPS)"`. When the recommended maximum can't be read, `total_mb` and `free_mb` are `None`.
  - `dw._apple_chip_name() -> str` is cached and never raises.

Nothing downstream needs changing. The worker reads these in the process that holds the models (`dw/worker.py:702`). The UI meter (`ui/src/App.svelte:115`) shows once `total_mb` is non-zero. `_allocated_mb()` in `dw/workflow.py:251` starts reporting real figures.

- [ ] **Step 1: Write the failing tests.** In `tests/test_device_helpers.py`, replace `test_mps_reports_zeroed_known_stats` with:

```python
    def test_mps_reports_real_figures(self, monkeypatch):
        mb = 1024 * 1024
        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "mps")
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(dw, "_apple_chip_name", lambda: "Apple M5 Pro")
        monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 1000 * mb)
        monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 1500 * mb)
        monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 64000 * mb)

        stats = dw.device_memory_stats()

        assert stats == {
            "available": True,
            "device_name": "Apple M5 Pro (MPS)",
            "allocated_mb": 1000.0,
            "reserved_mb": 1500.0,
            "free_mb": 62500.0,
            "total_mb": 64000.0,
        }

    def test_mps_without_a_capacity_reading_reports_none(self, monkeypatch):
        def unavailable():
            raise RuntimeError("no Metal device")

        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "mps")
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(dw, "_apple_chip_name", lambda: "Apple M5 Pro")
        monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 0)
        monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 0)
        monkeypatch.setattr(torch.mps, "recommended_max_memory", unavailable)

        stats = dw.device_memory_stats()

        assert stats["available"] is True
        assert stats["total_mb"] is None
        assert stats["free_mb"] is None


class TestDeviceCapacity:
    def test_mps_capacity_is_the_recommended_max(self, monkeypatch):
        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "mps")
        monkeypatch.setattr(
            torch.mps, "recommended_max_memory", lambda: 62 * 1024**3
        )

        assert dw.device_capacity_gb() == 62.0

    def test_cpu_has_no_capacity(self, monkeypatch):
        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "cpu")

        assert dw.device_capacity_gb() is None

    def test_a_failing_probe_is_none_not_an_exception(self, monkeypatch):
        def broken():
            raise RuntimeError("boom")

        monkeypatch.setattr(dw, "get_device_type", lambda device=None: "mps")
        monkeypatch.setattr(torch.mps, "recommended_max_memory", broken)

        assert dw.device_capacity_gb() is None

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="needs an Apple Silicon GPU"
    )
    def test_this_mac_reports_a_capacity(self):
        assert dw.device_capacity_gb("mps") > 1.0
```

- [ ] **Step 2: Run the tests to confirm they fail.**

Run: `venv/bin/python -m pytest tests/test_device_helpers.py -k "mps or Capacity" -v`
Expected: FAIL. `_apple_chip_name` and `device_capacity_gb` don't exist yet, and the figures are zero.

- [ ] **Step 3: Implement.** In `dw/__init__.py`, add `import functools` and `import subprocess` beside `import os`. Replace the MPS branch of `device_memory_stats` (the `elif device_type == "mps" ...:` block) with:

```python
    elif (
        device_type == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        # Unified memory: 'reserved' is everything the Metal driver holds for
        # this process, and the ceiling is Metal's recommended working set -
        # the figure an allocation past which starts to page
        stats["available"] = True
        stats["device_name"] = f"{_apple_chip_name()} (MPS)"
        stats["allocated_mb"] = torch.mps.current_allocated_memory() / 1024 / 1024
        stats["reserved_mb"] = torch.mps.driver_allocated_memory() / 1024 / 1024
        try:
            total = torch.mps.recommended_max_memory() / 1024 / 1024
            stats["total_mb"] = total
            stats["free_mb"] = max(total - stats["reserved_mb"], 0.0)
        except (RuntimeError, AttributeError):
            pass
```

Update the docstring's MPS sentence to: "MPS reports allocated (tensors), reserved (everything the Metal driver holds) and total (Metal's recommended working set); free is total minus reserved." Then add after the function:

```python
@functools.lru_cache(maxsize=1)
def _apple_chip_name():
    """The chip's marketing name ('Apple M5 Pro'), so a reading from an 8 GB M1
    and one from a 128 GB M4 Max are not reported as the same device."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return result.stdout.strip() or "Apple Silicon"
    except (OSError, subprocess.SubprocessError):
        return "Apple Silicon"


def device_capacity_gb(device=None):
    """How much memory the accelerator can hold, in GiB - Metal's recommended
    working set on MPS, the card's total on CUDA - or None where there is no
    accelerator or the probe fails. Used where a check needs a ceiling for a
    device no curated 'cost' entry describes."""
    if not _TORCH_AVAILABLE:
        return None
    try:
        device_type = get_device_type(device)
        if device_type == "mps":
            return torch.mps.recommended_max_memory() / 1024**3
        if device_type == "cuda":
            index = torch.device(device or get_device()).index or 0
            return torch.cuda.get_device_properties(index).total_memory / 1024**3
    except (RuntimeError, AttributeError, AssertionError):
        return None
    return None
```

- [ ] **Step 4: Run the tests to confirm they pass.**

Run: `venv/bin/python -m pytest tests/test_device_helpers.py tests/test_device.py -v`
Expected: all PASS, and the real-Mac test passes on this box.

- [ ] **Step 5: Check the whole suite for callers that assumed zeros.**

Run: `venv/bin/python -m pytest -q -x -k "memory or device or observed" 2>&1 | tail -5`
Expected: PASS. If a test asserted `"Apple Silicon (MPS)"` or zeroed MPS figures, update it to the new contract. Those assertions pinned the old defect.

- [ ] **Step 6: Commit.**

```bash
venv/bin/ruff format dw/__init__.py tests/test_device_helpers.py
git add dw/__init__.py tests/test_device_helpers.py
git commit -m "feat(mps): report real unified-memory figures and the chip name"
```

---

### Task 5: The VRAM estimate checks against the device actually serving

**Files:**
- Modify: `dw/vram_estimate.py` (`vram_estimate_errors`, `apply_vram_estimate`)
- Modify: `dw/workflow.py` (the call at ~line 786 in validation, `apply_vram_estimate` at ~line 997, and the `from . import ...` line at ~line 94)
- Create: `tests/test_vram_estimate.py`

**Interfaces:**
- Consumes: `dw.get_device_type()` and `dw.device_capacity_gb()` from Task 4.
- Produces:
  - `vram_estimate_errors(definition, arguments=None, supplied=(), device_type=None, capacity_gb=None)`
  - `apply_vram_estimate(definition, variables, device_type=None, capacity_gb=None)`
  - Rule:
    1. With `device_type=None`, every entry is checked, as today.
    2. Otherwise, the entries whose `device` backend equals `device_type` are checked.
    3. If none match and `capacity_gb` is not None, a single synthesized entry `{"name": f"this {device_type} device (recommended maximum)", "vram_gb": round(capacity_gb, 1)}` is checked.
    4. If none match and capacity is unknown, every entry is checked. This is the conservative fallback.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_vram_estimate.py`:

```python
"""vram_estimate refuses a shape against the device actually serving it.

Every catalog 'cost' entry is a CUDA card, so checking all of them refused a
64 GB Mac against a 24 GB RTX 3090 - and filtering to matching entries alone
would have switched the guard off on a Mac entirely, since none match.
"""

from dw.vram_estimate import apply_vram_estimate, vram_estimate_errors
import pytest


def definition(extra_cost=()):
    # 768 * 512 * 121 voxels * 200 bytes = 8.86 GiB, + 20 base = 28.86 GiB
    return {
        "variables": {"width": 768, "height": 512, "num_frames": 121},
        "cost": [
            {"device": "cuda", "name": "RTX 3090", "vram_gb": 24, "minutes": 1.8},
            *extra_cost,
        ],
        "vram_estimate": {
            "voxel_variables": ["width", "height", "num_frames"],
            "base_gb": 20,
            "bytes_per_voxel": 200,
        },
    }


def test_no_device_checks_every_entry_as_before():
    errors = vram_estimate_errors(definition())
    assert len(errors) == 1 and "RTX 3090" in errors[0]["message"]


def test_a_cuda_box_checks_its_cuda_entries():
    errors = vram_estimate_errors(definition(), device_type="cuda", capacity_gb=24)
    assert len(errors) == 1 and "RTX 3090" in errors[0]["message"]


def test_a_mac_with_room_is_not_refused_by_a_cuda_card():
    assert vram_estimate_errors(definition(), device_type="mps", capacity_gb=62) == []


def test_a_mac_without_room_is_refused_against_its_own_capacity():
    errors = vram_estimate_errors(definition(), device_type="mps", capacity_gb=16)
    assert len(errors) == 1
    assert "mps" in errors[0]["message"]
    assert "RTX 3090" not in errors[0]["message"]


def test_a_mac_whose_capacity_is_unknown_keeps_the_conservative_check():
    errors = vram_estimate_errors(definition(), device_type="mps", capacity_gb=None)
    assert len(errors) == 1 and "RTX 3090" in errors[0]["message"]


def test_a_curated_mps_entry_wins_over_the_measured_capacity():
    mac = {"device": "mps", "name": "M5 Pro 64 GB", "vram_gb": 26, "minutes": 9}
    errors = vram_estimate_errors(definition([mac]), device_type="mps", capacity_gb=62)
    assert len(errors) == 1 and "M5 Pro 64 GB" in errors[0]["message"]


def test_an_indexed_cost_device_matches_its_backend():
    card = {"device": "cuda:1", "name": "second card", "vram_gb": 48, "minutes": 1}
    d = definition()
    d["cost"] = [card]
    assert vram_estimate_errors(d, device_type="cuda", capacity_gb=24) == []


def test_run_time_backstop_takes_the_device_too():
    d = definition()
    apply_vram_estimate(d, d["variables"], device_type="mps", capacity_gb=62)
    with pytest.raises(ValueError):
        apply_vram_estimate(d, d["variables"], device_type="mps", capacity_gb=16)
```

- [ ] **Step 2: Run the tests to confirm they fail.**

Run: `venv/bin/python -m pytest tests/test_vram_estimate.py -v`
Expected: FAIL with `TypeError: unexpected keyword argument 'device_type'`.

- [ ] **Step 3: Implement in `dw/vram_estimate.py`.** Add this helper above `vram_estimate_errors`:

```python
def _entries_for(cost, device_type, capacity_gb):
    """The cost entries a projection is checked against on this device.

    Entries measured on this backend first. A backend no entry describes - a
    Mac, against a catalog measured on CUDA cards - is checked against its own
    capacity when it can report one, and against every entry when it cannot,
    so an unreadable ceiling never turns the guard off."""
    entries = [entry for entry in cost if isinstance(entry, dict)]
    if device_type is None:
        return entries
    matching = [
        entry
        for entry in entries
        if str(entry.get("device", "")).split(":")[0] == device_type
    ]
    if matching:
        return matching
    if capacity_gb is not None:
        return [
            {
                "name": f"this {device_type} device (recommended maximum)",
                "vram_gb": round(capacity_gb, 1),
            }
        ]
    return entries
```

Change the signature to `def vram_estimate_errors(definition, arguments=None, supplied=(), device_type=None, capacity_gb=None):`. Replace the loop header and its `isinstance` guard:

```python
    for entry in cost:
        if not isinstance(entry, dict):
            continue
```

with:

```python
    for entry in _entries_for(cost, device_type, capacity_gb):
```

Change `apply_vram_estimate` to take and forward the two arguments:

```python
def apply_vram_estimate(definition, variables, device_type=None, capacity_gb=None):
    ...
    errors = vram_estimate_errors(
        definition, variables, device_type=device_type, capacity_gb=capacity_gb
    )
```

Append this to the module docstring: "The entries checked are the serving device's own (see `_entries_for`); `bytes_per_voxel` was calibrated on CUDA, so a check against a Mac's capacity is an estimate of an estimate."

- [ ] **Step 4: Wire the device in at both call sites in `dw/workflow.py`.** Extend the import at ~line 94 to `from . import get_device, empty_device_cache, device_memory_stats, device_capacity_gb, get_device_type`. At the validation call (~line 786):

```python
+vram_estimate_errors(
    self.workflow_definition,
    arguments,
    supplied=set(arguments or {}),
    device_type=get_device_type(),
    capacity_gb=device_capacity_gb(),
)
```

At the run-time call (~line 997):

```python
            apply_vram_estimate(
                workflow_def,
                variables,
                device_type=get_device_type(),
                capacity_gb=device_capacity_gb(),
            )
```

- [ ] **Step 5: Run the tests to confirm they pass.**

Run: `venv/bin/python -m pytest tests/test_vram_estimate.py tests/test_catalog_structure.py -q`
Then: `venv/bin/python -m pytest -q -k "vram or validat" 2>&1 | tail -5`
Expected: all PASS. A pre-existing test that expected an RTX 3090 refusal from `Workflow.validation_errors` while running on this Mac is now environment-dependent. Pin it with `monkeypatch.setattr("dw.workflow.get_device_type", lambda device=None: "cuda")` rather than deleting it.

- [ ] **Step 6: Check it on a real template.**

Run: `venv/bin/python -m dw.validate workflows/templates/ltx2/text-to-video.json`
Expected: valid on this 64 GB Mac.

- [ ] **Step 7: Commit.**

```bash
venv/bin/ruff format dw/vram_estimate.py dw/workflow.py tests/test_vram_estimate.py
git add dw/vram_estimate.py dw/workflow.py tests/test_vram_estimate.py
git commit -m "fix(vram_estimate): check against the serving device, not every CUDA card"
```

---

### Task 6: Attention slicing becomes opt-in, and the fp16 warning is corrected

**Files:**
- Modify: `dw/pipeline_processors/pipeline.py` (~lines 312-326 slicing block; ~line 1823 fp16 warning; add helper near `auto_cpu_offload_enabled` ~line 1509)
- Modify: `dw/workflow_schema.json:595-598` (`disable_attention_slicing` description)
- Modify: `docs/ACCELERATION.md:218`, `docs/WORKFLOW_GUIDE.md:1292-1293`
- Test: create `tests/test_attention_slicing.py`

**Interfaces:**
- Produces: `attention_slicing_requested(configuration: dict) -> bool` in `dw/pipeline_processors/pipeline.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_attention_slicing.py`:

```python
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
```

- [ ] **Step 2: Run the test to confirm it fails.**

Run: `venv/bin/python -m pytest tests/test_attention_slicing.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement.** Add this helper to `pipeline.py` beside `auto_cpu_offload_enabled`:

```python
def attention_slicing_requested(configuration):
    """Whether a pipeline's attention runs sliced. Opt-in on every backend: it
    used to be automatic on MPS, where it measured 2.4x slower on UNet attention
    than the SDPA it replaces (bf16, M5 Pro) - SDPA is already the
    memory-efficient path. 'disable_attention_slicing' is accepted and ignored."""
    return bool(configuration.get("enable_attention_slicing", False))
```

Replace the slicing block's condition and comments (~lines 312-317) with:

```python
            # Attention slicing trades speed for memory and is opt-in - see
            # attention_slicing_requested for why it is no longer automatic on MPS
            if attention_slicing_requested(self.configuration):
```

Keep the inner `has_method` branch as it is, but change its comment to "Modular pipelines have no attention slicing - skip rather than fail". Then replace the fp16 warning message (~line 1831) with:

```python
        logger.warning(
            f"{component_name} loads in float16 on MPS, which can produce NaN "
            "values (black images) on Apple Silicon - bfloat16 is the usual fix"
        )
```

- [ ] **Step 4: Update the schema and docs.**
  - In `dw/workflow_schema.json`, set the `disable_attention_slicing` description to: `"Deprecated and ignored: attention slicing is opt-in on every backend (enable_attention_slicing)."`
  - In `docs/ACCELERATION.md:218`, replace the sentence "Enabled automatically on MPS (unified memory benefits from slicing) unless `disable_attention_slicing` is set." with "Opt-in on every backend. It is not automatic on MPS, because PyTorch's SDPA is already memory-efficient and slicing measured 2.4x slower on UNet attention on Apple Silicon. It only affects UNet/ControlNet models."
  - In `docs/WORKFLOW_GUIDE.md:1292-1293`, replace "Enabled automatically on MPS unless `disable_attention_slicing` is set." with "Opt-in on every backend (UNet/ControlNet models only)."

- [ ] **Step 5: Run the tests and the schema check.**

Run: `venv/bin/python -m pytest tests/test_attention_slicing.py tests/test_catalog_structure.py -q && venv/bin/python -m dw.validate workflows/templates/text-to-image.json`
Expected: PASS, then valid.

- [ ] **Step 6: Commit.**

```bash
venv/bin/ruff format dw/pipeline_processors/pipeline.py tests/test_attention_slicing.py
git add dw/pipeline_processors/pipeline.py dw/workflow_schema.json docs/ACCELERATION.md docs/WORKFLOW_GUIDE.md tests/test_attention_slicing.py
git commit -m "perf(mps): make attention slicing opt-in; it slowed UNet attention 2.4x"
```

---

### Task 7: CPU fallbacks on MPS are no longer silent

**Files:**
- Modify: `dw/__init__.py:27-30` (the `warnings.filterwarnings` block)
- Test: create `tests/test_mps_fallback_warning.py`

**Interfaces:**
- Produces: after `import dw`, torch's MPS CPU-fallback `UserWarning` is shown, and every other `UserWarning` is still ignored.

- [ ] **Step 1: Write the failing test.** Use a fresh interpreter, because the filters are installed at import and pytest adds its own.

```python
"""torch announces an op the MPS backend lacks, when PYTORCH_ENABLE_MPS_FALLBACK
is set, with a UserWarning - and dw ignored every UserWarning, so a step that
quietly ran part of its model on the CPU left no trace. The fallback variable is
set by the fp4-fp8-for-torch-mps autoload install.sh adds on macOS."""

import subprocess
import sys

PROBE = """
import warnings, dw
fallback = ("The operator 'aten::_linalg_eigvals' is not currently supported "
            "on the MPS backend and will fall back to run on the CPU. "
            "This may have performance implications.")
with warnings.catch_warnings(record=True) as caught:
    warnings.warn(fallback, UserWarning)
    warnings.warn("some other library chatter", UserWarning)
print(len(caught), caught[0].message if caught else "")
"""


def test_only_the_fallback_warning_gets_through():
    result = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    count, _, message = result.stdout.strip().splitlines()[-1].partition(" ")
    assert count == "1"
    assert "fall back to run on the CPU" in message
```

- [ ] **Step 2: Run the test to confirm it fails.**

Run: `venv/bin/python -m pytest tests/test_mps_fallback_warning.py -v`
Expected: FAIL with `count == "0"`.

- [ ] **Step 3: Implement.** In `dw/__init__.py`, directly after the three `warnings.filterwarnings("ignore", ...)` lines, add:

```python
# ...except torch's notice that an op the MPS backend lacks ran on the CPU. With
# PYTORCH_ENABLE_MPS_FALLBACK set (the fp4-fp8-for-torch-mps autoload sets it)
# that is otherwise a step that got several times slower with nothing to say why.
# Later filters take precedence, so this one wins over the blanket ignore above
warnings.filterwarnings(
    "default",
    message=r".*not currently supported on the MPS backend and will fall back",
    category=UserWarning,
)
```

- [ ] **Step 4: Run the test to confirm it passes.**

Run: `venv/bin/python -m pytest tests/test_mps_fallback_warning.py -v`
Expected: PASS.

- [ ] **Step 5: Confirm it with a real op on this Mac.**

Run: `PYTORCH_ENABLE_MPS_FALLBACK=1 venv/bin/python -c "import dw, torch; torch.linalg.eigvals(torch.randn(4,4,device='mps'))" 2>&1 | grep -c "fall back"`
Expected: `1`.

- [ ] **Step 6: Commit.**

```bash
venv/bin/ruff format dw/__init__.py tests/test_mps_fallback_warning.py
git add dw/__init__.py tests/test_mps_fallback_warning.py
git commit -m "fix(mps): let torch's CPU-fallback warning through the UserWarning filter"
```

---

### Task 8: Template defaults that fail, download twice, or come from odd places

**Files:**
- Modify: `workflows/templates/upscale-diffusion.json:10`, `workflows/templates/upscale-spandrel.json:9`, `workflows/templates/image-to-image.json:5`, `workflows/models/flux-torchao.json:38`, `workflows/templates/attention-processor.json:24`, `workflows/templates/controlnet-component.json:37`, `dw/workflows/test.json:16`
- Create: `tests/test_template_defaults.py`

**Interfaces:** none (data only).

- [ ] **Step 1: Write the failing catalog test.** Create `tests/test_template_defaults.py`:

```python
"""A catalog default must run: no placeholder hosts, and one spelling per repo.

upscale-* defaulted to https://example.com/image.jpg (not an image), and
flux-torchao spelled FLUX.1-dev two ways - the Hub cache is keyed on the
spelling, so the transformer downloaded twice (~23 GB)."""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = sorted([*ROOT.glob("workflows/**/*.json"), *ROOT.glob("dw/workflows/*.json")])
PLACEHOLDER = re.compile(r"https?://(www\.)?example\.(com|org|net)/")
WITHDRAWN = ("runwayml/",)


def strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from strings(value)
    elif isinstance(node, str):
        yield node


def test_no_default_points_at_a_placeholder_host():
    offenders = [
        f"{path.relative_to(ROOT)}: {s}"
        for path in FILES
        for s in strings(json.loads(path.read_text()))
        if PLACEHOLDER.search(s)
    ]
    assert offenders == []


def test_no_withdrawn_repos():
    offenders = [
        f"{path.relative_to(ROOT)}: {s}"
        for path in FILES
        for s in strings(json.loads(path.read_text()))
        if s.startswith(WITHDRAWN)
    ]
    assert offenders == []


def test_one_spelling_per_repo_within_a_file():
    for path in FILES:
        repos = {
            s
            for s in strings(json.loads(path.read_text()))
            if re.fullmatch(r"[\w.-]+/[\w.-]+", s)
        }
        lowered = {}
        for repo in repos:
            lowered.setdefault(repo.lower(), set()).add(repo)
        clashes = [names for names in lowered.values() if len(names) > 1]
        assert clashes == [], f"{path.relative_to(ROOT)}: {clashes}"
```

- [ ] **Step 2: Run the test to confirm it fails.**

Run: `venv/bin/python -m pytest tests/test_template_defaults.py -v`
Expected: three FAILs naming the upscale templates, the two `runwayml/` files and `flux-torchao.json`. If the repo-spelling test flags anything else, treat it as a real finding and fix it the same way.

- [ ] **Step 3: Fix the data.**
  - `upscale-diffusion.json`, `upscale-spandrel.json` and `image-to-image.json`: set the default `location` to `https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png`. This was verified 200 `image/png` on 2026-09-25, and it's the image diffusers' own docs use. `image-to-image`'s previous default came from a third-party repo named `sage-permission-probe`.
  - `flux-torchao.json:38`: change `black-forest-labs/Flux.1-Dev` to `black-forest-labs/FLUX.1-dev`.
  - `attention-processor.json:24` and `controlnet-component.json:37`: change `runwayml/stable-diffusion-v1-5` to `stable-diffusion-v1-5/stable-diffusion-v1-5`.
  - `dw/workflows/test.json:16`: change `"torch.float16"` to `"torch.bfloat16"`. This is the system test, and the engine's own warning says fp16 is unsafe on MPS. SD 1.5 runs in bf16 on CUDA too.

- [ ] **Step 4: Run the tests and validate the touched files.**

Run: `venv/bin/python -m pytest tests/test_template_defaults.py tests/test_catalog_structure.py -q && for f in workflows/templates/upscale-diffusion.json workflows/templates/upscale-spandrel.json workflows/templates/image-to-image.json workflows/models/flux-torchao.json workflows/templates/attention-processor.json workflows/templates/controlnet-component.json; do venv/bin/python -m dw.validate $f || echo "FAIL $f"; done`
Expected: PASS, and no `FAIL` lines.

- [ ] **Step 5: Commit.**

```bash
venv/bin/ruff format tests/test_template_defaults.py
git add workflows dw/workflows/test.json tests/test_template_defaults.py
git commit -m "fix(templates): runnable default inputs, one repo spelling, no withdrawn repos"
```

---

### Task 9: Docs catch up with Mac behavior

**Files:**
- Modify: `CLAUDE.md` (lines 26-27 `dw.test` comment; line 324 MPS gotcha; line 576 `music` -1; the *Cross-Platform Device Support* section)
- Modify: `docs/ACCELERATION.md:343-352` (MPS Notes)
- Modify: `docs/DEPENDENCIES.md:58`
- Modify: `README.md:35`, `docs/TESTING.md:67` (the `dw.test` description)

**Interfaces:** none.

- [ ] **Step 1: CLAUDE.md.**
  - Replace the `# Basic system test (torch, diffusers import check)` comment with `# System test - downloads SD 1.5 (a few GB) and generates one image`.
  - In the MPS gotcha at line 324, change "no autocast, no bitsandbytes, no flash_attn, no triton, no torch.compile" to "no bitsandbytes, no flash_attn, no triton, no torch.compile (`torch.autocast("mps")` works on torch 2.14, but dw does not use it)".
  - At line 576, change "`music`, an mp3, keeps -1" to "`music`, an mp3, normalizes to -3 as well (#362)".
  - In *Cross-Platform Device Support*, after the `resolve_device()` paragraph, add: "The same translation reaches the settings that carry a device or a CUDA-only feature: SDNQ `quantization_device`/`return_device` go through `resolve_device()`, and `use_quantized_matmul(_conv)` is turned off on MPS, because it falls back to `torch._int_mm`, ~500x slower there (`portable_quantization_arguments`, `config_objects.py`). Group-offload `use_stream`/`record_stream` are dropped when the onload device is not CUDA/XPU. `vram_estimate` checks against the serving device's own `cost` entries, else its `device_capacity_gb()`. Attention slicing is opt-in everywhere. Each adaptation logs a warning."
- [ ] **Step 2: ACCELERATION.md MPS Notes.** Replace the bullet list with:

```markdown
- No flash-attn, no Triton, no bitsandbytes - `attention_backend` is effectively CUDA-only; use `"native"`-family backends or leave it unset on MPS. `compile` is skipped with a warning (inductor support on MPS is immature). A workflow that loads a bitsandbytes checkpoint (`flux2-dev`, `multi-image-reference`) or a TorchAO int4 config (`flux-torchao`) has no Mac path yet.
- Workflows written for CUDA are adapted rather than refused: a `cuda` device becomes `mps`, SDNQ's `quantization_device` follows it and its quantized matmul is switched off (it runs through `torch._int_mm`, ~500x slower on MPS), group-offload CUDA streams are dropped, and `"offload": "sequential"` becomes `"model"`. Each change is logged as a warning.
- Attention slicing is opt-in; it made UNet attention 2.4x slower on Apple Silicon.
- `float16` can produce NaN values on Apple Silicon - use `bfloat16`; dw only warns, it doesn't override the dtype for you.
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO` defaults to `0.0` (use all unified memory) unless already set in the environment.
- Ops the MPS backend lacks: with `PYTORCH_ENABLE_MPS_FALLBACK=1` (set by the `fp4-fp8-for-torch-mps` package install.sh adds) they run on the CPU, and dw logs torch's warning when one does. Without it they raise.
- Memory figures are real: allocated, driver-reserved, and Metal's recommended working set as the total. `vram_estimate` checks a Mac against that total, since the catalog's `cost` entries are CUDA cards; the per-voxel figures were calibrated on CUDA.
- Cost estimates quoted on a Mac are the CUDA figures (`basis: "other_device"`) until the Mac's own runs build observed history - expect them to be optimistic.
- Offloading has less benefit than on CUDA, since unified memory is already shared between CPU and GPU.
```

- [ ] **Step 3: DEPENDENCIES.md:58.** Replace the macOS line with: `**macOS (MPS):** fp4-fp8-for-torch-mps (FP8/FP4 dtypes for Metal; autoloads on every torch import, overrides mm/linear/copy on the MPS dispatch key and sets PYTORCH_ENABLE_MPS_FALLBACK=1), fluidtop (a GPU/power TUI; needs sudo)`
- [ ] **Step 4: README.md:35 and docs/TESTING.md:67.** Rewrite the `dw.test` description to say that it runs `dw/workflows/test.json`, which downloads SD 1.5 and generates one image, so it exercises the whole stack rather than only the imports. Keep the surrounding sentence structure.
- [ ] **Step 5: Check.**

Run: `grep -n "import check\|keeps -1\|no autocast\|on by default (set .disable_attention_slicing" CLAUDE.md README.md docs/*.md`
Expected: no output.

- [ ] **Step 6: Commit.**

```bash
git add CLAUDE.md README.md docs/ACCELERATION.md docs/DEPENDENCIES.md docs/TESTING.md
git commit -m "docs: Mac behavior, dw.test's real scope, stale autocast and music lines"
```

---

### Task 10: Verify on this Mac (manual; downloads need Don's go-ahead)

**Files:** none. Record the results in the PR description.

- [ ] **Step 1: Run the full suite.**

Run: `venv/bin/python -m pytest -q 2>&1 | tail -5`
Expected: no failures beyond those already failing on `develop`. Check by running the same command on `develop` if any appear.

- [ ] **Step 2: Run the quick-start template, which covers UNet and the slicing change.**

Run: `time venv/bin/python -m dw.run workflows/templates/text-to-image.json`
Expected: an image written under `outputs/`, with no attention-slicing debug line. Record the wall time.

- [ ] **Step 3: Run the segment template (a download of about 1 GB).**

Run: `venv/bin/python -m dw.run workflows/templates/segment.json`
Expected: a mask written. It previously raised KeyError.

- [ ] **Step 4: Ask Don before this step.** LTX-2.5 text-to-video on a Mac is a download of tens of GB and a long run.

Run: `venv/bin/python -m dw.run workflows/templates/ltx2/text-to-video.json`
Expected: the log shows the three adaptation warnings (quantization device, quantized matmul, streams), and then either a video or a real memory or speed limit. Either result is the finding. Record peak `driver_allocated_memory` from the job's memory events, and the wall time.

- [ ] **Step 5: Open a PR against `develop`.** Its body summarizes Tasks 1-9 and pastes the Step 2-4 results. It lists the out-of-scope items below as follow-ups.

---

## Out of scope (follow-ups, each needs a measurement first)

- The `fp4-fp8-for-torch-mps` autoload is a global op override that nothing in the catalog uses. Decide whether to keep it on by default.
- Downgrade group offload and `residency: on_demand` on MPS the way sequential offload is downgraded. This needs a peak-memory A/B first.
- Add a GGUF or SDNQ path for `flux2-dev`, `multi-image-reference`, `community-pipeline` and `flux-torchao`.
- The fp16 defaults in `attention-processor`, `base-and-refiner`, `controlnet-component`, `depth-marigold`, `qr-code` and `surface-normals`. Also the fp16-on-MPS choice in `dw/tasks/upscale.py` and `restore_faces.py` (reviewer-reported, not reproduced).
- A `generator_device: "cpu"` option so a seed reproduces across CUDA and Mac.
- Task-model reloads per call in `restore_faces` and `image_utils` (reviewer-reported).
- `ltx-2.5` / `minimax-music3` skill wording about the Mac, once Task 10 Step 4 says what actually fits.
- The desktop-installers branch lacks torchaudio.
