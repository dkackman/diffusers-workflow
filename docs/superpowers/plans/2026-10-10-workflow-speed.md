# Workflow Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut wall time per job without touching model quality: stop reloading warm models when nothing about their weights changed, stop re-building per-frame helpers, decode video with all cores, and fix catalog defaults that multiply step counts for nothing.

**Architecture:** Three engine changes and one catalog change. (1) The pipeline cache key becomes a *weights identity*: LoRA scale/alpha and scheduler shift leave the hash, and a cache hit re-applies them with the in-place setters the engine already has. (2) A workflow switch in the worker evicts only pipelines the next workflow will not load, instead of everything. (3) Two task-level sinks (face-restore helper rebuilt per frame, single-threaded PyAV decode) are fixed in place. (4) Stale catalog defaults are corrected and pinned by a test.

**Tech Stack:** Python 3, pytest (`venv/bin/python -m pytest`, xdist `-n auto` in pytest.ini; use `-n0` when naming files), ruff, PyAV, diffusers, peft, facexlib.

**Spec:** The findings write-up in this session (speed audit of 2026-10-10). The ranked list: (11) LoRA scale/shift reloads, (7) workflow switch unloads all, (12) `release_pipeline` on a last step, (1) stale step counts, (6) stale first_block cache on turbo ref2va, (9) per-frame task sinks, (8/9) PyAV decode threads. Items needing a GPU measurement are listed under *Out of scope* at the end and are not tasks.

## Global Constraints

- Branch: `feat/workflow-speed` off `develop`. Never commit to `develop` or `master`.
- Gate for every task: `venv/bin/python -m pytest <named test files> -n0 -q` green, then `venv/bin/ruff format --check dw dw_mcp tests scripts && venv/bin/ruff check dw dw_mcp tests scripts` clean. Run the full suite (`venv/bin/python -m pytest tests/ -q`) once at the end of the branch.
- No `eval()`, `exec()`, `shell=True`. No new `av.open` outside `dw/media.py` (`tests/test_media_layering.py` enforces it).
- Model knowledge lives in the catalog and `plugins/dw/`, never in engine code.
- Every `av.open` that decodes picture frames sets `stream.thread_type = "AUTO"`; none that only reads headers does.
- Docs that describe a changed behaviour change in the same task (`docs/WORKFLOW_GUIDE.md`, `docs/ACCELERATION.md`, `docs/WORKER_GUIDE.md`).
- `docs/ARCHITECTURE.md` has a module count ratchet (`tests/test_arch_metrics.py`). No task adds a module, so the count must not change.

## Review Focus

1. A LoRA whose `model_name` changes between runs must still reload (only scale and alpha are runtime-mutable). Test in Task 4.
2. A LoRA nulled by a variable (`model_name: null`) keeps its *index* for adapter naming, so re-applying scales on a hit must skip nulled entries while still naming the survivors by their original index. Test in Task 4.
3. A cache hit for a pipeline with no `loras` and no `scheduler.shift` must apply nothing and call nothing on the model. Test in Task 4.
4. A workflow switch whose new definition fails to prepare (bad variable, missing asset) must fall back to the full cleanup, never leave the old models resident silently. Test in Task 5.
5. Face restore on a video must produce the same per-frame result with a cached helper as with a fresh one, so the helper's per-image state must be cleared before each frame. Test in Task 2.

---

### Task 1: Catalog defaults that multiply cost

**Files:**
- Modify: `workflows/templates/compose-workflows.json:10`
- Modify: `workflows/templates/prompt-weighting.json:24-25`
- Modify: `workflows/templates/minimax/shots-batch.json:100`
- Modify: `workflows/templates/minimax/reference-to-video.json`, `chain-matched-to-audio.json`, `chain-video-continuity.json`, `composable-references.json`, `voice-timbre-reference.json`, `generated-subject-reference.json` (remove the `cache` block from the ModularPipeline step's `configuration`)
- Create: `tests/test_catalog_speed_defaults.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a test that pins the three rules below for every template under `workflows/templates/`.

Rules the test pins:
1. A minimax template whose `lora_weight_name` default contains `turbo` and whose `num_inference_steps` default is at most 9 carries no `cache` block (RECIPES_24GB.md: first_block never skips on the turbo schedules).
2. `compose-workflows.json`'s `video_num_inference_steps` default equals the `num_inference_steps` default of `minimax/image-to-video.json` (the turbo LoRA's passes plus one).
3. Any step whose `model_name` contains `FLUX.1-schnell` runs at most 4 steps and guidance 0 (schnell is 4-step distilled; its guidance input is unused).

- [ ] **Step 1: Write the failing test**

```python
"""Catalog defaults that silently multiply a run's cost (speed audit 2026-10-10)."""

import glob
import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = sorted(
    glob.glob(os.path.join(REPO_ROOT, "workflows", "templates", "**", "*.json"), recursive=True)
)


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _default(definition, name):
    variable = definition.get("variables", {}).get(name)
    return variable.get("default") if isinstance(variable, dict) else variable


def _pipeline_steps(definition):
    for step in definition.get("steps", []):
        if "pipeline" in step:
            yield step


MINIMAX = [p for p in TEMPLATES if os.sep + "minimax" + os.sep in p]


@pytest.mark.parametrize("path", MINIMAX, ids=os.path.basename)
def test_turbo_schedules_carry_no_block_cache(path):
    definition = _load(path)
    weight = _default(definition, "lora_weight_name") or ""
    steps = _default(definition, "num_inference_steps")
    if "turbo" not in weight or steps is None or steps > 9:
        pytest.skip("not a turbo schedule")
    for step in _pipeline_steps(definition):
        configuration = step["pipeline"].get("configuration", {})
        assert "cache" not in configuration, (
            f"{os.path.basename(path)} step '{step['name']}' carries a cache block on a "
            f"{steps}-step turbo schedule; RECIPES_24GB.md says it never skips there"
        )


def test_compose_workflows_video_steps_match_the_turbo_default():
    compose = _load(os.path.join(REPO_ROOT, "workflows", "templates", "compose-workflows.json"))
    i2v = _load(
        os.path.join(REPO_ROOT, "workflows", "templates", "minimax", "image-to-video.json")
    )
    assert _default(compose, "video_num_inference_steps") == _default(
        i2v, "num_inference_steps"
    )


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: os.path.relpath(p, REPO_ROOT))
def test_schnell_runs_its_distilled_schedule(path):
    definition = _load(path)
    for step in _pipeline_steps(definition):
        pipeline = step["pipeline"]
        model = pipeline.get("from_pretrained_arguments", {}).get("model_name", "")
        if "FLUX.1-schnell" not in str(model):
            continue
        arguments = pipeline.get("arguments", {})
        assert arguments.get("num_inference_steps", 4) <= 4, path
        assert arguments.get("guidance_scale", 0.0) == 0.0, path
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_catalog_speed_defaults.py -n0 -q`
Expected: failures for the six ref2va templates (cache block present), compose-workflows (20 != 5), prompt-weighting (20 steps, guidance 4.0).

- [ ] **Step 3: Fix the templates**

- `compose-workflows.json:10`: `"video_num_inference_steps": 20` -> `5`.
- `prompt-weighting.json:24-25`: `"num_inference_steps": 20` -> `4`, `"guidance_scale": 4.0` -> `0.0`.
- In each of the six ref2va templates, delete the `"cache": { "type": "first_block", "threshold": 0.1 }` block from the ModularPipeline step's `configuration` (watch the trailing comma on the previous key). Do not touch `image-to-video.json`; its description already explains the absence.
- `shots-batch.json:100`: delete `"release_pipeline": true,`. It is the only step, so the release frees nothing within the run and only forces the next job to reload H3; Task 5 makes the worker drop it when a different workflow needs the card.
- If a template's `description` string mentions the cache or the release you removed, edit that sentence too (grep `first_block` and `release` in the changed files).

- [ ] **Step 4: Run the new test and the catalog tests**

Run: `venv/bin/python -m pytest tests/test_catalog_speed_defaults.py tests/test_catalog_structure.py tests/test_catalog_shape.py tests/test_h3_schedule.py tests/test_plugin_skills.py -n0 -q`
Expected: PASS. If `test_catalog_structure` complains about a description that no longer matches the derived shape or a `cost` entry, fix the description; do not change the `cost` minutes (they are measured values, now upper bounds).

- [ ] **Step 5: Validate the edited JSON against the schema**

Run: `for f in workflows/templates/compose-workflows.json workflows/templates/prompt-weighting.json workflows/templates/minimax/shots-batch.json workflows/templates/minimax/reference-to-video.json workflows/templates/minimax/chain-matched-to-audio.json workflows/templates/minimax/chain-video-continuity.json workflows/templates/minimax/composable-references.json workflows/templates/minimax/voice-timbre-reference.json workflows/templates/minimax/generated-subject-reference.json; do venv/bin/python -m dw.validate "$f" || echo "FAILED $f"; done`
Expected: no `FAILED` line.

- [ ] **Step 6: Ruff and commit**

```bash
venv/bin/ruff format --check dw dw_mcp tests scripts && venv/bin/ruff check dw dw_mcp tests scripts
git add workflows/templates tests/test_catalog_speed_defaults.py
git commit -m "fix(catalog): drop stale step counts, dead turbo caches and a no-op release"
```

---

### Task 2: Face restore keeps its detector between frames

**Files:**
- Modify: `dw/tasks/restore_faces.py:70-80`
- Test: `tests/test_restore_faces_helper_cache.py` (create)

**Interfaces:**
- Consumes: `dw.tasks.model_cache.cached_model(key, factory)`.
- Produces: nothing new; `restore_faces(image, model_name, device, **kwargs)` keeps its signature.

Background: `FaceRestoreHelper(...)` loads RetinaFace (and ParseNet when `use_parse`) in its constructor. `per_frame` calls `restore_faces` once per frame, so a video reloads detector weights per frame. The helper keeps per-image state (`all_landmarks_5`, `cropped_faces`, `restored_faces`, affine matrices, `det_faces`, `pad_input_imgs`) that `clean_all()` resets.

- [ ] **Step 1: Write the failing test**

```python
"""restore_faces builds its facexlib helper once per (settings, device), not per frame."""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from dw.tasks import model_cache


@pytest.fixture
def fake_facexlib(monkeypatch):
    """A facexlib whose helper records constructions and detects no faces."""
    constructed = []

    class FakeHelper:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.cleaned = 0

        def clean_all(self):
            self.cleaned += 1

        def read_image(self, image):
            self.image = image

        def get_face_landmarks_5(self, **kwargs):
            return 0

    module = types.ModuleType("facexlib.utils.face_restoration_helper")
    module.FaceRestoreHelper = FakeHelper
    package = types.ModuleType("facexlib")
    utils = types.ModuleType("facexlib.utils")
    monkeypatch.setitem(sys.modules, "facexlib", package)
    monkeypatch.setitem(sys.modules, "facexlib.utils", utils)
    monkeypatch.setitem(sys.modules, "facexlib.utils.face_restoration_helper", module)
    model_cache.clear_model_cache()
    yield constructed
    model_cache.clear_model_cache()


def test_two_frames_share_one_helper(fake_facexlib, monkeypatch):
    from dw.tasks import restore_faces as module

    descriptor = MagicMock(supports_half=False)
    monkeypatch.setattr(module, "_load_face_model", lambda path, device: descriptor)
    monkeypatch.setattr(
        "dw.tasks.upscale.resolve_model_path", lambda model_name, filename: "weights.pth"
    )
    frame = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

    module.restore_faces(frame, "m", device="cpu")
    module.restore_faces(frame, "m", device="cpu")

    assert len(fake_facexlib) == 1
    helper_kwargs = fake_facexlib[0]
    assert helper_kwargs["det_model"] == "retinaface_resnet50"


def test_helper_state_is_cleared_before_each_frame(fake_facexlib, monkeypatch):
    from dw.tasks import restore_faces as module

    descriptor = MagicMock(supports_half=False)
    monkeypatch.setattr(module, "_load_face_model", lambda path, device: descriptor)
    monkeypatch.setattr(
        "dw.tasks.upscale.resolve_model_path", lambda model_name, filename: "weights.pth"
    )
    frame = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    module.restore_faces(frame, "m", device="cpu")
    module.restore_faces(frame, "m", device="cpu")

    helper = model_cache._cache[
        ("restore_faces_helper", 1, 512, True, "cpu")
    ]
    assert helper.cleaned == 2


def test_different_settings_build_different_helpers(fake_facexlib, monkeypatch):
    from dw.tasks import restore_faces as module

    descriptor = MagicMock(supports_half=False)
    monkeypatch.setattr(module, "_load_face_model", lambda path, device: descriptor)
    monkeypatch.setattr(
        "dw.tasks.upscale.resolve_model_path", lambda model_name, filename: "weights.pth"
    )
    frame = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    module.restore_faces(frame, "m", device="cpu", face_size=512)
    module.restore_faces(frame, "m", device="cpu", face_size=256)
    assert len(fake_facexlib) == 2
```

Read `dw/tasks/restore_faces.py` first: if `_load_face_model` or `resolve_model_path` is imported or named differently from what the test patches, adjust the test's patch targets to the real names, not the implementation.

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_restore_faces_helper_cache.py -n0 -q`
Expected: `test_two_frames_share_one_helper` fails with 2 constructions; the cache-key test fails with KeyError.

- [ ] **Step 3: Cache the helper**

Replace the helper construction at `dw/tasks/restore_faces.py:70-77` with:

```python
    # facexlib's helper loads RetinaFace (and ParseNet with use_parse) in
    # its constructor. per_frame calls this once per frame, so a video was
    # reloading detector weights for every frame. The helper keeps per-image
    # state between calls; clean_all() resets it before each read
    def build_helper():
        return FaceRestoreHelper(
            upscale_factor=upscale_factor,
            face_size=face_size,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            use_parse=use_parse,
            device=torch.device(device),
        )

    face_helper = cached_model(
        ("restore_faces_helper", upscale_factor, face_size, use_parse, str(device)),
        build_helper,
    )
    face_helper.clean_all()
```

Keep the existing `read_image` call that follows.

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/test_restore_faces_helper_cache.py tests/test_restore_faces_video.py tests/test_face_track.py -n0 -q`
Expected: PASS.

- [ ] **Step 5: Note it in docs/TASKS.md**

Find the `restore_faces` entry in `docs/TASKS.md` and add one sentence: "The detector and parser are loaded once per settings/device and reused across frames; `release_models` drops them with the restorer."

- [ ] **Step 6: Ruff and commit**

```bash
venv/bin/ruff format --check dw dw_mcp tests scripts && venv/bin/ruff check dw dw_mcp tests scripts
git add dw/tasks/restore_faces.py tests/test_restore_faces_helper_cache.py docs/TASKS.md
git commit -m "perf(restore_faces): build the facexlib helper once, not per frame"
```

---

### Task 3: Frame decode uses every core

**Files:**
- Modify: `dw/media.py` — the picture-decoding opens: `video_shape` (~341, only its counting branch decodes), `read_frames_at` (~385), `read_frame_range` (~481), the frame counter (~515), `decode_audio_video` (~533), `decode_rgb_frames` (~561), and `read_thumbnails_and_track` (~577) if it decodes pictures.
- Test: `tests/test_media_decode_threads.py` (create)

**Interfaces:**
- Produces: a module-level helper in `dw/media.py`:

```python
def _threaded(stream):
    """A video stream set to decode on every core. libav defaults a
    software h264 decode to one thread; AUTO lets it pick frame and
    slice threading for the codec. Header-only reads do not call this."""
    stream.thread_type = "AUTO"
    return stream
```

- [ ] **Step 1: Write the failing test**

```python
"""Every picture-decoding open in dw/media.py decodes on every core."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dw import media


class FakeFrame:
    def __init__(self):
        self.pts = 0

    def to_ndarray(self, format="rgb24"):
        return np.zeros((2, 2, 3), dtype=np.uint8)


def _container(frames=1):
    container = MagicMock()
    stream = MagicMock()
    stream.average_rate = 8
    stream.frames = 0
    stream.start_time = 0
    stream.time_base = 1
    del stream.thread_type  # set by the code under test, asserted below
    container.streams.video = [stream]
    container.streams.audio = []
    container.decode.return_value = [FakeFrame() for _ in range(frames)]
    container.__enter__.return_value = container
    container.__exit__.return_value = False
    return container, stream


@pytest.mark.parametrize(
    "call",
    [
        lambda: media.decode_rgb_frames("x.mp4"),
        lambda: media.decode_audio_video("x.mp4"),
        lambda: media.video_shape("x.mp4"),
    ],
)
def test_decoders_set_auto_threading(call):
    container, stream = _container()
    with patch.object(media.av, "open", return_value=container):
        call()
    assert stream.thread_type == "AUTO"
```

Read each target function before running: `decode_audio_video` builds an `AudioResampler` only with an audio stream, and `video_shape` only decodes when `stream.frames` is falsy (the fake sets 0). If a function needs more fake attributes (`frame.time_base`, `stream.codec_context`), add them to the fake rather than changing the function.

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_media_decode_threads.py -n0 -q`
Expected: FAIL, `thread_type` not set (AttributeError or MagicMock inequality).

- [ ] **Step 3: Add `_threaded` and apply it**

Add `_threaded` near the top of `dw/media.py` (after the imports). In each picture-decoding function, wrap the stream once where it is taken: `stream = _threaded(container.streams.video[0])`, and in `decode_rgb_frames` change `container.decode(video=0)` to `container.decode(_threaded(container.streams.video[0]))`. In `video_shape`, call `_threaded(stream)` only inside the `if count is None:` branch. Leave `probe_media`, `_stream_info`, the duration helpers and the audio-only opens alone.

- [ ] **Step 4: Run the media tests**

Run: `venv/bin/python -m pytest tests/test_media_decode_threads.py tests/test_media_layering.py tests/test_video_utils.py tests/test_concat_videos.py tests/test_dissolve_videos.py -n0 -q`
Expected: PASS. (If a named file does not exist, drop it from the list; `ls tests | grep media` for the real names.)

- [ ] **Step 5: Ruff and commit**

```bash
venv/bin/ruff format --check dw dw_mcp tests scripts && venv/bin/ruff check dw dw_mcp tests scripts
git add dw/media.py tests/test_media_decode_threads.py
git commit -m "perf(media): decode picture frames with libav AUTO threading"
```

---

### Task 4: LoRA scale, alpha and scheduler shift no longer reload the pipeline

**Files:**
- Modify: `dw/step_cache.py:124-144` (`pipeline_cache_key`)
- Modify: `dw/pipeline_processors/adapters.py:104-155` (`load_loras`)
- Modify: `dw/pipeline_processors/components.py:~420-470` (`load_and_configure_scheduler`)
- Modify: `dw/pipeline_processors/pipeline.py` (`Pipeline`: new method `apply_runtime_settings`)
- Modify: `dw/pipeline_ownership.py:252-293` (`wrap_resident`)
- Modify: `docs/WORKFLOW_GUIDE.md` (the pipeline reuse / `release_pipeline` area near line 1319) and `docs/LORAS.md`
- Test: `tests/test_runtime_settings.py` (create); extend `tests/test_step_cache.py`

**Interfaces:**
- Produces, in `dw/step_cache.py`:

```python
RUNTIME_LORA_KEYS = ("scale", "alpha")

def weights_identity(pipeline_definition):
    """The pipeline definition with every runtime-mutable setting removed:
    a LoRA's scale and alpha (set_adapters / set_adapter_alpha) and a
    scheduler's shift (set_shift) change nothing about what is loaded."""
```

`pipeline_cache_key` hashes `weights_identity(...)` instead of the raw filtered dict.

- Produces, in `dw/pipeline_processors/adapters.py`:

```python
def adapter_settings(loras):
    """(adapter_names, adapter_weights, alphas) for the active entries of
    `loras`, naming an adapter by its own adapter_name or its ORIGINAL index
    (a nulled entry keeps its slot). Reads, never pops."""

def apply_adapter_settings(loras, pipeline):
    """Set each active LoRA's alpha then every scale on a pipeline that
    already holds them. No-op with no active entry."""
```

`load_loras` keeps its signature and its pop-then-load behaviour but computes names/weights/alphas through `adapter_settings` *before* popping, so the two paths cannot disagree.

- Produces, in `dw/pipeline_processors/components.py`:

```python
def apply_scheduler_shift(scheduler_definition, pipeline, component_name="scheduler"):
    """Set the shift a scheduler definition names on an already-loaded
    scheduler; nothing when the definition is None or names no shift."""
```

`load_and_configure_scheduler` calls it after loading the scheduler type.

- Produces, in `dw/pipeline_processors/pipeline.py`:

```python
    def apply_runtime_settings(self):
        """Re-apply what a cache hit may have changed without changing the
        weights: scheduler and audio_scheduler shift, LoRA scales and
        alphas. Called by wrap_resident; a fresh load applies the same
        values on its way through load()."""
```

- [ ] **Step 1: Write the failing key tests**

Append to `tests/test_step_cache.py`:

```python
from dw.step_cache import pipeline_cache_key, weights_identity


def _definition(**overrides):
    base = {
        "configuration": {"component_type": "FluxPipeline"},
        "from_pretrained_arguments": {"model_name": "black-forest-labs/FLUX.1-dev"},
        "loras": [
            {"model_name": "a/b", "weight_name": "x.safetensors", "scale": 1.0, "alpha": 16},
        ],
        "scheduler": {"shift": 6.0},
        "audio_scheduler": {"shift": 3.0},
        "arguments": {"prompt": "p"},
    }
    base.update(overrides)
    return base


def test_lora_scale_and_alpha_do_not_change_the_pipeline_key():
    a = _definition()
    b = _definition(loras=[{"model_name": "a/b", "weight_name": "x.safetensors", "scale": 0.5, "alpha": 8}])
    assert pipeline_cache_key(a) == pipeline_cache_key(b)


def test_scheduler_shift_does_not_change_the_pipeline_key():
    a = _definition()
    b = _definition(scheduler={"shift": 12.0}, audio_scheduler={"shift": 1.0})
    assert pipeline_cache_key(a) == pipeline_cache_key(b)


def test_lora_model_name_still_changes_the_pipeline_key():
    a = _definition()
    b = _definition(loras=[{"model_name": "a/c", "weight_name": "x.safetensors", "scale": 1.0}])
    assert pipeline_cache_key(a) != pipeline_cache_key(b)


def test_scheduler_type_still_changes_the_pipeline_key():
    a = _definition(scheduler={"scheduler_type": "X", "shift": 6.0})
    b = _definition(scheduler={"scheduler_type": "Y", "shift": 6.0})
    assert pipeline_cache_key(a) != pipeline_cache_key(b)


def test_weights_identity_leaves_the_definition_untouched():
    definition = _definition()
    weights_identity(definition)
    assert definition["loras"][0]["scale"] == 1.0
    assert definition["scheduler"]["shift"] == 6.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/bin/python -m pytest tests/test_step_cache.py -n0 -q`
Expected: ImportError on `weights_identity`.

- [ ] **Step 3: Implement `weights_identity` and use it in the key**

In `dw/step_cache.py`, above `pipeline_cache_key`:

```python
RUNTIME_LORA_KEYS = ("scale", "alpha")


def weights_identity(pipeline_definition):
    """The pipeline definition with every runtime-mutable setting removed.

    A LoRA's scale and alpha (set_adapters / set_adapter_alpha) and a
    scheduler's shift (set_shift) change nothing about what is loaded, so a
    run that only moves one of them must hit the warm pipeline and re-apply
    the value (Pipeline.apply_runtime_settings), not reload the stack.
    Everything else - the model, the LoRA files, quantization, placement,
    the scheduler type - still changes the key. Returns a new dict; the
    definition is the workflow's and is not edited.
    """
    identity = {
        k: v
        for k, v in pipeline_definition.items()
        if k not in ("arguments", "seed", "chain")
    }
    loras = identity.get("loras")
    if isinstance(loras, list):
        identity["loras"] = [
            {k: v for k, v in lora.items() if k not in RUNTIME_LORA_KEYS}
            if isinstance(lora, dict)
            else lora
            for lora in loras
        ]
    for name in ("scheduler", "audio_scheduler"):
        scheduler = identity.get(name)
        if isinstance(scheduler, dict):
            identity[name] = {k: v for k, v in scheduler.items() if k != "shift"}
    return identity
```

and make `pipeline_cache_key` serialize `weights_identity(pipeline_definition)`; update its docstring to say scale, alpha and shift are excluded and why.

- [ ] **Step 4: Run the key tests**

Run: `venv/bin/python -m pytest tests/test_step_cache.py -n0 -q`
Expected: PASS.

- [ ] **Step 5: Write the failing setter tests**

Create `tests/test_runtime_settings.py`:

```python
"""A cache hit re-applies LoRA scale/alpha and scheduler shift in place."""

from unittest.mock import MagicMock, patch

import pytest

from dw.pipeline_processors.adapters import adapter_settings, apply_adapter_settings
from dw.pipeline_processors.components import apply_scheduler_shift


def test_adapter_settings_name_by_original_index_and_skip_nulled():
    loras = [
        {"model_name": None, "scale": 0.3},
        {"model_name": "a/b", "scale": 0.5, "alpha": 8},
        {"model_name": "c/d", "adapter_name": "turbo"},
    ]
    names, weights, alphas = adapter_settings(loras)
    assert names == ["1", "turbo"]
    assert weights == [0.5, 1.0]
    assert alphas == {"1": 8}
    assert loras[1]["scale"] == 0.5  # read, not popped


def test_apply_adapter_settings_sets_alpha_then_scales():
    pipeline = MagicMock()
    calls = []
    with patch(
        "dw.pipeline_processors.adapters.set_adapter_alpha",
        side_effect=lambda p, n, a: calls.append(("alpha", n, a)),
    ):
        pipeline.set_adapters.side_effect = lambda n, w: calls.append(("scales", n, w))
        apply_adapter_settings(
            [{"model_name": "a/b", "scale": 0.5, "alpha": 8}], pipeline
        )
    assert calls == [("alpha", "0", 8), ("scales", ["0"], [0.5])]


def test_apply_adapter_settings_is_a_no_op_without_active_loras():
    pipeline = MagicMock()
    apply_adapter_settings([{"model_name": None}], pipeline)
    apply_adapter_settings(None, pipeline)
    pipeline.set_adapters.assert_not_called()


def test_apply_scheduler_shift_sets_the_shift():
    pipeline = MagicMock()
    apply_scheduler_shift({"shift": 12}, pipeline, "scheduler")
    pipeline.scheduler.set_shift.assert_called_once_with(12.0)


def test_apply_scheduler_shift_without_a_shift_touches_nothing():
    pipeline = MagicMock()
    apply_scheduler_shift(None, pipeline, "scheduler")
    apply_scheduler_shift({"scheduler_type": "X"}, pipeline, "scheduler")
    pipeline.scheduler.set_shift.assert_not_called()


def test_wrap_resident_reapplies_runtime_settings():
    from dw.pipeline_ownership import wrap_resident

    model = MagicMock()
    cached = MagicMock()
    cached.pipeline = model
    step_definition = {
        "name": "s",
        "pipeline": {
            "configuration": {"component_type": "X", "no_generator": True},
            "loras": [{"model_name": "a/b", "scale": 0.25}],
            "scheduler": {"shift": 9},
        },
    }
    with patch("dw.pipeline_ownership.emit_phase"):
        wrapper = wrap_resident(cached, step_definition, 1, "cpu", "/out", "p")
    assert wrapper.pipeline is model
    model.scheduler.set_shift.assert_called_once_with(9.0)
    model.set_adapters.assert_called_once_with(["0"], [0.25])


def test_wrap_resident_with_nothing_mutable_calls_nothing():
    from dw.pipeline_ownership import wrap_resident

    model = MagicMock()
    cached = MagicMock()
    cached.pipeline = model
    step_definition = {
        "name": "s",
        "pipeline": {"configuration": {"component_type": "X", "no_generator": True}},
    }
    with patch("dw.pipeline_ownership.emit_phase"):
        wrap_resident(cached, step_definition, 1, "cpu", "/out", "p")
    model.set_adapters.assert_not_called()
    model.scheduler.set_shift.assert_not_called()
```

Check how `get_component(pipeline, "scheduler")` resolves on a MagicMock (it may use `getattr` or a components dict); if it does not return `pipeline.scheduler`, give the mock what it reads rather than changing the helper. `wrap_resident` imports `emit_phase`; if it is imported under another name, patch that.

- [ ] **Step 6: Run them to verify they fail**

Run: `venv/bin/python -m pytest tests/test_runtime_settings.py -n0 -q`
Expected: ImportError on `adapter_settings`.

- [ ] **Step 7: Implement the setters**

In `dw/pipeline_processors/adapters.py`:

```python
def adapter_settings(loras):
    """(adapter_names, adapter_weights, alphas) for the active entries of
    `loras`. An adapter is named by its own adapter_name or its ORIGINAL
    index - a nulled entry keeps its slot, so the name a hit re-applies to
    is the name the load gave. Reads the entries, never pops them."""
    names, weights, alphas = [], [], {}
    for i, lora in enumerate(loras or []):
        if not isinstance(lora, dict) or lora.get("model_name") is None:
            continue
        name = lora.get("adapter_name") or str(i)
        names.append(name)
        scale = lora.get("scale")
        weights.append(1.0 if scale is None else float(scale))
        if lora.get("alpha") is not None:
            alphas[name] = lora["alpha"]
    return names, weights, alphas


def apply_adapter_settings(loras, pipeline):
    """Set each active LoRA's alpha, then every scale, on a pipeline that
    already holds the adapters. Alpha first: set_adapters is what recomputes
    each layer's scaling from it. No-op with no active entry."""
    names, weights, alphas = adapter_settings(loras)
    if not names:
        return
    for name, alpha in alphas.items():
        set_adapter_alpha(pipeline, name, alpha)
    logger.info(f"Setting adapter weights: {list(zip(names, weights))}")
    pipeline.set_adapters(names, weights)
```

Rewrite `load_loras` so the loop takes `names, weights, alphas = adapter_settings(loras)` once, pops `model_name`, `adapter_name`, `scale`, `alpha` from each active entry as it does now (they must not reach `load_lora_weights`), loads with the name from `adapter_settings`, then finishes with the same alpha-then-`set_adapters` sequence (call `apply_adapter_settings` on a snapshot taken before the pops, or keep the names/weights/alphas locals - either is fine, but the names must be the ones `adapter_settings` computed).

In `dw/pipeline_processors/components.py`, move the tail of `load_and_configure_scheduler` (from `shift = scheduler_definition.get("shift", None)` to the `set_shift` call) into `apply_scheduler_shift(scheduler_definition, pipeline, component_name="scheduler")`, which returns at once when `scheduler_definition` is None or has no `shift`, and call it from `load_and_configure_scheduler` after the type is loaded. Keep both `ValueError`s.

In `dw/pipeline_processors/pipeline.py`, on `Pipeline`:

```python
    def apply_runtime_settings(self):
        """Re-apply what a run may have changed without changing the weights:
        scheduler and audio_scheduler shift, LoRA scales and alphas. The
        pipeline cache keys on weights_identity, so a hit can carry any of
        these at a new value; load() applies the same values on its way
        through, so a cold load and a hit end in the same state."""
        apply_scheduler_shift(self.pipeline_definition.get("scheduler"), self.pipeline)
        apply_scheduler_shift(
            self.pipeline_definition.get("audio_scheduler"),
            self.pipeline,
            "audio_scheduler",
        )
        apply_adapter_settings(self.pipeline_definition.get("loras"), self.pipeline)
```

Import `apply_scheduler_shift` from `.components` and `apply_adapter_settings` from `.adapters` beside the existing imports.

In `dw/pipeline_ownership.py` `wrap_resident`, after the generator block and before `emit_phase("cached", ...)`:

```python
    # The key the hit matched on is the weights identity: a scale, alpha or
    # shift may differ from the values the resident model last ran with
    new_pipeline_wrapper.apply_runtime_settings()
```

- [ ] **Step 8: Run the setter tests and the neighbours**

Run: `venv/bin/python -m pytest tests/test_runtime_settings.py tests/test_step_cache.py tests/test_lora_alpha.py tests/test_lora_disable.py tests/test_h3_adapters.py tests/test_h3_schedule.py tests/test_pipeline_ownership.py tests/test_workflow_run.py -n0 -q`
Expected: PASS. (Drop a name that does not exist; `ls tests | grep -E "ownership|workflow_run|lora"`.) If a test asserted that a scale change produces a *different* key or a reload, read it: it was pinning the behaviour this task replaces, and its assertion flips.

- [ ] **Step 9: Docs**

`docs/WORKFLOW_GUIDE.md`, next to the `release_pipeline` description (~line 1324), add a short paragraph "What keeps a pipeline warm": the server's worker keeps a pipeline loaded across runs as long as what it *loads* is unchanged; a LoRA's `scale` and `alpha` and a scheduler's `shift` are applied in place on the warm pipeline, so iterating on them costs no reload; changing a LoRA's `model_name` or `weight_name`, the quantization, placement or scheduler type reloads. `docs/LORAS.md`: one sentence at the scale/alpha description saying the same. `docs/ARCHITECTURE.md`: if its map has a row for the pipeline cache key (grep `pipeline_cache_key`), extend the rule text with "keys on the weights identity (scale, alpha, shift excluded)".

- [ ] **Step 10: Ruff and commit**

```bash
venv/bin/ruff format --check dw dw_mcp tests scripts && venv/bin/ruff check dw dw_mcp tests scripts
git add dw/step_cache.py dw/pipeline_processors/adapters.py dw/pipeline_processors/components.py dw/pipeline_processors/pipeline.py dw/pipeline_ownership.py tests/test_runtime_settings.py tests/test_step_cache.py docs/WORKFLOW_GUIDE.md docs/LORAS.md docs/ARCHITECTURE.md
git commit -m "perf(cache): LoRA scale/alpha and scheduler shift re-apply on a warm pipeline instead of reloading"
```

---

### Task 5: A workflow switch keeps the pipelines the next workflow loads

**Files:**
- Modify: `dw/workflow_run.py` (new `pipeline_keys(workflow, arguments)` beside `cache_hits`, ~line 445)
- Modify: `dw/worker.py:349-367` (`_prepare_workflow`) and `:638-709` (`_cleanup_all` gains a `keep` parameter)
- Modify: `docs/WORKER_GUIDE.md` and `docs/SERVER.md:~224` (the "Workflow changed - releasing cached models" behaviour)
- Test: `tests/test_worker_execute.py` (extend) and `tests/test_workflow_run_keys.py` (create)

**Interfaces:**
- Produces, in `dw/workflow_run.py`:

```python
def pipeline_keys(workflow, arguments):
    """The set of pipeline cache keys a run of `workflow` with `arguments`
    would load its steps under - the same table run() and cache_hits()
    take (step_pipeline_keys over the prepared definition). Prepares the
    definition exactly as a run does and executes nothing. Sub-workflow
    steps load their own pipelines later and are not in the set."""
```

- Produces, in `dw/worker.py`: `_cleanup_all(self, keep=frozenset())` - every pipeline whose key is in `keep` survives; the rest, the task model cache and the step cache go as today. With `keep` empty the behaviour is exactly today's. Returns the same summary string, now naming how many pipelines were kept.

- `_prepare_workflow` on an identity change: compute `keep = pipeline_keys(new_workflow, command.get("arguments") or {}) & set(self.loaded_pipelines)`; on any exception, log at debug and use an empty set. If `keep` is non-empty, reply `Output(message="Workflow changed - keeping N warm model(s) the new workflow loads, releasing the rest...")`, else the existing message; then `self._reply(Output(message=self._cleanup_all(keep=keep)))`.

- [ ] **Step 1: Write the failing `pipeline_keys` test**

Create `tests/test_workflow_run_keys.py`, reusing `build_test_workflow_and_call_count_spy` from `tests/test_workflow_step_cache.py` (see its `TestCacheHits` class at ~line 1003 for the try/finally that stops its patchers):

```python
"""pipeline_keys() answers the worker's question on a workflow switch: which
resident pipelines does the next workflow load anyway."""

import copy

from dw import workflow_run as workflow_run_module
from dw.step_cache import step_cache, step_pipeline_keys
from tests.test_workflow_step_cache import build_test_workflow_and_call_count_spy


def test_pipeline_keys_match_what_a_run_keys_on(tmp_path):
    step_cache.clear()
    workflow, _ = build_test_workflow_and_call_count_spy(str(tmp_path))
    try:
        expected_def, _, _ = workflow_run_module.prepare_definition(
            workflow,
            copy.deepcopy(workflow.workflow_definition),
            {},
            workflow_run_module.run_base_dir(workflow),
        )
        expected = set(step_pipeline_keys(expected_def["steps"]).values())
        assert expected
        assert workflow_run_module.pipeline_keys(workflow, {}) == expected
    finally:
        for p in workflow._test_patcher:
            p.stop()


def test_an_unseeded_workflow_still_answers(tmp_path):
    step_cache.clear()
    workflow, _ = build_test_workflow_and_call_count_spy(str(tmp_path))
    del workflow.workflow_definition["seed"]
    try:
        assert workflow_run_module.pipeline_keys(workflow, {})
    finally:
        for p in workflow._test_patcher:
            p.stop()


def test_the_probe_writes_nothing(tmp_path):
    step_cache.clear()
    workflow, call_count = build_test_workflow_and_call_count_spy(str(tmp_path))
    try:
        workflow_run_module.pipeline_keys(workflow, {})
        assert list(tmp_path.iterdir()) == []
        assert call_count() == 0
    finally:
        for p in workflow._test_patcher:
            p.stop()
```

If `tests/test_workflow_step_cache.py` cannot be imported as `tests.test_workflow_step_cache` (check `tests/__init__.py` exists - it does), import it the way the other tests do.

Unlike `cache_hits`, `pipeline_keys` must not return `[]` for an unseeded workflow: a warm model is worth keeping whether or not step results are cacheable.

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_workflow_run_keys.py -n0 -q`
Expected: ImportError on `pipeline_keys`.

- [ ] **Step 3: Implement `pipeline_keys`**

In `dw/workflow_run.py`, after `cache_hits`:

```python
def pipeline_keys(workflow, arguments):
    """The set of pipeline cache keys a run of `workflow` with `arguments`
    loads its steps under - the table run() and cache_hits() take
    (step_pipeline_keys over the prepared definition). Prepares the
    definition exactly as a run does and executes nothing: the worker asks
    this before releasing the previous workflow's models, so a model the
    next workflow loads anyway stays warm. A sub-workflow step loads its
    own pipelines later and is not in the set; an unseeded workflow still
    answers, since warmth does not depend on the step cache.
    """
    output_root_token = activate_output_root(workflow.output_dir)
    try:
        workflow_def = copy.deepcopy(workflow.workflow_definition)
        base_dir = run_base_dir(workflow)
        workflow_def, _, _ = prepare_definition(
            workflow, workflow_def, arguments or {}, base_dir
        )
        return set(step_pipeline_keys(workflow_def.get("steps", [])).values())
    finally:
        deactivate_output_root(output_root_token)
```

`step_pipeline_keys` is already imported for `cache_hits`; if not, import it from `.step_cache`.

- [ ] **Step 4: Run it**

Run: `venv/bin/python -m pytest tests/test_workflow_run_keys.py -n0 -q`
Expected: PASS.

- [ ] **Step 5: Write the failing worker tests**

In `tests/test_worker_execute.py`, after `test_workflow_switch_evicts_cache_and_untouched_keys_dropped`:

```python
def test_workflow_switch_keeps_pipelines_the_new_workflow_loads():
    worker = _make_worker()
    _execute(worker, StubWorkflow())
    worker.loaded_pipelines["shared-key"] = object()
    worker.loaded_pipelines["old-only-key"] = object()

    with patch("dw.worker.workflow_run.pipeline_keys", return_value={"shared-key"}):
        messages = _execute(
            worker, StubWorkflow(), command=snapshot_command("/w/other.json")
        )

    assert "shared-key" in worker.loaded_pipelines
    assert "old-only-key" not in worker.loaded_pipelines
    outputs = [m["message"] for m in messages if m["type"] == "output"]
    assert any("keeping 1 warm" in m for m in outputs)


def test_workflow_switch_falls_back_to_a_full_release_when_keys_cannot_be_taken():
    worker = _make_worker()
    _execute(worker, StubWorkflow())
    worker.loaded_pipelines["shared-key"] = object()

    with patch(
        "dw.worker.workflow_run.pipeline_keys", side_effect=ValueError("bad variable")
    ):
        _execute(worker, StubWorkflow(), command=snapshot_command("/w/other.json"))

    assert worker.loaded_pipelines == {}


def test_full_cleanup_keep_spares_named_keys_only():
    worker = _make_worker()
    worker.loaded_pipelines = {"a": object(), "b": object()}
    worker._cleanup_all(keep={"a"})
    assert set(worker.loaded_pipelines) == {"a"}
```

Read `_execute`, `_make_worker` and `snapshot_command` at the top of the file first; `StubWorkflow` may need the `workflow_definition` attribute `pipeline_keys` is patched around (it is patched, so it does not). The existing `test_workflow_switch_evicts_cache_and_untouched_keys_dropped` patches `_cleanup_all` and asserts one call; it still holds (the call now carries `keep=`), so leave it or tighten it to `cleanup.assert_called_once_with(keep=set())`.

- [ ] **Step 6: Run them to verify they fail**

Run: `venv/bin/python -m pytest tests/test_worker_execute.py -n0 -q`
Expected: the three new tests fail (`keep` unexpected keyword; `shared-key` evicted).

- [ ] **Step 7: Implement the worker change**

`_cleanup_all(self, keep=frozenset())`: at `dw/worker.py:654-658` replace

```python
        self.loaded_pipelines.clear()
        self.shared_components.clear()
        self.prior_step_keys.clear()
```

with

```python
        kept = {k: v for k, v in self.loaded_pipelines.items() if k in keep}
        dropped = len(self.loaded_pipelines) - len(kept)
        # Rebuilt rather than cleared in place: the dict object is shared with
        # nothing, and a kept entry must keep its identity
        self.loaded_pipelines.clear()
        self.loaded_pipelines.update(kept)
        # Every run republishes shared components from the pipelines it hits
        # (workflow.py: publish_shared_components on a cache hit)
        self.shared_components.clear()
        # A step's prior key addresses an entry that is now gone - unless it
        # was kept, in which case a redefinition next run still releases it
        self.prior_step_keys = {
            name: key for name, key in self.prior_step_keys.items() if key in kept
        }
```

Leave `clear_model_cache()` and `step_cache.clear()` as they are. Append `f", kept {len(kept)} warm"` to the summary when `kept` is non-empty, and use `dropped` in the "Full cleanup complete" log line.

`_prepare_workflow`:

```python
        if identity != self.workflow_identity:
            if self.workflow_identity is not None:
                keep = self._keys_worth_keeping(command, job.workflow)
                if keep:
                    self._reply(
                        Output(
                            message=f"Workflow changed - keeping {len(keep)} warm "
                            "model(s) the new workflow loads, releasing the rest..."
                        )
                    )
                else:
                    self._reply(
                        Output(message="Workflow changed - releasing cached models...")
                    )
                self._reply(Output(message=self._cleanup_all(keep=keep)))
            self.workflow_identity = identity
```

with

```python
    def _keys_worth_keeping(self, command, workflow):
        """The resident pipeline keys the next workflow loads anyway. Any
        failure to prepare the definition answers the empty set - the run
        itself will report what is wrong, and a full release is the safe
        default on a card that cannot hold two stacks."""
        try:
            wanted = workflow_run.pipeline_keys(
                workflow, command.get("arguments") or {}
            )
        except Exception as e:
            logger.debug(f"Could not take the next workflow's pipeline keys: {e}")
            return set()
        return wanted & set(self.loaded_pipelines)
```

Check whether `_prepare_workflow` runs with the job's asset dir already active (`_activate_job` is phase 1, so yes); `pipeline_keys` may realize `asset:` variables and needs it.

- [ ] **Step 8: Run the worker tests**

Run: `venv/bin/python -m pytest tests/test_worker_execute.py tests/test_worker_pool.py tests/test_workflow_run_keys.py -n0 -q`
Expected: PASS.

- [ ] **Step 9: Docs**

`docs/WORKER_GUIDE.md` and `docs/SERVER.md` (~line 224, "that worker's pipelines and step cache are warm"): state that switching workflows releases only the models the next workflow does not load; a model both load stays warm (same weights identity, see Task 4's paragraph in WORKFLOW_GUIDE). Mention that `clear_memory` is still the way to empty the card by hand.

- [ ] **Step 10: Ruff and commit**

```bash
venv/bin/ruff format --check dw dw_mcp tests scripts && venv/bin/ruff check dw dw_mcp tests scripts
git add dw/worker.py dw/workflow_run.py tests/test_worker_execute.py tests/test_workflow_run_keys.py docs/WORKER_GUIDE.md docs/SERVER.md
git commit -m "perf(worker): keep warm pipelines a switched workflow loads anyway"
```

---

### Task 6: Full-suite gate and release note

**Files:**
- Modify: `docs/RELEASING.md` ("Next" section)

- [ ] **Step 1: Full suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all pass. The memory notes that `test_audio_utils`, `test_worker` lifecycle and `test_download_watch` can flake under xdist; re-run a failing one alone with `-n0` before treating it as a regression.

- [ ] **Step 2: Ruff on the whole tree**

Run: `venv/bin/ruff format --check dw dw_mcp tests scripts && venv/bin/ruff check dw dw_mcp tests scripts`

- [ ] **Step 3: Release note**

Under "Next" in `docs/RELEASING.md`, one bullet per task in user terms: warm pipelines survive a LoRA scale/alpha or shift change and a workflow switch; face restore and video decode faster on video inputs; compose-workflows and prompt-weighting defaults corrected; dead caches and a no-op release removed from the minimax templates.

- [ ] **Step 4: Commit**

```bash
git add docs/RELEASING.md
git commit -m "docs(release): note the workflow speed changes"
```

---

## Out of scope (needs a GPU measurement on lem before changing)

These came out of the same audit and are deliberately not tasks here. Each changes output or depends on a number this plan cannot produce on a Mac:

- Sequential -> model offload across the 19 Flux / Z-Image / Krea / Qwen-edit workflows (RECIPES_24GB measured model offload fastest for Flux; needs a per-family VRAM check at each template's canvas).
- `attention_backend: sage_hub` variable on the H3 and LTX templates (28-33% per step measured; needs `kernels` and `DIFFUSERS_TRUST_REMOTE_KERNELS=true`).
- Qwen-Image-2.1 Turbo8 LoRA (catalog `trial`) and a Flux dev 8-step LoRA; quality judgement needed.
- Kandinsky 6 text encoder int8 / on_demand residency; compile on its base transformer.
- qr-code at 150 steps; SD 1.5 templates in float32.
- Background video encode and skipping the post-write LUFS probe for in-memory intermediates; batching upscale tiles and RIFE pairs; SAM2 video propagation.
- Prompt and reference embedding cache per resident pipeline (chained-segments, restore-long, extend-clip).
- #815 on-disk cache of SDNQ-quantized weights.
