# H3 Sparse Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sparse` (comfy-kitchen `sol_attn`, tau 0.5) the default attention for MiniMax H3's generated-video rows on CUDA. It falls back to dense everywhere else. The server reports which backends it can run, and the UI offers them as a dropdown.

**Architecture:** A new `dw/sparse_attention.py` holds the tile map, the processor and the per-call decision. It attaches to a `MiniMaxH3Transformer3DModel` by replacing the processors on `transformer_blocks[*].attn`, and by adding an `nn.Module` forward pre-hook and post-hook. The hooks read `token_tags`/`position_ids`; the transformer's `forward` is never wrapped. `attention_backend: "sparse"` is intercepted before diffusers sees it, both at component level (`components.py`) and at pipeline level (`pipeline.py`). Every pipeline call enters `sparse_call(...)`. That context decides sparse or dense once, before the first forward, and emits the run-time fallback warning.

**Tech Stack:** Python 3.12, PyTorch, diffusers 0.41 dev (modular MiniMax-H3), comfy-kitchen 0.2.38, FastAPI/Pydantic, Svelte 5 + Vitest.

**Spec:** `docs/superpowers/specs/2026-10-17-h3-sparse-attention-design.md`. Evidence is on issue #931. The prototype being ported is `docs/superpowers/plans/931/h3_sol_proto.py` (a throwaway script, kept for reference; delete it with the final task).

## Global Constraints

- **Backend name:** `sparse`. Tau setting: `sparse_tau`, default `0.5`. Validate warning code: `sparse_attention_fallback`.
- **Dependency:** `comfy-kitchen>=0.2.38` in `pyproject.toml` `dependencies`. It's the single dependency source; `install.sh` installs `-e ".[server,dev]"`.
- **Kernel requirements:** CUDA sm_80+ (or HIP with matrix cores), bf16/fp16, head_dim 128. The check is `comfy_kitchen.sol_attn_is_available(device)`.
- **Sparse never refuses and never fails a run that dense would have run.**
- **Decided once per pipeline call, all or nothing.** A check that fails after the decision raises; it never drops that forward to dense.
- **Target video = the trailing run of video rows** (`token_tags == 0` after the last non-video row).
- **Tile coordinates come from ranking each axis's distinct `position_ids` values.** Tiles are 4x4x4 (64 rows). Prefix tiles are segment-pure, at most 64 rows each, and listed first. `sink_blocks = sink_q = [0, num_prefix_tiles]`.
- **No model knowledge beyond the class name `MiniMaxH3Transformer3DModel`** and the layout tensors diffusers passes it.
- **`sparse` never changes diffusers' process-wide active backend**, and is never added to `CUDA_ONLY_ATTENTION_BACKENDS`.
- **The server process must not initialize CUDA** to answer which backends it can run: `torch.cuda.is_initialized()` stays `False`.
- **`sparse_tau` is a runtime setting**, like a LoRA scale: excluded from `weights_identity`, and re-applied on a pipeline-cache hit.
- **Security rules from CLAUDE.md apply**; no new file, URL or subprocess surface.
- **Ratchets** (modules count, `test_dw_patch_targets`, catalog budgets) are moved only in their own commit, with the reason.

## Review Focus

1. **A cached H3 transformer re-used with a different backend.** The worker keeps models between jobs. A job asking for `sage_hub` or none, after a `sparse` job on the same weights, must run without the sparse processors. Owned by Task 4: `test_non_sparse_backend_uninstalls_sparse`.
2. **Pipeline-level `attention_backend: "sparse"` on a non-H3 pipeline** (SD 1.5, FLUX, LTX). It must run dense and warn once, and must never pass the string `sparse` to diffusers' `attention_backend()` context, which would raise. Owned by Task 4: `test_pipeline_level_sparse_on_non_h3_runs_dense`.
3. **A Mac (MPS) H3 run with the template default.** Validate gives one `sparse_attention_fallback` warning, not `attention_backend_unavailable`, and the run completes dense. Owned by Tasks 3 and 5: `test_fallback_on_non_cuda_device`, `test_sparse_on_mps_warns_fallback_not_unavailable`.
4. **A chained or multi-segment H3 job** (`chained-segments`, `shots-batch`). Each pipeline call decides afresh, and a later call can't inherit a stale tile map from a previous segment's shape. Owned by Task 3: `test_tile_map_rebuilt_when_layout_changes`, `test_each_call_decides_afresh`.
5. **The text token refiner and a Ref2VA reference image** must stay dense and exact: only `transformer_blocks[*].attn` is swapped, and reference rows go into the dense prefix. Owned by Tasks 2 and 3: `test_ref2va_reference_rows_in_prefix`, `test_refiner_processors_untouched`.

---

### Task 1: comfy-kitchen dependency

**Files:**
- Modify: `pyproject.toml` (the `dependencies` list, after `"kernels>=0.17.2"`)
- Modify: `docs/DEPENDENCIES.md`
- Test: `tests/test_dependencies_declared.py` (create if no equivalent exists; check first with `grep -l "kernels>=" tests/*.py`)

**Interfaces:**
- Produces: `import comfy_kitchen` works in every install.

- [ ] **Step 1: Write the failing test**

```python
import tomllib, pathlib

def test_comfy_kitchen_is_a_core_dependency():
    data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    assert any(d.replace(" ", "").startswith("comfy-kitchen>=0.2.38") for d in deps)
```

- [ ] **Step 2: Run it:** `venv/bin/python -m pytest tests/test_dependencies_declared.py -v -n0`. Expected: FAIL.
- [ ] **Step 3: Add the dependency.** Add `"comfy-kitchen>=0.2.38",` with a two-line comment, written in the style of the `kernels` entry: "Sol-Attn sparse attention for MiniMax H3's generated-video rows (#931); compiled CUDA/HIP kernels on Linux and Windows, an eager-only wheel elsewhere". Then add a row to `docs/DEPENDENCIES.md`'s table in its existing format.
- [ ] **Step 4: Install locally and check.** Run `venv/bin/pip install -e ".[server,dev]" && venv/bin/python -c "import comfy_kitchen; print(comfy_kitchen.__name__)"`. Expected: prints `comfy_kitchen`, and the test passes.
- [ ] **Step 5: Commit:** `git commit -m "build: comfy-kitchen dependency for H3 sparse attention (#931)"`

### Task 2: The tile map

**Files:**
- Create: `dw/sparse_attention.py` (module docstring: what sparse is, the layout rule, and the decided-once rule)
- Test: `tests/test_sparse_attention_tiles.py`

**Interfaces:**
- Produces:
  - `class SparseLayoutError(ValueError)`
  - `@dataclass(frozen=True) class TileMap`, with fields:
    - `dst: torch.Tensor`: long `(rows,)`, the packed row -> padded tiled row
    - `block_len: torch.Tensor`: int32 `(tiles,)`
    - `num_prefix_tiles: int`
    - `padded_rows: int`
    - `rows: int`
    - `video_grid: tuple[int, int, int]`
  - `build_tile_map(position_ids: torch.Tensor, token_tags: torch.Tensor, device) -> TileMap`

- [ ] **Step 1: Write the failing tests.** A helper `_layout(text, ref_rows, audio, t, h, w)` in the test builds `token_tags` and fractional `position_ids` the way diffusers does: text rows tag 1, reference-image rows tag 0, audio rows tag 2, then target video tag 0 last. Spatial positions use `np.linspace(left, left+ratio, n, endpoint=False) * 32`; time uses cumulative `5/3 * (1,4,4,4...)` from `text`.

```python
def test_t2va_grid_and_tiles():
    tags, pos = _layout(text=356, ref_rows=0, audio=414, t=37, h=17, w=30)
    tm = build_tile_map(pos, tags, "cpu")
    assert tm.video_grid == (37, 17, 30)
    assert tm.block_len.sum().item() == tm.rows == len(tags)
    assert tm.num_prefix_tiles == 13           # 356 -> 6 tiles, 414 -> 7
    assert len(tm.block_len) - tm.num_prefix_tiles == 10 * 5 * 8
    assert torch.unique(tm.dst).numel() == tm.rows
    assert (tm.block_len <= 64).all()

def test_ref2va_reference_rows_in_prefix():
    tags, pos = _layout(text=356, ref_rows=5506, audio=414, t=37, h=24, w=42)
    tm = build_tile_map(pos, tags, "cpu")
    assert tm.video_grid == (37, 24, 42)
    ref = torch.arange(356, 356 + 5506)
    assert (tm.dst[ref] < tm.num_prefix_tiles * 64).all()

def test_prefix_tiles_are_segment_pure():
    tags, pos = _layout(text=70, ref_rows=0, audio=70, t=4, h=4, w=4)
    tm = build_tile_map(pos, tags, "cpu")
    # 70 text -> 64 + 6, 70 audio -> 64 + 6: four prefix tiles, never one mixed
    assert tm.block_len[: tm.num_prefix_tiles].tolist() == [64, 6, 64, 6]

def test_rows_within_a_video_tile_are_spatially_local():
    tags, pos = _layout(text=8, ref_rows=0, audio=8, t=8, h=8, w=8)
    tm = build_tile_map(pos, tags, "cpu")
    # every full video tile spans at most 4 distinct values per axis
    ...

def test_not_a_full_grid_raises():
    tags, pos = _layout(text=8, ref_rows=0, audio=8, t=4, h=4, w=4)
    with pytest.raises(SparseLayoutError):
        build_tile_map(pos[:-1], tags[:-1], "cpu")

def test_no_target_video_raises():
    tags = torch.ones(128, dtype=torch.long)
    with pytest.raises(SparseLayoutError):
        build_tile_map(torch.zeros(128, 3, dtype=torch.float64), tags, "cpu")
```

- [ ] **Step 2: Run it:** `venv/bin/python -m pytest tests/test_sparse_attention_tiles.py -v -n0`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `build_tile_map`.** Port the prototype's `build_tile_map`, with the trailing-run target rule and per-axis ranking. Two differences from the prototype:
  - Every prototype `assert` becomes `raise SparseLayoutError(...)`.
  - Prefix segments are contiguous runs of equal `token_tags`. The prototype also keyed on the timestep index; the tile map no longer takes timestep indices.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -m "feat(attention): H3 sparse tile map (#931)"`

### Task 3: The processor, installation and the per-call decision

**Files:**
- Modify: `dw/sparse_attention.py`
- Test: `tests/test_sparse_attention_processor.py`

**Interfaces:**
- Consumes: `build_tile_map`, `TileMap`, `SparseLayoutError` (Task 2).
- Produces:
  - `SPARSE_BACKEND = "sparse"`, `DEFAULT_TAU = 0.5`
  - `install_sparse(component, tau: float) -> bool`. Returns `False`, changing nothing, when `type(component).__name__ != "MiniMaxH3Transformer3DModel"`. Idempotent: a second call only updates tau.
  - `uninstall_sparse(component) -> None`. Restores the original processor objects and removes both hooks. A no-op when nothing is installed.
  - `is_sparse_installed(component) -> bool`
  - `set_sparse_tau(component, tau: float) -> None`
  - `sparse_unavailable_reason(device, dtype, head_dim: int) -> str | None`. Returns `None` when the kernel can run. Otherwise one of: `"device <type> is not CUDA"`, `"compute capability <M>.<m> is below 8.0"`, `"dtype <dtype> is not bfloat16/float16"`, `"head_dim <n> is not 128"`, `"comfy-kitchen kernels unavailable: <exc>"`
  - `@contextmanager sparse_call(pipeline, label: str) -> Iterator[SparseCallReport]`. Resets every installed component's per-call decision on entry. On exit, if any installed component fell back, calls `emit_warning(f"{label}: sparse attention ran dense - {reason}", attention_backend="sparse", reason=reason)` once per call.
  - `@dataclass class SparseCallReport`: `ran_sparse: bool`, `reason: str | None`
  - Module-level `_sol_attn = None`, a seam: `_kernel()` returns `_sol_attn` if set, else imports `comfy_kitchen.sol_attn` and enters `ck.use_backend("cuda")`. Tests monkeypatch it.

- [ ] **Step 1: Write the failing tests.** They use a tiny real `MiniMaxH3Transformer3DModel` built from config: `num_layers=2`, `num_refiner_layers=1`, `num_attention_heads=2`, `attention_head_dim=128`, small `hidden_size`/`ffn_dim`/`text_dim`. Fake inputs come from Task 2's `_layout` helper (move it to `tests/sparse_layouts.py`).

```python
def test_install_swaps_only_block_processors(tiny_h3):
    refiner_before = [l.attn.processor for l in tiny_h3.token_refiner.refiner_blocks]  # adjust to the class's attribute
    assert install_sparse(tiny_h3, 0.5) is True
    assert all(type(b.attn.processor).__name__ == "SparseH3AttnProcessor" for b in tiny_h3.transformer_blocks)
    assert [l.attn.processor for l in tiny_h3.token_refiner.refiner_blocks] == refiner_before

def test_refiner_processors_untouched(tiny_h3): ...  # the same assertion after a forward

def test_install_on_non_h3_returns_false():
    assert install_sparse(torch.nn.Linear(2, 2), 0.5) is False

def test_signature_unchanged(tiny_h3):
    before = inspect.signature(tiny_h3.forward)
    install_sparse(tiny_h3, 0.5)
    assert inspect.signature(tiny_h3.forward) == before

def test_permutation_is_exact(tiny_h3, monkeypatch):
    # A fake kernel computing dense SDPA over the tiled tensors, padding
    # masked by block_len: the sparse path must equal the stock processor
    monkeypatch.setattr(sparse_attention, "_sol_attn", dense_sol_attn_fake)
    monkeypatch.setattr(sparse_attention, "sparse_unavailable_reason", lambda *a: None)
    ref = tiny_h3(**inputs).sample
    install_sparse(tiny_h3, 0.5)
    with sparse_call(SimpleNamespace(transformer=tiny_h3), "t") as report:
        out = tiny_h3(**inputs).sample
    assert report.ran_sparse
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

def test_fallback_on_non_cuda_device(tiny_h3, monkeypatch):
    warnings = capture(monkeypatch, sparse_attention, "emit_warning")
    install_sparse(tiny_h3, 0.5)
    ref_free = ...  # output with sparse uninstalled
    with sparse_call(SimpleNamespace(transformer=tiny_h3), "step") as report:
        out = tiny_h3(**inputs).sample
    assert not report.ran_sparse and "is not CUDA" in report.reason
    assert len(warnings) == 1 and "ran dense" in warnings[0][0]
    torch.testing.assert_close(out, ref_free)

def test_failure_after_decision_raises(tiny_h3, monkeypatch):
    # decided sparse on forward 1; forward 2 brings a layout that isn't a grid
    with pytest.raises(SparseLayoutError): ...

def test_each_call_decides_afresh(tiny_h3, monkeypatch): ...  # call 1 falls back, call 2 runs sparse
def test_tile_map_rebuilt_when_layout_changes(tiny_h3, monkeypatch): ...  # two shapes, two maps, both exact
def test_uninstall_restores_original_processors(tiny_h3): ...
def test_active_backend_unchanged(tiny_h3):
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry
    before = _AttentionBackendRegistry._active_backend
    install_sparse(tiny_h3, 0.5); uninstall_sparse(tiny_h3)
    assert _AttentionBackendRegistry._active_backend == before
```

`dense_sol_attn_fake(q, k, v, *, sink_blocks, sink_q, block_len, tau, **_)` lives in `tests/sparse_layouts.py`. It masks the padded key rows (rows `>= block_len[tile]` within each 64-row tile) and runs `F.scaled_dot_product_attention`.

- [ ] **Step 2: Run them:** `venv/bin/python -m pytest tests/test_sparse_attention_processor.py -v -n0`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - `SparseH3AttnProcessor` is the prototype's `sol_call`, reading its `TileMap` and tau from a per-component `_SparseState` that the forward pre-hook fills.
  - The pre-hook is `component.register_forward_pre_hook(hook, with_kwargs=True)`. On the call's first forward it computes the decision: `sparse_unavailable_reason(hidden_states.device, block dtype, attention_head_dim)`, then `build_tile_map`. A `SparseLayoutError` at decision time becomes a fallback with that reason.
  - On later forwards, a decided-sparse state rebuilds the map only when `(position_ids.shape, token_tags.sum())` changes, and lets `SparseLayoutError` propagate.
  - A decided-dense state makes the processor call the original processor object.
  - Keep the original processors on the state object, for `uninstall_sparse`.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -m "feat(attention): sparse H3 processor, install, per-call decision (#931)"`

### Task 4: Wiring into components, pipeline, schema and the runtime settings

**Files:**
- Modify:
  - `dw/pipeline_processors/components.py`: `configure_components` around line 124, and `set_component_attention_backend`
  - `dw/pipeline_processors/pipeline.py`: `_call_pipeline` around line 627-650, and the `apply_runtime_settings` function around line 143
  - `dw/step_cache.py`: `weights_identity`
  - `dw/workflow_schema.json`: lines 622 and 703 (the `attention_backend` descriptions) and the new `sparse_tau` siblings
- Test: `tests/test_sparse_attention_wiring.py`

**Interfaces:**
- Consumes: `install_sparse`, `uninstall_sparse`, `set_sparse_tau`, `sparse_call`, `SPARSE_BACKEND`, `DEFAULT_TAU` (Task 3).
- Produces: the config keys `attention_backend: "sparse"` and `sparse_tau: <number 0-4>`, at pipeline `configuration` level and on a `components.<name>` entry.

- [ ] **Step 1: Write the failing tests**

```python
def test_component_sparse_installs_and_never_reaches_diffusers(monkeypatch): ...
    # configure_components with {"transformer": {"attention_backend": "sparse", "sparse_tau": 0.7}}
    # -> install_sparse(transformer, 0.7) called; set_attention_backend never called

def test_component_sparse_default_tau(monkeypatch): ...  # no sparse_tau -> 0.5

def test_non_sparse_backend_uninstalls_sparse(monkeypatch): ...
    # a component with sparse installed, then configured "sage_hub" -> uninstall_sparse ran
    # before set_attention_backend("sage_hub")

def test_no_backend_uninstalls_sparse(monkeypatch): ...  # attention_backend absent on re-configure -> uninstalled

def test_pipeline_level_sparse_installs_on_h3_components_for_the_call(monkeypatch): ...
    # configuration {"attention_backend": "sparse"} -> diffusers attention_backend() never entered;
    # install_sparse called on each pipeline component whose class is MiniMaxH3Transformer3DModel

def test_pipeline_level_sparse_on_non_h3_runs_dense(monkeypatch): ...
    # no H3 component -> one emit_warning naming the pipeline, call proceeds

def test_every_call_enters_sparse_call(monkeypatch): ...

def test_sparse_tau_is_runtime_not_identity():
    a = {"configuration": {"components": {"transformer": {"attention_backend": "sparse", "sparse_tau": 0.5}}}}
    b = copy.deepcopy(a); b["configuration"]["components"]["transformer"]["sparse_tau"] = 1.0
    assert pipeline_cache_key(a) == pipeline_cache_key(b)

def test_cache_hit_reapplies_tau(monkeypatch): ...  # apply_runtime_settings -> set_sparse_tau(transformer, 1.0)

def test_schema_accepts_sparse_tau(): ...  # validate a minimal definition carrying sparse_tau at both levels
```

- [ ] **Step 2: Run them:** `venv/bin/python -m pytest tests/test_sparse_attention_wiring.py -v -n0`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **`components.py`:** `sparse` goes through `install_sparse(component, cfg.get("sparse_tau", DEFAULT_TAU))`, with the existing `emit_log` line. If `install_sparse` returns False, emit a warning that names the component ("not a MiniMax-H3 transformer; runs dense"). Any other value, or none, calls `uninstall_sparse(component)` first.
  - **`pipeline.py`:** `_call_pipeline` never hands `sparse` to `attention_backend()`. For pipeline-level `sparse` it installs on the H3 components for the call and uninstalls in the ExitStack callback. `sparse_call(self.pipeline, self.segment_label or self.name)` is always entered.
  - **`apply_runtime_settings`:** walks `configuration.components` for `sparse_tau` and calls `set_sparse_tau`.
  - **`weights_identity`:** drops `sparse_tau` the way it drops a LoRA scale.
  - **Schema:** add `sparse_tau` (`"type": "number", "minimum": 0`) with a description naming the default 0.5 and #931. Both `attention_backend` descriptions mention `'sparse'`.
- [ ] **Step 4: Run them, plus the neighbours.** Run `venv/bin/python -m pytest tests/test_sparse_attention_wiring.py tests/test_attention_backend_trace.py tests/test_pipeline_components.py tests/test_schema.py tests/test_configuration_schema.py -v -n0`. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -m "feat(attention): 'sparse' attention_backend at component and pipeline level (#931)"`

### Task 5: Validate warning

**Files:**
- Modify: `dw/acceleration_warnings.py` (module docstring list, a new `SPARSE_FALLBACK = "sparse_attention_fallback"`, and the loop at about lines 239-268)
- Test: add to the existing test module for `acceleration_warnings` (find it with `grep -l "attention_backend_unavailable" tests/*.py`)

**Interfaces:**
- Consumes: `SPARSE_BACKEND` (Task 3).

- [ ] **Step 1: Write the failing tests**

```python
def test_sparse_on_mps_warns_fallback_not_unavailable():
    warnings = acceleration_warnings(context_for(device_type="mps", component_backend="sparse"))
    assert any("sparse_attention_fallback" in w and "'mps'" in w and "dense" in w for w in warnings)
    assert not any("attention_backend_unavailable" in w for w in warnings)

def test_sparse_on_cuda_is_silent(): ...
def test_sparse_unknown_device_is_silent(): ...  # matches the module's "unresolvable stays silent" rule
```

- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** In the backend loop, a `sparse` value on a resolved non-CUDA device appends `f"{render_path(path)}: {SPARSE_FALLBACK}: sparse attention needs CUDA; this server is '{device}', so the run uses dense attention{where}"`. Add one line for it to the module docstring list.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -m "feat(validate): sparse_attention_fallback warning (#931)"`

### Task 6: The server reports the backends it can run

**Files:**
- Modify: `dw/sparse_attention.py`: add `available_attention_backends(device_type: str) -> list[str]`
- Modify: `dw/server/api_models.py` (`HealthInfo`), `dw/server/routes/system.py` (`health`)
- Regenerate: `ui/src/lib/generated/openapi.json` via `scripts/dump_openapi.py` (under `constraints-openapi.txt`), then `cd ui && npm run gen:api`
- Test: `tests/test_sparse_attention_backends.py`, plus the existing health route test (`grep -l "api/health" tests/*.py`)

**Interfaces:**
- Produces: `HealthInfo.attention_backends: list[str]`, with the description "Attention backends this server's device can run, for the UI's dropdown; 'sparse' only where comfy-kitchen's compiled kernels loaded".
- The list order:
  - `"native"`, then the other diffusers `native`-family names (`_native_flash`, `_native_efficient`, `_native_math`, `_native_cudnn` on CUDA only, `flex`).
  - On CUDA, then `sparse` (when the compiled extension imported), then the sorted `CUDA_ONLY_ATTENTION_BACKENDS`.
  - Names not in diffusers' `AttentionBackendName` are dropped.

- [ ] **Step 1: Write the failing tests**

```python
def test_mps_lists_no_cuda_backends():
    names = available_attention_backends("mps")
    assert "native" in names and "sparse" not in names and "sage_hub" not in names

def test_cuda_lists_sparse_when_kernels_present(monkeypatch):
    monkeypatch.setattr(sparse_attention, "_kernels_importable", lambda: True)
    assert "sparse" in available_attention_backends("cuda")

def test_listing_never_initializes_cuda():
    available_attention_backends("cuda")
    assert not torch.cuda.is_initialized()

def test_health_carries_attention_backends(client): ...
```

- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** `_kernels_importable()` checks `comfy_kitchen.backends.cuda._EXT_AVAILABLE` (or the registry's CUDA availability) and catches every exception. It never calls `torch.cuda.get_device_capability`; the per-card check stays at run time (Task 3). Then regenerate the OpenAPI JSON and the TS types.
- [ ] **Step 4: Run them, plus the OpenAPI drift test** (`grep -l openapi.json tests/*.py`). Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -m "feat(server): health reports available attention backends (#931)"`

### Task 7: UI dropdowns

**Files:**
- Create: `ui/src/lib/attentionBackends.svelte.ts`: a shared `$state` holding the last `health.attention_backends`, set from `App.svelte`'s health poll (line ~148)
- Modify: `ui/src/lib/editor.ts`: add `attentionBackendVariables(definition: unknown): Set<string>`
- Modify: `ui/src/lib/editor/VariablesForm.svelte` (a new `backendVariables?: Set<string>` prop), `ui/src/lib/pages/WorkflowPage.svelte`, `ui/src/lib/pages/EditorPage.svelte` (pass the prop), `ui/src/lib/editor/PipelineOptions.svelte` (lines ~249-253: a `<select>`)
- Test: `ui/src/lib/editor.test.ts`, `ui/src/lib/editor/VariablesForm.test.ts`

**Interfaces:**
- Consumes: `HealthInfo.attention_backends` (Task 6).
- Produces:
  - `attentionBackendVariables(def)`: the names X for every `"attention_backend": "variable:X"` anywhere in `def.steps`, at pipeline configuration and component level, including nested `workflow` steps' inline definitions when present.

- [ ] **Step 1: Write the failing tests**

```ts
it('finds variables feeding attention_backend at component level', () => {
  const def = { steps: [{ pipeline: { configuration: { components: { transformer_ref: { attention_backend: 'variable:attention_backend' } } } } }] }
  expect(attentionBackendVariables(def)).toEqual(new Set(['attention_backend']))
})
it('ignores literals and other keys', ...)
it('renders a select with default (none) plus the server list for a backend variable', ...)
it('keeps a current value the server does not list, marked unavailable', ...)  // e.g. sage_hub opened on a Mac
it('falls back to the text field when the server list is empty or missing', ...)
```

- [ ] **Step 2: Run them:** `cd ui && npx vitest run src/lib/editor.test.ts src/lib/editor/VariablesForm.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - A select whose options are "default (none)" -> empty string, then the server's list.
  - A value not in the list stays selectable, labelled `<name> (not available on this server)`.
  - In override mode, the empty option means "use the workflow default".
  - Follow the existing field styling.
  - Don't use `vi.resetModules` in the Svelte component tests (standing rule).
- [ ] **Step 4: Run the UI gates:** `cd ui && npm run check && npx vitest run && npm run lint`. Expected: clean.
- [ ] **Step 5: Commit:** `git add` only the files named above (never `git add -A ui`; Don edits `package.json`), then `git commit -m "feat(ui): attention backend dropdowns from the server's list (#931)"`

### Task 8: H3 template defaults, docs and skill note

**Files:**
- Modify: the 19 files `grep -l '"attention_backend"' workflows/templates/minimax/*.json` lists (not `music.json`, `music-video-cuts.json`)
- Modify: `docs/ACCELERATION.md` (*Attention Backends*), `docs/RECIPES_24GB.md` and `docs/QUANTIZATION.md` (their H3 rows note the default backend), `plugins/dw/skills/minimax-h3/references/quality-and-speed.md`, and the `plugins/dw` version per `plugins/dw/README.md`
- Test: `tests/test_h3_sparse_defaults.py`, plus the existing `test_template_defaults.py`, `test_catalog_speed_defaults.py`, `test_plugin_skills.py`

**Interfaces:**
- Consumes: the config keys from Task 4.

- [ ] **Step 1: Write the failing test**

```python
H3 = [p for p in glob("workflows/templates/minimax/*.json") if '"attention_backend"' in open(p).read()]

def test_nineteen_h3_templates():
    assert len(H3) == 19

@pytest.mark.parametrize("path", H3)
def test_h3_defaults_to_sparse(path):
    d = json.load(open(path))
    assert d["variables"]["attention_backend"] == "sparse"
    assert d["variables"]["sparse_tau"] == 0.5
    text = json.dumps(d["steps"])
    assert text.count('"sparse_tau": "variable:sparse_tau"') == text.count('"attention_backend": "variable:attention_backend"')
    assert "attention_backend" in d["cost_drivers"]
```

- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Edit the templates, docs and skill note.**
  - **Templates:** in each one, set `"attention_backend": "sparse"`, add `"sparse_tau": 0.5`, and put `"sparse_tau": "variable:sparse_tau"` next to every `"attention_backend": "variable:attention_backend"`. Leave `cost` minutes unchanged (spec: stage 3 replaces them).
  - **`docs/ACCELERATION.md`:** a `sparse` subsection covering:
    - what it does
    - the #931 measured table (from the spec)
    - `sparse_tau`
    - the CUDA sm_80+ need, and the dense fallback with its two warnings
    - that it moves composition relative to dense, unlike sage, so a draft and its final use the same backend
  - **`quality-and-speed.md`:** one paragraph under *Ramp*, step 1. Templates now default to `sparse`. Every ramp row keeps `attention_backend=sage_hub` at both tiers until #931 stage 3 measures the sparse rows, because sparse moves staging and sage doesn't. Check the skill byte cap test still passes.
- [ ] **Step 4: Run them:** `venv/bin/python -m pytest tests/test_h3_sparse_defaults.py tests/test_template_defaults.py tests/test_catalog_speed_defaults.py tests/test_plugin_skills.py -v -n0`, then the full suite `venv/bin/python -m pytest tests/ -q`. Expected: PASS. Move a catalog budget ratchet only in its own commit, with the measured reason.
- [ ] **Step 5: Commit:** `git commit -m "feat(templates): H3 defaults to sparse attention, tau 0.5 (#931)"`

### Task 9: Deploy and GPU smoke

**Files:** none changed unless the smoke finds a defect.

- [ ] **Step 1: Deploy.** Run `scripts/deploy.sh` to lem and to mini-ai, following the lem restart and mini-ai LaunchDaemon notes in memory. Confirm `venv/bin/python -c "import comfy_kitchen"` on both.
- [ ] **Step 2: lem T2VA smoke.**
  - Run `python -m dw.run workflows/templates/minimax/video-with-audio-768p.json seed=42` against lem.
  - Expected: completes with no `sparse attention ran dense` warning, and the log shows the backend line.
  - The transformer time is within 15% of the prototype's 87.7 s for 4 forwards.
- [ ] **Step 3: lem Ref2VA smoke.** Run `reference-to-video` at `width=1344 height=768 seed=42` with the defaults. Expected: completes in about 450 s wall, no fallback warning. Pull the clip; it should match the prototype's `tau05.mp4` closely (same seed).
- [ ] **Step 4: Backend switch on a warm worker.** Re-run the same Ref2VA with `attention_backend=sage_hub`. Expected: no sparse processors (a dense-speed transformer time), and the job completes.
- [ ] **Step 5: mini-ai.** Run `python -m dw.validate workflows/templates/minimax/reference-to-video.json` against mini-ai. Expected: one `sparse_attention_fallback` warning, and no `attention_backend_unavailable`. Open the UI against mini-ai and confirm the dropdown lists no `sparse`.
- [ ] **Step 6: Report.** Comment on #931 with job ids, timings and any deviations.

### Task 10: Stage 3 measurements and the ramp (lem, Don judges)

**Files:**
- Modify: `plugins/dw/skills/minimax-h3/references/quality-and-speed.md` (*Quality ladder*, *Ramp*), and the `cost` minutes on `reference-to-video`, `dialogue-short` and `video-with-audio-768p`
- Test: `tests/test_plugin_skills.py`, `tests/test_catalog_speed_defaults.py`

- [ ] **Step 1: Queue the runs.** Use the rung arguments exactly as `quality-and-speed.md` states them, on `reference-to-video` at 1344x768 with 4 references as in #863, at seeds 42, 7 and 1234:
  - B3 and L5 with `attention_backend=sparse`, and again with `sage_hub`.
  - L1 and L7 with `sparse`.

  That's 18 runs (6 rungs × 3 seeds), queued on lem with `dw.run`; nohup any runner script. Record the job ids, wall time, peak reserved memory and the transformer per-forward time.
- [ ] **Step 2: Hand the clips to Don** in pairs: sparse vs sage per cinema rung and seed, and L1 -> L7 under sparse per seed. Ask for:
  - the skin verdict on L5 sparse vs sage
  - whether staging carried B3 -> L5 and L1 -> L7 under sparse
- [ ] **Step 3: Update the ramp from his verdicts only.**
  - New ladder rows (`B3s`, `L5s`, `L1s`, `L7s`) with minutes and his face scores.
  - The *Ramp* path table moves a path to `sparse` only where he said it holds.
  - Replace "attention never moves composition" with the sage/sparse distinction.
  - The three templates' `cost` minutes come from the measured runs.
- [ ] **Step 4: Run** `venv/bin/python -m pytest tests/test_plugin_skills.py tests/test_catalog_speed_defaults.py -v -n0`. Expected: PASS, within the byte cap.
- [ ] **Step 5: Commit**, then comment on #931 with the table. File the lightx2v Turbo-SLA follow-up issue (spec non-goal).
