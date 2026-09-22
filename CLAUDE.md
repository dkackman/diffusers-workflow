# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`diffusers-workflow` is a declarative workflow engine for HuggingFace Diffusers. Users define AI image/video generation pipelines in JSON — variable substitution, multi-step composition, cross-step data flow, and utility tasks — without writing Python. Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU.

## Common Commands

```bash
# Install
bash ./install.sh && source ./activate

# Run a workflow - templates/text-to-image.json uses a small, ungated model and a literal
# prompt, so it needs no Hugging Face login and downloads only a few GB
python -m dw.run workflows/templates/text-to-image.json
python -m dw.run workflows/templates/text-to-image.json prompt="a cat" num_images_per_prompt=4

# A gated model (e.g. workflows/models/flux-dev.json) needs Hugging Face auth
# first: huggingface-cli login

# Validate a workflow against schema
python -m dw.validate workflows/models/z-image.json

# Basic system test (torch, diffusers import check)
python -m dw.test

# Interactive REPL
python -m dw.repl

# HTTP server + web UI (http://127.0.0.1:8765, API docs at /docs)
python -m dw.serve
```

## Architecture

### Server & Web UI

`dw/serve.py` runs a FastAPI app over the same persistent worker the REPL uses,
queueing jobs FIFO and persisting history to `~/.diffusers_helper/jobs.sqlite`.
See docs/SERVER.md, `dw/server/CLAUDE.md` and `ui/CLAUDE.md`.

### MCP Server

The stdio MCP server lives in `dw_mcp/` — see `dw_mcp/CLAUDE.md` and docs/MCP.md.

### Claude Code plugin

`.claude-plugin/marketplace.json` publishes the `dw` plugin in `plugins/dw/`: one
composition skill per model family (`minimax-h3`, `minimax-music3`, `ltx-2.5`) that
chooses a template for a request's shape and states the family's hard rules, plus
cross-cutting composition skills (`script-to-video`, `series-episodes`) - shapes above
the families that orchestrate the decision trees and cast consistency across multiple
generations. Every skill the directory holds is named in `plugins/dw/README.md` and here,
pinned by the same test. Model knowledge lives there and in the catalog, never in engine
code; every number a skill states is pinned to a diffusers symbol by
`tests/test_plugin_skills.py`. `plugin.json`'s version is the engine's, bumped by
`scripts/release.sh`. Adding or re-auditing a family is `.claude/skills/model-family-onboarding/`.

### REPL Architecture

The REPL (`dw/repl.py`) uses a **persistent worker subprocess** (`dw/worker.py`) to keep GPU models cached between runs. Communication is via `multiprocessing.Queue`. Worker management is in `dw/repl_worker.py`, command handlers in `dw/repl_commands.py`.

**Critical**: Uses `multiprocessing.set_start_method("spawn")` for CUDA/MPS compatibility.

### Workspaces on the server

`dw.serve` can hold several workspaces under one root: the root's own
`workflows/assets/outputs` are the `default` workspace, a named one is a
subdirectory beside them (`named_workspace`, `create_workspace` in
`dw/workspace.py`), and `prompts/` at the root is shared by all of them - there
is one prompt library, because `prompt:` is shared by reference. Routes take an
optional `workspace`; omitting it means the default, so pre-workspace calls are
unchanged. A job carries its own `output_dir`, `asset_dir` and `workflow_dir`
(`JobManager.submit`), so it stays in its workspace whatever the manager serves
next; the worker activates the asset root per job (`activate_asset_dir`), which
is the one root that could not stay process-wide. `jobs.sqlite` has a
`workspace` column, backfilled to `default`. `common/assets` at the root is
the one asset library every workspace shares - assets are otherwise per
workspace, which is wrong for a recurring cast a later workspace still has to
reach. It sits on every workspace's asset search path behind that workspace's
own library (so a workspace name shadows a shared one), is tagged `origin:
common` by `GET /api/assets`, and is written to only when a call says so
(`?shared=true` on uploads, `"shared": true` on keep, `shared=True` over MCP).
Reserved names: `workflows`, `prompts`, `assets`, `outputs`, `exports`,
`common`. The web UI is organised by workspace, with a sidebar listing every
workspace on the server (`ui/src/lib/Sidebar.svelte`) and the selected one
named in the hash (`#/ws/<name>/...`).

The web UI has a page for it: `ui/src/lib/pages/AssetsPage.svelte` (#165)
reads `GET /api/assets` and shows the library the way the gallery shows
outputs, tagged by `origin` so a shadowed or read-only entry is visible
before a 403 explains it.

### Workflow sources

`dw/workflow_sources.py` is the server's workflow search path: the writable
directory first (the workspace's `workflows/`), then any `--examples-dir`, each
read-only. Reads (`listing`, `find_workflow`) span every root front-to-back so an
earlier name shadows a later one; `PUT /api/workflows` always resolves through
`writable_source`, so saving something opened from a read-only root writes a copy
rather than overwriting it, and `DELETE` on a read-only root answers 403. A job
carries the root it is confined to (`JobManager.submit(workflow_dir=...)`), so an
examples workflow runs confined to the examples directory rather than to the
writable one. Packaged `dw/workflows/` is off the path - it is what `builtin:`
sub-workflow steps name, resolved in `dw/workflow.py`.

The prompt and asset libraries have the same shape: each `--examples-dir`
brings the `prompts/` and `assets/` beside it (`example_libraries` in
`dw/workspace.py`), pinned into `DW_PROMPT_PATH` / `DW_ASSET_PATH` by
`dw.serve` so the spawned worker resolves as the API does. `prompt_search_path`
/ `asset_search_path` put the workspace's own library first, so a workspace
name shadows an example's; `GET /api/prompts` and `GET /api/assets` span the
path and tag each entry with its `origin`; writes (`PUT /api/prompts`, uploads,
keep-as-asset) only ever land in the workspace, and deleting a read-only prompt
answers 403.

### Workspaces

`dw/workspace.py` resolves the one directory a run's content belongs to -
`workflows/`, `prompts/`, `assets/`, `outputs/`. Order: `--workspace` >
`DW_WORKSPACE` > the `workspace` setting > the working directory when it holds
any of `workflows/`, `prompts/` or `outputs/` > `~/diffusers-workspace`. A
checkout satisfies rule four, so every default lands where it did before
workspaces existed. Resolution creates nothing; an entry point about to write
calls `ensure()` (or creates the one folder it needs). `set_workspace` pins the
root *and* how it was chosen into the environment, so a spawned worker does not
read an inferred workspace back as one the user named - `get_prompt_dir` yields
to its older discovery (`./prompts`, then the walk up from the workflow file)
for an inferred workspace but not for an explicit one. `--workflow-dir`,
`--output-dir` and `--prompt-dir` each still override one folder. See docs/WORKSPACES.md;
the later stages (workflow search path, run directories, `asset:`/`output:`
references) are documented above in *Workflow sources* and *Type System*.

### Type System

`arguments.py` + `type_helpers.py` handle dynamic type conversion during workflow loading:
- Keys ending in `_type` or `_dtype`, or named `dtype`, are auto-converted: `"FluxPipeline"` → loaded from `diffusers`, `"torch.bfloat16"` → `torch.bfloat16`
- Values wrapped in `{}` are escaped (stay as strings): `"{nf4}"` → `"nf4"`
- Dotted names use full module path: `"sdnq.SDNQConfig"` → `importlib.import_module("sdnq").SDNQConfig`
- Values prefixed with `constant:` read a value declared in python rather than copying it
  into JSON: `"constant:diffusers.pipelines.ltx2.utils.DISTILLED_SIGMA_VALUES"`. Resolved
  in `realize_args`, validated by `validate_constant_name()`; anything callable is refused
- Values prefixed with `asset:` resolve to the path of a file in the asset library:
  `"asset:iris.png"` or `"asset:gyre/frames/web.mp4"`. Resolved in `realize_args` before
  every other convention (`dw/assets.py`), rooted at the library rather than the workflow
  file, confined to it, and then loaded by whatever would have loaded a path written
  there. The library is `DW_ASSET_DIR` / `--asset-dir`, else the workspace's `assets/`
  when a workspace was named, else `./assets` if it exists, else found by walking up
  from the workflow file's directory
- Values prefixed with `output:` resolve to the path of a file an earlier run wrote:
  `"output:ltx2/Gyre/latest/still.png"`. The name is `<workflow identity>/<run id>/<file>`
  under the output root, and `latest` in the run-id position picks the newest run that
  holds the file (run ids sort by their UTC timestamp; a failed or fully-cached run holds
  only a manifest and is skipped). Resolved in `realize_args` beside `asset:` (`dw/runs.py`),
  against the output root `Workflow.run` activates, and confined to it
- A generated file becomes a stable input with `POST /api/assets/keep` (gallery "Keep as
  asset", MCP `keep_output`): it is hard-linked, else copied, from the workspace's outputs
  into its assets under a chosen name, so later workflows reference `asset:name` rather
  than a run id that pruning would break
- Values prefixed with `prompt:` load a stored prompt's `text` from the prompt library:
  `"prompt:name"` or `"prompt:folder/name"`. Resolved in `realize_args` (`dw/prompts.py`),
  rooted at the library rather than the workflow file. The library is `DW_PROMPT_DIR` /
  `--prompt-dir`, else `./prompts` if it exists, else found by walking up from the
  workflow file's directory
- A step's `result.subfolder` names a subfolder of the run directory for that step's
  files - by convention `final` for the deliverable and `intermediate` for the rest; any
  relative path (`shots/act-1`); `variable:`/`item:` allowed; no default. Mechanics under
  *Result subfolders* in Critical Gotchas
- A step carrying `for_each` (a list, or `variable:` naming one) is expanded by
  `expand_for_each` (`dw/for_each.py`) into one ordinary step per entry, named
  `<step>@<entry name or index>`, immediately after `replace_variables` in
  `Workflow.run` and, with the caller's arguments folded, in `validation_errors`.
  Inside a member `item:` / `item:field` is the entry (any type, spliced whole);
  a later step reads the group with `gather:<step>` (a list; splices inside a
  list); two groups over the same list pair by key (`slice` inside `shot@x` is
  `slice@x`). `previous_result:` naming a group is a directed error. `@` is
  reserved in step names; entry names are validated and unique; 32 entries max;
  `release_pipeline`/`release_models` survive on the last member only. The
  realized workflow keeps `for_each`; the manifest names the members. An entry
  of a list-valued variable may reference another variable
  (`"from_file": "variable:character_a_voice"`); `resolve_variable_values`
  (`dw/variables.py`) replaces those once, before `realize_args`, refusing a
  cycle, and `undeclared_variable_references` walks inside list/dict variable
  values too. The catalog derives `lists` (`list_fields`, `dw/for_each.py`):
  the fields an entry takes are the `item:` references the steps make, `name`
  first; an entry key no step reads is a validation warning
  (`entry_field_warnings`). A `cost` entry may carry `per_entry`
  (`{variable, minutes, entries}`), measured, never derived. An empty
  `for_each` list is an error; `expanded_definition` realizes constants first.
- Every run directory holds `workflow.json` beside its manifest: the *realized*
  workflow, with the run's arguments folded into the variable defaults, the seed
  it used, stored prompt text inlined and `output:.../latest/...` pinned to the
  run it resolved to. Written by `realize_workflow` (`dw/realize.py`) at run
  start, best effort. Over MCP, `get_job_workflow` reads it back and
  `save_workflow` names it; `export_job` bundles the run

The same conventions, written for an agent composing a workflow over MCP, are
the `Authoring a workflow from an agent` section of docs/WORKFLOW_GUIDE.md;
change both when one changes.

### LTX-2.5 IC-LoRAs

`templates/ltx2/generative-upscale` was the only IC-LoRA use in the catalog;
three more conditioning templates join it (#151, #152), all through
`LTX2InContextPipeline` + `LTX2ReferenceCondition`, all at
`reference_downscale_factor: 1` (the upscaler's is 2). `reference-sheet`
drives Ingredients — the family's only identity route, and the first two
templates here whose reference is a file the workflow did not make; the sheet
is a still, so a `loop_frames` step (`dw/tasks/video_utils.py`, the video
analogue of `loop_audio`) laps it into the static video the LoRA reads
through its 121-frame bucket. `restore-deblur` and `restore-decompression`
each invert one defect and no other. Every number in the three is the vendor
card's and is pinned by `tests/test_ltx2_ic_loras.py`; the trained caption
form is a *different* genre from a T2V shot caption, so those stored prompts
are tagged `ic-lora` and `tests/test_ltx_prompt_library.py` checks them
against their own convention rather than the 150-220-word paragraph rule.
The weights are `gated: auto` on Hugging Face — per repo, so a box that pulls
one can still 403 on another. A `loras` entry counts toward
`plan.downloads_required` (`_collect_sources`, `dw/plan.py`): it names its repo
under `model_name` directly rather than through `from_pretrained_arguments`, so
the walk used to miss it and a box holding every base weight but not the
IC-LoRA answered `[]` and then pulled it mid-run.

### Quantization Support

Quantization configs are defined per-component in workflow JSON and instantiated in `config_objects.py`. Supported frameworks: BitsAndBytes, TorchAO, GGUF, SDNQ, optimum-quanto. The `config_type` field is a free-form string — new quantization backends work automatically via dynamic import.

SDNQ pre-quantized models use a different pattern: `pre_load_modules` imports sdnq (registers with diffusers), then the entire pipeline loads from the pre-quantized repo. Optional `sdnq_optimize` applies quantized matmul post-load (CUDA/XPU only).

### Cross-Platform Device Support

`dw/__init__.py` handles device detection (CUDA > MPS > CPU) and platform-specific optimizations:
- **CUDA**: TF32 matmul, cuDNN benchmark, deterministic mode (configurable via settings)
- **MPS**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (use all unified memory), autocast warnings suppressed, attention slicing enabled by default
- **CPU**: Warning displayed

Detection is overridden by the `DW_DEVICE` environment variable (single run) or the `device` setting (standing), either of which can name a specific accelerator such as `cuda:1`. Device placement is explicit throughout — no default torch device is set, since that would build models directly in VRAM and defeat offloading. Compare backends with `get_device_type()` rather than `== "cuda"`, which a device like `cuda:1` would fail.

A step can override the device it runs on: `device` in a pipeline `configuration` (also the default for that pipeline's components), in a component `configuration`, or in a task's `arguments`.

Every device a workflow names passes through `resolve_device()`, which translates a backend this machine does not have into the one it does and warns — a `cuda` workflow runs on a Mac and an `mps` one runs on a CUDA box. Only the backend is translated: an index survives when the backend matches (`cuda:1` on a single-GPU CUDA box stays a genuine error) and is dropped when it does not. `cpu` is never rewritten, since pinning a step to the CPU is how a GPU-specific problem gets ruled out. Translation happens before anything reads the backend, so the MPS accommodations (the sequential-offload downgrade, attention slicing, the compile skip) fire for a translated device too.

A `components` entry can additionally set `residency: "on_demand"`, which rests the component on the CPU and wraps its `forward`/`encode`/`decode` to move it to the device around each call (`apply_on_demand_placement` in `pipeline.py`). The wrappers use `functools.wraps` because callers introspect the signature — MiniMax H3's denoiser picks its arguments from `signature(transformer.forward)`. It is mutually exclusive with `group_offload` on the same component, and like `group_offload` it suppresses the wholesale `pipeline.to(device)` at load.

Settings in `~/.diffusers_helper/settings.json`: `device`, `enable_tf32`, `cudnn_benchmark`, `cudnn_deterministic`, `log_level`, `log_filename`.

## Security Rules

All entry points use `dw/security.py`. When adding features:
- Validate paths with `validate_path()` / `validate_workflow_path()` / `validate_output_path()`
- Validate variable names with `validate_variable_name()` (pattern: `^[a-zA-Z_][a-zA-Z0-9_-]*$`)
- Validate URLs with `validate_url()` (http/https only)
- Sanitize subprocess args with `sanitize_command_args()`
- **Never** use `eval()`, `exec()`, or `shell=True`
- Path traversal (`../`) is blocked

CodeQL knows about these validators, which is why the scan is quiet: the local
query pack in `.github/codeql/dw-security/` models them as sanitizers for
`py/path-injection`, because the built-in query recognizes a
normalize-then-check only as a local barrier guard and so cannot see one that
lives in another module and returns the safe value. This is what makes code
scanning useful here rather than 26 identical false positives - but it only
holds while new filesystem access goes through a validator. Reaching the disk
some other way is a real alert, so treat one as a finding rather than as more
of the old noise. `validate_path(path, base)` is modeled as a barrier only when
`base` is not `None`; with `None` it is normalization only, and the path stays
reportable. Scanning is advanced setup (`.github/workflows/codeql.yml`) for the
same reason - default setup cannot load a pack.

## Critical Gotchas

- **Schema validation runs before variable substitution** — variable defaults must match expected JSON types (use `25` not `"25"` for numbers)
- **`previous_result:` references are checked statically too** — once the schema
  passes, `previous_result_reference_errors` (`dw/previous_results.py`) reports any
  literal `previous_result:` or `from_previous_result` naming no *earlier* step, with
  the JSON path it sits at. References otherwise resolve lazily per step, so a step
  renamed in one place and not another failed only when the run reached it, after
  every step before it had generated. The definition is substituted before the check,
  so a reference spelled by a *declared* variable is checked by its value; one spelled
  by an undeclared variable is itself a validation error (below)
- **`for_each` expands before the reference check** — `validation_errors` substitutes
  (the caller's `arguments` when they are all good, else the defaults) and expands
  first, so `gather:` and `item:` errors carry the path of the template step
  (`steps[0].for_each[1].name`). Expansion records each expanded step's *source* index
  (`expand_for_each(definition, source_indices)`), so a reference error always carries a
  path in the file the author wrote, and one inside a member names the member in its
  message. An undeclared `variable:` is a validation error at the path it sits at, not a
  warning and not a complaint about the `for_each` list that did substitute: once a
  `variables` block exists, `replace_variables` refuses an undeclared reference, so it is
  a run that cannot start
- **The two MiniMax cut templates take one `shots` list** — since the stage-2
  rewrite (2026-09-11) `templates/minimax/dialogue-short` and `music-video`
  have no `shot_N_*` variables; a scripted caller passes `shots` (entries
  `{name, prompt, references, num_frames}` and `{name, prompt, start_frame}`).
  The members are `shot@<name>` in the manifest and the gallery. This is the
  breaking change the next release note should name. The CLI and REPL only
  take `name=value` strings, and a string handed to a list variable is
  comma-split - so `shots` can only be supplied over the API/MCP (a JSON
  body); `python -m dw.run` runs the templates' default list
- **A reference name is checked for its shape before the queue, and `@` is
  part of it** — a `for_each` member is `<group>@<entry>` and the files it
  writes carry that `@` in their base name, which `OUTPUT_REFERENCE_PATTERN`
  and `ASSET_REFERENCE_PATTERN` refused: a whole class of files the server
  itself named could not be named back to it, so `output:` on a shot a
  list-driven template produced forced a re-render or an `upload_asset` round
  trip (#162). `@` is safe in a path — not a separator, not `..`, and
  containment is still `validate_path`'s — and a name still may not *start*
  with one. The other half is that the refusal arrived at run time, after the
  queue, from a message that described a *valid* name and never said which
  character it objected to: `reference_name_errors` (`dw/reference_names.py`)
  now checks the shape of every `asset:`/`prompt:`/`output:` reference in the
  definition in `validation_errors`, and `_name_fault` (`dw/security.py`)
  names the offending character and position. Shape only — *existence*
  depends on the workspace and on what pruning has taken, so it stays where it
  was: the validate route resolves the caller's `arguments` against the
  workspace, and the definition's own references resolve at run time
- **Cartesian product explosion** — multiple `previous_result` references multiply: 4 images × 3 masks = 12 iterations
- **Component sharing requires exact key matching** between `shared_components` and `reused_components`
- **Built-in workflows** need explicit argument mapping: `"prompt": "variable:prompt"`
- **MPS differences from CUDA**: no autocast, no bitsandbytes, no flash_attn, no triton, no torch.compile. Model offloading has less benefit on unified memory, and `"offload": "sequential"` is downgraded to `"model"` with a warning there (`place_component`) — per-submodule streaming hands back no residency when the CPU and the accelerator share one pool. `exclude_from_cpu_offload` is sequential-only and does not survive the downgrade.
- **`{}`-escaped strings** in JSON arguments: `"{nf4}"` stays as string `"nf4"`, without braces it would try to load as a type
- **A stored prompt's `text` may not begin with a reference prefix** (`variable:`, `previous_result:`, `constant:`, `asset:`, `output:`, `prompt:`) — the engine rejects it to prevent double resolution or iteration expansion
- **Audio+video muxing**: pipelines that generate audio alongside video (LTX-2) have the two muxed into one `video/mp4` file with PyAV in `result.py`
- **A caller's `arguments` are checked before anything is queued** -
  `argument_errors` (`dw/variables.py`) folds them into the declared variables
  exactly as `set_variables` does at the top of a run, so an undeclared name or
  a value that will not coerce is a 400 from `POST /api/jobs` rather than a
  job that fails on its first step, and `POST /api/validate` takes the same
  `arguments` (plus an `asset:`/`prompt:`/`output:` existence check against
  the workspace) so the free pre-flight covers the part the caller wrote.
  A workflow that declares no variables takes no arguments at all - those were
  dropped in silence, since `Workflow.run` only substitutes when a `variables`
  block exists. A valid `POST /api/validate` answer also carries `plan`
  (`dw/plan.py`): the fingerprint of the work, step and list counts,
  `downloads_required` and a cost `estimate` with its `basis` - the number an
  agent quotes, with `basis` saying whether it was measured for this list
  (`catalog`/`per_entry`) or extrapolated over one the caller resized
  (`derived`); `plan: null` when it could not be built, never a changed
  verdict. `acknowledged_cost` on `POST /api/jobs` / `rerun` takes `true`
  (recorded) or the plan's `{fingerprint, minutes, downloads}` (checked - 409
  with the current plan when the fingerprint or the required downloads
  changed; `minutes` never compared), and the job records `acknowledged:
  none | boolean | bound`. `cached_steps` is the worker's answer to a
  `probe_cache` command (`Workflow.cache_hits`, which shares
  `_prepare_definition` / `_cache_lookup` with `run` so the two cannot drift).
  The web UI reads the fields only: the editor lists the plan under a valid
  verdict (`describePlan`, `ui/src/lib/plan.ts`), and a job queued `bound`
  says so on the job page and in the jobs list; the UI itself sends no
  acknowledgement
- **A failed run still reports what it wrote** — the worker carries its partial
  manifest on the error and cancelled messages as well as on success, and the
  "Previous result not found" error names the steps that ran even after
  `release_unreferenced_results` has dropped their results
- **Run directories**: each execution writes `<output_dir>/<workflow identity>/<run id>/`
  with a `manifest.json` beside its files (`dw/runs.py`, `Workflow.effective_output_dir`).
  Identity is the workflow's path under a `workflows/` tree, else its file name, else its
  `id`; the run id is `<UTC timestamp>-<8 hex of the spec>`, with a `-N` counter if taken.
  A sub-workflow inherits the parent's run directory and writes no manifest of its own.
  `--output-layout flat` / `DW_OUTPUT_LAYOUT` / the `output_layout` setting restores the
  old layout. The gallery groups a workflow's runs under one folder by stripping the run
  id (`strip_run_id`). The realized workflow is written into the same directory as
  `workflow.json` (`dw/realize.py`, `write_realized_workflow`), and the manifest's
  `workflow` block carries `realized`, `prompts` (the stored prompts inlined) and
  `sub_workflows` (path -> SHA-256). A job records the run it was
  (`run_id`/`run_dir` on `Job` and in `jobs.sqlite`), which is how
  `JobManager.realized` finds the file. `exports` is a reserved workspace name:
  `POST /api/jobs/{id}/export` gathers one finished job into
  `<workspace>/exports/<job id>/` and `GET /exports/<job id>.zip` streams it.
- **Result subfolders**: a step's `result.subfolder` (`dw/subfolders.py`) puts its files
  in a subfolder of the run directory - `<run>/final/x.mp4` - by convention `final` or
  `intermediate`; the engine treats no name specially and there is no default.
  `Workflow.step_output_dir` computes the directory once and hands it to both
  `Result.save` and the pipeline wrapper, so a chain's `save_segments` spill follows it.
  Shape is `SUBFOLDER_PATTERN` (the `output:` segment rule, so a subfolder is
  `output:`-addressable up to `OUTPUT_REFERENCE_PATTERN`'s seven-segment ceiling),
  checked by `subfolder_errors` in `validation_errors` after
  `for_each` expansion and again at run time; containment is `validate_output_path`
  against the run directory. Manifest entries and `step_end` carry `subfolder`.
  `split_run_path` finds the run id anywhere in a path, so `strip_run_id` still groups a
  workflow's runs. Gallery entries carry it too; `GET /api/gallery?subfolder=` and MCP
  `list_gallery(subfolder=)` filter on it. The web UI reads the field only:
  the gallery page offers a subfolder pick once any entry has one, and the
  job page sections results under `final/` / `intermediate/` headings (or
  whatever the step named) (`sectionBySubfolder`, `ui/src/lib/results.ts`),
  unchanged for a run that chose none. `file_base_name` may not contain a
  separator - it is a name, not a path - and it *replaces* the derived
  `<workflow id>-<step name>.<index>` base rather than prefixing it (#100), so
  two steps in one subfolder that set the same one collide onto
  `output_file_path`'s `-2` counter.
  Every `workflows/templates/**` file with two or more saving steps
  marks each one `final`/`intermediate` (`tests/test_template_subfolders.py` pins the rule;
  `dw/workflows/` builtins stay unmarked - a role is the parent's to assign). That moved
  the templates' outputs into `<run>/final/` and `<run>/intermediate/`: an
  `output:<template>/latest/x` reference keeps resolving but stops advancing past the last
  pre-change run (`keep_output` is the stable form), and a seeded template's first run
  after the change regenerates rather than hitting the step cache (the key includes
  `result`). Those two, and a stray `subfolder` key becoming live, are the release-note
  items beside the `shots` list change. Gallery names for a template's runs now read
  `<template>/<run id>/final/<file>`, so an `output:` reference built from one carries the
  `final/` segment
- **A task argument's numeric domain is declared, not inferred** — a task
  command's argument schema is its implementation's signature, which says
  nothing about range, so `dw/task_domains.py` declares the domains that are
  not a judgement call (a count or a rate above zero, an offset zero or above)
  and `validation_errors` reports a literal outside one at its JSON path. The
  commands check the same table at run time (`check_arguments`), which is the
  only layer that sees a value arriving from a `variable:` or an earlier step.
  Both defects it closed were silent successes rather than failures:
  `slice_audio(num_frames=-10)` reached Python's slice semantics and returned
  the track minus its last ten frames (#139), and
  `resample_audio(target_sample_rate=0)` left the samples alone and then hit
  `DEFAULT_AUDIO_SAMPLE_RATE` at save, writing a 44100 Hz header over a 32 kHz
  waveform (#140) — which is why `_as_track` now refuses a non-positive rate
  outright: relabelling a waveform changes its speed and pitch, and the save
  default makes a missing rate look like a valid one. Adding a domain means
  one entry in the table; `tests/test_task_domains.py` pins every entry to a
  real parameter of a real command so a rename cannot leave one checking
  nothing
- **`cost` is curated, `observed` is derived, and they are different fields** —
  `dw/workflow_schema.json` defines `cost` as *"Never derived"*, so nothing
  writes one; `dw/server/observed_cost.py` reports a sibling built from this
  box's own `jobs.sqlite` rows (#93). Four rules, each a way the naive median
  would lie: runs are bucketed by the workflow's declared `cost_drivers` (a
  list driver on its *length*, so two four-shot runs are comparable however
  different their prompts) and the bucket reported is the one the *defaults*
  give, keeping it comparable to a curated figure; `cold_minutes` and
  `warm_minutes` are separate, each with its own run count, and only the cold
  one is comparable to `cost` (wall clock including model load); a run whose
  every manifest entry is `reused` wrote nothing and is excluded; and a run
  whose persisted events hit `MAX_PERSISTED_EVENTS` without a `loading` phase
  is `unclassified_runs` rather than assumed warm. Everything comes off the
  job row in one query, so a figure survives a pruned run directory, and the
  aggregate caches against `JobHistory.watermark()` rather than a file mtime —
  a job landing changes every figure and changes no file. The compact listing
  carries only `observed_minutes`/`observed_runs` (#101 budget); the full
  block is in the full listing and `GET /api/workflows/{name}/variables`. The
  raw `GET /api/workflows/{name}` is left verbatim, since the editor saves
  what it reads back. A `cost_drivers` entry naming no declared variable is
  dropped, and `tests/test_observed_cost.py` sweeps the catalog for one.
  `plan.estimate` quotes the observed figure ahead of the curated one
  (`basis: "observed"`, with `runs`) — `basis: "unknown"` has to mean nobody
  has a number, not nobody curated one (#154). Only the *cold* median, only
  when the history is this backend's, and only for the bucket the caller's
  own arguments fall in (`ObservedCosts.observed(name, definition,
  arguments)`); a resized list finds no bucket and falls back to the curated
  figure. Nothing is added for a composed child, since an observed run
  already ran it. An inline definition has no catalog name, so no history
- **An observed figure below three runs is not the same statistical basis as
  a dozen, and says so** — a single-run `observed_minutes` was quoted at the
  same authority as a twelve-run one, and ran ~3x pessimistic doing it
  (#301). Below `SMALL_N_THRESHOLD` (3) runs, `estimate()`'s `_tempered`
  (`dw/plan.py`) blends the observed minutes toward the workflow's curated
  `cost` when one exists — proportional to how thin the history is, one run
  counting for a third of the blend — rather than quoting the raw point
  figure; where no curated figure exists to blend toward (including the
  #268 child-rollup case, which has none by construction), the minutes are
  left alone and `low_confidence: true` is added to the estimate instead, so
  a caller has something machine-checkable beyond having to know to inspect
  `runs` itself. No new range/uncertainty-band math — that was considered
  and rejected as more surface than the problem needs
- **An H3 adapter is checked against the partition its step denoises on** —
  `ref2va` loads `transformer_ref` alone, so diffusers puts whatever
  `lora_weight_name` names straight onto it: an FL2VA turbo LoRA on a
  reference step runs, succeeds, and only retains identity worse (#149,
  #155). `dw/adapter_compatibility.py` refuses the mispairing in
  `validation_errors` (so `POST /api/validate` and the pre-queue check both
  catch it, at `arguments.<name>` when the caller supplied it) and *warns*
  on a file name carrying neither `ref2v` nor `fl2v` — the name of a future
  reference-trained checkpoint cannot be predicted, so the escape hatch
  stays open while the one documented mistake is closed. The workflow names
  and the partition each denoises against are diffusers'
  (`MiniMaxH3Blocks._workflow_map`, pinned by `tests/test_h3_adapters.py`);
  the file-name convention is MiniMax's and is swept against the catalog's
  own defaults
- **An elided step says whether anyone decided it** — `warn_elided` used to
  tell every caller their reference was probably misspelled, including the
  one who deliberately passed `singer_reference` and so bought the elision
  `music-video` advertises (#146, #157). `overriding_variables`
  (`dw/elision.py`) compares the definition as *written* against the
  substituted steps: a step reached only through a variable whose value no
  longer names it was replaced on purpose, and its record carries
  `overridden_by` and drops the diagnosis. A variable no step reads is not
  how the step was reached, so that case keeps the old wording
- **A deliverable with no audio headroom warns** — a track at or above
  −0.5 dBFS is written anyway and said out loud (`warn_without_headroom`,
  `dw/result.py`, kind `audio_no_headroom`), for both a saved audio file and
  a muxed video: a clipped file succeeds, and a consumer that cannot listen
  had `peak_dbfs` with no rule to read it against — `get_gallery_metadata`'s
  hint taught the near-silent end of the range only (#158). A warning, not a
  gain change: what level a deliverable sits at is the workflow's to decide,
  and `normalize_audio` is the step that decides it. The two Music 3
  templates decide it now (#159) - `music` and `music-video` peak-normalize
  to -1 dBFS, the level `assemble-and-score` has always used, because the
  warning was firing on their own defaults every run. `music-video`
  normalizes only the track going into the mux, not the slices that condition
  the shots, so the picture is unchanged; `music`'s deliverable moves to the
  new `balanced` step, which renames the file an `output:` reference names
- **A deliverable is measured as written, not as handed to the writer** —
  `warn_without_headroom` reads the waveform, and the encoder sits downstream
  of it: a song normalized to exactly -1.0 dBFS came back out of
  `music-video`'s AAC mux at **+0.94**, so a clean default run shipped a
  clipped file and nothing warned (#161). `warn_if_written_above_full_scale`
  (`dw/result.py`, kind `audio_clipped`) probes the file it just wrote and
  warns when it decodes at or above 0 dBFS — whatever the encoder did, that
  is the number a consumer's decoder sees. Only for a file that can carry a
  soundtrack, and silent when `warn_without_headroom` already spoke for that
  file, since two warnings would be two answers to one mistake. The encode's
  overshoot is material-dependent — about 0.1 dB on an mp3 and about 1.9 dB
  on the AAC mux of the same song — so no target chosen up front can be
  *known* to be enough, which is why reading the file back is the half that
  stops the next instance. The half that fixes this one: every template
  whose deliverable ends in a `pair_audio` mux (`music-video`,
  `assemble-and-score`, `dissolve-between-shots`) normalizes to **-3 dBFS**;
  `music`, an mp3, keeps -1
- **A variable's bound is declared by the author, checked three times** — a
  model's own rule about a value (H3's `num_frames` is `17 * n + 5` from 124
  to 345) is a property of the model, so it lives in the workflow rather than
  in engine code, as a `variable_constraints` entry (`dw/variable_constraints.py`).
  One shape, not two: it takes a chain step's `frame_snap` field names, and a
  chain writes `"frame_snap": "constraint:num_frames"` rather than repeating
  the numbers. `snap: "up"` rounds an off-grid value to the next legal one
  and warns (at validation *and* through `emit_warning`, so it reaches the
  job's `warnings`); without `snap` an off-grid value is refused. The bounds
  hold for the value the run will use, matching diffusers' own
  `align_num_frames`, which snaps before it range-checks — so 108 is accepted
  (it becomes 124) and 346 refused (it would become 362). LTX-2.5's templates
  declare the `8 * n + 1` grid with *no* `snap`, because those pipelines floor
  an off-grid count rather than raising: rounding up here would be a second
  silent change to the length. Checked in `validation_errors` (so
  `POST /api/validate`, `validate_workflow` and the pre-queue check all
  refuse it at `arguments.<name>` / `variables.<name>`), at run time in
  `apply_constraints` before anything loads, and reported beside the default
  by `list_workflows` (terse) / `get_workflow(variables_only=true)` — that
  last part is what stops the next consumer picking 61 (#96).
  `tests/test_variable_constraints.py` sweeps the whole catalog and pins every
  declared number to the diffusers symbol it derives from. A constraint key is
  a plain variable name and is matched wherever a value by that name sits -
  top-level variable *or* a field of a `for_each` entry (#145), the latter only
  where a step consumes that field as `item:<name>` (`entry_constraint_fields`),
  so the bound follows the value into the pipeline argument rather than the
  name into the JSON. An entry violation is reported at
  `arguments.shots[0].num_frames`, and the rule is reported beside the field in
  the catalog's `lists` block as well as in `constraints`
- **Step cache**: a process-wide singleton (`dw/step_cache.py`) consulted by every `Workflow.run`, including server jobs; entries are keyed by `(workflow id, step name)` and validated against the output
  *root*, never the per-run directory - a run directory is new every execution and would
  defeat the cache; disabled entirely when the workflow sets no `seed`; a hit reports the earlier run's files with `reused: true` and writes nothing new; `memory clear` drops it. This is why "Run again" on a seeded workflow finishes instantly and generates nothing - the job page says so when every step was reused, and `POST /api/jobs/{id}/rerun` with `{"new_seed": true}` (MCP `rerun_job(new_seed=True)`) draws a fresh seed into the workflow's seed variable, which is the way to get a different image

## JSON Workflow Structure

The workflow schema is at `dw/workflow_schema.json` — read it for the full structure.

File paths in workflows are relative to the workflow file. Built-in workflows use `"builtin:filename.json"` (resolves to the packaged `dw/workflows/` — distinct from the top-level `workflows/` folder of runnable examples).
