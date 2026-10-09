# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`diffusers-workflow` is a declarative workflow engine for HuggingFace Diffusers. Users define AI image/video generation pipelines in JSON — variable substitution, multi-step composition, cross-step data flow, and utility tasks — without writing Python. Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU.

## Common Commands

```bash
# Install
bash ./install.sh && source ./activate

# HTTP server + web UI (http://127.0.0.1:8765, API docs at /docs)
python -m dw.serve

# Run a workflow - dw.run only queues the job on a running dw.serve. This
# template uses a small, ungated model, so it needs no Hugging Face login
python -m dw.run workflows/templates/text-to-image.json
python -m dw.run workflows/templates/text-to-image.json prompt="a cat" num_images_per_prompt=4

# A gated model (e.g. workflows/models/flux-dev.json) needs Hugging Face auth
# first: huggingface-cli login

# Validate a workflow against schema
python -m dw.validate workflows/models/z-image.json

# Unit tests - parallel across every core (pytest-xdist, `-n auto` in
# pytest.ini); naming test files runs them in one process
pytest tests/ -v
pytest tests/test_security.py -v

# System test - downloads SD 1.5 (a few GB) and generates one image
python -m dw.test
```

## Where things are

`docs/ARCHITECTURE.md` is the map: concept, owning module, the rule that holds across
the seam, and the test or check that enforces it. Open it before grepping - it names the
module, and that module's docstring holds the detail.

- `dw.serve` runs every job in a persistent spawned worker (one per `--devices` card) (`dw/worker.py`, managed by
  `dw/worker_manager.py`) that keeps models cached between runs, so a change to engine
  code needs a server restart before a job sees it.
- The packaged `dw/workflows/` is what a `builtin:` step names (resolved in
  `dw/library.py`); it is not the top-level `workflows/` folder of runnable examples.
- The workflow schema is `dw/workflow_schema.json` - read it for the full structure.
- The HTTP server is `dw/server/` (see `dw/server/CLAUDE.md`), the web UI is `ui/` (see
  `ui/CLAUDE.md`), and the stdio MCP server is `dw_mcp/` (see `dw_mcp/CLAUDE.md`).

The guides in `docs/`, by topic:
- `WORKFLOW_GUIDE.md` - writing workflow JSON. It owns the reference conventions
  (`variable:`, `previous_result:`, `constant:`, `asset:`, `output:`, `prompt:`, `item:`,
  `gather:`): its *Authoring a workflow from an agent* section, *References* first, and
  the *Type System* section.
- `WORKSPACES.md` - workspaces, the library search paths, run directories and manifests.
- `SERVER.md` - the HTTP API and web UI; `REMOTE.md` - using a server on another machine.
- `MCP.md` - the MCP tool surface; `AGENT_LOOP.md` - the automated implementer/tester loop.
- `WORKER_GUIDE.md` - the persistent worker and its queue protocol.
- `TASKS.md` - the utility task commands (`dw/tasks/`).
- `ACCELERATION.md` - caching, compile, attention backends, device settings, MPS;
  `QUANTIZATION.md` - per-component quantization backends; `RECIPES_24GB.md` - tested
  combinations of both per model family.
- `LORAS.md`, `IP_ADAPTER.md`, `PROMPT_WEIGHTING.md` - those features, one each.
- `SECURITY.md` - the validators and trust gate in depth; `SECURITY_QUICKREF.md` - the
  same at a glance.
- `TESTING.md` - the test suite; `DEPENDENCIES.md` - installation; `RELEASING.md` - releases.

### Claude Code plugin

`.claude-plugin/marketplace.json` publishes the `dw` plugin in `plugins/dw/`: one
composition skill per model family (`minimax-h3`, `minimax-music3`, `ltx-2.5`,
`kandinsky-6`) that chooses a template for a request's shape and states the family's
hard rules, plus
cross-cutting composition skills (`script-to-video`, `series-episodes`) - shapes above
the families that orchestrate the decision trees and cast consistency across multiple
generations. Every skill the directory holds is named in `plugins/dw/README.md` and here,
pinned by `tests/test_plugin_skills.py`; that README says how the rest is pinned and
versioned.

## Security Rules

All entry points use `dw/security.py`'s validators (paths, URLs, subprocess arguments) and the trust gate is `dw/trust.py`. When adding features:
- Validate paths with `validate_path()` / `validate_workflow_path()` / `validate_output_path()`
- Validate variable names with `validate_variable_name()` (pattern: `^[a-zA-Z_][a-zA-Z0-9_-]*$`)
- Validate URLs with `validate_url()` (http/https only)
- Sanitize subprocess args with `sanitize_command_args()`
- **Never** use `eval()`, `exec()`, or `shell=True`
- Path traversal (`../`) is blocked

CodeQL models these validators as sanitizers (`.github/codeql/dw-security/`; the map's
*CodeQL path-injection model* row). That only holds while filesystem access goes through
one: reaching the disk some other way is a real alert, so treat one as a finding, not
noise. `validate_path(path, base)` is a barrier only when `base` is not `None`; with
`None` it is normalization only, and the path stays reportable.

## Before you edit

- **Model knowledge lives in `plugins/dw/` and the catalog, never in engine code.** A
  model's rule about a value is declared in the workflow (`dw/variable_constraints.py`).
- **Compare backends with `get_device_type()`, not `== "cuda"`** - a device like
  `cuda:1` fails the string compare (`dw/__init__.py`).
- **`{}` keeps a value a string**: under a `*_type`/`*_dtype`/`dtype` key `"{nf4}"` stays
  `"nf4"`; unbraced it is loaded as a type, and fails at load time, after validation has
  passed (`docs/WORKFLOW_GUIDE.md` *Types and escaping*; `dw/arguments.py`).
