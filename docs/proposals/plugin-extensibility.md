# Proposal: a plugin mechanism for third-party samplers, pipelines, and tasks

Status: **not committed** - scoping notes for a future decision, prompted by
comparing dw against `Krea2_ComfyUI_Advanced`, a ComfyUI custom-node pack that
hand-implements a model-specific generation recipe (custom solvers, per-stage
LoRA weighting, multi-stage progressive denoising, an automated sweep/scoring
harness). None of that is buildable in dw today without editing dw's own
source. This document scopes what a general, third-party plugin mechanism
would need - one that lets an installed package register new samplers,
pipeline classes, and task commands the way dw's own built-ins are
registered, without those additions requiring changes inside `dw/` itself,
and with the result usable through the MCP server as well as the CLI.

## What already works, unmodified

dw has one real extension mechanism already: dotted-path class resolution,
gated by an allowlist. It covers more of the surface than expected.

- **Dotted `*_type` resolution** (`dw/type_helpers.py:18-31`) - any
  `*_type` / `*_dtype` / `dtype` / `config_type` key resolves via
  `importlib.import_module` on the dotted path given, not restricted to the
  `diffusers` package at the call site.
- **Custom `scheduler_type` / `component_type`**
  (`dw/pipeline_processors/pipeline.py:1192`, `:1518`) - both are consumed
  generically (`.from_config(...)`, `.from_pretrained(...)`), with no
  hardcoded restriction to diffusers classes. A third-party `SchedulerMixin`
  or `DiffusionPipeline` subclass runs as-is.
- **`pre_load_modules`** (`dw/pipeline_processors/pipeline.py:235-239`) -
  imports a list of modules for side effects before pipeline load. This is
  the registration hook a plugin would use, identical to the pattern SDNQ
  already uses to register itself with diffusers.
- **The trust gate** (`dw/security.py:84-117`) - `TRUSTED_TOP_LEVEL_PACKAGES`
  allowlists top-level package names; anything outside it requires the whole
  process to run with `--trust-workflows`. This is global, not per-package.

Net effect: a plugin author can already ship a package containing a custom
scheduler or pipeline class and reference it by dotted path - but only if the
operator either trusts every workflow on that server, or the plugin ships
inside the `dw` namespace itself (as `dw/community_pipelines/` does).

## Gaps

Three things stand between "a class can be swapped in" and "a package
installs like a ComfyUI custom-node pack":

1. **No plugin surface for task commands.** `command` dispatches through
   `_COMMAND_REGISTRY` (`dw/tasks/task.py:24`), populated only by
   `@register_command` decorators written inside `dw/tasks/task.py` itself
   (lines 35-60). There is no entry-point discovery, no plugin directory
   scan. New orchestration logic - a sweep harness, per-stage LoRA blending,
   PNG-embedded-settings extraction - needs a code change inside dw, full
   stop.
2. **Trust is process-wide, not per-plugin.** `DW_TRUST_WORKFLOWS`
   (`dw/security.py:100-117`) is one boolean set at server startup. There's
   no way to vet and allow one specific plugin package without also
   accepting arbitrary dotted-path imports from every other workflow the
   server ever runs.
3. **Multi-stage orchestration is only partly a gap.** Progressive
   multi-stage generation forces ComfyUI into a monolithic custom node
   because nodes are its unit of composition. dw already composes this way
   via `previous_result:` chaining across separate pipeline steps. The
   genuinely missing piece is a new numerical technique inside a stage (a
   new solver, variance repair), which is the task/scheduler-plugin question
   above, not a new orchestration primitive.

## MCP propagation

Would a plugin's new capabilities show up over MCP once installed, or does
`dw_mcp` need its own changes? Checked against the live code - `dw_mcp`'s
tools are thin, generic REST wrappers, not per-capability logic.

| Capability | Surfaces via MCP automatically? | Notes |
| --- | --- | --- |
| New task command | Yes | `list_tasks` / `get_task` re-read the registry live, per request - no `dw_mcp` change needed. |
| Custom class, exported into `diffusers`' namespace | Yes | `list_classes` / `get_class` do live `dir(diffusers)` + `inspect.signature` introspection. |
| Custom class, in its own package | No | Runs fine via dotted-path resolution, but invisible to the catalog - `load_allowed_class` only resolves dotted names against `ALLOWED_MODULES = ("sdnq",)`. |
| Argument validation for a dotted `scheduler_type`/`component_type` | No | `workflow_argument_warnings()` explicitly skips dotted/escaped component types - a bad plugin call only surfaces at job runtime. |
| Trust-state visibility | No | No REST endpoint or MCP tool exposes whether the server was started with `--trust-workflows`; a client only learns by getting a 400. |

## Proposed scope

Two additions, both inside dw proper - `dw_mcp` needs no changes for the
task-command half, and only a small allowlist extension elsewhere.

### 1. Task-command plugin registry

Discover packages via Python entry points (e.g. group `dw.tasks`) at server
startup; each entry point calls the existing
`register_command(name, implementation=..., provided=...)` convention
already used by built-ins. Introspection, schema generation, and MCP
exposure fall out for free because `list_tasks`/`get_task` already read the
registry live.

Touches: `dw/tasks/task.py`, `dw/serve.py`, `pyproject.toml`.

### 2. Per-plugin trust list

Replace the single `--trust-workflows` switch's implicit "trust everything"
with an operator-maintained list of named, installed packages - a
`settings.json` entry or `plugins.json`, optionally pinned by version.
`require_trusted_dotted_name()` and `require_trusted_pre_load_modules()`
check this list in addition to `TRUSTED_TOP_LEVEL_PACKAGES`.

Touches: `dw/security.py`, and `dw/introspection.py` (extend
`ALLOWED_MODULES` to read the same list, closing the catalog-visibility gap
from the MCP table above).

## Open questions

- Should a trusted-plugin package be allowed to register a task command
  whose `implementation` path points outside `dw` and outside itself - i.e.
  can plugins call each other - or should each plugin's commands be
  sandboxed to its own module tree?
- Does `workflow_argument_warnings()` get taught to resolve *trusted* dotted
  names (closing the validation gap above) as part of this work, or is that
  separable follow-up?
- Is a version/hash pin on the per-plugin trust list worth the maintenance
  cost, or is package-name trust (as `TRUSTED_TOP_LEVEL_PACKAGES` already
  does) sufficient given dw already assumes a locally-controlled Python
  environment?
- Should plugin discovery be eager at server startup only, or does the REPL
  / one-shot `dw.run` path need the same entry-point scan?

Citations are file:line references at the time of writing and may drift.
