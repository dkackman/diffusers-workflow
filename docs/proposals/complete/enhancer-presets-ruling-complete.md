# Where enhancer presets live: a ruling, not a migration (#696)

Written by model `claude-opus-5-5` via provider `anthropic`, at close-out
2026-10-08, from plan v2 on #696 and the stage thread (#781).

## The ask

The 2026-10-07 develop review asked whether `dw/server/enhancers.py`
`PRESETS` (an `"h3"` preset with `intended_models: ["minimax-h3"]`, and a
`"t2i"` preset naming `z-image`/`flux`) is UI wiring or model knowledge
leaked into server code, against CLAUDE.md's rule that model knowledge lives
in `plugins/dw/` and the catalog. If the latter, the fix was to move each
preset into its builtin's metadata and discover `PRESETS` from
`dw/workflows/`.

Verdict: **don't build the migration; record a ruling.** `PRESETS` is the
Enhance panel's menu: a label, the curated LLMs, preselect hints, and which
builtin or task each entry runs. The model knowledge it points at, the
Context-IR spec, already lives in `dw/workflows/h3_context_ir.json`.

Don's answers (2026-10-08): Q1, move `_T2I_SYSTEM_PROMPT` into a
`t2i_augment.json` builtin - **no**, it stays in Python. Q2, build the full
migration anyway - **no**.

## Why not the migration

- **Value was low.** No issue or field report had asked for an enhancer for a
  new family, and none of the server's 104 retained jobs was an `enhance_*`
  job.
- **Inertia was high.** `dw/workflow_schema.json` is
  `additionalProperties: false` at the top level, so discovery meant a new
  top-level workflow key (`enhancer`) every authoring agent and the workflow
  guide would have to learn, beside `shape`/`traits`/`summary`. It also
  needed an ordering field, since the UI preselects `presets[0]`
  (`ui/src/lib/enhancer.svelte.ts`).
- **Poor reversibility.** Removing a schema key that builtins carry is a
  `breaking-change`; the ruling is undone by reopening #696.
- **The acceptance couldn't be met without a break.** The `"h3"` default
  sits in `dw/server/routes/library.py`, `dw_mcp/prompts.py` and
  `dw_mcp/tools_authoring.py`, pinned by two tests. Keeping the key leaves a
  family name in `dw/server/`; renaming it breaks the MCP surface.

## What was built

| Stage | Issue | Merge on `develop` | What it did |
|---|---|---|---|
| 1 record the ruling | #781 | `5aa376b1` (merge of `c1332ef8`) | A new *Prompt enhancement* row in `docs/ARCHITECTURE.md` (owner `enhancers.py`: `PRESETS`, `_T2I_SYSTEM_PROMPT`; enforced by `tests/test_prompt_library.py` and `tests/test_server.py::TestEnhance`), and the same rule in the `enhancers.py` module docstring. No code changed. |

The tester added two regression cases that pin the MCP surface the ruling
leaves alone: **C-F376** (`list_enhancers` returns `h3` then `t2i`, every
field pinned) and **C-F377** (`enhance_prompt` with no `preset` builds the
same job as `preset="h3"`; `t2i` builds a different one; a missing
`acknowledged_cost` and an unknown preset are refused before queueing).
Both passed on mini-ai (mps). The architecture review passed with no
findings.

## Bounces

- **#781: none.**

Cost is left out: the stage comments don't record `usage:` figures. The
plan's estimate was about $1.

## Deferred, and why

- **Discovering presets from builtin metadata** (an `enhancer` block in
  `dw/workflows/`). *Would come back if* a third preset is requested, an
  enhancer is asked for a family other than H3 or t2i, or user/workspace
  libraries are meant to contribute presets. The map row states the trigger.
- **`_T2I_SYSTEM_PROMPT` into a builtin** - declined (Q1). It is
  family-agnostic text; the map row names it as the one tolerated case.
- **Removing `enhance_prompt`/`list_enhancers`** to save MCP description
  budget - raised in the plan as a possible follow-up, not filed. Worth its
  own issue only if the Enhance panel turns out to be unused.

## Design corrections found against the issue

- There was no "Prompt enhancement" row in `docs/ARCHITECTURE.md`; the stage
  added one rather than editing it.
- The `"h3"` default is in three places, not one (`routes/library.py`,
  `dw_mcp/prompts.py`, `dw_mcp/tools_authoring.py`).
- `t2i` is not a workflow: it is `task: text_generation` with a Python system
  prompt, so "moving" it meant writing a new builtin first.
- `tests/test_prompt_library.py` imports `PRESETS` to check `intended_model`
  spellings - the one real model-knowledge consumer, and it stays valid.
