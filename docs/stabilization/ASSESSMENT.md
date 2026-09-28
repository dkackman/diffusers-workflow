# Architecture assessment (2026-09-28)

Repo copy of the shared doc, for agents that cannot open it:
https://claude.ai/code/artifact/4e21fcdc-ab14-4575-9b52-3805e6a66030 . The
shared doc is canonical; this copy is refreshed at each phase gate.

## Verdict

Line-level quality is good; system-level structure is not. Each ticket added
a new module or a new copy of an existing rule and nothing consolidated them,
so copies now disagree - and that drift is where the correctness bugs come
from. Since 2026-08-01: 421 fix / 200 feat / 15 refactor commits; 89 new
engine modules; `validation_errors` is 22 checks, each walking the
definition; one validate request expands the definition 7 times; `app.py` is
a 4,691-line closure with 65 routes; the "unresolved prefix" tuple is
redefined in ~9 modules with differing contents; 5 audio decode entry
points. Bug #415 was fixed twice (`JobManager.submit` and
`worker._handle_execute`) because admission runs at three entry points.

## Bugs (fixed in Phase 0 unless noted)

| # | Defect | Where | Evidence |
| --- | --- | --- | --- |
| B1 | Validation (`expanded_definition`), the run (`_prepare_definition`) and the record (`realize_workflow`) prepare the definition three ways; a constraint-snapped value is validated and recorded pre-snap | `dw/workflow.py`, `dw/realize.py` | fixed, Phase 0 |
| B2 | A step-cache hit still runs `create_step_action`, which loads a released pipeline just for bookkeeping | `dw/workflow.py` | fixed, Phase 0 |
| B3 | `sub_workflow_warnings` returns dicts among string warnings (UI shows `[object Object]`), reports expanded indices, ignores caller arguments | `dw/workflow.py`, `dw/server/app.py` | fixed, Phase 0 |
| B4 | `_prune_detail_cache` iterates a live module dict shared by request threads | `dw/server/app.py` | fixed, Phase 0 |
| B5 | `assign_run_version` is max+1 with no lock | `dw/runs.py` | fixed, Phase 0 |
| B6 | `--mcp` mount shares one client, so `use_workspace` switches every session | `dw/server/mcp_mount.py` | confirmed; by design (single-user mount, stateless HTTP). Revisit in Phase 3 with the router split if multi-agent use of one server becomes a requirement |
| B7 | Step-cache key ignores `pipeline_reference` / `reused_components` sources | `dw/step_cache.py` | fixed, Phase 0 |
| B8 | Worker validates before activating the job's asset root | `dw/worker.py` | fixed, Phase 0 |
| B9 | Media probes in validation decode whole files, uncached, per check (Phase 2) | `dw/media_info.py` + preflight modules | reported |
| B10 | Broad `except Exception` hides warning-pass failures; `submit_job` maps all errors to 400 (Phase 2) | `dw/workflow.py`, `dw/server/app.py` | reported |

## Findings by subsystem

- **Validation:** 22 checker modules with private step loops, prefix
  constants, path attribution and media probes; findings as strings or
  dicts; four rules written in both the check and the task.
- **Execution core:** three prepare paths; `Workflow.run` is 559 lines; run
  state on reusable instance attributes; reference prefixes parsed at ~54
  sites.
- **`result.py` / `pipeline.py`:** 1,830 and 2,395 lines of mixed concerns;
  `result.py` imports upward into `tasks`; model-family names in engine code.
- **Server:** `create_app` closure; admission three times per submit; library
  root precedence written in both `workspace.py` and `app.py`.
- **MCP:** correctly a thin HTTP client; only the shared-session defect (B6).
- **Worker/REPL:** one lifecycle, reply handling written twice.
- **Tasks/DSP:** `audio_utils.py` 2,182 lines with a hand-rolled limiter,
  compressor and biquad; `av.open` in 7 modules.
- **Tests:** 78% behaviour-only, but 116/134 modules imported by path and
  284 `patch("dw...")` targets - in-file refactors are safe, module moves
  are expensive.
- **Docs:** docs churn exceeds code churn; CLAUDE.md files total 964 lines.

## Decisions (2026-09-28)

- Hard freeze until the base is solid and harness guardrails exist: no new
  templates, prompts, tasks, MCP tools or modules. Testers keep running; the
  implementer fixes their bugs through existing code only. The refactor
  itself may change surface (routes, tools, file layout) where the
  consolidation requires it, but adds no feature or functionality the
  refactor does not require.
- The REPL is removed (Phase 1); `WorkerManager` survives as
  `dw/worker_manager.py`.
- `python -m dw.run` becomes a thin client of `dw.serve` (Phase 1).
- Breaking HTTP/MCP changes are allowed in Phase 3 with a version bump.
- Build vs. buy: where a battle-tested, maintained library covers something,
  the custom implementation is deleted. Standing guardrail afterwards.
- Execution: subagent-driven, on `stabilization/phase-N` branches.

## Guardrail principles (installed in Phase 4)

Guardrails are mechanical - CI checks and harness gates that pass or fail -
not more prose in agent context. Metrics ratchet against
`docs/stabilization/baseline.json` (produced by `scripts/arch_metrics.py`):
modules, files over 1,000 lines, functions over 150 lines, reference-prefix
literals outside their owner, test patches of `dw.` paths, CLAUDE.md lines,
duplicate-code blocks, and from Phase 1 on, functions over cyclomatic
complexity 15 and import cycles. Change coupling, package instability and
LCOM4 are gate reports, not gates (ROADMAP.md, "Metrics"). A ticket is done when it works, no metric regressed,
and it added no second copy of an existing rule.
