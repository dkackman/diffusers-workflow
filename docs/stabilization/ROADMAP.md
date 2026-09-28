# Stabilization roadmap

The single home for the 2026-09-28 architecture stabilization. Why it exists:
[ASSESSMENT.md](ASSESSMENT.md). Shared view of the same content:
https://claude.ai/code/artifact/4e21fcdc-ab14-4575-9b52-3805e6a66030

Each phase ends at a gate that passes or fails. A phase's detailed plan is
written when the previous gate passes, not before - the code each later
phase works on is what the earlier phases leave behind.

| Phase | Scope | Gate | Plan | Status |
| --- | --- | --- | --- | --- |
| 0 | Freeze, baseline metrics, fix B1-B8 | B1-B5, B7, B8 fixed with regression tests; B6 documented as a design limit; deployed to lem; `baseline.json` committed | [phase-0.md](phase-0.md) | done 2026-09-28 (`stabilization-gate-0`) |
| 1 | Metrics v2 first (see below); remove the REPL; one prepare pipeline; one admission service; `dw.run` becomes a thin client of `dw.serve` | Validation sees the definition the run sees; the server admits a request once (one `Workflow`, one expansion); every entry point reaches the worker through the server; ratchets re-baselined | [phase-1.md](phase-1.md) | in progress |
| 2 | Seams in place: `references.py`, validation context + check registry, shared task rules, step cache, typed worker protocol | `validation_errors` is a registry loop; no prefix literals outside `references.py` | written at gate 1 | - |
| 3 | Structural moves: `app.py` routers + services, `LibraryPath`, split `result.py` / `pipeline.py`, one media + dsp module | No module over 1,000 lines, no function over 150; suite and lem smoke green | written at gate 2 | - |
| 4 | Context diet (CLAUDE.md <= 250 lines total) and guardrails installed | Guardrails live in dw CI and the harness; freeze lifted | written at gate 3 | - |

## Metrics

Two kinds, kept small on purpose.

**Ratchets.** These live in `scripts/arch_metrics.py`. Each is lower-is-better, and `--check` against `baseline.json` fails the build when one gets worse.

- Phase 0 set: modules, modules over 1,000 lines, functions over 150 lines, reference-prefix literals, test `patch("dw...")` targets, CLAUDE.md lines, duplicate blocks.
- Added at the start of Phase 1 ("metrics v2"), with a re-baseline in the same commit:
  - **Cyclomatic complexity:** count of functions above 15. Uses ruff's C901, which is already a dev dependency, so nothing new is installed. Today: 21.
  - **Import cycles:** count of strongly connected components larger than one module in the `dw/` + `dw_mcp/` import graph. Target 0. The graph comes from `grimp` via `import-linter`, a new dev dependency; stage C reuses it for the layering contracts, so one library serves both jobs.

**Gate reports.** These are produced at each phase gate by `scripts/arch_report.py`, which lands in Phase 1. They are written into the Gate reports section below. They are reports, never gates.

1. **Change coupling:** file pairs that change together in `git log`, measured as shared commits and the coupling degree, plus hotspots (churn × total complexity per file). This is the signal for a rule living in two places, like the #415 pair of `server/jobs.py` and `worker.py`. It is a short git-log script, because code-maat is a separate Java tool.
2. **Instability per package:** Ca, Ce and I = Ce / (Ca + Ce), from the same grimp graph. The engine core should be stable (low I) and the server volatile (high I).
3. **LCOM4:** on `Workflow`, `Pipeline`, `Result`, `JobManager` and the app factory. It is a before/after check that the Phase 3 splits improved cohesion. It is hand-rolled over `ast` (about 40 lines), since there is no maintained library for it; this is the one build-vs-buy exception.

Deliberately not adopted: function points, maintainability index, Halstead, coverage %, and comment density.

**Every gate report carries the full metrics table.** It has one column per gate, starting with "before Phase 0" (`3afd70e9`), so the trend reads left to right. The rows are:

- every ratchet;
- functions over cyclomatic complexity 15 and over 30;
- the complexity distribution (function count, median, mean, and how many functions exceed 10, 15, 20 and 30);
- the ten most complex functions, with file:line.

`scripts/arch_report.py` (Phase 1) produces it for any commit or tag. Until then it is computed by hand, the way gate 0's was.

Each gate is tagged `stabilization-gate-N`, so any report can be recomputed for an earlier gate. Gate 0's reports are computed from its tag once `arch_report.py` exists.

## Gate reports

(Filled at each gate: the metrics diff against the previous gate, the top change-coupling pairs and hotspots, package instability, and LCOM4 for the tracked classes from Phase 3 on.)

### Gate 0 (2026-09-28, develop 6a88c746)

- Ratchets: every Phase 0 metric equal to `baseline.json` (`--check` exit 0). Values: 133 modules, 10 over 1,000 lines, 19 functions over 150 lines, 93 prefix literals, 285 test patch targets, 964 CLAUDE.md lines, 21 duplicate blocks.
- B2 on lem (RTX 3090), `templates/ltx2/two-stage` rerun with the same seed: 78.8 s before Phase 0, **0.77 s** after. A cold run is about 160-174 s throughout.
- Two gate-time follow-ups came from that real-GPU timing, not from the tests:
  - Task 11: a cached step defers its pipeline load until a step that actually runs borrows it.
  - Task 12: borrowed-pipeline cache keys are hashed from the definition as written, because `Pipeline.load` edits its definition in place.
  Mocks could not see either. A real-model timing stays in every gate.
- Change-coupling, instability and LCOM4 reports for this gate are computed from the tag once `scripts/arch_report.py` lands in Phase 1.

| Metric (lower is better) | Before Phase 0 (`3afd70e9`) | Gate 0 (`fb03dae3`) |
| --- | --- | --- |
| Engine + MCP modules | 133 | 133 |
| Modules over 1,000 lines | 10 | 10 |
| Functions over 150 lines | 19 | 19 |
| Functions over cyclomatic complexity 15 (ruff C901) | 21 | 21 |
| Functions over cyclomatic complexity 30 (ruff C901) | 4 | 4 |
| Duplicate-code blocks, cross-file (pylint symilar) | 21 | 21 |
| Reference-prefix literals | 93 | 93 |
| Test `patch("dw...")` targets | 284 | 285 |
| CLAUDE.md lines, all files | 964 | 964 |
| Import cycles | not measured | not measured (Phase 1) |

The two columns match because Phase 0 fixed correctness, not structure. The 285 includes the metrics script's own test fixture.

Complexity distribution at gate 0 (mccabe, 1,485 functions): median 3, mean 4.2. 69 functions are over 10, 27 over 15, 10 over 20 and 4 over 30. mccabe counts nested functions inside their parent, which is why it reports 27 over 15 where ruff reports 21. ruff scores each function separately, and it is the ratchet.

The ten most complex (mccabe, with nested functions included):

| Complexity | Function |
| --- | --- |
| 510 | dw/server/app.py:747 `create_app` |
| 79 | dw_mcp/server.py:65 `build_server` |
| 41 | dw/repl_commands.py:644 `WorkflowCommands._workflow_run` |
| 35 | dw/workflow.py:1279 `Workflow.run` |
| 28 | dw/arguments.py:103 `realize_args` |
| 27 | dw/result.py:811 `Result.save_artifact` |
| 24 | dw/media_info.py:19 `probe_media` |
| 24 | dw/locations.py:612 `_walk` |
| 22 | dw/tasks/loop_bed.py:397 `find_loop_bed` |
| 21 | dw/teacache.py:89 `_create_flux_teacache_forward` |

## Working rules for the duration

- Hard freeze: no net-new features or functionality. The refactor may change
  surface (routes, tools, file layout) where consolidation requires it, and
  nothing more.
- Refactor work happens on `stabilization/phase-N` branches, merged to
  `develop` and deployed to lem at each gate.
- The harness keeps running testers and field-bug fixes (freeze prompt:
  [harness/stage-a-freeze.md](harness/stage-a-freeze.md)). It does not touch
  files listed in [hot-zone.txt](hot-zone.txt), which names what the current
  phase is restructuring.
- Metrics: `scripts/arch_metrics.py` (Phase 0, Task 1) is the one source of
  numbers for the phase gates here and for the harness's ratchet later.
- No new count-pinning tests; every new test fails before its fix.
