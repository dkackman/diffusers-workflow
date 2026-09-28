# Stabilization roadmap

The single home for the 2026-09-28 architecture stabilization. Why it exists:
[ASSESSMENT.md](ASSESSMENT.md). Shared view of the same content:
https://claude.ai/code/artifact/4e21fcdc-ab14-4575-9b52-3805e6a66030

Each phase ends at a gate that passes or fails. A phase's detailed plan is
written when the previous gate passes, not before - the code each later
phase works on is what the earlier phases leave behind.

| Phase | Scope | Gate | Plan | Status |
| --- | --- | --- | --- | --- |
| 0 | Freeze, baseline metrics, fix B1-B8 | B1-B8 fixed with regression tests and deployed to lem; `baseline.json` committed | [phase-0.md](phase-0.md) | planned |
| 1 | Remove the REPL; one prepare pipeline; one admission service; `dw.run` becomes a thin client of `dw.serve` | Validation sees the definition the run sees; a submit validates once | written at gate 0 | - |
| 2 | Seams in place: `references.py`, validation context + check registry, shared task rules, step cache, typed worker protocol | `validation_errors` is a registry loop; no prefix literals outside `references.py` | written at gate 1 | - |
| 3 | Structural moves: `app.py` routers + services, `LibraryPath`, split `result.py` / `pipeline.py`, one media + dsp module | No module over 1,000 lines, no function over 150; suite and lem smoke green | written at gate 2 | - |
| 4 | Context diet (CLAUDE.md <= 250 lines total) and guardrails installed | Guardrails live in dw CI and the harness; freeze lifted | written at gate 3 | - |

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
