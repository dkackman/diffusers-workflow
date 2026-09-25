# Proposal backlog: benefit vs. complexity ranking

Written 2026-09-20, after auditing every file in `docs/proposals/` against the
current codebase (fully-implemented proposals were deleted; the proposals
found partially implemented are split into `*-complete.md` design docs plus
`*-partial.md` remaining-work trackers alongside this file). Ranking is by
benefit vs. added complexity/risk, highest ROI first. Updated 2026-09-20
(second pass) after the `tier1-proposals` branch shipped four of the five
original Tier 1 items and both fully-finished proposals (`score-and-select`,
`script-to-video-agent-skill`) were removed.

Since 2026-09-23 every open item below is also a GitHub issue labeled
`feature` (#374, #377, #379, #380, and #244 for `resume.md`; #375 was declined, #376 and #378 shipped), parked with Don. Its
`priority:N` label mirrors the tier here. Work starts from the issue.

## Tier 1 — do these first (small, scoped, clear payoff)

None open. The last item, the H3 video mux headroom warning, shipped. Its
remaining deferred fix is recorded in
`complete/h3-video-mux-headroom-warning-complete.md`.

## Tier 2 — solid ROI, moderate scope

2. **orphaned-run-directories.md** — real, recurring disk-usage annoyance
   (leftover manifests invisible to gallery/asset listings); the proposal
   already recommends the simple option (A). Moderate but bounded work.
5. **mcp-job-notifications.md** — improves reliability of the wait/poll loop
   (cursor-based, `failure_kind`), but there's no reported live pain forcing
   this yet; medium complexity touching the event/job-record schema.

## Tier 3 — high benefit, but big lifts (stage carefully, don't take all at once)

7. **resume.md** — meaningful for expensive multi-step runs that crash, but
   rehydrating the step cache from disk manifests is a correctness-sensitive
   engine change (step identity matching, partial-state edge cases). High
   complexity.
8. **sweeps-and-comparison.md** — valuable once doing real side-by-side
   comparisons, but it's a new job-schema field, batch semantics, and a new
   UI page. Its own doc says "nothing here should be implemented without a
   fresh look."
9. **maintenance-screen.md — the full UI page** — Phase 0 (WAL mode) shipped
   2026-09-20; the maintenance/observability page itself (orphan listing,
   disk usage, job pruning) is the remaining, much bigger ask.

## Shipped since the ranking

- **Closing the xfail security tests** (#407, stages #409-#413), shipped
  2026-09-24. Record, including what was deferred (a UI Content-Security-Policy,
  Playwright in CI): `complete/xfail-security-tests-complete.md`.

## Declined

Kept in `declined/` with the reason at the top, so a revival starts from
the analysis rather than repeating it.

- **declined/workspace-folders.md** — grouped workspace names (`QA/EP1`).
  Declined 2026-09-23 on #375: thin demonstrated value against a loosened
  security boundary and a change that can't be taken back.

## Backlog ideas with no doc on file

- **A larger Qwen-Image catalog entry** — the 20B, Apache-2.0 Qwen-Image
  checkpoint (distinct from the smaller Qwen-Image-2.1 onboarded
  2026-09-20) would be a real catalog expansion. No design doc exists for
  it yet; write one before starting.
