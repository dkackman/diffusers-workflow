# Proposal backlog: benefit vs. complexity ranking

Written 2026-09-20, after auditing every file in `docs/proposals/` against the
current codebase (fully-implemented proposals were deleted; the proposals
found partially implemented are split into `*-complete.md` design docs plus
`*-partial.md` remaining-work trackers alongside this file). Ranking is by
benefit vs. added complexity/risk, highest ROI first. Updated 2026-09-20
(second pass) after the `tier1-proposals` branch shipped four of the five
original Tier 1 items and both fully-finished proposals (`score-and-select`,
`script-to-video-agent-skill`) were removed.

## Tier 1 — do these first (small, scoped, clear payoff)

1. **h3-video-mux-headroom-warning-partial.md** — fixes (1)-(4) shipped
   2026-09-20 (`bbe4adb`); item 5 (updating the M-F008 regression case
   wording against the external suite) is still owed, and fix (2) (a
   `normalize_audio` gain stage on the H3 video templates) stays deferred
   pending a real clipped-in-practice case.

## Tier 2 — solid ROI, moderate scope

2. **orphaned-run-directories.md** — real, recurring disk-usage annoyance
   (leftover manifests invisible to gallery/asset listings); the proposal
   already recommends the simple option (A). Moderate but bounded work.
3. **workspace-folders.md** — one regex relax + a depth-2 listing walk + UI
   grouping. Low complexity, meaningful convenience for the growing
   series-episodes workflow.
4. **mcp-context-cost-partial.md** — cheap and safe, but the remaining
   payoff is small (~1.1k tokens of connect cost), and recommendation 6 in
   `mcp-context-cost-complete.md` suggests the harness-level fix (deferred
   MCP schemas) may already make this moot — worth confirming that before
   spending effort here.
5. **mcp-job-notifications.md** — improves reliability of the wait/poll loop
   (cursor-based, `failure_kind`), but there's no reported live pain forcing
   this yet; medium complexity touching the event/job-record schema.

## Tier 3 — high benefit, but big lifts (stage carefully, don't take all at once)

6. **output-assessment-partial.md (stages 2-4)** — the highest-value item on
   the list; it's the actual fix for the motivating problem (#193's
   undetected 33ms drift). But it's a multi-stage engine feature (boundary
   persistence, 5 new probe tasks, a rules table, new routes, a new skill).
   `output-assessment-complete.md` already stages it into 4
   independently-landable pieces — treat each as its own decision rather
   than one big yes/no.
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

## Backlog ideas with no doc on file

- **A larger Qwen-Image catalog entry** — the 20B, Apache-2.0 Qwen-Image
  checkpoint (distinct from the smaller Qwen-Image-2.1 onboarded
  2026-09-20) would be a real catalog expansion. No design doc exists for
  it yet; write one before starting.
