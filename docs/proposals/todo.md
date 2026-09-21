# Proposal backlog: benefit vs. complexity ranking

Written 2026-09-20, after auditing every file in `docs/proposals/` against the
current codebase (fully-implemented proposals were moved to `complete/` by
Don; the four found partially implemented were split into `*-complete.md` +
`*-partial.md` pairs alongside this file). Ranking is by benefit vs. added
complexity/risk, highest ROI first.

## Tier 1 — do these first (small, scoped, clear payoff)

1. **h3-video-mux-headroom-warning-partial.md** — an active bug: a warning
   fires wrongly on stock template defaults and breaks a regression assertion
   (M-F008). Fix is scoped to holding one warning until a probe that already
   runs reports back. Small, contained, fixes something broken today.
2. **maintenance-screen.md — Phase 0 only (WAL mode)** — literally a one-line
   `PRAGMA journal_mode=WAL` on the jobs DB connection. Real concurrency
   benefit, near-zero risk. Split this from the rest of the proposal — the
   full UI page is a separate, much bigger ask (see Tier 3).
3. **score-and-select-partial.md** — all the engine work (the hard part) is
   already shipped. What's left is authoring one catalog template and
   pinning it in tests. Low effort, unlocks the "generate 4, keep the
   sharpest" pattern end-to-end.
4. **script-to-video-agent-skill.md** — pure composition of mechanisms that
   already exist (family templates, prompt-writing skills, cost gate,
   cast-consistency). Zero engine changes, just a skill file. Potentially
   high leverage (prose script → finished video) for the lowest cost on this
   list.

(**step-callback-lead-in-instrumentation.md**, originally item 2 here, turned
out to already be implemented — `dw/events.py`'s generic phase-stall watchdog,
shipped for #176 after this proposal was written, is exactly its recommended
Option A. Renamed to `step-callback-lead-in-instrumentation-complete.md`
2026-09-20.)

## Tier 2 — solid ROI, moderate scope

6. **orphaned-run-directories.md** — real, recurring disk-usage annoyance
   (leftover manifests invisible to gallery/asset listings); the proposal
   already recommends the simple option (A). Moderate but bounded work.
7. **workspace-folders.md** — one regex relax + a depth-2 listing walk + UI
   grouping. Low complexity, meaningful convenience for the growing
   series-episodes workflow.
8. **mcp-context-cost-partial.md** — cheap and safe, but the remaining
   payoff is small (~1.1k tokens of connect cost), and recommendation 6 in
   that same doc suggests the harness-level fix (deferred MCP schemas) may
   already make this moot — worth confirming that before spending effort
   here.
9. **mcp-job-notifications.md** — improves reliability of the wait/poll loop
   (cursor-based, `failure_kind`), but there's no reported live pain forcing
   this yet; medium complexity touching the event/job-record schema.

## Tier 3 — high benefit, but big lifts (stage carefully, don't take all at once)

10. **output-assessment-partial.md (stages 2-4)** — the highest-value item on
    the list; it's the actual fix for the motivating problem (#193's
    undetected 33ms drift). But it's a multi-stage engine feature (boundary
    persistence, 5 new probe tasks, a rules table, new routes, a new skill).
    The doc already stages it into 4 independently-landable pieces — treat
    each as its own decision rather than one big yes/no.
11. **resume.md** — meaningful for expensive multi-step runs that crash, but
    rehydrating the step cache from disk manifests is a correctness-sensitive
    engine change (step identity matching, partial-state edge cases). High
    complexity.
12. **sweeps-and-comparison.md** — valuable once doing real side-by-side
    comparisons, but it's a new job-schema field, batch semantics, and a new
    UI page. Its own doc says "nothing here should be implemented without a
    fresh look."
13. **qwen-image-wan-onboarding.md** — real catalog expansion (20B
