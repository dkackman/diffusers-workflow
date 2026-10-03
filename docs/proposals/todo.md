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
- **`attribute_voices`, which reference singer sings each line** (#485,
  stages #494-#495), shipped 2026-09-27. Record, including what was
  deferred (template wiring, demucs weights in `downloads_required`):
  `complete/attribute-voices-complete.md`.
- **H3 Ref2VA VRAM ceiling: references, `for_each` members, hand-built
  workflows** (#479, stages #501-#502), shipped 2026-09-27. Record, including
  what was deferred (the host-memory half, measured calibration):
  `complete/h3-vram-ceiling-references-complete.md`.
- **A true-peak limit mode for `normalize_audio`** (#474, stages
  #496-#497), shipped 2026-09-27. Record, including what was deferred
  (holding the ceiling on the encoded film, true-peak in the post-encode
  check, `compress_audio`'s limit mode):
  `complete/normalize-audio-limiter-complete.md`.
- **`join_into_song`, a spoken scene breaking into a song** (#486, stages
  #513-#514), shipped 2026-09-27. Record, including what was deferred (a
  template, register-external-output, the `concat_videos` silent-input
  desync filed as #553): `complete/join-into-song-complete.md`.
- **`find_loop_bed`, ranked room-tone loop windows** (#218, stages
  #544-#545), shipped 2026-09-27. Record, including what was deferred (a
  sync route, in-workflow wiring into `slice_audio`, thresholds for louder
  rooms): `complete/find-loop-bed-complete.md`.
- **`ltx2/refine-clip`, LTX-2.5's two-stage refine on an existing mp4**
  (#543, stage #549), shipped 2026-09-28 as one template with no engine
  task. Record, including what was deferred (a 1x refine and the
  VAE-encode task it needs, encoding the source soundtrack into audio
  latents): `complete/ltx2-refine-clip-complete.md`.
- **`ltx2/upscale-clip`, LTX-2.5's generative 2x upscale of an existing
  mp4** (#542, stage #548), shipped 2026-09-28 as one template that keeps
  the source's soundtrack. Record, including what was deferred (probing
  the source at validate, source audio for the restore templates, the
  vendor's Refine-Details IC-LoRA): `complete/ltx2-upscale-clip-complete.md`.

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
