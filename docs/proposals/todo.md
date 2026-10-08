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
`feature` (#374, #379, #380, and #244 for `resume.md`; #375 was declined, #376, #377 and #378 shipped), parked with Don. Its
`priority:N` label mirrors the tier here. Work starts from the issue.

## Tier 1 — do these first (small, scoped, clear payoff)

None open. The last item, the H3 video mux headroom warning, shipped. Its
remaining deferred fix is recorded in
`complete/h3-video-mux-headroom-warning-complete.md`.

## Tier 2 — solid ROI, moderate scope

2. **orphaned-run-directories.md** — real, recurring disk-usage annoyance
   (leftover manifests invisible to gallery/asset listings); the proposal
   already recommends the simple option (A). Moderate but bounded work.

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

- **`window_video` and `join_windows` host memory: range reads and a uint8
  join** (#695, stage #780), shipped 2026-10-08. A window reads only its
  frame range and the soundtrack; the join no longer decodes the source's
  picture or holds float32 copies. Record, including what was cut (the
  file-backed `safe_get`, with its trigger):
  `complete/window-memory-complete.md`.

- **Finishing tasks: tonal controls in `grade`, `sharpen`, `film_grain`,
  `apply_lut`** (#603, stages #633-#637), shipped 2026-10-07. Seven tonal
  `grade` parameters, a seeded per-frame grain, a strict `.cube` parser
  (and `.cube` uploads), a `palette` look built in memory, and an optional
  look step in `series-episodes`. Record, including what was not built
  (`make_lut`, a `.cube` output, the plugin version bump):
  `complete/finishing-tasks-complete.md`.

- **Script-adherence check: `check_script` and `templates/check-script`**
  (#609, stages #643-#645), shipped 2026-10-07. A take's transcript aligned
  to its expected lines, with shot-aware findings; the by-eye transcript
  check is retired from the loop and the skills. Record, including what was
  not built (an `assess_output` input, a wider per-shot clip rule):
  `complete/check-script-complete.md`.

- **H3 dialogue rules: probes and `minimax-h3/references/dialogue.md`**
  (#608, stages #640-#642), shipped 2026-10-07. Fifteen probe rows on lem
  (`audits/2026-10-h3-dialogue-probes.md`); one H3 rule confirmed, plus a
  Music3 vocal-stem check. Record, including what was not built (the
  `speech` validate warning, since H3 didn't clip; the prompt-length budget
  and the negation lint, with their triggers):
  `complete/h3-dialogue-rules-complete.md`.

- **Overlap windowing for long video sources: `window_video`,
  `join_windows` and `templates/ltx2/restore-long`** (#601, stages #628-#630
  and #658), shipped 2026-10-06. Record, including what was deferred (the H3
  grid, per-window audio, long refine/upscale templates, with their
  triggers): `complete/overlap-windowing-complete.md`.

- **Music-video timing stack: `analyze_beats`, `plan_cuts` and
  pad-then-trim shots** (#600, stages #625-#627), shipped 2026-10-06. Record,
  including what was deferred (per-shot `kind`/`singer`, stage D, and Q4's
  optional `for_each` entry fields):
  `complete/music-video-timing-complete.md`.

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
- **`fit_to_model` / `restore_to_source`, an exact size and frame-count
  round trip for v2v** (#602, stages #631-#632), shipped 2026-10-06, with
  `upscale-clip` and `refine-clip` rewired to letterbox and restore. Record,
  including what was deferred (the pair in the restore templates, the blend,
  the anchors in #613): `complete/fit-to-model-restore-complete.md`.
- **One `wait_for_job` call that covers a long render** (#377, stage
  #546), shipped 2026-09-28: `DW_MCP_MAX_WAIT_SECONDS=1800` on lem's unit
  and one wait rule in the guide and skills, with no code change. Record,
  including what was deferred (the progress heartbeat, stage #547, not
  needed since no long wait was cut): `complete/mcp-long-wait-complete.md`.
- **Promoting an H3 take to 1344x768 in latent space** (#471, stage #499;
  #500 not built), shipped 2026-10-03 as two tasks, `upscale_h3_latents` and
  `decode_h3_latents`, with an inline workflow in the guide and no template.
  Don's gate found the preview ~39% faster than a native 768p render but
  soft in the faces. Record, including what was deferred (the refine pass,
  persisted latents, task weights in `downloads_required`):
  `complete/h3-latent-upscale-complete.md`.
- **Bulk download of a job's outputs** (#592, stage #595), shipped
  2026-10-05 with no new tool: `export_job` now reports the ungated
  `/exports` zip as `auth_required: false` on a token server and tells the
  agent to fetch it, and the multi-job skills name it as how a project goes
  home. Record, including what was deferred (a multi-job bundle, an
  outputs-only zip, removing an export over MCP):
  `complete/job-export-bulk-download-complete.md`.
- **Checking the lip-sync target** (#488, stage #617), shipped 2026-10-05
  from shipped parts: `transcribe_audio` never emits a null `end`, an
  `attribute-lines` template, a mixed-line `uncertain` rule in
  `attribute_voices`, and the loop in `docs/TASKS.md`. Record, including
  what was deferred (the VLM probe, anatomy and travel-direction checks, a
  real two-singer round): `complete/lip-sync-target-check-complete.md`.
- **H3 audio-hold and a refine pass after latent upscale** (#598, stages
  #618-#621), shipped 2026-10-06. It adds `hold_audio` (opt-in: the
  templates keep the audio reference after an A/B went against hold) and
  `refine_strength` (opt-in engine surface). `templates/minimax/upscale-refine`
  failed Don's gate: 11.53 min against 9.59 for a native 768p render, which
  is sharper, so it was reverted (#664, 2026-10-07). Record, including what was deferred (a re-measure of hold, the
  3-pass variant, #612's arm of the A/B):
  `complete/h3-audio-hold-refine-complete.md`.
- **`templates/ltx2/refine-in-place`, a same-size LTX refine with a
  strength knob** (#606, stages #638-#639), shipped 2026-10-06. It adds the
  `LTX2RefinePipeline` community pipeline and five lem-tuned sigma ladders
  selected by `strength` 0-4. Record, including what was deferred
  (per-segment strength decay in `chained-segments`, with its triggers):
  `complete/ltx2-refine-in-place-complete.md`.
- **Temporal face repair, `templates/ltx2/face-repair`** (#599, stages
  #622-#624), shipped 2026-10-06. It adds the `crop_face_track` and
  `paste_face_track` tasks (YuNet over tiles, one tracked face, a
  distance gate, 8n+1 crops, feathered strength-scaled paste-back) and a
  same-size LTX refine of the crop with five lem-tuned ladders. Record,
  including what was deferred (rotation passes, landmark-affine paste,
  multi-face tracking, with their triggers):
  `complete/face-repair-complete.md`.
- **H3 multi-frame guides, `guides` on `t2va`/`fl2va`** (#611, stages
  #648-#650), shipped 2026-10-07. It adds the `guides` argument (clips of
  1, 5 or 17m+5 frames at any frame 17j, up to 4), `"audio": true` on a
  guide, and `continuity: "guide"` on `templates/minimax/chained-segments`
  (opt-in; its A/B held the seams better than `last_frame` at ~40% more
  wall time, and the default is Don's call). Record, including what was
  deferred (persisted latents, stale-marking, guides on `ref2va`, a total
  guide-frames cap, with their triggers): `complete/h3-guides-complete.md`.
- **One job per GPU** (#462, stages #675-#678), shipped 2026-10-07.
  `dw.serve --devices cuda:0,cuda:1` runs one pinned worker per card behind
  one queue, with a VRAM-fit dispatcher, backfill, rerun and identity
  affinity, per-card memory, and `device` on jobs. lem runs both 3090s.
  Record, including what was deferred (a host-RAM gate, an overlap flag on
  observed cost, per-card catalog cost, with their triggers):
  `complete/multi-gpu-workers-complete.md`.

## Declined

Kept in `declined/` with the reason at the top, so a revival starts from
the analysis rather than repeating it.

- **declined/workspace-folders.md** — grouped workspace names (`QA/EP1`).
  Declined 2026-09-23 on #375: thin demonstrated value against a loosened
  security boundary and a change that can't be taken back.
- **declined/mcp-job-notifications.md** — a gapless cursor and a
  `failure_kind` on `wait_for_job`. Superseded 2026-10-03 by #377's smaller
  plan: terminal status is sticky, so the cursor closes no gap, and the
  proposed OOM label would have missed CUDA OOMs. Reopen triggers are at the
  top of the doc.

## Backlog ideas with no doc on file

- **A larger Qwen-Image catalog entry** — the 20B, Apache-2.0 Qwen-Image
  checkpoint (distinct from the smaller Qwen-Image-2.1 onboarded
  2026-09-20) would be a real catalog expansion. No design doc exists for
  it yet; write one before starting.
