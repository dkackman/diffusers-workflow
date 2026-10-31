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
`feature` (#374, #380, and #244 for `resume.md`; #375 and #379 were declined, #376, #377 and #378 shipped), parked with Don. Its
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
9. **maintenance-screen.md — the full UI page** — Phase 0 (WAL mode) shipped
   2026-09-20; the maintenance/observability page itself (orphan listing,
   disk usage, job pruning) is the remaining, much bigger ask.

## Shipped since the ranking

- **H3 quality ramp: draft -> final tiers** (#862, stage #882), shipped
  2026-10-15. A `## Ramp` section in the `minimax-h3` skill's
  `references/quality-and-speed.md`, not a new skill: TV L1 -> L7, cinema
  B3 -> L5/P1, ship a close-up draft, with the arguments per rung, what
  carries (with #861's seed counts), what to approve per tier, how to judge
  and how to quote cost. Record, including what was deferred (MagCache M2
  as the cinema-final cache, a top-level skill once a second family has a
  ladder): `complete/h3-quality-ramp-complete.md`.
- **Dispatch holds a queued job for host RAM** (#869, stages #909-#910),
  shipped 2026-10-15. A job waits `queued`, with a `queue_wait` event and a
  live `waiting_for`, while its projected peak (the max observed for its
  memory key, else the list fit) won't fit beside the running jobs. It is
  never refused, and never held alone. Record, including what was deferred
  (a validate "would wait" prediction, frame count in the key):
  `complete/host-ram-dispatch-gate-complete.md`.
- **`plan.estimate` scales the curated figure** (#866, stages #905-#908),
  shipped 2026-10-15. A move of `num_inference_steps`, `width`/`height` or
  `num_frames` off the defaults now scales the catalog or inherited figure
  in proportion (`basis: "derived"`, `scaled` names the axes) instead of
  quoting `unknown` or the flat figure. Any other declared driver (H3
  weights, LoRA, attention, cache) keeps the figure with `low_confidence`
  and `differs`. The fingerprint and step cache are keyed only by what
  changes the output. Record, including what was cut (compile as a driver,
  the learned fixed + per-pass fit) and follow-ups #911 and #912:
  `complete/estimate-scaling-complete.md`.

- **H3 adapter check reads the LoRA catalog first** (#870, stage #888),
  shipped 2026-10-14. On the server, validate and admission look up the
  LoRA in the shared catalog: a `proven` entry whose `workflow` puts it on
  the other partition refuses the step (Realism People on `ref2va`), a
  `trial`/`rejected` one only warns, naming the entry; uncatalogued files
  keep the filename rule. Record, including what was cut (reading the
  safetensors header, any Hub request from validation):
  `complete/catalog-adapter-check-complete.md`.

- **Running a sweep: a written method, not a server feature** (#872, stage
  #890), shipped 2026-10-14. `## Running a sweep` in `docs/WORKSPACES.md`
  (`get_guide("workspaces", section="Running a sweep")`) writes down the
  loop the H3 quality matrix ran with local scripts: one workspace per
  experiment, factors as variables, a pinned seed, one noted `run_workflow`
  per variant with `wait_for_job` between, `set_note` assessments in the
  `score N/5 · judged_by <who>` form. It supersedes
  `sweeps-and-comparison.md` (#379), now deleted. Record, including what was
  cut (`sweep=` on `run_workflow`, structured notes, an experiment object,
  `arguments` on `rerun_job`): `complete/sweeps-complete.md`.

- **`compare_outputs`: aligned grids and diff metrics across runs** (#871,
  stage #889), shipped 2026-10-14. A read-only MCP tool and `POST
  /api/gallery/compare` return one grid of 2–8 outputs × 1–8 moments, plus
  PSNR, SSIM and mean abs diff against a baseline. One shared crop is the
  region shown and measured; per-output crops are display only. This is the
  read-only comparison half of the old `sweeps-and-comparison.md`, without
  batches or a UI. Record,
  including what was cut (`crop="faces"` as a deferred stage B,
  `same_staging`, saving the grid): `complete/compare-outputs-complete.md`.

- **Per-step timing and step-cache skip counts** (#867, stages #885-#886),
  shipped 2026-10-14. Each pipeline step's manifest entry carries `timing`
  (load, encode, denoise, passes, first and median pass, decode, save), and
  a cached step adds `timing.cache` (forwards, computed, skipped,
  `hit_rate`) plus one `cache_stats` event. Record, including what was cut
  (`compile_s`, counts for `taylorseer`/`faster`/`text_kv`, with their
  triggers): `complete/step-timing-complete.md`.

- **H3 templates expose `attention_backend` and `cache`** (#865, stages
  #883-#884), shipped 2026-10-14. Every MiniMax-H3 template takes both as
  variables (default `null`, wired to the H3 step only). `cache` is a
  whole-object variable: the schema accepts `variable:` there, and
  validation checks the substituted object (`cache_errors`). The tier
  presets were cut; the tiers live in the `minimax-h3` skill as argument
  sets (#862). Record: `complete/h3-template-acceleration-complete.md`.

- **Validate warnings for acceleration settings that can't pay off** (#868,
  stage #887), shipped 2026-10-14. `validate_workflow` and `save_workflow`
  warn (never refuse) on `cache_short_schedule` (a `first_block`/`mag` cache
  under 20 steps), `taylorseer_state_unestimated`,
  `compile_fullgraph_with_cache` and `attention_backend_unavailable` (a
  CUDA-only backend on a server that isn't CUDA). Compile under offload is
  docs only. Record, including what was deferred (TaylorSeer state sizing,
  a `mag` step-mismatch check): `complete/acceleration-warnings-complete.md`.

- **Copy runs and assets between workspaces** (#853, stages #859-#860),
  shipped 2026-10-13. `copy_to_workspace` and `POST
  /api/workspaces/{destination}/copy` copy whole runs (with manifests) and
  asset-library entries (with their notes and kept shots) between two
  workspaces under one root. The copy is all or nothing, never overwrites
  and is idempotent. Media is hard-linked, sidecars are copied, and
  colliding versions are renumbered. A move is a copy followed by the
  delete tools. Record, including what was deferred (a UI, job rows for
  copied runs, workflows and prompts): `complete/workspace-copy-complete.md`.
- **List and clear dw's own disk caches** (#828, stages #832-#834), shipped
  2026-10-11. A registry (`dw/disk_caches.py`) of `inductor`, `triton` (Triton's
  own `~/.triton/cache`, left in place) and the `quantized` cache, with
  `GET /api/caches`, `DELETE /api/caches/{name}` and a per-entry delete for the
  quantized cache, a Caches panel on the server page, and sizes in
  `get_server_info`'s `caches` (no MCP clear tool). Clearing works on an idle
  warm worker without a restart, and only for a root that carries dw's
  `.dw-cache` marker. Eviction stays manual. Record:
  `complete/disk-caches-complete.md`.
- **On-disk cache of SDNQ on-the-fly-quantized components** (#815, stages
  #825-#827), shipped 2026-10-11. Opt-in with `DW_QUANTIZED_CACHE=1`, rooted at
  `DW_QUANTIZED_CACHE_DIR` (default `<HF_HOME>/dw-quantized`), on both the H3
  modular path and the LTX-2.5 per-component path. Measured on lem: an H3 cold
  load saves 63.5 s (page cache warm) to 93.9 s (cold). Record, including what
  was cut or deferred (other backends, the LTX measurement rows; listing and
  clearing shipped in #828): `complete/quantized-cache-complete.md`.
- **`dw/server/jobs.py` split into `pool` / `job_results` /
  `worker_memory`; device ordinal and card as stored fields** (#693,
  stages #776-#777), shipped 2026-10-08. The `slots[0]` shims are gone
  (`running_job_id()` serves `/api/health`). `Job` and the history DB carry
  `device_ordinal`/`device_card` (additive columns, SQL backfill), `device`
  is derived from them, and `card_of`/`ordinal_of` are deleted. REST and
  MCP are unchanged. Record, including what was deferred (history readers,
  API fields, multi-card checks on lem): `complete/jobs-split-device-fields-complete.md`.
- **H3 guides in the VRAM estimate** (#694, stages #778-#779), shipped
  2026-10-08. An optional `vram_estimate.bytes_per_guide_voxel` (28.71 on the
  H3 t2va templates and `chained-segments`, which also gained a `cost` and an
  estimate) charges each guide's snapped frames on the canvas, plus a
  `continuity: "guide"` chain's guide, probed alike at validate, admission and
  run. Measured on lem: `h3-guide-vram.md`. Record, including what was cut or
  deferred (`hold_audio` and refine terms, calibration margins, with their
  triggers): `complete/h3-guide-vram-complete.md`.
- **Task layer consolidation: one registration pattern, one coercion,
  shared image ops** (#692, stages #773-#775, fix-forward #785), shipped 2026-10-08. A task's
  domains, choices, static check and media arguments are declared on
  `register_command`, and the old tables are derived views. Task numbers
  are coerced by `whole_number`/`real_number`, and validate and run agree.
  `crop_face_track` takes its grid from the workflow. Record:
  `complete/task-layer-consolidation-complete.md`. The follow-up is #784.
- **`h3_blocks.py` split into `h3_rules` / `h3_hold` / `h3_guides`, and the
  H3 copies of diffusers pinned** (#691, stages #770-#771), shipped
  2026-10-08. Drift tests pin the refine sigma grid to stock
  `set_timesteps` and the guide chunk sizes to the H3 video VAE; the rules
  module is torch-free. Record, including the cut catalog move of
  `guide_frames`: `complete/h3-blocks-split-complete.md`.

- **Enhancer presets ruled UI wiring, not model knowledge** (#696, stage
  #781), shipped 2026-10-08. `PRESETS` stays in `dw/server/enhancers.py` as
  the Enhance panel's menu; a new *Prompt enhancement* map row states the
  rule and its trigger (a third preset moves the menu into builtin
  metadata). Record, including the declined migration:
  `complete/enhancer-presets-ruling-complete.md`.

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
- **A locked H3 frame that lip-syncs to a song** (#852, stages
  #877-#879), shipped 2026-10-14. Four renders on lem found Ref2VA
  `[keyframe completion + audio reuse]` with a first frame holds a wide
  two-shot and lip-syncs the right singer; a last frame doesn't help. One
  `minimax-h3` skill line; no template, since `music-video` takes the
  recipe with arguments. Record: `complete/h3-keyframe-audio-complete.md`.
- **A `stacked_vocals` warning from `separate_stems`** (#947, stage #955),
  shipped 2026-10-19. A stereo check on the vocals stem names the spans
  that hold layered voices (a Music 3 chorus). Record, including what was
  deferred (the voice/pitch conversion task, structured spans, the check in
  `attribute_voices`): `complete/stacked-vocals-warning-complete.md`.
- **`remove_object`, prompt-free LaMa object removal** (#956, stages
  #973-#974), shipped 2026-10-22. A task that fills a white-means-fill mask
  from the surrounding image, with no prompt and fp32 on any device, using
  pinned `big-lama.safetensors` loaded through spandrel. It comes with the
  `remove-object` and `segment-and-remove` templates and an
  erase-versus-repaint routing rule. `segment` now keeps SAM2's best-rated
  mask. Record, including what was deferred (temporal video removal,
  per-frame masks, a larger model for big regions, #975's download
  warning): `complete/remove-object-complete.md`.
- **Task weights in `plan.downloads_required`** (#975, stages #977-#978),
  shipped 2026-10-23. Each model-backed task declares
  `required_weights(arguments)`. Its loader reads the same resolution, and
  the plan lists those repos, plus single files at their pinned revision
  and facexlib's URLs, so a bound acknowledgement covers them. Record,
  including what was deferred (fetchers outside the HF cache, pinned-file
  `download_model`, `gated` on file entries):
  `complete/task-weights-downloads-complete.md`.
- **H3 rules from WanGP's field notes** (#985, stages #988-#989), shipped
  2026-10-25. Probed on lem at int8 + turbo: FL2VA with `hold_audio` keeps
  the first frame's pixels and lip-syncs, so it is now the skill's
  locked-frame route (Ref2VA keyframe completion the fallback); leaving
  words out of `<d>` under a held track was refuted. The `h3` enhancer
  writes `<d>` blocks again. Record, including what was dropped (tail-frame
  discard: dw chains have no per-window end image) and deferred (keyframe
  strength, not reachable from a workflow):
  `complete/h3-wangp-field-notes-complete.md`.
- **Look measures in `analyze_seams`** (#1031, stage #1042), shipped
  2026-10-26. Every seam record carries signed `saturation_step`,
  `lightness_step` and `contrast_step`, a local-rate `look_ratio` with its
  `look_ratio_measure`, and a `detail_ratio`, taken from a 128x72 colour
  copy during the one decode. Measures only, no rules. Record, including
  what was deferred (rules until #1033 measures real chains, take drift,
  apparent age): `complete/seam-look-measures-complete.md`.
- **H3 AISkills claims: style-skills pointer and three probes** (#1059,
  stages #1070-#1072), shipped 2026-10-30. `minimax-h3/references/style.md`
  points a style-heavy request at MiniMax's eight style packs, by name. On
  lem, P2 (timbre reference replays source words), P3 (style image pulls in
  its person) and P5 (screen side holds handedness) were all refuted
  (`audits/2026-10-h3-aiskills-probes.md`) and listed as probed-and-not-a-rule.
  Record, including what was deferred (P1 anime cadence, P4 POV) and dropped
  (P6, P7): `complete/h3-aiskills-probes-complete.md`.
- **Chain following consecutive windows of a long source video** (#1061,
  stages #1074-#1076), closed 2026-10-31 after its probe only. On lem, H3
  ref2va followed a driving window with no cut in it, but a window holding
  a source cut dropped the first shot and the carry broke the seam, so the
  gate failed and the `match_video` chain key and `chain-follow-video`
  template were not built. Record, with the design and what would reopen it
  (a lighter H3 build, or #1045's guide path):
  `complete/chain-follow-video-complete.md`.

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
