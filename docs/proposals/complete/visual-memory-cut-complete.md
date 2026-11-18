# Visual memory at a cut: an earlier shot as an H3 video reference (#1398)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-18). Don filed the issue on 2026-11-16 from KadoBOT/ComfyUI-H3-Continuity-Studio
(MIT, read at a2ddb97): after a hard cut H3 sees only the new prompt, so the room, props
and wardrobe can change. The source hands H3 earlier moments of the film as native video
references, and reports that a whole earlier clip copies its lip movement onto a new
speaking shot. The plan comment on #1398 (v2) holds the full sweep. This file records
what was measured and shipped.

## Verdict: guidance only. A short window of the establishing shot holds the place

Measured on one scene, one seed, ref2va turbo on lem (`bench/records/visual-memory-cut.md`),
and Don viewed the arms on 2026-11-17:

- **No memory (B0, B0s):** the reverse angle loses three of four markers: the lamp colour,
  the window sky and the green coat.
- **A 22-frame window from the middle of the establishing wide (B2):** all four hold, the
  shot stays a true reverse, and it costs **1.35x** a control pass.
- **The previous clip whole (B1s, S1s, 960x544):** the lamp, shelf and window hold, but the
  coat becomes a **purple dress** in both arms. The shot is pulled back toward A's angle,
  and it costs **4.4x** a control pass, above the skill's 3.4x. At 1344x768 it does not fit
  a 24 GB card (B1, job `e6ecb7df55ec`, out of memory).
- **The last frames of the speaking shot (S2):** H3 replays them for about 1.4 s, then cuts
  inside the clip to a reverse that holds all four markers. A cut built this way has to
  trim that opening.
- **Lip copy:** not reproduced. S1s shows no mouth copy by frames or in Don's viewing, so
  the rule is shipped as the source's claim, not as measured.

## What the design sweep settled

- **The pixel route already existed for both shapes.** The whole clip is
  `MiniMaxH3VideoReference` `from_file: output:…`. A window is `trim_video` →
  `from_previous_result`. So "latent-direct vs pixel" was "new engine surface vs nothing",
  and the latent path was cut: finished clips don't persist latents (`dw/result.py`), and no
  ref2va input takes an encoded latent.
- **The fl2va-with-guides narrated-take shape can't take a reference at all.** Any reference
  sends a shot to ref2va, which refuses guides, anchors and carry (`dw/guides.py`). The
  feature reaches the ref2va cut shapes only (`dialogue-short`, and script-to-video and
  series-episodes shots rendered through it).
- **The plugin already offered A's tail as a video reference** (`joins.md`) with no
  lip-copy warning. Recording that hazard was a reason to probe rather than defer.
- **Cut:** the latent-direct path, a template option, and the source's modes ("same scene",
  "selected moments", "cast").

## Plan history

- **v1 (design, 2026-11-16).** Verdict "build smaller": one lem probe (E, A, B0, B1, B2, A′,
  S1, S2 at 1344x768, 5-frame windows), then guidance. Approved at defaults: Q1 [scope]
  build smaller, Q2 [spend] ~70 min lem, Q3 [taste] Don views the arms, Q4 [scope] no
  #1323 sheet arm, Q5 [none] guidance in `cuts.md` with the rule echoed in `joins.md`.
- **v2 (stage A build, 2026-11-17).** B1 ran out of memory at 1344x768, and 5 frames was
  refused (dw needs at least 13 frames), so the plan's 22-frame fallback was used. The
  whole-clip arms moved to 960x544 with their own control (B0s, B1s, S1s), and a replay
  check was added. Q6 [loosen, spend ~16 min over] approved at the default on 2026-11-17.
  No scope change.

## What was built

### Stage A (#1401): the probe and its record (bench only)

Commits `e94da31d` and `bf74f93d`, merged into `develop` as `96a6afb5`, and the record
fix `0e9832f3`. No deploy.

- `bench/workflows/visual-memory-shot.json` (portrait plus an optional whole clip) and
  `bench/workflows/visual-memory-window.json` (a `trim_video` window as the reference).
- `bench/records/visual-memory-cut.md`: job ids, wall and per-pass times, the refusal and
  out-of-memory errors quoted verbatim, the per-arm marker and framing read, the replay
  and mouth reads, and `bench/cut_windows.py pair` numbers rerun on lem.
- Lem time: about 93 min against the ~86 approved. A whole-clip pass cost 4.4x, not the
  3.4x the estimate assumed.

### Stage B (#1402): guidance and attribution (plugin only)

Commits `8fe04bd8` and `3cebf05a`, merged as `afcd99dd`, and the fix `1fa5ffc0`, merged
as `9d6ebc5f`. No deploy.

- `plugins/dw/skills/minimax-h3/references/cuts.md`, *Visual memory at a cut*: the marker
  table as Don viewed it, the costs, the out-of-memory limit, S2's replay, and the recipe
  (a 22-frame window from the middle of the establishing shot, `start_frame` on the 17
  grid, `audio: null`, within 3 videos and 12 files). It also has the lip-copy rule, as the
  source's claim.
- `plugins/dw/skills/script-to-video/references/joins.md`: the "A's tail as a video
  reference" option carries the speech rule and points at `cuts.md`.
- `docs/ATTRIBUTIONS.md`: a KadoBOT/ComfyUI-H3-Continuity-Studio entry (MIT, concept only,
  a2ddb97).

Deviations, accepted at review:
1. **No manual plugin version bump.** `plugins/dw/README.md` ties the plugin version to the
   engine's, and `scripts/release.sh` bumps both.
2. **The record kept its PRELIMINARY marking through stage B.** This close-out points the
   record at Don's verdict.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1401 | 1 re-plan (B1 out of memory, v2) and 1 docs-review bounce (a "four of six seconds" miscount and no rerun evidence, fixed in `0e9832f3`) | Session cost not recorded on the stage. GPU: about 93 min of lem, one H3 job at a time |
| #1402 | 1 tester bounce (a `cuts.md` line saying `validate_workflow` missed a reference under 13 frames, stale after #1422; fixed in `1fa5ffc0`) | Session cost not recorded on the stage. No GPU |

The stage B verify also turned up #1428, an engine-side issue (unknown keys on an H3
reference were not refused), filed and closed on its own.

## Deferred and open

- **The latent-direct path:** not built. *Trigger:* a window reference that holds the place
  but reads visibly softer or shifted against its source frames, as a VAE re-encode would
  explain. The probe showed no such sign.
- **A template option for the window:** not built. *Trigger:* a field run showing agents
  fail to wire `trim_video` → `from_previous_result` by hand.
- **The lip copy:** not reproduced on one sample. The rule stays as the source's claim.
- **#1323 (the storyboard sheet across cuts):** still parked on its own trigger. This probe
  ran no sheet arm (Q4).
- **Known limits:** one scene and one seed, and the whole-clip arms ran at a smaller size
  than the window arms. Face sharpness is not comparable across sizes.
