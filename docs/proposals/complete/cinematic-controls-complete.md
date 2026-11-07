# Cinematic controls: camera/look vocabulary, shot document, storyboard/animatic, face-drift check (#1141)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-02). Plan v1 was approved by Don on 2026-11-01 with every default
(no later version). There were seven stages, #1142–#1148, and all were
verified on 2026-11-02. Stages A–F were built in Don's interactive session.
The loop built stage G and the two bounce fixes.

Sources:
- The audit: `docs/proposals/audits/2026-11-higgsfield/README.md`. It covers
  Higgsfield's catalog, the #OpenSourceHiggsfield clones and their licences,
  and dw's gaps.
- The full plan with implementation tasks:
  `docs/superpowers/plans/2026-11-01-cinematic-controls.md`. It stays in the
  tree as the task-level record.
- Tier 2 (re-shoot, 9 angles, speed LoRAs, relight, dubbing, swap) is a
  separate feature, `docs/proposals/cinematic-addons.md`. It is still open.

Concepts only, no code: Popcorn (Higgsfield) and the coverage set from
Shot Board (MIT). The face-drift probe takes its approach from CreatorOS
(YuNet + SFace) and was re-implemented.

## The idea

Higgsfield is a preset layer over other people's models. The audit
concluded that **dw's gaps are mostly vocabulary and presentation, not
capability**. So tier 1 adopts Higgsfield's director vocabulary and its
plan-board-shoot loop on the families dw already runs: H3, LTX-2.5,
Kandinsky 6 and FLUX 3 Klein. It adds no new generator.

## Decisions (Don, 2026-11-01)

- **Named moves are hints, not promises.** The skill tells the agent to
  describe the move and check it with `get_output_frames`. Stage G measures
  how often each family honours it.
- **Q1 [scope], yes:** two task commands (`stills_to_video`,
  `analyze_face_drift`) and two skills (`cinematography`, `storyboard`).
  No MCP, REST or workflow-syntax surface.
- **Q2 [none], yes:** the module ratchet goes 236 → 237 for
  `dw/tasks/face_drift.py`, rather than growing `face_track.py` past its
  ceiling.
- **Q3 [trust], yes:** SFace weights (Apache-2.0, ~37 MB) at a pinned
  revision.
- **Q4 [spend], yes, after A–F verify:** about 1 h of lem GPU for stage G.

## What was built

### A: `cinematography` skill (#1142)
- `plugins/dw/skills/cinematography/` holds a closed vocabulary of six
  axes (`references/vocabulary.md`): shot size, move, lens, light,
  era/stock and tempo.
- It has one compile table per family: `compile-h3.md`,
  `compile-ltx-2.5.md`, `compile-kandinsky-6.md` and
  `compile-flux3-klein.md`. On Klein, moves compile to framing.
- `references/grades.md` holds five named grades: noir, teal-orange,
  70s stock, VHS and bleach-bypass. Each is an `apply_lut` palette plus a
  `film_grain` setting.
- *A hint, not a promise* gives the `get_output_frames(count=12)` check.
  *Not in scope* covers Music 3 (no camera) and any other family (stop).
- Each of the four family skills has a pointer line.
- The H3 sentences come from MiniMax's prompt guide §4.3 and the stored
  `minimax/fox_dawn_context_ir` prompt, because the H3 references had no
  camera wording.

### B: shot document, fix ladder, H3 lint (#1143)
- `script-to-video/references/shot-document.md` defines:
  - the shot list, saved as `prompt:<project>/shots` (a JSON array);
  - each caption, saved as `prompt:<project>/shot-NN` and referenced by name;
  - a revision, which re-saves and re-runs only the changed shots.
- Every shot carries `shot_size`, `move`, `lens`, `light` (ids from
  `cinematography`) and `duration`. A `null` means the model chooses.
- The fix ladder: regional edit or face repair, then re-prompt, then
  reseed, then ask the user. It allows 3 tries per shot and a per-project
  GPU-minute cap quoted from `plan.estimate`.
- The H3 prompt lint flags an overfull beat, a cut with no new information
  and a missing `overall_soundscape`.

### C: `stills_to_video` task (#1144)
- `stills_to_video(images, durations, fps=24)` holds each still for its
  duration. It returns a clip with one shot record per still, documented in
  `docs/TASKS.md`.
- Its caps:
  - fps in (0, 120];
  - 600 s in total;
  - 36000 frames;
  - a 4 GiB frame budget, checked at run time because image size isn't
    known at validate.
- Validate refuses bad literal durations or fps, a count mismatch and
  either cap. Mixed sizes are refused naming `images`.
- `images` is a declared media argument, so validate path-gates each entry
  and names the index (`images[N]`). This came from the bounce.

### D: `storyboard` skill + `animatic` template (#1145)
- The skill covers:
  - the Popcorn shape on Klein: up to 8 boards, with references cited as
    "image one";
  - the six-cut coverage set of one moment, redrawn from a hero frame;
  - a board as the start frame of an H3, LTX-2.5 or K6 shot;
  - the animatic recipe: boards, a TTS scratch track, a Music 3 bed, then
    the template.
- Boards are kept as `asset:boards/<scene>-NN.jpg`.
- `workflows/templates/animatic.json` (shape `utility`) runs
  `stills_to_video` and then `pair_audio(fit="video")`. The audio is
  trimmed or padded to the clip, with a warning. The manifest carries shots
  `still_1..N`.

### E: `analyze_face_drift` + `face-drift-check` (#1146)
- `analyze_face_drift(video, reference, shots=None, threshold=0.363,
  samples_per_shot=8)` runs YuNet detection and SFace embedding, on CPU or
  CUDA. Both weights are at pinned revisions and listed in the plan's
  weights. SFace is `opencv/face_recognition_sface` @ `3d708243…`.
- Findings come in the `assess.py` answer shape: `face_drift` (median
  below threshold) and `no_face`. Each finding's `at` and each
  `measurements[]` entry carry `shot`, `index`, `start_frame`,
  `num_frames`, `start` and `end` (seconds; null without fps), so shots
  that share a name stay distinct (fix-forward #1165, `bec9cbe6`).
- Validate refuses `threshold` outside [-1, 1] and `samples_per_shot`
  outside 1–64.
- `workflows/templates/face-drift-check.json` is the template. The
  `series-episodes` and `script-to-video` skills point at it.

### F: sitcom multicam grammar (#1147)
- `series-episodes/references/sitcom-multicam.md` lays out:
  - a wide master held throughout;
  - per-speaker singles from `templates/minimax/dialogue-short`;
  - `transcribe_audio` (`timestamps: "segment"`), then `plan_cuts`
    (`segment_by: "line"`) on the H3 grid for cuts on line boundaries.
- `SKILL.md` has one pointer line to it.

### G: named-move trial (#1148)
- 18 clips: 6 moves × H3, LTX-2.5 and K6, at seed 42, on one fixed subject
  at `wide`. Each was judged by its frames. The job ids, prompts and
  observations are in `docs/proposals/audits/2026-11-higgsfield/move-trial.md`.
- The verdicts are in the skill's table. Each family compile table also
  has a `measured (#1148)` column, and the 7 moves not in the trial read
  `not measured`.

| move | H3 | LTX-2.5 | K6 |
|---|---|---|---|
| static | honoured | honoured | honoured |
| dolly-in | honoured | honoured | honoured |
| dolly-out | honoured | honoured | honoured |
| truck-left | partial | honoured | partial |
| crane-up | honoured | honoured | partial |
| orbit | honoured | honoured | partial |

No cell was ignored, so no move carries "hint only, rarely honoured". The
skill defines that marking for a future trial.

## Deviations from the plan

- **D's estimate.** The plan said `validate_workflow` "quotes a near-zero
  GPU estimate". The animatic loads no model, so the estimate is reported
  as `unknown` with the reason "task-only workflow" until the box has run
  it. After that it quotes the template's own run history (`basis:
  "observed"`, `minutes: 0.0`), by design (#439/#921). The final check
  failed C-F647 on that, and fix-forward #1164 took it to Don as a record
  conflict. He ruled to amend the case, not the engine: C-F647 accepts
  `unknown` with the task-only reason, or `observed` with `minutes` under
  0.1 (case change dkackman/harnest#174). No code changed.
- **E's same-person fixture.** The final check flagged `face_drift` on
  `ep6-shot1-priya.mp4` against `priya-portrait.jpg` (median 0.352 under
  0.363). Fix-forward #1165 checked the measurement first; Don ruled the
  code right: that clip is not the portrait's person. The threshold stays
  0.363. C-F651 arm 1 was re-pointed at a fresh H3 render conditioned on
  the portrait (`asset:qa-cast/h3-priya-portrait-i2v.mp4`, median 0.483,
  no findings; case change dkackman/harnest#177). The same check found
  findings named only the shot, so #1165 added per-shot index, frames and
  times to them.
- **C's caps.** Beyond the plan's 600 s cap, C adds a 36000-frame cap and a
  4 GiB run-time byte budget. So 600 s at 120 fps is refused.
- **C's silence.** The plan said "an AudioVideo that is silent". After the
  bounce, it is a full-length silent mono 48 kHz track, not a clip with no
  audio stream, so the clip goes through `assemble-and-score`'s
  `resample_audio` path. `resample_audio`'s refusal of a video with no
  track was deliberately left alone, because many commands share it.
- **G's download.** The K6 weights
  (`kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers`, 24.9 GB) were
  not on lem, so the first K6 trial job downloaded them. This is the
  catalog model already in use, and the download was counted as part of
  Q4's approved spend.

## Bounces per stage

| Stage | Bounces | What |
|---|---|---|
| A #1142 | 1 | C-F629: the `light` axis had no per-value sentence in the LTX-2.5, H3 and Klein compile tables (only K6 had one), which also blocked part of C-F633. Fixed with a five-phrase `light` table in each (`285b5375`, merged `d4256478`). |
| B #1143 | 0 | |
| C #1144 | 1 | C-F645: `assemble-and-score` failed at `resample_audio` on a stills clip with no audio track. SE-F060: some `images` path probes were not refused at validate. The details were filed privately. Fixed with a silent track and `images` as a declared media argument (`6f8e12eb`, merged `2c371a1d`, arch review pass). |
| D #1145 | 0 | |
| E #1146 | 0 | |
| F #1147 | 0 | |
| G #1148 | 0 | |
| Final check | 1 | C-F647: the `animatic` estimate was `observed`, not `unknown`. A record conflict, not a build miss: fix-forward #1164 carried Don's ruling to amend the case; re-run passed on develop @ 06f08f37. |
| Final check | 2 | C-F651 arm 1: `face_drift` on the same person. The fixture's clip was not the portrait's person (Don's ruling, threshold unchanged); fix-forward #1165 re-pointed the case's fixture and added per-shot times to findings (`bec9cbe6`, merged `eb8d1424`); all three arms passed on lem @ eb8d1424. |

Cost: no stage comment recorded `usage:` figures, so none is given here.
G used the approved ~1 h of lem GPU (18 clips) plus the K6 download above.

## Deferred and left open

- **Tier 2 add-ons** (re-shoot, angles, speed LoRAs, relight, dubbing,
  swap): `docs/proposals/cinematic-addons.md`, a separate feature.
- **Moves not in the trial** (7 of the vocabulary's moves) and **Klein**
  (stills: moves compile to framing) read `not measured`. Back if a user
  reports one being ignored, or when a new trial is approved.
- **The trial's single seed and subject.** The verdicts are one sample per
  cell. A partial on truck-left (H3, K6) and crane-up/orbit (K6) may move
  with seed or subject.
- **`resample_audio` on a trackless video** still refuses. Back if another
  task needs to produce trackless clips for the scoring path.
- Pre-existing, found in the A–F review and filed separately: #1154.
