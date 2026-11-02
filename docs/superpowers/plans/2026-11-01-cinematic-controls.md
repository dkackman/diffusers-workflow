# Cinematic Controls (tier 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give users and agents Higgsfield's director vocabulary on dw's existing families. That means named camera, lens, light, look and grade choices compiled to each family's caption, a saved shot document, storyboards and timed animatics, a face-drift check between shots, and a sitcom multicam grammar. Every named choice is a hint, never a promise.

**Architecture:**
- **Model knowledge stays in the plugin.** It goes into two new cross-cutting skills (`cinematography`, `storyboard`) and into reference files under the existing skills, which are all at their 12 KiB cap.
- **Engine surface:** two new task commands, `stills_to_video` and `analyze_face_drift`, and two templates that run them. No MCP tool, REST route or workflow syntax changes.
- **The shot document is stored in the existing prompt library** (`save_prompt` / `prompt:`).

**Tech Stack:** Python 3 (`dw/tasks/`), OpenCV `cv2.FaceDetectorYN` + `cv2.FaceRecognizerSF`, workflow JSON templates, plugin skill markdown, pytest.

**Spec:** [docs/proposals/audits/2026-11-higgsfield/README.md](../../proposals/audits/2026-11-higgsfield/README.md), tier "Adopt now". Don ruled on 2026-11-01 that named moves are **hints**.

## Global Constraints

- Model knowledge lives in `plugins/dw/` and the catalog, never in engine code (CLAUDE.md, "Before you edit").
- Every `SKILL.md` stays at or under `SKILL_SIZE_LIMIT = 12 * 1024` bytes (`tests/test_plugin_skills.py:23`). Detail goes in `references/`.
- A new skill is named in `plugins/dw/README.md` and in `CLAUDE.md`'s plugin paragraph, and pinned by `tests/test_plugin_skills.py`.
- **Presets are hints.** Skill text never promises a move happens. It says to describe the move in the caption and check it with `get_output_frames`.
- **No camera-body brand names** (ARRI, RED, Cooke, Venice) as presets. Use abstract looks only.
- Code from no-licence, GPL or AGPL repos is never copied. Ideas only, with the project named.
- Weights are read at a pinned revision. SFace: `opencv/face_recognition_sface`, file `face_recognition_sface_2021dec.onnx`, revision `3d7082438a6e4551e840c9b2bb60b71e8da4b524`, Apache-2.0. YuNet: as in `dw/tasks/face_track.py:65-67`.
- Paths and assets go through `dw/security.py` validators. No `eval`, `exec` or `shell=True`.
- `docs/stabilization/baseline.json` `modules` goes 236 → 237 for the one new module (`dw/tasks/face_drift.py`). That is Q2 below.
- Compare backends with `get_device_type()` (`dw/__init__.py`).
- Warnings a caller must see go through `dw.events.emit_warning`, not `logger.warning`.

## Review Focus

1. **`stills_to_video` with stills of different sizes.** It must refuse and name both sizes, not stretch silently. Pinned in Task 3.
2. **`stills_to_video` with a duration shorter than one frame** (0.01 s at 24 fps). Each still must get at least one frame, and the shot map must still land on the frame grid. Pinned in Task 3.
3. **`analyze_face_drift` with a reference image that has no face, or two faces.** No face: refuse with a message naming the reference. Two faces: use the largest and say so in `notes`. Pinned in Task 4.
4. **`analyze_face_drift` on a shot where nobody is in frame** (an establishing shot). It must return a `no_face` finding for that shot, not a drift finding and not a crash. Pinned in Task 4.
5. **An agent following the `cinematography` skill for a family it doesn't cover** (Music 3, an image-only family). The skill must say the vocabulary is for video families plus Klein stills and stop there. Pinned by a text assertion in Task 1.

---

## Stages (the plan Don approves; each is one GitHub stage issue)

Stage letters match the harnest stage issues. Each stage's **acceptance intent** is what the source-blind tester writes `pending:` cases from.

### Stage A: Camera-and-look vocabulary (`cinematography` skill)

**Builds:**
- **A new skill `plugins/dw/skills/cinematography/`** with a closed vocabulary. It has six axes: shot size, camera move, lens character, light, era/stock, and genre/tempo.
- **One compiled sentence per value per family**, for H3, LTX-2.5, Kandinsky 6, and FLUX 3 Klein (stills).
- **A named-grade table**: noir, teal-orange, 70s stock, VHS and bleach-bypass. Each one is an `apply_lut` `palette` plus a `film_grain` setting.
- **A "presets are hints" rule** and the check that goes with it.
- **One pointer line in each family skill.**

**Acceptance intent (over MCP and the plugin):**
- A cold agent asked for "a slow dolly-in on a close-up, noir look" for LTX-2.5 gets a caption containing that family's compiled move sentence and shot size. It runs the noir grade through `apply_lut` with the table's palette, and that validates clean.
- The same request for Kandinsky 6 produces exactly one camera sentence.
- `validate_workflow` stays clean for a template run with a compiled caption.
- The skill names no camera brands.
- Asked for a camera move on Music 3, the skill says it doesn't apply.

### Stage B: Shot document, fix ladder, H3 lint (`script-to-video` references)

**Builds:**
- **New shot-list fields** in `script-to-video`: `shot_size`, `move`, `lens`, `light`, `duration`.
- **Saving the shot list** as a prompt-library entry, `prompt:<project>/shots`, holding the JSON text. Each shot's compiled caption is saved as `prompt:<project>/shot-NN`, and the workflow references it by name. A revision re-saves and re-runs only the changed shots.
- **A written fix ladder**: regional edit or face repair, then re-prompt, then reseed, then ask the user. Caps: 3 tries per shot (the existing budget) and a GPU-minute cap per project, quoted from `plan.estimate`.
- **An H3 prompt lint checklist**: overfull beat, a cut with no new information, missing `overall_soundscape`.

**Acceptance intent:**
- An agent following the skill on a 3-shot script ends with `list_prompts` showing `<project>/shots` and `<project>/shot-01..03`.
- Each run's arguments reference `prompt:<project>/shot-NN`, which `get_job` shows.
- After a one-shot revision, only that shot's job is queued again.
- `get_prompt` on `<project>/shots` returns JSON with the five new fields on every shot.

### Stage C: `stills_to_video` task

**Builds:** the task command `stills_to_video(images, durations, fps=24)`. It holds each still for its duration and returns an `AudioVideo` that is silent, with one shot record per still. It refuses mismatched sizes, durations that are non-positive or the wrong length, an fps outside (0, 120], and a total over 600 s.

**Acceptance intent:**
- Three 512×512 stills with durations [1, 2, 0.5] at 24 fps produce a 84-frame clip of 3.5 s. Its manifest has 3 shots at frames 0, 24 and 72.
- A single duration, given as a number, applies to every still.
- A duration of 0.01 s still gives one frame.
- Each of these is refused with a message naming the argument: mixed sizes, `durations` with the wrong count, a duration of 0 or less, fps 0, fps 121, and more than 600 s in total.
- The clip goes on to `concat_videos` / `assemble-and-score` like any other clip.

### Stage D: Storyboard and animatic (`storyboard` skill + `animatic` template)

**Builds:**
- **A new skill `plugins/dw/skills/storyboard/`.**
  - **Popcorn shape on FLUX 3 Klein:** up to 8 frames, references cited as "image one".
  - **Shot Board's six-cut coverage set:** establishing, medium, face close-up, over-the-shoulder, insert and low angle. You pick a hero frame and redraw the others from it.
  - **Board to start frame:** a board frame becomes an H3, LTX or K6 start frame.
- **An animatic recipe:** boards, then a TTS scratch track, then a Music 3 bed, then the new template `workflows/templates/animatic.json`. The template uses `stills_to_video` and an audio overlay through existing tasks.

**Acceptance intent:**
- `list_workflows(shape="shot")`, or whichever shape the catalog derives, lists `animatic`.
- `validate_workflow` on it with 3 board assets, durations and an audio asset is clean and quotes a near-zero GPU estimate.
- Running it gives a video whose length is the sum of the durations, whose audio is the provided track trimmed or padded to that length, and whose `get_output_frames` at each shot's midpoint shows that board.

### Stage E: `analyze_face_drift` task + `face-drift-check` template

**Builds:**
- **The task command** `analyze_face_drift(video, reference, shots=None, threshold=0.363, samples_per_shot=8)`.
  - It runs YuNet detection and SFace embedding on CPU or CUDA.
  - It reports the cosine similarity, per shot, between the reference face and the best-matching face in sampled frames.
  - Findings are `face_drift` (median below `threshold`) and `no_face`. They are places to look, in the `assess.py` answer shape.
- **A template** `workflows/templates/face-drift-check.json`.
- **Skill pointers** in `series-episodes` and `script-to-video`.

**Acceptance intent:**
- Each of these produces the expected findings:
  - a clip of the cast portrait animated by H3/LTX: `face_drift` absent;
  - the same check against a different person's portrait: `face_drift` on every shot;
  - a clip with an establishing shot: `no_face` on that shot only;
  - a reference with no face: a refusal naming `reference`.
- The plan or estimate lists the SFace weights.
- A threshold outside [-1, 1] is refused.

### Stage F: Sitcom multicam grammar (`series-episodes` reference)

**Builds:**
- **A sitcom section** in `series-episodes/references/`: a wide master held throughout, per-speaker singles from `dialogue-short`, and cuts on line boundaries from `transcribe_audio` and `plan_cuts`.
- **One pointer line** in `SKILL.md`.

**Acceptance intent:**
- A cold agent asked for "a sitcom scene, two characters, multicam" names the master, the singles and the line-boundary cut procedure, and gets each tool name right.
- `validate_workflow` on the cut it builds is clean.

### Stage G: Measure named moves (trial → measured), spend-gated

**Builds:** nothing in code. For each family (H3, LTX-2.5, K6), run 6 moves — static, dolly in, dolly out, truck left, crane up, orbit — at one seed, on a fixed subject. Judge each with `get_output_frames` and record honoured / partial / ignored in the `cinematography` skill's per-family table. Moves that are ignored are marked "hint only, rarely honoured".

**Acceptance intent:** the skill's table carries a measured verdict for each of the 18 cells, with the job ids that support it in `docs/proposals/audits/2026-11-higgsfield/move-trial.md`.

### Questions for Don

- **Q1 [scope]:** Add two task commands (`stills_to_video`, `analyze_face_drift`) and two new skills, with no MCP, REST or syntax changes. **Default:** yes.
- **Q2 [none]:** The module ratchet goes 236 → 237 for `dw/tasks/face_drift.py`. **Default:** yes. The alternative is growing `face_track.py` (893 lines) past its ceiling.
- **Q3 [trust]:** A new weight download: SFace, Apache-2.0, 37 MB, at a pinned revision. **Default:** yes.
- **Q4 [spend]:** Stage G uses about 18 lem clips at draft settings, roughly 45–60 GPU minutes. **Default:** yes, after A–F verify.

---

## Implementation tasks

### Task 1: `cinematography` skill (Stage A)

**Files:**
- Create: `plugins/dw/skills/cinematography/SKILL.md`, `plugins/dw/skills/cinematography/references/vocabulary.md`, `references/compile-h3.md`, `references/compile-ltx-2.5.md`, `references/compile-kandinsky-6.md`, `references/compile-flux3-klein.md`, `references/grades.md`
- Modify: `plugins/dw/skills/{minimax-h3,ltx-2.5,kandinsky-6,flux3-klein}/SKILL.md`, one pointer line each, staying under the cap (trim elsewhere in the same file if a skill is over).
- Modify: `plugins/dw/README.md`, `CLAUDE.md` plugin paragraph, `.claude-plugin/marketplace.json` and the plugin version, following the README's versioning rule.
- Test: `tests/test_plugin_skills.py`

**Interfaces:**
- Produces: the axis names and value ids that Tasks 2, 5 and 7 cite:
  - `shot_size` ∈ {extreme-wide, wide, full, medium, medium-close, close-up, extreme-close-up}
  - `move` ∈ {static, dolly-in, dolly-out, truck-left, truck-right, crane-up, crane-down, orbit, tracking, whip-pan, push-in, handheld, crash-zoom}
  - `lens` ∈ {wide, normal, long, anamorphic, vintage}
  - `light` ∈ {high-key, low-key, golden-hour, practical, noir}
  - `era` ∈ {8mm, 16mm, 35mm, vhs-dv, digital}
  - `tempo` ∈ {still, measured, brisk, frantic}
  - grade ids ∈ {noir, teal-orange, 70s-stock, vhs, bleach-bypass}

- [ ] **Step 1: Write the failing tests** in `tests/test_plugin_skills.py`:
  - `test_cinematography_skill_is_registered`: the skill dir is in the README's list and in CLAUDE.md.
  - `test_cinematography_names_no_camera_brands`: none of `ARRI`, `RED `, `Cooke`, `Venice`, `Alexa` appear anywhere under the skill dir.
  - `test_cinematography_vocabulary_covers_every_family`: each of the four `compile-*.md` contains a row for every `move` and `shot_size` id above.
  - `test_cinematography_says_presets_are_hints`: `SKILL.md` contains the phrase `a hint, not a promise`.
  - `test_cinematography_scopes_out_audio_families`: `SKILL.md` names `minimax-music3` as out of scope.
  - `test_grades_are_valid_palettes`: every palette in `grades.md`'s fenced JSON passes `dw.tasks.lut.check_lut_source(palette=...)`.
  - `test_family_skills_point_at_cinematography`: each of the 4 family `SKILL.md` contains `cinematography`.
- [ ] **Step 2:** `venv/bin/python -m pytest tests/test_plugin_skills.py -n0 -k cinematograph -v`. Expected: FAIL.
- [ ] **Step 3: Write the skill.**
  - Write each family's compiled sentences from that family's own caption spec. Read `ltx-2.5/SKILL.md` and its references for the shot-type/motion/viewpoint rule, `kandinsky-6` for the one camera sentence, and `minimax-h3/references` for Context-IR camera language.
  - For LTX, add the Lightricks card's compile rule: name what comes into view during the move.
  - The `SKILL.md` body covers: when to use it; the axes; "a hint, not a promise", and how to check with `get_output_frames`; how to pick a grade; and which skill each family's compile table lives under. Attribute ideas to Aither (MIT), CreatorOS (Apache-2.0) and OpenField (MIT) in a references footer.
- [ ] **Step 4:** Run the same pytest. Expected: PASS. Then run `venv/bin/python -m pytest tests/test_plugin_skills.py -n0`. Expected: all PASS, size caps included.
- [ ] **Step 5: Commit** with `feat(plugin): #<stage A> - cinematography skill: camera/look vocabulary compiled per family`.

### Task 2: Shot document, fix ladder, H3 lint (Stage B)

**Files:**
- Create: `plugins/dw/skills/script-to-video/references/shot-document.md`, `references/fix-ladder.md`, `plugins/dw/skills/minimax-h3/references/prompt-lint.md`
- Modify: `plugins/dw/skills/script-to-video/SKILL.md`. Its step 2 shot list gains the 5 fields, and pointer lines go to the three references, staying under the cap.
- Test: `tests/test_plugin_skills.py`

**Interfaces:**
- Consumes: Task 1's axis ids.
- Produces: prompt names `<project>/shots` (JSON text, an array of shot objects) and `<project>/shot-NN` (two digits, from 01). The shot-object keys are `id, cast, location, opening_state, ending_state, beat, must_not, join, shot_size, move, lens, light, duration`.

- [ ] **Step 1: Failing tests:**
  - `test_shot_document_names_its_prompt_keys`: `shot-document.md` contains `<project>/shots` and `<project>/shot-NN` and every key above.
  - `test_shot_document_uses_only_existing_tools`: every backticked `snake_case(` tool name in the three new files is a tool `dw_mcp` registers. Reuse the existing tool-name helper in that test file, if there is one.
  - `test_fix_ladder_order`: `fix-ladder.md` lists `regional-edit`, then `face-repair`, then re-prompt, reseed and ask, in that order.
  - `test_h3_lint_checks`: `prompt-lint.md` names `overall_soundscape`.
- [ ] **Step 2:** Run with `-k "shot_document or fix_ladder or h3_lint"`. Expected: FAIL.
- [ ] **Step 3: Write the references.** Recompiling only changed shots means re-saving `shot-NN` and calling `rerun_job` or `run_workflow` for that shot alone. Credit seriesmith (MIT), OpenReel (MIT) and open-higgsfield-skills (MIT) for ideas. Picky (GPL) is credited as an idea source only, with no text copied.
- [ ] **Step 4:** Run the full `tests/test_plugin_skills.py -n0`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat(plugin): #<stage B> - persisted shot document, fix ladder, H3 prompt lint`.

### Task 3: `stills_to_video` task (Stage C)

**Files:**
- Modify: `dw/tasks/video_utils.py`. Add `stills_to_video`.
- Modify: `dw/tasks/video_handlers.py`. Register the command with `domains={"fps": POSITIVE}`, plus the `static_check` for the counts it can see before run time.
- Modify: `docs/TASKS.md`. Add an entry beside `loop_frames`.
- Test: `tests/test_stills_to_video.py` (new)

**Interfaces:**
- Produces: `stills_to_video(images, durations, fps=24) -> AudioVideo`, where:
  - `images` is a list of PIL images, arrays, or image location dicts. Load them the way `ingredients_grid` loads its images.
  - `durations` is a float or a list of floats in seconds.
  - The return is `AudioVideo(frames=uint8 (n,h,w,3), audio=None, sample_rate=None, fps=fps, shots=[...])`. Shot records follow `dw/shots.py`, one per still with its start/end frame.
- Constants in `video_utils.py`: `MAX_STILLS_FPS = 120`, `MAX_STILLS_SECONDS = 600`.
- Frames per still = `max(1, round(duration * fps))`.

- [ ] **Step 1: Failing tests** in `tests/test_stills_to_video.py`:
  - `test_three_stills_hold_for_their_durations`: three 512×512 stills, `[1, 2, 0.5]`, fps 24 → `frames.shape == (84, 512, 512, 3)`, `fps == 24`, shot starts `[0, 24, 72]`.
  - `test_one_duration_applies_to_all`: `durations=2` with 2 stills at 12 fps → 48 frames.
  - `test_a_sub_frame_duration_still_gets_one_frame`: `[0.01]` at 24 → 1 frame.
  - `test_mismatched_sizes_refused`: 512×512 and 512×768 → ValueError naming both sizes.
  - `test_wrong_duration_count_refused`: 3 stills and 2 durations → ValueError naming `durations`.
  - `test_non_positive_duration_refused`: `[1, 0]` → ValueError.
  - `test_fps_bounds`: 0 and 121 refused, 120 accepted.
  - `test_total_over_600_seconds_refused`: `[601]` → ValueError naming 600.
  - `test_result_concats_with_concat_videos`: the output passed to `concat_videos` with itself gives 2× the frames.
  - `test_command_registered`: the registry has `stills_to_video`.
- [ ] **Step 2:** `venv/bin/python -m pytest tests/test_stills_to_video.py -n0 -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the function and its handler, following `_handle_loop_frames`. The silent `AudioVideo` must save as a video, whatever `result` path an image-less `AudioVideo` uses. Check `dw/media_types.py` save handling. If audio `None` isn't accepted, use `audio=numpy.zeros((1, 0))` with `sample_rate=None`, and pin that in a test.
- [ ] **Step 4:** Run the same pytest, then `venv/bin/python -m pytest tests/ -q` and `ruff check dw tests`. Expected: PASS, clean.
- [ ] **Step 5: Commit** with `feat(tasks): #<stage C> - stills_to_video: hold each still for a duration`.

### Task 4: `analyze_face_drift` task (Stage E)

**Files:**
- Create: `dw/tasks/face_drift.py`.
- Modify: `dw/tasks/face_track.py`. Expose the YuNet loader for reuse: rename `_detector` to `face_detector`, or add a public wrapper. Don't copy it.
- Modify: `dw/tasks/video_handlers.py` (or the handler file that holds `analyze_shots`), `dw/tasks/weights.py` if it needs a registry entry, `docs/TASKS.md`, and `docs/stabilization/baseline.json` (`modules` 237).
- Test: `tests/test_face_drift.py` (new)

**Interfaces:**
- Consumes: `assess.media_from`, `assess.resolve_shots`, and `assess._answer` (or its public equivalent) for the answer shape.
- Produces: `analyze_face_drift(video, reference, shots=None, threshold=0.363, samples_per_shot=8) -> dict`. The dict has these keys:
  - `measurements`, a list with one entry per shot: `{shot, frames_sampled, faces_found, similarity_median, similarity_min}`;
  - `findings`;
  - `rules_applied`;
  - `shots_source`;
  - `notes`.
- Constants: `SFACE_REPO = "opencv/face_recognition_sface"`, `SFACE_FILE = "face_recognition_sface_2021dec.onnx"`, `SFACE_REVISION = "3d7082438a6e4551e840c9b2bb60b71e8da4b524"`.
- Similarity is `FaceRecognizerSF.match(..., FR_COSINE)`. The 0.363 default is OpenCV's published cosine threshold for SFace.

- [ ] **Step 1: Failing tests** in `tests/test_face_drift.py`. Inject a fake detector and recognizer, as `tests/test_face_track.py` does, so no weights download.
  - `test_same_face_no_drift_finding`.
  - `test_other_face_drift_on_every_shot`.
  - `test_empty_shot_reports_no_face_only_there`: shot 1 has no detections, shot 2 matches → `no_face` at shot 1 only.
  - `test_reference_without_face_refused`: ValueError naming `reference`.
  - `test_reference_with_two_faces_uses_largest_and_notes_it`.
  - `test_threshold_bounds`: -1.01 and 1.01 refused.
  - `test_samples_per_shot_caps_reads`: a 100-frame shot reads 8 frames.
  - `test_required_weights_lists_sface_and_yunet`.
  - `test_command_registered`.
- [ ] **Step 2:** `venv/bin/python -m pytest tests/test_face_drift.py -n0 -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Per sampled frame: detect all faces, align and crop each with `recognizer.alignCrop`, embed it, and keep the best cosine match against the reference.
  - Per shot: the median and minimum of those best values over the frames that had a face.
  - Rules: `face_drift` when the median is below `threshold`; `no_face` when no sampled frame had a face.
  - Load the recognizer through `model_cache.cached_model`, like YuNet.
- [ ] **Step 4:** Run the same pytest, the full suite, `ruff`, and `venv/bin/python scripts/arch_metrics.py` (or the ratchet test). Expected: PASS, with modules at 237.
- [ ] **Step 5: Commit** with `feat(tasks): #<stage E> - analyze_face_drift: SFace identity check per shot`.

### Task 5: `animatic` and `face-drift-check` templates, `storyboard` skill (Stages D + E)

**Files:**
- Create: `workflows/templates/animatic.json`, `workflows/templates/face-drift-check.json`, `plugins/dw/skills/storyboard/SKILL.md`, `plugins/dw/skills/storyboard/references/coverage-set.md`, `references/animatic.md`.
- Modify: `plugins/dw/README.md`, `CLAUDE.md`, and the plugin version (as in Task 1); `series-episodes/SKILL.md` and `script-to-video/SKILL.md` get one pointer line each to `face-drift-check`.
- Test: `tests/test_plugin_skills.py`, plus whatever the catalog uses to validate every template (find the test that loads every `workflows/templates/**/*.json`, such as `tests/test_templates*.py`, and confirm it picks the new files up).

**Interfaces:**
- Consumes: `stills_to_video` (Task 3), `analyze_face_drift` (Task 4), and the existing audio tasks for laying a track under a clip. Use the same pairing step `assemble-and-score.json` uses.
- `animatic.json` variables: `boards` (a list of image refs), `durations` (a number or a list), `fps` (default 24), `soundtrack` (an audio ref).
- `face-drift-check.json` variables: `video`, `reference`, `threshold` (default 0.363).

- [ ] **Step 1: Failing tests:**
  - `test_storyboard_skill_registered`.
  - `test_storyboard_names_six_coverage_cuts`: establishing, medium, close-up, over-the-shoulder, insert and low angle all appear in `coverage-set.md`.
  - `test_storyboard_caps_boards_at_eight`.
  - Template validation passes for both new templates (`python -m dw.validate workflows/templates/animatic.json`).
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3: Write the templates and the skill.** Board generation in the skill uses the existing Klein `multi-reference` and `consistent-set` templates. Credit Higgsfield's Popcorn (concept) and Shot Board (MIT).
- [ ] **Step 4:** Run the full suite and `python -m dw.validate` on both templates. Expected: PASS, clean.
- [ ] **Step 5: Commit** with `feat(catalog): #<stage D>/#<stage E> - animatic + face-drift-check templates, storyboard skill`.

### Task 6: Sitcom multicam grammar (Stage F)

**Files:**
- Create: `plugins/dw/skills/series-episodes/references/sitcom-multicam.md`.
- Modify: `plugins/dw/skills/series-episodes/SKILL.md` (one pointer line, under the cap).
- Test: `tests/test_plugin_skills.py`.

- [ ] **Step 1: Failing tests:**
  - `test_sitcom_reference_names_its_tools`: names `dialogue-short`, `transcribe_audio` and `plan_cuts`, and each backticked tool or template exists.
  - `test_series_points_at_sitcom`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3: Write the reference.** The wide master is held through the scene, singles come per speaker, cuts land on line boundaries from the transcript's word timings, and the master is the fallback when a single fails the face-drift check. Credit One-shot Multicam (MIT).
- [ ] **Step 4:** Run the full `tests/test_plugin_skills.py -n0`. Expected: PASS.
- [ ] **Step 5: Commit** with `feat(plugin): #<stage F> - sitcom multicam grammar`.

### Task 7: Move trial (Stage G, after A–F verify and Q4)

- [ ] For each of H3 (draft tier), LTX-2.5 (distilled) and K6, run the 6 moves at seed 1234 on one subject. The prompt is the family's compiled sentence from Task 1. Each run gets a `note` naming the move.
- [ ] Judge each with `get_output_frames` (first, middle and last frame) as honoured, partial or ignored.
- [ ] Write `docs/proposals/audits/2026-11-higgsfield/move-trial.md` with the 18 cells and their job ids, and update each `compile-*.md` table with a `measured` column.
- [ ] Commit with `docs(plugin): #<stage G> - named-move trial verdicts`.
