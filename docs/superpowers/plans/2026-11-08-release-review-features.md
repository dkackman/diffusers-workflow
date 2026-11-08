# Release-review features (0.20.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the five remaining "even better if" features from the 0.20.0 readiness review (rerun overrides, chain resume, LUFS level matching, a narration script check, the job-page timeline strip) on top of the review-defect fixes.

**Architecture:** Each feature is one task in its own module or a roomy existing one; every new knob defaults off or to today's behaviour; run-time findings reach the caller as job warnings; disk paths go through `dw/security.py`/`dw/locations.py`. Tasks A, B, D, E are backend (Python, pytest), F is the Svelte UI plus one small exporter change. They are independent and run in parallel worktrees off `develop` after `fix/release-review-0-20-0` merges.

**Tech Stack:** Python 3.14, FastAPI/Pydantic, PyAV, numpy/torch, pytest (-n auto); Svelte 5, Vite, vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-11-08-release-review-features-design.md`

## Global Constraints

- Run Python tests with the main checkout's venv: `/Users/don/src/dkackman/diffusers-workflow/venv/bin/python -m pytest <file> -q -n0`; the full suite with `DW_DEVICE=cpu ... -m pytest tests/ -q -p no:cacheprovider`.
- Before every commit: `ruff check dw dw_mcp tests scripts && ruff format --check ...`; before the task's last commit: `python scripts/arch_metrics.py --check docs/stabilization/baseline.json` exits 0 (no module over 1,100 lines, no new function over the C901 limit, no new import cycle); UI tasks: `cd ui && npm run metrics -- --check ../docs/stabilization/ui/baseline.json` (600-line file ceiling).
- Modules near the ceiling: `chain.py` 1,014, `plan.py` 1,086, `validation.py` 1,070, `workflow_run.py` 1,096, `routes/media.py` 1,093, `server/jobs.py` 1,100, `components.py` 1,046. New code of any size goes in a new module; the seam map `docs/ARCHITECTURE.md` gets one row per new rule (module, rule, enforcing test).
- Commit headlines use a scope, never `fix: #N` / `feat: #N` (that auto-closes the issue). End every commit message with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- MCP docstrings: one added sentence per tool; `tests/test_mcp_surface_budget.py` must stay green.
- The JSON routes the UI calls declare response models in `dw/server/api_models.py`; a changed model needs `python scripts/dump_openapi.py` (only under the pins in `constraints-openapi.txt` - skip the dump if the venv's FastAPI differs and say so) and `cd ui && npm run gen:api`.
- No GPU this week: every test is CPU-only and mocks model loads.

## Review Focus

1. Rerun override of the seed variable together with `new_seed: true` - expect a 400 naming both, not a silent choice (Task 1, step 1 test 3).
2. Resume where the kept run used a different `num_frames` or `trim_frames` - expect a refusal naming the segment and both counts, never a plan change (Task 2, step 1 test 2).
3. `match_levels: "lufs"` on a shot under 0.4 s (`integrated_lufs` None) - expect the rms fallback with a warning naming the shot, not a crash or silence (Task 3, step 1 test 3).
4. `check_narration` on a transcript that repeats a take (a looped speech model) - expect the repeats as `inserted` on that take and `ok: false`, not a mis-alignment that fails every later take (Task 4, step 1 test 3).
5. An `.otioz` exported before this release (clips with empty `metadata`) on the job page - expect the strip to draw with no copy buttons, not a blank panel (Task 5, step 1 test 2).

---

### Task 1: `rerun_job` argument overrides

**Files:**
- Modify: `dw/server/jobs.py:497-540` (`rerun_spec`), `dw/server/routes/jobs.py:413-500` (`RerunRequest`, route), `dw/server/api_models.py:413` (`rerun_of` neighbourhood), `dw_mcp/diagnose.py:376-396`, `dw_mcp/tools_jobs.py:181-205`
- Docs: `docs/MCP.md` (rerun_job), `docs/SERVER.md` (rerun route), `plugins/dw/skills/long-form/references/quality-and-speed.md` (one line: promote with `rerun_job(..., arguments={"tier": "best"})`)
- Test: `tests/test_server.py` (route), `tests/test_mcp_diagnose.py` (tool), `tests/test_api_contract.py` if a new response key is added

**Interfaces:**
- Produces: `JobManager.rerun_spec(job_id, new_seed=False, overrides=None) -> (spec, arguments) | None`; `RerunRequest.arguments: dict[str, Any] | None = None`; `diagnose.rerun_job(client, job_id, acknowledged_cost=False, new_seed=False, arguments=None)`; MCP `rerun_job(job_id, acknowledged_cost=False, new_seed=False, arguments: dict | None = None)`; the job's stored spec gains `"overrides": {...}` when any were given and `JobDetail` reports it as `overrides: dict = sometimes(...)`.

- [ ] **Step 1: Write the failing route tests** in `tests/test_server.py` next to the existing rerun tests: (1) `test_rerun_overrides_merge_onto_the_stored_arguments` - submit a job with `{"prompt": "a", "num_images_per_prompt": 1}` on the text-to-image fixture, rerun with `{"arguments": {"num_images_per_prompt": 2}}`, assert the new job's `arguments["num_images_per_prompt"] == 2` and `arguments["prompt"] == "a"` and `overrides == {"num_images_per_prompt": 2}`; (2) `test_rerun_override_of_an_undeclared_variable_is_refused` - `{"arguments": {"nope": 1}}` -> 400 with `Unknown variable 'nope'` in detail; (3) `test_rerun_override_of_the_seed_with_new_seed_is_refused` - a seeded fixture, `{"new_seed": true, "arguments": {"seed": 7}}` -> 400 naming `seed` and `new_seed`.
- [ ] **Step 2: Run them**: `pytest tests/test_server.py -k rerun -n0 -q` - expect 3 FAIL (422 or key errors).
- [ ] **Step 3: Implement** `rerun_spec(..., overrides=None)`: merge `{**arguments, **overrides}` before the `new_seed` block; when `overrides` names `self.seed_variable(job_id)` and `new_seed` is true raise `ValueError("'new_seed' and an override of the seed variable 'X' cannot both be given")`; `RerunRequest.arguments` with a description; the route passes `overrides=body.arguments`; `manager.rerun(...)` stores `overrides` in the job spec (route passes it through; `describe` emits it `sometimes`). Unknown variables are already refused by `admit_for` -> `dw/variables.py:283`; assert the message surfaces as 400.
- [ ] **Step 4: Run** the three tests -> PASS; run `tests/test_api_contract.py tests/test_job_history_parity.py -n0 -q` -> PASS.
- [ ] **Step 5: MCP**: add `arguments` to both `rerun_job` functions, one docstring sentence: "`arguments` overrides variables for the rerun (a draft promoted with `{"tier": "best"}`); validate first, since an override changes the plan a bound acknowledgement must name." Test in `tests/test_mcp_diagnose.py`: the posted body carries `arguments`. Run `tests/test_mcp_diagnose.py tests/test_mcp_surface_budget.py -n0 -q` -> PASS.
- [ ] **Step 6: Docs** (MCP.md, SERVER.md, the skill line) and a seam-map row is not needed (no new rule). ruff, ratchet, commit: `feat(server,mcp): rerun_job takes argument overrides - promote a draft in one call`.

### Task 2: `chain.resume_from`

**Files:**
- Create: `dw/pipeline_processors/chain_resume.py` (all resume logic)
- Modify: `dw/workflow_schema.json` (`$defs/chain.resume_from`: `{"oneOf": [{"type": "string"}, {"type": "null"}]}`, description from the spec), `dw/pipeline_processors/chain_config.py:119` (`self.resume_from`), `dw/pipeline_processors/chain.py:91-240` (hook calls only; keep under 1,100 lines), `dw/pipeline_processors/chain_rules.py` (validate rule), `dw/plan.py` or `dw/cost_scaling.py` (estimate N - k), `workflows/templates/minimax/chained-segments.json` (`save_segments: true`, `resume_from` variable), `workflows/templates/ltx2/chained-segments.json` (`resume_from` variable), `docs/WORKFLOW_GUIDE.md` (chain section), `plugins/dw/skills/long-form/references/run-and-repair.md`, `docs/ARCHITECTURE.md` (row)
- Test: `tests/test_chain_resume.py` (new), `tests/test_chain.py` (template defaults), `tests/test_guide_chain_validation.py` or the chain-rules test file (validate rule)

**Interfaces:**
- Consumes: `chain_repair.segment_files(chain_output) -> list[str]` (siblings in index order, complete-set check); `chain_spill.SegmentSpill`, `_decode_segment(path) -> uint8 tensor`; `chain_continuity.GuideContinuity/LastFrameContinuity.extract(artifact)`; `_SegmentCarry(frames, audio, sample_rate, audio_latents)`; `dw/media.py` audio decode (`decode_rgb_frames`, the probe for audio); `dw/locations.py`/`dw/runs.py resolve_output_reference` for the reference.
- Produces: `chain_resume.ResumeState` with `kept: list[str]` (copied files in the new spill), `count: int`, `carry: _SegmentCarry | frame`, `anchor_frames`, `keystone_frame`, `warning: str`; `chain_resume.load(config, pipeline, spill, base_dir) -> ResumeState | None`; `chain_resume.check_plan(kept_paths, plan) -> None` (raises ValueError naming segment, kept frames, planned frames); `chain_resume.failure_hint(spill) -> str | None` (the `output:` reference of the first spilled file, for the failure warning).

- [ ] **Step 1: Write the failing tests** in `tests/test_chain_resume.py` with `tests/test_chain.py`'s `FakePipeline` and a tmp output dir: (1) `test_a_last_frame_chain_resumes_after_k_kept_segments` - run a 4-segment chain with `save_segments: true` whose fake pipeline raises on segment 3, assert 2 segment files exist and the raised error's job warning (`chain_resume_hint`) names the first file's `output:` reference; then run the same chain with `resume_from` set to that reference, assert the fake pipeline was called exactly 2 times, the output has 4 shots with `segment_file` set on all four, and a `chain_resumed` warning "Resumed 2 of 4 segments from ..." was recorded; (2) `test_resume_refuses_a_kept_segment_of_another_length` - kept files from `num_frames: 29` under a plan of 37 -> ValueError containing "segment 1", "29" and "37"; (3) `test_guide_resume_warns_about_the_decoded_audio_guide` - `continuity: guide` resume, the `chain_resumed` warning mentions the decoded track; (4) `test_latent_carry_and_resume_are_refused_at_validate` (chain_rules finding); (5) `test_kept_files_are_copied_not_moved` - originals still exist after the resumed run.
- [ ] **Step 2: Run** `pytest tests/test_chain_resume.py -n0 -q` -> 5 FAIL.
- [ ] **Step 3: Implement** `chain_resume.py`: resolve `resume_from` through `resolve_output_reference` (bare paths refused by it); `segment_files` for siblings; `check_plan`; copy files into `spill` (append to `spill.paths`, set `spill.frame_count`) via `shutil.copyfile` to `output_file_path(...)` names; build the carry with the configured continuity's `extract` over a decoded artifact of the last kept file (frames tensor -> PIL via `frames_as_pil_list` if `extract` wants PIL; audio decoded from the mp4; `audio_latents=None`); `anchor_frames`/`keystone_frame` from the first file; look_hold: call `hold.observe`-equivalent over each kept segment's frames in order if LookHold exposes it, else set `hold` to observe the last kept only and say which in the module docstring. In `chain.py`: after `spill` is built, `resumed = chain_resume.load(...)`; skip `segment.index < resumed.count` in the loop (lay their shots down from the files' frame counts); set `carry`, `anchor_frames`, `keystone_frame` from `resumed`; record the warning through the existing run-warning path (`argument_warnings`/the chain's `warn` helper). In the `finally`/except path: when `spill` has paths and the run is raising, record `chain_resume_hint`.
- [ ] **Step 4: Run** the five tests -> PASS; `pytest tests/test_chain.py tests/test_chain_duration.py tests/test_chain_reroll.py -n0 -q` -> PASS.
- [ ] **Step 5: Templates and estimate**: `minimax/chained-segments` `save_segments: true` (keep `keep_segments: "variable:keep_segments"`), both chain templates gain `resume_from` (type string, default null, description from the spec) wired to the chain block; `chain_segment_count`/the estimate price N - k when the reference resolves (a `dw/plan.py` hook calling `chain_resume.kept_count(definition, arguments, base_dir) -> int`), else N with the plan text "priced for every segment; resume_from is not resolvable at validate". Tests: `tests/test_chain.py` template default assertion updated with the reason; an estimate test in `tests/test_plan.py`.
- [ ] **Step 6: Docs**: WORKFLOW_GUIDE chain section (`resume_from`, the failure hint), run-and-repair.md recipe "a chain that died at segment k", the ARCHITECTURE row ("a resume never changes the plan" - `chain_resume.check_plan`, `tests/test_chain_resume.py`). ruff, ratchet, full suite; commit `feat(chain): resume_from reuses a dead chain's kept segments`.

### Task 3: `match_levels: "lufs"`

**Files:**
- Modify: `dw/tasks/joins.py:501-600` (`LEVEL_MEASURES` source, `DEFAULT_MATCH_DBFS`, `match_levels`), the module that defines `LEVEL_MEASURES` (grep `LEVEL_MEASURES =`), `dw/tasks/concat_videos.py:36-125` (`match_levels_lufs` argument + docstring), `dw/tasks/dissolve_videos.py` if it exposes `match_levels`, `dw/task_domains.py` (choice list), `dw/tasks/video_handlers.py` (domains), `docs/TASKS.md` concat_videos, `plugins/dw/skills/long-form/references/audio.md` (one line)
- Test: `tests/test_concat_videos.py`, a new `tests/test_match_levels_lufs.py` for the pure function

**Interfaces:**
- Consumes: `dsp.integrated_lufs(samples, rate) -> float | None`, `dsp.true_peak_dbfs(samples) -> float`, the existing `match_levels_held` warning path.
- Produces: `joins.match_levels(waveforms, measure, target_dbfs=None, command=..., names=None, sample_rate=None, target_lufs=None)`; `concat_videos(..., match_levels_lufs=None)`; `DEFAULT_MATCH_LUFS = -16.0`; `LUFS_TRUE_PEAK_CEILING_DBTP = -1.0`.

- [ ] **Step 1: Write the failing tests** in `tests/test_match_levels_lufs.py`: (1) two 2 s pink-noise-like waveforms at different gains come out within 0.5 LU of -16 LUFS each; (2) a waveform whose gain to -16 would pass -1 dBTP is held there and the `match_levels_held` warning names it; (3) a 0.2 s shot is matched by rms and a warning `match_levels_short_shot` names "video 1"; (4) `concat_videos(match_levels="lufs", match_levels_dbfs=-20)` raises naming `match_levels_lufs`; (5) `-23` target works when passed.
- [ ] **Step 2: Run** -> 5 FAIL (`'lufs'` not in LEVEL_MEASURES).
- [ ] **Step 3: Implement**: add `"lufs"` to `LEVEL_MEASURES`; `match_levels` takes `sample_rate` and `target_lufs` (default `DEFAULT_MATCH_LUFS`); for `"lufs"` the per-shot measure is `integrated_lufs(w, sample_rate)`, gain = target - measured, ceiling by `true_peak_dbfs(w * gain) <= -1.0` (hold and warn as rms does), None -> rms path at `DEFAULT_MATCH_DBFS["rms"]` plus the short-shot warning; `concat_videos` passes `sample_rate` (already resolved in `_prepare_inputs`) and `match_levels_lufs`; refuse `match_levels_dbfs` with `"lufs"`.
- [ ] **Step 4: Run** the new file and `tests/test_concat_videos.py tests/test_dissolve_videos.py tests/test_introspection.py -n0 -q` -> PASS.
- [ ] **Step 5: Domains and docs**: `task_domains`/`video_handlers` choice list gains `lufs` and `match_levels_lufs: NON_POSITIVE`; TASKS.md concat_videos rows for `match_levels` (add lufs, -16 default, -23 EBU R128 note, -1 dBTP hold, short-shot fallback) and `match_levels_lufs`; audio.md: "deliver at -16 LUFS: `match_levels: lufs`". ruff, ratchet, commit `feat(audio): match_levels lufs - a BS.1770 delivery target on the join`.

### Task 4: `check_narration`

**Files:**
- Create: `dw/tasks/narration_check.py`, `workflows/templates/check-narration.json`, `tests/test_narration_check.py`
- Modify: `dw/tasks/task.py:26` (import for registration), `dw/tasks/model_handlers.py` or the new module's `_register` (follow `text_takes._register`), `docs/TASKS.md` (section after *Splitting Text into Takes*), `plugins/dw/skills/long-form/references/audio.md`, `workflows/templates/README.md` if templates are listed there, `tests/test_catalog_structure.py` fixtures if the catalog count is pinned
- Test: `tests/test_narration_check.py`, `tests/test_bench_workflows.py`-style template validation (`python -m dw.validate workflows/templates/check-narration.json`)

**Interfaces:**
- Consumes: `audio_transcription.model_source(arguments)`, `required_weights(arguments)`, `cached_model(key, loader)`, `waveform_and_rate`, `_downmixed_mono`, `_ASR_SAMPLE_RATE`; `text_takes.split_takes(text, max_seconds, words_per_second)`.
- Produces: `narration_check.normalize(text) -> list[str]`; `narration_check.align(expected_tokens, heard_tokens) -> list[tuple[str, int|None, int|None]]` (opcode-level pairs); `narration_check.check_narration(takes, audio, device="cpu", sample_rate=None, model_name=None, max_word_error_rate=0.1, transcribe=None) -> dict` where `transcribe` is an injectable callable `(waveform, rate, device, model_name) -> str` (tests pass a fake; production default uses the shared loader); result `{"takes": [...], "checked": int, "failed": [int], "worst": {"index": int, "word_error_rate": float} | None}`; warnings `narration_take_mismatch` per failing take through the task warning channel (`dw/task_problems.py` or the pattern `split_takes`'s neighbours use - grep `warn(` in `dw/tasks/audio_slice.py`).

- [ ] **Step 1: Write the failing tests**: (1) `test_a_word_perfect_narration_passes` - takes `["the cat sat.", "on the mat."]`, fake transcript "The cat sat on the mat" -> every take `ok`, `failed == []`; (2) `test_a_dropped_word_is_attributed_to_its_take` - transcript "the cat on the mat" -> take 0 `dropped == ["sat"]`, rate 1/3, `ok False` at 0.1, take 1 ok; (3) `test_a_looped_take_shows_as_inserted_and_later_takes_still_align` - transcript "the cat sat the cat sat on the mat" -> take 0 `inserted == ["the", "cat", "sat"]`, take 1 ok; (4) `test_the_lengths_need_not_match_but_a_string_is_split` - `takes="..."` string splits with `split_takes` defaults; (5) `test_each_failing_take_warns` - the warning text names the index, the rate and the first dropped words.
- [ ] **Step 2: Run** `pytest tests/test_narration_check.py -n0 -q` -> 5 FAIL.
- [ ] **Step 3: Implement** the module; `normalize` lower-cases, strips punctuation with `re.sub(r"[^\w\s]", " ", ...)`, splits on whitespace; `align` uses `difflib.SequenceMatcher(None, expected, heard, autojunk=False).get_opcodes()`; attribute by expected-token index ranges per take, inserted tokens to the take whose expected range the opcode's `i1` falls in (the last take when at the end); `word_error_rate = (len(dropped) + len(inserted)) / max(1, expected_words)`; register the command `check_narration` with `returns="json"`, `required_weights="dw.tasks.narration_check.required_weights"` (re-export of the transcription module's), `consumes_device=True`, domains `max_word_error_rate` in [0, 1].
- [ ] **Step 4: Run** -> PASS; `pytest tests/test_task_domains.py tests/test_introspection.py -n0 -q` -> PASS.
- [ ] **Step 5: Template** `workflows/templates/check-narration.json`: variables `text` (string, default null), `takes` (list, default null), `audio` (required reference, `asset:`/`output:`), `max_word_error_rate` (0.1), `model_name`; one step `check` with the task, `result.content_type: application/json`, `file_base_name: narration-check`; description and a cost row (text task, no GPU minutes beyond the ASR; copy `split-takes`' row shape). `python -m dw.validate workflows/templates/check-narration.json` passes; the catalog-structure tests pass (update a pinned count with the reason).
- [ ] **Step 6: Docs**: TASKS.md section (arguments table, result shape, the warning), audio.md: "after `narration-long`, run `templates/check-narration` on the narration with the same text; re-speak the takes it lists", ARCHITECTURE row if the warning channel rule is new. ruff, ratchet, commit `feat(tasks): check_narration - the script against the speech, per take`.

### Task 5: the job-page timeline strip

**Files:**
- Create: `ui/src/lib/timeline.ts`, `ui/src/lib/timeline.test.ts`, `ui/src/lib/job/TimelineStrip.svelte`, `ui/src/lib/job/TimelineStrip.test.ts`
- Modify: `ui/src/lib/job/JobResults.svelte:36-70` (fetch metadata for `.otioz` too; render the strip), `dw/tasks/timeline_export.py:66-120,269-330` (clip `metadata.dw.source`), `dw/tasks/timeline_export.py:370-470` (`summarize_bundle` surfaces `source`), `docs/SERVER.md` web UI section (one sentence), `docs/TASKS.md` export_timeline (metadata.dw.source)
- Test: `tests/test_timeline_export.py` (source in the clip metadata and the summary), the two vitest files; `ui/e2e` only if `tests/` already has an `.otioz` fixture the e2e server can serve (else skip and say so)

**Interfaces:**
- Consumes: `api.galleryMetadata(file, workspace)` -> `{ media: {kind: "timeline", name, otio_version, has_audio_track, tracks: [{kind, name, clip_count, seconds, items: [{type, name, start_frame, frames, rate, seconds, media?, source?}]}], members: string[] } }`; `CopyButton.svelte` (`text`, `title` props).
- Produces: `timeline.ts`: `type TimelineSummary`, `type TimelineItem`, `isTimeline(media: unknown): media is TimelineSummary`, `shares(track: TimelineTrack): number[]` (each item's seconds / track seconds, gaps included), `label(item): string` ("name · 4.2 s"); exporter: each OTIO Clip's `metadata: {"dw": {"source": "<output:... reference>"}}` when known; summary item `source: str | None`.

- [ ] **Step 1: Write the failing tests**: Python in `tests/test_timeline_export.py`: (1) an exported joined bundle's clips carry `metadata.dw.source == "output:<run>/<file>"` for the exported video (derive the reference with the helper `dw/runs.py` uses to name a run file; grep `def output_reference_for` / `is_output_reference` and the gallery route's `reference` field), and `summarize_bundle` returns it as `source`; (2) a bundle whose clips have empty `metadata` summarizes with `source: None`. vitest `timeline.test.ts`: `isTimeline` true/false, `shares` sums to 1 with a gap, `label` format. Component test `TimelineStrip.test.ts`: renders one block per item with its label, a copy button only for items with `source`, none for a legacy summary.
- [ ] **Step 2: Run** `pytest tests/test_timeline_export.py -n0 -q` and `cd ui && npx vitest run src/lib/timeline.test.ts src/lib/job/TimelineStrip.test.ts` -> FAIL.
- [ ] **Step 3: Implement** the exporter metadata (joined mode: the video's reference on every clip; sources mode: each source's), the summary `source`, `timeline.ts`, `TimelineStrip.svelte` (flex row per track, `flex-basis` from `shares`, min block width 48 px with horizontal scroll, hatched gaps via a CSS gradient on the existing tokens, name + seconds label, `CopyButton` with `text={item.source}`; audio track row under the video row; members list as `DownloadLink`s when a member is one of the job's files), and the `JobResults` hook: include `.otioz` files in the metadata fetch (keep the image comment accurate) and render `<TimelineStrip summary={...}/>` under that file's row when `isTimeline(fileMeta[file]?.media)` - note `fileMeta` stores `r.metadata` today; store the whole response for `.otioz` or add a sibling map `fileMedia`.
- [ ] **Step 4: Run** the tests -> PASS; `npx svelte-check --threshold error`, `npm run lint`, `npx vitest run`, `npm run build` -> clean.
- [ ] **Step 5: Docs** (SERVER.md one sentence under the job page; TASKS.md export_timeline notes `metadata.dw.source`). `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` exits 0; Python ruff + ratchet; commit `feat(ui,tasks): the job page draws an .otioz timeline; clips carry their source reference`.

---

## Merge order

`fix/release-review-0-20-0` first (it changes `ltx2/chained-segments.json`, `chain_config.py`, `chain.py`). Then A, D, E, F in any order; B last, rebased onto the result (it touches both chain templates and `chain.py`). After each merge: full suite, both ratchets.
