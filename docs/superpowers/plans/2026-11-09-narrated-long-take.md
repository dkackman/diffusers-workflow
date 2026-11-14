# Narrated Long Take Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One unbroken, lip-synced H3 take of a speaking subject, four minutes or more, driven by a narration made first. A cold agent can plan, probe, continue and repair it from the catalog, the skills and MCP alone.

**Architecture:**
- **Narration first.** Two new utility tasks turn a speech track into a retimed narration and per-segment lines laid on the chain's frame grid.
- **The take.** A guide chain holds each segment's slice of that narration (`match_audio` + `hold_audio`) and muxes the unsliced track over the take.
- **The drift counter.** It joins the chain's latent carry, a new latent keystone and the late capstone, applied every Nth segment. It is benched against today's presets and promoted only if it wins.
- **Running it.** A probe stops after k segments, a resume continues it, per-segment identity and lip-sync measures name failures, and repair holds the narration.
- **The agent layer.** One template, a skill rule and a guide section carry it.

**Tech Stack:** Python 3.12, PyTorch, diffusers modular MiniMax-H3, the dw engine, Whisper (`transformers`), OpenCV YuNet/SFace, pytest (xdist), the dw plugin (Markdown skills), `gh`, lem (2x RTX 3090) over ssh.

**Spec:** `docs/proposals/narrated-long-take.md` (approved by Don 2026-11-09). Executors read both. Issue #1308 (ruling posted 2026-11-09) is Phase 0.

## Global Constraints

- **Branch.** Work on `feat/narrated-long-take` off `develop`. Merge to `develop` at each deploy gate (Tasks 10, 15), then push. Never push to `master`.
- **lem.** Don deploys lem and restarts its server. When a deploy is due, ask him to run `scripts/deploy.sh develop` and to set `DW_EXPERIMENTAL_REFINE_WINDOWS=1` in lem's checkout `.env` for the bench period (Task 10). Remove the flag again at Task 15.
  - One int8 H3 job per host at a time.
  - Read run state over MCP or HTTP; never via `pgrep` self-matches.
- **Model knowledge lives in the catalog and `plugins/dw/`, never in engine code.** Prompt structure comes from template variables (a `prompt_template` with placeholders). No H3 prompt text goes in `dw/`.
- **Licences.**
  - GPL-3.0 (geiru), AGPL-3.0 (H3MD), no-redistribution (Smite79) and unlicensed (Longshot) sources contribute concepts only.
  - matlowai (CC BY 4.0) and NerdNotes (CC BY-SA) are cited in `docs/ATTRIBUTIONS.md` *Ideas reimplemented here*.
- **Security.** Any new media argument goes through `dw/locations.py` / `PIPELINE_MEDIA_ARGUMENTS` and the `dw/security.py` validators. Never `eval`, `exec` or `shell=True`.
- **Ratchets.**
  - `scripts/arch_metrics.py --check docs/stabilization/baseline.json` passes on every commit. New modules raise `"modules"` from 261 in a commit that says why (precedent 06bbc3e8).
  - No function over 150 lines, no new import cycles.
  - `test_dw_patch_targets` must not rise: mock through module-level seams with `monkeypatch`, not `patch("dw....")`.
- **Budgets.**
  - `SKILL_SIZE_LIMIT` 12288 bytes per SKILL.md: long-form has about 950 free, minimax-h3 about 430.
  - `COMPACT_BUDGET` 14_100 and `FILTERED_BUDGET` 1_500 in `tests/test_catalog_structure.py`: a raise is logged in the comment above it, with the measurement.
  - `SURFACE_BUDGET` 16_000 for MCP tool text: add no MCP tool. New behaviour is reached through tasks and templates.
- **Tests.**
  - TDD.
  - `venv/bin/python -m pytest -n0 tests/<file>.py` for one file. The full suite (`venv/bin/python -m pytest tests/`) passes before every merge.
  - `ruff format` and `ruff check` are clean.
  - New owners get a row in `docs/ARCHITECTURE.md`; `tests/test_architecture_map.py` checks it.
- **Commits.** Use `feat(narration): ...`, `feat(chain): ...` or `fix(chain): #1308 - ...`, always with a scope. Never a bare `fix: #N`. End every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Defaults.**
  - `shot` and `tier` defaults on `chained-segments` do not change.
  - The new `narrated` preset ships only on the bench verdict (Task 12).
  - Every knob stays reachable. A renamed key is labelled `breaking-change` in the close-out.
- **Checkpoints.** Don evaluates at CP0-CP4 (spec *Bench budget*). Each is a short report in chat, with gallery URLs, per-segment scores and a recommendation. Nothing past a checkpoint starts until he answers.

## Review Focus

- **A narration whose last word ends mid-segment.**
  - Expected: the last segment is padded to the grid, the held slice is zero-padded past the track's end (as `slice_samples` does), and the take is trimmed back to the track's length.
  - Test: `test_take_ends_at_the_narration_end`, Task 5.
- **A word longer than the slack around a planned boundary**, for example a boundary that can only fall inside "Tranquility".
  - Expected: `narration_lines` reports it in `mid_word_boundaries` with the word and the segment; it does not fail.
  - Expected: `retime_narration` never stretches a gap past `boundary_slack_seconds`.
  - Test: `test_boundary_inside_a_long_word_is_reported_not_forced`, Task 2.
- **A narration with no detectable words**, such as music or silence.
  - Expected: both tasks refuse with a message naming the input, not an empty layout.
  - Test: `test_no_words_is_refused`, Task 2.
- **`stop_after` larger than the plan, and resume of a take that was not stopped.**
  - Expected: `stop_after` ≥ plan length renders the whole plan and logs it.
  - Expected: resuming a complete take still refuses with today's "nothing is left to render".
  - Test: `test_stop_after_past_the_plan_renders_it_all`, Task 6.
- **A held-audio chain whose guide also carried audio.**
  - Expected: the overlap is held once, in the segment's own slice. No guide audio rows are laid in.
  - Test: `test_held_chain_guide_carries_no_audio`, Task 5.

---

## Phase 0: #1308 (b), strengthened

### Task 1: A passed segment missing its opening words re-rolls, then reports

**Files:**
- Modify: `dw/pipeline_processors/chain_check.py` (`_opens_mid_line` :478, `check_words` :515-597, docstring :527 and :538-540)
- Modify: `dw/pipeline_processors/chain.py` (`_checked_attempt` :517-584, `_record_check` :638-692)
- Modify: `docs/WORKFLOW_GUIDE.md` :2857-2869, `plugins/dw/skills/minimax-h3/references/dialogue.md` :72-92, `plugins/dw/skills/long-form/references/run-and-repair.md` :90-93, `docs/ARCHITECTURE.md` :83
- Test: `tests/test_chain_words.py`, `tests/test_chain_reroll.py`

**Interfaces:**
- Produces: a words finding `{"rule": "line_opening_missing", "severity": "warn", "at": {"seconds": float, "word": str}, "value": <index of the line's first aligned word in the heard list>, "threshold": 0, "missing": [str], "says": str}` on a *passed* segment.
- Produces: the `words` dict gains `"first_word_start": float | None`, the start of the first kept chunk.
- Produces: `CheckResult.rerolls` is true when words passed but carries `line_opening_missing`. `_record_check` emits `chain_check_failed` with `kind`, `segment` and `findings` when every attempt carried it.

- [ ] **Step 1: Write failing tests** in `tests/test_chain_words.py`:
  - `test_passed_segment_missing_its_opening_words_is_reported`. Line "In order to form a more perfect union"; heard `words("Its to form a more perfect union", start=0.0)`; minimum 0.8. Assert `passed is True`, one finding with rule `line_opening_missing`, `missing == ["in", "order"]`, `answer["first_word_start"] == 0.0`.
  - `test_whole_line_has_no_opening_finding`. Heard is the whole line. Assert no `line_opening_missing`.
  - `test_opening_compared_after_number_folding`. Line "Amendment 1. Congress shall"; heard "Amendment one Congress shall". No finding.

  In `tests/test_chain_reroll.py`, using `FakeWords`:
  - `test_opening_missing_rerolls_while_retries_remain`. Attempt 1 returns `line_opening_missing`, attempt 2 a clean pass, `retries=1`. Assert 2 attempts and attempt 2 kept.
  - `test_opening_missing_on_every_attempt_is_reported`. Assert one `chain_check_failed` warning with `every_attempt_failed` true and the finding named in the text.
- [ ] **Step 2: Run them.** `venv/bin/python -m pytest -n0 tests/test_chain_words.py tests/test_chain_reroll.py -q`. Expected: the 5 new tests FAIL.
- [ ] **Step 3: Implement.**
  - Replace `_opens_mid_line`'s first-chunk compare with an alignment: fold heard words as `_seam_head` does (:414-428), `script_check.align([expected], heard)`, and take the expected words before the first aligned one as `missing`.
  - Fire when `missing` is non-empty and the segment has a previous line (carried segment) or `head_seconds > 0`.
  - Keep `line_clipped_at_head` for the failed path unchanged.
  - In `with_words`, a passed words dict carrying `line_opening_missing` adds `"words_opening"` to `failed` without setting `passed=False`. Do not add it to `WARN_ONLY`, so it re-rolls.
  - Fix the docstring sentence at :538-540 to say which findings fail the segment and which re-roll without failing.
- [ ] **Step 4: Run tests.** Same command; expected PASS. Then `venv/bin/python -m pytest tests/test_chain*.py -q`, all pass.
- [ ] **Step 5: Docs.**
  - WORKFLOW_GUIDE :2857-2869: the finding, the re-roll and the warning.
  - dialogue.md: keep the cue paragraph and add one sentence: "a segment that still opens past its first words re-rolls once; for a long monologue use a narrated take".
  - run-and-repair.md: a bullet for `line_opening_missing`.
  - ARCHITECTURE row :83: name `_opens_mid_line` and the new tests.
  - Run `venv/bin/python -m pytest -n0 tests/test_plugin_skills.py tests/test_architecture_map.py tests/test_guide_pointers.py -q`; expected PASS.
- [ ] **Step 6: Commit.** `fix(chain): #1308 - a passed segment missing its opening words re-rolls, then reports`

---

## Phase 1: Narration first (CP0)

### Task 2: `retime_narration` and `narration_lines` tasks

**Files:**
- Create: `dw/tasks/narration_retime.py` (both tasks and shared helpers; raise `"modules"` to 262)
- Modify: `dw/tasks/task.py` (import line), `docs/TASKS.md` (new section `## Retiming a Narration` after `## Checking a Narration`), `docs/ARCHITECTURE.md` (row "Narration retime and segment layout")
- Test: `tests/test_narration_retime.py`

**Interfaces:**
- Consumes:
  - `transcribe_audio(audio, device, sample_rate, timestamps="word", model_name=...)`, `dw/tasks/audio_transcription.py:76`, through a module-level seam `_word_chunks(waveform, rate, device, model_name) -> list[dict]` that tests monkeypatch.
  - `script_check.align`; `script_lines.normalize_words` / `fold_numbers`.
  - `waveform_and_rate`, `as_track` (`audio_utils.py:730/:86`); `crossfade_concat` (`joins.py:475`).
  - `find_loop_bed` (`loop_bed.py:774`), `loop_audio` (`audio_utils.py:589`).
  - `plan_segments(total_frames, segment_frames, trim_frames, frame_snap)` (`chain_config.py:550`). If importing it from a task creates a cycle, move the pure function to `dw/chain_plan.py` and re-export it from `chain_config`.
- Produces:
  - `retime_narration(audio, text=None, comma_seconds=0.3, sentence_seconds=0.6, section_seconds=1.0, min_gap_seconds=0.12, segment_frames=None, guide_frames=0, fps=24, boundary_slack_seconds=0.25, model_name=WHISPER_DEFAULT_MODEL, sample_rate=None, device="cpu") -> {"audio": AudioTrack, "report": JsonRecord}`.
    - `report` keys: `words` `[{word, start, end}]` (retimed times), `pauses` `[{after, class, before_seconds, after_seconds}]`, `duration_seconds`, `boundaries` `[{segment, seconds, in_gap: bool}]` (present when `segment_frames` is set), `mid_word_boundaries` `[{segment, word}]`.
  - `narration_lines(audio, segment_frames, guide_frames=0, fps=24, text=None, prompt_template=None, fill=None, model_name=..., sample_rate=None, device="cpu") -> dict` (JSON).
    - Keys: `lines` `[str]` (one per planned segment: the words whose midpoint falls in that segment's kept span), `prompts` `[str]` (only when `prompt_template` is given: the template with `{line}` and every `fill` key substituted), `segments` `[{index, start_seconds, end_seconds, line}]`, `mid_word_boundaries`, `total_frames`.
  - Registered commands `retime_narration` (artifact; `audio/wav`, report saved as JSON beside it) and `narration_lines` (`returns="json"`). Domains: the seconds arguments `NON_NEGATIVE`, `fps` `POSITIVE`; `segment_frames` and `guide_frames` whole numbers.
  - Pause class rule: a gap after a word ending `,;:` → comma; `.?!` → sentence; a paragraph break in `text` (blank line) or a line starting "Amendment" → section. When `text` is given, punctuation comes from the script through `align`, else from Whisper's chunk text. Gaps under `min_gap_seconds` are left alone.
  - Boundary rule: the kept-span ends from `plan_segments` (segment 1 ends at `segment_frames`, then every `segment_frames - guide_frames`) are boundary times. Each gap nearest a boundary is lengthened or shortened by at most `boundary_slack_seconds` so the boundary falls inside it. If no gap is reachable, report it in `mid_word_boundaries`.
  - Fill: inserted silence is the source's quietest loopable bed (`find_loop_bed`, then `loop_audio`), equal-power faded. The result is never digital zero.

- [ ] **Step 1: Write failing tests** with a synthetic voiced/silent track (220 Hz bursts as "words", like `voiced()` in `test_chain_words.py`) and a monkeypatched `_word_chunks`:
  - `test_comma_and_sentence_gaps_take_their_class_lengths`. Gaps 1.4 s after "Union," and 2.0 s after "America.". Retimed gaps are 0.3 ± 0.02 and 0.6 ± 0.02 s.
  - `test_gaps_shorter_than_min_gap_are_untouched`.
  - `test_section_break_from_a_blank_line_in_text`.
  - `test_inserted_gap_is_room_tone_not_zero`. The RMS of an inserted span is above -90 dBFS and below the voiced level.
  - `test_boundaries_land_in_gaps_on_the_chain_grid`. `segment_frames=192`, `guide_frames=22`, fps 24. Every reported boundary has `in_gap` true, and each boundary second equals the `plan_segments` kept-span end within one frame.
  - `test_boundary_inside_a_long_word_is_reported_not_forced` (Review Focus).
  - `test_no_words_is_refused` (Review Focus). `ValueError` naming `retime_narration`.
  - `test_lines_split_words_by_kept_span_midpoint`. `narration_lines` on a track with known chunk times gives the expected `lines` list and `len(lines) == len(plan)`.
  - `test_prompts_fill_the_template`. `prompt_template="A {scene}. <d>{line}</d>"`, `fill={"scene": "park"}`. `prompts[i] == f"A park. <d>{lines[i]}</d>"`.
  - `test_fill_placeholder_missing_is_refused`. The template names `{speaker}`, which `fill` lacks: `ValueError` naming the placeholder.
  - `test_registry_rules`. Both commands are listed by `list_tasks`, and `task_argument_errors` refuses `fps: 0`.
- [ ] **Step 2: Run them; expect FAIL** (module missing). `venv/bin/python -m pytest -n0 tests/test_narration_retime.py -q`
- [ ] **Step 3: Implement** in `dw/tasks/narration_retime.py`:
  - `word_map(waveform, rate, text, device, model_name) -> list[dict]`, with `class` on each gap.
  - `kept_span_ends(total_frames, segment_frames, guide_frames, fps) -> list[float]`.
  - The two task functions and a `_register()` called at module bottom (the `text_takes.py` pattern).
  - Keep each function under 60 lines; the ratchet counts.
- [ ] **Step 4: Run tests; expect PASS.** Then `venv/bin/python scripts/arch_metrics.py --check docs/stabilization/baseline.json` after raising `modules` to 262.
- [ ] **Step 5: Docs.**
  - TASKS.md section: argument tables for both commands, the report and lines shapes, what is refused.
  - The ARCHITECTURE row, naming `tests/test_narration_retime.py::test_boundaries_land_in_gaps_on_the_chain_grid`.
- [ ] **Step 6: Commit.** `feat(narration): retime_narration and narration_lines - pauses by punctuation class, boundaries on the chain grid`

### Task 3: Chain `prompts` and `lines` from one step's list

**Files:**
- Modify: `dw/previous_results.py` (`resolve_chain_prompts` :200), `dw/pipeline_processors/chain_config.py` (where `lines` is read), `dw/pipeline_processors/chain_rules.py` (`chain_lines_problem` :343)
- Modify: `docs/WORKFLOW_GUIDE.md` (`### Chained Video Generation`, the `prompts`/`lines` paragraph :2718)
- Test: `tests/test_chain_prompts_reference.py`

**Interfaces:**
- Produces:
  - A chain block's `prompts` or `lines` may be one string `previous_result:<step>.<key>` naming a JSON list of strings. It resolves to that list before the plan is made.
  - A list entry that is itself a reference keeps today's behaviour.
  - Validate accepts the unresolved string; at run, a non-list or a non-string entry is refused with the step and key named.

- [ ] **Step 1: Failing tests.**
  - `test_prompts_resolve_from_one_list_reference`. A fake previous result `{"lines": {"prompts": ["a", "b", "c"]}}`; after resolution `chain["prompts"] == ["a", "b", "c"]`.
  - `test_lines_resolve_from_one_list_reference`.
  - `test_list_reference_to_a_non_list_is_refused`.
  - `test_validate_accepts_an_unresolved_list_reference`. `validate_workflow` on a minimal t2va chain using `previous_result:lines.prompts` has no error.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement** in `resolve_chain_prompts`; add the same resolution for `lines` (rename the function to `resolve_chain_lists` and keep the old name as a thin alias only if other callers exist). Relax `chain_lines_problem` for an unresolved reference.
- [ ] **Step 4: Run; expect PASS,** plus `venv/bin/python -m pytest tests/test_chain*.py tests/test_validation*.py -q`.
- [ ] **Step 5: Commit.** `feat(chain): prompts and lines may be one previous_result list`

### Task 4: `narration-retime` template, then CP0

**Files:**
- Create: `workflows/templates/narration-retime.json`
- Modify: `tests/test_catalog_structure.py` (`UTILITIES` :136; `COMPACT_BUDGET` raise if measured over, with its log line), `plugins/dw/skills/long-form/references/audio.md` (a "Retime a narration" recipe)

**Interfaces:**
- Produces: template `templates/narration-retime`.
  - Variables: `narration` (audio), `script` (text, may be null), `comma_seconds`, `sentence_seconds`, `section_seconds`, `num_frames` (default 192), `guide_frames` (default 22).
  - Steps:
    - `retime` (`retime_narration` with `segment_frames: variable:num_frames`), saving `final` audio and its report;
    - `check` (`check_narration` on `previous_result:retime.audio` against `script`, when `script` is given).

- [ ] **Step 1:** Write the template. Description: what it makes, that `num_frames`/`guide_frames` must equal the take's, and that its report's `mid_word_boundaries` should be empty before a take is rendered.
- [ ] **Step 2:** Run `venv/bin/python -m pytest -n0 tests/test_catalog_structure.py tests/test_examples.py tests/test_template_subfolders.py tests/test_plugin_skills.py -q`. Fix any budget overrun with a logged raise.
- [ ] **Step 3: Commit.** `feat(narration): narration-retime template`
- [ ] **Step 4: Prepare CP0** (needs lem on this commit; this is deploy gate 0). Merge to `develop`, push, and ask Don to deploy (no flag yet).
  - Source: `ConstitutionWalkFinal/20261108-202744-78f95d6f/final/ConstitutionWalkFinal-film.3-0.0.mp4` in workspace `constitution-walk-2`. Extract its audio as an asset in a new workspace `constitution-walk-3`.
  - The script: the 48 lines from the takes' prompts, Preamble and each amendment separated by blank lines.
  - Run `templates/narration-retime` with `num_frames: 192`, `guide_frames: 22`.
- [ ] **Step 5: CP0 report to Don:**
  - the retimed track (gallery URL), its duration, and the pause table by class;
  - the `check_narration` result;
  - `mid_word_boundaries`;
  - the end of Amendment 3 in seconds, and the segment count to it (the probe's `stop_after`).

  Don listens and tunes the class lengths. Rerun until he approves the track. The approved track is the bench narration.

---

## Phase 2: The held-audio take (control arm)

### Task 5: Guide chains held to a narration

**Files:**
- Modify: `dw/pipeline_processors/chain_config.py` (`_read_guide` :263: no guide audio carry when `holds_audio`), `dw/pipeline_processors/guides.py` (new `held_audio_warnings`, registered in `WARNING_CHECKS` validation.py:686)
- Test: `tests/test_chain_held_guide.py`

**Interfaces:**
- Produces:
  - Under `continuity: guide` with `match_audio` and a step `hold_audio`, the guide carries video only (`carry_audio` forced false, logged once). Each segment's `hold_audio` is its slice from `audio_start_frame` (already the case, `chain.py:1030`).
  - Validate warnings:
    1. A chain step with `hold_audio` and no `match_audio`: "every segment holds the whole track from its start; set match_audio: true to give each segment its slice".
    2. A held-audio chain whose `num_frames % 3 != 0`: "H3's audio latents run at 40 a second; <n> frames is off the 3-frame audio grid; the nearest on-grid lengths are 141, 192, 243, 294, 345".
- [ ] **Step 1: Failing tests** (fakes as in `TestMatchAudioHold`, tests/test_chain.py:440-570, but `continuity: guide`):
  - `test_held_chain_guide_carries_no_audio` (Review Focus). The segment 2 call's guide has no `"audio"` key and no `audio_latents`.
  - `test_held_guide_slices_start_at_the_guide_span`. Slice starts are `[0, frames_to_samples(192-22), ...]`.
  - `test_take_ends_at_the_narration_end` (Review Focus). Final frames == round(track seconds × 24), and the final audio equals the source track.
  - `test_hold_without_match_audio_warns`.
  - `test_off_grid_held_chain_warns`. 209 warns, 192 does not.
- [ ] **Step 2: Run; expect FAIL.** `venv/bin/python -m pytest -n0 tests/test_chain_held_guide.py -q`
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run; expect PASS,** plus `tests/test_chain*.py tests/test_validation*.py`.
- [ ] **Step 5: Commit.** `feat(chain): a guide chain held to a narration carries no guide audio; held-audio validate warnings`

### Task 6: Probe and continue (`stop_after`, resume from a probe)

**Files:**
- Modify: `dw/pipeline_processors/chain_config.py` (read `stop_after`), `dw/pipeline_processors/chain.py` (`run_chain` loop :171 stops after k; `_stitched` :789 trims the source track to the rendered frames), `dw/pipeline_processors/chain_rules.py` (`stop_after` problem; `chain_resume_problem` :730 accepts latent carry, see Task 9)
- Test: `tests/test_chain_probe.py`

**Interfaces:**
- Produces:
  - Chain key `stop_after: int | null`, a whole number ≥ 1. The plan is unchanged; only segments `0..k-1` render. The take is muxed with the source track trimmed to the rendered frames, and the run logs `chain_stopped_after` with k and the plan length.
  - The manifest's chain record carries `"planned_segments": len(plan)`.
  - `resume_from` naming a stopped take's last segment file continues at segment k+1 under the same plan, through the existing `chain_resume.load` path.

- [ ] **Step 1: Failing tests:**
  - `test_stop_after_renders_k_and_trims_the_track`. Plan 5, `stop_after` 2: 2 pipeline calls, final audio length == rendered frames' duration.
  - `test_stop_after_past_the_plan_renders_it_all` (Review Focus).
  - `test_resume_continues_a_stopped_probe`. Run with `stop_after` 2 and keep segments, then the same chain with `resume_from` set to the probe's segment-001 file. Calls are made only for segments 3..5, the final frame count equals a full run's, and the final audio equals the full source track.
  - `test_resume_of_a_complete_take_still_refuses`.
  - `test_stop_after_zero_is_refused`.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run; expect PASS,** plus `tests/test_chain_resume.py tests/test_chain*.py`.
- [ ] **Step 5: Docs.** WORKFLOW_GUIDE `### Chained Video Generation`: a "Probe, then continue" paragraph (stop_after, then resume_from the probe's last segment file, and what is re-encoded at that seam).
- [ ] **Step 6: Commit.** `feat(chain): stop_after renders a probe; resume_from continues it`

### Task 7: Identity and lip-sync measures

**Files:**
- Create: `dw/tasks/lip_sync.py` (raise `"modules"` to 263)
- Modify: `dw/tasks/face_drift.py` (public `face_similarities(frames, reference, samples, device=None) -> list[float]` from its existing scoring), `dw/pipeline_processors/chain_check.py` (measures `face_drop`, `lip_sync`), `dw/pipeline_processors/chain_rules.py` (`CHAIN_CHECK_FIELDS` :216: replace reserved `face` with `face_drop_max`, `lip_sync_min`; `chain_check_finding` :309), `dw/pipeline_processors/chain.py` (`_checked_attempt`: fold the new measures with segment 1's frame 0 as the face reference and the segment's held slice as the sync audio), `dw/tasks/task.py`, `docs/TASKS.md` (`### measure_lip_sync` beside `### measure_sync`), `docs/WORKFLOW_GUIDE.md` (`#### Checking each segment` table :2816), `docs/ARCHITECTURE.md`
- Test: `tests/test_lip_sync.py`, `tests/test_chain_check_face_sync.py`

**Interfaces:**
- Produces:
  - `lip_sync_score(frames, waveform, sample_rate, fps, max_lag_seconds=0.4) -> {"correlation": float | None, "lag_seconds": float | None, "frames_with_face": int}`.
    - The mouth box comes from YuNet's two mouth-corner landmarks, extended down by the corner distance.
    - The signal is the mean absolute frame difference in that box, correlated (Pearson) against the audio RMS envelope at the frame rate, at the best lag within ±`max_lag_seconds`.
  - Task `measure_lip_sync(video, audio=None, max_lag_seconds=0.4)` (JSON). The audio defaults to the video's own track.
  - Chain check measures:
    - `face_drop`: the median face similarity of segment 1 minus this segment's, both against segment 1's frame 0.
    - `lip_sync`: the correlation.
  - Both are added to `MEASURES` and to `WARN_ONLY`. Ceilings `face_drop_max` and `lip_sync_min` default `None`: measure and record, never fail, until the bench sets them. Values are shown in `shot["check"]` and in `chain_check_failed` when a ceiling is set.

- [ ] **Step 1: Failing tests.**
  - `test_lip_sync.py`, synthetic: frames with a mouth box whose brightness follows a known envelope, plus a stubbed landmark seam `_mouth_box`.
    - `test_synced_mouth_scores_high_at_zero_lag`: correlation > 0.8, |lag| < 1 frame.
    - `test_shifted_audio_scores_at_its_shift`: a 0.3 s shift gives lag ≈ 0.3 s.
    - `test_no_face_returns_none`.
  - `test_chain_check_face_sync.py`:
    - `test_face_drop_is_measured_and_warn_only`;
    - `test_lip_sync_is_measured_and_warn_only`;
    - `test_face_drop_max_sets_a_failing_ceiling_without_reroll`;
    - `test_check_refuses_unknown_face_key`.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run; expect PASS,** plus `tests/test_face_drift*.py tests/test_chain*.py tests/test_task_registry_rules.py`.
- [ ] **Step 5: Calibrate the sync measure (CPU, no GPU), and record it.** Run `measure_lip_sync` locally (the venv) over the constitution-walk-2 kept segment files in `/private/tmp/claude-501/-Users-don-src-dkackman-diffusers-workflow/6b6bf187-e737-46c5-a185-518466476aa3/scratchpad/cw2/ws/` (`rsync` them again if the scratchpad is gone):
  - Good: each segment with its own audio.
  - Bad: each segment with its audio shifted 0.5 s, and with the next segment's audio.

  Write the distributions to `bench/records/narrated.md` § *Sync calibration*.
  - **Rule:** if at least 90% of good segments score above every bad one at the same segment, the measure stays as built.
  - **Otherwise:** remove `lip_sync` from the chain check (keep the task) and say in the skill to judge sync by eye.
- [ ] **Step 6: Commit.** `feat(chain): face_drop and lip_sync segment measures, warn-only; measure_lip_sync task`

### Task 8: `templates/minimax/narrated-take`

**Files:**
- Create: `workflows/templates/minimax/narrated-take.json`
- Modify: `tests/test_catalog_structure.py` (budgets, with the log line)

**Interfaces:**
- Consumes: Tasks 2, 3, 5, 6, 7; the chained-segments pipeline block (copy its `configuration`, `from_pretrained_arguments` `workflow: fl2va`, scheduler shifts and attention variables).
- Produces: template `templates/minimax/narrated-take`, shape `sequence`, traits derived (`has-audio`, `chained`, `image-conditioned`).
  - Variables:
    - `image`, `narration` (the retimed track), `script`;
    - `scene`, `speaker`, `soundscape`;
    - `prompt_template` (default: the H3 Context-IR form of the constitution-walk-2 prompt with `{scene}`, `{speaker}`, `{line}`, `{soundscape}` placeholders, and no cue or breath clause);
    - `tier` (presets `standard`, `best` copied from chained-segments; default `best`);
    - `shot` (presets: `dialogue` = `{continuity: guide}`; `narrated` added in Task 12);
    - `num_frames` 192, `guide_frames` 22;
    - `stop_after` (default null; the description says to set it for the probe);
    - `resume_from` null, `keep_segments` true, `seed`.
  - Steps:
    - `lines` (`narration_lines`: `audio: variable:narration`, `segment_frames: variable:num_frames`, `guide_frames`, `prompt_template`, `fill: {scene, speaker, soundscape}`);
    - `take` (chain: `continuity: guide`, `match_audio: true`, `prompts: "previous_result:lines.prompts"`, `lines: "previous_result:lines.lines"`, `stop_after`, `resume_from`, `check: {"jump_ratio_max": ..., "face_drop_max": null, "lip_sync_min": null}` with the chain-check defaults, `retries: 1`; step `hold_audio: variable:narration`, `image: variable:image`).
  - Description, in under 120 characters for the summary sentence: the shape; that the narration comes from `templates/narration-retime`; the probe/continue procedure; and that `num_frames`/`guide_frames` must match the retime's.
  - No `cost` row until Task 11 measures one.

- [ ] **Step 1:** Write the template.
- [ ] **Step 2:** `venv/bin/python -m pytest -n0 tests/test_catalog_structure.py tests/test_examples.py tests/test_template_subfolders.py tests/test_template_defaults.py -q` passes, budget raise logged if needed.
- [ ] **Step 3:** A validate test, `tests/test_narrated_take_template.py::test_validates_with_a_probe` (`stop_after: 12`), asserts no errors and that the plan's segment count comes from the narration.
- [ ] **Step 4: Commit.** `feat(minimax): narrated-take template`

---

## Phase 3: The drift candidate (behind the flag)

### Task 9: Latent carry under a held narration, latent keystone, `anchor_every`

**Files:**
- Modify: `dw/pipeline_processors/chain_rules.py` (`chain_latent_carry_problem` :427: drop the `match_audio` and `hold_audio` refusals; `chain_resume_problem` :740: drop the latent-carry refusal; keystone slot `"latent"`; `anchor_every`), `dw/pipeline_processors/chain.py` (`_with_latent_tail` :892 passes no audio latents when `config.holds_audio`; segment 1's frame-0 latent kept; `anchor_every` gating of keystone and capstone in `_segment_arguments` :907-943), `dw/pipeline_processors/chain_continuity.py` (`inject` carries `keystone_latents`), `dw/pipeline_processors/h3_guide_steps.py` (layout writes `keystone_latents` over the guide's latent-0 rows; a restore step puts the carried latent 0 back after the denoise loop, before decode), `dw/pipeline_processors/chain_resume.py` (`ResumeState` re-encodes segment 1's decoded frame 0 for the latent keystone)
- Test: `tests/test_h3_latent_carry.py` (update :192-198, :234), `tests/test_chain_latent_keystone.py`, `tests/test_chain_anchor_every.py`

**Interfaces:**
- Produces:
  - `experimental_latent_carry: true` combines with `match_audio` and `hold_audio` and carries video latents only.
  - `keystone_slot: "latent"` needs `keystone: segment_1`, guide continuity and the flag. Segment 1's frame-0 latent is locked into each later segment's guide latent 0 for the whole denoise and restored before decode.
  - `anchor_every: int` (≥ 1, default 1) applies the keystone and `experimental_capstone` on segments with 0-based index `i ≥ 1` and `i % anchor_every == 0`. It is refused without a keystone or capstone.
  - `resume_from` combines with latent carry: the first resumed seam is a decoded, re-encoded guide (logged as `chain_resumed`, with that fact). The keystone latent is re-encoded from the decoded file.

- [ ] **Step 1: Failing tests.**
  - `test_latent_carry_with_held_audio_carries_video_only` (`LatentOutputs` fakes, test_h3_latent_carry.py:70).
  - Update the two refusal tests to expect acceptance.
  - `test_latent_keystone_rows_written_and_restored`: a layout-step unit test with fake latents. During the loop row 0 equals the keystone; after the restore step it equals the carry's.
  - `test_latent_keystone_needs_keystone_segment_1`.
  - `test_anchor_every_3_anchors_segments_3_6_9`: indices 3, 6, 9 of 10 get the keystone and capstone guides, the others none.
  - `test_anchor_every_without_an_anchor_is_refused`.
  - `test_resume_with_latent_carry_reencodes_the_first_seam`.
- [ ] **Step 2: Run; expect FAIL.** `venv/bin/python -m pytest -n0 tests/test_h3_latent_carry.py tests/test_chain_latent_keystone.py tests/test_chain_anchor_every.py -q`
- [ ] **Step 3: Implement.** Read `h3_guide_steps.py` :283-530 first (`carried_guide_latents`, `_hold_guide_audio`, `GATED_GUIDE_ROWS`). The restore step is a modular block appended after the denoise loop (`h3_windows.py`'s `insert_refine_windows` :429 shows how a block is inserted), inserted only when a guide carries `keystone_latents`.
- [ ] **Step 4: Run; expect PASS,** plus `tests/test_h3*.py tests/test_chain*.py`.
- [ ] **Step 5: Bench arms.** Create `bench/arms/narrated-control.json`, `narrated-candidate.json`, `narrated-static-long.json` in the existing arms format (`tests/test_bench_workflows.py:65`). Each arm runs `templates/minimax/narrated-take` on the CP0 narration with `stop_after` = the CP0 segment count to the end of Amendment 3:
  - control: `shot: dialogue`;
  - candidate: `experimental_latent_carry: true`, `keystone: segment_1`, `keystone_slot: latent`, `experimental_capstone: {"enter_below_sigma": 0.7}`, `anchor_every: 3`;
  - static-long: `keystone: segment_1`, `keystone_slot: last`.

  Seeds 1 and 2.
- [ ] **Step 6: Commit.** `feat(chain): latent carry under a held narration, latent keystone, anchor_every (experimental)`

### Task 10: Deploy gate 1

- [ ] **Step 1:** Full suite, ruff, ratchet: `venv/bin/python -m pytest tests/ -q && ruff check && venv/bin/python scripts/arch_metrics.py --check docs/stabilization/baseline.json`. All pass.
- [ ] **Step 2:** Merge `feat/narrated-long-take` into `develop`, push. Ask Don to deploy and set `DW_EXPERIMENTAL_REFINE_WINDOWS=1` in lem's `.env`.
- [ ] **Step 3:** Smoke over MCP: `validate_workflow` each arm, quote `plan.estimate`, confirm no errors and the expected held-audio warnings (none at 192 frames).

---

## Phase 4: Bench (CP1, CP2)

### Task 11: CP1 and CP2 runs

- [ ] **Step 1: CP1 runs.** Control seed 1, then candidate seed 1, one at a time (int8, one per host). Watch the kept segment files as they land. At segment 4 (about 30 s), pull `get_output_frames(at=...)` and `compare_outputs(texture_bands=true)` against segment 1. If an arm is already clearly worse than the control, cancel it and note why.
- [ ] **Step 2: CP1 measures** per take:
  - per-segment `face_drop` and `lip_sync` from the job's shots;
  - `assess_output(probe="analyze_seams")`;
  - `compare_outputs(texture_bands=true)` against segment 1;
  - `analyze_face_drift` against the cast still;
  - frame saturation per segment (the HSV method from the assessment);
  - wall minutes.

  Write them to `bench/records/narrated.md` (title format `# Narrated take: ...`, setup lines, an arm table and a *Labels* section, as in `words.md`).
- [ ] **Step 3: CP1 report to Don:** both probes' gallery URLs side by side, the table, the face frames at 0/30/60/85 s, and a recommendation. Wait for his read.
- [ ] **Step 4: CP2 runs.** static-long seed 1, then seed 2 of all three arms. Same measures and record.
- [ ] **Step 5: CP2 report:** the three arms ranked and the proposed winner with its numbers.
  - **If the candidate loses:** propose the second-line arms (no capstone; stat anchor `anchor: segment_1, anchor_strength: 0.15`; per-segment keystones from a still progression) at 90 s seed 1, with a checkpoint after each pair. That plan comes back to Don before any build.
  - **If nothing beats the control:** take the spec's *If the bench fails* route.
- [ ] **Step 6: Record the measured cost.** The winner's wall minutes per segment and the template's `cost` row, measured at its defaults.

---

### Task 11a: `keystone_images`, the moving keystone (Amendment 1)

The spec is the proposal's *Amendment 1*. A chain key `keystone_images` holds one image
per segment. Under `keystone: "segment_1"` with `keystone_slot: "last"`, each later
segment *k* (1-based) on the `anchor_every` cadence takes entry *k-1* as its fl2va
`last_image`, in place of segment 1's frame 0. Segment 1 takes entry 0 as its own
`last_image` only when the step has no `last_image`.

**Files:**
- `dw/pipeline_processors/chain_rules.py`: the refusals, and `place_keystone` taking a
  per-segment frame.
- `dw/pipeline_processors/chain_config.py`: reads the list.
- `dw/pipeline_processors/chain.py`: per-segment placement, segment 1 included, and the
  job-log note.
- `dw/previous_results.py`: `resolve_chain_lists` also resolves `keystone_images`, same
  rules as `prompts`: a list of refs/paths, or one `previous_result:<step>.<key>`. An
  image-producing step (FLUX 3 Klein edit with `for_each`) hands on its list of images;
  the first artifact per entry.
- `dw/workflow_schema.json`: the chain key.
- `workflows/templates/minimax/narrated-take.json`: optional `keyframes` variable (list,
  default null; a null drops the key), wired to `keystone_images`. A non-null value
  also needs `keystone: segment_1` and `keystone_slot: last`. Add a `shot` preset,
  `walk_keyframed`, that sets both.
- `bench/workflows/narrated-take.json`: re-sync; the pin test enforces it.
- Docs: WORKFLOW_GUIDE *Chaining with a guide*, the `keystone_slot` row plus a
  `keystone_images` row.
- Tests: new `tests/test_chain_keystone_images.py`.

**Refusals,** each naming the key and the allowed values:
- the key without `keystone: segment_1` and `keystone_slot: last`;
- a list shorter than the planned segment count (the `match_audio` plan, `segments`, or
  `duration_seconds`), naming both counts. A probe (`stop_after`) still needs the full
  plan's length, because the probe is continued;
- an entry that is not an image reference;
- the key on a non-H3 or non-fl2va step;
- the key on a step that sets its own `last_image`.

**Behaviour pins:**
- `test_segment_k_takes_entry_k_as_last_image`: 4 segments, a fake step; the recorded
  `last_image` per segment is entries 0..3.
- `test_anchor_every_skips_off_cadence_segments`: with `anchor_every: 2`, segments at
  1-based positions 2 and 4 get a last image and position 3 gets none (keep the
  0-based `i % n == 0` rule as `config.anchors(i)` has it).
- `test_resume_keeps_entry_k_for_segment_k`: a resume after segment 2 places entries 2
  and 3 on segments 3 and 4.
- `test_previous_result_list_resolves`: one `previous_result:keys.images` naming a
  4-image list resolves in order.
- One test per refusal.
- The held-narration path: a `match_audio` plus `hold_audio` chain places last images
  with guide audio forced off as today. This is a regression pin, not new behaviour.

**Ratchet:** chain.py (1,095), plan.py (1,099) and validation.py (1,093) sit at the
1,100-line ceiling. New logic goes in `chain_rules.py`, or a new module if that file
nears the ceiling. `scripts/arch_metrics.py --check docs/stabilization/baseline.json`
must be no worse than base (oversize 2, complex 3).

Commit: `feat(chain): keystone_images - a per-segment last image (moving keystone)`.

### Task 11b: Keyframe series and review (Amendment 1)

This is controller work and needs no code. lem runs on cuda:0 by day.

1. **Avenue plate.** Draw one wide plate on FLUX 3 Klein: the s3 avenue and colonnade,
   empty, matching the speaker still's light and lens.
2. **Recession.** Derive 12 plates for the probe. The camera tracks back, so the
   colonnade recedes: plate *k* is the wide plate centre-cropped to a field that widens
   by a constant ratio per segment. Fit the ratio to an unhurried walk of about 1 m/s
   over 7.1 s per segment.
3. **Composite.** For each plate, a FLUX 3 Klein multi-reference edit: picture 1 is the
   speaker still (identity, wardrobe, framing), picture 2 is plate *k*. Use one seed and
   one prompt for the whole series.
4. **Measure.** `analyze_face_drift` of each keyframe against the speaker still, plus a
   contact sheet. Keyframes outside the band of the static-long take's segments get
   redrawn.
5. **Review.** Don approves the series before Task 11c.

Record the results in `bench/records/narrated.md`, *Moving keystone: keyframes*.

### Task 11c: Moving-keystone bench (Amendment 1)

- **Arm:** `bench/arms/narrated-moving-keystone.json`. The control's arguments plus
  `keystone: segment_1`, `keystone_slot: last`, and `keystone_images` = the 12 approved
  keyframes as `asset:` refs. Seed 1, `stop_after: 12`. The full 43-entry list is drawn
  only at CP3.
- **Measures:** CP1's set (`face_drop` per segment, saturation, `analyze_seams`, eye at
  0/30/40/60/85 s), plus a walk check: the backdrop advances and there is no pull-back.
  Don judges it against `static_long` (face) and the candidate (walk).
- **Then:** seed 2. CP3 and CP4 take the winner, extending the keyframe series to all
  43 segments.

## Phase 5: Promote and finish the agent layer

### Task 12: The `narrated` preset, promoted

**Files:**
- Modify: `dw/pipeline_processors/chain_rules.py`, `h3_rules.py`, `h3_guides.py`, `components.py` :816-819. Rename the winning experimental keys to stable ones (`latent_carry`, `capstone`). The gated-row loop denoiser is installed whenever a chain uses a gated guide, independent of the flag; `DW_EXPERIMENTAL_REFINE_WINDOWS` stays only for the windowed refine.
- Modify: `workflows/templates/minimax/narrated-take.json` (preset `shot: narrated` with the winning bundle, made the default; `cost` row from Task 11; `check` ceilings `face_drop_max` / `lip_sync_min` from the bench where it set them)
- Modify: bench arms to the stable keys
- Test: the tests touched in Task 9, with the stable keys. `tests/test_h3_capstone.py`: `test_gated_denoiser_is_a_pass_through_without_gated_rows` (identical outputs with and without the wrapper on a fake transformer).

- [ ] **Step 1:** Failing tests for the stable keys and the pass-through.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement the rename and the install rule. Update every test and doc mentioning the experimental keys (`grep -rn "experimental_latent_carry\|experimental_capstone" dw tests docs plugins workflows bench`).
- [ ] **Step 4:** Run the full suite; expect PASS.
- [ ] **Step 5: Commit.** `feat(chain): narrated preset - <winning bundle>; latent carry and capstone leave the experimental flag (breaking-change: key rename)`

(If the bench winner was `static_long` or the control, this task only adds the preset and the cost row; no rename.)

### Task 13: Repair a narrated take

**Files:**
- Modify: `dw/tasks/chain_repair.py` (`chain_repair_cut` :363 returns `held_audio`: the master track's slice under the repair render's frames when the take is a match_audio take, else None; `repair_geometry` :87 allows the last segment with a head guide only), `workflows/templates/minimax/repair-segment.json` (the `tier` presets copied from chained-segments; the render step takes `hold_audio: previous_result:cut.held_audio`; the `segment` constraint allows N)
- Test: `tests/test_chain_repair.py`

**Interfaces:**
- Produces:
  - A narrated take repairs at its tier, holding its narration under the re-rendered span.
  - The splice keeps the frame count, so the master track stays in sync (`_spliced_track` already keeps `take.audio` for a matched take).
  - The last segment is repairable.

- [ ] **Step 1: Failing tests:**
  - `test_matched_take_repair_holds_the_master_slice`: slice start == `frames_to_samples(start_frame)`, length == render frames.
  - `test_last_segment_repair_has_no_tail_guide`.
  - `test_repair_keeps_the_take_frame_count`.
  - `test_canvas_mismatch_with_the_take_is_refused` (existing `_check_canvas`).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5: Commit.** `feat(chain): repair a narrated take - held master slice, take tier, last segment`

### Task 14: Skills, guide, attributions

**Files:**
- Modify: `plugins/dw/skills/long-form/SKILL.md`
  - §2: a "Narrated take" row.
  - §3: the 35 s rule gains its measured exception, one sentence: "A speaking subject in one place is a narrated take: `templates/narration-retime`, then `templates/minimax/narrated-take` with `stop_after` for a probe of about 90 s, then `resume_from` to continue (`bench/records/narrated.md`)."
  - §4: `face_drop`, `lip_sync`, `line_opening_missing`.
  - Stay under 12288 bytes.
- Modify: `plugins/dw/skills/long-form/references/run-and-repair.md` (the full procedure: narration → listen → probe → judge with which tools and which numbers → continue → repair), `references/levers.md` (the narrated preset and its verdict; latent carry, latent keystone, capstone and `anchor_every` verdicts)
- Modify: `plugins/dw/skills/minimax-h3/SKILL.md` (one line under *Longer than one clip*: a monologue longer than about 35 s is a narrated take, read `long-form`; under 430 bytes), `plugins/dw/skills/script-to-video` (route a long monologue to the shape)
- Modify: `docs/WORKFLOW_GUIDE.md` (new `### A narrated long take` under `## Pipeline Configuration`, after `### Chained Video Generation`: the procedure, the audio-grid rule, the probe and continue, what each measure means and its ceiling)
- Calls on the spec's warnings: the spec's "a cue clause in a held-audio prompt" and "a probe size that cannot judge drift" warnings are not built as validate checks (both are prompt or intent heuristics that would misfire). The guide section and the template description state both rules instead.
- Modify: `docs/ATTRIBUTIONS.md` (matlowai/h3-long-video-research, NerdNotes tests 16-18, geiru and H3MD concepts)
- Modify: `tests/test_plugin_skills.py` (pin the new rule's phrase and `bench/records/narrated.md` in long-form's citations, as `:1885` does)

- [ ] **Step 1:** Write the pins first and run `venv/bin/python -m pytest -n0 tests/test_plugin_skills.py tests/test_guide_pointers.py -q`; expect FAIL.
- [ ] **Step 2:** Write the text. Every pointer must resolve through `get_guide` (`tests/test_guide_pointers.py`).
- [ ] **Step 3:** Run the pins plus `tests/test_server_guides.py`; expect PASS.
- [ ] **Step 4: Commit.** `docs(long-form): the narrated take - skill rule, procedure, guide section, attributions`

### Task 15: Deploy gate 2

- [ ] **Step 1:** Full suite, ruff, ratchet; all pass.
- [ ] **Step 2:** Merge to `develop`, push. Ask Don to deploy and to remove `DW_EXPERIMENTAL_REFINE_WINDOWS` from lem's `.env` (unless Task 12 kept a piece behind it).
- [ ] **Step 3:** Smoke: `validate_workflow` on `templates/minimax/narrated-take` with defaults and the CP0 narration. The quoted estimate uses the new cost row, and the preset is `narrated`.

---

## Phase 6: The constitution walk (CP3, CP4), the drill, close-out

### Task 16: CP3 and CP4

- [ ] **Step 1: CP3.** Continue the winner's seed-1 probe with `resume_from` (its last segment file) and `stop_after` = the segment count to the end of Amendment 6.
  - Measures as in Task 11, plus the continued seam (`analyze_seams` on that seam, `get_output_frames(seams=true)`).
  - Report to Don.
- [ ] **Step 2: Repair, if needed.** Any segment Don or the measures flag is repaired with `templates/minimax/repair-segment` before continuing.
- [ ] **Step 3: CP4.** Continue to the end. Report the full take: duration, face and sync per segment, saturation curve, seams, and the comparison against the 2026-11-08 cut on the three defects.
- [ ] **Step 4:** `set_note` the keeper; add the CP3/CP4 rows to `bench/records/narrated.md`.

### Task 17: Cold drill

- [ ] **Step 1:** Make `~/testing/nt-drill-1`. Run a headless session the way `bench/records/drill.md:1-10` does (`claude -p --model sonnet --strict-mcp-config` with the dw MCP only, `--plugin-dir plugins/dw`).
  - Turn 1 (verbatim): "Here is a portrait: <asset name>. Make one continuous shot of her walking toward the camera reciting this, no cuts: <a 60-120 s public-domain monologue (the Gettysburg Address)>."
  - Turn 2: the scripted go-ahead after it quotes a cost.
- [ ] **Step 2: Judge the transcript.**
  - Did it read `long-form`, pick `narration-retime` then `narrated-take`, probe with `stop_after`, judge with the measures, continue with `resume_from`, and finish?
  - Every guess or wrong turn becomes a fix in the skill, guide, template description or a validate warning. Rerun once after the fixes.
- [ ] **Step 3:** Record it in `bench/records/drill.md` as a new ask.

### Task 18: Close-out

- [ ] **Step 1:** `git mv docs/proposals/narrated-long-take.md docs/proposals/complete/narrated-long-take-complete.md` and add a *Verdicts* section (each arm, each gap G1-G5 with what shipped).
- [ ] **Step 2: Issues:**
  - #1308: verify on lem against the #1300 repro with the cue (Task 1's behaviour: segment 3 re-rolls or reports), then close with the job ids.
  - #1225: close as built (the narrated preset) or note its verdict.
  - #1187: comment with the capstone's verdict. #1150: comment with the latent carry's.
  - File follow-ups only for measured gaps, as `owner:don` ideas.
- [ ] **Step 3:** Merge, push, update memory (`constitution-walk-2-assessment-2026-11-09.md` → the outcome).
