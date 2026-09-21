# Tier 1 Proposals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the four remaining Tier 1 items from `docs/proposals/todo.md` — a correctness fix to the H3 video-mux headroom warning, WAL mode on the jobs database, one catalog template exercising the already-shipped `select`/`judge` reducer, and a new composition-only plugin skill (`script-to-video`) — each independently landable.

**Architecture:** Four unrelated, independently-shippable changes bundled into one plan because they were ranked together. Task 1 touches `dw/result.py` (engine warning logic). Task 2 touches `dw/server/jobs.py` (one line, a SQLite pragma). Task 3 adds one new workflow JSON file plus a small structural test. Task 4 adds one new plugin skill file with no engine changes. Do these in any order; there is no dependency between tasks.

**Tech Stack:** Python 3, pytest, JSON workflow definitions, no new dependencies.

**Spec:**
- Task 1: `docs/proposals/h3-video-mux-headroom-warning-partial.md`
- Task 2: `docs/proposals/maintenance-screen.md` (Phase 0 only — the rest of that proposal is out of scope here)
- Task 3: `docs/proposals/score-and-select-partial.md` (engine already shipped; this task is section 3, "One catalog template")
- Task 4: `docs/proposals/script-to-video-agent-skill.md`

## Global Constraints

- Never use `eval()`, `exec()`, or `shell=True` (repo-wide security rule, `dw/security.py`).
- `cost` on a workflow is curated from a real measured run and is never fabricated (CLAUDE.md: "`cost` is curated, `observed` is derived... nothing writes one"). Task 3's template ships with **no** `cost` field until someone measures a real run on GPU hardware — do not invent a number.
- Every `workflows/templates/**` file with two or more saving steps marks each saving step's `result.subfolder` as `final` or `intermediate` (`tests/test_template_subfolders.py` pins this).
- A step whose task command returns a scalar (`judge` does — `dw/tasks/task.py` declares `returns="scalar"`) must carry **no** `result` block at all — `dw/scalar_result_validation.py` refuses one.
- Run the full suite with the local dev venv (`venv/bin/python -m pytest`), not a fresh install — torch and the rest of the test deps are already there. Do not attempt to run anything against a live GPU box from this environment; where a step needs a real GPU run (Task 3's cost measurement), the plan says so explicitly and stops short of doing it here.

---

### Task 1: H3 video-mux headroom warning — hold the pre-encode prediction until the post-encode probe answers

**Files:**
- Modify: `dw/result.py:115-139` (`warn_without_headroom`), `dw/result.py:686` and `dw/result.py:778-793` (`save_artifact`), `dw/result.py:893-899` and `dw/result.py:932-935` (`save_audio_video`)
- Test: `tests/test_result.py` (modify `TestTheHeadroomWarning.test_a_video_is_probed_even_though_the_waveform_already_warned`, add two new tests)

**Interfaces:**
- Consumes: existing `_peak_dbfs(waveform)`, `emit_warning(message, **data)`, `probe_media(path)` — unchanged.
- Produces: `warn_without_headroom(waveform, file_name, emit=True)` — new `emit` keyword, default `True` (audio-only callers are unaffected). `Result._predicted_peak_dbfs` — new instance attribute, the peak `warn_without_headroom` measured for a video mux without emitting yet, or `None`.

The bug: `dw/result.py` already runs `warn_if_written_above_full_scale` (the post-encode, ground-truth probe) unconditionally for a video mux, but the *pre-encode* `warn_without_headroom` warning still fires immediately and unconditionally before that probe runs. Result: a clean H3 video mux (measures fine after encoding) still carries a stale `audio_no_headroom` warning that nothing ever retracts, so `warnings: []` assertions on stock template defaults fail even though the file is fine. Fix: for a **video** mux only, measure the pre-encode peak without emitting, run the post-encode probe as today, and only fall back to emitting the pre-encode warning if the post-encode probe couldn't get a ground-truth answer at all. Audio-only saves are untouched — `warn_without_headroom` still emits immediately for them, exactly as before.

- [ ] **Step 1: Write the failing test for the new reconciliation behavior**

Replace the existing test that asserts today's (buggy) both-warnings behavior, and add a case for the probe-fails fallback. In `tests/test_result.py`, inside `class TestTheHeadroomWarning` (the class containing `test_a_video_is_probed_even_though_the_waveform_already_warned`, currently around line 1625):

```python
    def test_a_clean_video_mux_drops_the_stale_prediction(self):
        """#174 amendment: the pre-encode prediction fires on H3's own
        soundtrack every run, and the post-encode probe already proved the
        written file is fine - the caller should see nothing, not a stale
        warning about a file that turned out clean."""
        with patch(
            "dw.media_info.probe_media",
            return_value={"peak_dbfs": -1.12, "kind": "video"},
        ):
            warnings = self.events_from(lambda: self.save_muxed(torch.ones((2, 100))))

        assert warnings == []

    def test_a_dirty_video_mux_reports_only_the_measured_clip(self):
        """The post-encode probe found a real clip - report that, not the
        pre-encode guess, so the caller gets one answer with a real number."""
        with patch(
            "dw.media_info.probe_media",
            return_value={"peak_dbfs": 0.94, "kind": "video"},
        ):
            warnings = self.events_from(lambda: self.save_muxed(torch.ones((2, 100))))

        assert [w["kind"] for w in warnings] == ["audio_clipped"]
        assert warnings[0]["peak_dbfs"] == pytest.approx(0.94, abs=0.01)

    def test_an_unprobeable_video_mux_falls_back_to_the_prediction(self):
        """No ground truth available (a broken/short file) - the pre-encode
        guess is the only signal there is, so it still reaches the caller."""
        with patch("dw.media_info.probe_media", side_effect=OSError("truncated")):
            warnings = self.events_from(lambda: self.save_muxed(torch.ones((2, 100))))

        assert [w["kind"] for w in warnings] == ["audio_no_headroom"]
```

And change the existing `test_a_video_is_probed_even_though_the_waveform_already_warned` (it currently asserts `kinds == {"audio_no_headroom", "audio_clipped"}`, which is exactly the bug this task fixes) to:

```python
    def test_a_video_is_probed_even_though_the_waveform_already_warned(self):
        """#174: suppressing the post-encode probe whenever the pre-encode
        check already fired assumed the encoder only ever adds overshoot -
        true for the mp3s #159/#161 measured, backwards for an H3 video mux,
        whose AAC mux can land under full scale after starting over it. A
        video always gets the ground-truth post-encode read, and once that
        read is in, it - not the pre-encode guess - is what the caller sees."""
        with patch(
            "dw.media_info.probe_media",
            return_value={"peak_dbfs": 0.94, "kind": "video"},
        ):
            warnings = self.events_from(lambda: self.save_muxed(torch.ones((2, 100))))

        kinds = {w["kind"] for w in warnings}
        assert kinds == {"audio_clipped"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_result.py -k "TestTheHeadroomWarning" -v`
Expected: `test_a_video_is_probed_even_though_the_waveform_already_warned` FAILS (still asserts the old two-warning set once you change the assertion, current code produces `{"audio_no_headroom", "audio_clipped"}` which fails against `{"audio_clipped"}`); `test_a_clean_video_mux_drops_the_stale_prediction` FAILS (current code emits `audio_no_headroom` unconditionally); `test_an_unprobeable_video_mux_falls_back_to_the_prediction` PASSES already (current code already falls back to the pre-encode warning when the probe fails, so no regression risk there — note this as a baseline, not a new behavior).

- [ ] **Step 3: Add the `emit` parameter to `warn_without_headroom`**

In `dw/result.py`, replace the `warn_without_headroom` function (currently lines 115-139):

```python
def warn_without_headroom(waveform, file_name, emit=True):
    """Say when the soundtrack about to be written is at or over full scale.

    A clipped deliverable is invisible to the consumer this server is built
    for: the job succeeds, and an agent that cannot listen has `peak_dbfs`
    and no rule to read it against - `get_gallery_metadata` teaches the
    near-silent end of the range and said nothing about the other one (#158).
    A warning rather than a change to the mix: what level a deliverable
    should sit at is the workflow's to decide, and `normalize_audio` is the
    step that decides it.

    `emit=False` measures and returns the predicted peak without emitting
    the warning - used for a video mux, where the caller holds the emission
    until the post-encode probe (`warn_if_written_above_full_scale`) has a
    ground-truth answer, and only falls back to this prediction if that
    probe could not measure the written file at all (#174 amendment).
    """
    peak = _peak_dbfs(waveform)
    if peak is None or peak < HEADROOM_WARN_DBFS:
        return None
    if emit:
        emit_warning(
            f"The soundtrack written to {file_name} peaks at {peak:+.1f} dBFS, "
            f"which leaves no headroom below full scale - an mp3 or AAC encode "
            f"of it decodes above 0 dBFS and clips. Add a 'normalize_audio' "
            f"step (peak_dbfs: -1) before the step that saves it, or "
            f"'match_levels' on the join that made it.",
            kind="audio_no_headroom",
            file=file_name,
            peak_dbfs=round(peak, 2),
        )
    return peak
```

- [ ] **Step 4: Hold the prediction in `save_audio_video`'s two call sites**

In `dw/result.py`, in `save_audio_video`, replace (currently lines 893-899, the segment-backed branch):

```python
            audio = None
            if artifact.audio is not None and sample_rate is not None:
                audio = as_audio_track(artifact.audio)
                self._no_headroom_warned = (
                    warn_without_headroom(artifact.audio, os.path.basename(output_path))
                    is not None
                )
```

with:

```python
            audio = None
            if artifact.audio is not None and sample_rate is not None:
                audio = as_audio_track(artifact.audio)
                self._predicted_peak_dbfs = warn_without_headroom(
                    artifact.audio, os.path.basename(output_path), emit=False
                )
                self._no_headroom_warned = self._predicted_peak_dbfs is not None
```

And replace (currently lines 932-935, the muxed branch):

```python
        logger.debug(f"Muxing audio at {sample_rate}Hz into {output_path}")
        self._no_headroom_warned = (
            warn_without_headroom(audio, os.path.basename(output_path)) is not None
        )
```

with:

```python
        logger.debug(f"Muxing audio at {sample_rate}Hz into {output_path}")
        self._predicted_peak_dbfs = warn_without_headroom(
            audio, os.path.basename(output_path), emit=False
        )
        self._no_headroom_warned = self._predicted_peak_dbfs is not None
```

- [ ] **Step 5: Initialize the new attribute and reconcile after the post-encode probe**

In `dw/result.py`, in `save_artifact`, replace the initializer (currently line 686):

```python
        self._no_headroom_warned = False
```

with:

```python
        self._no_headroom_warned = False
        self._predicted_peak_dbfs = None
```

Then replace the post-encode block (currently lines 778-793):

```python
        # The level of what was actually written, which is the only one the
        # consumer will hear: the encode's own overshoot sits between the
        # waveform `warn_without_headroom` measured and this (#161). Only a
        # file that can carry a soundtrack, so an image never pays a probe.
        # The pre-encode warning only suppresses this for a plain audio
        # save - a video mux's overshoot is not reliably positive (#174),
        # so a video always gets the ground-truth post-encode check
        if content_type.startswith("audio") or content_type.startswith("video"):
            warn_if_written_above_full_scale(
                output_path,
                already_warned=(
                    self._no_headroom_warned
                    if content_type.startswith("audio")
                    else False
                ),
            )
```

with:

```python
        # The level of what was actually written, which is the only one the
        # consumer will hear: the encode's own overshoot sits between the
        # waveform `warn_without_headroom` measured and this (#161). Only a
        # file that can carry a soundtrack, so an image never pays a probe.
        # The pre-encode warning only suppresses this for a plain audio
        # save - a video mux's overshoot is not reliably positive (#174),
        # so a video always gets the ground-truth post-encode check
        if content_type.startswith("audio") or content_type.startswith("video"):
            written_peak = warn_if_written_above_full_scale(
                output_path,
                already_warned=(
                    self._no_headroom_warned
                    if content_type.startswith("audio")
                    else False
                ),
            )
            # A video's pre-encode prediction was held rather than emitted
            # (#174 amendment): the post-encode probe is the ground truth,
            # so a clean or genuinely-clipped result each get exactly one
            # answer - nothing here, or `audio_clipped` from the probe
            # itself. The only time the held prediction is worth anything is
            # when the probe could not measure the file at all, in which
            # case it is the one signal available and is surfaced late
            # rather than dropped silently
            if (
                content_type.startswith("video")
                and self._no_headroom_warned
                and written_peak is None
            ):
                emit_warning(
                    f"The soundtrack written to {os.path.basename(output_path)} "
                    f"was predicted to peak at {self._predicted_peak_dbfs:+.1f} "
                    f"dBFS before encoding, and the written file could not be "
                    f"re-measured to confirm whether the mux corrected it - add "
                    f"a 'normalize_audio' step (peak_dbfs: -1) before the step "
                    f"that saves it, or 'match_levels' on the join that made it.",
                    kind="audio_no_headroom",
                    file=os.path.basename(output_path),
                    peak_dbfs=round(self._predicted_peak_dbfs, 2),
                )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_result.py -k "TestTheHeadroomWarning" -v`
Expected: all PASS, including the three from Step 1.

- [ ] **Step 7: Run the full result/audio/video test modules to check for regressions**

Run: `venv/bin/python -m pytest tests/test_result.py tests/test_job_warnings.py -v`
Expected: all PASS. `test_the_muxed_deliverable_is_measured_too` and `test_a_video_with_a_quiet_track_is_not_warned_about` (unmodified, earlier in the same class) must still pass unchanged — they exercise the case where the post-encode probe cannot run at all (mocked `encode_video`, no real file on disk), which is exactly the fallback path this task adds, not a new behavior for those two.

- [ ] **Step 8: Update the regression case note in the spec doc**

In `docs/proposals/h3-video-mux-headroom-warning-partial.md`, add a line under "Remaining steps" item 5 (or wherever fits) noting this was implemented and the date, and cross-reference the commit once made — do this as part of the commit in Step 9, not before, so the commit hash is real.

- [ ] **Step 9: Commit**

```bash
git add dw/result.py tests/test_result.py docs/proposals/h3-video-mux-headroom-warning-partial.md
git commit -m "$(cat <<'EOF'
fix(engine): #174 amendment - hold the pre-encode headroom warning for a video mux until the post-encode probe answers

A video mux's pre-encode audio_no_headroom warning fired unconditionally,
independent of what the post-encode ground-truth probe found - so a clean
H3 video mux still carried a stale warning about a file that turned out
fine, and warnings: [] assertions on stock template defaults failed.
Scoped to content_type.startswith("video") only; audio-only saves are
unaffected.
EOF
)"
```

---

### Task 2: WAL mode on the jobs database

**Files:**
- Modify: `dw/server/jobs.py:151-152` (`JobHistory._connect`)
- Test: `tests/test_jobs_listing.py` or a new small test in `tests/test_server.py`-adjacent file — added inline below as `tests/test_job_history_wal.py`

**Interfaces:**
- Consumes: `JobHistory(db_path)` constructor (unchanged signature) — see `tests/test_jobs_listing.py:53`, `JobHistory(str(tmp_path / "jobs.sqlite"))`.
- Produces: nothing new externally; `_connect()`'s returned `sqlite3.Connection` now has `journal_mode` set to `wal`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_job_history_wal.py`:

```python
"""WAL mode on the jobs database - a real concurrency benefit (readers do
not block behind a writer) for a near-zero-risk one-line change. See
docs/proposals/maintenance-screen.md, "Phase 0"."""

from dw.server.jobs import JobHistory


def test_the_jobs_database_uses_wal_mode(tmp_path):
    history = JobHistory(str(tmp_path / "jobs.sqlite"))

    with history._connect() as connection:
        (mode,) = connection.execute("PRAGMA journal_mode").fetchone()

    assert mode.lower() == "wal"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_job_history_wal.py -v`
Expected: FAIL, `mode` is `'delete'` (SQLite's default), not `'wal'`.

- [ ] **Step 3: Set WAL mode in `_connect`**

In `dw/server/jobs.py`, replace (currently lines 151-152):

```python
    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=5)
```

with:

```python
    def _connect(self):
        # WAL mode lets a reader (the web UI polling job status, an MCP
        # get_job call) proceed without blocking behind whatever write the
        # worker is mid-transaction on, and vice versa - the default
        # rollback-journal mode takes a database-wide lock for the
        # duration of a write. journal_mode is a property of the database
        # file, not the connection, but PRAGMA is cheap and idempotent, so
        # it is set on every connect rather than assumed to have stuck.
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_job_history_wal.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full jobs/server test suite to check for regressions**

Run: `venv/bin/python -m pytest tests/test_jobs_listing.py tests/test_jobs_reused.py tests/test_server.py tests/test_observed_cost.py -v`
Expected: all PASS. WAL mode changes on-disk file layout (adds a `-wal` and `-shm` file beside `jobs.sqlite`) but not query semantics, so no behavioral change is expected anywhere else.

- [ ] **Step 6: Commit**

```bash
git add dw/server/jobs.py tests/test_job_history_wal.py
git commit -m "$(cat <<'EOF'
fix(server): enable WAL mode on the jobs database

A reader (job status poll, MCP get_job) no longer blocks behind whatever
write the worker is mid-transaction on. One-line change, no schema or
query changes. See docs/proposals/maintenance-screen.md, "Phase 0".
EOF
)"
```

---

### Task 3: `best-of-n-to-video` catalog template

**Files:**
- Create: `workflows/templates/best-of-n-to-video.json`
- Test: `tests/test_best_of_n_template.py` (new, small, structural — the existing parametrized `tests/test_examples.py` sweep already covers schema/reference validation for every file under `workflows/`, since it auto-discovers workflow files)

**Interfaces:**
- Consumes: `select` task (`dw/tasks/select.py`, `returns="artifact"`, no schema change needed), `judge` task (`dw/tasks/judge.py`, `returns="scalar"` — **no `result` block allowed** on the `judge` step), `for_each`/`gather:` (existing engine features, no change), the existing `workflows/templates/ltx2/image-to-video.json` sub-workflow (referenced by relative path, the same way `workflows/templates/sub-workflow.json` references `./lora.json`).
- Produces: nothing consumed by other tasks in this plan — a standalone catalog addition.

This task authors **one** template demonstrating the shape score-and-select's engine work (already shipped) exists for: fan out N stills from one prompt, judge each, keep the best, spend the expensive video stage on it alone. No engine code changes.

Design notes settled during planning (deviating slightly from the original proposal text, which named a hypothetical `templates/flux/best-of-n-to-video` path and a `builtin:` sub-workflow reference — neither matches the actual repo convention):

- There is no `workflows/templates/flux/` subdirectory in the catalog (only `ltx2/` and `minimax/` have subdirectories; everything else, including `text-to-image.json`'s Stable Diffusion 1.5 pipeline, is flat under `workflows/templates/`). This template goes at `workflows/templates/best-of-n-to-video.json`, flat, matching the majority convention.
- `builtin:` names only the packaged `dw/workflows/` sub-workflows (CLAUDE.md); a catalog template composing another catalog template uses a relative `"workflow": {"path": "./ltx2/image-to-video.json", ...}` step, the same way `workflows/templates/sub-workflow.json` references `./lora.json`.
- No per-candidate seed. The workflow-schema `step.seed` field only accepts a literal `variable:` reference (`dw/workflow_schema.json` pattern `^variable:`), so a `for_each` entry cannot vary it via `item:seed`. Simpler and schema-safe: declare **no** top-level `seed` (step-cache is then correctly disabled for this template, since the whole point is a fresh random draw each run), and let each `still@<name>` for_each member draw its own independent random generation from an identical prompt — that alone produces four different candidates.
- `judge` returns a scalar (`dw/tasks/task.py` declares `returns="scalar"`), so the `judge` step must carry **no** `result` block — `dw/scalar_result_validation.py` refuses one and this is unconditional, not a style choice.
- No `cost` entry. Per the Global Constraints above, a curated cost figure has to be measured on real GPU hardware, which this planning pass has no access to — ship without one and note in the description that it is pending a measured run (this matches how newly-onboarded templates elsewhere in the catalog are handled before their first field-tested run).

- [ ] **Step 1: Write the template file**

Create `workflows/templates/best-of-n-to-video.json`:

```json
{
    "id": "best-of-n-to-video",
    "shape": "shot",
    "summary": "Generate several stills from one prompt, keep the sharpest by a VLM judge's score, and spend a video render on that one alone.",
    "description": "The README's headline shape, made deterministic: fan out N Stable Diffusion 1.5 stills from one prompt ('still', a for_each group so each candidate is named and individually cached), score each against a rubric with a vision-language model ('judge', paired to 'still' by for_each position), keep the highest score with no agent in the loop ('pick', the 'select' task's argmax rule), and run the LTX-2.5 image-to-video template on the winner alone ('video', a sub-workflow step). No seed is set: the point is N independently-drawn candidates from one prompt, so step-cache reuse is correctly disabled here rather than pinned to a fixed draw. See docs/proposals/score-and-select-complete.md for the engine design this exercises.",
    "cost_drivers": ["candidates", "num_inference_steps"],
    "variables": {
        "prompt": "a red fox sitting in a snowy forest clearing at dawn, photograph, sharp focus",
        "candidates": ["a", "b", "c", "d"],
        "rubric": "How sharp and well-composed is this photograph? Penalize blur, extra or malformed limbs, and flat or blown-out lighting.",
        "scale": [0, 10],
        "video_prompt": "prompt:best-of-n-to-video/video-motion"
    },
    "steps": [
        {
            "name": "still",
            "for_each": "variable:candidates",
            "pipeline": {
                "configuration": {
                    "component_type": "StableDiffusionPipeline"
                },
                "from_pretrained_arguments": {
                    "model_name": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "torch_dtype": "torch.float32",
                    "safety_checker": null
                },
                "arguments": {
                    "prompt": "variable:prompt",
                    "num_inference_steps": 25
                }
            },
            "result": {
                "content_type": "image/jpeg",
                "embed_metadata": true,
                "subfolder": "intermediate"
            }
        },
        {
            "name": "judge",
            "for_each": "variable:candidates",
            "task": {
                "command": "judge",
                "arguments": {
                    "image": "previous_result:still",
                    "rubric": "variable:rubric",
                    "scale": "variable:scale"
                }
            }
        },
        {
            "name": "pick",
            "task": {
                "command": "select",
                "arguments": {
                    "candidates": ["gather:still"],
                    "scores": ["gather:judge"],
                    "rule": "argmax"
                }
            },
            "result": {
                "content_type": "image/jpeg",
                "embed_metadata": true,
                "subfolder": "intermediate"
            }
        },
        {
            "name": "video",
            "workflow": {
                "path": "./ltx2/image-to-video.json",
                "arguments": {
                    "image": "previous_result:pick",
                    "prompt": "variable:video_prompt"
                }
            },
            "result": {
                "subfolder": "final"
            }
        }
    ]
}
```

`video_prompt` references a stored prompt (`prompt:best-of-n-to-video/video-motion`) rather than a literal string, since the LTX-2.5 trained caption form is a different genre from the still's plain photographic prompt (CLAUDE.md's IC-LoRA note makes the same distinction for a different family) and a placeholder literal here would either be wrong-shaped or misleadingly specific. Step 2 below saves that stored prompt.

- [ ] **Step 2: Save the stored video-motion prompt**

This template references `prompt:best-of-n-to-video/video-motion`, which must exist as a stored prompt for `realize_args` to resolve it (CLAUDE.md, `prompt:` convention). Create it directly as a file rather than requiring a running server for this planning step — check where the workspace's `prompts/` directory actually lives for the checkout-as-workspace case (CLAUDE.md, workspace resolution: rule four, a checkout with a `workflows/` directory satisfies default workspace resolution, so its `prompts/` sits at the repo root):

Run: `ls prompts/ 2>/dev/null || echo "no prompts/ directory at repo root yet"`

If it exists, create `prompts/best-of-n-to-video/video-motion.json` (matching the shape of an existing stored prompt file — check one first, e.g. `find prompts -name '*.json' | head -1` and read it for the exact `{"text": ...}` shape) with a short LTX-2.5-appropriate motion caption, e.g.:

```json
{
    "text": "A red fox sits in a snowy forest clearing at dawn. Its ears twitch, breath visible in the cold air, as soft snowfall drifts past in the low golden light. The camera holds a slow, steady push in."
}
```

If no `prompts/` directory exists at the repo root, do not create one speculatively — instead change `video_prompt`'s default in the template to a literal string with the same content (drop the `prompt:` prefix), and note in the template's `description` that a stored prompt is preferred once the workspace has a prompt library. Verify which case applies before writing the template file in Step 1, and adjust before committing.

- [ ] **Step 3: Validate the template structurally**

Run: `venv/bin/python -m dw.validate workflows/templates/best-of-n-to-video.json`
Expected: valid (schema-conformant); it will report a `plan` only if it can resolve `prompt:`/`asset:` references against a workspace, which needs no GPU.

- [ ] **Step 4: Write the structural test**

Create `tests/test_best_of_n_template.py`:

```python
"""The best-of-n-to-video template wires select/judge the way
docs/proposals/score-and-select-complete.md designed: a for_each fan-out,
a scalar judge with no result block, and select's argmax over the paired
gather: lists. This is a structural check - the reducer engine itself is
tested in tests/test_select.py / tests/test_judge.py, and schema/reference
validity is already swept by the parametrized tests in test_examples.py."""

import json
import os

TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "workflows", "templates", "best-of-n-to-video.json"
)


def load():
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def steps_by_name(definition):
    return {step["name"]: step for step in definition["steps"]}


def test_the_judge_step_carries_no_result_block():
    """judge returns a scalar - a result block on it is refused by
    dw/scalar_result_validation.py, so the template must never carry one."""
    steps = steps_by_name(load())

    assert "result" not in steps["judge"]


def test_pick_selects_by_argmax_over_the_paired_gather_lists():
    steps = steps_by_name(load())
    pick_task = steps["pick"]["task"]

    assert pick_task["command"] == "select"
    assert pick_task["arguments"]["rule"] == "argmax"
    assert pick_task["arguments"]["candidates"] == ["gather:still"]
    assert pick_task["arguments"]["scores"] == ["gather:judge"]


def test_still_and_judge_share_the_same_for_each_list():
    """judge's previous_result:still must resolve to the matching member by
    position, which only holds if both steps for_each the same variable."""
    steps = steps_by_name(load())

    assert steps["still"]["for_each"] == steps["judge"]["for_each"]


def test_the_template_declares_no_cost_yet():
    """A curated cost figure has to be measured on real GPU hardware - see
    the Global Constraints note in docs/superpowers/plans/
    2026-09-20-tier1-proposals.md. Once someone measures a real run, this
    test should be updated (or removed) alongside adding the real number."""
    definition = load()

    assert "cost" not in definition


def test_saving_steps_are_marked_final_or_intermediate():
    steps = load()["steps"]
    saving_steps = [s for s in steps if "result" in s]

    for step in saving_steps:
        assert step["result"].get("subfolder") in ("final", "intermediate"), (
            f"step '{step['name']}' saves without a subfolder marking"
        )
```

- [ ] **Step 5: Run the new test and the catalog-wide sweep**

Run: `venv/bin/python -m pytest tests/test_best_of_n_template.py tests/test_examples.py tests/test_template_subfolders.py -v`
Expected: all PASS. `tests/test_examples.py` auto-discovers the new file and runs schema validation, untrusted-mode validation, and reference resolution against it with no changes needed on your part; `tests/test_template_subfolders.py` checks the final/intermediate marking rule against every template including this one.

- [ ] **Step 6: Note the pending GPU-measured cost as a follow-up**

This plan stops here for Task 3 — do not fabricate a `cost` entry. Once this template has been run once on real GPU hardware (e.g. the `lem` box used elsewhere in this project's history), add a `cost` block with the measured minutes and, if a plugin skill later quotes this template, pin it the way `tests/test_plugin_skills.py` pins the others.

- [ ] **Step 7: Commit**

```bash
git add workflows/templates/best-of-n-to-video.json tests/test_best_of_n_template.py
# also `git add prompts/best-of-n-to-video/video-motion.json` if Step 2 created it
git commit -m "$(cat <<'EOF'
feat(catalog): best-of-n-to-video - the select/judge reducer's first template

Fans N Stable Diffusion 1.5 stills from one prompt, scores each with a
VLM judge, keeps the argmax with no agent in the loop, and spends an
LTX-2.5 render on the winner alone. Exercises the score-and-select engine
work (already shipped) end to end; no engine changes here. cost is
intentionally absent pending a measured GPU run.
EOF
)"
```

---

### Task 4: `script-to-video` plugin skill

**Files:**
- Create: `plugins/dw/skills/script-to-video/SKILL.md`
- Modify: `plugins/dw/README.md` (name the new skill, per CLAUDE.md: "Every skill the directory holds is named in `plugins/dw/README.md` and here [CLAUDE.md]")
- Modify: `/Users/don/src/dkackman/diffusers-workflow/CLAUDE.md` (add the skill's name to the "Claude Code plugin" section's list)
- Test: `tests/test_plugin_skills.py` — no code change needed; it auto-discovers every `plugins/dw/skills/*/SKILL.md` via `glob.glob`, so the new file is automatically swept by `test_a_skill_has_a_triggering_description_under_the_size_cap` and `test_every_catalog_name_a_skill_quotes_resolves`.

**Interfaces:**
- Consumes: no engine code — composes existing MCP tools and skills by name only (`get_server_info`, `list_workflows`, `dw:series-episodes`, `h3-prompt-writing`, `validate_workflow`, `plan.estimate`, `run_workflow`, `acknowledged_cost`, `get_job_events`, `get_gallery_metadata`, `rerun_job`, `templates/assemble-and-score`).
- Produces: nothing consumed elsewhere in this plan.

This is pure documentation/composition work: no engine code changes, matching the proposal's own framing ("Nothing above is a new engine capability"). The two hard requirements from `tests/test_plugin_skills.py` that the file must satisfy:
1. Frontmatter `name` must equal the directory name (`script-to-video`), and `description` must be present and over 40 characters.
2. The whole file must be ≤ 12,288 bytes (`SKILL_SIZE_LIMIT = 12 * 1024`).
3. Every backtick-quoted name matching `` `(templates|models)/...` `` must resolve to a real file under `workflows/` — so every catalog name quoted below must be checked against the actual catalog before committing (Step 3).

- [ ] **Step 1: Confirm which catalog names are safe to quote**

Run this before writing prose, since a misremembered template name fails the pinning test:

```bash
ls workflows/templates/minimax/
ls workflows/templates/ltx2/
ls workflows/templates/assemble-and-score.json
ls workflows/templates/best-of-n-to-video.json  # from Task 3, if committed first
```

Only quote (in backticks, as `` `templates/...` ``) names this confirms exist. If Task 3 has not landed yet when this task is done, drop the `best-of-n-to-video` mention from Step 6's remediation table entry rather than quoting a file that doesn't exist yet — the pinning test will otherwise fail.

- [ ] **Step 2: Write the skill file**

Create `plugins/dw/skills/script-to-video/SKILL.md`:

```markdown
---
name: script-to-video
description: Use when a dw MCP server is connected and the user hands over a script or prose and wants it turned into a video end to end - decomposing it into a shot list, casting recurring characters once, picking each shot's template and family, validating cost before spending, running and diagnosing, and handing the finished shots to the cut-and-score pass. Does not cover model auto-tuning or writing new pipeline JSON.
---

# Script to video on a dw server

Turning a script into shots is the one piece none of the other skills
cover. Everything downstream of a shot list already exists: `dw:minimax-h3`
and `dw:ltx-2.5` pick a template per shot's shape, `h3-prompt-writing`
writes the structured prompt, `dw:series-episodes` keeps a cast consistent,
and `templates/assemble-and-score` cuts and scores the result. This skill
is the decision tree that connects them, in order. It does not add anything
to the engine - see "Not in scope" below for what it deliberately declines.

## 1. Read the script and the server

`get_server_info` for device and workspace, the way every dw skill starts.
Read the script as prose and dialogue, the way a person would - this skill
does not parse screenplay formats.

## 2. Decompose into a shot list, not a shot count

The unit that matters is *shots*, not lines or scenes: one scene may be
several shots, and `dw:minimax-h3`'s sweet spot is 4-6 second shots (the
`17*n+5` frame grid). For each shot, name it, note which character(s)
appear, what happens, and roughly how long it runs. This is genuine
reasoning work with no existing mechanism to lean on - the one step where
"an LLM does this well" is actually true here, unlike auto-tuning a slow
model.

## 3. Cast recurring characters once, before any shot generates

Read `dw:series-episodes` step 0 before drawing anything: every character
appearing in more than one shot gets a portrait (and voice clip, if they
speak) drawn once and `keep_output`'d as `asset:cast/<name>`, referenced
from every later shot with `from_file`, never `from_previous_result`. A
script with one continuous scene and no recurring cast can skip this and
let the family template draw its own portrait. This is the step that goes
missing silently - nothing validates "the same character looks the same"
across separately generated shots, so it has to be deliberate.

## 4. Per shot: pick the shape, write the prompt

Silent action or establishing shots go through a `dw:ltx-2.5` template
(`templates/ltx2/text-to-video`, `templates/ltx2/image-to-video`, or a
chained one for a longer take). Anything with dialogue, a held voice, or a
scored montage goes through `dw:minimax-h3`'s `templates/minimax/dialogue-short`
or `templates/minimax/music-video`, prompted with `h3-prompt-writing`'s
structure (`subject_definitions`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`) rather
than the raw script line. Read the chosen family's own skill before writing
a single prompt - the hard rules (frame count, canvas, reference limits)
live there, not here.

## 5. Validate and quote cost before queuing

`validate_workflow` against the chosen template with the assembled `shots`
argument, read `plan.estimate` and `plan.downloads_required`, and state the
cost before calling `run_workflow`. Only pass `acknowledged_cost` once a
human (or an explicit pre-authorization) has signed off - this is a real
gate, not a formality, and skipping it here is the one thing this skill
must not do even when running unattended.

## 6. Run, read warnings, retry per shot

After the job completes, read `get_job_events` / `get_gallery_metadata` for
the warnings the engine already emits (`audio_no_headroom`, `audio_clipped`,
an elision diagnostic) and `rerun_job(new_seed=True)` a shot that reads
wrong - not by inventing a new heuristic, by reading what is already
surfaced.

## 7. Cut, score, deliver

Once every shot exists, hand off to `templates/assemble-and-score` (or the
`dw:series-episodes` five-beat procedure, if this script is one episode of
a series) for the recut/bed/match_levels/normalize/pair pass. That skill
already owns this step - do not re-derive it here.

## Not in scope

- **Automatic model speed optimization.** If a model is slow, that is a
  human-in-the-loop task informed by `observed_cost`, not something this
  skill attempts unsupervised - the search space is large and a wrong
  choice can silently change output quality.
- **Workflow authoring from scratch.** This skill picks among existing
  catalog templates; a script whose shape no template covers is a
  "propose a new template" conversation, not something to improvise at
  run time.
- **Full unattended autonomy.** Step 5's cost acknowledgment is a real gate
  every time, not a one-time setup step.
```

- [ ] **Step 3: Verify the pinning test passes**

Run: `venv/bin/python -m pytest tests/test_plugin_skills.py -v`
Expected: all PASS, including the new `script-to-video` parametrized cases of `test_a_skill_has_a_triggering_description_under_the_size_cap` and `test_every_catalog_name_a_skill_quotes_resolves`. If a name fails to resolve, fix the quoted name (do not add a file to make it resolve) — Step 1 already confirmed which names are safe.

- [ ] **Step 4: Check the size cap directly**

Run: `wc -c plugins/dw/skills/script-to-video/SKILL.md`
Expected: under 12288. If over, trim prose in sections 2/4/6 first (they carry the most restate-able detail) before cutting a whole section.

- [ ] **Step 5: Name the skill in the two places CLAUDE.md requires**

In `plugins/dw/README.md`, add `script-to-video` to wherever the existing skills (`minimax-h3`, `minimax-music3`, `ltx-2.5`, `series-episodes`) are listed, with a one-line description matching the frontmatter.

In `/Users/don/src/dkackman/diffusers-workflow/CLAUDE.md`, under "### Claude Code plugin", update the sentence listing the composition skills to include `script-to-video`, describing it the same way the existing sentence describes `series-episodes` (a shape above the family skills).

- [ ] **Step 6: Run the full plugin test file once more after the doc edits**

Run: `venv/bin/python -m pytest tests/test_plugin_skills.py -v`
Expected: all PASS (README/CLAUDE.md edits are prose-only and not asserted on directly by this test file, but re-running confirms nothing else broke).

- [ ] **Step 7: Commit**

```bash
git add plugins/dw/skills/script-to-video/SKILL.md plugins/dw/README.md CLAUDE.md
git commit -m "$(cat <<'EOF'
feat(plugin): script-to-video skill - the shot-list decomposition step

Composes existing mechanisms (family template pickers, prompt-writing
skills, the cast-consistency procedure, the cost gate, warning-driven
retry, and the cut-and-score handoff) into the one decision tree that was
missing: turning a prose script into the shots the rest of the catalog
already knows what to do with. No engine changes.
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers the entirety of `h3-video-mux-headroom-warning-partial.md`'s "Remaining steps" 1-4 (item 5's regression-suite-wording update is a tester/doc action folded into Step 8, not separate code). Task 2 covers `maintenance-screen.md`'s Phase 0 in full (the rest of that proposal — the Maintenance UI page — is explicitly out of scope for this plan, per the Tier 1 ranking that split it out). Task 3 covers `score-and-select-partial.md`'s "One catalog template" section in full except the plugin-skill-naming follow-on, which is noted as a future step once a skill actually needs to name it (this plan's Task 4 skill does not depend on Task 3's template, and Task 3's template does not currently need naming from any skill). Task 4 covers `script-to-video-agent-skill.md` in full, choosing "plugin skill" over "docs guide" per the proposal's own stated leaning.
- **Placeholder scan:** No TBD/TODO left in any step; the one intentionally-omitted value (Task 3's `cost` field) is explained, not left vague, and is verified absent by a test rather than silently skipped.
- **Type/interface consistency:** `warn_without_headroom(waveform, file_name, emit=True)` is used identically at all three call sites (Task 1, Steps 3-4); `self._predicted_peak_dbfs` is initialized once (Step 5) and read once (Step 5's reconciliation block) with no other writer introduced.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-20-tier1-proposals.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
