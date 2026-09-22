# Remaining work: output assessment (#193) - stages 2-4

Split from `output-assessment-complete.md` on 2026-09-20. Stage 1 (evidence:
mux-aware `get_output_audio` with excerpts, new `get_output_frames`) shipped
and merged to `develop` (`525d0f9`/`48425e9`/`3c6ce20`). Stages 2-4 below are
not started. All design decisions, vocabulary, and section numbering below
refer to the complete doc, which is the authoritative design reference — this
file only tracks what remains to build against that design.

## Stage 2 - Boundaries (original Section 3)

- `shots` attribute on `AudioVideo` and plain frames results:
  `[{name, start_frame, num_frames, start_sample, num_samples,
  overlap_frames?, hard_cut?}]`.
- Populated by `concat_videos`, `dissolve_videos`, and `run_chain`
  (`dw/pipeline_processors/chain.py`); `start_sample` measured from the
  actual accumulated waveform length, never derived from `start_frame / fps`.
- Names assigned in the workflow layer (`Workflow.run`) from the step's
  pre-resolution `gather:` reference list, not in the task.
- Every `AudioVideo(` constructor gets a stated rule per the table in the
  original Section 3 (`pair_audio` recomputes sample fields; `interpolate_frames`
  rescales frame fields; `stabilize`/`loop_frames` carry through unchanged;
  `slice_audio`/`trim_*`/anything changing frame count drops shots); a test
  enumerates constructor sites and fails on one the table doesn't name.
- Manifest gains a `shots` entry beside `subfolder`.
- File-level persistence: a post-write remux pass tags the mp4 container
  (`container.metadata["comment"]` with a `dw.shots` JSON blob), surviving
  `os.link` (so `keep_output`, `upload_asset`, and pruned run directories keep
  it); `read_embedded_metadata` folds the tag read into `probe_media`.
- Precedence in every probe: caller's `shots` argument > manifest > file tag >
  none.
- `hard_cut` per-shot flag; `concat_videos` gains a `hard_cuts` argument.
- `get_gallery_metadata` gains `media.shots`.

## Stage 3 - Probes and rules (original Sections 1, 4, 5)

- `dw/tasks/assess.py`: five probe tasks (`analyze_shots`, `analyze_seams`,
  `analyze_sync_drift`, `analyze_av_alignment`, `analyze_continuity`), each
  CPU-only (numpy + PyAV), streaming rather than materializing every frame.
- `dw/assessment_rules.py`: the rules table (`{name, probe, field, comparator,
  threshold, severity, says}`), pinned by `tests/test_assessment_rules.py`
  against real probe fields. `av_correlation` measured but ships with no rule
  until lem data sets one.
- `dw/server/assess.py` + routes in `dw/server/app.py`:
  `GET /api/gallery/{name:path}/assess`, `/inspect`, `/frames` (evidence tool,
  if not already covered by stage 1's `get_output_frames` route), `/audio`
  excerpt support (stage 1 covers the MCP/media side; verify route parity).
- MCP: `dw_mcp/assess.py` — `assess_output(name, detail=False, workspace=None)`,
  `inspect_output(name, probe, workspace=None, **arguments)`.
- `list_tasks` gains a fourth list, `assessment`, for the five probes;
  `docs/TASKS.md` gets a `### <probe>` section each.
- Guide: `docs/WORKFLOW_GUIDE.md` gains a `## Assessing a run's output`
  section.
- Testing: synthetic-media probe tests (no models) per the original Section 7
  — a known 6 dB step flags `seam_level_step` at the right seam and nowhere
  else, a 267-sample-per-shot drift is reported, a dissolve doesn't flag a
  step, a `hard_cut` boundary doesn't flag a jump, memory/budget/route tests.

## Stage 4 - Skill and discovery (original Section 6)

- New plugin skill `plugins/dw/skills/judging-output/` (`dw:judging-output`),
  named in `plugins/dw/README.md` and `CLAUDE.md`.
- Body: look-before-measuring, `assess_output`, drill-down via
  `inspect_output`/`get_output_frames(seams=...)`, the remediation table
  (mapping each rule to a template variable/fix), "recut don't regenerate
  when you can," and the honest tier-1 limits.
- Family skills' "run and judge" steps shrink to a pointer:
  `minimax-h3` step 4, `series-episodes`, and `ltx-2.5` each hand off to
  `dw:judging-output` instead of carrying their own judging prose.
- New test in the `test_plugin_skills.py` style pinning the skill's quoted
  thresholds to `dw/assessment_rules.py`.
- MCP server instructions block gains one sentence describing the
  assess/look/listen loop.
- **Field test on lem**: `assess_output` over MCP on an existing
  `music-video` and `dialogue-short` run, a `chained-segments` run, and the
  #197 repro cut — this is what finalizes the skill's threshold prose and
  decides `av_correlation`'s rule (currently unset).

## Release note items owed once stages 2-4 land

`get_output_audio` no longer refuses `video/mp4` and moved off the `/outputs`
mount (stage 1, may already be owed); manifest entries and mp4 files carry
`shots`; `list_tasks` has a fourth list; `minimax-h3` step 4 is rewritten.
