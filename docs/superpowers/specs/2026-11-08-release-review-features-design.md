# Release-review features for 0.20.0: six "even better ifs"

Written 2026-11-08 from the 0.20.0 readiness review of `develop` (52a05434).
The review found the long-form feature set (#1199) sound and listed six
things that would make it feel like one feature rather than many. Don asked
for all six, with the five review defects, before the release. The defects
are on `fix/release-review-0-20-0` and are not described here except where a
feature leans on one.

Every design call below that Don did not make is marked *(my call)*.

## Intent

A long piece now has tiers, duration chains, per-segment checks, repair in
place, long narration and an editor export. The seams between those are
where a user still does hand work: resubmitting a draft at a better tier by
copying its workflow, redoing a whole chain when segment 15 of 20 died,
remembering a template's frame grid, levelling a delivered piece in dBFS
when editors talk in LUFS, finding a dropped sentence in a finished
narration by ear, and reading an `.otioz` by its JSON. Each feature removes
one of those.

## Goals, non-goals, anti-goals

Goals: each feature is reachable from the MCP surface and the catalog
templates with no new JSON authoring for the common case; each is covered
by unit tests runnable on this Mac (no GPU is available this week - lem is
down); each ships with its guide and skill sentence; the architecture
ratchets stay at baseline.

Non-goals: new UI beyond the timeline strip (the editor and job page get no
other change); re-rolling a failed narration take automatically (report
only in this cut); a resume that survives a change of plan (different
segment count or frames); broadcast-grade loudness compliance beyond the
integrated measure and a true-peak hold.

Anti-goals: anything that spends GPU time without the cost gate; a
behaviour change on an existing template's default output (every new knob
defaults off or to today's behaviour); a module pushed over the 1,100-line
ceiling or a function over the complexity ratchet.

## Constraints that hold across all six

- Engine code holds no model knowledge; a rule about a value lives in the
  template's `variable_constraints` (CLAUDE.md).
- Every disk path goes through `dw/security.py` validators with a non-None
  base; references resolve through `dw/locations.py`.
- A run-time warning must reach the caller as a job warning, not only the
  log (the #85 rule).
- The MCP tool surface has a text budget (`SURFACE_BUDGET`, 16,000 tokens,
  `tests/test_mcp_surface_budget.py`); a tool docstring grows by a sentence,
  not a paragraph, and the guide carries the rest.
- Modules near the 1,100-line ceiling: `chain.py` 1,014, `plan.py` 1,086,
  `components.py` 1,046, `h3_rules.py` 1,057, `validation.py` 1,070,
  `workflow_run.py` 1,096, `routes/media.py` 1,093, `server/jobs.py` 1,100.
  New code of any size goes in a new or roomy module; the UI ceiling is 600
  lines per file.
- The architecture seam map (`docs/ARCHITECTURE.md`) gets a row for each new
  rule, named by module and enforcing test.

## Feature A: promote a draft - `rerun_job` with argument overrides

**What.** `POST /api/jobs/{id}/rerun` and the MCP `rerun_job` tool take an
optional `arguments` object of variable overrides merged onto the stored
job's arguments before admission. `rerun_job(job_id, arguments={"tier":
"best"}, acknowledged_cost=...)` is the one-call promotion of a draft.

**Behaviour.**
- Merge is shallow, by variable name, overrides win; `new_seed` still draws
  after the merge (so an override of the seed variable and `new_seed`
  together is refused with a message naming both).
- Admission is exactly a fresh submit's: the merged arguments are validated
  against the workflow's variables, and a bound `acknowledged_cost` is
  checked against the plan of the *merged* arguments (the route already
  plans from `arguments`; the docstring's claim that a rerun's fingerprint
  cannot change becomes "a fresh seed never changes it; an override may").
- An override of a variable the workflow does not declare is refused at the
  route (400) naming the variable, as `run_workflow` does.
- The job record stores the overrides it was made with (a `rerun_of` with
  `overrides`), so `get_job` shows what differed from the parent *(my call:
  reuse the existing rerun provenance field if one exists; otherwise add
  `overrides` beside the parent id)*.

**Surface.** `RerunRequest.arguments: dict | None`; `rerun_spec(job_id,
new_seed, overrides)`; MCP `rerun_job(..., arguments: dict | None = None)`
with one added docstring sentence: "`arguments` overrides variables for the
rerun (a draft promoted with `{"tier": "best"}`); validate first, since an
override changes the plan a bound acknowledgement must name." `docs/MCP.md`
and `docs/SERVER.md` get the paragraph; the long-form skill's
quality-and-speed reference says "promote with rerun_job".

**Out.** No UI control in this cut.

## Feature B: resume a chain from its kept segments

**What.** A chain block key `resume_from`: a reference (`output:` or
`variable:`) naming one segment file of an earlier run of the same chain.
The run reuses that run's finished segments as segments 1..k and renders
only k+1..N.

**Why it is possible now.** `save_segments` writes each finished segment as
a playable mp4 (frames trimmed to the timeline, audio muxed) and a crashed
chain leaves them on disk (`chain_spill.py`); `chain_repair.segment_files`
already gathers a file's siblings and checks the set is complete.

**Behaviour.**
- The engine gathers the named file's siblings (reuse `segment_files`),
  takes them in index order, and requires k < N and that every kept file's
  frame count equals the plan's frame count for that segment index (after
  the plan's own trims). A mismatch is refused naming the segment, the kept
  count and the planned count: a resume never silently changes the plan.
- Supported continuity: `last_frame` (carry = the kept file's last frame)
  and `guide` (carry = the kept file's last `guide_frames` frames and their
  audio, read from the mp4). The kept file's audio is AAC, so the first
  resumed seam's audio guide is re-encoded from a decoded track and has no
  latents; Don accepted that (2026-11-08) with the `chain_resumed` warning
  saying so. Every later seam is a live carry as usual. `experimental_latent_carry` and a resume are
  refused together at validate (latents are not on disk). `anchor:
  segment_1` and `keystone: segment_1` read segment 1's frames from the
  first kept file. `look_hold` re-derives its state by observing the kept
  segments in order *(my call; the builder may instead refuse look_hold +
  resume if re-deriving is not faithful - say which in the plan)*.
- `check`/`retries` do not re-check kept segments; the first rendered
  segment is checked against the last kept one's seam as usual.
- The kept files are copied, not moved, into the new run's spill under its
  own naming, so the new take is self-contained and `keep_segments` names
  them on its shots; the originals stay where they were.
- The run logs one line "Resumed k of N segments from <file>" and the job
  carries a `chain_resumed` info-level warning with the same text so the
  caller sees it.
- A chain that fails after at least one segment was spilled now logs, and
  reports as a job warning, the first segment file's `output:` reference
  and that `resume_from` continues it. The builder must verify that a
  failed run's segment files are reachable through an `output:` reference
  (a failed job may write no manifest); if they are not, make the failed
  run write enough of a manifest for `output:` to resolve, and say so in
  the plan.
- Cost: validate's estimate for a chain with a resolvable `resume_from`
  prices N - k segments; an unresolvable one (the reference is a variable
  with no value yet) prices N and says so in the plan text.

- `minimax/chained-segments` sets `save_segments: true` (Don, 2026-11-08):
  the final output is unchanged, each finished segment is written beside
  the take while the chain runs and deleted at the end unless
  `keep_segments`, so a crash at any segment leaves something to resume
  from. The notes name it as a default change; the cost row's wall time is
  unchanged (the spill is already how `ltx2/chained-segments` runs).

**Surface.** Schema `$defs/chain.resume_from`; `ChainConfig.resume_from`;
a `resume_from` variable (default null) on `minimax/chained-segments` and
`ltx2/chained-segments` only; `docs/WORKFLOW_GUIDE.md` chain section; the
long-form skill's run-and-repair reference gets the "a chain that died at
segment k" recipe beside the repair recipe.

## Feature C: `frame_snap` derived from the step's constraint

Shipped as review defect 1 on `fix/release-review-0-20-0`: a
length-planned chain with no `frame_snap` whose `num_frames` is a variable
with a `modulus` constraint derives its snap from that constraint and logs
it; one with neither gets a validate warning. The LTX template also declares
`frame_snap: "constraint:num_frames"` explicitly. Nothing further here.

## Feature D: a delivery loudness target on the join

**What.** `match_levels: "lufs"` on `concat_videos` (and the other callers
of `joins.match_levels`: `join_into_song`, `dissolve_videos` if it takes
`match_levels`): each shot is gained to one integrated loudness (BS.1770,
`dw/dsp.py integrated_lufs`), the measure editors deliver to.

**Behaviour.**
- Target: `match_levels_lufs`, default -16 LUFS (streaming/web delivery;
  the docs name -23 as the EBU R128 broadcast figure). `match_levels_dbfs`
  with `"lufs"` is refused naming the right key.
- Ceiling: the gain is held where the shot's true peak would pass -1 dBTP
  (`dsp.true_peak_dbfs`), reported as the existing `match_levels_held`
  warning.
- A shot too short to measure (`integrated_lufs` returns None under 0.4 s)
  is matched by rms to the equivalent target and the join carries a
  warning naming the shot *(my call)*.
- The join's result records the per-shot gains as the rms path does, and
  the gallery metadata's `integrated_lufs` of the output is what a user
  checks the result against.

**Surface.** `LEVEL_MEASURES` gains "lufs"; `dw/task_domains.py` choice
list; `docs/TASKS.md` concat_videos section; the long-form skill's audio
reference states "deliver at -16 LUFS: `match_levels: lufs`". No template
default changes.

## Feature E: a narration script check

**What.** A task `check_narration`: given the script (the takes' text, or
the text itself) and the finished narration audio, transcribe the audio and
report where the speech departs from the script, per take.

**Why this shape.** A step cannot be switched on by a variable (the step
schema has no conditional key), and the narration templates do not save
their take clips. Checking the joined narration against the joined script
needs neither: it works on any narration, including one made before this
release, and runs only when asked *(my call, after the builder check)*.

**Behaviour.**
- Arguments: `takes` (list of strings, or one string that is split with
  `split_takes`' own splitter at the same defaults), `audio` (the narration:
  a path, `output:` or `previous_result:`), `model_name` (default
  `transcribe_audio`'s), `max_word_error_rate` (default 0.1), `device`.
- Normalise both texts (lower-case, punctuation stripped, digits kept),
  align the heard tokens to the concatenated script tokens with
  `difflib.SequenceMatcher`, and attribute each script token's outcome to
  the take that owns it. Per take: `index`, `expected_words`,
  `heard_words`, `word_error_rate`, `dropped`, `inserted`, `ok`; totals
  `checked`, `failed`, `worst`. Repeated words (a looped take) show as
  `inserted` on the take they fall in.
- Each failing take raises a `narration_take_mismatch` job warning naming
  the take index, its rate and the first few dropped words. The result is
  a JSON dict.
- The transcription reuses `dw/tasks/audio_transcription.py`'s loader and
  cache (import `model_source`, `required_weights`, `cached_model`; do not
  duplicate); `required_weights` names the ASR model for the plan.
- Unit tests mock the transcriber; no model download in the suite.

**Surface.** `dw/tasks/narration_check.py` (new module), registered like
`split_takes`; `docs/TASKS.md` section under *Splitting Text into Takes*;
a new template `templates/check-narration` (`text` or `takes`, `audio`,
`max_word_error_rate`) that saves `narration-check.json`; the long-form
skill's audio reference says to run it after `narration-long` and which
take to re-speak. The narration templates do not change.

**Out.** Re-rolling a failed take (a follow-up; the result carries the
indices a re-roll would need).

## Feature F: show the timeline on the job page

**What.** When a job's outputs include an `.otioz`, the job page draws the
timeline summary the gallery metadata route already returns (#1254):
tracks as rows, each clip a block whose width is its share of the
timeline, labelled with its name and seconds; gaps hatched; the member
files listed as downloads.

**Behaviour.**
- `JobResults` fetches gallery metadata for `.otioz` outputs as it does for
  images today, and renders a new `TimelineStrip.svelte` (in
  `ui/src/lib/job/`) from `media` when `media.kind == "timeline"`.
- Each clip shows a one-click copy of its source `output:` reference when
  the summary carries one - what `repair-segment`'s
  `chain_take`/`chain_output` take - since the UI has no argument prefill
  to link into *(my call)*. Today the bundle records no provenance (clip
  `metadata` is empty), so `export_timeline` writes each clip's source
  reference into its OTIO `metadata.dw.source` when it knows it (the
  exported video's own `output:` reference, and in `sources` mode each
  source's), and `summarize_bundle` surfaces it as `source` on the clip.
  Bundles exported before this show no copy button *(my call)*.
- Responsive: the strip scrolls horizontally under a minimum block width;
  keyboard-reachable copy buttons; works in both themes with the existing
  tokens. No new dependency.
- Types: a `TimelineSummary` type in `ui/src/lib/timeline.ts` with a
  vitest for the share/label maths; `GalleryMetadata.media` stays
  `dict[str, Any]` server-side (no OpenAPI change). A component test for
  the strip; an e2e only if a fixture `.otioz` is cheap to add.

**Surface.** `ui/CLAUDE.md`/`docs/SERVER.md` web UI section: one sentence.

## Left to the builder

- Exact names of helpers and where tests live, within the modules named.
- Feature B's look_hold re-derivation choice, and the check that a failed
  run's segment files resolve through `output:` (the running manifest is
  written at run start; whether it lists the spilled files is the question).
- Whether Feature D's short-shot fallback should be a refusal instead.
- Feature F's visual proportions and copy affordance.

## Order and merging

The defects branch merges first. The six features are independent and run
in parallel worktrees off that merge: A (server + MCP), B (chain), D
(joins), E (new task + templates), F (UI). Conflicts are expected only in
`docs/WORKFLOW_GUIDE.md`, `docs/TASKS.md`, the skill references and the
two chain templates (B and C both touch `ltx2/chained-segments.json`);
merge B after the defects branch and rebase.
