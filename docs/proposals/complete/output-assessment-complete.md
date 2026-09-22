# Proposal: output assessment - let an agent measure, look at and listen to what it made

Status: proposed, 2026-09-20. Scopes issue #193 (agent has no way to hear or
watch generated output), folding in #210 (no `VideoContent` in the MCP SDK)
and the per-segment `analyze_cut` proposal from #193's thread. Written from
a design conversation with Don, every section agreed; revised the same day
after a review pass against the code (the *Review findings* section records
what changed and why). Stage 1 (evidence) is on branch `feat/193-evidence`
(3736b01), full suite green, pending merge to `develop`. Stages 2-4 not
started.

## The goal

An agent that drives `dw` over MCP should be able to assess the quality of a
generated deliverable and decide whether to redo or tweak it. Four use cases
drive the design:

1. **Lip-sync synchronisation** - is the speech where the mouth is.
2. **Seam transitions** - is each join in a cut clean, in the audio and in
   the picture.
3. **Continuity between clips** - does shot B look like it belongs after
   shot A.
4. **Audio/video contextual alignment** - does the sound swell where the
   picture does; does the score reach its climax with the cut.

Today the agent has `get_output_image` (stills only), `get_output_audio`
(audio only, refuses `video/mp4`), `get_gallery_metadata` (whole-file
duration/level, `envelope=true` for second-by-second rms/peak keyed to time
rather than to shots), `analyze_audio` (whole-track), `transcribe_audio`,
and `frame_grid` (#245 - a contact sheet, but only reachable by authoring a
workflow; no MCP tool wraps it, and the `minimax-h3` skill still says "you
cannot watch a video"). Nothing measures a seam, a drift, a continuity
break or an alignment, and nothing gets a frame *pair* at a boundary into
the agent's context. The issue's own motivating miss - a five-shot cut at
exactly the right duration whose audio drifted 267 samples per shot and was
33 ms late by the end - passed every check that exists.

## Decisions taken (with Don, 2026-09-20)

| Question | Decision |
|---|---|
| Where does judgement live? | **Tier 1: the engine measures, the agent judges.** Deterministic numbers plus evidence the agent can perceive (frames as `ImageContent`, audio excerpts). No ML judges in the engine. Tier 2 (CLIP/DINO similarity, SyncNet-class sync) and tier 3 (a VLM judge) are marked `TODO` where they would slot in, not built. |
| How does the agent invoke a measurement? | **Both paths, one implementation.** Measurements are engine tasks (composable into a template's own self-check step, recorded in the manifest, cacheable) *and* run synchronously in the server process over an MCP tool for an output or asset, so a question about a finished file does not queue behind a 20-minute render. |
| Where do shot boundaries come from? | **Persisted at join time, caller override.** The engine made the cut, so it records where the seams are; asking the agent to reconstruct that is the arithmetic that produced the 33 ms miss. |
| Lip sync at tier 1? | **Drift only.** Per-shot and cumulative audio-vs-video offset. Onset-frame evidence and motion correlation were considered and deferred with SyncNet to tier 2 - too weak to be worth teaching. |
| Who holds the thresholds? | **The engine emits `findings`** from one declared rules table, so an agent over bare MCP gets a place to look without the plugin; the skill teaches what to *do* about a finding. |
| Surface shape | **Narrow probes plus one verdict call.** `assess_output` runs every applicable probe and returns findings plus a one-line summary per probe; `inspect_output(probe=...)` returns one probe's full body. Fat verdict, narrow detail. |
| Are findings verdicts? | **No. The agent is the ultimate authority.** Nothing in the engine acts on a finding: no probe fails, no job fails, no template refuses to save, and `assess_output` never says pass or fail. A finding is a place to look. Severity is `info`/`warn` only - `error` was dropped because it reads like a verdict. |

## Vocabulary

The boundary list is called **`shots`** everywhere - on the artifact, in the
manifest, in the file tag, as the probe argument. Not `segments`: that word
already means chain *generation* segments throughout the engine
(`SegmentedFrames`, `save_segments`, `chain.py`'s `Segment`), and a grep
that finds both is a grep that finds neither. A **seam** is the boundary
between two adjacent shots; seam *i* sits between shot *i* and shot *i+1*.

## Section 1 - Probes (engine tasks, `dw/tasks/assess.py`)

All CPU-only (numpy + PyAV), read-only: the frames and waveform pass through
unmodified, and what comes back is diagnostics rather than an artifact, as
`analyze_audio` does. Each takes a video or audio in any shape a result
carries (a path, an `asset:`/`output:` reference, a `previous_result:`,
a generated `AudioVideo`), an optional `shots` list (section 3), and
returns `{...measurements, findings: [...], rules_applied: [...],
shots_source: "argument" | "manifest" | "file" | "none"}`.

**A probe over a path streams.** The existing loaders (`load_audio_video`,
`load_audio`) materialise every frame as a PIL image - a four-minute 1080p
clip is ~36 GB that way - and both route the track through
`_fit_audio_to_frames`, which snaps audio to the frame count within
`AUDIO_FIT_TOLERANCE_SECONDS` (0.25 s) and so *hides the very drift
`analyze_sync_drift` exists to see*. So a probe given a path decodes in one
`probe_media`-style pass (`dw/media_info.py`) that keeps only what the
rules need: the waveform (~46 MB for four minutes of 48 kHz stereo), one
motion scalar per frame, a fixed-size downsampled luma/Lab image per frame,
the frame pair either side of each seam, and the container's own stream
durations. A probe given an in-memory `AudioVideo` (the task-step path)
reads those from the object it has. Nothing ever holds "the decoded file".

| Probe | Answers | Measurements |
|---|---|---|
| `analyze_shots` | per-shot level - the "whole-file mean hides a 6.5 dB jump" case | per shot: `peak_dbfs`, `rms_dbfs`, `crest_db`, `low/mid/high_dbfs` (the `analyze_audio` numbers, per shot); `rms_range_db` across shots |
| `analyze_seams` | is each join itself clean, in audio and in picture | per seam, audio: `level_step_db` (rms of the 250 ms either side), `floor_dbfs` (rms of the 20 ms centred on the join - the "seam floor"), `click_db` (peak of the 2 ms window at the join over the peak of its neighbours), `spectral_shift` (band-share change across the join). Video: `frame_delta` (mean absolute difference between the last frame of A and the first of B, both downsampled to a fixed size) and `typical_delta` (the *90th percentile* of inter-frame difference inside each shot, floored - a median is near zero on a static talking head and would flag every cut after one), reported as `jump_ratio` |
| `analyze_sync_drift` | lip-sync drift (tier 1) | per shot: `offset_ms` at its start and end, read as `start_sample / rate - start_frame / fps` from the recorded boundaries, and `cumulative_ms`; whole file: `length_delta_ms` = the container's audio stream duration minus its video stream duration, *after subtracting the codec's declared priming* (AAC's 1024-sample block is 21 ms at 48 kHz and 43 ms at H3's 24 kHz - over the threshold on a clean file if left in). **TODO tier 2:** SyncNet / AV-HuBERT offset and confidence per shot |
| `analyze_av_alignment` | does the sound swell where the picture does | the audio rms envelope and the video motion envelope (mean inter-frame difference), both at 4 Hz and normalised; `correlation` (Pearson, whole file and per shot), `audio_peak_s`, `motion_peak_s`, `peak_offset_s`, `cut_density` (seams per 10 s, when there are shots) against the audio envelope |
| `analyze_continuity` | does shot B look like it belongs after shot A | for each adjacent pair, on the fixed-size downsampled frames: `color_distance` (Lab histogram distance, last frame of A vs first of B), `luma_step`, `sharpness_ratio` (Laplacian variance of B's first frame over A's last - catches "drift sharpening into noise late in a chain"; resolution- and quantisation-dependent, so only ever compared *within* one file, never against a constant). **TODO tier 2:** CLIP/DINO embedding similarity, face identity |

`analyze_seams` and `analyze_continuity` both look at the boundary frames;
the split is *cut hygiene* (is the join itself clean) versus *content* (does
B belong after A), kept separate so a deliberate hard cut between two
locations trips continuity and not seams.

`analyze_sync_drift`'s per-shot `offset_ms` measures the *recorded*
boundaries, so it is only as honest as the recording: section 3 requires
`start_sample` to be read from the joined waveform's actual length at each
seam, never derived from `start_frame / fps`, or the probe is a tautology.
`length_delta_ms` is the one number read from the file itself and is what
catches a file whose boundaries were never recorded.

A dissolve's shot B starts at the first blended frame, and its seam is at
`start_frame + overlap_frames / 2`; the seam probe widens its windows to
the whole fade rather than measuring a step in the middle of one.

`frame_grid` (#245) stays as it is - it is the evidence half of this.
`analyze_audio` stays as the single-track probe and is on the `inspect`
whitelist.

**A probe step saves as JSON.** A dict result is written whole only under
`content_type: application/json` (`Result.save`'s JSON branch); under any
other type the dict branch explodes it per key and a scalar raises. The
schema requires `application/json` on a step whose command is a probe, and
`scalar_result_validation` refuses anything else.

## Section 2 - Evidence tools (sight and sound into the agent's context)

The tools that let an agent look at a flagged finding rather than trust the
number. The MCP SDK has no `VideoContent` (#210); the ruling there stands:
frames go back as `ImageContent`, sound as the existing audio path.

- **`get_output_frames(name, at=None, seams=None, count=None,
  max_dimension=512, workspace=None)`** - new MCP tool in `dw_mcp/media.py`
  over `GET /api/gallery/{name:path}/frames`. Returns a list of images
  (base64, the `get_output_image` shape) each labelled with its time and
  frame index. Three selectors, mutually exclusive:
  - `at: [seconds | "frame:N", ...]` - specific moments.
  - `seams: true | [indices]` - for each seam, the last frame before and the
    first frame after, composed into *one* side-by-side tile per seam with
    the shot names burned in. This is the seam and continuity evidence in
    one image.
  - `count: N` - `frame_grid` reached without authoring a workflow: N evenly
    spaced frames as one contact sheet.

  Same `_encode_within_budget` path as `get_output_image`, capped at
  `MAX_RETURNED_BYTES` across the whole call; over budget it halves the
  dimension before it drops tiles, and the answer says which it did. The
  server decodes with PyAV, seeking to each timestamp rather than reading the
  clip through.

- **`get_output_audio` becomes mux-aware and takes an excerpt.** It moves
  from the raw `/outputs` mount to `GET /api/gallery/{name:path}/audio?start=&duration=`.
  On `video/mp4` the server extracts the track and returns it as WAV. New
  `start`/`duration` seconds pull the two seconds around seam 3 rather than
  refusing a four-minute cut; the answer carries `excerpt: {start, duration,
  of}` so it names itself as a cut (#204 objected to *silent* truncation).
  **The whole-file refusal stays**: without `start`/`duration` a file over
  `MAX_RETURNED_BYTES` base64 is refused exactly as today, and the refusal
  now says to ask for an excerpt. An audio-only file under budget with no
  excerpt asked for is returned as-is in its own encoding, not transcoded.

- **`get_gallery_metadata`** gains `media.shots` (section 3), so the
  existing `envelope` can be read against shot boundaries.

## Section 3 - Boundary persistence

Video files carry no embedded metadata today (only PNG/JPEG do; a video's
`get_gallery_metadata` comes from job history via the manifest). Three
layers, and a precedence:

- **The artifact carries it.** `AudioVideo` and a plain frames result gain
  an optional `shots` attribute: `[{name, start_frame, num_frames,
  start_sample, num_samples, overlap_frames?, hard_cut?}]`.
  - **Who populates it**: `concat_videos`, `dissolve_videos` and
    `run_chain` (`dw/pipeline_processors/chain.py` - the other join in the
    catalog, `templates/ltx2/chained-segments` and `extend-clip`, and the
    place late-chain sharpening actually happens; it already knows every
    seam from `Segment.head_trim` and `config.plan`).
  - **`start_sample` is measured, not derived.** The join task records each
    shot's `start_sample` from the length of the waveform it has actually
    accumulated when the shot is appended, so a per-shot codec overrun is
    *in* the record. This is what makes `analyze_sync_drift`'s `offset_ms`
    a measurement.
  - **Names are assigned in the workflow layer, not the task.** A task
    receives resolved objects and can only name a shot by position
    (`video_names` says "video 2"). `Workflow.run` renames the positional
    entries from the step's pre-resolution `gather:` reference list, the
    way `selected_field` already reads `step_data` - so a shot that came
    through `gather:shot` is `shot@intro`, and a finding says "between
    `shot@intro` and `shot@chase`" rather than "seam 2". A shot with no
    such origin keeps its index as its name.
  - **Every `AudioVideo(` constructor has a stated rule**, and a test fails
    when a new constructor appears without one:

    | Constructor | Rule |
    |---|---|
    | `concat_videos`, `dissolve_videos`, `run_chain` | *populate* |
    | `pair_audio` | frame side survives; `start_sample`/`num_samples` recomputed from the new track's rate, and with `fit: "video"` the last shot's `num_samples` is re-measured |
    | `interpolate_frames` | frame fields rescaled by the frame multiplier; sample fields unchanged |
    | `stabilize`, `loop_frames` (same frame count) | carried through unchanged |
    | `slice_audio`, `trim_*`, anything returning an `AudioTrack` or a different frame count | *dropped* - there are no frames to carry a boundary on, or the timeline changed |
    | `result.py`, `task.py`, `video_utils.py` wrappers | carried through unchanged |
- **The manifest records it.** The manifest entry grows `shots` beside
  `subfolder` when the artifact has them; `get_job`, `get_gallery_metadata`
  (`media.shots`) and the server-side probes read it from there. The
  realized `workflow.json` is untouched.
- **The file keeps it - by a remux pass after the write.** `result.py`
  does not own the container: `encode_video` and `export_to_video` are
  diffusers' (`diffusers.utils`), and neither takes metadata. So after
  either has written an mp4 whose artifact has `shots`, `result.py` opens
  the file, copies its packets to a sibling with
  `container.metadata["comment"] = json.dumps({"dw": {"shots": [...]}})`,
  and replaces it. The chain's spilled `SegmentedFrames` path gets the same
  pass. The tag round-trips through PyAV and survives `os.link`, so
  `keep_output` (a hard link), `upload_asset` and a pruned run directory
  all keep it, and `analyze_seams` on an `asset:` still knows the cut.
  `read_embedded_metadata` dispatches on extension before it reaches PIL;
  the metadata route folds the tag read into `probe_media` so a video is
  opened once per call, not twice.
- **Precedence** in every probe: the caller's `shots` argument > manifest
  > file tag > none. With none, the per-shot probes answer over one shot
  with `shots_source: "none"`, and `analyze_seams` reports zero seams - an
  answer, not an error.

`hard_cut` is a per-shot flag a template may set on an entry (a cut
between two locations by design), which suppresses `seam_frame_jump` for
that boundary. `concat_videos` gains a `hard_cuts` argument naming entries;
nothing sets it by default.

## Section 4 - Rules and `findings`

One table, `dw/assessment_rules.py`, in the `dw/task_domains.py` style: each
rule is `{name, probe, field, comparator, threshold, severity, says}`, and
`tests/test_assessment_rules.py` pins every rule to a real field of the
real probe's answer so a rename cannot leave one checking nothing. The
module docstring states the authority rule: **these are places to look, not
verdicts; nothing in the engine acts on a finding.** Starting thresholds,
each tunable in that one place:

| rule | probe / field | flags when | severity |
|---|---|---|---|
| `shot_level_spread` | `analyze_shots.rms_range_db` | > 3 dB | warn |
| `seam_level_step` | `analyze_seams.level_step_db` | > 3 dB | warn |
| `seam_click` | `analyze_seams.click_db` | > 12 dB | warn |
| `seam_hole` | `analyze_seams.floor_dbfs` | < −50 dBFS while both sides > −30 | warn |
| `seam_frame_jump` | `analyze_seams.jump_ratio` | > 8, unless the boundary is `hard_cut` | info |
| `sync_drift` | `analyze_sync_drift.offset_ms` at any shot end | > 40 ms | warn |
| `sync_length` | `analyze_sync_drift.length_delta_ms` (priming already subtracted) | > 40 ms | warn |
| `av_peak_offset` | `analyze_av_alignment.peak_offset_s` | > 2 s | info |
| `continuity_color` | `analyze_continuity.color_distance` | > 0.25 (normalised) | warn |
| `continuity_sharpness` | `analyze_continuity.sharpness_ratio` | < 0.5 or > 2 | warn |

`av_correlation` is **measured but has no rule yet**: on a 20 s five-shot
clip the 4 Hz envelopes are 80 samples, motion is per-cut and audio is
continuous, so a threshold set before the stage-4 field test would be a
coin flip. The number is reported; the rule lands from lem data.

A finding:

```json
{
  "rule": "seam_level_step",
  "severity": "warn",
  "at": {"seam": 2, "between": ["shot@intro", "shot@chase"], "seconds": 10.33},
  "value": 4.2,
  "threshold": 3.0,
  "says": "level steps 4.2 dB up across the join"
}
```

Findings are a new shape that *reads like* `job.warnings` (run-time warnings
carry a `kind`, not a severity); `severity` is `info` / `warn` only. Every
probe answer ends with `rules_applied` so an agent that got no findings
knows what *was* checked; `assess_output` adds `not_applicable: {probe:
why}` (no audio track, no shots, a still).

## Section 5 - API and MCP surface, and where the code runs

**Engine** - `dw/tasks/assess.py`, `dw/assessment_rules.py`. The five
probes are ordinary task commands, so a workflow can wire one as a
self-check step (by convention `intermediate`, `content_type:
application/json`) and its answer lands in the manifest. CPU-only; no torch
model is loaded.

**Server** - `dw/server/assess.py`, routes in `dw/server/app.py`, in the
existing `{name:path}/metadata|thumbnail|download` family:

- `GET /api/gallery/{name:path}/assess?detail=false` - runs every
  applicable probe *in the server process* (a sync `def`, so FastAPI runs
  it in the threadpool the way `metadata?envelope=true` runs today; the
  worker and its queue are never involved) and answers `{probes: {name:
  summary}, findings, not_applicable, shots_source}`; `detail=true`
  includes each probe's full body.
- `GET /api/gallery/{name:path}/inspect?probe=analyze_seams&...` - one
  probe's full body. `probe` is checked against the whitelist (the five
  probes plus `analyze_audio`) *before* any other query parameter is looked
  at, and only that probe's declared arguments are forwarded - a query
  string never reaches a task signature unfiltered. Anything else is a 400
  naming the list.
- `GET /api/gallery/{name:path}/frames?at=|seams=|count=&max_dimension=` -
  JSON list of base64 PNG tiles with labels.
- `GET /api/gallery/{name:path}/audio?start=&duration=` - mux-aware, WAV
  for an excerpt or an extracted track, the file's own bytes otherwise.
- All four accept an `asset:` name the way `metadata` does (#127), and
  `workspace`. Paths go through `validate_output_path` / the asset resolver
  as every other route does.
- One request, one streaming pass: `assess` decodes the file once and hands
  the reduced per-frame/per-sample data to every probe. Nothing is held
  across requests.

**MCP** - `dw_mcp/assess.py`, wired in `server.py`: `assess_output(name,
detail=False, workspace=None)`, `inspect_output(name, probe, workspace=None,
**arguments)`, `get_output_frames(...)`, and the widened `get_output_audio`.
None needs `acknowledged_cost` - nothing here spends GPU minutes.
`assess_output`'s description carries the authority rule in one sentence.

**Web UI** - reads the fields only: the job page lists a probe step's
findings under its result the way it lists warnings. Nothing else in the
first cut.

## Section 6 - Skill and discovery

### One skill, family-agnostic: `dw:judging-output`

A new plugin skill in `plugins/dw/skills/judging-output/`, named in
backticks in `plugins/dw/README.md` and `CLAUDE.md` as
`tests/test_plugin_skills.py` requires. It is the *procedure* for the
redo-or-tweak loop, and it is one skill rather than a section in each
family's because the loop is the same for an H3 cut and an LTX chain, and
because `minimax-h3` (12284 of 12288 bytes) and `ltx-2.5` (12254) cannot
absorb it.

Trigger line: use when a dw run has finished and the user wants to know
whether it is good, or when a family skill's "run and judge" step hands off
here.

Body, in order:

1. **Look before measuring**: `get_output_frames(count=12)` for the shape,
   `get_output_audio(start, duration)` around anything you mean to judge;
   what you can see and hear outranks any number below.
2. **Then `assess_output`**, and read the findings as places to look, not
   verdicts. One paragraph on each rule family: what the number is, what
   usually caused it in this catalog, and what it looks like when it is
   *fine* (a hard cut trips `seam_frame_jump`; a whispered line trips
   `shot_level_spread`).
3. **Drill down with `inspect_output(probe=...)` and
   `get_output_frames(seams=[n])`** on a flagged finding.
4. **Remediation table** - the part only the skill can hold, because it
   names template variables: a `seam_level_step` on an H3 cut →
   `match_levels_dbfs`; a `seam_hole` → `audio_bleed_ms`; a `sync_drift`
   → the #197 class, re-run through `pair_audio` `fit`; a
   `continuity_color` between two shots → regenerate B with A's last frame
   (`chain-video-continuity`) or the shared cast asset; `av_peak_offset` on
   a music video → re-slice the score (`start_frame`) or recut; a
   `continuity_sharpness` late in a chain → shorten the chain or reseed the
   drifting shot. Each row says whether the fix is a *recut* (cheap: the
   shots exist, `output:` them) or a *regenerate* (a new run, quote the
   cost).
5. **Recut, don't regenerate, when you can**: the `series-episodes` recipe
   for `output:`-referencing a run's `intermediate/` shots.
6. What tier 1 cannot tell you (identity across shots, whether the mouth
   matches the words) and that the frames are the way to check those - the
   honest limit, stated once.

The family skills' "run and judge" steps shrink to a pointer: `minimax-h3`
step 4 ("You cannot watch a video", ~700 bytes) is replaced by a shorter
step naming `get_output_frames` and handing off to `dw:judging-output` -
the replacement is budgeted to be shorter than what it replaces, since the
skill has four bytes of headroom; `series-episodes` "watch two episodes
back to back" points at the same place; `ltx-2.5` likewise. The
thresholds the new skill quotes are pinned to `dw/assessment_rules.py` by a
new test in the `test_plugin_skills.py` style (that file's existing pins are
to diffusers symbols; this is the same pattern against a different table).

### How the capability reaches an agent without the plugin

Three layers, each already the way this repo communicates a capability:

- **Tool descriptions** (`dw_mcp/server.py` docstrings) are the only thing
  a bare MCP client reads. `assess_output`'s says what it checks, that the
  findings are places to look, and that `inspect_output`, `get_output_frames`
  and `get_output_audio` are the drill-down. `get_gallery_metadata`'s
  description and its existing `next` hint point at `assess_output` for a
  cut ("whole-file numbers cannot see inside a join"). No hint is added to
  `get_job` - it has none today and #101's budget applies to every job
  read.
- **The guide** (`docs/WORKFLOW_GUIDE.md`, read over `get_guide`) gains a
  `## Assessing a run's output` section (top-level headings are what
  `list_guides` indexes), holding the probe table, the findings shape and
  the authority rule - so an agent told "check the seams" can look the
  mechanism up rather than guess at it.
- **The task catalog** (`list_tasks` / `get_task`): each probe's docstring
  is its `get_task` description. `list_tasks` has no categories today (three
  flat lists from `dw/introspection.py`, consumed by the web UI editor), so
  the five probes are listed under a new fourth list, `assessment`, rather
  than tagged - a new field the editor can ignore. `docs/TASKS.md` gets one
  `### <probe>` section each, as `analyze_audio` has.

The MCP server's own instructions block (the text in `dw_mcp/server.py` that
a client shows at connect) gains one sentence in the loop description:
"after a run, `assess_output` measures the deliverable and returns places to
look; `get_output_frames` and `get_output_audio` let you look and listen."

## Section 7 - Testing

- **Synthetic media, no models.** Every probe is tested on clips built in
  the test (numpy frames + a synthesised tone), the way `test_video_utils.py`
  tests `frame_grid`: a cut with a known 6 dB step at seam 2 must flag
  `seam_level_step` at seam 2 and nowhere else; a track padded by 267 samples
  per shot must report the drift the issue measured; a dissolve must not
  flag a step; a `hard_cut` boundary must not flag a jump; a static shot
  followed by a cut must not flag `seam_frame_jump` on the 90th-percentile
  baseline. Stage 3's tests pass `shots` as the argument, so they land
  without stage 2.
- **The rules table is pinned** (`tests/test_assessment_rules.py`): every
  rule names a field the probe actually returns; every threshold in the
  skill's prose matches the table.
- **Every `AudioVideo(` constructor has a rule**: a test enumerates the
  constructor sites and fails on one the table in section 3 does not name.
- **Boundary persistence round-trips**: `concat_videos` → save → manifest
  entry → `read_embedded_metadata` on the remuxed mp4 → the same list;
  `keep_output` keeps it; `pair_audio` recomputes the sample fields; a
  `slice_audio` drops it. `start_sample` on a synthetic cut whose shots
  each overrun by 267 samples records the overrun.
- **Memory**: `assess` on a long synthetic clip holds no per-frame image
  list (assert on peak RSS or on the streaming reader never being asked for
  a full frame list).
- **Budget**: `get_output_frames` on a 4K clip lands under
  `MAX_RETURNED_BYTES` and says it halved the dimension.
- **Routes**: the four `/api/gallery/{name}/...` routes refuse a traversal,
  answer 400 for an unknown probe naming the whitelist, forward only a
  probe's declared arguments, and accept an `asset:` name.
- **Field test on lem**: `assess_output` over MCP on an existing
  `music-video` and `dialogue-short` run, a `chained-segments` run, and the
  #197 repro cut, before the skill's threshold prose is finalised and before
  `av_correlation` gets a rule - the starting thresholds above are guesses
  until they have met real output.

## Not in scope

- Any model-based judge (tiers 2 and 3) - each `TODO` in section 1 is the
  slot.
- Automatic remediation: no template re-runs itself on a finding.
- `score-and-select` wiring: a probe's numbers are exactly what that
  proposal's reducer would read, but that proposal stands on its own.
- Web UI beyond listing findings.
- Findings on stills (composition, artifacts) - a different problem with a
  different evidence tool already in place (`get_output_image`).

## Staging

Four independently landable pieces, each its own branch off `develop`:

1. **Evidence**: mux-aware `get_output_audio` with excerpts;
   `get_output_frames` and its route. Closes the perceptual half of #193
   on its own.
2. **Boundaries**: `shots` on the artifact (every constructor ruled), the
   manifest, the remux tag; `media.shots` in metadata; the workflow-layer
   rename from `gather:`.
3. **Probes and rules**: the five tasks, the streaming reader, the table,
   `assess` / `inspect` routes and tools; guide section; `list_tasks`
   fourth list. Tests use the `shots` argument; a probe on a file with no
   boundaries answers `shots_source: "none"`.
4. **Skill**: `dw:judging-output`, the family-skill rewrites, the field
   test that sets the thresholds and decides `av_correlation`'s rule.

Release note items: `get_output_audio` no longer refuses `video/mp4` and
moved off the `/outputs` mount; manifest entries and mp4 files carry
`shots`; `list_tasks` has a fourth list; `minimax-h3` step 4 is rewritten.

## Review findings (2026-09-20)

A review pass against the code changed the design in these ways:

- `result.py` does not own the mp4 container (diffusers' `encode_video` /
  `export_to_video` write it), so the file tag is written by a post-write
  remux pass, not "at mux time".
- The existing loaders materialise every frame and snap audio to the frame
  count within 0.25 s, so probes stream and `analyze_sync_drift` reads
  stream durations from the container.
- A task cannot see `for_each` names; the workflow layer assigns them from
  the `gather:` reference list.
- `start_sample` must be measured from the accumulated waveform or the
  drift probe measures its own record.
- `run_chain` is the other join and populates `shots`; every `AudioVideo(`
  constructor has an enumerated rule.
- `segments` renamed `shots` to keep clear of chain segments.
- `sync_length` subtracts codec priming (43 ms at 24 kHz); `jump_ratio`
  uses a 90th-percentile floored baseline; `av_correlation` has no rule
  until lem data; `get_output_audio` keeps its whole-file refusal; probe
  steps require `application/json`; `list_tasks` gets a fourth list rather
  than a tag; no `next` hint on `get_job`.
