# Proposal: output assessment - let an agent measure, look at and listen to what it made

Status: proposed, 2026-09-20. Scopes issue #193 (agent has no way to hear or
watch generated output), folding in #210 (no `VideoContent` in the MCP SDK)
and the per-segment `analyze_cut` proposal from #193's thread. Written from
a design conversation with Don, every section agreed. No code changes yet.

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
| Lip sync at tier 1? | **Drift only.** Per-segment and cumulative audio-vs-video length. Onset-frame evidence and motion correlation were considered and deferred with SyncNet to tier 2 - too weak to be worth teaching. |
| Who holds the thresholds? | **The engine emits `findings`** from one declared rules table, so an agent over bare MCP gets a place to look without the plugin; the skill teaches what to *do* about a finding. |
| Surface shape | **Narrow probes plus one verdict call.** `assess_output` runs every applicable probe and returns findings plus a one-line summary per probe; `inspect_output(probe=...)` returns one probe's full body. Fat verdict, narrow detail. |
| Are findings verdicts? | **No. The agent is the ultimate authority.** Nothing in the engine acts on a finding: no probe fails, no job fails, no template refuses to save, and `assess_output` never says pass or fail. A finding is a place to look. Severity is `info`/`warn` only - `error` was dropped because it reads like a verdict. |

## Section 1 - Probes (engine tasks, `dw/tasks/assess.py`)

All CPU-only (numpy + PyAV), read-only: the frames and waveform pass through
unmodified, and what comes back is diagnostics rather than an artifact, as
`analyze_audio` does. Each takes a video or audio in any shape a result
carries (a path, an `asset:`/`output:` reference, a `previous_result:`,
a generated `AudioVideo`), an optional `segments` list (section 3), and
returns `{...measurements, findings: [...], rules_applied: [...],
segments_source: "argument" | "manifest" | "file" | "none"}`.

| Probe | Answers | Measurements |
|---|---|---|
| `analyze_segments` | per-shot level - the "whole-file mean hides a 6.5 dB jump" case | per segment: `peak_dbfs`, `rms_dbfs`, `crest_db`, `low/mid/high_dbfs` (the `analyze_audio` numbers, per segment); `rms_range_db` across segments |
| `analyze_seams` | is each join itself clean, in audio and in picture | per seam, audio: `level_step_db` (rms of the 250 ms either side), `floor_dbfs` (rms of the 20 ms centred on the join - the "seam floor"), `click_db` (peak of the 2 ms window at the join over the peak of its neighbours), `spectral_shift` (band-share change across the join). Video: `frame_delta` (mean absolute pixel difference between the last frame of A and the first of B) and `typical_delta` (median inter-frame difference inside each shot), reported as the ratio `jump_ratio`; `luma_step`; `color_step` (mean Lab distance) |
| `analyze_sync_drift` | lip-sync drift (tier 1) | per segment: `audio_seconds`, `video_seconds`, `offset_ms` at the segment's start and end, `cumulative_ms`; whole file: `length_delta_ms` (audio minus video). **TODO tier 2:** SyncNet / AV-HuBERT offset and confidence per segment |
| `analyze_av_alignment` | does the sound swell where the picture does | the audio rms envelope and the video motion envelope (mean inter-frame difference), both at 4 Hz and normalised; `correlation` (Pearson, whole file and per segment), `audio_peak_s`, `motion_peak_s`, `peak_offset_s`, `cut_density` (seams per 10 s, when there are segments) against the audio envelope |
| `analyze_continuity` | does shot B look like it belongs after shot A | for each adjacent pair: `color_distance` (Lab histogram distance, last frame of A vs first of B), `luma_step`, `sharpness_ratio` (Laplacian variance of B's first frame over A's last - catches "drift sharpening into noise late in a chain"). **TODO tier 2:** CLIP/DINO embedding similarity, face identity |

`analyze_seams` and `analyze_continuity` both look at the boundary frames;
the split is *cut hygiene* (is the join itself clean) versus *content* (does
B belong after A), kept separate so a deliberate hard cut between two
locations trips continuity and not seams.

A dissolve's seam is the midpoint of the overlap, and the segment entry
carries `overlap_frames`, so the seam probe widens its windows to the whole
fade rather than measuring a step in the middle of one.

`frame_grid` (#245) stays as it is - it is the evidence half of this.
`analyze_audio` stays as the single-track probe and is on the `inspect`
whitelist.

## Section 2 - Evidence tools (sight and sound into the agent's context)

The tools that let an agent look at a flagged finding rather than trust the
number. The MCP SDK has no `VideoContent` (#210); the ruling there stands:
frames go back as `ImageContent`, sound as the existing audio path.

- **`get_output_frames(name, at=None, seams=None, count=None,
  max_dimension=512, workspace=None)`** - new MCP tool in `dw_mcp/media.py`
  over `GET /api/gallery/{name}/frames`. Returns a list of images (base64,
  the `get_output_image` shape) each labelled with its time and frame index.
  Three selectors, mutually exclusive:
  - `at: [seconds | "frame:N", ...]` - specific moments.
  - `seams: true | [indices]` - for each seam, the last frame before and the
    first frame after, composed into *one* side-by-side tile per seam with
    the segment names burned in. This is the seam and continuity evidence in
    one image.
  - `count: N` - `frame_grid` reached without authoring a workflow: N evenly
    spaced frames as one contact sheet.

  Same `_encode_within_budget` path as `get_output_image`, capped at
  `MAX_RETURNED_BYTES` across the whole call; over budget it halves the
  dimension before it drops tiles, and the answer says which it did. The
  server decodes with PyAV, seeking to each timestamp rather than reading the
  clip through.

- **`get_output_audio` becomes mux-aware and takes an excerpt.** On
  `video/mp4` it extracts the track (`load_audio` already special-cases video
  extensions) and returns it as WAV. New `start`/`duration` seconds pull the
  two seconds around seam 3 rather than refusing a four-minute cut; the
  answer carries `excerpt: {start, duration, of}` so it names itself as a
  cut. (#204 objected to *silent* truncation; an excerpt that says what it
  is answers a different question.) Server: `GET
  /api/gallery/{name}/audio?start=&duration=`.

- **`get_gallery_metadata`** gains `media.segments` (section 3), so the
  existing `envelope` can be read against shot boundaries.

## Section 3 - Boundary persistence

Video files carry no embedded metadata today (only PNG/JPEG do; a video's
`get_gallery_metadata` comes from job history via the manifest). Three
layers, and a precedence:

- **The artifact carries it.** `AudioVideo` and a plain frames result gain
  an optional `segments` attribute: `[{name, start_frame, num_frames,
  start_sample, num_samples, overlap_frames?, hard_cut?}]`. `concat_videos`
  and `dissolve_videos` populate it from their inputs. A segment's `name` is
  the input's `for_each` entry name when it came through `gather:` /
  `shot@x`, else its index - which is what makes a finding say "between
  `shot@intro` and `shot@chase`" rather than "seam 2". `pair_audio`,
  `slice_audio`, `trim`/`loop` tasks and `_fit_audio_to_frames` carry the
  list through when the frame timeline is unchanged and *drop* it when they
  change it - never silently wrong.
- **The manifest records it.** The manifest entry grows `segments` beside
  `subfolder` when the artifact has them; `get_job`, `get_gallery_metadata`
  (`media.segments`) and the server-side probes read it from there. The
  realized `workflow.json` is untouched.
- **The file keeps it.** At mux time PyAV writes the same list into the mp4
  container's `comment` tag as JSON (`{"dw": {"segments": [...]}}`), and
  `read_embedded_metadata` grows a video branch. That is what survives
  `keep_output` (a hard link), `upload_asset` and a pruned run directory,
  so `analyze_seams` on an `asset:` still knows the cut.
- **Precedence** in every probe: the caller's `segments` argument >
  manifest > file tag > none. With none, the per-segment probes answer over
  one segment with `segments_source: "none"`, and `analyze_seams` reports
  zero seams - an answer, not an error.

`hard_cut` is a per-segment flag a template may set on an entry (a cut
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
| `segment_level_spread` | `analyze_segments.rms_range_db` | > 3 dB | warn |
| `seam_level_step` | `analyze_seams.level_step_db` | > 3 dB | warn |
| `seam_click` | `analyze_seams.click_db` | > 12 dB | warn |
| `seam_hole` | `analyze_seams.floor_dbfs` | < −50 dBFS while both sides > −30 | warn |
| `seam_frame_jump` | `analyze_seams.jump_ratio` | > 8, unless the boundary is `hard_cut` | info |
| `sync_drift` | `analyze_sync_drift.offset_ms` at any segment end | > 40 ms | warn |
| `sync_length` | `analyze_sync_drift.length_delta_ms` | > 40 ms | warn |
| `av_peak_offset` | `analyze_av_alignment.peak_offset_s` | > 2 s | info |
| `av_correlation` | `analyze_av_alignment.correlation` | < 0.2 | info |
| `continuity_color` | `analyze_continuity.color_distance` | > 0.25 (normalised) | warn |
| `continuity_sharpness` | `analyze_continuity.sharpness_ratio` | < 0.5 or > 2 | warn |

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

Severity is `info` / `warn`, the run-time warnings' vocabulary minus
`error`, so it reads like `job.warnings`. Every probe answer ends with
`rules_applied` so an agent that got no findings knows what *was* checked;
`assess_output` adds `not_applicable: {probe: why}` (no audio track, no
segments, a still).

## Section 5 - API and MCP surface, and where the code runs

**Engine** - `dw/tasks/assess.py`, `dw/assessment_rules.py`. The five
probes are ordinary task commands, so a workflow can wire one as a
self-check step (by convention `intermediate`) and its answer lands in the
manifest as `application/json`, the way `analyze_audio`'s does. CPU-only; no
torch model is loaded.

**Server** - `dw/server/assess.py`, routes in `dw/server/app.py`:

- `GET /api/gallery/{name}/assess?detail=false` - runs every applicable
  probe *in the server process* (not the worker: no queue, no GPU, the same
  place `metadata?envelope=true` runs today) and answers `{probes: {name:
  summary}, findings, not_applicable, segments_source}`; `detail=true`
  includes each probe's full body.
- `GET /api/gallery/{name}/inspect?probe=analyze_seams&...` - one probe's
  full body, arguments as query parameters. The whitelist is the five
  probes plus `analyze_audio`; anything else is a 400 naming the list.
- `GET /api/gallery/{name}/frames?at=|seams=|count=&max_dimension=` -
  JSON list of base64 PNG tiles with labels.
- `GET /api/gallery/{name}/audio?start=&duration=` - WAV, mux-aware.
- All four accept an `asset:` name the way `metadata` does (#127), and
  `workspace`. Paths go through `validate_output_path` / the asset resolver
  as every other route does.
- A decoded file is cached per `(path, mtime)` for the life of one request
  only - five probes must not decode a four-minute cut five times, and
  nothing is held across calls.

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

A new plugin skill in `plugins/dw/skills/judging-output/`, listed in
`plugins/dw/README.md` and pinned by `tests/test_plugin_skills.py` like the
others. It is the *procedure* for the redo-or-tweak loop, and it is one
skill rather than a section in each family's because the loop is the same
for an H3 cut and an LTX chain, and because `minimax-h3` and `ltx-2.5` are
each within 40 bytes of the 12 KiB skill cap and cannot absorb it.

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
   `segment_level_spread`).
3. **Drill down with `inspect_output(probe=...)` and `get_output_frames(seams=[n])`** on a flagged finding.
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
step 4 ("You cannot watch a video") is rewritten to name
`get_output_frames` and hand off to `dw:judging-output`; `series-episodes`
"watch two episodes back to back" points at the same place;
`ltx-2.5` likewise. That rewrite frees bytes rather than spending them.

### How the capability reaches an agent without the plugin

Three layers, each already the way this repo communicates a capability:

- **Tool descriptions** (`dw_mcp/server.py` docstrings) are the only thing
  a bare MCP client reads. `assess_output`'s says what it checks, that the
  findings are places to look, and that `inspect_output`, `get_output_frames`
  and `get_output_audio` are the drill-down. `get_gallery_metadata`'s
  description points at `assess_output` for a cut ("whole-file numbers
  cannot see inside a join"). `get_job`'s `next` hint for a finished
  video job names `assess_output`.
- **The guide** (`docs/WORKFLOW_GUIDE.md`, read over `get_guide`) gains a
  section *Assessing a run's output*, indexed like the others, holding the
  probe table, the findings shape and the authority rule - so an agent that
  was told "check the seams" can look the mechanism up rather than guess at
  it. `list_guides` is what makes a section findable by name.
- **The task catalog** (`list_tasks` / `get_task`): each probe's docstring is
  its `get_task` description, and `list_tasks` tags the five with a
  `assessment` category so they sort together. `docs/TASKS.md` gets one
  section per probe as `analyze_audio` has.

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
  flag a step; a `hard_cut` boundary must not flag a jump.
- **The rules table is pinned** (`tests/test_assessment_rules.py`): every
  rule names a field the probe actually returns; every threshold in the
  skill's prose matches the table (the `test_plugin_skills.py` pattern).
- **Boundary persistence round-trips**: `concat_videos` → save → manifest
  entry → `read_embedded_metadata` on the mp4 → the same list; `keep_output`
  keeps it; a `slice` that changes the timeline drops it.
- **Budget**: `get_output_frames` on a 4K clip lands under
  `MAX_RETURNED_BYTES` and says it halved the dimension.
- **Routes**: the four `/api/gallery/{name}/...` routes refuse a traversal,
  answer 400 for an unknown probe naming the whitelist, and accept an
  `asset:` name.
- **Field test on lem**: `assess_output` over MCP on an existing
  `music-video` and `dialogue-short` run, and on the #197 repro cut, before
  the skill's threshold prose is finalised - the starting thresholds above
  are guesses until they have met real output.

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
2. **Boundaries**: `segments` on the artifact, the manifest, the mp4 tag;
   `media.segments` in metadata.
3. **Probes and rules**: the five tasks, the table, `assess` / `inspect`
   routes and tools; guide section; `list_tasks` category.
4. **Skill**: `dw:judging-output`, the family-skill rewrites, the field
   test that sets the thresholds.

Release note items: `get_output_audio` no longer refuses `video/mp4`;
manifest entries and mp4 files carry `segments`; `minimax-h3` step 4 is
rewritten.
