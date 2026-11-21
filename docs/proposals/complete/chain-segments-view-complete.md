# A per-segment view of a chain: measures, a segment table, repair and continue (#1451)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the idea on 2026-11-18, from a UI support assessment of
v0.19.0..develop (0e85aabc). Plan v2 is the plan comment on #1451 and holds the full
sweep. This file records what was built, the deviations, and what stays open.

## Verdict: build smaller (three stages, A → B → C)

- **Who gains:**
  - MCP sessions driving chains. Before this, a passing segment recorded only
    `attempts`, `face_drop` and `words`. Its jump, look, cut and texture ratios, and
    every ceiling, were computed and then dropped.
  - Don in the browser. The job page showed one chip parsed from phase text, and a
    finished job showed nothing.
- **Corrections to the idea:**
  - The data had to be on the manifest, not in an event: a finished job keeps 200
    events and its SSE stream is empty.
  - A chain that died records no shots, so it can't be resumed from the page.
  - No still is written per segment. The thumbnail is the segment's mp4 or the take
    seeked to the segment's start.
- **Cut (Q2):** live rows while a chain runs, resuming a died chain, and "resume from
  here" on an arbitrary row. Only **Continue the chain** was kept.
- **Rejected alternative:** a structured `chain_segment` event (see the first
  correction).

## Plan history

- **v1 (2026-11-18).** Three stages. The advisor approved it on Don's behalf at
  2026-11-19T05:10Z, with every question's default and four adjustments:
  - acceptance runs on mini-ai with LTX-2.5, no H3;
  - Repair is enabled only for the H3 family (the only `repair-segment` template is
    `templates/minimax/repair-segment`);
  - Continue needs the workflow to declare variables for both `stop_after` and
    `resume_from`;
  - `audio_dbfs` stays a top-level shot key and is mirrored into `measures`.
- **v2 (2026-11-19).** Wrote the ruling into the plan. No scope change.

**Rulings (advisor, all defaults):** Q1 [default] `measures` on every checked shot;
Q2 [scope] cut as listed; Q3 [scope] the tester verifies B and C with vitest,
playwright, both themes' screenshots and a real job page; Q4 [none] names as proposed.

## What was built

### Stage A (#1474): `measures` on chain shot records (engine + REST schema)

Commit `6fde2e5d`, merged to `develop` at `34949ebf`.

- `chain_check.measures_record`, called from `chain.py` `_record_check`, writes
  `shot["measures"]` on every checked segment, from the kept attempt, pass or fail.
  Each entry is `{value, ceiling, bound, warn_only, failed}`:
  - the ceiling is the `SegmentCheck` value in force (so an override shows), or `null`
    when the measure has none;
  - `warn_only` comes from `WARN_ONLY` (texture and face_drop);
  - `failed` is the check's own verdict, not a fresh comparison;
  - `words` is its similarity against `words_similarity_min`, `bound: "min"`;
  - `audio_dbfs` stays top-level and is mirrored with `bound: "min"`.
- A chain with the check off gets no key. Segment 1 carries only what it measured
  (`{audio_dbfs}` or `{}`).
- `dw/server/api_models.py` types `ManifestEntry.shots` as `list[ShotRecord]`
  (`extra="allow"`, with `ShotMeasure`), regenerated into `ui/src/lib/generated/`.
- MCP is unchanged: `get_job` and `get_gallery_metadata` pass the key through.
- The architecture review passed: `measures_record` lives in `chain_check.py`, the
  module the map already names as the owner.

### Stage B (#1475): the segment table on the job page (UI, read-only)

Commit `0d27adec`, merged to `develop` at `249ce1b8`.

- `ui/src/lib/job/ChainSegments.svelte`, mounted once on `JobPage.svelte` after
  Results. The pure rows are in `ui/src/lib/chainRows.ts`. A step is chained when its
  shots carry `planned_segments`.
- Each row has:
  - a lazy thumbnail (the `segment_file`, otherwise the take at `start_frame / fps`);
  - attempts "n of m", plus "kept seed N";
  - one cell per measure (`value ≤ ceiling`, `≥` for a min bound, or "(no ceiling)"),
    with a pass, warn or fail verdict read from the engine's fields;
  - `resumed` and `(repaired)` marks.
- The header reads "k of N planned". Unknown measure names render generically, after
  the known ones.
- **Deviations:**
  - "m" and the fps come from the job's definition and arguments, not the shot. When
    they're missing, the cell reads "n attempts" and the seek uses the segment's share
    of the take.
  - Segment 1 with `{}` reads "nothing measured", and a job before stage A reads
    "not recorded".

### Stage C (#1476): Repair and Continue (UI over existing routes)

Commit `9b7a940e` (merge `8e50f369`), with the bounce fix `fd59627d` (merge
`adedc601`).

- **Repair this segment** (`ui/src/lib/chainActions.ts` `repairAction`):
  - **Family rule:** the chain step's `model_name` (a literal, or a `variable:`
    resolved against the job) must equal a `model_name` that
    `templates/minimax/repair-segment` loads. The template is read live, so no
    family is hard-coded.
  - Disabled reasons, in this order: no template for the family; segment 1; a
    repaired take; segments not kept.
  - Enabled, it writes a one-shot `dw-workflow-prefill` to sessionStorage and opens
    the template with `chain_take` and `segment` filled. Nothing runs until Run.
- **Continue the chain** (`continueAction`), shown once above the table when the take
  is short of its plan, every row has a segment file, and `stop_after` and
  `resume_from` are both declared variables. It prices the rerun with `POST
  /api/validate?sizes=false`, confirms with the estimate, then POSTs
  `/api/jobs/{id}/rerun` with the last segment as `resume_from` and `stop_after`
  null. It isn't limited to H3.

## Verification and bounces

Everything was verified on mini-ai (MPS) with the tester's `chain-1451` fixture, an
LTX-2.5 chain at 256², 25 frames, 3 segments, `jump_ratio_max` 6.5. No H3 was run.
The enabled Repair path was checked on a mocked H3 job.

| Stage | Cases | Bounces | Notes |
|---|---|---|---|
| A #1474 | C-F1073 to C-F1078 | 0 | Parked once: C-F1078's REST/OpenAPI clause couldn't be run with the tester's tools. dkackman/harnest#253 made it MCP-only, and then it passed. |
| B #1475 | C-F1079 to C-F1082, SE-F073 part 1 | 0 | C-F1080's "no `cut_ratio` cell" contradicted C-F1079 and C-F1082; amended in dkackman/harnest#259. SE-F073 part 2 was retargeted to #1476 (dkackman/harnest#260). |
| C #1476 | C-F1083 to C-F1087 | 1 | C-F1086: the Continue confirm had no estimate. The plan cited a "confirm-and-estimate flow", but Run again has none, so the fix prices the rerun through validate. |

No stage comment reported a `usage:` figure, so cost is not recorded. The plan's
estimate was about $13 ($4, $5, $4).

## Deferred, and what brings each back

- **Live rows during a running chain:** if Don asks. It needs a per-segment event with
  path redaction and relative paths.
- **Resuming a chain that died:** if the engine writes a partial manifest on failure.
- **Resume from an arbitrary row:** if a field report asks.
- **Repair for other families:** when a `repair-segment` template exists for one. The
  family rule reads the template, so Repair enables with no UI change.
- **Repair of two adjacent segments** (#1231): a later row action.
- **Per-shot audio drift** (#1442): any new `measures` entry renders by name.
