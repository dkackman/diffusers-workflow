# Findings and Assess in the gallery, and a job's event log (#1452)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the idea on 2026-11-18, from a UI support assessment of
v0.19.0..develop (0e85aabc). Two pieces of evidence the server already served had no
caller in `ui/src`: the assessment findings for an output (on the metadata, and from
the assess route) and a job's persisted event log. Plan v1 is the plan comment on #1452,
and it still holds the full sweep. This file records what was built, the deviations,
and what stays open.

## Verdict: build smaller (two UI-only stages)

- **Who gains:** Don in the browser. MCP sessions already had `assess_output` and
  `get_job_events`, so they gain nothing.
- **Corrections to the idea:**
  - The assess route is a **GET** with query params (`probe`, `detail`, `workspace`),
    not a POST.
  - A finished job keeps only its **last 200** events (`MAX_PERSISTED_EVENTS`), so the
    section is "Event log", not "Full event log", and it shows the route's `note`.
  - Metadata findings are only the level rules (`full_scale`, `near_silent`). Assess
    returns those and the probe findings, so its reply replaces the metadata list
    rather than adding to it.
- **Cut:** crop and texture bands on the compare page. `ComparePage.svelte` never calls
  `POST /api/gallery/compare`; it plays 2-4 files side by side. Adding them means a
  second, server-rendered view with that route's crop limits. See "Deferred".
- **Cheaper alternatives rejected:** doing nothing (ask a Claude session), and showing
  metadata findings only (the level rules, never the probes; it became stage A's first
  half).

## Plan history

- **v1 (2026-11-18).** Two UI-only stages, A (gallery) and B (job page). The advisor
  approved it on Don's behalf at 2026-11-19T05:10Z, in the batch lane, with every
  question's default. Adjustments, written into the stage bodies with no new version:
  - verify stage A on a non-H3 output on mini-ai (no H3, no lem);
  - stage B's note case needs a job the server actually trimmed;
  - stage B's log is its own component, so #1451's `JobPage.svelte` work only meets it
    at one mount point.

**Rulings (advisor, all defaults):**
- Q1 [none]: local narrowed types in `ui/src/lib/types.ts`, no server `response_model`
  (one would filter `probes`, which MCP reads under `detail`).
- Q2 [none]: the names `findings`, `api.assess`, `api.eventLog`, `fetchEventLog`, the
  labels "Findings" and "Event log", and a 5,000-event paging cap.

## What was built

### Stage A (#1477): findings and Assess in the gallery detail (UI)

Merged to `develop` at `fc1ad4bf` (commit `10691e8f`). UI only.

- **Metadata findings:** `GalleryPage.select()` keeps `r.findings` and passes it to
  `GalleryDetail`. `ui/src/lib/gallery/GalleryFindings.svelte` renders a "Findings"
  region: a severity chip (warn first, as the server orders them), the server's `says`,
  and `at` as m:ss.s when present. An empty list reads "No findings".
- **Assess button** in the detail header, before "open file". It calls `api.assess(name)`
  (`GET /api/gallery/{name}/assess`, workspace-scoped like every gallery call), reads
  "Assessing…" and is disabled while it runs, and replaces the metadata list with the
  reply. A muted line under the list names `rules applied: …` and each probe that
  didn't apply, with the server's reason.
- **Disabled on a still**, titled "a still has no shots, seams or soundtrack to
  measure" (the server's own words).
- **A failed Assess** shows the server's `detail` inline (`role="alert"`), and the
  earlier list stays.
- **Held for the page's life**, keyed by workspace + file name, never persisted.
- **Colours:** new tokens `--finding-warn` / `--finding-info` in all four theme blocks
  of `app.css`, AA on the panel in both themes, distinct from `--live` and `--select`.
- **Types:** `LevelFinding` (alias of the generated `Schemas['LevelFinding']`) and
  `AssessReply`, each commented with its route.
- **Tests:** vitest in `api.test.ts`, `GalleryDetail.test.ts` and `GalleryPage.test.ts`;
  playwright `ui/e2e/gallery-findings.spec.ts` in both themes.
- **Docs:** `SERVER.md` web UI Gallery section; `ARCHITECTURE.md` Level findings row.

**Deviations** (accepted at verify):
1. `tests/test_ui_twins.py` lists the assess route in `NOT_IN_CONTRACT`, with a
   comment. The twin guard fails any route `api.ts` calls that has no `response_model`,
   and Q1 chose local types.
2. Assess is also disabled for a non-media file, titled "not a video or audio file"
   (the server's other `not_applicable` reason).
3. The result cache lives in `GalleryPage`, so it also survives the panel closing, not
   only output switches.

**Bounces: 0.** The first verify on mini-ai (develop @ `6ef6ecf1`) passed C-F1088 to
C-F1094 and SE-F074. Assess matched `assess_output` on a near-silent wav (one
`near_silent` warn) and on a silent `stills_to_video` mp4 (no findings), both made with
CPU tasks in `qa-1452`. A disabled button can't be force-clicked by `ui-check`, so
C-F1090 and C-F1091 rest on the `disabled` attribute and the request log.

### Stage B (#1478): the event log on the job page (UI)

Merged to `develop` at `7e3067d7` (commit `d75d8cb0`). UI only.

- **Types and API:** `EventLogReply` in `types.ts`; `api.eventLog(id, {after, limit,
  kinds})`, with `kinds` sent as a repeated param.
- **Paging:** `fetchEventLog()` in `ui/src/lib/job/eventLog.ts`, pure over
  `api.eventLog`. It asks for 500 a page, follows `truncated` from each `last_seq`, and
  stops at 5,000 (`capped` only when the server still had more).
- **Section:** `ui/src/lib/job/EventLog.svelte`. `JobPage` gains only the import and one
  mount point after the Log panel. It is collapsed by default on every job, makes no
  request until opened, and never writes `JobPage`'s `events`, so the flow view and
  step chips are unchanged.
- **Rows:** seq, time (`+` the `at` duration), kind, and the message or the rest of the
  payload as compact JSON. A client-side kind select is built from the kinds present.
  The server's `note` shows above the rows. A 404 reads "no event log for this job".
- **Refresh** pages on from `after=last_seq`, offered only while the job is live.
- **Tests:** vitest in `api.test.ts`, `eventLog.test.ts` (follows `truncated`, caps at
  7,000 events, not capped at exactly 5,000) and `JobPage.test.ts`; playwright
  `ui/e2e/job-event-log.spec.ts` on a mocked trimmed log, both themes.
- **Docs:** a job-page Event log paragraph in `SERVER.md`'s Jobs bullet.

**Deviations** (accepted at verify):
1. A warning is filed under `warning · <kind>` (e.g. `warning · phase_stall`), so the
   filter separates a stall from a clip.
2. Each Refresh pages on for up to another 5,000, and the cap line names the count
   held: "showing the first N".
3. `tests/test_ui_twins.py` lists the event-log route in `NOT_IN_CONTRACT`, as stage A
   did for assess.

**Bounces: 0.** The first verify on mini-ai (develop @ `7e3067d7`) passed C-F1095,
C-F1097 to C-F1099, C-F1101 and SE-F075, with two arms short:
- **C-F1096's note arm** wasn't exercised on a real answer: the finished job used
  (74 events) has no note. SE-F075 rendered a mocked note as text.
- **C-F1100 arm (b)** (the 5,000 cap against paging mocks) wasn't run: `ui-check`'s
  static mocks can't make `seq` follow each request's `after`. Filed as
  dkackman/harnest#254. The cap is covered by `eventLog.test.ts`.

Stage costs aren't recorded, because the stage comments name no `usage:` figures. The
plan estimated about $6. Neither stage used GPU time; the fixtures were CPU tasks and
one small text-to-image job.

## Deferred, and what brings each back

- **Crop and texture bands on the compare page** (the cut). *Back if* Don asks to see
  the compare grid or texture-band table in the browser. It would be its own stage,
  about $5: a server-rendered `POST /api/gallery/compare` view honouring the route's
  crop limits (`media_compare.py`).
- **A real trimmed log on the job page.** The note was seen only mocked. Any finished
  job past 200 events shows it; check it the next time one is opened.
- **The cap arm of C-F1100**, once harnest#254 gives `ui-check` request-aware mocks.

## Known limits

- Local types can drift from the server. Each names its route and the vitest fixtures
  are copied from real replies, so a renamed key fails a test. A later `response_model`
  on either route can replace the local type and drop the `NOT_IN_CONTRACT` entry.
- Assess on a long video holds a server thread for seconds; the button allows one
  request at a time, the same cost an MCP caller pays.
- #1451 (per-segment chain view) also edits `JobPage.svelte`; it meets this feature
  only at the Event log mount point.
- Verified on mini-ai (MPS) only. Nothing here is accelerator-specific.
