# Run again with changes, preset promotions and the overrides chip (#1448)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the idea on 2026-11-18, from a UI support assessment of
v0.19.0..develop. `POST /api/jobs/{id}/rerun` already took `arguments` (overrides
merged over the original's) and recorded them as `overrides`, and MCP's `rerun_job`
used them for the draft→standard promotion. The web UI offered only **Run again** and
**New seed**, and nothing rendered `job.overrides`. Plan v2 is the plan comment on
#1448, and it still holds the full sweep. This file records what was built, what was
cut and what stays.

## Verdict: build smaller (one UI-only stage)

- **The case is parity.** The browser was the only client that couldn't rerun with
  changes. Before this, promoting a draft from the browser meant retyping every
  variable on the workflow page, which lost `rerun_of`, `overrides` and the kept seed.
  Demand evidence was thin: 6 of the last 200 jobs were reruns, 5 of them in harness
  workspaces.
- **Correction to the issue:** the jobs-list chip was not UI-only. `Job.summary()` and
  the history list query never fill `JobSummary.overrides` (only `detail()` does), and
  live/history summary parity is pinned. That half was stage B, and it was cut.
- **Correction to the issue:** the job page doesn't know the presets.
  `GET /api/jobs/{id}/workflow` strips `presets` because the bundles are already folded
  in. So the promotions fetch the workflow file as it is now, by name.
- **Cheaper alternatives rejected:** doing nothing (loses the provenance and the seed),
  and promotion buttons with no form (`VariablesForm` override mode already did the
  form, so leaving it out saved little).

## Plan history

- **v1 (2026-11-18).** Two stages: A (UI) and B (the jobs-list chip title, a small
  server change). The advisor approved it on Don's behalf at 2026-11-19T04:49Z for
  stage A only, with three minor adjustments:
  - no H3/minimax job admitted on mini-ai, not even to cancel it. Promotion is checked
    with a tiny SD 1.5 workflow with a `size` selector in the tester's own workspace;
  - a test that a numeric `30` typed over a run value of `30` sends nothing;
  - the overrides chip is its own self-contained element, so #1449 and #1458 can add
    siblings beside it.
- **v2 (2026-11-19).** Folded in the ruling. Stage B removed, no other scope change.

**Rulings (advisor):**
- Q1 [scope]: cut stage B. It is refiled after #1449 stage A if wanted.
- Q2 [none]: no confirm. A promotion queues on click, and its title names the change
  and says the value comes from the workflow file as it is now.

## What was built

### Stage A (#1471): rerun with changes, preset promotions, overrides chip

Merged to `develop` at `fa7141a2` (commit `0db9f94c`). UI only: no engine, MCP, REST or
syntax change, and the architecture review passed it.

- **`api.rerunJob(id, { newSeed, arguments })`** sends `arguments` only when there are
  any. **Run again** and **New seed** send their old bodies, `{"new_seed":false}` and
  `{"new_seed":true}`, with no `acknowledged_cost`.
- **"Run again with changes…"** (`ui/src/lib/job/RerunWithChanges.svelte`), on the
  job page of a finished job. It opens a modal with `VariablesForm` in override mode
  over every declared variable, ids prefixed `rerun-var-`, and this run's values
  (its arguments over the declared defaults) as placeholders.
  - A field counts as changed when its trimmed text is non-empty and differs from
    the run's displayed value. Only changed fields are sent, so a typed `30` over `30`
    sends nothing.
  - **Queue** is disabled and the form says "nothing changed" until something
    differs.
  - A blank seed keeps the drawn one. A typed seed is an override. The form never
    sends null.
  - A server 400 shows in the modal (`role=alert`) and the modal stays open. On
    success the page opens the new job.
- **Preset promotions.** One button per *other* value of each preset selector that the
  run declares (for example `size: medium`). A click requeues with only
  `{selector: value}` and no confirm. The title says the bundle comes from the workflow
  file as it is now, and names any bundle key the original set explicitly ("keeps
  width=640 set explicitly"), since an explicit argument beats the bundle.
- **Overrides chip.** `changed: size, steps` beside "rerun of <id>", in muted `<code>`,
  titled with one `name = value` line per override. It is absent when `overrides` is
  absent or empty.
- **Tests:** `RerunWithChanges.test.ts` (11), the chip in `JobPage.test.ts` (2), the
  request body in `api.test.ts` (4). `JobsPage.test.ts` is unchanged and green.
- **Docs:** `docs/SERVER.md`, `README.md`, and the "A different result" row of
  `docs/ARCHITECTURE.md`.

**Deviations from the plan** (all accepted at verify):
1. Promotions look the workflow up by `job.workflow_name` (the catalog name), not
   `job.workflow` (a display name). A job from an inline definition (`workflow_name`
   null), or one whose workflow can no longer be fetched, has no promotions. The form
   still works for both.
2. That lookup lives inside `RerunWithChanges`, not `JobPage`.
3. The run's current selector value is left out of the promotions, rather than shown
   as a pressed button.

**Bounces (1 failure, 1 hold).**
- **Verify 1, hold.** No code failure, but several cases couldn't be observed with the
  tester's `ui-check` runner (rerun POST stubs, `title` attributes, GET status mocks,
  duplicate ids). The advisor handed it back once `ui-check` @ `3485d32` closed those
  gaps.
- **Verify 2, failed C-F1056's seed clause.** A plain **Run again** drew a fresh seed.
  The UI sent exactly `{"new_seed":false}`, and an MCP `rerun_job` with no arguments
  did the same, so the fault was the server's rerun path. It was filed as #1492 and
  fixed in `37f8ee1e` ("a plain rerun keeps the original's drawn seed"). The stage
  went back to the tester with no code change. Sending the seed as an override from the
  UI would have been wrong, because it would put `seed` in `overrides`.
- **Verify 3 (develop @ `37f8ee1e`), passed** C-F1056. C-F1053 to C-F1063 and SE-F072
  had passed in the earlier passes.

Stage costs aren't recorded: the stage comments name no `usage:` figures. The plan
estimated $4–6. GPU use was a handful of 256 px, 4-step SD 1.5 images on mini-ai.

## Cut, and what brings each back

- **The overrides on the jobs list** (stage B, Q1). It needs `Job.summary()` and the
  history list query to fill `JobSummary.overrides`, which moves the pinned parity
  test. If Don wants it, it is filed as its own issue after #1449 stage A, which edits
  the same `JobSummary`/`recent_summaries`/parity test.
- **Sending null** (for example "draw a fresh seed" as an override). That stays the
  **New seed** button. The server can't take null as "unset", so a promotion keeps
  explicitly-set bundle keys rather than clearing them.

## Known limits

- **Promotion values come from the file as it is now.** If the workflow changed since
  the run, a value may be gone, and the server's refusal (with the allowed list) shows
  in the modal. The button title says so.
- **A typed number for a null-default variable stays a string**
  (`dw/variables.py`). This matches the workflow page.
- **The modal isn't passed `presets` or `constraints`** (#1447's props), so its fields
  stay plain text, with no bundle dropdown or rule hints.
- **#1483** (#1458 stage 3, the Requests UI) was blocked on this issue. It edits the
  same `JobHeader.svelte` and `JobsPage.svelte`, and rebases onto this.
- The plan's timings weren't measured on mini-ai, and CUDA hasn't seen the stage.
  Nothing in it is accelerator-specific.
