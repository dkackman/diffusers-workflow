# Requests: a ledger that joins the jobs made for one ask (#1458)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the feature on 2026-11-18, from the same UI assessment as
#1446-#1457. An agent breaks an ask ("create XYZ") into many jobs, and nothing joined
them: jobs linked only by `workspace` and `rerun_of`, `note` was free text, and the
script-to-video shot document was prompt-library text, shared across workspaces and
invisible to the jobs list. Plan v3 is the plan comment on #1458, and it still holds the
full sweep (file:line). This file records what was built, what was cut and what stays.

## Verdict: build smaller (four stages)

- **The case.** Don's driving sessions hand-rolled the ledger in notes (`lego-league`,
  48 jobs: `"ep01 s7 I2V"`, `"ep01 s6 sfx (reroll clip)"`), several asks shared one
  workspace, and the skills marked keepers with `set_note` and logged progress in the
  workspace note. No field report asked for it by name; the demand was the repeated
  pattern.
- **Don's decisions (2026-11-18):** a ledger, not an orchestrator; one request belongs
  to one workspace; "request" in the UI with a distinct module name in code; the
  initiating prompt kept verbatim and the settled goal beside it, the Q&A between them
  not stored.
- **Cheaper alternatives rejected:** a `req:<id>/<item>` note convention plus a text
  filter (what `lego-league` already did, and what wasn't working), and the shot
  document alone. Neither gives the UI a plan to show.

**Corrections to the proposal** (plan v2):
1. "Copies with it": no. A workspace copy carries no job records, so a copied ledger
   would show every item as planned. `requests.json` is not copied; `export_job`'s
   `job.json` carries `request`/`item`.
2. A new root file makes `delete_workspace` refuse with 409, unless it joins
   `_foreign_entries`' allow-list. Stage 1 added it.
3. "Accepted" is stored as a run id with the job id beside it: run ids survive a copy
   and are what the gallery groups by.
4. "One `request` argument" carries the item too: `request="<id>/<item-key>"`.

## Plan history

- **v1 (2026-11-18).** Five stages. The advisor approved it on Don's behalf at
  2026-11-19T05:50Z with the Q1-Q7 defaults and six adjustments: re-staging so the MCP
  tools ship with the ledger, a comment-survival unit test, Don as stage 3's verifier,
  file/argument/estimate/text/unknown-field caps, API-token auth on PUT, and lock-free
  reads.
- **v2 (2026-11-19).** Folded the ruling in. Four stages.
- **v3 (2026-11-19, `loosen`).** Stage 2's build found that "`move_job` clears the link"
  rested on a false assumption: `move_job` only reorders the waiting queue, and nothing
  moves a job record between workspaces. The item and its acceptance intent were
  removed. Don approved v3. The stage-1 cases stood, and the tester added C-F1146 for
  "Deleting things" and asked the curator to retire C-F1138 (dkackman/harnest#263).
- Between v2 and v3 the approval label went missing without a comment. A design
  session recorded that and Don re-confirmed it.

**Rulings (advisor, on Don's behalf):** Q1 [scope] cut the page edits and runnable
items to follow-up ideas; Q2 [default] the ledger isn't copied with a workspace; Q3
[scope] the per-item comment box stays in; Q4 [none] the `id/key` argument form; Q5
[none] `get_request` always returns every comment with its timestamp; Q6 [none]
`dw/ledger.py`, `<workspace>/requests.json`, the key pattern; Q7 [taste] Don looks at
the UI once.

## What was built

### Stage 1 (#1481): the ledger, its file, REST and two MCP tools

Merged to `develop` at `3e0ab98b` (merge `a25620ff` of `c3376ac5`); bounce fix
`115d39b6`, merged as `2dc957cb`.

- `dw/ledger.py`: the model, validation, derived state and file I/O.
  `validate_ledger_key` in `dw/security.py` (`^[a-z0-9][a-z0-9-]{0,63}\Z`).
- `<workspace root>/requests.json`: atomic writes under the per-directory lock,
  lock-free reads, on the `_foreign_entries` allow-list. A no-root workspace reads as
  empty and refuses writes with 409 `no_workspace_root`.
- REST: `GET /api/workspaces/{name}/requests`, `GET .../requests/{id}`, `PUT
  .../requests/{id}` (whole plan, revisioned, API token, no query token).
- Every cap and rule in the plan's Security and Writes sections: write-once prompt,
  `409 stale_revision` with the current revision, drop-on-remove versus delete, un-drop,
  `depends_on` unknown/self/cycle, unknown item fields refused, a `by: user` comment
  surviving an agent write. 97 cases in `tests/test_ledger.py`.
- MCP `set_request` and `get_request` (`dw_mcp/ledger.py`), `get_request` read-only.
  Tool count 65 → 67. `get_guide("workspaces", section="Requests")`.
- **Deviation:** the routes live in `dw/server/routes/library.py` beside the other
  workspace routes, not a new `routes/ledger.py`, so the module count rose by the two
  the plan names. That +2 (271 → 273 by `arch_metrics.py`) needed a waiver: the
  advisor approved it as the approved design and added `arch-approved`; `baseline.json`
  `modules` was raised accordingly.

### Stage 2 (#1482): the job link and the `run_workflow` argument

Merged to `develop` at `8cdaead8` (merge of `a19b0fda`); architecture fix `fcdd5271`,
merged as `e2b16a86`.

- `POST /api/jobs` takes `request`/`item`, validated in admission against the ledger
  (read without its lock): 422 `unknown_request`, `request_closed`, `unknown_item`,
  `item_dropped`, before anything queues.
- `request`/`item` on the job record, `summary()`, two indexed history columns (every
  hand-written list), `RERUN_SPEC_KEYS` (single and restart-batch reruns keep the link;
  a rerun into a closed request or dropped item still runs, with
  `rerun_ledger_link_<code>`) and `JOB_RECORD_KEYS`.
- `GET /api/jobs?request=<id>[&item=<key>]`, `GET /api/gallery?request=`.
- Derived item state (`dropped`, `done`, `running`, `queued`, `failed`, `planned`),
  the result (accepted run if it exists, else the latest succeeded one, with
  `accepted_missing`), attempts, `spent_minutes` and `remaining_minutes`. `accepted`
  on `set_request` must be a succeeded job linked to that item.
- `run_workflow(request="<id>/<item-key>")`. The surface budget measured 15_957.5 of an
  unraised 16_000.
- **Deviation (from the architecture bounce):** `"Demo/cat"` is refused by the server's
  422, not a client-side copy of the key rule; the client only splits on one `/`.

### Stage 3 (#1483): the Requests UI

Merged to `develop` at `78841cce` (`1ca8db65`); bounce fix `f502680f`, merged as
`c888bec2`.

- A Requests section (`routes.ts`, `RequestsPage.svelte`) and an open-requests panel
  on the workspace overview.
- The request page (`RequestPage.svelte`, `request/RequestItem.svelte`): prompt and
  goal, items in order with state badge, result thumbnail, attempts and estimate;
  spent versus remaining; revision history; "This request no longer exists." on a 404;
  a 5 s poll while an item is queued or running.
- The comment box, the page's one edit: it PUTs the plan as read with one comment
  replaced and `by: "user"`; a stale 409 shows inline with Reload.
- A `req` hash filter and `request › item` row label on the jobs list, a request filter
  chip (with a picker) in the gallery, and a breadcrumb in `JobHeader.svelte`.
- **Deviations:** the row label is not a button (it sits inside the row's link); the
  request page links to "its jobs" and "its outputs"; the comment box is its own
  textarea, not `NoteEditor`.
- **Verifier changed by Don (2026-11-19):** the tester verifies behaviour in the
  browser through `ui-check`; Don judges the layout once, separately, not blocking.

### Stage 4 (#1484): skill adoption (plugin only)

Merged to `develop` at `f7538e91` (`41f17545`).

- script-to-video (step 2), long-form (section 1), storyboard ("The request (three
  boards or more)") and series-episodes (one per episode) open a request at the
  shot-list step, for three or more jobs only.
- The shared procedure is `plugins/dw/skills/script-to-video/references/request.md`:
  the id rule, the item table (`shot-NN` → `prompt:<project>/shot-NN`, cast sheets,
  `score`, a `cut` depending on the shots), running with `request=`, accepting with
  `set_request` instead of `set_note`, re-reading after `stale_revision`, reading
  comments before a revision pass, dropped versus removed, closing.
- Text pins in `TestSkillRequests` (`tests/test_plugin_skills.py`).
- **Deviation:** no plugin version bump; `plugin.json` already matches `pyproject.toml`,
  and `scripts/release.sh` bumps both together.

## Bounces per stage

- **Stage 1: one verify bounce** (plus the module-count waiver hold before merge).
  C-F1126 and SE-F079: an id with `/`, or an empty one, matched no route because the
  router decodes `%2F`, so a GET got a bare 404 and a PUT a 405 instead of the id
  rule's 400. Fixed by `{request_id:path}`. The spent/remaining assertions moved to
  stage 2 (dkackman/harnest#261).
- **Stage 2: one architecture bounce, one verify bounce.** The review found a second
  owner of the key rule in `dw_mcp/diagnose.py`, a copy of `job_run_directory`, and the
  ledger lookup outside admission; all three fixed in `fcdd5271`. The verify failed
  C-F1146 step 2: an unfiltered `list_jobs` hides a deleted workspace's jobs. That is
  #944's design, not this stage's; Don ruled the default (#944 stands, the case
  narrowed to `get_job`), and the stage passed with no code change. The build also
  stopped once for the v3 re-plan.
- **Stage 3: one verify bounce.** The gallery dropped the item half of
  `req=<id>/<item>`, and `/api/gallery` ignored `item`. Fixed in both, with a filtered
  empty state.
- **Stage 4: none.** C-F1139 (the storyboard request step: three boards done, spent
  0.51 min equal to the summed durations) and C-F1140 passed first time.

Stage costs aren't recorded: the stage comments name no `usage:` figures. The plan
estimated about $21 ($8, $6, $5, $2). GPU use was a handful of small
`templates/text-to-image` and `qr_code` jobs on mini-ai.

## Cut, and what brings each back

- **User edits from the request page** (drop, reorder, accept another attempt, "group
  into a request" on selected jobs): #1485, a parked `idea`. "Group into a request"
  would be the first route to rewrite finished jobs' history rows, so it gets its own
  design.
- **Runnable items** ("run" and "run what's ready" from the page, from an item's stored
  workflow and arguments): #1486, a parked `idea`. Stored arguments stay inert until
  then, and that follow-up must send them through normal admission.
- **`move_job` clearing the link** (v3): nothing moves a job between workspaces. A
  cross-workspace job move would be its own `scope` feature.
- **A `list_jobs` request filter over MCP**: Don's limit was two tools and one
  argument; `get_request` lists an item's attempts.

## Known limits

- **The MCP surface budget has about 42 tokens of headroom** (15_957.5 of 16_000).
  The next tool or parameter needs narrative moved to a guide, or a `scope` ruling to
  raise the budget.
- **An unfiltered `list_jobs` hides a deleted workspace's linked jobs** (#944). Their
  `request`/`item` survive under `"<ws> (deleted)"` and `get_job` shows them; the job
  page's breadcrumb reads "request no longer exists".
- **Comments are always returned** (Q5): there is no "changed since last read".
- **The H3-shaped skill steps** (script-to-video, series-episodes) are covered by text
  pins only; their first real use on lem is the field check.
- **Don's one look at the UI layout** (Q7) is separate and doesn't block.
- The plan's timings weren't measured on mini-ai, and CUDA hasn't seen the stages.
  Nothing in them is accelerator-specific.
