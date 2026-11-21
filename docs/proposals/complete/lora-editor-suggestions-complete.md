# Catalog LoRA suggestions in the LoRA editor (#1456)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the idea on 2026-11-18 from a UI support assessment of
v0.19.0..develop: `GET /api/loras/recommend` and `/api/lora-schema` had no caller in
`ui/src`, and `LorasEditor.svelte` gave no hint which LoRAs suit a step's base. Plan v2
is the plan comment on #1456, and it still holds the full sweep. This file records what
was built, what was cut, and what stays open.

## Verdict: build smaller (one stage, catalog only)

- **Cut: Hub candidates (`/api/loras/recommend`).** That route searches the Hugging
  Face Hub on every call, runs one search per server at a time (a concurrent call gets
  `hub_error`), and can't filter by partition, so its suggestions can fail `validate`
  for a partitioned base (H3). An editor panel calling it would make the UI reach an
  outside service by itself. *Brings it back:* Don asking for an explicit "search the
  Hub" button, a `trust` call and its own issue.
- **Cut: `/api/lora-schema` in the UI.** Suggestions don't need it. *Brings it back:* a
  catalog-entry editor (the "UI LoRA page" the catalog design,
  `docs/superpowers/specs/2026-10-03-lora-catalog-design.md`, left out of scope).
- **Cut: suggestions for a `variable:` base.** The server resolves variables only for a
  saved workflow, never for the unsaved draft. *Brings it back:* a step needing it in
  practice (the draft's variables passed down four component levels).
- **Corrections to the idea:**
  - The editor did not "offer the catalog": nothing in `ui/src` called any
    `/api/loras*` route, and the editor only added a blank row.
  - It was not "UI only": `tests/test_ui_twins.py` refuses an `/api/` literal in
    `api.ts` unless the route is in `UI_READ_ROUTES`, which requires a declared
    response schema. `GET /api/loras` had none, so the stage declared one.
  - The right route is the catalog-only `GET /api/loras?model=&workflow=`, which filters
    by exact base and partition and returns rejected rows too (the client drops them).

## Plan history

- **v1 (2026-11-18).** One stage. The advisor approved it on Don's behalf at
  2026-11-19T05:25Z in the batch lane, taking every question's default:
  - Q1 [none]: catalog-only `GET /api/loras`, never `/recommend`.
  - Q2 [none]: one explanatory line for a `variable:` base.
  - Q3 [none]: declare a response model on the existing route (typing, not new surface).
  - Adjustments: the row model declares every catalog field or sets `extra="allow"`
    (strict mode forbids extras); a required pytest loads every shipped `loras/` entry
    through it; acceptance 1 compares bodies before and after, not "intact"; no request
    to `huggingface.co` either; show each entry's `workflow` constraint in stage A.
- **v2 (2026-11-19, decompose).** Folded those adjustments in. No scope change.

No re-plan.

## What was built

### Stage A (#1480): catalog suggestions in the LoRA editor (server typing + UI)

Merged to `develop` at `29e7f72d` (branch `feat/1456-A-lora-suggestions`: `aadc0b1e`
server, `5f7b9caf` UI, plus tests/docs), and the bounce fix at `548a7665` (`4488f9e4`).

- **Server.** `GET /api/loras` declares `response_model=LoraList` with
  `response_model_exclude_unset=True` (`dw/server/routes/loras.py`; `LoraScale`,
  `LoraEvidence`, `LoraRow`, `ResolvedBase`, `LoraList` in
  `dw/server/api_models_library.py`). Every field of `dw/lora_catalog_schema.json` is
  declared, with no `extra="allow"`; optional keys stay absent, an int scale stays an
  int. The handler is unchanged. `/api/loras` is in `UI_READ_ROUTES`, and
  `ui/src/lib/generated/` is regenerated.
- **UI.** `api.listLoras({model, workflow?})`, and a "From the catalog" panel in
  `ui/src/lib/editor/LorasEditor.svelte`:
  - fetches after a 300 ms debounce, for a literal `owner/name` base only, adding
    `workflow` when the step names a literal partition (a non-empty string with no
    `:`);
  - drops rejected rows and sorts proven, then trial, then by name;
  - shows name, status, scale (default plus range), description, use_when, trigger, and
    the entry's `workflow` when the step has no literal partition;
  - adds an entry in one click as `{model_name, weight_name?, revision?, scale}`
    (scale falls back to 1.0, `weight_name` copied verbatim); an entry already on the
    step (model + weight file) shows a disabled "added";
  - a non-literal base, zero matches and a fetch error each get one muted line (the
    error reads "Catalog suggestions are unavailable right now.");
  - existing rows show `revision` read-only.
- **Tests.** `tests/test_server_loras.py::TestDeclaredResponse` (full-field row, no
  `scale` / null `workflow` row, `resolved` absent) and the required
  `test_every_shipped_entry_loads_through_the_row_model`; `LorasEditor.test.ts` (11
  cases, mocked `api`); `StepEditor.test.ts` mocks `listLoras`; Playwright
  `ui/e2e/lora-suggestions.spec.ts` on a fixture `ui/e2e/fixtures/loras/` tree (one
  trial, one rejected), asserting no `/recommend` or `huggingface.co` request and,
  after the bounce, that the panel text follows `--ink` through every theme.
- **Docs.** "In the web UI" in `docs/LORAS.md`; `docs/SERVER.md` notes the listing is
  declared as `LoraList`.

**Deviations:** the contract cases live in `tests/test_server_loras.py` beside the
route's other tests (`test_api_contract.py` only gained the `UI_READ_ROUTES` entry);
the optional `.withicon` cleanup needed no change.

**Verified** on mini-ai (mps) through MCP and the ui-check runner: C-F1109–C-F1117,
C-F1119, SE-F077 and SE-F078 pass; C-F1118 (b) and (c) pass; C-F1120 passes in the
light theme. No LoRA was downloaded and no job ran.

### Fix-forward #1501: a failed catalog fetch shows its line (C-F1119 c/d)

The feature's final check (at `058736f9`) failed C-F1119 (c) and (d). With
`/api/loras` stubbed to a 500 or aborted, the panel showed only its heading. The
`'loading'` state rendered nothing under the heading, so a fetch that never settled in
the browser looked the same as a failed one. Commit `e9a82b67` was merged to `develop`
as `5d245a10` (branch `feat/1456-A-catalog-fetch-error`):
- the loading state now shows "Loading catalog suggestions…";
- a fetch still unsettled after 8 s counts as failed and shows "Catalog suggestions are
  unavailable right now.";
- Playwright routes `/api/loras?*` to a 500, an abort and a request that is never
  answered, and asserts the line each time; vitest covers the loading line and the
  timeout.

It was verified on mini-ai at `4dc724fe`: C-F1119 (a)–(e) pass, and C-F1120 passes in
both themes. A late answer after the timeout replaces the error line.

## Bounces

Stage A: one.
- **C-F1120, dark theme.** The entry name, description and heading set no colour and
  inherited it from `.panel.step`, which transitions `color` over 0.15 s, so a read
  right after the theme toggle saw light-theme ink. Fixed by giving them
  `color: var(--ink)`, with a Playwright check that reads them with no wait after each
  toggle.

Feature final check: one (C-F1119 c/d), fixed forward in #1501, which then passed
with no bounce.

## Cost

Not recorded: no stage comment names a `usage:` figure. The plan estimated $3–5.

## Open points

- **C-F1118 passed in the final check.** It was checked against the plan's order
  (proven first, then by name), which is the subject of dkackman/harnest#255 / #256. The
  tester removed its `pending: #1480` line.
- **C-F1120's dark theme** was re-run in #1501's verify, by toggling the theme and
  reloading. It passes.
- **The rest of the step editor lags the theme switch** the same way (any text
  inheriting from `.panel`). It predates this feature; fixing it means changing the
  global `.panel` transition in `ui/src/app.css`.
- **Side findings filed:** #1495 / #1497, a page error on `variable:cache` when
  expanding the ref2va template step.
- The cuts above (Hub search, `lora-schema`, `variable:` bases).
