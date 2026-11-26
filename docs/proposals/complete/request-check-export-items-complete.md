# Check and export jobs get their own request items (#1617)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-26). Plan v1 on #1617 is the full design. This file records what
shipped, what was left out, and why.

**Result:** guidance only, no engine change. The request guidance now tells an
agent to give a check and an export their own items. A per-shot check links to
`check-NN` (`depends_on: ["shot-NN"]`), and an export links to `export`
(`depends_on: ["cut"]`). A check never links to the shot it checks, and an
export never links to `cut`. The ledger's rules are unchanged: every succeeded
job linked to an item is a take of it, and with nothing accepted the latest
one is the item's `result`.

## The problem

From #1523's field report ("Whitefish Point", items 5 and 8):

- A `templates/check-script` run linked to `fitz-light/shot-03` was listed
  beside the render in `attempts`. With nothing accepted, `_result`
  (`dw/ledger_views.py`) takes the latest succeeded job, so the shot's
  `result` would have been the check JSON. A check and a take that both
  succeed also raise a false `choose`.
- `run_workflow(request="fitz-light/export")` was refused with `has no item
  'export'`. The export was then linked to `cut`, where it competed for
  `cut`'s result.

The ask was a stored non-take mark on the link (`link="check"`, or one
inferred from the workflow).

## Why guidance instead of the mark

- The ledger already models the right thing: a check or an export is planned
  work of its own. Both work today with `set_request`, including items added
  mid-request.
- Item 8's refusal was the server working as designed: the plan had no
  `export` item.
- The mark would have cost about 4 stages ($14–20): a job-store column and
  migration, the view rules across status, result, revised, needs, cover,
  ready, remaining and attempt numbering, a new `run_workflow` parameter
  (REST, MCP, OpenAPI) and the UI twin. Every reader of "succeeded" would
  have to keep the new concept true. Removing it later would break MCP and
  REST.
- Inferring the mark from the workflow is worse: shape `utility` covers
  concat, mux and slice, which are deliverables.
- The evidence was one field report.

Don's delegate approved plan v1 as written on 2026-11-25 (Q1: build smaller;
Q2: item keys `check-NN` and `export`).

## Design as built

Stage 1, #1699 (`35b122d2`, merged as `49d397b0`). Plugin text only:

- **`plugins/dw/skills/script-to-video/references/request.md`**
  - The items table has a `check-NN` row (`templates/check-script`, or
    `check-narration` / `face-drift-check`, `depends_on: ["shot-NN"]`) and an
    `export` row (the packaging workflow, `depends_on: ["cut"]`).
  - "Run each item" states the linking rule and its reason (take, latest
    result, false `choose`). It also says to rerun `check-NN` after a new take
    of `shot-NN`, and to add a missing check or export item with `set_request`
    first, because the server refuses the link.
  - The result-rule sentence ends "so link only takes of that item to it."
- **`plugins/dw/skills/series-episodes/SKILL.md`** ("Run and judge"): link
  `check-script` to its own `check-<episode item>` item, or run it unlinked
  as a one-off, with a pointer to `request.md`.
- **`plugins/dw/skills/long-form/SKILL.md`** (section 1, beside its
  `request.md` pointer): "A check or export job links to its own item
  (`check-NN`, `export`), not to what it reads."

`docs/WORKSPACES.md`, `docs/MCP.md` and `docs/SERVER.md` are unchanged: the
rule they state didn't move.

### Deviations from the plan

- **No plugin version bump.** The plugin's version is the engine's
  (`plugins/dw/README.md`; `scripts/release.sh` bumps both and
  `tests/test_plugin_skills.py` holds them equal), so a plugin-only stage
  leaves it alone, as #1425 did.
- **The long-form line sits in section 1**, not beside the export step: the
  file was 14 bytes under its 12 KiB cap. Three sentences were shortened
  without changing their meaning. The tester accepted it.

## Stage

- **Stage 1, #1699:** no bounces. The verify on lem (develop `065d183c`) ran
  C-F1309 and C-F1310 live over MCP (a `check-01` link left `shot-01` with one
  attempt and its own `result`, and no `choose`; the `export` link was refused,
  then succeeded once `set_request` added the item, without touching `cut`)
  and read the three skill files for C-F1311.
  - Two case clauses went beyond the plan (`check-01`'s `result.output` as the
    check JSON, which the ledger gives as `null`; an `unknown_item` code the
    MCP error doesn't carry). They were not counted, and amendments are filed
    as dkackman/harnest#301.

No stage comment gave a `usage:` figure, so this record leaves cost out. The
plan estimated about $2–3.

## Left out, and what would bring it back

- **The stored non-take mark** (a `link=` argument on `run_workflow`, its job
  column and view rules): comes back with a second field report where check
  items clutter a request or can't express the check, for example one check
  spanning several takes of the same item, or a check that has to run before
  its shot is `done`.
- **Inferring non-takes from the workflow:** would need a new declared trait
  and a catalog read at submit time.
- **A rerun of `check-NN` after a retake** is a second take of `check-NN`.
  With nothing accepted there, it should raise `choose` on the check item. The
  guidance doesn't say to accept the newer check. The tester noted this at
  spec time; no case pins either behavior. It comes back if field reports show
  a false `choose` on check items.
- **Enforcement:** an agent can still link a check to its shot by habit. The
  guidance lowers the chance; only the mark above would enforce it.

Adjacent: #1613 (assemble-grade-export as one job) would make the `export`
item one job instead of several, and doesn't conflict.
