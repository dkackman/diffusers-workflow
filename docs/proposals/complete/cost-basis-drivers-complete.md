# The curated cost figure states its basis and measured-at settings (#1454)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the idea on 2026-11-18 from a UI support assessment of
v0.19.0..develop: `WorkflowCard.cost_drivers`, `constraints`, `variable_names` and
`WorkflowList.cost_basis` were served but unread by the web UI, so a card quoted "~3
min" with no word on what settings or basis that figure was for. Plan v1 is the plan
comment on #1454, and it still holds the full sweep. This file records what was built,
what was cut, and what stays open.

## Verdict: build smaller (one UI-only stage)

- **Cut: the per-field "moves the cost" mark** in the run form. Read off
  `cost_drivers`, it would be wrong both ways. Some declared drivers don't move the
  figure (the unscalable ones, e.g. H3 `weights_dtype`/`attention_backend`/`cache`,
  `num_images_per_prompt`, `tier`: changing one only sets `low_confidence`). Some
  undeclared things do (`for_each` list lengths, guides and references, chain
  retries, preset selectors). Sorting them apart in the UI would re-implement
  `dw/cost_scaling.py`, which the UI row of `docs/ARCHITECTURE.md` forbids ("derives
  nothing of its own"). The per-run answer is validate's `plan.estimate`, which #1446
  stage B shows on the run page.
- **What would bring it back:** a server field classifying each driver (scales /
  flags only). That is new REST surface, so it would be its own issue for Don.
- **Correction to the idea:** `WorkflowList.cost_basis` is not "what the figure was
  measured at". It is the constant `"curated"` (`dw/server/routes/library.py`), meaning
  a maintainer measured it once. The settings are the card's `cost_drivers` map (name
  to preset-applied default, `dw/server/catalog.py`), and the hardware is
  `cost[0].name ?? cost[0].device`.
- **Left unread on purpose:** `constraints` (owned by #1447) and `variable_names` (it
  duplicates the definition's keys).
- **Cheaper alternatives rejected:** the tooltip clause only (leaves the run page with
  no curated figure), and nothing.

## Plan history

- **v1 (2026-11-18).** One stage. The advisor approved it on Don's behalf at
  2026-11-19T05:25Z in the batch lane, taking every question's default:
  - Q1 [none]: cut the per-field mark.
  - Q2 [none]: word `curated` as "a maintainer's figure", like the MCP text.
  - Q3 [none]: lists print as an entry count, objects as `{…}`.
  - Adjustments: ship the "other values re-price it in the plan" hint only if #1446
    stage B's plan box is on develop; read expected values from the live card; check a
    preset template whose bundle and raw defaults differ, or use a mock if none does;
    keep the formatter free of driver-specific rules.

No re-plan.

## What was built

### Stage A (#1479): basis and settings in the card tooltip and on the run page (UI)

Commit `b2d80f08`, merged to `develop` as `74870156`.

- **`ui/src/lib/cost.ts` (new):**
  - `costMeasuredAt(drivers)` prints the card's `cost_drivers` map as
    `name=value, …` in map order. A scalar prints as itself, a list as `<n> entries`,
    `null` as `unset`, an object as `{…}`; a value over 40 characters is cut with `…`.
    An empty or missing map gives `''`. No driver-specific rules.
  - `costBasisWords(basis)` maps `curated` to "a maintainer's figure"; any other value
    prints raw, and an empty or null basis prints nothing.
  - `costFigure` and `costMeasuredOn` hold the caption format and the "measured on
    <where>[ at …]" clause, shared by both pages.
- **Catalog card (`WorkflowsPage.svelte`):** `cardTitle`'s line reads "a maintainer's
  figure, measured on <where> at <drivers>" (no `at` clause without drivers, no line
  for a null cost). The caption (`~N min · X GB VRAM`) is unchanged.
- **Run page (`WorkflowPage.svelte`):** a line `aria-label="curated cost"`, "a
  maintainer's figure: ~N min · X GB VRAM, measured on … at …", built from the card in
  the listing the page already fetched, with the muted hint "· other values re-price
  it in the plan". The hint **shipped**, because #1446 stage B (#1468, the plan box)
  was on develop. A null cost prints nothing.
- **Docs:** `docs/SERVER.md` now documents the card's `cost_drivers` as a map from
  driver to preset-applied default (a gap before), and its workflow-page paragraph
  describes the curated line.
- **Tests:** `cost.test.ts` (every formatter case), `WorkflowsPage.test.ts` (tooltip
  with drivers, list driver, no drivers, null cost, unchanged caption; unknown basis
  and hostile value printed literally), `WorkflowPage.test.ts` (curated line shows with
  a card value that differs from the definition, sits before the observed text, absent
  for a null cost). `npm test` 861 passing; `check` and `lint` clean.

**Deviation: the observed figure moved.** It used to sit inline in the head row next
to Run. To put the curated line above it, both now sit in one `div.costs` block under
the head row, curated first; the observed line keeps its tooltip. The head row holds
only the title, Edit, Duplicate and Run.

**Verified** on mini-ai (mps), develop @ `b8d9757e`, through the web UI with the
ui-check runner: C-F1102–C-F1108 and SE-F076 all pass. Nothing was queued.

## Bounces

Stage A: none. Built once, verified first time.

## Cost

Not recorded: neither stage comment names a `usage:` figure. The plan estimated ~$2.

## Open points

- **No live preset template was found whose bundle-applied driver values differ from
  its raw defaults.** On `pika-h3/text-to-video` they are equal, so acceptance 2's
  "follows the card" was shown with a mocked `cost_drivers.width` (and in
  `WorkflowPage.test.ts`). The code reads the card, never `workflow.variables`.
- **Verification limits:** ui-check can't fetch JSON, so card values were compared
  through rendered text and `get_workflow` defaults; "key removed" and "card missing
  from the list" were approximated by mocking to null; C-F1108's tooltip check sampled
  two cards (all 55 captions were checked).
- **Entry choice:** the card shows `cost[0]`, while the engine quotes this device's
  entry. The tooltip names the entry's device, so it is honest about which figure it
  is; the run-page hint points at the per-device plan.
- **The cut mark** needs a server driver classification first (above).
