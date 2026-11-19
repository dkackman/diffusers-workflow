# The run form shows preset bundles and declared variable rules (#1447)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the idea on 2026-11-18, from a UI support assessment of
v0.19.0..develop. 13 catalog workflows declare `presets` and 66 declare
`variable_constraints`, but `VariablesForm.svelte` read only `required`. A preset
selector was a free-text field with no list of its bundles, and a rule (grid, range,
minimum length, word rate) was discovered as a refusal after Run. Plan v1 is the plan
comment on #1447, and it still holds the full sweep. This file records what was
built, what was cut and what stays.

## Verdict: build smaller (two UI-only stages, no GPU)

- **Cut:** marking a value that breaks a rule before submit, client-side. #1446 stage
  B already validates the run form as you type and places each `arguments.<name>`
  finding under its field in the server's own words. Those cover what a UI copy of
  the rules can't: the snap warning, the H3 fixed limits in `dw/pipeline_limits.py`
  (never served), the selector-not-allowed error and list-entry paths. A UI checker
  would copy `dw/variable_constraints.py`'s rule order and drift from it, and break
  the map row "the UI … derives nothing of its own".
- **Cheaper alternatives rejected:** the dropdown alone (leaves rules unstated); raw
  `variable_constraints` JSON under each field (hard to read); a server-served
  phrase from the engine's `_accepted` on `/api/workflows/{name}/variables` (new
  REST surface and a regen). The served phrase stays the fallback if the labels drift.
- **No server change was needed:** the run page already fetches the full definition,
  which carries both `presets` and `variable_constraints`.

## Plan history

- **v1 (2026-11-18).** Batch lane, every question `[none]`. The advisor approved it on
  Don's behalf with four minor adjustments, folded into the stage bodies without a
  new version:
  - the schema-coverage pin falls back to `tests/test_ui_twins.py` if vitest can't
    import `dw/workflow_schema.json` (it could, so no fallback was used);
  - the ARCHITECTURE sentence names its enforcer;
  - the "(not a preset)" option's value is the raw current value;
  - the `constraints`/`presets` props stay independent of #1446's `findings`.

**Rulings (advisor, all defaults):**
- Q1 [none]: one fixed label per declared key, in the engine's `_accepted` form.
- Q2 [none]: define mode keeps the selector a text field, with a "sets …" line and
  its bundle names.
- Q3 [none]: a bundle's values are placeholders only, never copied into `arguments`,
  so a later change to the template's bundle still reaches a rerun.

## What was built

### Stage A (#1469): each declared rule and its reason under the field

Merged to `develop` at `d687fa1c` (branch `feat/1447-A-constraint-hints`, `04b3e254`).

- **`constraintHints(definition)`** in `ui/src/lib/editor.ts`: variable → lines, one
  fixed label per key, every number from the declaration.
  - `min_frames`/`max_frames`: "lo to hi", "lo or more", "hi or less".
  - `modulus`/`remainder`: "m * n + r".
  - `snap:"up"`: "an off-grid value rounds up".
  - `min_length`: "a non-empty string/list" or "at least N characters/entries", by
    the declared default's type (a null default reads as a string).
  - `max_words_per_second` + `seconds_from`: "about R words a second of F (a
    warning)".
  - `required` keeps the field's existing mark and adds no line.
  - Keys of one rule join with ", "; the `reason` follows after " - ".
  - An unknown key falls back to `key: value`.
- **List-entry rules** show under the list as "each entry's `<field>`: …". The list
  a field belongs to is read the way `dw/for_each.py:list_fields` reads it: steps
  with `for_each: variable:<list>` and their `item:<field>` references. The plan's
  `shots[].num_frames` spelling was wrong: in the files the key is a plain field name
  matched through `item:`. A key that is both a top-level variable and an entry field
  shows under both.
- **Schema-coverage pin:** `editor.test.ts` imports `dw/workflow_schema.json` directly
  and fails if a `$defs.variable_constraint.properties` key has no label.
- **Form:** a `constraints` prop on `VariablesForm.svelte`, rendered as a muted
  `ul.hints` under the field in both modes, plain text. `WorkflowPage` passes it from
  the fetched definition, `EditorPage` from the live document.
- **Docs:** `docs/SERVER.md` workflow-page paragraph; `docs/ARCHITECTURE.md` row "The
  UI reads engine fields", naming `editor.test.ts` and the coverage case.

**Bounces (0 failures, 1 hold).** The first verify couldn't run: the tester's shell
had no browser (the same gap as #1467). The advisor handed it back once Don had
approved the tester's `ui-check` runner. The second verify (develop @ `d687fa1c`)
passed C-F1038 to C-F1044 and SE-F070.

### Stage B (#1470): a preset selector as a dropdown, with the chosen bundle's values as placeholders

Merged to `develop` at `e838574d` (branch `feat/1447-B-preset-dropdown`, `146205a6`).

- **`presetView(definition)`**: per selector, its bundle names (exactly the keys of
  `presets.<selector>`) and `sets`, the union of the variables its bundles set, minus
  the selector itself (as `dw/presets.py` skips it).
- **`presetPlaceholder(definition, overrides, name)`**: the chosen bundle (the
  override, else the declared default) gives a variable's placeholder, only while its
  field is empty. This is the one client-side layering, in `effective_variables`
  order, for display only.
- **Form:** a `presets` prop on `VariablesForm.svelte`. In override mode a selector is
  a `<select>`: "default (`<declared>`)" with value `''`, one option per bundle, and
  "`<v>` (not a preset)" carrying the raw value for a non-bundle current value. A
  reference or a non-string declared default stays a text field. Under it, "sets
  `<a>`, `<b>`". Define mode keeps the text field and adds "bundles: …". The backend
  select is untouched.
- **Pages:** `WorkflowPage` passes `workflow.presets`, `EditorPage` `ed.doc.presets`.
- **Docs:** `docs/SERVER.md` workflow-page paragraph; the ARCHITECTURE row names
  `presets` and the placeholder-only layering.

**Bounces (0).** The first verify (develop @ `e838574d`) passed C-F1045 to C-F1052
and SE-F071 with `ui-check`; stubbed submits carried only the selector and typed
values. C-F1049 passed as not exercised (see Known limits); `validate_workflow`
confirmed the server still refuses a non-bundle selector value. SE-F071's wording
(`default (…)` rather than `(not a preset)` for a mocked declared value) is a case
amendment for the harness, not a code finding.

Stage costs aren't recorded: the stage comments name no `usage:` figures. The plan
estimated ~$2 each, ~$4 in all. The feature used no GPU time.

## Cut, and what brings each back

- **Client-side violation marks before submit.** They come back as a stage C, for Don
  to rule on, if #1446's as-you-type check proves too slow to read as "before
  submit". It would copy the engine's rule order.
- **A server-served rule phrase** (alternative c). It comes back if the UI labels
  drift from the engine's `_accepted` wording; it is new REST surface, Don's call.
- **Editing presets or constraints in the editor.** Define mode only shows them.

## Known limits

- **Label drift:** the coverage pin catches a new constraint key, not a reworded
  engine phrase. That costs wording only, never a verdict.
- **The rerun modal** (`RerunWithChanges.svelte`, #1448) isn't passed `presets` or
  `constraints`, so it keeps plain text fields.
- **No URL prefill:** the run page reads no query string, so a non-bundle selector
  value reaches the form only if `overrides` already holds it. "(not a preset)" is
  pinned in `VariablesForm.test.ts`.
- **Side observation from stage A's verify, not filed:** in the editor, clicking a
  mode button right after typing in the JSON pane seemed to drop the edit; blurring
  first applied it.
- The plan's timings weren't measured on mini-ai, and CUDA hasn't seen either stage.
  Neither has anything accelerator-specific.
