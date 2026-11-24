# UI quality phase 5: creating - run form, catalog, editor. Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a person can find the workflow they want, fill it in with their own files or an earlier step's result through one reference input, see in one plain sentence whether it will run and why not, and run it; an author can chain a step, rename it, and jump from a validation error to the field that caused it, without the editor breaking what they wrote.

**Architecture:** One building block, `ReferenceField` (`ui/src/lib/ui/ReferenceField.svelte`, pure logic in `ui/src/lib/referenceField.ts`): a media or reference value is edited through it on the run form, in the rerun dialog and in the step editor - kind-filtered suggestions with earlier steps' `previous_result:` first, upload, browse, and a preview drawn by `OutputTile`. It replaces `MediaArgumentInput` and `BrowseButton`. Around it: one verdict and one plan renderer for the run check and the editor (`verdictOf`, `PlanLines`, survey D9), a run form split into Essentials and Advanced that remembers what was typed, a catalog search that matches words, and editor fixes (step rename that rewrites references, variable names and types, component slots from the pipeline, linked validation). One server task adds an opt-in `inputs=true` to `GET /api/workflows`: each variable that feeds a media argument, with its kind.

**Tech stack:** Svelte 5 runes, vitest with @testing-library/svelte, Playwright, ESLint flat config, Node 24 type stripping for `scripts/arch-metrics.mjs`. Server: FastAPI 0.143.0 / Pydantic 2.14.1 for the OpenAPI dump, pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-ui-quality-design.md`. This plan covers:
- principle P8 (*One reference input*) and the building-blocks row *Reference input* (`ReferenceField`);
- the phase-5 row of *Phases*: WF-6..14, WF-17, WF-19..24, JB-5, JB-13, and the journey gates `chain-a-step.spec.ts` and `first-image.spec.ts`;
- decision 6 (*Gate Run on a refused check*, approved): Run becomes **Fix N issues** on the check's own refused verdict, and the server stays the authority;
- survey item D9 (`PlanLines`);
- the phase-4 carry-over: `raw_media` from 4 to 1 (Suggest 1 and MediaArgumentInput 2 go; ChainSegments 1 stays, a thumb seeked to a segment's first frame, which a poster cannot be).

The audit (`docs/superpowers/specs/2026-11-22-ui-audit.md`, Tier 2 §A, §B *Catalog search*, §C, §D, Tier 3 *Editor*) has the longer text. Its reviewer numbers map: *workflows 6..24* = WF-6..24, *jobs 5* = JB-5, *jobs 13* = JB-13 (the seed warning shown to the person running a job, the same warning as WF-9).

**Out of scope:**
- Phase 6 request behaviour (the request run dialog's engine text, RQ-10, stays phase 6's, though it may adopt `verdictOf` later).
- The prompt editor beyond the read-only banner (Task 17).
- #1599 timestamps, and every #1600-#1611 item not listed under *Folded in* below.

**Folded in** (each in code the task already rewrites):
1. #1600 - EditorPage `save()` can reject unhandled after a successful save (Task 18).
2. #1600 - Ctrl/Cmd+S under an open dialog or during IME composition falls through to the browser (Task 18).
3. #1600 - A step restored by Undo returns at its old raw index, not its old place (Task 14).
4. #1600 - A step restored under a new name is not announced (Task 14).
5. #1600 - The MediaPicker toggle does nothing during a search, and the picker shares one collapse key across tabs and workspaces (Task 4).
6. #1600 - EnhancePanel's job link uses `workspace.current` (Task 9; WorkerList keeps it, it needs the server field).
7. #1600 - Re-clicking the shape filter clears to "all" with no cue (Task 12).
8. #1600 - The StepEditor name field lost its 480px cap: checked by eye in Task 14's screenshot.
9. #1601 - Stale `~` comments in `WorkflowPage.svelte` and `WorkflowsPage.svelte` (Tasks 7 and 12; `runPlan.ts` stays with #1601).
10. #1601 - The shared file bar's `wf-folder`/`wf-savename` ids and workflow wording (Task 17).
11. #1601 - The editor's back label is the leaf where the page's h1 is the full path (Task 17: the label becomes the full path).
12. #1601 / #1610 - Prettier on EnhancePanel and VariablesForm (Tasks 9 and 13).
13. #1602 - No `editorKeys` test for a read-only document, a busy skip, or an unrelated Ctrl key (Task 18).
14. #1602 - `WorkflowPage.test.ts`'s back-label regex accepts either label (Task 7).
15. #1606 - The picker's current tile and hover look alike (Task 4).
16. #1606 - `ui/src/lib/api.ts` at 600 lines: Task 1 adds to it and pays for it, leaving it at or below 598.
17. #1610 - The editor's primary-button pin checks only the initial state; also check dirty, builtin and running (Task 17).

## Measured at the start (develop 78b66483)

**UI metrics** (`cd ui && node scripts/arch-metrics.mjs`): `files_over_size_ceiling 0, complex_functions 13, long_functions 0, import_cycles 0, modules_in_import_cycles 0, prefix_literals 3, a11y_suppressions 1, raw_error_toasts 0, empty_outside_loadstate 0, unwatched_reads 0, window_dialogs 0, unguarded_mutations 0, banned_terms 0, raw_media 4, duplicate_blocks 18`. `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` prints only `complex_functions: 11 -> 13` and `prefix_literals: 0 -> 3`, both pre-existing.
- `raw_media` 4: `MediaArgumentInput.svelte:204 img`, `:206 video`, `ChainSegments.svelte:238 video`, `Suggest.svelte:136 img`.
- `duplicate_blocks` 18; in files this phase rewrites: `ValidationPanel:74` and `RunCheckPanel:53` (the `.plan` list, D9), `VariablesForm:142`, `PipelineOptions:218`, `EditorPage:3`/`:234`, `WorkflowsPage:313`.

**Python** (`venv/bin/python scripts/arch_metrics.py`, the main checkout's venv): `modules 281`, `claude_md_lines 129`, `modules_over_size_ceiling 5`. `dw/server/api_models.py` 1105, `dw/server/job_history.py` 1000, `dw/server/routes/jobs.py` 1033, `dw/server/routes/library.py` 1000, `dw/server/api_models_library.py` 760, `dw/server/catalog.py` 436, `dw/server/catalog_shape.py` 546. `ui/CLAUDE.md` 12 lines.

**Page sizes** (limit 600): `app.css` 600, `Sidebar` 584, `FlowView` 576, `App.svelte` 561, `RequestItem` 550, `AssetsPage` 519, `OverviewPage` 514, `EditorPage` 509, `ModelsPage` 505, `JobPage` 502, `patterns.css` 485, `PromptEditorPage` 461, `StepEditor` 458, `GalleryPage` 454, `WorkflowPage` 434, `WorkflowsPage` 433, `PipelineOptions` 372, `VariablesForm` 322, `ArgumentsEditor` 310, `LorasEditor` 275, `MediaArgumentInput` 251, `Suggest` 223, `StepList` 213, `DocumentFileBar` 201. TS: `api.ts` 598, `editor.ts` 552, `flow.ts` 554, `editorShell.svelte.ts` 543.

**Findings checked against the code:**

| ID | State today | Where |
|---|---|---|
| WF-6 | open: a media argument's field offers `asset:`/`output:` only; once it holds `previous_result:x` it drops to the plain text branch, whose `previous_result:` rows come after every variable and prompt | `ArgumentsEditor.svelte:104-121,152-167`, `referenceSuggestions.ts:22-42` |
| WF-7 | open: `bind:value={step.name}` renames per keystroke, nothing rewrites references, and `StepModes` is keyed by name so the step falls back to compact | `StepEditor.svelte:164-168`, `stepModes.svelte.ts:20` |
| WF-8 | open: the check "informs; it never gates Run" | `RunCheckPanel.svelte:2-4`, `runCheck.svelte.ts:4` |
| WF-9 / JB-13 | open: the step cache's warning says "This workflow sets no 'seed'" when it sets `variable:seed` with a null default, and the job page lists every warning in a warn panel | `dw/step_cache.py:798-820`, `JobPage.svelte:338-344` |
| WF-10 | open: a gated repo reads "a job would be admitted, with warnings"; `access_blocked` is only in the mono plan line | `runCheck.svelte.ts:74-84`, `plan.ts:78` |
| WF-11 | open: run-form variables are text fields; the browse button shows only once a value is already a reference; no upload; no kind filter | `VariablesForm.svelte:221-236`, `BrowseButton.svelte:33` |
| WF-12 | open: `overrides` is reset to the prefill on every visit | `WorkflowPage.svelte:126-133` |
| WF-13 | partly done: the filters are in the hash (phase 3, D7). Search is one substring over name and description, and the count is `listed.length` only | `WorkflowsPage.svelte:105-117,161` |
| WF-14 | open: `N vars` | `WorkflowsPage.svelte:289-293` |
| WF-17 | open: no Enter, no name check, a new variable is `''` so `30` is stored as `"30"` | `VariablesForm.svelte:131,140-144,274-283` |
| WF-19 | open: `COMPONENT_SLOTS` is a fixed list | `editor.ts:188-204`, `PipelineOptions.svelte:40-42` |
| WF-20 | open: placeholders only | `LorasEditor.svelte:143-160` |
| WF-21 | open: plan lines, `root: message` findings and the curated cost line are engine wording; every variable is one flat list | `RunCheckPanel.svelte`, `WorkflowPage.svelte:262-276,313-335` |
| WF-22 | partly done: the workflow page says `examples, read-only`; the editor knows `ed.writable` and shows nothing | `WorkflowPage.svelte:225-232`, `EditorPage.svelte:331-411` |
| WF-23 | open: `path: message` text, validated only on the button | `ValidationPanel.svelte:34-44`, `EditorPage.svelte:254-273` |
| WF-24 | open: Enhance is on the prompt editor only | `EnhancePanel.svelte`, `PromptEditorPage.svelte:87` |
| JB-5 | partly done: double-submit is guarded (phase 2). Values are placeholders, there is no New seed, overrides are sent as strings | `RerunWithChanges.svelte:32-60,125-136` |

**Found already done:** WF-13's filters in the URL (phase 3); WF-15 and WF-16 (phases 1 and 3); WF-8's "unreadable refusal" half - a refused submit is already `describeError`'s title with the server text behind `Details` (phase 2), so what is left is gating Run; JB-5's double-submit (phase 2, JB-6); the `previous_result:` validity check (`danglingReferenceDetails`) exists, it is only the offering that is wrong.

## Global constraints

- **MCP behaviour is unchanged.** The one server change (Task 1) is additive and opt-in: `GET /api/workflows` without `inputs=true` answers key for key as before. Task 19 changes engine text MCP sees and runs **only if Don answers Question 4 yes**. OpenAPI is regenerated only with the pinned overlay on `PYTHONPATH`: `PYTHONPATH=/private/tmp/claude-501/-Users-don-src-dkackman-diffusers-workflow/c38aa355-7a1b-44f4-aee3-aaaef6ebc96d/scratchpad/p3/openapi-overlay venv/bin/python scripts/dump_openapi.py && cd ui && npm run gen:api` (if the overlay is gone, rebuild it as phase 4 did: `venv/bin/python -m pip install --target <dir> -c constraints-openapi.txt fastapi pydantic`). Never hand-edit `ui/src/lib/generated/`.
- **Size limits.**
  - Python `modules` stays 281. No task adds a module; a new test file is not one. A task that finds it needs a module stops and asks Don.
  - `claude_md_lines` stays 129: `ui/CLAUDE.md` stays 12 lines, rewritten, never grown.
  - `dw/server/api_models.py` ≤ 1105 (untouched), `dw/server/job_history.py` ≤ 1000 (untouched), `dw/server/routes/jobs.py` ≤ 1041 (untouched), `dw/server/routes/library.py` ≤ 1000 (Task 1 pays for its parameter).
  - `ui/src/lib/api.ts` ≤ 600 (Task 1 pays for its option). No page over 600 lines; `app.css` is at 600, so a new global rule goes in `src/patterns.css`.
  - No function over complexity 10 or 60 lines, in TS or Python.
- **The UI metrics gate.** Before Task 1: `node scripts/arch-metrics.mjs > "$SCRATCH/ui-p5.json"` (any scratch folder). Each task's gate is `node scripts/arch-metrics.mjs > "$SCRATCH/ui-now.json" && node scripts/arch-metrics.mjs --compare "$SCRATCH/ui-now.json" "$SCRATCH/ui-p5.json"`, and it prints nothing. `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` prints only the two pre-existing lines. Every count that is 0 stays 0. A task that lowers a count lowers that key in `baseline.json` in the same commit.
- **The phase gate** is `e2e/journeys/first-image.spec.ts` and `e2e/journeys/chain-a-step.spec.ts` passing, and `raw_media` at 1 (Task 20).
- **Staging and commits.** Stage files by name; never `git add -A ui` (Don edits `ui/package.json`), never add or force-add `.superpowers/`. Headlines use a scope (`ui(run): …`, `ui(editor): …`, `server(catalog): …`, `docs(…)`), never `fix: #N`; to name an issue write `refs #N` in the body. Each commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Checks.** UI, from `ui/`: `npm run check`, `npm run lint`, `npx vitest run <file>`, `npx playwright test <file>` (port 8971). Python: `venv/bin/python -m pytest <file> -q`, `venv/bin/python scripts/arch_metrics.py`. Tests run with `DW_STRICT_RESPONSES=1`.
- **Security.** Server code validates with `dw/security.py` against a real base; no `eval`, `exec` or `shell=True`. The variable-name rule in the UI is a twin of `dw/references.py`'s `VARIABLE_NAME_CHARACTERS`, pinned by `tests/test_ui_twins.py`.
- WCAG contrast is guidance, never a check. Names, focus, `aria-pressed`/`aria-expanded` and a dialog's label are in scope.
- **Building blocks.** `ReferenceField` gets a UI row in `docs/ARCHITECTURE.md` in Task 5; the rows a task changes (*UI overlays and suggestions* for `Suggest`, *The UI reads engine fields* for `verdictOf`/`planSummary`, *Media tiles* for the `row` size) are edited in that task.
- **Known flaky tests**: the smoke "editor opens a workflow with introspected arguments" (#1603) and `tests/test_ledger_job_link.py::TestDelete::test_a_request_made_again_under_the_id_takes_the_jobs_back`. Not this phase's.
- **Copy fixed by this plan.** Use it exactly; `<…>` is filled in.
  - **Verdict** (`verdictOf`), lead then remedy:
    - refused: `A job would be refused: <n> <issue|issues> to fix.` / no remedy.
    - gated (a `downloads_required` entry with `access_blocked`): `This model needs Hugging Face access.` / `Accept its terms on Hugging Face, then sign in on the server with hf auth login.`
    - a repo that is not there (`not_found`): `A model this workflow downloads can't be found.` / `Check the model name in the workflow.`
    - valid with a note under a field you filled: `Ready to run. Read the notes under <n> <field|fields>.`
    - otherwise valid: `Ready to run.`
    - the check failed: `Couldn't check this workflow.`, then `ErrorLine`.
    - while checking: `Checking…`
  - **Run gate:** the Run button reads `Fix <n> <issue|issues>`, titled `Shows what would stop this job`; pressing it scrolls to and focuses the first issue.
  - **Plan:** the summary line `<estimate> · downloads <gb> GB first` (the second half only when a download has a size), then `<Details summary="What a job will do">` holding the plan lines.
  - **Author notes:** `<Details summary="<n> <note|notes> for the workflow's author">`.
  - **Run form:** the Advanced fold `Advanced (<n>)`; a seed variable with no default has the placeholder `random each job`; when values were restored, the hint `Restored what you last typed here.` with the button **Reset to defaults**.
  - **ReferenceField:** placeholder `type, drop or choose <a|an> <kind>`; **Choose…** titled `Choose <a|an> <kind> from the assets or earlier jobs`; **Upload…** titled `Upload <a|an> <kind> from this computer, or drop one here`; a dropped or picked file of the wrong kind: `That isn't <a|an> <kind> file.`; a `previous_result:` row's detail `the result of step <name>`.
  - **Enhance:** a prompt field's button **Enhance…**; the dialog `Enhance the prompt`; its use button **Use this prompt**.
  - **Rerun:** the hint `Each field holds what this job used. Change what should differ.`; the checkbox `New seed`; with nothing changed and New seed off, the Queue button is disabled and the line reads `nothing changed`.
  - **Catalog:** placeholder `search workflows…`; a card's input line `needs <names>` (variable names joined by `, `).
  - **Editor:** read-only banner `This is an example, so it can't be changed in place. Save makes your own copy in <workspace>.` (prompt editor: the same with no change); variable name errors `A variable name starts with a letter or _ and holds letters, numbers, _ and -.` and `There is already a variable called <name>.`; step name errors `A step name can't be empty or hold a space, . or @.` and `Another step is already called <name>.`; the rename toast `Renamed step <old> to <new>` plus `, and <n> <reference|references> to it` when n > 0; the restore toast for a renamed step `Step <old> came back as <new>`; a validation error's button `Go to <step <name>|variable <name>>`.

## Review focus

1. **A reference value the field cannot preview** - an `output:` that names a deleted file, an `asset:` of the wrong kind, a hand-typed URL, `previous_result:` in a run form. The field keeps the text, shows the kind's glyph or nothing, never a broken image, and never rewrites the value. Pinned in Task 5 (`a reference to a missing file keeps its text and shows the glyph`, `a typed URL is kept as typed`).
2. **Renaming a step that other steps reference in every shape** - `previous_result:a`, `previous_result:a.frames`, `previous_result:a@cold_open`, `{"from_previous_result": "a"}`, `gather:a`, and a step called `ab` that must not be touched. Pinned in Task 14 (`rewrites every shape of reference and nothing that only starts with the name`).
3. **The check answering late** - Run pressed while a check is in flight, or the verdict flipping from refused to ready as the person types. Run uses the latest verdict, and a refused verdict that clears re-enables Run without a reload. Pinned in Task 3 (`Run comes back once the check stops refusing`, `a check in flight does not block Run`).
4. **Restored run values that no longer fit** - a remembered value for a variable the workflow no longer declares, or for another workspace's workflow of the same name. Stale keys are dropped and drafts are keyed by workspace and name; a prefill wins over a draft. Pinned in Task 8 (`drops a remembered variable the workflow no longer has`, `a prefill wins over the draft`, `drafts are per workspace`).
5. **An upload that lands after the person moved on** - a slow upload finishing after the field was cleared or the dialog closed. It never writes into a field that changed meanwhile. Pinned in Task 5 (`an upload that finishes after the field changed leaves it alone`).

---

### Task 1: `inputs=true` on the workflow listing (server)

What WF-11's kind filter and WF-14's "needs" line read: which variables feed a media argument, and what kind. Additive and opt-in.

**Files:**
- Modify: `dw/server/catalog_shape.py` (`input_kinds`), `dw/server/catalog.py` (the detail), `dw/server/routes/library.py` (`list_workflows`), `dw/server/api_models_library.py` (`WorkflowCard.inputs`)
- Modify: `ui/src/lib/generated/*` (regenerated only), `ui/src/lib/api.ts` (`listWorkflows`), `ui/src/lib/types.ts` (`InputKind`)
- Test: `tests/test_catalog_inputs.py` (new test file), `tests/test_api_contract.py` and `tests/test_ui_twins.py` (unchanged; must pass)
- Docs: `docs/SERVER.md` (the parameter); ARCHITECTURE, a row in *Server*: *A workflow's inputs* - owner `catalog_shape.input_kinds`; rule "the media kind a variable takes is read from the argument it feeds, by the same `_MEDIA_ARGUMENTS` list the `needs-input-media` trait reads; listed only when asked"; enforced by `tests/test_catalog_inputs.py`

**Interfaces:**
- `catalog_shape._MEDIA_ARGUMENTS` becomes `_MEDIA_ARGUMENT_KINDS: dict[str, str]` (`image`, `mask_image`, `urls` → `image`; `video`, `source_video`, `videos`, `clip` → `video`; `audio`, `reference_audio` → `audio`), and `_needs_input_media` reads its keys, so the trait is unchanged.
- `catalog_shape.input_kinds(steps, variables) -> dict[str, str]`: each `variable:<name>` a step argument in `_MEDIA_ARGUMENT_KINDS` names (one level into a list), not filled inside (`_filled_inside`), → that argument's kind; a `variable:` reference inside a `{"location": …}` object → its `media_type`'s kind (`image/…` → `image`, …), else `file`. First kind wins; sorted by name.
- The catalog detail carries `inputs` always (it is cached); `list_workflows(..., inputs: bool = False)` drops it from every detail unless `inputs` is true. `WorkflowCard.inputs: dict[str, InputKind] = sometimes("Each variable that feeds a media argument, and the kind it takes. Only with inputs=true.")`, `InputKind = Literal["image", "video", "audio", "file"]`.
- `routes/library.py` stays ≤ 1000 lines: fold the parameter into the docstring's existing sentence and drop a blank-line pair, nothing else.
- `api.ts`: `listWorkflows: (opts?: { inputs?: boolean }) => …` with `appendQuery`; still a read. `api.ts` stays ≤ 598 lines (pay with a one-line arrow elsewhere if needed).

- [ ] **Step 1: Write the failing tests** in `tests/test_catalog_inputs.py`:
  - `test_a_variable_feeding_an_image_argument_is_an_image_input` (`{"image": "variable:photo"}` → `{"photo": "image"}`)
  - `test_mask_and_video_and_audio_arguments_name_their_kinds`
  - `test_a_variable_whose_default_is_a_step_result_is_not_an_input` (`_filled_inside`)
  - `test_a_location_object_takes_its_media_type` and `test_a_location_object_without_one_is_a_file`
  - `test_the_listing_says_nothing_about_inputs_unless_asked` (the key is absent from every detail)
  - `test_the_listing_carries_inputs_when_asked` (the inpaint template lists its image and mask variables)
  - `test_the_needs_input_media_trait_is_unchanged` (every workflow under `workflows/` keeps its traits)
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_catalog_inputs.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement**, regenerate OpenAPI as in Global constraints, then `cd ui && npm run gen:api`.
- [ ] **Step 4: Run** the file, `tests/test_api_contract.py`, `tests/test_ui_twins.py`, `tests/test_catalog*.py`, then `wc -l dw/server/routes/library.py ui/src/lib/api.ts` (≤ 1000, ≤ 598), `venv/bin/python scripts/arch_metrics.py` (`modules 281`), `cd ui && npm run check`. Expected: PASS.
- [ ] **Step 5: Commit** `server(catalog): the variables a workflow takes files through, and their kinds, when asked`.

### Task 2: One verdict and one plan, for the run check and the editor

WF-10, the engine-text half of WF-21, D9, and the run-page half of JB-13.

**Files:**
- Create: `ui/src/lib/workflow/PlanLines.svelte`, `ui/src/lib/workflow/PlanLines.test.ts`, `ui/src/lib/findings.test.ts`
- Modify: `ui/src/lib/runCheck.svelte.ts` (`verdictOf`), `ui/src/lib/plan.ts` (`planSummary`), `ui/src/lib/findings.ts` (`authorNotes`), `ui/src/lib/workflow/RunCheckPanel.svelte`, `ui/src/lib/editor/ValidationPanel.svelte`
- Test: `ui/src/lib/runCheck.test.ts`, `ui/src/lib/plan.test.ts`, `ui/src/lib/workflow/RunCheckPanel.test.ts`, `ui/src/lib/editor/ValidationPanel.test.ts`
- Docs: ARCHITECTURE *The UI reads engine fields* adds `verdictOf` and `planSummary` (they read `valid`, `errors`, `access_blocked`, `not_found` and the estimate, and derive nothing else)

**Interfaces:**
- `runCheck.svelte.ts`: `export type Verdict = { tone: 'good' | 'warn' | 'bad'; text: string; remedy?: string; issues: number }` and `export function verdictOf(validation: ValidationResult | null, error: unknown, placed: PlacedFindings): Verdict | null`, with the Global constraints copy, checked in this order: error, not valid (`issues` = errors + bare refusal), gated, not found, a placed warning, ready. `RunCheck.verdict` becomes a getter that is not used any more and is removed; pages call `verdictOf`.
- `plan.ts`: `export function planSummary(plan: Plan): string` (`describeEstimateTotal`'s text with its `~` free wording, `formatApprox`, plus the download half).
- `findings.ts`: `PlacedFindings` gains `author: Finding[]` - the warnings that name no shown variable. `general` keeps only errors and the bare refusal. Placement rules are otherwise unchanged.
- `PlanLines.svelte` props `{ plan: Plan; open?: boolean }`: the summary line, then `Details` (`What a job will do`) holding `describePlan`'s lines joined by `\n`. The `.plan` list CSS goes (the D9 duplicate).
- `RunCheckPanel` props become `{ verdict: Verdict | null; checking: boolean; plan: Plan | null; checkError: unknown; findings: Finding[]; author: Finding[] }`: the verdict lead and remedy, `PlanLines`, each error finding with an `id="issue-<i>"` (Task 3 scrolls to them), then the author `Details`.
- `ValidationPanel` uses `verdictOf` and `PlanLines`; in the editor every warning stays visible (the reader is the author), so it passes no `author` split.

- [ ] **Step 1: Write the failing tests.**
  - `runCheck.test.ts`: `a refused check counts its issues`, `a gated download is its own verdict with the hf remedy`, `a repo that isn't there says so`, `a note under a filled field asks to read it`, `notes for the author alone read Ready to run`.
  - `plan.test.ts`: `the summary names the estimate and the download size`, `no download, no download half`.
  - `findings.test.ts`: `a warning naming a shown variable goes under it`, `a warning naming nothing is the author's`, `errors are never the author's`.
  - `PlanLines.test.ts`: `shows the summary and keeps the lines behind What a job will do`.
  - `RunCheckPanel.test.ts`, `ValidationPanel.test.ts`: `both say the same verdict for the same answer`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/runCheck.test.ts src/lib/plan.test.ts src/lib/findings.test.ts src/lib/workflow/ src/lib/editor/ValidationPanel.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** WorkflowPage passes `verdictOf(runCheck.validation, runCheck.error, placed)` and `placed.author`.
- [ ] **Step 4: Run** the tests, `npx vitest run src/lib/pages/WorkflowPage.test.ts src/lib/pages/EditorPage.test.ts`, and the metrics gate. Expected: PASS; `duplicate_blocks` below 18. Lower it in `baseline.json`.
- [ ] **Step 5: Commit** `ui(run): one verdict and one plan, in plain words, for the run check and the editor`.

### Task 3: Run waits for a check that refuses

WF-8, decision 6.

**Files:**
- Modify: `ui/src/lib/pages/WorkflowPage.svelte`
- Test: `ui/src/lib/pages/WorkflowPage.test.ts`

**Interfaces:**
- Consumes: Task 2's `verdictOf` (`issues`) and the `issue-<i>` ids; findings under a field keep `data-findings-for`.
- When the latest verdict has `tone: 'bad'` and `issues > 0`, the Run button reads `Fix <n> <issue|issues>` (Global constraints) and its press scrolls the first issue into view (a field's finding first, in form order, else `#issue-0`) and focuses its field. A gated or not-found verdict leaves Run as **Run**. `running.busy` and `!workflow` disable it as today. Nothing else gates it: no verdict yet, a check in flight or a failed check leave Run enabled.

- [ ] **Step 1: Write the failing tests** in `WorkflowPage.test.ts`: `a refused check turns Run into Fix 2 issues, which focuses the first field`, `Run comes back once the check stops refusing`, `a check in flight does not block Run`, `a gated verdict keeps Run`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/WorkflowPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the test and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(run): a refused check turns Run into Fix N issues`.

### Task 4: Suggestion rows draw their pictures through `OutputTile`

Half of the `raw_media` carry-over; folds #1600 item 5 and #1606 item 15.

**Files:**
- Modify: `ui/src/lib/ui/suggest.ts` (`SuggestItem.media`), `ui/src/lib/ui/Suggest.svelte`, `ui/src/lib/ui/OutputTile.svelte` (`size: 'row'`), `ui/src/patterns.css` (`.filetile.row`), `ui/src/lib/refcomplete.svelte.ts`, `ui/src/lib/assets/MediaPicker.svelte`
- Test: `ui/src/lib/ui/Suggest.test.ts`, `ui/src/lib/refcomplete.test.ts`, `ui/src/lib/ui/OutputTile.test.ts`, `ui/src/lib/assets/MediaPicker.test.ts`

**Interfaces:**
- `SuggestItem` drops `thumb` and `glyph` and gains `media?: MediaFile`. `Suggest` renders `<OutputTile file={item.media} size="row" caption="none" alt="" />` for a row with `media`.
- `OutputTile`'s `size` gains `'row'`: 32 px square, no caption, no badges.
- `refcomplete`: `assetChoices` and `outputRow` build `media` with `assetMedia(asset, assetLibrary.workspace ?? workspace.current)` and `outputMedia(file, outputLibrary.workspace ?? workspace.current)`; `hasThumbnail` is deleted (use `hasPicture`).
- `MediaPicker`: the collapse key is `collapsed-media-picker:<workspace>:<tab>`; while `query` is not empty every group is open and its `GroupHeading` is `disabled`, titled `Showing every match while the search is on`; the current tile carries `active` and a visible `current` badge, so it no longer looks like hover.

- [ ] **Step 1: Write the failing tests.**
  - `Suggest.test.ts`: `a row with a picture draws it as a tile`, `a row without one draws its glyph`.
  - `refcomplete.test.ts`: `an output row's thumbnail names the listing's workspace`.
  - `OutputTile.test.ts`: `a row tile has no caption`.
  - `MediaPicker.test.ts`: `group toggles are disabled while searching`, `the collapse state is per workspace and tab`, `the current tile says current`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/ui/Suggest.test.ts src/lib/refcomplete.test.ts src/lib/ui/OutputTile.test.ts src/lib/assets/MediaPicker.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/lora-suggestions.spec.ts`, and the metrics gate. Expected: PASS; `raw_media` 3. Lower it in `baseline.json`.
- [ ] **Step 5: Commit** `ui(media): suggestion rows draw their pictures as tiles; the picker keeps its folds per tab`.

### Task 5: `ReferenceField`

The building block. P8; replaces `MediaArgumentInput`.

**Files:**
- Create: `ui/src/lib/ui/ReferenceField.svelte`, `ui/src/lib/ui/ReferenceField.test.ts`, `ui/src/lib/referenceField.ts`, `ui/src/lib/referenceField.test.ts`
- Delete: `ui/src/lib/editor/MediaArgumentInput.svelte` and its test (its cases move to `ReferenceField.test.ts`)
- Modify: `ui/src/lib/mediaArguments.ts` (`MediaKind` gains `'audio'`)
- Docs: ARCHITECTURE, a UI row *Reference input*: owners `ReferenceField.svelte`, `referenceField.ts`; rule "a media or reference value is edited only through ReferenceField - run form, rerun dialog, step editor; earlier steps' results are offered first; the preview is an OutputTile"; enforced by the two test files and `raw_media`. *UI overlays and suggestions* names `ReferenceField` beside `Suggest`.

**Interfaces:**
- `referenceField.ts` (pure):
  - `export type FieldKind = MediaKind | 'file'`
  - `export function earlierResults(workflow: Record<string, any>, stepIndex: number, kind?: FieldKind): string[]`: `referenceSuggestions`' `previous_result:` entries for steps before `stepIndex`, kept when the step's `result.content_type` starts with `kind`, or for a video step its `.frames` when `kind` is `image` and `.audio` when `audio`; every one when `kind` is absent or `file`.
  - `export function fieldRows(text: string, kind: FieldKind | undefined, earlier: string[]): SuggestItem[] | null`: an empty field or a `previous_result:` prefix lists `earlier` first (detail per Global constraints), then `referenceCompletions(text, kind)`'s rows; otherwise `referenceCompletions(text, kind)`.
  - `export function previewOf(value: string, kind: FieldKind | undefined): MediaFile | null`: an `asset:` → `assetMedia` of the listed asset, an `output:` → `outputMedia`, an `http(s)://` or `/` URL → a `MediaFile` with `thumb` the URL for an image and none otherwise, anything else (`previous_result:`, `variable:`, unknown names) → null.
- `ReferenceField.svelte` props: `{ id: string; value: string; kind?: FieldKind; earlier?: string[]; placeholder?: string; onchange: (value: string) => void; oninput?: (value: string) => void; children?: Snippet }`.
  - A `Suggest` with `complete={(t) => fieldRows(t, kind, earlier ?? [])}`, **Choose…** opening `MediaPicker` on `kind`, **Upload…** with a hidden file input `accept="<kind>/*"` and `dropFiles`, then the preview as `OutputTile size="strip"` from `previewOf` (a just-picked file shows its blob as `thumb` while uploading).
  - The upload is `MediaArgumentInput`'s (`uploadAll`, create-only, `ErrorLine` on failure), keyed to the value it started from: when it lands and the field no longer holds that value, it is reported in a toast and not written.
  - A file whose MIME type is not the kind: the Global constraints line, nothing uploaded.
  - `children` renders after the buttons (Task 9's **Enhance…**).

- [ ] **Step 1: Write the failing tests.**
  - `referenceField.test.ts`: `offers earlier results of the kind first`, `offers a video step's frames to an image field`, `an empty field lists earlier results before the prefixes`, `previews an asset, an output and an image URL`, `previews nothing for previous_result`.
  - `ReferenceField.test.ts`: the moved `MediaArgumentInput` cases, plus `a reference to a missing file keeps its text and shows the glyph`, `a typed URL is kept as typed`, `an upload that finishes after the field changed leaves it alone`, `refuses a dropped file of the wrong kind`, `the preview is a tile, never a raw img or video`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/referenceField.test.ts src/lib/ui/ReferenceField.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `ArgumentsEditor` keeps compiling by importing `ReferenceField` where it imported `MediaArgumentInput` with `kind` and no `earlier`; Task 6 does the rest.
- [ ] **Step 4: Run** the tests, `npx vitest run src/lib/editor/`, and the metrics gate. Expected: PASS; `raw_media` 1. Lower it in `baseline.json`.
- [ ] **Step 5: Commit** `ui(reference): ReferenceField, one input for a file or a reference`.

### Task 6: The step editor takes references through `ReferenceField`

WF-6.

**Files:**
- Modify: `ui/src/lib/editor/ArgumentsEditor.svelte`, `ui/src/lib/editor/StepEditor.svelte` (passes `workflow` and `index`), `ui/src/lib/editor/StepList.svelte`, `ui/src/lib/pages/EditorPage.svelte` (passes the document down; no new logic)
- Test: `ui/src/lib/editor/ArgumentsEditor.test.ts`, `ui/src/lib/editor/StepEditor.test.ts`

**Interfaces:**
- Consumes: Task 5's `ReferenceField`, `earlierResults`.
- `ArgumentsEditor` gains `earlier: (kind?: FieldKind) => string[]`. A value is a `ReferenceField` when `mediaKindFor(parameter, key)` names a kind - **whether or not it already holds a reference**, so `previous_result:x` stays in the same field - or when it is a reference string (`isReference`); its `kind` is the media kind or undefined. `withMediaLocation` still keeps an object value's shape. Other text stays `Suggest`; `BrowseButton` is no longer used here.
- `StepEditor` builds `earlier` from `earlierResults(workflow, index, kind)`.

- [ ] **Step 1: Write the failing tests.** `ArgumentsEditor.test.ts`: `an image argument offers the earlier step's result first`, `an image argument holding previous_result stays a reference field`, `a video step's frames are offered to an image argument`, `a later step's result is never offered`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/editor/ArgumentsEditor.test.ts src/lib/editor/StepEditor.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/step-model.spec.ts e2e/smoke.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(editor): media arguments offer earlier steps' results first, and keep them`.

### Task 7: The run form and the rerun dialog take files through `ReferenceField`

WF-11; folds #1601's stale `~` in `WorkflowPage.svelte` and #1602's back-label test.

**Files:**
- Modify: `ui/src/lib/editor/VariablesForm.svelte` (`kinds`), `ui/src/lib/pages/WorkflowPage.svelte`, `ui/src/lib/job/RerunWithChanges.svelte`
- Delete: `ui/src/lib/assets/BrowseButton.svelte` (no user left)
- Test: `ui/src/lib/editor/VariablesForm.test.ts`, `ui/src/lib/pages/WorkflowPage.test.ts`, `ui/src/lib/job/RerunWithChanges.test.ts`, `ui/src/lib/freshness.test.ts`

**Interfaces:**
- Consumes: Task 1's `inputs`, Task 5's `ReferenceField`.
- `VariablesForm` gains `kinds?: Record<string, FieldKind>`. In override mode a variable in `kinds`, or one whose value or default is an `asset:`/`output:` reference, is a `ReferenceField` with that kind (no `earlier`; a run form has no steps before it). In define mode a variable holding such a reference is a `ReferenceField` with no kind. Everything else is as today.
- `WorkflowPage`'s `catalog` resource reads `api.listWorkflows({ inputs: true })` and passes `card?.inputs ?? {}`; the Variables hint drops its `prompt:`/`asset:`/`output:` sentence for `Leave a field blank to use the workflow's own default.`
- `RerunWithChanges` reads `api.listWorkflows({ inputs: true })` as a `resource` (`watch: ['workflows']`, `lazy` until opened) and passes `details[job.workflow_name]?.inputs`.
- `WorkflowPage.test.ts`'s back-label test asserts `← Workflows` on a workspace route and `← Examples` on a shared one, separately.

- [ ] **Step 1: Write the failing tests.**
  - `VariablesForm.test.ts`: `an image input is a reference field filtered to images`, `an audio input offers only audio`, `a variable holding an asset is a reference field without a kind`.
  - `WorkflowPage.test.ts`: `the inpaint form offers upload and browse for its image and mask`, `the back link names Workflows here and Examples there`.
  - `RerunWithChanges.test.ts`: `an image variable can be swapped through the picker`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/editor/VariablesForm.test.ts src/lib/pages/WorkflowPage.test.ts src/lib/job/RerunWithChanges.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, `npx vitest run src/lib/freshness.test.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(run): the run form and the rerun dialog take files by upload, browse or reference`.

### Task 8: The run form: Essentials first, and what you typed is kept

WF-21 (the Essentials/Advanced half), WF-12, and the runner-facing half of WF-9.

**Files:**
- Create: `ui/src/lib/workflow/runForm.ts`, `ui/src/lib/workflow/runForm.test.ts`
- Modify: `ui/src/lib/editor/VariablesForm.svelte` (`advanced`), `ui/src/lib/pages/WorkflowPage.svelte`
- Test: `ui/src/lib/pages/WorkflowPage.test.ts`

**Interfaces:**
- `runForm.ts`:
  - `export function essentials(workflow: WorkflowDefinition, card: WorkflowCard | undefined): Set<string>`: required variables, `card.inputs`' keys, `card.cost_drivers`' keys, preset selectors (`presetView`), and free text - a string default that is empty, holds whitespace, or is a `prompt:` reference. *(Question 2: if Don picks another rule, only this function changes.)*
  - `export function seedVariable(workflow: WorkflowDefinition): string | null`: the name `workflow.seed`'s `variable:` reference names.
  - `export function draftKey(workspace: string, name: string): string` (`run-draft:<workspace>:<name>`), `export function readDraft(key: string, declared: string[]): Record<string, string>` (drops keys not in `declared` and blanks), `export function writeDraft(key: string, overrides: Record<string, string>): void` (removes the entry when all blank), through `storage.ts`.
- `VariablesForm` gains `advanced?: Set<string>`: those keys render after the rest under a `GroupHeading` (`Advanced (<n>)`, collapsed), opened when any of them holds an override or a finding. A seed variable (`seedVariable`) with a null default has the Global constraints placeholder.
- `WorkflowPage`: on a name change `overrides = { ...readDraft(key, names), ...takePrefill() }`; when `readDraft` returned anything, the hint and **Reset to defaults** (clears `overrides` and the draft). Every override change writes the draft (debounced 300 ms). A Run keeps the draft.

- [ ] **Step 1: Write the failing tests.**
  - `runForm.test.ts`: `required, inputs, cost drivers, selectors and free text are essentials`, `dtype and backend strings are advanced`, `finds the seed variable`, `drops a remembered variable the workflow no longer has`, `drafts are per workspace`.
  - `WorkflowPage.test.ts`: `advanced variables are folded until one holds a value`, `a seed with no default says random each job`, `values typed here are back on the next visit`, `a prefill wins over the draft`, `Reset to defaults clears the form and the draft`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/workflow/runForm.test.ts src/lib/pages/WorkflowPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `WorkflowPage` stays below 470 lines; logic stays in `runForm.ts`.
- [ ] **Step 4: Run** the tests and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(run): essentials first, advanced folded, and the form keeps what you typed`.

### Task 9: Enhance on the run form

WF-24; folds #1600 item 6 and the EnhancePanel prettier item.

**Files:**
- Create: `ui/src/lib/workflow/EnhanceDialog.svelte`, `ui/src/lib/workflow/EnhanceDialog.test.ts`
- Modify: `ui/src/lib/editor/EnhancePanel.svelte` (job link workspace), `ui/src/lib/enhancer.svelte.ts` (`jobWorkspace`), `ui/src/lib/editor/VariablesForm.svelte` (`onenhance`), `ui/src/lib/pages/WorkflowPage.svelte`
- Test: `ui/src/lib/editor/EnhancePanel.test.ts`, `ui/src/lib/pages/WorkflowPage.test.ts`

**Interfaces:**
- `Enhancer` gains `jobWorkspace = $state('')`, set from the submitted job's `workspace`; `EnhancePanel` links `jobHref(enhancer.jobId, enhancer.jobWorkspace)`.
- `EnhanceDialog.svelte` props `{ open: boolean (bindable); idea: string; onuse: (text: string) => void }`: a `Modal` labelled `Enhance the prompt`, owning one `Enhancer` (loaded when first opened, `stop()` on destroy), its `idea` set from `idea` on open, `EnhancePanel` inside; `EnhancePanel`'s `onuse` calls `onuse(enhancer.takeResult())` and closes.
- `VariablesForm` gains `onenhance?: (key: string) => void`: free-text variables (Task 8's free-text rule, exported from `runForm.ts` as `isFreeText(value: unknown): boolean`) show **Enhance…** after their field. `WorkflowPage` opens one `EnhanceDialog` for the key and writes the result into `overrides[key]`.

- [ ] **Step 1: Write the failing tests.** `EnhanceDialog.test.ts`: `opens with the field's text as the idea`, `Use this prompt writes the result back and closes`. `EnhancePanel.test.ts`: `the job link names the job's own workspace`. `WorkflowPage.test.ts`: `only free-text fields offer Enhance`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/workflow/EnhanceDialog.test.ts src/lib/editor/EnhancePanel.test.ts src/lib/pages/WorkflowPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement**, then `npx prettier --write src/lib/editor/EnhancePanel.svelte`.
- [ ] **Step 4: Run** the tests and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(run): Enhance a prompt from the run form`.

### Task 10: Run again with changes starts from the job's values

JB-5.

**Files:**
- Create: `ui/src/lib/workflow/typedArguments.ts`, `ui/src/lib/workflow/typedArguments.test.ts`
- Modify: `ui/src/lib/job/RerunWithChanges.svelte`, `ui/src/lib/pages/WorkflowPage.svelte` (`runArguments` uses `typedArguments`)
- Test: `ui/src/lib/job/RerunWithChanges.test.ts`, `ui/src/lib/pages/WorkflowPage.test.ts`

**Interfaces:**
- `typedArguments(overrides: Record<string, string>, declared: Record<string, unknown>): Record<string, unknown>`: blanks dropped; each value through `coerce(widgetFor(undefined, declared[key]), raw)`, so a number default sends a number, a boolean a boolean, an object or list parsed JSON; a reference stays a string.
- `RerunWithChanges`: opening fills `overrides` with `displayValue(values[key])` for every key (the job's values, not placeholders); `changed` compares each against that start; `newSeed = $state(false)` is a `New seed` checkbox; Queue sends `api.rerunJob(job.id, { newSeed, arguments: typedArguments(changed, values) })` and is enabled when `changedCount > 0 || newSeed`. The hint is the Global constraints copy.

- [ ] **Step 1: Write the failing tests.** `typedArguments.test.ts`: `a number default sends a number`, `a reference stays text`, `a blank is dropped`. `RerunWithChanges.test.ts`: `opens holding the job's values`, `sends only what changed, typed`, `New seed alone queues a rerun with a new seed`, `nothing changed and no new seed keeps Queue disabled`. `WorkflowPage.test.ts`: `Run sends 30 as a number where the default is a number`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/workflow/typedArguments.test.ts src/lib/job/RerunWithChanges.test.ts src/lib/pages/WorkflowPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(job): Run again with changes starts from the job's values, with New seed`.

### Task 11: The job page keeps the author's notes folded

JB-13 (and WF-9 as a person running a job sees it).

**Files:**
- Modify: `ui/src/lib/pages/JobPage.svelte:338-344` and its `.warnings` CSS
- Test: `ui/src/lib/pages/JobPage.test.ts`

**Interfaces:**
- Consumes: Task 2's `placeFindings(… ).author`.
- The job's `warnings` go through `placeFindings({ valid: true, errors: [], warnings: job.warnings, error: null }, Object.keys(job.arguments ?? {}))`: a warning naming an argument this job was given is a `.warn` line, as today; the rest are one `Details` with the Global constraints author summary, collapsed. No warn-edge panel when only author notes remain.

- [ ] **Step 1: Write the failing tests** in `JobPage.test.ts`: `the seed warning is folded under the author's notes`, `a warning about an argument this job was given stays in view`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/JobPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `JobPage` does not grow past 502.
- [ ] **Step 4: Run** the test and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(job): notes for a workflow's author fold away on the job page`.

### Task 12: The catalog finds by words, counts, and says what a workflow needs

WF-13, WF-14; folds #1600 item 7 and #1601's `~` in `WorkflowsPage.svelte`.

**Files:**
- Create: `ui/src/lib/workflow/catalogSearch.ts`, `ui/src/lib/workflow/catalogSearch.test.ts`
- Modify: `ui/src/lib/pages/WorkflowsPage.svelte`
- Test: `ui/src/lib/pages/WorkflowsPage.test.ts`

**Interfaces:**
- `catalogSearch.ts`: `export function catalogMatches(names: string[], details: Record<string, WorkflowCard>, query: string): string[]`. The query is split into words, lower-cased, with the stop words `a an and for of on the to with` dropped. Each workflow's text is its name, description, summary, shape, kinds and traits. A workflow's score is how many words its text holds; the result is the workflows with the highest score above 0, best first, then by name - so `video of a cat` lists the video workflows and `flux image` only those holding both. An empty query (after stop words) returns `names`. *(Question 3.)*
- `WorkflowsPage`: `visible` uses `catalogMatches` after the shape and trait filters; the listing is read with `{ inputs: true }`; the count is `{ n: visible.length, of: listed.length, noun }`; the placeholder is `search workflows…`; the card's `N vars` becomes `needs <names>` from `Object.keys(detail.inputs ?? {})`, or nothing. The shape `Segmented` keeps a second press on the selected shape as the clear, and the `all` option shows as selected after it (the #1600 cue).

- [ ] **Step 1: Write the failing tests.** `catalogSearch.test.ts`: `video of a cat finds the video workflows`, `every matched word ranks higher`, `stop words alone match everything`, `matches traits and kinds`. `WorkflowsPage.test.ts`: `the count reads n of m while searching`, `a card says what it needs`, `a card with no inputs says nothing about variables`, `a second press on a shape clears it to all`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/workflow/catalogSearch.test.ts src/lib/pages/WorkflowsPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `WorkflowsPage` shrinks.
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/journeys/consistency.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(catalog): search by words, count n of m, and cards that say what they need`.

### Task 13: Variables in the editor: Enter, names and types

WF-17; folds the VariablesForm prettier item.

**Files:**
- Create: `ui/src/lib/editor/variableTypes.ts`, `ui/src/lib/editor/variableTypes.test.ts`
- Modify: `ui/src/lib/references.ts` (`VARIABLE_NAME_CHARACTERS`), `ui/src/lib/editor/VariablesForm.svelte`, `tests/test_ui_twins.py`
- Test: `ui/src/lib/editor/VariablesForm.test.ts`

**Interfaces:**
- `references.ts`: `export const VARIABLE_NAME_CHARACTERS = 'abc…XYZ0123456789_-'`, the engine's string; `tests/test_ui_twins.py::test_the_ui_variable_name_characters_are_the_engines` pins it.
- `variableTypes.ts`:
  - `export type VariableType = 'text' | 'number' | 'boolean' | 'json' | 'none'`
  - `export function variableNameProblem(name: string, taken: string[]): string | null`: the Global constraints copy, or null.
  - `export function typeOf(value: unknown): VariableType` (`null` → `none`, an object or list → `json`).
  - `export function inferValue(raw: string): unknown`: `true`/`false`, a finite number, `null`, JSON starting `{` or `[` that parses; else the text. A reference is text.
  - `export function asType(raw: string, type: VariableType): unknown`.
- `VariablesForm` define mode: the new-name input adds on Enter and on the button, refusing (inline `.error`, `aria-invalid`) when `variableNameProblem` says so; a new variable's first commit uses `inferValue`; each row has a small `select` (`aria-label="type of <name>"`) of the five types that re-coerces the value with `asType`; later commits coerce to the row's type.

- [ ] **Step 1: Write the failing tests.** `variableTypes.test.ts`: `names follow the engine's rule`, `a taken name is refused`, `infers numbers, booleans, null and JSON`, `a reference stays text`. `VariablesForm.test.ts`: `Enter adds a variable`, `a bad name says why and adds nothing`, `30 in a new variable is stored as 30`, `the type select turns 30 back into text`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/editor/variableTypes.test.ts src/lib/editor/VariablesForm.test.ts` and `venv/bin/python -m pytest tests/test_ui_twins.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement**, then `npx prettier --write src/lib/editor/VariablesForm.svelte`.
- [ ] **Step 4: Run** the tests and the metrics gate. Expected: PASS; `VariablesForm` stays below 400 lines.
- [ ] **Step 5: Commit** `ui(editor): variables add on Enter, follow the engine's name rule and keep their types`.

### Task 14: Renaming a step carries its references

WF-7; folds #1600 items 3, 4 and 8.

**Files:**
- Create: `ui/src/lib/stepRename.ts`, `ui/src/lib/stepRename.test.ts`
- Modify: `ui/src/lib/editor/StepEditor.svelte` (the name field), `ui/src/lib/editor/StepList.svelte` (`renameStep`, undo), `ui/src/lib/stepModes.svelte.ts` (`rename`)
- Test: `ui/src/lib/editor/StepList.test.ts`, `ui/src/lib/stepModes.test.ts`

**Interfaces:**
- `stepRename.ts`:
  - `export function stepNameProblem(name: string, others: string[]): string | null` (Global constraints copy).
  - `export function renameStep(workflow: Record<string, any>, from: string, to: string): { workflow: Record<string, any>; count: number }`: a new document where every string `previous_result:<from>` - exactly, or followed by `.` or `@` - and `gather:<from>` (same rule) has `<from>` replaced by `<to>`, every `from_previous_result` value equal to `<from>` (or starting `<from>.`/`<from>@`) likewise, and the step's own `name` set; `count` is the references changed. Walk with the reference prefixes from `references.ts`; nothing that only starts with `<from>` (`<from>x`) changes.
- `StepModes.rename(from: string, to: string): void` moves the mode.
- `StepEditor`'s name field is a plain `value` with an `onchange` (not `bind:value`): a problem shows inline and keeps the old name; otherwise `onrename(to)`. `StepList.onrename` calls `renameStep`, assigns the steps and variables back, `modes.rename`, and toasts the Global constraints copy. The field gets `max-width: 480px` (#1600 item 8).
- `StepList.removeStep`'s Undo puts the step back before the step that followed it, if still present (as `VariablesForm` does), and toasts `Step <old> came back as <new>` when `freeName` changed it.

- [ ] **Step 1: Write the failing tests.** `stepRename.test.ts`: `rewrites every shape of reference and nothing that only starts with the name` (all five shapes from *Review focus* 2, and a step `ab`), `counts what it changed`, `refuses an empty name, a dot, an @ and a space`, `refuses a taken name`. `StepList.test.ts`: `a rename keeps the step's mode and its references`, `undo puts a step back beside its old neighbour`, `a step that comes back renamed says so`. `stepModes.test.ts`: `rename moves the mode`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/stepRename.test.ts src/lib/editor/StepList.test.ts src/lib/stepModes.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/smoke.spec.ts`, the metrics gate, and a 1440 px screenshot of a step head (`node scripts/ux-capture.mjs` or the smoke run) to check the name field's width by eye.
- [ ] **Step 5: Commit** `ui(editor): renaming a step carries every reference to it`.

### Task 15: Component slots come from the pipeline

WF-19.

**Files:**
- Create: `ui/src/lib/editor/componentSlots.ts`, `ui/src/lib/editor/componentSlots.test.ts`
- Modify: `ui/src/lib/editor/PipelineOptions.svelte`
- Test: `ui/src/lib/editor/PipelineOptions.test.ts`

**Interfaces:**
- `componentSlots.ts`: `export function slotsFrom(description: ClassDescription | null): string[] | null`: the `__init__` parameters of `classDescription(component_type, 'init')` whose annotation or doc type names a class (not `bool`, `int`, `float`, `str`, `dict`, `list`, none of them `Optional[...]` of those), in signature order; null when the description is null or `accepts_kwargs` (a modular pipeline), so the caller falls back to `COMPONENT_SLOTS`.
- `PipelineOptions` reads the description as a `resource` debounced like `ArgumentsEditor`'s (`latestOnly`, 300 ms, `watch: []`); the add-slot select lists `slotsFrom(...) ?? COMPONENT_SLOTS` minus active ones; an active slot the pipeline does not list stays shown, marked `not a component of <class>` in `.warn`.

- [ ] **Step 1: Write the failing tests.** `componentSlots.test.ts`: `lists the constructor's component parameters`, `skips flags and numbers`, `a modular pipeline falls back`. `PipelineOptions.test.ts`: `offers the slots StableDiffusionPipeline takes`, `a slot the pipeline lacks is marked`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/editor/componentSlots.test.ts src/lib/editor/PipelineOptions.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, `venv/bin/python -m pytest tests/test_ui_twins.py -q` (`COMPONENT_SLOTS` stays the schema's), and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(editor): component slots come from the pipeline's constructor`.

### Task 16: LoRA rows are labelled

WF-20.

**Files:**
- Modify: `ui/src/lib/editor/LorasEditor.svelte:140-165`
- Test: `ui/src/lib/editor/LorasEditor.test.ts`

**Interfaces:**
- Each row is a `.formgrid` of three labelled fields - `Model`, `Weight file`, `Scale` - with ids from `$props.id()` and the row index; the placeholders stay as examples; the remove button's label names the row's model (`remove <model_name>` or `remove this LoRA` when blank).

- [ ] **Step 1: Write the failing test** `every LoRA field has a label`, `remove names the LoRA`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/editor/LorasEditor.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the test, `npx playwright test e2e/lora-suggestions.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(editor): LoRA rows are labelled`.

### Task 17: The editors say when they hold an example

WF-22; folds #1601 items 10 and 11 and #1610 item 17.

**Files:**
- Modify: `ui/src/lib/editor/DocumentFileBar.svelte`, `ui/src/lib/pages/EditorPage.svelte` (back label), `ui/src/lib/pages/PromptEditorPage.svelte` (drops its own read-only line if any)
- Test: `ui/src/lib/editor/DocumentFileBar.test.ts`, `ui/src/lib/pages/EditorPage.test.ts`, `ui/src/lib/pages/PromptEditorPage.test.ts`

**Interfaces:**
- `DocumentFileBar` renders, when `ed.loadState === 'loaded'` and `!ed.writable`, one `.hint` banner with the Global constraints copy, `<workspace>` being `workspace.current`. Its ids become `doc-folder`/`doc-savename`, and its copy names "this file", not "the workflow".
- `EditorPage`'s back label is the full path (`name`), matching the h1.
- `EditorPage.test.ts`: the primary-button pin also checks dirty, an example (builtin) and running.

- [ ] **Step 1: Write the failing tests.** `DocumentFileBar.test.ts`: `an example says Save makes your own copy`, `a workspace file says nothing`. `PromptEditorPage.test.ts`: `an example prompt says so`. `EditorPage.test.ts`: `the back link is the full path`, `Run is the primary button when dirty, on an example and while running`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/editor/DocumentFileBar.test.ts src/lib/pages/EditorPage.test.ts src/lib/pages/PromptEditorPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Update every selector that names `wf-folder`/`wf-savename` (`grep -rn "wf-folder\|wf-savename" src e2e`).
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/journeys/editor-safety.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(editor): an example says that Save makes your own copy`.

### Task 18: Validation is live, and each error goes to its place

WF-23; folds #1600 items 1 and 2 and #1602 item 13.

**Files:**
- Create: `ui/src/lib/findingPlace.ts`, `ui/src/lib/findingPlace.test.ts`
- Modify: `ui/src/lib/editor/ValidationPanel.svelte` (`onjump`), `ui/src/lib/pages/EditorPage.svelte` (a `RunCheck` over the document; `save()`'s post-save steps), `ui/src/lib/editorShell.svelte.ts` (`editorKeys`)
- Test: `ui/src/lib/editor/ValidationPanel.test.ts`, `ui/src/lib/pages/EditorPage.test.ts`, `ui/src/lib/editorShell.test.ts`

**Interfaces:**
- `findingPlace.ts`: `export type Place = { kind: 'step'; index: number; name: string } | { kind: 'variable'; name: string }` and `export function findingPlace(path: string | null, doc: Record<string, any>): Place | null`: `steps[<i>]…` → that step (null when the index is out of range), `variables.<name>…` or `arguments.<name>…` → that variable when declared, else null.
- `ValidationPanel` gains `doc` and `onjump: (place: Place) => void`: an error with a place renders as its text plus a `Go to …` button.
- `EditorPage`: `const check = new RunCheck((sizes, signal) => api.validate({ workflow: $state.snapshot(ed.doc), sizes }, signal))`, `check.changed(ed.saveName || 'new', snapshot)` from an `$effect` over the document while `ed.loadState === 'loaded'`, `onDestroy(check.stop)`; the panel shows `check.validation` (the Validate button still forces one through `validate()`, which also sets it). `onjump` for a step: `modes.set(step, 'full')`, then `focusStep(name)`; for a variable: focus `#wfvar-<name>`.
- `save()`'s post-save steps (`catalog.adopt`, `renameRoute`) are in a `try`, reported with `notify.failure('update the editor after saving', e)`.
- `editorKeys`: Ctrl/Cmd+S is `preventDefault`ed before the overlay and composition checks; it then acts only when no overlay is open and nothing is composing.

- [ ] **Step 1: Write the failing tests.** `findingPlace.test.ts`: `places a step path`, `places a variable path`, `an index past the end places nothing`. `ValidationPanel.test.ts`: `an error with a place offers Go to`. `EditorPage.test.ts`: `a change is checked without pressing Validate`, `Go to opens the step and scrolls to it`, `a failure after a save is reported, not thrown`. `editorShell.test.ts`: `Ctrl+S under a dialog never reaches the browser`, `a read-only document ignores Ctrl+S`, `a busy skip still prevents the default`, `an unrelated Ctrl key is left alone`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/findingPlace.test.ts src/lib/editor/ValidationPanel.test.ts src/lib/pages/EditorPage.test.ts src/lib/editorShell.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `EditorPage` ends at or below 509 lines.
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/journeys/editor-safety.spec.ts e2e/smoke.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(editor): validation is live, and each error goes to its step or variable`.

### Task 19: The seed warning tells the truth (only if Don answers Question 4 yes)

WF-9's engine half. Skip this task, and say so at the phase exit, unless Question 4 is answered yes. Question 5 (a default seed in the templates) is separate; its answer adds Step 3b only if yes.

**Files:**
- Modify: `dw/step_cache.py` (`unseeded_cache_warnings`)
- Test: `tests/test_step_cache.py` (or the file that holds `unseeded_cache_warnings`' tests today; `grep -rln unseeded_cache_warnings tests`)

**Interfaces:**
- When `seed` is `variable:<name>` and that variable's value is null: `"The seed variable '<name>' has no value, so each run draws a random seed and the step cache is off - every step regenerates on every run. Give '<name>' a default, or pass one, to make a repeat run reuse what it already produced"`. When `seed` is absent, the current text. The return shape and the silence for task-only workflows are unchanged.

- [ ] **Step 1: Write the failing tests** `test_a_seed_variable_with_no_value_is_named` and `test_no_seed_at_all_keeps_the_original_warning`.
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_step_cache.py -q -k seed`. Expected: FAIL.
- [ ] **Step 3: Implement.** Step 3b (only on Question 5 yes): set `"seed": 42` in `workflows/templates/text-to-image.json` and the editor's blank workflow (`emptyWorkflow` in `ui/src/lib/editor.ts`), and run `venv/bin/python -m dw.validate workflows/templates/text-to-image.json`.
- [ ] **Step 4: Run** the tests and `venv/bin/python -m pytest tests/test_validation.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `engine(cache): the seed warning names the seed variable that has no value`.

### Task 20: The journeys, the rules for the next agent, and the phase gate

**Files:**
- Create: `ui/e2e/journeys/first-image.spec.ts`, `ui/e2e/journeys/chain-a-step.spec.ts`
- Modify: `ui/GLOSSARY.md` (*Before you write markup*), `ui/CLAUDE.md` (rewrite, no new line), `docs/ARCHITECTURE.md` (*UI architecture ratchet* says `raw_media` holds ChainSegments only), `docs/stabilization/ui/baseline.json`

**Interfaces:**
- The fixture needs nothing new: the journeys use the copied `workflows/templates/`, the `e2efind00001` job and the asset library.
- `GLOSSARY.md` gains rows: a file or reference to fill in (`ReferenceField`); whether a run will work (`verdictOf`, `PlanLines`); a workflow's search (`catalogMatches`).
- `ui/CLAUDE.md` bullet 2's "(`raw_media` holds the rest at 4)" becomes "(`raw_media` holds the rest at 1); a file or reference to fill in is a `ReferenceField`", paid for by trimming the same bullet's "(`src/lib/poll.ts` says what they promise)", which the ARCHITECTURE row already says. Still 12 lines.

- [ ] **Step 1: Write the journeys.**
  - `first-image.spec.ts`:
    1. On Workflows, type `text to image`: the count reads `<n> of <m> workflows` and `text-to-image` is listed first.
    2. Open it: the verdict reads `Ready to run.` once checked; `Advanced (<n>)` is folded; the seed field (open Advanced) says `random each job`.
    3. Type `abc` into `num_images_per_prompt` (an Essential: it is a cost driver): Run reads `Fix 1 issue`; pressing it focuses that field. Set `2`: Run is back.
    4. Type a prompt `a lighthouse at dusk`, leave the page for Gallery, come back: the prompt is still there and `Restored what you last typed here.` shows.
    5. With `page.route('**/api/jobs', …)` answering the POST with job `e2efind00001` (assert the body sends `num_images_per_prompt: 2` as a number and the prompt), press Run: the job page opens and its Results show `lighthouse.png`.
    6. Open `inpaint`: its image field offers **Upload…** and **Choose…**; Choose lists images only; pick `lighthouse.png` from earlier jobs: the field holds an `output:` reference and the preview is a tile.
  - `chain-a-step.spec.ts`:
    1. New workflow from the `text-to-image` example; add a pipeline step; in its `image` argument press ↓ in the empty field: the first row is `previous_result:main`; take it. The field stays a reference field.
    2. Rename step `main` to `still`: the toast says `Renamed step main to still, and 1 reference to it`; the second step's field reads `previous_result:still`; both steps keep their detail level.
    3. Type `bad name` as the step name: the step-name error shows and the name stays `still`.
    4. Break a variable path (delete the `prompt` variable): within two seconds the validation panel shows an error with `Go to`; pressing it focuses the field it names.
    5. Add a variable on Enter as `steps` with `30`: the JSON view shows `"steps": 30`.
    6. Replace the second step's `image` value with an asset chosen through **Choose…**: the field holds `asset:<name>` and the preview is a tile.
- [ ] **Step 2: Run** `cd ui && npx playwright test e2e/journeys/first-image.spec.ts e2e/journeys/chain-a-step.spec.ts`. Expected: PASS. A failure names the task that left a gap; fix it there.
- [ ] **Step 3: Write the rules** as in Interfaces.
- [ ] **Step 4: Run** `npm run check`, `npm run lint`, `npm run build`, `npm test`, `npx playwright test`, the metrics gate, `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` (only the two pre-existing lines), `venv/bin/python scripts/arch_metrics.py` (`modules 281`, `claude_md_lines 129`), `wc -l dw/server/api_models.py dw/server/job_history.py dw/server/routes/jobs.py dw/server/routes/library.py ui/src/lib/api.ts` (≤ 1105, 1000, 1041, 1000, 600), and `venv/bin/python -m pytest tests/test_ui_twins.py tests/test_api_contract.py tests/test_catalog_inputs.py -q`. Expected: PASS; `baseline.json` reads `raw_media 1` and `duplicate_blocks` at its measured value, below 18.
- [ ] **Step 5: Commit** `ui(run): the first-image and chain-a-step journeys, and the rules for the next agent`.

---

## Phase exit

- `raw_media` is 1 in `docs/stabilization/ui/baseline.json` (ChainSegments' seeked thumb); `MediaArgumentInput` and `BrowseButton` are gone and `ReferenceField` is the one reference input.
- Every metric that was 0 is 0; `duplicate_blocks` is below 18; nothing is above `$SCRATCH/ui-p5.json`.
- `modules` 281, `claude_md_lines` 129, the size caps hold, no page over 600.
- `GET /api/workflows` answers as before unless asked for `inputs`; Task 19 ran only on Don's yes, or the exit note says it was skipped.
- ARCHITECTURE has the rows *Reference input* and *A workflow's inputs*.
- `first-image.spec.ts` and `chain-a-step.spec.ts` hold the phase.
