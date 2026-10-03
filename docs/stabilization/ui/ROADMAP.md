# UI stabilization roadmap

The UI's turn at what the engine got from 2026-09-28 to 2026-10-01
([../ROADMAP.md](../ROADMAP.md)). Why it exists: [ASSESSMENT.md](ASSESSMENT.md).
The approach is the engine's: measure first, fix what is wrong before moving
anything, give every rule one owner, then restructure, then install the
guardrails that keep it.

Each phase ends at a gate that passes or fails. A phase's detailed plan is
written when the previous gate passes, not before: what each later phase
works on is what the earlier phases leave behind. Phase 0's plan is
[phase-0.md](phase-0.md).

| Phase | Scope | Gate | Plan | Status |
| --- | --- | --- | --- | --- |
| 0 | UI ratchet and baseline; the gate reports count `.svelte`; fix the five rules the UI and the engine disagree on today (U1-U5) | `ui/scripts/arch-metrics.mjs --check` runs in CI and preflight against a committed baseline; U1-U5 fixed, each with a test that failed first; tagged `ui-stabilization-gate-0` | [phase-0.md](phase-0.md) | done 2026-10-01 (`ui-stabilization-gate-0`) |
| 1 | One owner per rule: prefixes in `src/lib/references.ts` only; engine vocabularies read from the server or pinned; the server takes over what `dw_mcp` computes client-side (its stage 2), then `dw_mcp` consolidates (stage 3) | `prefix_literals` 0; every UI and `dw_mcp` copy of an engine rule is either gone or pinned by a test, and has a seam-map row | [phase-1.md](phase-1.md) (staged: 1a-1c) | done 2026-10-01 (`ui-stabilization-gate-1`) |
| 2 | Primitives: overlays and comboboxes on Bits UI, behind `src/lib/ui/` wrappers styled from `app.css` tokens | No hand-rolled focus trap, Escape chain or DOM sniffing for open dialogs; no false `aria-modal`; `<datalist>` gone; e2e green | [phase-2.md](phase-2.md) | done 2026-10-02 (`ui-stabilization-gate-2`); freeze lifted |
| 3 | Structural moves: one editor shell under `EditorPage` and `PromptEditorPage`; `JobPage` split by job; one polling helper; `api.ts` de-duplicated and its cycle broken; shared layout styles | `files_over_size_ceiling` 0, `import_cycles` 0, `long_functions` 0; suite, e2e and a lem UI smoke green | [phase-3.md](phase-3.md) | done 2026-10-02 (`ui-stabilization-gate-3`) |
| 4 | Contract and guardrails: response models on the routes the UI reads, generated TS types, e2e on PRs into develop, the harness ratchets the UI, `ui/CLAUDE.md` triaged | A server response change that breaks the UI fails CI; the harness refuses a UI ratchet rise without `arch-approved` | [phase-4.md](phase-4.md) (staged: 4a-4d) | done 2026-10-02 (`ui-stabilization-gate-4`) |

## Decisions

Recorded 2026-10-01; Don ruled on both open questions the same day.

- **Freeze (yes, Don 2026-10-01; lifted at gate 2, 2026-10-02).** The engine pass froze features until its guardrails
  existed. The UI equivalent: no new pages, components or UI features until
  gate 2; the implementer fixes tester bugs in `ui/` through existing code,
  and a fix that needs a new component waits or is labelled `stabilization`.
- **Component library: Bits UI (approved, Don 2026-10-01).** Headless, so the
  `app.css` token system, the achromatic chrome and the WCAG AA pairs in
  `ui/CLAUDE.md` stay as they are. It is native to Svelte 5 runes, covers every
  hand-built widget (dialog, alert dialog, popover, combobox, tooltip, toggle
  group), and is the most used and actively maintained of the headless options
  (2.19.4 released 2026-10-01; about 1.26M downloads a week). Pages never
  import it directly: each primitive is wrapped once in `src/lib/ui/`.
- **Not Bootstrap.** Bootstrap 5.3's palette is chromatic and its colour
  modes are a second theme system beside `data-theme`; making it fit means a
  Sass rebuild and re-proving every AA pair. Its Svelte bindings
  (`@sveltestrap/sveltestrap` 7.1) are still Svelte 4 syntax, and Bootstrap 6
  is unreleased, so adopting 5.3 now means a second migration. Bootstrap plus
  a headless library is incoherent: both own a dialog's markup and state.
  shadcn-svelte (Bits UI plus Tailwind) is coherent but replaces the token
  system; its useful idea, wrapping each primitive locally, is adopted
  without the Tailwind layer.
- **Scale, honestly.** About 330 lines of hand-built overlay code exist.
  Phase 2 is about correct semantics and fewer special cases, not size; the
  size is in page layout and scoped styles, which is Phase 3.

## Metrics

**Ratchets.** `ui/scripts/arch-metrics.mjs` (Phase 0) measures `ui/src`,
tests excluded. Every metric is lower-is-better, and `--check` against
`baseline.json` in this directory fails the build when one gets worse. The
counting rules are in the script's header comment. Measured 2026-10-01 at
develop `b290dab4`:

| Ratchet | Rule | Today |
| --- | --- | --- |
| `files_over_size_ceiling` | files over 600 raw lines, `<style>` included | 7 |
| `complex_functions` | ESLint `complexity` over 10, script blocks only | 14 |
| `long_functions` | ESLint `max-lines-per-function` over 60 | 2 |
| `import_cycles` | strongly connected components over relative imports | 1 |
| `modules_in_import_cycles` | their total size | 2 |
| `prefix_literals` | strings starting with a reference prefix outside `src/lib/references.ts` | 29 |
| `a11y_suppressions` | `svelte-ignore a11y_*` comments | 7 |

ESLint does not see `{#if}`/`{#each}` in templates, where much of a page's
branching is; the gate report counts template blocks per file as a proxy
rather than inventing a ratchet for it.

**Gate reports.** Written into this file at each gate, never gates
themselves: the ratchet table with one column per gate, SLOC by `.svelte`
block (script, markup, style), the ten largest files, the ten most complex
functions, template block counts for the largest pages, and change coupling
within `ui/src` and between `ui/src` and `dw/server`, from `git log`.

Each gate is tagged `ui-stabilization-gate-N`.

## Gate reports

### Gate 0 (2026-10-01, develop 367bcb57)

Gate criteria:
- **Ratchet in CI and preflight:** met. `ui/scripts/arch-metrics.mjs --check` runs in the `ui` CI job and in `npm run preflight` against `baseline.json`.
- **U1-U5 fixed, each with a test that failed first:** met.
  - U1 is `flow.ts`'s dangling check. Its resolver is now shared with the flow graph's chips and edges.
  - U2 is `references.ts` and `isReference`.
  - U3 is workspace names, checked against the shared `tests/fixtures/workspace_names.json`.
  - U4 is `output_kinds` on the job detail and on file-carrying run events.
  - U5 is the content types.
- **Tagged:** met.

Suite results on develop:
- pytest: 7,734 passed (`DW_DEVICE=cpu`).
- Integration: 5 passed, on the GPU.
- vitest: 374 passed.
- Playwright e2e: 108 passed.
- Both ratchets: clean.

The final whole-branch review (one fresh reviewer) found three problems, fixed in `a6928dbe`:
- A running job's outputs showed as links until its manifest existed.
- A `from_previous_result` spelled `variable:...` was flagged as dangling.
- The graph's chips still split names on the first dot.

Five minor findings are deferred to Phase 1:
- The check still accepts some references the engine refuses (`gather:` on a step without `for_each`, a group named from outside it).
- The name length counts UTF-16 units.
- MCP `get_job` now carries `output_kinds` undocumented.
- Two weak pins.
- Two Review Focus cases are tested only at unit level.

The folder and prompt name patterns in the two editor pages are ASCII-only copies of a Unicode rule. They are the U3 problem again, and Phase 1's one-owner work takes them.

lem smoke, `367bcb57` deployed:
- A `normalize_audio` job renders three `<audio>` players.
- A `text-to-image` job renders its image.
- `assemble-and-score` renders its video.
- No page errors.
- `GET /api/jobs/{id}` carries `output_kinds` for historical jobs.

#### Ratchets

| Ratchet | Before Phase 0 (`12aa9204`) | Gate 0 |
| --- | --- | --- |
| `files_over_size_ceiling` | 7 | 7 |
| `complex_functions` | 14 | 14 |
| `long_functions` | 2 | 2 |
| `import_cycles` | 1 | 1 |
| `modules_in_import_cycles` | 2 | 2 |
| `prefix_literals` | 29 | 15 |
| `a11y_suppressions` | 7 | 7 |

#### SLOC

`.svelte` non-blank lines by block: script 3,744, markup 4,104, style 3,293.
The UI as the engine report counts it (`svelte_code_lines` plus pygount):
12,969 code lines at `stabilization-gate-4`.

#### Largest files (raw lines)

| Lines | File |
| --- | --- |
| 1075 | `src/lib/pages/EditorPage.svelte` |
| 954 | `src/lib/pages/PromptEditorPage.svelte` |
| 875 | `src/lib/pages/JobPage.svelte` |
| 802 | `src/lib/editor/StepEditor.svelte` |
| 671 | `src/lib/api.ts` |
| 669 | `src/lib/pages/AssetsPage.svelte` |
| 612 | `src/lib/pages/GalleryPage.svelte` |
| 596 | `src/lib/pages/ModelsPage.svelte` |
| 525 | `src/lib/pages/WorkflowsPage.svelte` |
| 496 | `src/App.svelte` |

#### Most complex functions (ESLint `complexity`)

| Complexity | Function |
| --- | --- |
| 19 | `src/lib/digest.ts:73` `pipelineDigest` |
| 17 | `src/lib/routes.ts:67` `legacyRedirect` |
| 16 | `src/lib/editor.ts:100` `widgetFor` |
| 15 | `src/lib/progress.ts:30` `stepProgress` |
| 15 | `src/lib/results.ts:142` `unsavedSteps` |
| 13 | `src/lib/editor.ts:59` `mediaKindFor` |
| 12 | `src/lib/api.ts:112` `describeContents` |
| 12 | `src/lib/pages/JobPage.svelte:151` `exportJob` |
| 12 | `src/lib/pages/WorkflowsPage.svelte:167` (arrow function) |
| 12 | `src/lib/routes.ts:39` `parseView` |

Template blocks (`{#if}` and `{#each}`), the branching ESLint does not see:
- `JobPage` 41
- `EditorPage` 24
- `WorkflowsPage` 22
- `StepEditor` 21
- `ModelsPage` 19
- `PromptEditorPage` 19


### Gate 1 (2026-10-01, develop 59f64f8e)

Gate criteria:
- **`prefix_literals` 0:** met. `references.ts` is the only speller, through `reference()` and `referenceName()`.
- **Every UI and `dw_mcp` copy of an engine rule gone or pinned, with a seam-map row:** met.
  - Pinned for the UI: the shared case files (`reference_cases.json`, where the engine decides each case, and `workspace_names.json`) and `tests/test_ui_twins.py`, which pins shapes, traits, component slots, offload modes, content types, the writer settings and the terminal states.
  - Pinned for `dw_mcp`: `tests/test_mcp_twins.py`.
  - Gone from `dw_mcp`: image fitting, crop, decode limits, the probe list, the merge patch, run-directory derivation and the level thresholds. The server owns all of them now.
- **`mcp-assessment.md` stages 2 and 3:** done, M-item by M-item.

Suite results:
- pytest: 7,781 passed (`DW_DEVICE=cpu`).
- Integration: 5 passed, on the GPU.
- vitest: 407 passed.
- Playwright e2e: 108 passed.
- Both ratchets: clean.

The final whole-branch review (one fresh reviewer) found three problems, fixed in `d46442eb`:
- The image route gave a bare 500 on an undecodable file.
- The editor's same-list check compared raw values where the engine compares resolved, key-order-free values.
- The full-scale finding gave weaker advice than `audio_qc`.

Ten minor findings are deferred. They are listed as `Final: minor (deferred)` in the branch's ledger, and the main ones are:
- No 64 px floor on `/image` and `/frames`.
- Double encodes.
- A workflow DELETE that does not take the save lock.
- No version check for a newer `dw-mcp` against an older server.
- A combined prompt-name length the editor does not check.

The modules ratchet rose 165 → 167 for `dw/server/inline_media.py` and `dw_mcp/confine.py`, each rise in its own named commit, approved by Don with the plan.

lem smoke, `59f64f8e` deployed, over MCP:
- `get_output_image` with a crop returned 200x120 from an 896x1152 source, cut by the server.
- `get_output_frames` with `hear` returned two tiles plus two excerpts.
- `get_gallery_metadata` on a scored cut carried `findings: []` (peak -2.9 dBFS) and the next hint with no thresholds.
- `save_workflow` patch mode merged a description onto the stored version.
- `delete_output(job_id=)` removed a scratch run whole.
- A second delete of the same run surfaced a bug: it answered "Invalid run directory", because the containment check required the path to exist. Fixed test-first (`4d25dd57`), redeployed, and re-smoked: the second delete now says "already gone".

In the browser:
- The job pages render audio players, images and video with no page errors.
- The editor's warnings are covered by unit and e2e tests.

`dw_mcp` is 4,405 lines, against 4,597 at the assessment.

#### Ratchets

| Ratchet | Before Phase 0 | Gate 0 | Gate 1 |
| --- | --- | --- | --- |
| `files_over_size_ceiling` | 7 | 7 | 7 |
| `complex_functions` | 14 | 14 | 14 |
| `long_functions` | 2 | 2 | 2 |
| `import_cycles` | 1 | 1 | 1 |
| `modules_in_import_cycles` | 2 | 2 | 2 |
| `prefix_literals` | 29 | 15 | 0 |
| `a11y_suppressions` | 7 | 7 | 7 |

`.svelte` non-blank lines by block: script 3,753, markup 4,107, style 3,293.
The largest files and most complex functions are unchanged from gate 0 apart
from a few lines: `EditorPage` is 1,077 lines and `widgetFor` sits at
`editor.ts:106`. Phase 3 is where they move.


### Gate 2 (2026-10-02, develop c3c0cd57)

Gate criteria:
- **No hand-rolled focus trap, Escape chain or DOM sniffing for open dialogs:** met.
  - `focusTrap.ts` is deleted.
  - `App.svelte` keeps one Escape branch, for the mobile drawer. The drawer is a layout region, not an overlay.
  - `dialogOpen()` is replaced by the layer count (`ui/src/lib/ui/layers.svelte.ts`).
- **No false `aria-modal`:** met. The status and token popovers are non-modal dialogs anchored to their trigger.
- **`<datalist>` gone:** met. All 13 are replaced by `Suggest`.
- **e2e green:** met.

Bits UI 2.19 is imported only under `ui/src/lib/ui/` (ESLint `no-restricted-imports`), behind `ConfirmDialog`, `Modal`, `Popover` and `Suggest`. Overlay styles and one stacking scale (`--layer-*`) are in `app.css`.

Suite results:
- pytest: 7,783 passed (`DW_DEVICE=cpu`).
- Integration: 5 passed, on the GPU.
- vitest: 424 passed.
- Playwright e2e: 109 passed. One new e2e opens a suggestion list in a real browser.
- Both ratchets: clean.

Three real-browser bugs jsdom could not show were found by that e2e, fixed in `app.css` before review:
- The list sat behind the page (a z-index on a non-positioned element).
- The list was 2px wide.
- Its rows had shrunk to 8px.

The final whole-branch review found two `Suggest` defects that silently changed workflow values, fixed in `9b233c7b`:
- Enter took the first matching suggestion instead of the typed text, and so did Ctrl/Cmd+Enter (validate & run).
- Picking the same suggestion twice did nothing, because Bits treated it as a deselect.

Minor findings are deferred:
- One Escape closes both an overlay and the open mobile drawer.
- Home and End move the list highlight while a suggestion list is open.
- `aria-expanded` can read true with no list showing.
- A pick may fire `onchange` twice; every caller is idempotent.
- Focus return after the confirm is not asserted.
- A misplaced test helper comment.

lem smoke, `c3c0cd57` deployed, headless Chromium:
- `?` opens the help, and Escape closes it.
- The status popover opens; its trigger closes it again; an outside click closes it.
- The token popover opens.
- A delete confirm on a scratch asset opens, and Cancel keeps the asset. The asset was removed through the API afterwards.
- The editor's pipeline field offers 22 suggestions for `Flux`, and Enter keeps `Flux`.
- No page errors.

#### Ratchets

| Ratchet | Before Phase 0 | Gate 0 | Gate 1 | Gate 2 |
| --- | --- | --- | --- | --- |
| `files_over_size_ceiling` | 7 | 7 | 7 | 7 |
| `complex_functions` | 14 | 14 | 14 | 13 |
| `long_functions` | 2 | 2 | 2 | 2 |
| `import_cycles` | 1 | 1 | 1 | 1 |
| `modules_in_import_cycles` | 2 | 2 | 2 | 2 |
| `prefix_literals` | 29 | 15 | 0 | 0 |
| `a11y_suppressions` | 7 | 7 | 7 | 3 |

### Gate 3 (2026-10-02, develop 329b7d0a)

Gate criteria:
- **`files_over_size_ceiling` 0:** met (was 7). The largest file is now `ModelsPage.svelte` at 597 lines; `EditorPage` is 456, down from 1,077.
- **`import_cycles` 0:** met. `api.ts`'s cycle is broken by `apiUrls.ts` and `workspaceState.svelte.ts`.
- **`long_functions` 0:** met (was 2).
- **Suite, e2e and a lem UI smoke green:** met.

What moved:
- **One editor shell.** `DocumentEditor` (`editorShell.svelte.ts`) holds the document, its baseline and dirty flag, the JSON draft, the remembered view, the save path and the one tab-close guard. Both `EditorPage` and `PromptEditorPage` build on it, with `ViewSwitch`, `FolderPicker` and `EditorBody`.
- **`JobPage` split by job.** `JobHeader`, `JobProgress` and `JobResults` sit under `lib/job/`. `JobResults` is keyed on `job.id`, so a late metadata reply from the previous job lands on a destroyed instance.
- **The editors' parts are components.** `StepList`, `StepModes`, `ValidationPanel` and `WorkflowFileBar` serve the workflow editor; `Enhancer` and `EnhancePanel` serve the prompt editor; `PipelineOptions` and `StepDigestList` serve `StepEditor`. The gallery detail and three asset components moved too.
- **One polling helper** (`poll.ts`), and **shared layout styles** (`.pagehead`, `.withicon` and others in `app.css`).
- **Seam-map rows** in `docs/ARCHITECTURE.md` for the editors' document, UI polling and workspace state in the UI.

Suite results:
- pytest: 7,784 passed (`DW_DEVICE=cpu`).
- Integration: 5 passed, on MPS.
- vitest: 468 passed. 44 of them are new, pinning the shell, the enhancer, step modes and the split pages.
- Playwright e2e: 109 passed.
- Both ratchets: clean.

The final whole-branch review found no behaviour regression in job switching, the enhancer download, the step list, the leave-page guard or workspace scoping. One Important finding, a GalleryPage test that asserted before its second listing landed and failed about two full runs in five, was fixed in `5ea428bf`. Preflight also caught the default-workspace twin pin reading the file the constant had left (`0a0666e5`).

Minor findings are deferred:
- A late `refresh()` on the job page can bring back the previous job after a navigation. This predates the phase.
- No test that the enhancer download stops when the page unmounts, and no page-level leave-page test on `EditorPage`.
- `PipelineOptions` re-runs its scheduler lookup when a step goes compact and back to full.

lem smoke, `329b7d0a` deployed, headless Chromium:
- A job page shows its header, Run again and its media. Switching to another job replaces the header and results, with none of the first job's left behind.
- The workflow editor's four views switch. Adding a step marks the document unsaved, and the JSON view shows it in Monaco.
- The prompt editor opens a stored prompt under its name. The enhancer panel shows its model, device and Generate controls, and comes back after a switch to the JSON view and back.
- The gallery's detail plays a video. The assets detail shows its delete control.
- Idle on the overview, the UI makes 4 API calls in 12 seconds, to `/api/health` and `/api/memory` only.
- No page errors.

#### Ratchets

| Ratchet | Before Phase 0 | Gate 0 | Gate 1 | Gate 2 | Gate 3 |
| --- | --- | --- | --- | --- | --- |
| `files_over_size_ceiling` | 7 | 7 | 7 | 7 | 0 |
| `complex_functions` | 14 | 14 | 14 | 13 | 12 |
| `long_functions` | 2 | 2 | 2 | 2 | 0 |
| `import_cycles` | 1 | 1 | 1 | 1 | 0 |
| `modules_in_import_cycles` | 2 | 2 | 2 | 2 | 0 |
| `prefix_literals` | 29 | 15 | 0 | 0 | 0 |
| `a11y_suppressions` | 7 | 7 | 7 | 3 | 3 |

### Gate 4 (2026-10-02, develop e578d7e4)

Gate criteria:
- **A server response change that breaks the UI fails CI:** met.
  - 46 routes - every JSON route `api.ts` calls - declare Pydantic response models in `dw/server/api_models.py` (70 models, 824 lines).
  - `types.ts` re-exports the types `openapi-typescript` generates from the OpenAPI document, which `scripts/dump_openapi.py` commits at `ui/src/lib/generated/`. No response type is hand-written any more.
  - The chain: `tests/test_api_contract.py` fails when the committed document is stale; CI's "Response contract" step fails when the generated types are; `npm run check` fails wherever the UI reads a field the server stopped sending. `tests/test_ui_twins.py` fails when `api.ts` calls a JSON route with no declared model.
  - Demonstrated on a scratch worktree: renaming `HealthInfo.worker_alive` on the server failed the freshness test, then - regenerated - failed svelte-check in `StatusPopover.svelte` and `ServerPage.svelte`, the two places that read it.
- **The harness refuses a UI ratchet rise without `arch-approved`:** met (harnest `fd527d2`, `2be80d8`, pushed).
  - `check_ui` measures `ui/` at the merge base and the work's HEAD with develop's `ui/scripts/arch-metrics.mjs` and asks its `--compare` which metrics rose; the waiver is the engine's, against `docs/stabilization/ui/baseline.json`. It fails closed on every failure it can meet (no node, no `ui/node_modules`, a crashing compare, an unreadable cached measurement, any exception).
  - Demonstrated on dw: a 12-branch function in `ui/src/lib/format.ts` reports `complex_functions: 12 -> 13` - with an inline `eslint-disable complexity` above it too, since the metrics script no longer honours inline directives or a branch's own ignore patterns.
- **e2e before develop moves:** e2e runs on PRs into develop or master and on every push to develop (the agent loop pushes develop directly). The merge's own push ran it: GitHub CI run 37036348274 on `e578d7e4` - backend, ui (with the new "Response contract" step) and e2e all passed.

The contract's rules, beside the seam map's new "The UI's response contract" row:
- The payload does not change: an absent key stays absent (`response_model_exclude_unset`), a key sent only in some states is `sometimes()` (generated as `key?: T`), an int stays an int.
- Runtime is lenient: an undeclared key passes through, and a response its model rejects is logged and sent as built rather than turned into a 500 (on `POST /api/jobs` the job is queued by then). Tests, the e2e fixture server and the dump run strict (`DW_STRICT_RESPONSES=1`). lem's journal held no "does not match its model" line across the four deploys.
- Outside the contract, by design: the agent-only `view=compact` and `only_orphans` listings, the verbatim workflow/prompt GETs, the event stream, JSON Schema documents, files and zips.

What declaring the models found:
- `/api/validate` would have answered 500 for any uncached gated model (`gated` is `"auto"`/`"manual"`, not a bool) - caught by the 4b review before it shipped.
- A history job with a null manifest, and the Jobs page's name filter on a job with no workflow name (it threw) - fixed test-first.
- Keys `types.ts` never declared (`/api/server`'s `trust_workflows` and `runtime`; memory's `stale`/`reason`/`age_seconds`; a gallery or asset file's `text` kind; a plan estimate's `observed` basis).
- A scheduler whose `-inf` default answered 500 now names it.

`ui/CLAUDE.md` is 11 lines (12 before): the engine-derived-fields rule is the contract's seam row, `--live` placement is `ui/scripts/design-rules.test.ts`, and WCAG AA stays guidance (Don: a nice-to-have, not a guardrail). Engine ratchet: `modules` 167 -> 168 (`api_models.py`), `claude_md_lines` 129 -> 128.

Suite results (develop `e578d7e4`):
- pytest: 7,854 passed (`DW_DEVICE=cpu`); integration 5 passed on MPS.
- vitest: 474 passed. Playwright e2e: 109 passed against the strict fixture server.
- harnest: every test file passes (`test-ui-ratchet.sh` 31).
- Both ratchets: clean.

Each stage had a fresh whole-branch review. Findings fixed test-first: the dump script touching `~/.diffusers_helper` and reading another checkout's `dw` (4a); the gated-model 500 and runtime leniency for a rejected response (4b); the `-inf` default (4c); the UI ratchet's fail-open paths, the inline-directive bypass and stale UI tools (4d).

Deferred minors (detail in each stage's review):
- Response key order changed in places: declared fields first, then extras (`/api/memory`'s `info`, a history job's `historical`/`spec`, a manifest entry's `parent_step`, a workflow card's `configures_missing`, a prompt card's `text_chars`). Keys and values are unchanged - a release-note line.
- The freshness test follows whatever FastAPI/Pydantic CI resolves; a release that changes OpenAPI output would fail unrelated PRs.
- The coverage pin is path-level, not method-level; some 4c state pins are looser than planned.
- A `.txt` gallery or asset file renders in the audio branch (predates Phase 4; the type is now honest about `text`) - a UI follow-up.
- The UI ratchet's refusal hints are Python-flavoured; its message before develop had `--compare` named no way out; no test runs both ratchets at once.
- e2e runs twice per develop push while a release PR is open.

After the gate (branch `ui-stabilization/minors`, harnest 9893924): FastAPI and Pydantic are pinned in `constraints-openapi.txt` for CI and the dump; the coverage pin compares method and path; the 4c entries pin exact keys; e2e skips the release PR's own run; the UI ratchet gives UI hints, parks on a broken `--compare`, and refuses both rises at once. The `.txt` rendering is #573. The key-order note waits for the release.

lem smoke, `e578d7e4` deployed, headless Chromium and MCP:
- Every converted route the UI reads answers 200, including the agent views; the DPM scheduler's default reads `-inf`.
- The status popover, Server, Models, Jobs, a job page, the editor (validate, pipeline suggestions, a step's parameters), Prompts, Gallery and Assets render with no page errors.
- `get_health`, `get_memory`, `get_job`, `validate_workflow`, `list_workflows`, `list_assets`, `get_prompt` answer as before.

#### Ratchets

| Ratchet | Before Phase 0 | Gate 0 | Gate 1 | Gate 2 | Gate 3 | Gate 4 |
| --- | --- | --- | --- | --- | --- | --- |
| `files_over_size_ceiling` | 7 | 7 | 7 | 7 | 0 | 0 |
| `complex_functions` | 14 | 14 | 14 | 13 | 12 | 12 |
| `long_functions` | 2 | 2 | 2 | 2 | 0 | 0 |
| `import_cycles` | 1 | 1 | 1 | 1 | 0 | 0 |
| `modules_in_import_cycles` | 2 | 2 | 2 | 2 | 0 | 0 |
| `prefix_literals` | 29 | 15 | 0 | 0 | 0 | 0 |
| `a11y_suppressions` | 7 | 7 | 7 | 3 | 3 | 3 |

## Working rules for the duration

- A rule the engine owns is read from the server or pinned to its owner by a
  test; it is never re-derived in the UI.
- Every fix starts with a test that fails on today's code.
- No count-pinning assertions; baseline numbers live in `baseline.json`.
- Comments explain why the code is the way it is, not the ticket history.
- A ratchet only rises in a commit that names the rise and why, and the
  harness requires `arch-approved` for it, as for the engine.
