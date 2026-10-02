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
| 3 | Structural moves: one editor shell under `EditorPage` and `PromptEditorPage`; `JobPage` split by job; one polling helper; `api.ts` de-duplicated and its cycle broken; shared layout styles | `files_over_size_ceiling` 0, `import_cycles` 0, `long_functions` 0; suite, e2e and a lem UI smoke green | [phase-3.md](phase-3.md) | not started |
| 4 | Contract and guardrails: response models on the routes the UI reads, generated TS types, e2e on PRs into develop, the harness ratchets the UI, `ui/CLAUDE.md` triaged | A server response change that breaks the UI fails CI; the harness refuses a UI ratchet rise without `arch-approved` | written at gate 3 | |

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

## Working rules for the duration

- A rule the engine owns is read from the server or pinned to its owner by a
  test; it is never re-derived in the UI.
- Every fix starts with a test that fails on today's code.
- No count-pinning assertions; baseline numbers live in `baseline.json`.
- Comments explain why the code is the way it is, not the ticket history.
- A ratchet only rises in a commit that names the rise and why, and the
  harness requires `arch-approved` for it, as for the engine.
