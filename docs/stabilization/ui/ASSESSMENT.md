# UI assessment (2026-10-01, develop 2b2b82ce)

The engine stabilization (gates 0-4) never read `ui/` and has no ratchet over
it. This is the survey that scopes a pass: what is there, where it already
disagrees with the engine, and what tooling a ratchet needs.

## The gate reports undercounted the UI

Every gate's *SLOC by layer* row says the UI is about 2,270 code lines.
`scripts/arch_report.py` lists `.svelte` in `SLOC_SUFFIXES`, but pygount has
no lexer for it and returns `SourceState.unknown` with 0 lines, so only the
`.ts` files were counted. `ui/src` without tests is **15,508 raw lines**:
11,623 in `.svelte`, the rest `.ts` and `app.css`, over three times
`dw_mcp`'s 4,597. The measurement needs fixing before any UI baseline is taken.

## What is there

- **Tooling is good and green.** ESLint (typescript-eslint, eslint-plugin-svelte,
  recommended presets only), svelte-check (0 errors, 0 warnings), vitest (36
  files, 340 tests), prettier, Playwright (7 specs). CI's `ui` job runs
  prettier, lint, check, test and build; `e2e` runs only on PRs into master.
  `tsconfig.app.json` is `strict`.
- **One API client.** Every `fetch` and `EventSource` is in `ui/src/lib/api.ts`.
  Response types are hand-written in `types.ts` (379 lines), and nothing checks
  them against the server: no route declares a `response_model`, so the
  OpenAPI document has no response schemas to generate from.
- **Layering holds.** `App` imports pages, pages import components, everything
  imports `lib/*.ts`; no `.ts` imports a `.svelte` but `main.ts`. One import
  cycle: `lib/api.ts` <-> `lib/workspace.svelte.ts`.
- **Size.** 14 non-test files over 400 lines: `EditorPage.svelte` 1,075,
  `PromptEditorPage.svelte` 954, `JobPage.svelte` 855, `editor/StepEditor.svelte`
  802, `api.ts` 671, then `AssetsPage`, `GalleryPage`, `ModelsPage`,
  `WorkflowsPage`, `App`, `app.css`, `FlowView`, `OverviewPage`, `ServerPage`.
  In the big pages about 30% is `<style>`.
- **Complexity.** ESLint's `complexity` at 10 finds 14 functions over, led by
  `pipelineDigest` 19 (`digest.ts`), `legacyRedirect` 17 (`routes.ts`),
  `widgetFor` 16 (`editor.ts`). It does not see `{#if}`/`{#each}` in templates,
  where most of the branching is.

## Second owners of engine rules

Disagreeing today:

| # | UI copy | Owner | Disagreement |
| --- | --- | --- | --- |
| U1 | `flow.ts`: `danglingReferenceDetails` | `dw/previous_results.py` | A second reference validator. Flags a for_each member reference (`previous_result:g@a`) the engine accepts; misses `{from_previous_result: ...}` the engine flags; splits names on the first `.` where the engine uses `reference_resolves_to`. |
| U2 | `editor.ts:98`: `isReference` | `dw/references.py` | Knows 4 of the prefixes; not `item:`, `gather:`, `asset:`, `output:`. `widgetFor` can then offer a number or checkbox widget for such a value, and `coerce` would overwrite it (read from the code, not reproduced in a browser). |
| U3 | `workspaceActions.ts:25` name pattern | `dw/security.py`: `WORKSPACE_NAME_PATTERN` | The UI refuses dots the engine allows, and has no 100-character cap. |
| U4 | `JobPage.svelte:323` extension regexes | `dw/server/outputs.py`: `MEDIA_KINDS` | Misses `.bmp`, `.mov`/`.mkv`/`.avi` and all audio. Gallery and Assets read the server's `kind` instead. |
| U5 | `editor.ts:146`: `CONTENT_TYPES` (a closed `<select>`) | `dw/content_types.py` | Lacks flac, ogg, opus, aiff; such a step shows a blank selection. |

Agreeing today, unpinned: terminal job states (twice in the UI, `api.ts` and
`JobPage.svelte`), `CACHE_TYPES`, the member separator `@`, the unsaved
reason, `DEFAULT_WORKSPACE`, the event names. About 35 reference-prefix
literals across `editor.ts`, `flow.ts`, `prompts.ts` and five components; the
Python `prefix_literals` ratchet does not read them. Run version and the plan
estimate are read from the server, never computed, as `ui/CLAUDE.md` requires.

Duplication inside the UI: `EditorPage` and `PromptEditorPage` are parallel
copies (`setView`, `savePath`, `applyJson`, `onKeydown`, `save`, the folder
regex); `gb()` three times though `format.ts` exists for it; `promptText`
twice; four polling loops; near-copy helpers in `api.ts` (`scoped` /
`workspaceScopedPath`, `withToken` / `withTokenIn`, and the header and error
handling in `fetchJson` / `downloadResponse`).

## Proposed pass

Ordered so the measurement exists before anything is moved, as the engine
pass did.

1. **Measure and ratchet** (no UI code changes).
   - Fix `arch_report.py`'s UI SLOC (count `.svelte`).
   - Extend the ratchet to `ui/src`: modules over a line ceiling, ESLint
     `complexity` and `max-lines-per-function` counts, import cycles
     (`madge --circular --extensions ts,svelte`, a new devDependency, or a
     short script over resolved imports), and prefix literals outside one UI
     owner module. Baseline today's numbers in `baseline.json`; CI already
     runs the `ui` job, so the check goes there.
   - Seam-map rows for the UI's copies, so each has an owner named.
2. **Fix the live disagreements.** U1: drop the client-side validator and show
   the server's `validate` findings, which the editor already fetches, or, if
   the offline check is wanted, make it a thin wrapper over one prefix module.
   U2-U5: one `references.ts` owner for prefixes; read media kind and content
   types from the server (a field on an existing response) instead of a list.
   Twin-constant pins where a list must stay in the UI.
3. **Consolidate.** One editor shell under `EditorPage` and
   `PromptEditorPage`; one polling helper; `format.ts` and the `api.ts`
   helpers de-duplicated; the `api.ts` <-> `workspace` cycle broken. Split the
   pages over the ceiling along the jobs they do (`JobPage`: events, ETA,
   export, media).
4. **Contract (optional, the expensive part).** `response_model`s on the
   routes, then generated TS types (openapi-typescript) in place of
   `types.ts`, checked in CI. Running `e2e` on PRs into develop is a cheaper
   step toward the same goal.

Stage 2 of the [dw_mcp pass](../mcp-assessment.md) moves image, frame and
metadata logic to new server routes; the UI should consume the same routes, so
that stage and this one's stage 2 share their server work.
