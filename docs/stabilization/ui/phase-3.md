# UI Phase 3: Structural Moves - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every file in `ui/src` is under the 600-line ceiling, the import graph has no cycle and no function is over 60 lines, by extracting cohesive pieces from the seven oversized files and consolidating the duplicates the assessment found. Nothing a user sees changes.

**Architecture:** One structural move per task, each green on its own. New homes:
- `ui/src/lib/job/` for the job page's parts;
- `ui/src/lib/editor/` for the editor shell's components;
- `ui/src/lib/editorShell.svelte.ts` (a `DocumentEditor` class shared by the workflow and prompt editors);
- `ui/src/lib/enhancer.svelte.ts`, `ui/src/lib/stepModes.svelte.ts`;
- `ui/src/lib/poll.ts`, `ui/src/lib/outputMeta.ts`;
- `ui/src/lib/workspaceState.svelte.ts` (breaks the cycle);
- `ui/src/lib/apiErrors.ts`, `ui/src/lib/apiUrls.ts`.

`api.ts` keeps re-exporting what the pages and the test mocks use (`vi.mock('../api')` mocks by module id).

**Tech Stack:** Svelte 5 (runes), TypeScript, Vitest + @testing-library/svelte, Playwright.

**Spec:** [ROADMAP.md](ROADMAP.md) Phase 3 row and gate; [ASSESSMENT.md](ASSESSMENT.md) *Duplication inside the UI*.

## Global Constraints

- A refactor: rendered markup, classes, accessible names, titles and behaviour stay as they are. Every e2e selector keeps working; the full e2e suite runs at the end of each task that moves markup.
- Each task ends with the UI ratchet checked, `cd ui && npm run metrics -- --check ../docs/stabilization/ui/baseline.json`, and a metric that falls is rewritten into the baseline in that task's commit. None may rise. `complex_functions` (13) is held: a moved complex function moves intact, and only Task 4 splits one.
- A component that the pages' tests mock through `'../api'` imports what it needs from `'../api'` or `'../../api'`, never from `apiErrors.ts`/`apiUrls.ts` directly.
- Line numbers below are from develop `fee6ff01`; each move also names the code by content, since earlier tasks shift lines.
- A moved piece's styles move with its markup. A rule used on both sides of a split is copied, not globalised, unless the task says otherwise (global `.error`/`.hint` would restyle pages that use those class names without a rule).
- Commands from `ui/`: `npx vitest run`, `npm run check`, `npm run lint`, `npx prettier --write src e2e`, `npm run metrics -- --check ../docs/stabilization/ui/baseline.json`, `npm run build && DW_E2E_PYTHON=../venv/bin/python npx playwright test`.
- Commits: conventional prefix (`refactor(ui)`); end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Branch `ui-stabilization/phase-3` from develop.

## Review Focus

1. A job's results reset when the job page moves to another job: no metadata, file or seed from the previous job shows, and the metadata fetch order JobPage.test pins holds (Task 8).
2. In the prompt editor, the enhancer's model download survives switching views (form/split/JSON) and stops when the page is left (Task 10).
3. A step added, removed or moved in the workflow editor after the step list is extracted still edits the same workflow object the JSON view and flow view show (Task 11).
4. An unsaved edit still triggers the browser's leave-page guard in both editors after the shell takes over the guard (Tasks 10-11).
5. A request scoped to a job's own workspace (download URL, metadata) still goes to that workspace after `api.ts`'s scoping helpers are merged (Task 6).

---

### Task 1: Delete duplicated rules; one page-head class

- [ ] Delete the scoped `.withicon` and `.flex` rules that duplicate `app.css`'s globals (262-271): `.withicon` in `LorasEditor`, `VariablesForm`, `EditorPage` (~792), `JobPage` (~685), `ModelsPage` (~475), `PromptEditorPage` (~766), `WorkflowPage` (~290); `.flex` in `ComponentEditor`, `StepEditor` (~593), `EditorPage` (~789), `JobPage` (~682), `PromptEditorPage` (~763), `SchemaPage` (~69). Compare each body with the global before deleting; keep any that differ, and record which.
- [ ] Add to `app.css`: `.pagehead { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.8rem; margin-bottom: var(--space-4); } .pagehead.baseline { align-items: baseline; }`. In `AssetsPage` (~503), `PromptsPage` (~143), `WorkflowsPage` (~342), `JobsPage` (~198) replace `class="head"` with `class="pagehead"` and delete their identical `.head` rule; `GalleryPage` (~465) uses `class="pagehead baseline"`. Leave every other `.head` (the editor's sticky head, Job, Server, Models, Overview differ).
- [ ] Run: vitest, check, lint, metrics `--check`, full e2e. Expected: all pass. Commit `refactor(ui): drop scoped copies of global rules; one page-head class`.

### Task 2: Shared formatting and output-metadata helpers

**Files:** `ui/src/lib/format.ts`, create `ui/src/lib/outputMeta.ts`, `App.svelte`, `StatusPopover.svelte`, `ModelsPage.svelte`, `JobPage.svelte`, `GalleryPage.svelte`; tests `format.test.ts`, `outputMeta.test.ts`.

**Interfaces:** `gbFromMb(mb: number): string` and `gbFromBytes(bytes: number): string` (both `toFixed(1)`, no unit - not `formatBytes`, which formats differently); `promptText(value: unknown): string`; `describeOutputMeta(meta: Record<string, unknown> | null): { model: string; seed: unknown; prompt: string; negativePrompt: string }` (JobPage's `describe`, ~280-289, with its `promptText`, ~244).

- [ ] Failing tests first: `gbFromMb(1536) === '1.5'`, `gbFromBytes(1610612736) === '1.5'`; `promptText` and `describeOutputMeta` cases copied from what JobPage and GalleryPage show today (a string prompt, a `{text}` object, a missing seed). Run: FAIL (no exports).
- [ ] Implement; replace the three `gb` copies (App ~83 and StatusPopover ~17 take MB; ModelsPage ~144 takes bytes) and the two `promptText`/describe copies (JobPage ~244/280, GalleryPage ~206-222). Gallery's seed/prompt derivations become `describeOutputMeta(metadata)`.
- [ ] vitest, check, lint, metrics, e2e. Commit `refactor(ui): one home for the GB and output-metadata helpers`.

### Task 3: One polling helper

**Files:** create `ui/src/lib/poll.ts` + `poll.test.ts`; `App.svelte` (~67-81), `ServerPage.svelte` (~30-42), `JobsPage.svelte` (~39-52), `ModelsPage.svelte` (~64-68, ~84-92).

**Interfaces:** `poll(fn: () => unknown, ms: number, opts?: { immediate?: boolean }): () => void` (immediate defaults to true: call `fn` once now, then every `ms`; returns the stop function) and `sleep(ms: number): Promise<void>`.

- [ ] Failing test with fake timers: immediate call, then one call per interval, none after stop; `immediate: false` skips the first. Run: FAIL.
- [ ] Implement; each site becomes `$effect(() => poll(async () => { ... }, ms))`, keeping its interval and immediacy (App/Server 5000 immediate; Jobs 3000 immediate, reading its filter before `return poll(...)`; Models 2000 not immediate, guarded by `if (!updating) return` / `if (!anyActive) return`). The prompt editor's download wait stays a sequential loop (it must not overlap requests) and moves in Task 10, using `sleep`.
- [ ] vitest, check, lint, metrics, e2e. Commit `refactor(ui): one polling helper`.

### Task 4: Split the two long functions

- [ ] `digest.ts` `pipelineDigest` (~73-143): extract `pipelineSummary(cfg, pre, argCount)`, `mainLine(step, cfg, pre)`, `componentsLine(pipeline)`, `lorasLine(pipeline)`, `schedulerLine(pipeline)`, `accelerationLine(cfg)` beside the existing `argsLine`; `pipelineDigest` returns `{ summary, lines: [...mainLine(...), ...argsLine(args), ...componentsLine(...), ...lorasLine(...), ...schedulerLine(...), ...accelerationLine(...)] }` in today's order. `digest.test.ts` pins the output; run it before and after.
- [ ] `monaco.ts` `setupMonaco` (~21-107): `PALETTES` to module scope; `configureSchemas(): Promise<void>` (~25-55); `defineThemes(): void` (~85-105); `setupMonaco` keeps its guard and calls them in today's order.
- [ ] metrics: `long_functions` 0, `complex_functions` falls (12 expected). Rewrite the baseline. vitest, check, lint, e2e (the JSON editor specs exercise Monaco). Commit `refactor(ui): split pipelineDigest and setupMonaco`.

### Task 5: Break the api/workspace import cycle

- [ ] Create `ui/src/lib/workspaceState.svelte.ts` holding `DEFAULT_WORKSPACE` and the `workspace` `$state` object (moved from `workspace.svelte.ts` ~6, ~18-26, with their doc comment). `workspace.svelte.ts` imports them and re-exports: `export { DEFAULT_WORKSPACE, workspace } from './workspaceState.svelte'`, so no importer changes. `api.ts` imports from `./workspaceState.svelte`.
- [ ] metrics: `import_cycles` 0, `modules_in_import_cycles` 0. Rewrite the baseline. vitest (`workspace.svelte.test.ts`, the page tests that set `workspace.current`), check, lint. Commit `refactor(ui): workspace state in its own module; the api cycle is gone`.

### Task 6: api.ts - one scoping rule, one request path, smaller file

**Interfaces** (in `apiUrls.ts`, re-exported by `api.ts` where pages use them): `scopeTo(path: string, ws: string): string` (no default: an undefined workspace meant "unscoped" before); `addToken(url: string): string`; `withToken(url: string, ws: string = workspace.current): string`; `outputUrl` (moved). In `api.ts`: `send(path: string, init?: RequestInit, scope?: boolean): Promise<Response>` (bearer, scope, `ApiError` on a non-ok answer) under both `fetchJson` and `downloadResponse`. `apiErrors.ts` takes `ApiError`, `errorDetail`, `describeContents` (~107-172).

- [ ] First, a failing-or-pinning pass in `api.test.ts`: the job's-own-workspace URL tests (~183, ~235) already pin scoping; add one asserting a failed download throws an `ApiError` whose message is the server's detail (today it throws a plain `Error`; nothing checks the type, so this is the one intended difference - record it).
- [ ] Implement: `scoped`/`workspaceScopedPath` become `scopeTo`; `withToken`/`withTokenIn` become `withToken(url, ws?)`; `exportZipUrl`'s token tail uses `addToken`; `fetchJson`/`downloadResponse` call `send`. Move the URL helpers and the error helpers out; `api.ts` re-exports `ApiError`, `errorDetail`, `outputUrl`, and keeps `streamJobEvents`, `TERMINAL_STATUSES`, `fetchOutputText`.
- [ ] vitest, check, lint, metrics (`api.ts` under 600; about 535), e2e. Commit `refactor(ui): api.ts has one scoping rule and one request path`.

### Task 7: GalleryPage - the detail panel

- [ ] Extract `ui/src/lib/gallery/GalleryDetail.svelte`: markup ~342-462, `keepAsAsset` (~48-86), the metadata-derived workflow/args/prompt/seed (~200-222, now via `describeOutputMeta`), `openAsWorkflow` (~224-231), styles ~563-612. Props `{ file: GalleryFile; metadata: Record<string, unknown> | null; metadataLoading: boolean; sourceJob: { id: string; status: string } | null; onremove: () => void; onclose: () => void }`. `select()` and `removeFile` stay in the page (moving the fetch into a child effect would stop a re-click from refetching).
- [ ] vitest (`GalleryPage.test.ts` unchanged), check, lint, metrics, e2e. Commit `refactor(ui): the gallery's detail panel is its own component`.

### Task 8: JobPage - header, progress, results

Three commits, one per piece, each green.

- [ ] `ui/src/lib/job/JobHeader.svelte`: markup ~343-436, `rerun` (~135-143), `exporting` + `exportJob` (~145-186, complexity 12, moved intact), styles `.head`/`.seed` (~675-692). Props `{ job: JobDetail; jobId: string; seed: number | undefined; runVersion: number | undefined; running: boolean; cancelPending: boolean; seedVariable: string | null }`.
- [ ] ETA as pure functions in `progress.ts` with tests first: `estimateEta(stepTimes: number[], denoise): number | null` (body of ~229-237) and `nextStepTimes(times: number[], event: JobEvent, now: number): number[]` (the branch at ~103-114); then `ui/src/lib/job/JobProgress.svelte` (markup ~449-490, styles ~693-698 and ~728-775), props `{ steps: string[]; finishedSteps: string[]; listStep: string | undefined; running: boolean; progress: StepProgress; etaSeconds: number | null }`.
- [ ] `ui/src/lib/job/JobResults.svelte`: markup ~534-652, `fileMeta` and its effect (~256-277), `describe` → `describeOutputMeta`, `allReused`/`sections`/`sectioned` (~293-301), `fileUrl` (~330-331), styles ~671-674 and ~776-874. Props `{ job: JobDetail; fileGroups: StepGroup[]; unsaved: UnsavedStep[]; kinds: Record<string, string | null>; seedVariable: string | null }`. The page wraps it in `{#key jobId}`, which replaces the `fileMeta = {}` reset. Before this move, add a JobPage test (Review Focus 1): render job A with an image, then switch `jobId` to job B with another, and assert A's metadata is gone and the fetch order is `[A's image, B's image]`.
- [ ] After each: vitest (JobPage.test unchanged otherwise), check, lint, metrics, e2e. Commits `refactor(ui): the job page's header / progress / results are components`.

### Task 9: AssetsPage and StepEditor

- [ ] AssetsPage: `ui/src/lib/assets/AssetDetail.svelte` (markup ~446-500, styles ~634-665; props `{ asset: AssetFile; busy: boolean; onremove: (a: AssetFile) => void; onclose: () => void }`), `ShadowedAssets.svelte` (markup ~412-439, `SHADOWED_BY` ~46-50, styles ~548-563 and ~604-633; props `{ entries: ShadowedAsset[] }`; `leaf` moves to `names.ts` as `leafName`), `LibraryRow.svelte` (markup ~317-356, styles ~516-547; props `{ section; open: boolean; shared: boolean; busy: boolean; ontoggle; onupload: (t: 'workspace' | 'shared') => void }`). One commit each.
- [ ] StepEditor: `ui/src/lib/editor/PipelineOptions.svelte` (the components/loras/scheduler/acceleration `<details>` ~382-529, script ~159-208, styles ~625-681; props `{ pipeline: Record<string, any> (bindable); index: number; openSection: string }`, keeping `configuration = $derived(pipeline.configuration ?? {})` as written) and `StepDigestList.svelte` (markup ~302-316, styles ~728-759; props `{ lines: DigestLine[]; onopen: (section: string) => void }`). `.hint`/`.error` rules are copied to each side that uses them.
- [ ] After each: vitest, check, lint, metrics, e2e (step-model.spec's `.digestline`, `.summary`, `.panel.step` and responsive.spec's `.step .grid` keep their classes). Commits per piece.

### Task 10: PromptEditorPage onto the editor shell and an enhancer module

- [ ] **First, coverage:** PromptEditorPage has no unit test. Add `PromptEditorPage.test.ts` (mock `'../api'` as `EditorPage.test.ts` does): it mounts a stored prompt; switching form → JSON → form keeps an edit; Ctrl+S saves to the folder/name shown; a dirty doc arms `beforeunload` (dispatch the event and assert `defaultPrevented`); choosing an enhancer preset and starting a model download, then switching view, keeps the download state (Review Focus 2). Run: PASS on today's code (these pin behaviour the moves must keep).
- [ ] `ui/src/lib/editorShell.svelte.ts`: `class DocumentEditor` with `doc`, `baseline`, `jsonDraft`, `jsonParseFailed`, `view`, `busy`, `saveName`, `folder`, `newFolder` as `$state`; `dirty` as `$derived`; constructor `{ viewKey: string; views: readonly EditorView[]; legacyView?: () => EditorView | null }` registering the JSON-mirror and `beforeunload` effects (constructed during component init); methods `setView`, `applyJson`, `load`, `markSaved`, `directory`, `savePath`, `commitNewFolder`, `takeImport(importKey, folderKey)`. Unit-test the class through a tiny harness component (view persistence, `applyJson` on bad JSON sets `jsonParseFailed`, `savePath` with a new folder).
- [ ] Components: `editor/ViewSwitch.svelte` (props `{ view; options: { view; label; title; icon }[]; onselect }`, carrying `.viewswitch`/`.activebtn`), `editor/FolderPicker.svelte` (props `{ folder (bindable); newFolder (bindable); folders: string[]; id?: string; newFolderTitle: string }`, carrying `.folderpick`/`.newfolder`), `editor/EditorBody.svelte` (props `{ view; jsonDraft; onjson; schema?: 'prompt'; hint; stickyTop: string; form: Snippet }`, carrying `.editwrap`/`.jsoncol` with `style:top={stickyTop}`). Titles and button text stay verbatim (EditorPage.test finds them).
- [ ] PromptEditorPage uses the shell and the three components; then `ui/src/lib/enhancer.svelte.ts` (`class Enhancer` holding ~102-115's state; `load`, `pickPreset`, `preselect(intended)`, `refreshModels`, `downloadModel` - the sequential wait loop with `sleep` - `generate`, `cancel`, `stop`, `takeResult`, `intendedModels(details)`; `preselect` stays called on load and on the intended-model field's change, not as an effect) and `editor/EnhancePanel.svelte` (props `{ enhancer: Enhancer; onuse }`; markup ~627-735, styles ~858-895).
- [ ] After each commit: vitest, check, lint, metrics, e2e (responsive.spec opens a prompt editor). PromptEditorPage under 600 (about 470).

### Task 11: EditorPage onto the shell; validation, file bar and step list

- [ ] EditorPage uses `DocumentEditor`, `ViewSwitch`, `FolderPicker`, `EditorBody` (flow view stays `{#if view === 'flow'}<FlowView/>{:else}<EditorBody>`). Every `workflow.` in markup becomes `ed.doc.`.
- [ ] `editor/ValidationPanel.svelte` (markup ~578-623, styles ~935-945 and ~994-1024; props `{ validation: ValidationResult; ondismiss: () => void }`).
- [ ] `editor/WorkflowFileBar.svelte` (markup ~499-576, `savePreview` ~147-151, styles ~797-867; props `{ ed: DocumentEditor; workflowDir: string; folders: string[]; fileOpen (bindable) }`).
- [ ] `ui/src/lib/stepModes.svelte.ts` (`class StepModes` replacing ~54-69 and the restore calls: `of`, `set` (persists), `mark` (no persist, as `addStep` does), `setAll`, `restore`, `reset`), then `editor/StepList.svelte` (markup ~645-715, `addStep`/`removeStep`/`moveStep` ~258-288, styles ~880-894 and ~952-993 including `.steprow.flowlit :global(.panel.step)`; props `{ steps (bindable); modes: StepModes; folder: string; flow: StepFlow[]; problemsByStep: Map<number, string[]>; stepReferences: string[][]; hovered (bindable) }`). Before this move, add an EditorPage test (Review Focus 3): add a step, move it up, remove another, then switch to JSON and assert the JSON shows the same steps in the same order.
- [ ] After each: vitest, check, lint, metrics, full e2e (step-model.spec drives the step list). EditorPage under 600 (about 470).

### Task 12: Gate 3

- [ ] metrics `--write`: `files_over_size_ceiling` 0, `import_cycles` 0, `modules_in_import_cycles` 0, `long_functions` 0; nothing else above gate 2. If a file is still over 600, the gate has not passed - split it before going on.
- [ ] Seam-map rows: the editor shell (`ui/src/lib/editorShell.svelte.ts`: one save/view/JSON/leave-guard rule for both editors), polling (`ui/src/lib/poll.ts`), workspace state (`ui/src/lib/workspaceState.svelte.ts`).
- [ ] `scripts/preflight.sh` (pytest with `DW_DEVICE=cpu` on an MPS Mac; integration without it).
- [ ] One fresh-reviewer pass with this plan's Review Focus; Critical and Important fixed test-first.
- [ ] With Don's word: merge, push, deploy to lem; browser smoke on lem: a job page with outputs (header, progress of a finished job, results), the workflow editor (add/move/remove a step, switch views, validate), the prompt editor (switch views, the enhancer panel opens), gallery and assets details, server page polling.
- [ ] Gate report in `ROADMAP.md` (ratchet column, file sizes before/after, smoke); tag `ui-stabilization-gate-3` (annotated); push the tag on Don's word.
