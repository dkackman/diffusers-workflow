# UI quality phase 3: consistency and vocabulary. Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the same concept looks, acts and is named the same on every page. Every route is a titled page with one header. Every destructive action follows one policy. Every button that changes something is an `action()`. No copy breaks the glossary. Each of these is held by a metric or a test that fails when the next page gets it wrong.

**Architecture:** Three new mechanical rules go in first: `window_dialogs`, `unguarded_mutations` and `banned_terms`, plus `duplicate_blocks`, all in `ui/scripts/arch-metrics.mjs`. The banned terms come from `ui/src/lib/terms.ts`, which `ui/GLOSSARY.md` explains. Each later task then burns down a count and does not raise any other. The page frame is `PageHeader` and `pageFrame`, its labels come from `SECTION_LABELS` in `routes.ts`, and an unknown route renders a not-found page. The P5 policy lives in `ui/src/lib/destructive.ts` (the confirms) and in `action({ undo })` (the undos). The repeated concepts from the duplication survey become one piece each: `Segmented`, `StatusChip`, `hashState`, `nameDialog`, `DocumentFileBar`, a shared note draft, and the global classes in a new `src/patterns.css`.

**Tech stack:** Svelte 5 runes, vitest with @testing-library/svelte, Playwright, ESLint flat config (with eslint-plugin-svelte's AST for the copy rule), Node 24 type stripping (`arch-metrics.mjs` imports `src/lib/terms.ts`). There is no server change.

**Spec:** `docs/superpowers/specs/2026-11-22-ui-quality-design.md`. This plan covers:
- principles P4, P5 (and its policy table), P9 and P10;
- the building-blocks rows *Disclosure of engine text*, *Page frame* and *Vocabulary*;
- the metrics `window_dialogs`, `unguarded_mutations` and `banned_terms`;
- *Keeping it from coming back* (`duplicate_blocks`, and the design rules for tokens, status colours and the segmented style);
- the ★ vocabulary rows, and decision 4 (Update from GitHub).

The findings are SH-10, SH-11, SH-13, SH-18, XC-10..13, XC-16, RQ-6, RQ-12, WF-15 and WF-18. Their text is in `docs/superpowers/specs/2026-11-22-ui-audit.md`: Tier 3 *Shell*, *Requests*, *Accessibility* and *Other*; Tier 2 §D, the WF-18 undo; §F *Vocabulary*. The survey items are D3, D4, D5, D6, D7, D10, D11, D12, D14, D15 and D16. Their file:line occurrences are in `docs/superpowers/specs/2026-11-22-ui-duplication-survey.md`, under §3-§7, §10-§12, §14, §16-§18, §20-§23 and *Shared pieces that pages bypass*. Each is named in the task that absorbs it.

**Out of scope:**
- **#1599**, the created and last-changed timestamps pass. Task 1's glossary and Task 12's duration format are what it builds on. This phase does no timestamp inventory or fill-in.
- **#1597** (note gaps) and **#1598** (deferred minors from 2a and 2b) are separate issues. Task 19 touches `NoteEditor`, which #1597 also touches; that task names the overlap and does none of #1597.
- Behaviour behind a vocabulary row: "Accepting also stars the file" is a change to the request flow, so it goes to phase 6. Here only the words change.

## Measured at the start (develop b581ea58)

| Count | Value | Where |
|---|---|---|
| `window.confirm` / `window.prompt` / `alert(` calls | **1** | `src/lib/gallery/GalleryDetail.svelte:95` (Keep as asset). Phase 1 already removed the AssetsPage upload prompt. |
| Inline `onclick={() => api.<write>(…)}` | **0** | The spec's literal pattern no longer occurs, so the metric counts what the pattern stood for (Task 1). |
| `api.<write>` calls in `.svelte` outside `action()` | **25** | CachesPanel 2, GalleryDetail 2, RerunWithChanges 1, AssetsPage 3, EditorPage 1, GalleryPage 4, JobsPage 1, ModelsPage 4, PromptEditorPage 1, WorkflowPage 2, NoteList 3, RequestMenu 1 |
| Copy that breaks Task 1's rules | **62** | `~` before a figure 7, run as a noun 38 (2 of them `Run note`), Keep as asset 5, star to keep 1, the session wording 6, held for your review 2, "a/the template" 2, result (latest) 1 |
| Repeated 6-line blocks across `.svelte` files | **48** | the Task 1 algorithm |
| `var(--x)` with no definition | **2 real** (`--danger` ×2, `--sunk` ×4) | `--bits-combobox-anchor-width` is Bits UI's own |
| Python `modules` / `claude_md_lines` | 281 / 129 | `ui/CLAUDE.md` is 12 lines |

UI metrics at the start: `files_over_size_ceiling 0, complex_functions 13, long_functions 0, import_cycles 0, modules_in_import_cycles 0, prefix_literals 3, a11y_suppressions 3, raw_error_toasts 0, empty_outside_loadstate 0, unwatched_reads 0`. Against `docs/stabilization/ui/baseline.json`, develop is already red on `complex_functions 11 -> 13` and `prefix_literals 0 -> 3`. Neither is this phase's to fix, and neither may rise.

Page sizes (the limit is 600):

| File | Lines |
|---|---|
| `app.css` | 600, at the limit |
| `EditorPage` | 591 |
| `ModelsPage` | 565 |
| `AssetsPage` | 558 |
| `GalleryPage` | 550 |
| `Sidebar` | 547 |
| `PromptEditorPage` | 542 |
| `App.svelte` | 529 |
| `JobsPage` | 525 |
| `WorkflowsPage` | 500 |
| `StepEditor` | 500 |

## Global constraints

- **MCP behaviour is unchanged.** This phase makes no server change, so there is no OpenAPI regeneration. Every route keeps its payload.
- **No new Python module.** `modules` stays at 281. `claude_md_lines` stays at 129: a rule added to `ui/CLAUDE.md` is paid for by rewriting its existing lines, never by adding one. The spec's "before you write markup" table therefore goes in `ui/GLOSSARY.md` (Task 21), and `ui/CLAUDE.md` gets one rewritten pointer to it.
- **Size limits.** No page over 600 lines, and no function over complexity 10 or 60 lines.
  - `app.css` is at 600 and must not grow. New global classes go in `src/patterns.css` (created in Task 15), imported in `main.ts` after `frames.css`.
  - `EditorPage` (591) may only shrink. Logic a task adds to a page near the limit moves into the `.ts` module the task names.
- **The UI metrics gate: no metric rises.** Before Task 1, run `node scripts/arch-metrics.mjs > "$SCRATCH/ui-start.json"`, where `SCRATCH` is any scratch folder. After Task 1, run `node scripts/arch-metrics.mjs > "$SCRATCH/ui-p3.json"` too: it holds the new metrics' starting counts.
  - Each task's gate is `node scripts/arch-metrics.mjs > "$SCRATCH/ui-now.json" && node scripts/arch-metrics.mjs --compare "$SCRATCH/ui-now.json" "$SCRATCH/ui-p3.json"`, and it must print nothing.
  - `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` may print only the two lines that are already red.
  - A task that burns a count down lowers that key in `baseline.json` in the same commit.
  - By Task 21, `window_dialogs`, `unguarded_mutations` and `banned_terms` are 0 in `baseline.json`, and `duplicate_blocks` is below 48.
- **The phase gate** is `ui/scripts/copy-rules.test.ts`, whose source case asserts no violations (Task 21).
- **Staging and commits.** Stage files by name; never `git add -A ui`, because Don edits `ui/package.json` locally. Commit headlines use a scope (`ui(…)`, `docs(…)`), never `fix: #N`. Each commit ends with the `Co-Authored-By` line.
- **Checks.** UI checks run from `ui/`: `npm run check`, `npm run lint`, `npx vitest run <file>`, and `npx playwright test <file>`, which uses port 8971. Python: `venv/bin/python scripts/arch_metrics.py`.
- WCAG contrast is guidance only, never a check. Keyboard and screen-reader basics (focus, names, `aria-pressed`, `aria-expanded`) are in scope.
- **Building blocks.** Each new building block gets a row in the UI table of `docs/ARCHITECTURE.md`, in the task that builds it, naming its owner module and its test.
- **Known flaky test.** `tests/test_ledger_job_link.py::TestDelete::test_a_request_made_again_under_the_id_takes_the_jobs_back` is not this phase's to fix. Don't chase it.
- **Copy fixed by this plan.** Use it exactly. `<…>` is filled in.
  - **Durations:**
    - an estimate: `about 3 min` (the measured value to 2 decimals, as `formatMinutes` keeps it today)
    - under a minute: `under 1 min`
    - an hour or more: `about 1 h 20 min`
    - an ETA: `about <duration> left`
    - host RAM: `host RAM about <gb> GB (<basis>)`
    - a total: `total about <m> min`
    - Measured time (`formatSpent`, `formatDuration`) has no "about".
  - **Destroying a file**, via `destroyConfirm`:
    - title: `Delete <name>?` for one, or `Delete <n> <nouns>?` for several
    - message: `This removes <it|them> on disk.`, then `It is starred.` or `<k> of them <is|are> starred.` when that applies, then the caller's consequence sentence
    - buttons: **Delete** (danger) and **Cancel**
  - **Deleting a workspace:**
    - title: `Delete workspace "<name>"?`
    - message: `This permanently removes <n> outputs (<k> starred), <n> assets and <n> workflows (<size>). The prompt library and the shared asset library are kept.` A folder with no files is left out. With no starred files, drop ` (<k> starred)`.
    - buttons: **Delete workspace** (danger) and **Cancel**
  - **Managing the default workspace** (SH-11): `The default workspace is the server's own folder, so it can't be hidden or deleted.`
  - **Update diffusers** (decision 4):
    - title: `Update diffusers from GitHub?`
    - message: `This changes the server's Python environment and restarts the idle worker.`
    - button: **Update diffusers**
  - **Deleting a model:**
    - title: `Delete <repo> (<gb> GB) from the model cache?`
    - message: `The next workflow that needs it downloads it again.`
    - button: **Delete model**
  - **Undo toasts**, each with **Undo**:
    - `Starred <leaf>` and `Unstarred <leaf>`
    - `Removed step <name>` and `Removed variable <name>`
    - `Marked "<title>" done` and `Abandoned "<title>"`
  - **Save as asset:**
    - the button: **Save as asset**
    - the name dialog: title `Save as asset`, field label `Name in the asset library`, buttons **Save** and **Cancel**
    - on an existing name: `"<name>" is already in the asset library.`, with **Replace** (danger), **Keep both** (it saves as `freeName`'s suggestion, which the button's title states) and **Cancel**
    - success: `Saved as asset <name>`
  - **Not found:**
    - h1: `Page not found`
    - text: `There is no page at <path>.`
    - link: **Go to the overview**
  - **Skip link:** `Skip to content`.
  - **The header meter's state word:** `running` with a job in flight, else `idle`.
  - **An empty Compare:** `Nothing to compare yet.`, then `Tick two or more files in the Gallery, then press Compare.`, with the link **Open the Gallery**.
  - **No examples:** `No examples on this server.`, with the remedy `Start dw.serve with --examples-dir <folder> to list some.`
  - **Notes:**
    - the person's own notes: `Your note on this job`, `Your note on this file`, `Your note on this asset`, `Your note on this workspace`
    - the agent's channel keeps `Note for the agent`
  - **The hold** (vocabulary row; the RQ-1 behaviour is phase 6):
    - the lead: `<n items> waiting for your OK`
    - with the agent listening: `The agent will run <it|them> at <time> if it's connected (in <countdown>), unless you leave a note.`
    - without: `The agent isn't connected, so nothing runs at <time> unless it reconnects. Run <it|them> here, or leave a note.`
    - due, listening: `Review time is up: the agent should be starting <it|them>.`
    - due, not listening: `Review time passed while the agent wasn't connected: nothing ran.`
  - **The note channel:** `The agent is connected; it will see this within about 10 seconds.` and `The agent isn't connected. It reads this the next time it updates the plan.`
  - **The version badge's title:** `<ordinal> job of this workflow in this folder`, for example `2nd job of this workflow in this folder`.
  - **A take:** `latest take` replaces `result (latest)`.

## Review focus

1. **Deleting something that holds starred files.** Two cases: a workspace whose outputs include starred files, and a single starred file in the gallery detail. The confirm states the starred count, and its button is styled as danger. Pinned in Task 7 (`the workspace confirm counts what it removes, starred included`, `a starred file's confirm says so`).
2. **Undo after the world moved on.** Three cases:
   - Undo a removed step after adding another one: the step goes back where it was, and nothing is lost.
   - Undo an unstar after the file was deleted: the toast says it couldn't, through `notify.failure`, with no unhandled rejection.
   - Undo a "Mark done" after an agent reopened the request: the store's `stale_revision` path, not a silent overwrite.

   Pinned in Task 9.
3. **Back after narrowing and scrolling a long list, while it reloads.** The filters and the scroll position come back, and a quiet reload during the restore does not jump to the top. Pinned in Task 18 and in the journey in Task 21.
4. **Focus moves to the new page's h1 only on a real navigation.** Typing in a filter (a query-only hash change), a quiet reload and the first load never move focus. Pinned in Task 3 (`PageHeader focuses its h1 on a path change, never on a query change or the first render`).
5. **The name dialog with a bad name, an existing name and Escape.**
   - An invalid name keeps **Save** disabled and says why.
   - An existing one asks Replace, Keep both or Cancel.
   - Escape or Cancel at either step does nothing, with no error toast (the GA-13 shape).

   Pinned in Task 10.

---

### Task 1: The new metrics, the glossary and the copy rules

The spec's `window_dialogs`, `unguarded_mutations` and `banned_terms`, plus `duplicate_blocks` from *Keeping it from coming back*.

**Files:**
- Create: `ui/src/lib/terms.ts`, `ui/GLOSSARY.md`, `ui/scripts/copy-rules.test.ts`
- Modify: `ui/scripts/arch-metrics.mjs`, `ui/scripts/arch-metrics.test.ts`, `docs/stabilization/ui/baseline.json`
- Docs: `docs/ARCHITECTURE.md`, a new UI row *Vocabulary*

**Interfaces:**
- Produces, in `terms.ts`. It must be dependency-free and use only erasable TypeScript, because Node imports it without a build:
  - `export type Banned = { id: string; pattern: RegExp; use: string; why: string }`
  - `export const BANNED: Banned[]`. Its ids and patterns:
    - `tilde`: `/(^|[\s(·])~(\s?\d|$)/`. Use `about 3 min`.
    - `keep-asset`: `/\bkeep(s|ing)? (it |this file |\S+ )?as (an? )?(input )?asset\b/i`. Use `Save as asset`.
    - `star-keep`: `/\bstar to keep\b/i`. Use `Star`.
    - `latest-result`: `/\bresult \(latest\)/i`. Use `latest take`.
    - `run-noun`: `/\b(this|the|a|each|every|earlier|last|latest|first) runs?\b(?! (it|them|again|now|here|the|a|this|once|only)\b)|\bRun note\b/i`. Use `job` (or `take` inside a request).
    - `session`: `/\bsession\b/i`. Use `the agent is connected` / `isn't connected`.
    - `held-review`: `/held for your review/i`. Use `waiting for your OK`.
    - `starter`: `/\bstarters?\b|\b(a|the) template\b/i`. Use `an example`.
  - `export const COPY_OWNERS = new Set(['src/lib/terms.ts'])`
  - "attempt" is deliberately not banned: in the UI it names only a chain segment's retry inside one job (`chainRows.ts`). GLOSSARY.md says so.
- Produces, in `arch-metrics.mjs`:
  - `export const WRITES`: the "writes" section of `NOT_READS`, split out. `NOT_READS` becomes `[...WRITES, ...checks, ...urlBuilders]`, so `misclassified` is unchanged.
  - **`window_dialogs`**: every `CallExpression` whose callee is `window.confirm`, `window.prompt` or `window.alert`, or a bare `alert`, `confirm` or `prompt` identifier. Detail: `file:line`.
  - **`unguarded_mutations`**: in a `.svelte` file, an `api.<WRITES>` call that is lexically inside no `action(` call and inside no function declaration whose name is referenced within an `action(` call's arguments in the same file. That is one level, so a helper like `JobHeader`'s `exportOrReplace` counts as guarded. Detail: `file:line name`.
  - **`banned_terms`**: an `arch/copy` rule over every source file outside `COPY_OWNERS`. It reads:
    - a string `Literal` containing whitespace;
    - a `TemplateLiteral` whose quasis are joined with `"0"`, if the result contains whitespace;
    - `SvelteText`, except inside `<style>`;
    - `SvelteLiteral`.

    Each `BANNED.pattern` that matches counts once. Detail: `file:line id`.
  - **`duplicate_blocks`**, over `.svelte` files:
    - Lines are trimmed and their whitespace collapsed.
    - Blank lines and trivial ones are dropped. Trivial means matching `/^([{}()\[\];,]*|<\/?[a-z]+>|\/\/.*|\/\*.*|\*.*|\*\/|-->|<!--.*)$/i`.
    - Sliding 6-line windows are hashed with sha1. A window whose hash occurs in two or more files is shared.
    - The count is the maximal runs of consecutive shared windows, per file, summed over files.
  - Known limits, added to the header comment:
    - a write made in a `.ts` helper (`workspaceActions.ts`, `uploads.ts`, `promptFile.ts`) or through a named import (`setNote`) is not counted;
    - copy assembled from single words, or across `{…}` in markup, can slip through.

- [ ] **Step 1: Write the failing tests.**
  - `copy-rules.test.ts` (`@vitest-environment node`):
    - `each rule catches what it bans and spares what it allows`. A table per id, with these samples:
      - `tilde`: bad `'~3 min'`, `'· ~'`; good `'about 3 min'`, `'~/.cache'`.
      - `keep-asset`: bad `'Keep as asset'`, `'keep this file as an input asset'`; good `'Keep mine'`, `'Keep both'`, `'Keep editing'`.
      - `run-noun`: bad `'this run used'`, `'earlier runs'`, `'Run note'`; good `'Run it here'`, `'the pipeline runs once'`, `'nothing runs until you press Run'`.
      - `session`: bad `'No session is listening'`; good `'sessionStorage'`.
      - `starter`: bad `'the blank starter'`; good `'workflows/templates'`.
    - `every banned term's replacement is in GLOSSARY.md`: each `use` occurs in the file.
  - `arch-metrics.test.ts`: each new metric over throwaway files in a temp `src`, through `measure(files)`:
    - `window_dialogs counts window.prompt and a bare confirm, not confirmDialog`
    - `unguarded_mutations spares a write inside action() and in a function action() names`
    - `banned_terms skips style text and terms.ts`
    - `duplicate_blocks counts one shared 6-line run once per file`
- [ ] **Step 2: Run** `cd ui && npx vitest run scripts/copy-rules.test.ts scripts/arch-metrics.test.ts`. Expected: FAIL (no `terms.ts`, no metrics).
- [ ] **Step 3: Implement**, as in Interfaces.
  - `GLOSSARY.md` has two tables. Below them, a "Before you write markup" heading is left for Task 21.
  - The first table, **Words**, has the columns term, means, say, don't say. It holds every vocabulary row of the spec, plus `attempt` (a chain segment's retry only), `take` (only inside a request), and `card` (queue text) versus `GPU` (the hardware line).
  - The second, **Formats**, holds durations (`about 3 min`, `under 1 min`), the measured time `formatDuration`, and the ordinal version title.
- [ ] **Step 4: Run** the two test files, then `node scripts/arch-metrics.mjs`. Expected: PASS, and the measurement reads `window_dialogs 1, unguarded_mutations 25, banned_terms 62, duplicate_blocks 48`. If any differs, re-check the detail against *Measured at the start* before going on. Write the four keys into `baseline.json` at the measured values, and save `$SCRATCH/ui-p3.json`.
- [ ] **Step 5: Commit** `ui(metrics): window dialogs, unguarded mutations, banned terms and duplicate blocks; the glossary`.

### Task 2: Section labels, the breadcrumb and unknown routes

SH-18 (the breadcrumb uses internal keys; a bad section is not corrected) and XC-12 (unknown routes).

**Files:**
- Modify: `ui/src/lib/routes.ts`, `ui/src/lib/Sidebar.svelte`, `ui/src/lib/Breadcrumb.svelte`, `ui/src/App.svelte` (the `{:else}` chain gains `missing`)
- Create: `ui/src/lib/pages/NotFoundPage.svelte`
- Test: `ui/src/lib/routes.test.ts`, `ui/src/lib/Breadcrumb.test.ts`, `ui/src/lib/pages/NotFoundPage.test.ts`
- Docs: ARCHITECTURE, a new UI row *Routes and their names*

**Interfaces:**
- Produces, in `routes.ts`:
  - `export const SECTION_LABELS`, of type `{ ws: Record<WsSection, string>; shared: Record<SharedSection, string>; server: Record<ServerSection, string>; groups: { shared: string; server: string } }`. The labels:
    - workspace sections: Overview, Workflows, Jobs, Requests, Gallery, Compare, Assets, Editor
    - shared sections: Prompts, Prompt editor, Assets, Examples
    - server sections: Models, Schema, Status
    - groups: Shared, Server
  - `RouteView` gains `| { kind: 'missing'; path: string }`.
  - `parseView`:
    - `ws/<name>` with no section stays the overview;
    - `ws/<name>/<unknown>` is `missing`, so the URL is left as typed;
    - an unknown `shared/…` or `server/…` is `missing`.
  - `legacyRedirect`'s `default:` case returns `null`, so `parse()` yields `missing`. The named legacy cases stay.
  - `sectionLabel(view: RouteView): string`. `missing` reads `Not found`.
- `Sidebar`'s three item lists take their `label` from `SECTION_LABELS`, so the sidebar and the breadcrumb cannot drift.
- `Breadcrumb` shows these labels. The workspace crumb keeps its name and dot.
- `NotFoundPage` has no props and reads `route.view.path`. It uses the copy in Global constraints, and its link goes to `wsHref(workspace.current, 'overview')`.

- [ ] **Step 1: Write the failing tests:**
  - `routes.test.ts`:
    - `an unknown workspace section is missing, not the overview`
    - `an unknown top-level path is missing`
    - `every section has a label`, which iterates `WS_SECTIONS`, `SHARED_SECTIONS` and `SERVER_SECTIONS`
  - `Breadcrumb.test.ts`: `shows the sidebar's labels` (`#/shared/prompt-edit/x` reads `Shared / Prompt editor`), and `a missing route reads Not found`
  - `NotFoundPage.test.ts`: `names the path and links the overview`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/routes.test.ts src/lib/Breadcrumb.test.ts src/lib/pages/NotFoundPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `App.svelte` (529) stays under 600.
- [ ] **Step 4: Run** the three files, plus `npx vitest run src/lib/router.svelte.test.ts` and `npx playwright test e2e/chrome.spec.ts`. Expected: PASS.
- [ ] **Step 5: Commit** `ui(routes): one set of section labels; an unknown route is a not-found page`.

### Task 3: `PageHeader`, the tab title and focus on navigation, on the list pages

P10, the *Page frame* row, XC-10 (titles and focus) and D12 (survey §12). This task covers the list pages.

**Files:**
- Create: `ui/src/lib/ui/PageHeader.svelte`, `ui/src/lib/pageFrame.svelte.ts`
- Modify: `ui/src/lib/finishAlert.ts` (`showTitle`), `ui/src/lib/jobWatch.ts` (`tabTitle` and `endedTitle` take the page title)
- Modify: the list pages: `PromptsPage`, `WorkflowsPage` (Workflows and Examples), `AssetsPage`, `GalleryPage`, `JobsPage`, `RequestsPage`, `ComparePage`
- Test: `ui/src/lib/ui/PageHeader.test.ts`, `ui/src/lib/jobWatch.test.ts`, `ui/src/lib/pages/ComparePage.test.ts`, `ui/src/lib/pages/WorkflowsPage.test.ts`
- Docs: ARCHITECTURE, a new UI row *Page frame*

**Interfaces:**
- Produces:
  - `pageFrame.svelte.ts`:
    - `export const pageFrame = $state({ title: '' })`
    - `export function pageTitle(): string` returns `${pageFrame.title} · ${BASE_TITLE}`, or `BASE_TITLE` when the title is empty.
    - `BASE_TITLE` moves here from `jobWatch.ts` and is re-exported there.
  - `tabTitle(job, others, base = pageTitle())` and `endedTitle(job, base = pageTitle())`. The running-job prefix keeps its form, and `base` replaces the bare app name.
  - `finishAlert.showTitle` is exported as `refreshTitle()`, and `PageHeader` calls it when its title changes. `document.title` has this one writer.
  - `PageHeader.svelte` props: `{ title: string; secondary?: string; count?: { n: number; noun: string; plural?: string }; back?: { href: string; label: string }; actions?: Snippet; filter?: Snippet; baseline?: boolean }`.
    - It renders `.pagehead`, then an optional back link `← <label>` (the one back-link form, which replaces `nav.crumbs`, `← prompts` and `back to the gallery`), then `<h1 tabindex="-1">{title}</h1>`, then `secondary` in mono muted, then the count chip `<n> <noun|plural>`, then the `filter` snippet (styled `.pagehead .filter { max-width: 220px }`, which replaces the 5 page copies), then the actions.
    - It sets `pageFrame.title = title`.
    - It focuses the h1 after a navigation. That is when `route.parts`, the path, changes after the component first rendered. A query-only change (`setHashQuery`) never moves focus, and neither does the first render.
- Removed: the dead `.pagehead .flex` copies in Prompts and Workflows, and every page-local `.filter { max-width }`.
- `ComparePage` with fewer than 2 names shows the XC-16 empty copy through `LoadState`'s `empty`.
- `WorkflowsPage` under `examples` shows the no-examples copy (XC-16).

- [ ] **Step 1: Write the failing tests:**
  - `PageHeader.test.ts`:
    - `sets the tab title from its title`
    - `PageHeader focuses its h1 on a path change, never on a query change or the first render`
    - `renders the count with its noun`
    - `renders one back-link form`
  - `jobWatch.test.ts`: `the running title keeps the page's name` (`▶ 0% · ref-to-video +1 — Gallery · diffusers-workflow`).
  - `ComparePage.test.ts`: `with fewer than two files says how to pick them`.
  - `WorkflowsPage.test.ts`: `with no examples says how to add them`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/ui/PageHeader.test.ts src/lib/jobWatch.test.ts src/lib/pages/ComparePage.test.ts src/lib/pages/WorkflowsPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Every page touched loses lines. Run `wc -l` on `GalleryPage` and `AssetsPage` before and after.
- [ ] **Step 4: Run** the tests, `npx vitest run src/lib/pages/`, `npx playwright test e2e/smoke.spec.ts e2e/responsive.spec.ts`, and the metrics gate. Expected: PASS, with `duplicate_blocks` not above `ui-p3.json`.
- [ ] **Step 5: Commit** `ui(frame): PageHeader titles the tab and takes focus on navigation; list pages use it`.

### Task 4: `PageHeader` on every other page

D12's own-`.head` pages, and the rest of P10.

**Files:**
- Modify:
  - `OverviewPage`, `WorkflowPage`, `EditorPage`, `PromptEditorPage`, `ModelsPage`, `ServerPage` and `StatusPage`, `SchemaPage`, `NotFoundPage`
  - `ui/src/lib/job/JobHeader.svelte` (the job page's h1)
  - `ui/src/lib/request/RequestHeader.svelte` (the request page's h1)
- Test: each page's existing test file gains a title case

**Interfaces:**
- Consumes: Task 3's `PageHeader`.
- The h1s:
  - `JobHeader`'s h1 becomes the workflow's **file name**, from Task 13's `workflowName(job)`. Until Task 13 lands, use `job.workflow_name ?? job.workflow`. `secondary` is the JSON id when it differs (the ★ row).
  - The editors' h1 is the document name, or `New workflow` / `New prompt`. Their `back` is the section they came from, which replaces `nav.crumbs` and `← prompts`.
- Removed: the 7 local `.head` blocks (`ServerPage:337`, `SchemaPage:63`, `WorkflowPage:411`, `EditorPage:484`, `PromptEditorPage:374`, `ModelsPage:387`, `OverviewPage:449`), with their off-token gaps.

- [ ] **Step 1: Write the failing tests.** In each page test: `sets the tab title to <title>`. For example, `EditorPage.test.ts` checks that an open `e2e-card` gives `document.title === 'e2e-card · diffusers-workflow'`, and `JobPage.test.ts` checks that the h1 is the file name and the JSON id is shown as secondary.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/ src/lib/job/JobHeader.test.ts`. Expected: FAIL on the new cases.
- [ ] **Step 3: Implement.** `EditorPage` ends below 591 and `PromptEditorPage` below 542.
- [ ] **Step 4: Run** the same set, `npx playwright test e2e/journeys/editor-safety.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(frame): every page has one PageHeader`.

### Task 5: The shell's keyboard and screen-reader basics, and the header meter

XC-10 (skip link, drawer focus, dialog names) and SH-13 (no state word on the meter; icon-only controls).

**Files:**
- Modify: `ui/src/App.svelte`, `ui/src/lib/ui/ConfirmDialog.svelte`, `ui/src/lib/pages/OverviewPage.svelte` (the icon-only delete becomes a labelled button; it is converted to `action()` in Task 7)
- Test: `ui/src/lib/ui/ConfirmDialog.test.ts`, `ui/e2e/chrome.spec.ts`

**Interfaces:**
- `App.svelte`:
  - The skip link `<a class="skiplink" href="#main">Skip to content</a>` comes first in `.shell`. It is visible on focus only, styled in `App.svelte`'s own `<style>`. `<main id="main" tabindex="-1">`.
  - Opening the drawer focuses the sidebar's first link. Closing it (scrim, Escape or a navigation) returns focus to the menu button.
  - The meter button shows the state word (`running` / `idle`) beside the bars, and its `aria-label` is `${statusLabel}, ${meterText(metered)}`.
- `ConfirmDialog` renders `AlertDialog.Title` with `confirmState.title` as a visible heading, and the content is labelled by it (`aria-labelledby`), not by `aria-label`.

- [ ] **Step 1: Write the failing tests:**
  - `ConfirmDialog.test.ts`: `the dialog is named by its visible title`.
  - `chrome.spec.ts`:
    - `the first Tab reaches Skip to content, which moves focus to main`
    - `opening the drawer at 390px focuses its first link, and Escape returns focus to the menu button`
    - `the meter says idle or running`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/ui/ConfirmDialog.test.ts && npx playwright test e2e/chrome.spec.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `App.svelte` stays under 600. If it passes 560, move the theme cycle into `src/lib/theme.ts` (`cycleTheme()`, `initialTheme()`).
- [ ] **Step 4: Run** the same commands and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(shell): skip link, drawer focus, named dialogs and a meter that says what it shows`.

### Task 6: `Details`, and the inline errors that still show raw engine text

The *Disclosure of engine text* row: a one-line summary, the engine detail behind it, and a copy button.

**Files:**
- Create: `ui/src/lib/ui/Details.svelte`, `ui/src/lib/ui/ErrorLine.svelte`
- Modify:
  - `ui/src/lib/ui/LoadFailed.svelte` (its detail goes behind `Details`)
  - `ui/src/lib/CachesPanel.svelte` (`clearError`)
  - `ui/src/lib/ui/NoteEditor.svelte` (`error`; this overlaps #1597)
  - `ui/src/lib/job/EventLog.svelte`
  - `ui/src/lib/job/RerunWithChanges.svelte` (`error`)
  - `ui/src/lib/job/ChainSegments.svelte` (`continueEstimate`'s message)
  - `ui/src/lib/request/PlanItems.svelte` and `RequestMenu.svelte`: their `notify.error(\`Not saved: ${message}\`)` toasts become `notify.failure`, keeping the `stale_revision` copy
- Test: `ui/src/lib/ui/Details.test.ts`, `ui/src/lib/ui/ErrorLine.test.ts`, and the touched components' tests
- Docs: ARCHITECTURE: a new row *Engine detail*, and the *Error copy* row loses its "still show the raw message" sentence

**Interfaces:**
- `Details.svelte` props: `{ summary: string; detail: string; open?: boolean }`. It is a `<details>` whose `<summary>` is the one line, with the detail in a mono `<pre>` and a `CopyButton` (`copyText`). It renders nothing when `detail` is empty.
- `ErrorLine.svelte` props: `{ error: unknown; doing?: string }`.
  - It shows `describeError(error)`'s title (`Couldn't <doing>: <title>` when `doing` is given), then the remedy on its own line, then `<Details summary="Engine detail" detail={described.detail}>`.
  - It has `role="alert"`. It replaces each `e instanceof Error ? e.message : String(e)` shown inline in the files above.

- [ ] **Step 1: Write the failing tests:**
  - `Details.test.ts`: `shows the summary, hides the detail until opened, copies it`
  - `ErrorLine.test.ts`: `leads with the described title and keeps the engine text behind Details`
  - `CachesPanel.test.ts`: `a refused clear shows the server's reason in one line, the detail folded`
  - `RequestMenu.test.ts`: `a refused state change is a failure toast, not the raw message`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/ui/Details.test.ts src/lib/ui/ErrorLine.test.ts src/lib/CachesPanel.test.ts src/lib/request/`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, then `grep -rnE "instanceof Error \? \w+\.message" src --include=*.svelte` (expected: none in the files listed), then the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(errors): engine detail sits behind Details; inline errors lead with plain words`.

### Task 7: Destroying a user file asks one way (P5, row 2), with the workspace delete and the Manage panel

P5's "destroys a user file" row, SH-10 (the delete-workspace wording, and a button not styled as danger) and SH-11 (the Manage panel is all disabled controls).

**Files:**
- Create: `ui/src/lib/destructive.ts`
- Modify:
  - `ui/src/lib/gallery/deletion.ts` (it builds its messages through `destroyConfirm`)
  - `ui/src/lib/workspaceActions.ts`
  - `ui/src/lib/pages/OverviewPage.svelte` (the Manage panel)
  - `ui/src/lib/pages/GalleryPage.svelte` (single delete)
  - `ui/src/lib/pages/AssetsPage.svelte` (single and bulk delete)
  - `ui/src/lib/pages/PromptEditorPage.svelte` and `WorkflowPage.svelte` (delete)
  - `ui/src/lib/request/RequestMenu.svelte` (delete a request: the ledger record)
- Modify: `ui/eslint.config.js`, `ui/scripts/lint-rules.test.ts`
- Test: `ui/src/lib/destructive.test.ts`, `ui/src/lib/workspaceActions.test.ts`, `ui/src/lib/gallery/deletion.test.ts`, `ui/src/lib/pages/OverviewPage.test.ts`, `GalleryPage.test.ts`, `AssetsPage.test.ts`
- Docs: ARCHITECTURE, a new UI row *Destructive actions*, naming `destructive.ts`, `action()`'s `confirm` and `undo`, and the ESLint rule

**Interfaces:**
- Consumes: `action(fn, { doing, confirm })` and `ConfirmSpec` (`src/lib/action.svelte.ts`).
- Produces, in `destructive.ts`:
  - `destroyConfirm(subject: { name?: string; count?: number; noun: string; plural?: string; starred?: number; consequence?: string }): ConfirmSpec`. It builds the copy in Global constraints, with `danger: true` and `confirmLabel: 'Delete'`. A starred `name` reads `It is starred.`
  - `workspaceConfirm(name: string, contents: Record<string, { files: number; bytes: number }>, starred: number): ConfirmSpec` uses the workspace copy, with `confirmLabel: 'Delete workspace'` and `danger: true`.
  - `serverConfirm` and `undoToast` are added by Tasks 8 and 9.
- `deleteWorkspaceWithConfirm(name)` keeps its signature.
  - It reads `contents` off the unacknowledged 409's payload (`errorPayload`), not its message.
  - It counts starred outputs with `api.gallery()`, since the workspace is the current one, so the server is unchanged.
  - It asks through `workspaceConfirm`.
- Every delete above runs as `action(del, { doing, confirm: () => destroyConfirm(…) })`, through `ActionButton` or `.run`. This burns down `unguarded_mutations` by 7: AssetsPage 248 and 299, GalleryPage 154 and 207, PromptEditorPage 243, WorkflowPage 189, RequestMenu 56.
- The Overview Manage panel:
  - On `default`, it shows the SH-11 sentence, with no disabled buttons.
  - Elsewhere, it shows **Hide from sidebar** / **Show in sidebar**, and **Delete workspace** with an icon, a label and the `danger` style.
- The ESLint rule: `no-restricted-syntax` on `Property[key.name='confirmLabel'][value.value=/^(Delete|Remove)/]` everywhere in `src/` except `src/lib/destructive.ts`. Its message: `A delete asks through destroyConfirm() (src/lib/destructive.ts)`.

- [ ] **Step 1: Write the failing tests:**
  - `destructive.test.ts`:
    - `one file, one starred, several`
    - `the workspace confirm counts what it removes, starred included`
    - `a folder with no files is left out`
  - `workspaceActions.test.ts`: `asks with the counts from the server's refusal and deletes only on yes`
  - `deletion.test.ts` keeps its cases, and they now expect `danger`.
  - `GalleryPage.test.ts`: `a starred file's confirm says so`, and `a double click deletes once`.
  - `OverviewPage.test.ts`: `the default workspace says why it can't be managed, with no disabled buttons`.
  - `lint-rules.test.ts`: `a Delete confirm outside destructive.ts fails lint`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/destructive.test.ts src/lib/workspaceActions.test.ts src/lib/gallery/deletion.test.ts src/lib/pages/OverviewPage.test.ts src/lib/pages/GalleryPage.test.ts src/lib/pages/AssetsPage.test.ts scripts/lint-rules.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `GalleryPage` and `AssetsPage` must not grow: the bulk delete's confirm text moves into `deletion.ts` and `destructive.ts`.
- [ ] **Step 4: Run** the tests, `npm run lint`, and the metrics gate. Expected: PASS, with `unguarded_mutations` at 18. Lower it in `baseline.json`.
- [ ] **Step 5: Commit** `ui(policy): every delete of a user file names it, counts it and asks in danger`.

### Task 8: Changing the server asks what it costs (P5, row 3): models and caches

P5's "spends GPU time or changes the server" row, decision 4 (Update from GitHub; the SH-S1 confirm lands here), and survey §21's `DiskMeter` sites, which stay for Task 17.

**Files:**
- Modify: `ui/src/lib/destructive.ts` (`serverConfirm`), `ui/src/lib/pages/ModelsPage.svelte`, `ui/src/lib/CachesPanel.svelte`
- Test: `ui/src/lib/destructive.test.ts`, `ui/src/lib/pages/ModelsPage.test.ts`, `ui/src/lib/CachesPanel.test.ts`

**Interfaces:**
- `serverConfirm(spec: { title: string; message: string; confirmLabel: string }): ConfirmSpec` returns `danger: false`.
- In `ModelsPage`:
  - **Update diffusers** asks with the decision-4 copy.
  - **Delete model** asks with the model copy.
  - Starting and cancelling a download are actions with no confirm.
  - The `deleting` / `updating` locals are replaced by each action's `busy`. If a helper is needed, it lives in `src/lib/models.ts`.
- In `CachesPanel`, **Clear** and **Delete** run through `action` with `serverConfirm`. The message is today's wording, with `cost_of_clearing` as the second sentence. The root path stays out of the sentence and goes into `title`. A refusal is reported by Task 6's `ErrorLine`.
- This burns down `unguarded_mutations` by 6: ModelsPage 59, 70, 80 and 103, CachesPanel 43 and 78.

- [ ] **Step 1: Write the failing tests:**
  - `ModelsPage.test.ts`:
    - `Update diffusers asks first and runs only on yes`
    - `a second click while starting a download sends one request`
    - `deleting a model asks with its size`
  - `CachesPanel.test.ts`: `clear asks with the cost and runs once`
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/ModelsPage.test.ts src/lib/CachesPanel.test.ts src/lib/destructive.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `ModelsPage` ends below 565.
- [ ] **Step 4: Run** the tests and the metrics gate. Expected: PASS, with `unguarded_mutations` at 12. Lower the baseline.
- [ ] **Step 5: Commit** `ui(policy): server-changing actions say what they change; Update diffusers asks first`.

### Task 9: Reversible actions act at once and offer Undo (P5, row 1)

P5's reversible row, WF-18 (no undo for removing a step or variable) and RQ-6 (Mark done and Abandon have no confirm or undo). Accept and drop already undo (`PlanItems.undoable`), and stay as they are.

**Files:**
- Modify:
  - `ui/src/lib/destructive.ts` (`undoToast`)
  - `ui/src/lib/pages/GalleryPage.svelte` (`toggleStar`)
  - `ui/src/lib/editor/StepList.svelte` (`removeStep`)
  - `ui/src/lib/editor/VariablesForm.svelte` (`removeVariable`)
  - `ui/src/lib/request/RequestMenu.svelte` (`setState`)
- Test: `ui/src/lib/pages/GalleryPage.test.ts`, `ui/src/lib/editor/StepList.test.ts`, `ui/src/lib/editor/VariablesForm.test.ts`, `ui/src/lib/request/RequestMenu.test.ts`

**Interfaces:**
- `undoToast(message: string, revert: () => void | Promise<unknown>): void` calls `notify.undo(message, …)`. A rejected revert reports through `notify.failure('undo that', e)`.
- **Star:** `action(setStarred, { doing: 'star that file', undo: () => ({ message: 'Starred <leaf>' | 'Unstarred <leaf>', revert: () => api.setStarred(name, !next) }) })`. It is adopted locally as today. This burns GalleryPage 118.
- **Remove a step:** it snapshots `$state.snapshot(step)` and its index, removes it, and `undoToast('Removed step <name>', …)` re-inserts the snapshot at `min(index, steps.length)`, renaming it with the existing `-2` rule if the name is taken. The step's mode goes back to `full`.
- **Remove a variable:** it snapshots the key, the value and the key order. Undo puts it back at its place. A key that has been re-added meanwhile is left as it is, and the toast says `<name> is already back`.
- **Mark done and Abandon:** they push the state, then `undoToast(\`Marked "<title>" done\` | \`Abandoned "<title>"\`)`, whose revert pushes the previous state. A refused revert, `stale_revision` included, goes through the store's existing conflict path. Reopen and Delete are unchanged; Delete was Task 7's.

- [ ] **Step 1: Write the failing tests:**
  - `GalleryPage.test.ts`:
    - `starring offers Undo, which unstars`
    - `an undo that fails is reported, not thrown`
  - `StepList.test.ts`: `removing a step offers Undo, and the step comes back where it was after another was added`.
  - `VariablesForm.test.ts`: `removing a variable offers Undo; a key added again meanwhile is kept`.
  - `RequestMenu.test.ts`: `Mark done acts at once and Undo reopens`, `Abandon offers Undo`, and `an Undo refused as stale goes through the store's conflict path`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/pages/GalleryPage.test.ts src/lib/editor/StepList.test.ts src/lib/editor/VariablesForm.test.ts src/lib/request/RequestMenu.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and the metrics gate. Expected: PASS, with `unguarded_mutations` at 11. Lower the baseline.
- [ ] **Step 5: Commit** `ui(policy): star, remove a step or variable, mark done and abandon act at once with Undo`.

### Task 10: `nameDialog`, and Save as asset with Replace or Keep both

D11 (survey §11), the ★ row *Save as asset*, P5's overwrite row for a saved asset, and `window_dialogs` → 0. GA-10's reuse path ("Copy reference", "Use as input") stays in phase 4.

**Files:**
- Modify:
  - `ui/src/lib/confirm.svelte.ts` (`nameDialog`, and `confirmState.input`)
  - `ui/src/lib/ui/ConfirmDialog.svelte` (renders the field)
  - `ui/src/lib/gallery/GalleryDetail.svelte` (`keepAsAsset` becomes `saveAsAsset`)
- Create: `ui/src/lib/gallery/saveAsAsset.ts`, so `GalleryDetail` loses lines
- Test: `ui/src/lib/confirm.svelte.test.ts`, `ui/src/lib/ui/ConfirmDialog.test.ts`, `ui/src/lib/gallery/saveAsAsset.test.ts`, `ui/src/lib/gallery/GalleryDetail.test.ts`
- Docs: the ARCHITECTURE row *UI overlays and suggestions* names `nameDialog`

**Interfaces:**
- `nameDialog(question: string, opts: { title: string; label: string; initial?: string; confirmLabel?: string; validate: (name: string) => string | null }): Promise<string | null>`
  - It queues like `chooseDialog`, the same FIFO.
  - **Save** stays disabled while `validate` returns a message, and the message is shown under the field. Enter submits only when the name is valid.
  - Escape or Cancel answers `null`.
- `saveAsAsset(file: GalleryFile, existing: string[]): Promise<{ name: string } | null>` in `saveAsAsset.ts`:
  - It asks for the name through `nameDialog`, validated with `isNameSegment`.
  - It calls `api.keepOutput(file.name, name)`. On 409 `exists`, it asks with `chooseDialog` (the Global constraints copy).
    - **Replace** calls `keepOutput(…, true)`.
    - **Keep both** calls `keepOutput(…, freeName(name, existing))`.
    - **Cancel** returns `null`, with no toast.
  - `GalleryDetail` runs it as `action(saveAsAsset, { doing: 'save it as an asset', success: (r) => \`Saved as asset ${r.name}\` })`.
- The button's label is **Save as asset**, and its title is `save this file to the asset library, under a name later workflows can use`.
- This burns down `unguarded_mutations` by 2 (GalleryDetail 101 and 114) and `window_dialogs` by 1.

- [ ] **Step 1: Write the failing tests:**
  - `confirm.svelte.test.ts`:
    - `nameDialog answers the typed name`
    - `an invalid name keeps Save disabled and says why`
    - `Escape answers null`
    - `a nameDialog opened while a confirm is up waits its turn`
  - `saveAsAsset.test.ts`:
    - `an existing name asks Replace, Keep both or Cancel`
    - `Keep both saves under the free name`
    - `Cancel at either step makes no request and no toast`
  - `GalleryDetail.test.ts`: `offers Save as asset, never window.prompt`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/confirm.svelte.test.ts src/lib/ui/ConfirmDialog.test.ts src/lib/gallery/`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and the metrics gate. Expected: PASS, with `window_dialogs` at 0, `unguarded_mutations` at 9, and `banned_terms` down by the 5 `keep-asset` hits (to 57). Write all three into the baseline.
- [ ] **Step 5: Commit** `ui(dialogs): nameDialog replaces window.prompt; Save as asset asks Replace or Keep both`.

### Task 11: Every remaining mutating button is an `action()`

P4 is closed, and `unguarded_mutations` → 0.

**Files:**
- Modify:
  - `ui/src/lib/job/RerunWithChanges.svelte` (`queue`; its `submitting` flag goes)
  - `ui/src/lib/pages/EditorPage.svelte` and `WorkflowPage.svelte` (Run's `submitJob`, after the priced confirm they already have)
  - `ui/src/lib/pages/JobsPage.svelte` (`moveJob`)
  - `ui/src/lib/request/NoteList.svelte` (add, edit, delete)
  - `ui/src/lib/pages/GalleryPage.svelte` and `AssetsPage.svelte` (`downloadPicked`, the archive)
- Test: the existing test files of each, plus `ui/src/lib/request/NoteList.test.ts`
- Docs: the ARCHITECTURE row *Actions* drops "still plain handlers until phase 3"

**Interfaces:**
- Consumes: `action`, `ActionButton`.
- Each keeps the user-visible behaviour it has, and gains the busy guard and `notify.failure` copy.
  - `RerunWithChanges` keeps its inline refused-value message (Task 6's `ErrorLine`) by catching inside `fn` and returning `null`.
  - The two archive downloads move into one `downloadArchive(names, archive)` in `src/lib/picks.svelte.ts`, which is survey §19's hash-matched pair.

- [ ] **Step 1: Write the failing tests.** Per file, `a double click sends one request`. They are `RerunWithChanges.test.ts`, `EditorPage.test.ts` (Run), `WorkflowPage.test.ts` (Run), `JobsPage.test.ts` (move), `NoteList.test.ts` (add) and `GalleryPage.test.ts` (download).
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/job/RerunWithChanges.test.ts src/lib/pages/ src/lib/request/NoteList.test.ts`. Expected: FAIL on the new cases.
- [ ] **Step 3: Implement.** `EditorPage` must not grow: Run's `action` goes in `src/lib/editor/workflowFile.ts` beside the save.
- [ ] **Step 4: Run** the tests and the metrics gate. Expected: PASS, with `unguarded_mutations` at 0. Write it into the baseline.
- [ ] **Step 5: Commit** `ui(actions): every button that changes server state is an action`.

### Task 12: Durations and cost wording

WF-15 (cost wording; the `~` glyph), RQ-12 (the `~` glyph), and the *about 3 min* row. #1599 will build on this format. JB-14's raw-seconds ETA is fixed here, as a side effect of the format.

**Files:**
- Modify:
  - `ui/src/lib/format.ts` (`formatMinutes`, and a new `formatApprox`)
  - `ui/src/lib/plan.ts` (lines 92, 153, 166, 172 and 235)
  - `ui/src/lib/cost.ts` (the comment and `costFigure`)
  - `ui/src/lib/job/JobProgress.svelte:50`
- Test: `ui/src/lib/format.test.ts`, `ui/src/lib/plan.test.ts`, `ui/src/lib/cost.test.ts`, `ui/src/lib/job/JobProgress.test.ts`

**Interfaces:**
- `formatMinutes(minutes: number): string` returns `under 1 min`, or `about <n> min` with up to 2 decimals.
- `formatApprox(seconds: number): string` returns `under 1 min`, `about <m> min` with whole minutes, or `about <h> h <m> min` from an hour up.
- `plan.ts`'s lines use `formatMinutes`. Their wording stays apart from `~` → `about`, so `plan.test.ts`'s expectations change by that token only. That keeps this task a format change, not WF-8/WF-21's run-check rewrite (phase 5).
- `JobProgress` shows `about <formatApprox(eta)> left`.

- [ ] **Step 1: Write the failing tests.** Update the expectations in the four files to the Global constraints copy, and add `formatApprox crosses an hour` (`4800` → `about 1 h 20 min`).
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/format.test.ts src/lib/plan.test.ts src/lib/cost.test.ts src/lib/job/JobProgress.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, `npx vitest run src/lib/`, and the metrics gate. Expected: PASS, with `banned_terms` down by the 7 `tilde` hits (to 50). Lower the baseline.
- [ ] **Step 5: Commit** `ui(copy): durations read "about 3 min", never a tilde`.

### Task 13: One name for a job, and the workflow's file name everywhere

The ★ rows *job* and *Workflow name*, XC-13 (the finish toast names the JSON id; it already has **Open**), and *v2* (its tooltip).

**Files:**
- Modify:
  - `ui/src/lib/terms.ts` (`workflowName`, `ordinal`)
  - `ui/src/lib/jobWatch.ts` (`leaf`), `job/JobHeader.svelte`
  - `pages/JobsPage.svelte:129,289`, `pages/OverviewPage.svelte:218,227`, `assets/AssetDetail.svelte:43`
  - the version badges: Gallery, Compare and the gallery detail (`v{version}`)
  - the 36 `run-noun` sites in the `banned_terms` detail, which are all of them except the two `Run note` labels (Task 14's): FlowLegend, JobHeader, RerunWithChanges, JobResults, ChainSegments, MediaPicker, MediaArgumentInput, ValidationPanel, RunCheckPanel, `runCheck.svelte.ts`, `plan.ts:44`, `chainActions.ts`, `refcomplete.svelte.ts`, RequestFilter, GalleryPage, ComparePage, JobPage and WorkflowPage
- Test: `ui/src/lib/terms.test.ts`, `ui/src/lib/jobWatch.test.ts`, `ui/src/lib/job/JobHeader.test.ts`, `ui/src/lib/pages/JobsPage.test.ts`, and whichever existing tests assert the old copy
- Docs: ARCHITECTURE, the *Run-version surfaces* row's UI cell names the tooltip

**Interfaces:**
- `workflowName(job: { workflow_name?: string | null; workflow?: string | null }): string` returns the last path segment of `workflow_name`, else of `workflow`, else `'job'`.
  - `workflow_name` is the catalog name the job was run by, which is the file. `workflow` is the JSON id.
  - Every place that labels a job uses it: the tab title, the finish toast and its alert, the Jobs rows and their name filter, the Overview's job list, AssetDetail's provenance and the JobHeader h1.
- `ordinal(n: number): string` returns `1st`, `2nd`, `3rd`, `4th`, `11th`, `12th`, `13th`, `21st`.
- Every `v{n}` badge has `title={\`${ordinal(n)} job of this workflow in this folder\`}`.
- A run noun becomes `job` in each site. Inside a request (RequestFilter: `show only the jobs made for one request`), "the runs" becomes "the jobs". The verb "run" stays.

- [ ] **Step 1: Write the failing tests:**
  - `terms.test.ts`: `workflowName prefers the file name`, and the `ordinal` cases above.
  - `jobWatch.test.ts`: `finishedText names the file, not the JSON id`. For `{ workflow: 'E2ECard', workflow_name: 'e2e-card' }`, it starts `e2e-card`.
  - `JobsPage.test.ts`: `rows and the name filter use the file name`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/terms.test.ts src/lib/jobWatch.test.ts src/lib/job/ src/lib/pages/JobsPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Then run `node scripts/arch-metrics.mjs | jq '.detail.banned_terms'` until no `run-noun` hit is left.
- [ ] **Step 4: Run** `npx vitest run src/lib/` and the metrics gate. Expected: PASS, with `banned_terms` down by 36 (to 14). Lower the baseline.
- [ ] **Step 5: Commit** `ui(copy): one execution is a job, named by its workflow's file`.

### Task 14: The words for notes, takes, holds and examples

The ★ rows *Save as asset* (copy left over after Task 10), *Accept* (the words only), *Star*, *Notes*, *Waiting for your OK*, *Examples* and *GPU/card*. This finishes `banned_terms`: its last 14 hits, the two `Run note` labels among them.

**Files:**
- Modify:
  - `ui/src/lib/request/reviewText.ts`, `noteState.ts:58,60`, `runPlan.ts:98`, `AttemptStrip.svelte:43`
  - `ui/src/lib/gallery/GalleryDetail.svelte` (the star title at line 253, and the note labels at 294 and 304)
  - `ui/src/lib/pages/JobPage.svelte:326`, `OverviewPage.svelte` (the workspace note's label), `assets/AssetDetail.svelte` (the asset note's label)
  - `ui/src/lib/plan.ts:90,153` (`inherited from the template` becomes `inherited from <name>`, or `from an example` when there is none)
  - `ui/src/lib/CardMemoryList.svelte` (the hardware line says `GPU`)
  - any queue text that says `GPU` where it means a card (check with `grep -rn "GPU" src --include=*.svelte`)
- Test: `ui/src/lib/request/reviewText.test.ts`, `noteState.test.ts`, `runPlan.test.ts`, `AttemptStrip.test.ts`, `ui/src/lib/gallery/GalleryDetail.test.ts`, `ui/src/lib/pages/JobPage.test.ts`

**Interfaces:**
- `bannerText(group, listening, left)` keeps its signature and returns the hold copy in Global constraints. `REDIRECTED` and `runNowKeys` are unchanged; the RQ-1 rule is phase 6's.
- `NoteEditor`'s `label` props at each call site become `Your note on this <job|file|asset|workspace>`. The star's title becomes `star this file - press S`.

- [ ] **Step 1: Write the failing tests.** Update the expectations in the files above to the Global constraints copy.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/request/ src/lib/gallery/GalleryDetail.test.ts src/lib/pages/JobPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npx vitest run src/lib/`, `npx playwright test e2e/requests.spec.ts`, and the metrics gate. Expected: PASS, with `banned_terms` at 0. Write it into the baseline.
- [ ] **Step 5: Commit** `ui(copy): notes, holds, takes and examples use the glossary's words`.

### Task 15: `Segmented` and `.togglechip`

D4 (survey §4) and XC-10's `aria-pressed`.

**Files:**
- Create: `ui/src/lib/ui/Segmented.svelte`
- Delete: `ui/src/lib/editor/ViewSwitch.svelte`. `EditorPage` and `PromptEditorPage` use `Segmented`; until Task 20, the `EDITOR_VIEWS` list stays where it is.
- Modify:
  - `ui/src/lib/editor/StepEditor.svelte:205-222` and its CSS at 460-480, `.modeswitch`
  - `ui/src/lib/pages/WorkflowsPage.svelte:211-246` (the shape row is `Segmented`, the trait chips `.togglechip`)
  - `ui/src/lib/pages/GalleryPage.svelte:323-345` (`.rolechip` and Starred become `.togglechip`)
  - `ui/src/lib/assets/MediaPicker.svelte:162-179` (the tabs are `Segmented`)
  - `ui/src/lib/pages/PromptsPage.svelte:121-147` (the tag and model chips are `.togglechip`)
  - Create `ui/src/patterns.css`, imported in `ui/src/main.ts` after `frames.css`, holding `button.togglechip` and `.on`. Its header comment says it is `app.css`'s overflow, and why. Task 16 adds the rest.
  - `ui/scripts/design-rules.test.ts`
- Test: `ui/src/lib/ui/Segmented.test.ts`, `ui/scripts/design-rules.test.ts`
- Docs: ARCHITECTURE, a new UI row *Toggles*

**Interfaces:**
- `Segmented` props: `{ options: { value: string; label: string; title?: string; icon?: Component<{ size?: number }> }[]; value: string; onselect: (value: string) => void; label: string; size?: 'sm' | 'md' }`.
  - It renders `role="group"` with `aria-label={label}`, and one button per option with `aria-pressed`.
  - The selected style is one look, the accent outline. Survey §4 found two (accent outline and ink fill); this plan decides on the outline.
- `button.togglechip` is the one multi-select chip, the pill shape, with `aria-pressed`. `.on` is the ink fill.
- The design rule `the segmented and chip selected styles live only in Segmented.svelte and patterns.css`: no other `.svelte` `<style>` declares `aria-pressed` or `.activebtn` selectors.

- [ ] **Step 1: Write the failing tests:**
  - `Segmented.test.ts`: `marks the chosen option pressed and reports a pick`, and `is a named group`
  - `design-rules.test.ts`: the rule above
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/ui/Segmented.test.ts scripts/design-rules.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npx vitest run src/lib/`, `npx playwright test e2e/smoke.spec.ts`, and the metrics gate. Expected: PASS, with `duplicate_blocks` lower than before.
- [ ] **Step 5: Commit** `ui(toggles): one segmented switch and one filter chip, each telling screen readers its state`.

### Task 16: The shared classes, and a token rule that catches `--danger`

D3 (survey §3, `.formgrid`) and D15's CSS parts: §16 `table.datatable`, §17 the tone classes and the missing token, §18 `.panel > h2`, and §22 list rows. Also D16's `.frame` bypass (`.cardframe`).

**Files:**
- Modify: `ui/src/patterns.css` (Task 15 created it), the home of every class below.
- Modify:
  - the 9 form grids of §3
  - the 6 tables of §16
  - the `.error`, `.warn`, `.bad` and `.hint` copies of §17 (about 20 files)
  - `ModelsPage.svelte:423` and `VariablesForm.svelte:301` (`--danger` becomes `--bad` and `--warn` respectively, per §17)
  - the 4 `var(--sunk)` uses (`Suggest`, `AssetDetail`, `MediaPicker`, `AssetsPage`), which become `--panel-2`
  - the 6 panel headings of §18
  - JobsPage and RequestsPage's rows (§22; both collapse on `@media`)
  - `WorkflowsPage` and `OverviewPage`'s `.cardframe`, which become `.frame`
  - `ui/scripts/design-rules.test.ts`, whose `LIVE_FILES` gains `src/patterns.css` only if it uses `--live`
- Test: `ui/scripts/design-rules.test.ts`
- Docs: the `app.css` header comment names `patterns.css`. ARCHITECTURE gets a new UI row *Shared classes*.

**Interfaces:**
- `.formgrid`: `grid-template-columns: var(--label-w, minmax(140px, 40%)) 1fr`. Children `> label` share one style (600, muted). `.formgrid.withaction` adds a trailing `auto` track. One `@container (max-width: 400px)` rule collapses it to one column.
- `table.datatable` inside `.tablewrap`: `th` at `var(--t-xs)`, uppercase-free, with a border-bottom row rule. `.dense` makes it mono, for EventLog.
- `.error`, `.warn`, `.bad` and `.good` are colour only. `.hint` is muted, at `var(--t-sm)`.
- `.panel > h2` is flex, with its items centred and a gap of `var(--space-2)`.
- `.panel.list > .row` holds the shared row padding, radius and hover.
- New design rules:
  - `every var(--x) names a defined custom property`: one defined by `--x:` anywhere in `src`, or set with `style:--x`, or a `--bits-*` name.
  - `app.css stays at or under 600 lines`.

- [ ] **Step 1: Write the failing tests:** the two design rules. The first fails today on `--danger` and `--sunk`.
- [ ] **Step 2: Run** `cd ui && npx vitest run scripts/design-rules.test.ts`. Expected: FAIL, naming `--danger` and `--sunk`.
- [ ] **Step 3: Implement**, file by file, deleting each local copy as the global class takes over.
- [ ] **Step 4: Run** `npx vitest run`, `npm run ux:capture`, and a look at the Models, Server, Editor and Jobs captures in both themes. Then the metrics gate. Expected: PASS, with `duplicate_blocks` lower and no page larger than before.
- [ ] **Step 5: Commit** `ui(css): one form grid, table, tone and panel-heading rule; every token reference resolves`.

### Task 17: `StatusChip`, `DiskMeter`, `GroupHeading`, `jobHref`

D10 (survey §10), D15's §20 and §21, and D16's leftovers: `.row-live` on the Overview, `gbFromMb` in WorkerList, and §23's legacy `#/jobs/<id>` links.

**Files:**
- Create: `ui/src/lib/ui/StatusChip.svelte`, `ui/src/lib/ui/DiskMeter.svelte`, `ui/src/lib/ui/GroupHeading.svelte`, `ui/src/lib/collapsed.svelte.ts`
- Modify:
  - the status chips at `job/JobHeader.svelte:146`, `pages/JobsPage.svelte:308`, `pages/OverviewPage.svelte:325,334`, `request/AttemptStrip.svelte:31-37,80` (its `cancelled → idle` map goes) and `request/RunDialog.svelte:193`
  - the disk meters at `ModelsPage.svelte:176-183`, with its CSS at 439-452, and `CachesPanel.svelte:143-153`
  - the group headings in `FolderGroups.svelte`, `AssetsPage.svelte:61-65,151-158`, `assets/LibraryRow.svelte`, `assets/ShadowedAssets.svelte:24-28,52-63` and `MediaPicker.svelte:207`
  - `WorkerList.svelte:29,33`, `GalleryDetail.svelte:345`, `editor/EnhancePanel.svelte:93`, `App.svelte:257`, `OverviewPage.svelte:514` (`.row-live`)
  - `ui/src/lib/routes.ts` (`jobHref`)
- Test: `ui/src/lib/ui/StatusChip.test.ts`, `ui/src/lib/ui/DiskMeter.test.ts`, `ui/src/lib/ui/GroupHeading.test.ts`, `ui/src/lib/routes.test.ts`, `ui/scripts/design-rules.test.ts`
- Docs: ARCHITECTURE, a new UI row *Status and grouping*

**Interfaces:**
- `StatusChip` props: `{ status: JobStatus | string; label?: string }`. It owns the status-to-tone map (`.chip.<status>`). A request *item* chip (`itemState.ts`, `StateTag`) stays separate by design.
- `DiskMeter` props: `{ freeBytes: number; totalBytes?: number }`. It reads `<x> / <y> GB free` when the total is known, else `<x> GB free`.
- `collapsedState(key: string): { has(name: string): boolean; toggle(name: string): void }` is storage-backed, and replaces FolderGroups' and AssetsPage's own copies.
- `GroupHeading` props: `{ label: string; count?: number; collapsed: boolean; ontoggle: () => void; children?: Snippet }`. It is a button with `aria-expanded`.
- `jobHref(id: string, workspace?: string): string` returns `wsHref(workspace, 'jobs', id)` when it has a workspace. Without one it returns the legacy `#/jobs/<id>`, kept only where the caller has no workspace: `App.svelte`'s running link, since health carries none.
- A design rule: `job status colours are declared only in StatusChip.svelte and app.css`. No other `<style>` has a `.chip.(running|queued|succeeded|failed|cancelled)` selector.

- [ ] **Step 1: Write the failing tests:**
  - `StatusChip.test.ts`: `a cancelled job has one look everywhere`, which renders `cancelled` and expects `.chip.cancelled`
  - `GroupHeading.test.ts`: `tells screen readers whether it is open`
  - `DiskMeter.test.ts`: both texts
  - `routes.test.ts`: `jobHref names the job's workspace when known`
  - `design-rules.test.ts`: the status-colour rule
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/ui/ src/lib/routes.test.ts scripts/design-rules.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npx vitest run`, `npx playwright test e2e/requests.spec.ts e2e/smoke.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(pieces): one status chip, disk meter and group heading; job links name their workspace`.

### Task 18: Filters in the hash, the request filter on Jobs, and scroll on Back

D7 (survey §7), D6 (§6) and XC-11 (Back loses scroll and the gallery filters).

**Files:**
- Modify:
  - `ui/src/lib/router.svelte.ts` (`hashState`, plus scroll memory keyed by the entry's `dwIndex`)
  - `ui/src/App.svelte`: the navigation `$effect`'s `window.scrollTo(0, 0)` becomes the router's restore
  - `ui/src/lib/gallery/RequestFilter.svelte` (the `workspace` and `pickable` props)
  - `pages/JobsPage.svelte:30-64,248-266` (`hashState`, plus `RequestFilter` in place of its own chip)
  - `pages/RequestsPage.svelte:31-37`
  - `pages/GalleryPage.svelte` (text, roles, Starred)
  - `pages/WorkflowsPage.svelte` (shape, traits, text)
  - `pages/PromptsPage.svelte` (text)
  - `pages/AssetsPage.svelte` (text)
- Test: `ui/src/lib/router.svelte.test.ts`, `ui/src/lib/gallery/RequestFilter.test.ts`, `ui/src/lib/pages/JobsPage.test.ts`, `ui/src/lib/pages/GalleryPage.test.ts`
- Docs: ARCHITECTURE, a new UI row *A page's place*

**Interfaces:**
- `hashState<T>(key: string, initial: T, codec?: { parse: (raw: string) => T; format: (value: T) => string }): { value: T }`
  - The box is `$state`-backed. Writing it calls `setHashQuery` with every live box's value, so it merges with the other keys and never drops one.
  - A `hashchange` re-reads it. A hash with no query resets the box to `initial`.
  - The default codec is the string itself. Pages use `codec.list` (comma-joined) and `codec.flag` (`'1'` or absent), both exported.
  - It is created in component init, and cleans up through `$effect`.
- Scroll memory:
  - On every committed navigation, the router saves `scrollY` for the entry it leaves, keyed by `dwIndex`, in a module `Map` capped at 50 entries.
  - On arriving at an entry with a saved position by Back or Forward, it restores it. It retries on each animation frame for up to 1 s while the document is too short, and stops early on a user scroll, wheel, touch or key.
  - A pushed navigation scrolls to the top, as today.
  - Exported for tests: `restoreScroll(index: number, opts?: { frames?: number })`.
- `RequestFilter` gains:
  - `workspace?: string`, the workspace its picker lists. It defaults to `workspace.current`. Jobs-all passes `listingWorkspace`.
  - `pickable = true`
  - It uses `hashState('req', '')` for its value.

- [ ] **Step 1: Write the failing tests:**
  - `router.svelte.test.ts`:
    - `hashState writes its key and keeps the others`
    - `hashState follows Back`
    - `a hash with no query resets every box`
    - `restoreScroll waits for the page to be tall enough, and stops on a user scroll`
  - `RequestFilter.test.ts`: `with a workspace prop, it lists that workspace's requests`.
  - `JobsPage.test.ts`: `the request filter can be picked as well as cleared`.
  - `GalleryPage.test.ts`: `the text, role and Starred filters survive leaving and coming back`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/router.svelte.test.ts src/lib/gallery/RequestFilter.test.ts src/lib/pages/JobsPage.test.ts src/lib/pages/GalleryPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `router.svelte.ts` (267 lines) stays under 600. `JobsPage` ends shorter than 525.
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/workspaces.spec.ts e2e/journeys/agent-edits.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(router): filters live in the hash on every list, and Back restores where you were`.

### Task 19: One note-draft engine for both note editors

D5 (survey §5). **Overlap:** #1597 (note gaps) also touches `NoteEditor.svelte`. This task makes only the draft and key behaviour shared, and plans none of #1597's gaps. Whichever lands second rebases onto the other.

**Files:**
- Create: `ui/src/lib/noteDraft.svelte.ts`
- Modify: `ui/src/lib/ui/NoteEditor.svelte`, `ui/src/lib/request/ItemNote.svelte`, `ui/src/lib/request/draft.ts` (its helpers become the module's storage layer)
- Test: `ui/src/lib/noteDraft.svelte.test.ts`, `ui/src/lib/ui/NoteEditor.test.ts`, `ui/src/lib/request/ItemNote.test.ts`
- Docs: ARCHITECTURE, the *Keeping UI views fresh* row's NoteEditor clause names `noteDraft`

**Interfaces:**
- `class NoteDraft`, constructed with `{ key: string; saved: () => string; save: (text: string) => Promise<void> }`:
  - `text: string` (`$state`), persisted to `sessionStorage` under `key` through `draft.ts`'s `keepDraft` and `readDraft`
  - `atRisk: boolean`, from `draftAtRisk`
  - `keydown(e: KeyboardEvent): void`: Ctrl/Cmd+Enter saves, and Escape closes and keeps the draft
  - `cancel(): Promise<boolean>`: asks `Discard this draft of your note?` with **Discard** (danger) and **Keep editing** only when the draft differs from what was saved
  - `counter: string | null`: `<n> / 4000` within 10% of the limit, else `null`
- Both editors use it. `ItemNote` keeps its plan and intent coupling.

- [ ] **Step 1: Write the failing tests:**
  - `noteDraft.svelte.test.ts`:
    - `a draft survives a reload`
    - `Ctrl+Enter saves`
    - `Escape keeps the draft`
    - `cancel asks only when the draft differs`
  - `NoteEditor.test.ts`: `Ctrl+Enter saves a file note`, and `a half-written file note is there after a remount`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/noteDraft.svelte.test.ts src/lib/ui/NoteEditor.test.ts src/lib/request/ItemNote.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** The `NoteDraft` constructor is the one place a draft key is made: `note:<workspace>:<target>`, and the request keys keep their current form.
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/requests.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(notes): both note editors share one draft, keys and discard rule`.

### Task 20: `DocumentFileBar`, one views list and one save key

D14 (survey §14).

**Files:**
- Create: `ui/src/lib/editor/DocumentFileBar.svelte`, generalised from `WorkflowFileBar.svelte`, which is deleted
- Modify: `ui/src/lib/editorShell.svelte.ts` (`EDITOR_VIEWS`, `editorKeys`), `ui/src/lib/pages/EditorPage.svelte:75-102,251`, `ui/src/lib/pages/PromptEditorPage.svelte:53-72,126,268-292`
- Test: `ui/src/lib/editorShell.test.ts`, `ui/src/lib/editor/DocumentFileBar.test.ts`, `ui/src/lib/pages/EditorPage.test.ts`, `ui/src/lib/pages/PromptEditorPage.test.ts`
- Docs: the ARCHITECTURE row *The editors' document* names `DocumentFileBar` and `editorKeys`

**Interfaces:**
- `export const EDITOR_VIEWS: { value: EditorView; label: string; title: string; icon: Component }[]`. Each editor filters it by the views it has.
- `editorKeys(ed: DocumentEditor, handlers: { save: () => void; run?: () => void }): (e: KeyboardEvent) => void`
  - Ctrl/Cmd+S saves, and so does `'S'` with Caps Lock. Ctrl/Cmd+Enter runs when given.
  - It ignores the keys while `overlayOpen()`.
  - It flushes first, through `ed.flush()`.
- `DocumentFileBar` props: `{ ed: DocumentEditor; dir: string; folders: string[]; reference?: string; extra?: Snippet }`. The workflow editor's workspace and start-from go in `extra`. The prompt editor's reference hint and `CopyButton` go in `reference`.
- Both heads look the same: the "unsaved" chip, then the dirty dot on a **primary** Save.

- [ ] **Step 1: Write the failing tests:**
  - `editorShell.test.ts`:
    - `editorKeys saves on Ctrl+S and on Caps-Lock S`
    - `ignores keys under an overlay`
    - `flushes before saving`
  - `DocumentFileBar.test.ts`: `shows the folder, the name and the path for either document`.
  - `PromptEditorPage.test.ts`: `shows the unsaved chip like the workflow editor`.
- [ ] **Step 2: Run** `cd ui && npx vitest run src/lib/editorShell.test.ts src/lib/editor/DocumentFileBar.test.ts src/lib/pages/EditorPage.test.ts src/lib/pages/PromptEditorPage.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Both pages shrink. `editorShell.svelte.ts` (478 lines) stays under 600.
- [ ] **Step 4: Run** the tests, `npx playwright test e2e/journeys/editor-safety.spec.ts`, and the metrics gate. Expected: PASS.
- [ ] **Step 5: Commit** `ui(editor): one file bar, one views list and one save key for both editors`.

### Task 21: The phase gate, the rules for the next agent, and a journey

The copy ratchet at 0, the rules in `ui/CLAUDE.md`, and the spec's "before you write markup" table.

**Files:**
- Create: `ui/e2e/journeys/consistency.spec.ts`
- Modify: `ui/scripts/copy-rules.test.ts`, `docs/stabilization/ui/baseline.json`, `ui/GLOSSARY.md` (its *Before you write markup* table), `ui/CLAUDE.md` (rewrite lines, add none), `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes everything above.

- [ ] **Step 1: Write the gate and the journey.**
  - `copy-rules.test.ts`, `no copy breaks the glossary`: `(await measure()).detail.banned_terms` equals `[]`, and the failure message lists each `file:line id` with its `use`.
  - `consistency.spec.ts`:
    1. Each sidebar section's tab title and h1 match its sidebar label.
    2. `#/ws/default/nope` shows `Page not found`, and its link leads to the overview.
    3. In the Gallery, type a filter and tick Starred, scroll, open a file, and go Back: the filter, Starred and the scroll position are as they were.
    4. Typing in that filter leaves focus in the field.
    5. Star a file, then Undo: it is unstarred.
    6. Deleting a starred file asks with `It is starred.` and a danger button.
- [ ] **Step 2: Run** `cd ui && npx vitest run scripts/copy-rules.test.ts && npx playwright test e2e/journeys/consistency.spec.ts`. Expected: PASS. If the copy case fails, the failure names the line a task left behind; fix it here.
- [ ] **Step 3: Write the rules.**
  - `GLOSSARY.md` *Before you write markup*: a table of concept → component, one row each for a page head (`PageHeader`), a destructive action (`destroyConfirm` / `serverConfirm` / `undoToast` through `action()`), a mutating button (`action()` / `ActionButton`), engine text (`Details` / `ErrorLine`), a name to type (`nameDialog`), a segmented switch or filter chip (`Segmented` / `.togglechip`), a job status (`StatusChip`), a group heading (`GroupHeading`), a filter that survives Back (`hashState`), a form grid (`.formgrid`), a table (`table.datatable`), tone text (`.error`/`.warn`/`.bad`/`.hint`), a note (`NoteEditor` / `noteDraft`), and a document file bar (`DocumentFileBar`).
  - `ui/CLAUDE.md` stays at 12 lines.
    - Bullet 2's "a new mutating button is an `action()` / `ActionButton` (the job page's are; phase 3 converts the rest)" becomes "every mutating button is an `action()`; a destructive one asks or undoes through `src/lib/destructive.ts` (the P5 table)".
    - Its "(some inline errors still show it, not yet converted)" becomes "inline, `ErrorLine`".
    - Bullet 4 gains "a page's head is `PageHeader`; copy follows `ui/GLOSSARY.md`, whose banned terms `scripts/copy-rules.test.ts` holds at 0, and its *Before you write markup* table names the piece for each concept". That is paid for by cutting bullet 4's "`chooseDialog`/`confirmDialog` (`src/lib/confirm.svelte.ts`), a destructive choice `danger: true`" down to "a dialog is `src/lib/confirm.svelte.ts`'s".
  - The ARCHITECTURE *UI architecture ratchet* row names the four new metrics and `copy-rules.test.ts`.
- [ ] **Step 4: Run** `npm run check`, `npm run lint`, `npm run build`, `npm test`, `npx playwright test`, the metrics gate, `npm run metrics -- --check ../docs/stabilization/ui/baseline.json` (only the two pre-existing lines may print), `venv/bin/python scripts/arch_metrics.py` (`modules` 281, `claude_md_lines` 129) and `venv/bin/python -m pytest tests/test_ui_twins.py -q`. Expected: PASS. `baseline.json` reads `window_dialogs 0`, `unguarded_mutations 0`, `banned_terms 0`, and `duplicate_blocks` at its new measured value, below 48.
- [ ] **Step 5: Commit** `ui(copy): the copy ratchet at zero; rules and the markup table for the next agent`.

---

## Phase exit

- These metrics are 0 in `docs/stabilization/ui/baseline.json`: `window_dialogs`, `unguarded_mutations`, `banned_terms`, `raw_error_toasts`, `empty_outside_loadstate` and `unwatched_reads`.
- `duplicate_blocks` is below 48, and no other metric is above `$SCRATCH/ui-start.json`.
- There is no new Python module, and `claude_md_lines` is 129.
- `copy-rules.test.ts` holds the glossary.
- `design-rules.test.ts` holds the token, segmented and status-colour rules.
- `lint-rules.test.ts` holds the delete-confirm rule.
- `consistency.spec.ts` holds the page frame, Back and the P5 policy.
- Phases 4-7 are written next, against the code as it stands.
