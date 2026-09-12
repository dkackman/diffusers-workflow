# Output Subfolders - Stage 4 (UI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `result.subfolder` visible to a person in the web UI: the gallery filters on it, the job page groups a run's results under `final/` and `intermediate/` headings, and the TypeScript types carry the field the server already sends.

**Architecture:** Stages 1-3 landed the field everywhere the server speaks - every gallery entry, manifest entry and `step_end` event carries `subfolder` (`''` when a step chose none). Stage 4 is read-only consumption of that: the types learn the field (Task 1), `ui/src/lib/results.ts` learns to section step groups by subfolder (Task 2, pure function, vitest), `JobPage.svelte` renders the sections (Task 3), and `GalleryPage.svelte` grows a `<select>` beside its text filter (Task 4). The editor needs no work: its Monaco JSON views read the live `/api/schema`, which has described `result.subfolder` since stage 1. Docs close the stage (Task 5).

**Tech Stack:** Svelte 5 (runes), TypeScript, vitest + @testing-library/svelte, from `ui/`. Checks: `npm run check`, `npm run lint`, `npm test`. `ui/dist` is git-ignored and built on release - do not commit a build.

**Spec:** `docs/proposals/output-folders.md` - sections *The web UI* (~line 266), *Phasing* item 4, *Tests* → UI bullet.

## Global Constraints

- The field is `subfolder` on every surface; `''` means "at the run root". The engine treats no name specially and there is no default - the UI may *order* by the `final`/`intermediate` convention but must render any value.
- Spec, *The web UI*: "Gallery: a second, smaller filter for `subfolder` beside the folder filter, fed by `subfolders`, defaulting to everything. `grouping.ts` is unchanged - the server still supplies `folder`." Ruling for this plan: the gallery page loads the whole listing once (`api.gallery()`, `?limit=100000`) and filters client-side, so the control is fed by the distinct `subfolder` values of the loaded entries - the same set the server's `subfolders` lists - and no second request is made. The existing test `fetches the gallery listing exactly once on mount` stays as it is. Task 5 records this in the design.
- Spec, *The web UI*: "Job page: the manifest it already renders is grouped under headings by `subfolder` when any entry has a non-empty one; unchanged otherwise." "Unchanged otherwise" is binding: a run with every `subfolder === ''` (every pre-stage-1 job in history) renders exactly the DOM it renders today.
- Spec, *Phasing* 4: grouping is "live from `step_end`, confirmed from the manifest" - a `step_end` event's `subfolder` places the group while the job runs; the manifest entry's value wins once it arrives.
- Spec, *Tests* UI: "Gallery subfolder filter and job-page grouping (vitest)."
- `ui/CLAUDE.md` design system: mono (`--font-mono`) for anything the engine resolves literally - a subfolder name is one; sentence case, no ALL-CAPS labels; no accent colour; `--live` only for what is running. A `<select>` takes the global `select` style in `app.css` - no new colours.
- The gallery's folder grouping (`FolderGroups`, `folder`, `strip_run_id`) is untouched.
- A historical job whose manifest entries predate stage 1 has no `subfolder` key: the type is optional on `ManifestEntry` and the code reads `entry.subfolder ?? ''`. `GalleryFile.subfolder` is required - the server always sends it.
- Only the files this plan names change. No server, engine or MCP change; no `ui/e2e` change (the fixture writes unfoldered runs, and both spec files still pass).
- Commit trailer on every commit: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Run `npm test`, `npm run check` and `npm run lint` from `ui/` before every commit that touches `ui/`.

---

## File map

| File | Change |
|---|---|
| `ui/src/lib/types.ts` | `subfolder?: string` on `ManifestEntry`; new `StepEndEvent`; `subfolder: string` on `GalleryFile` |
| `ui/src/lib/results.ts` | step groups carry `subfolder`; new `sectionBySubfolder` |
| `ui/src/lib/results.test.ts` | tests for both |
| `ui/src/lib/pages/JobPage.svelte` | render sections |
| `ui/src/lib/pages/JobPage.test.ts` | **new** - rendering test for the sectioned and unsectioned cases |
| `ui/src/lib/pages/GalleryPage.svelte` | subfolder `<select>` |
| `ui/src/lib/pages/GalleryPage.test.ts` | `file()` factory gains `subfolder`; filter tests |
| `docs/SERVER.md`, `docs/proposals/output-folders.md`, `CLAUDE.md` | one sentence each |

---

### Task 1: The types carry `subfolder`

**Files:**
- Modify: `ui/src/lib/types.ts:14-35` (`ManifestEntry`, `JobEvent`) and `:198-206` (`GalleryFile`)
- Modify: `ui/src/lib/pages/GalleryPage.test.ts:16-24` (the `file()` factory, so `npm run check` stays green)

**Interfaces:**
- Produces: `ManifestEntry.subfolder?: string`; `StepEndEvent` (below); `GalleryFile.subfolder: string`. Tasks 2-4 import these.

- [ ] **Step 1: Edit `ui/src/lib/types.ts`**

Replace the `ManifestEntry` and `JobEvent` interfaces with:

```ts
export interface ManifestEntry {
  step: string
  files: string[]
  /** The step was served from the step cache: these files are an earlier
   * run's, republished, and nothing was generated for them this time. */
  reused?: boolean
  /** The in-run subfolder the step's `result.subfolder` chose - `final`,
   * `intermediate`, any relative path - `''` when it chose none. Absent
   * only on a job recorded before the field existed. */
  subfolder?: string
}
```

```ts
export interface JobEvent {
  seq: number
  event: string
  [key: string]: unknown
}

/** The `step_end` event, as the job page reads it: what the step saved and
 * where, before the manifest confirms it. */
export interface StepEndEvent extends JobEvent {
  event: 'step_end'
  step: string
  files?: string[]
  subfolder?: string
  reused?: boolean
}
```

In `GalleryFile`, after `folder: string` add:

```ts
  /** What followed the run id in the file's path - the `final` /
   * `intermediate` a step's `result.subfolder` chose, `''` for none. */
  subfolder: string
```

- [ ] **Step 2: Update the gallery test factory**

In `ui/src/lib/pages/GalleryPage.test.ts` the `file()` helper builds a `GalleryFile`; it now fails `npm run check` without the new field. Replace it with:

```ts
const file = (name: string, subfolder = ''): GalleryFile => ({
  name,
  folder: name.includes('/') ? name.split('/')[0] : '',
  subfolder,
  url: `/outputs/${name}`,
  kind: 'image',
  size: 1024,
  mtime: 1,
  label: name.split('/').pop()!.split('.')[0],
})
```

- [ ] **Step 3: Check and test**

Run from `ui/`: `npm run check && npm run lint && npm test`
Expected: all green (144 tests). `npm run check` must report 0 errors - grep the codebase for any other `GalleryFile` literal (`grep -rn "kind: 'image'" ui/src`) and add `subfolder: ''` wherever one is built.

- [ ] **Step 4: Commit**

```bash
git add ui/src/lib/types.ts ui/src/lib/pages/GalleryPage.test.ts
git commit -m "feat(ui): the manifest, step_end and gallery types carry subfolder

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `results.ts` sections step groups by subfolder

**Files:**
- Modify: `ui/src/lib/results.ts`
- Test: `ui/src/lib/results.test.ts`

**Interfaces:**
- Consumes: `ManifestEntry.subfolder?`, `StepEndEvent` from Task 1.
- Produces:
  ```ts
  export interface StepGroup { step: string; files: string[]; reused: boolean; subfolder: string }
  export function groupResultFiles(manifest: ManifestEntry[] | undefined, events: JobEvent[]): StepGroup[]
  export interface SubfolderSection { subfolder: string; groups: StepGroup[] }
  export function sectionBySubfolder(groups: StepGroup[]): SubfolderSection[]
  ```
  `sectionBySubfolder` returns **one** section with `subfolder: ''` when no group has a subfolder (the "unchanged otherwise" case, so the page can test `sections.length > 1 || sections[0].subfolder !== ''` to decide whether to show headings). Otherwise one section per distinct subfolder in first-appearance order, except that `final` is hoisted to the front - the deliverable is what a person opened the page for. Task 3 relies on exactly this ordering.

- [ ] **Step 1: Write the failing tests**

Append to `ui/src/lib/results.test.ts` (and add `sectionBySubfolder` to the import from `./results`; update the `stepEnd` helper to take an optional subfolder):

```ts
const stepEnd = (step: string, files: string[], subfolder?: string): JobEvent =>
  ({ seq: 0, event: 'step_end', step, files, subfolder }) as unknown as JobEvent
```

```ts
describe('groupResultFiles subfolder', () => {
  it('places a group from the live step_end and lets the manifest confirm it', () => {
    const groups = groupResultFiles(
      [{ step: 'episode', files: ['final/e.mp4'], subfolder: 'final' }],
      [stepEnd('episode', ['final/e.mp4'], 'final')],
    )
    expect(groups).toEqual([
      { step: 'episode', files: ['final/e.mp4'], reused: false, subfolder: 'final' },
    ])
  })

  it('reads the subfolder from the stream alone while the job runs', () => {
    const groups = groupResultFiles(undefined, [
      stepEnd('shot', ['intermediate/s.mp4'], 'intermediate'),
    ])
    expect(groups[0].subfolder).toBe('intermediate')
  })

  it('treats a manifest entry without the field as the run root', () => {
    const groups = groupResultFiles([{ step: 'old', files: ['a.png'] }], [])
    expect(groups[0].subfolder).toBe('')
  })

  it('lets the manifest win when it names a different subfolder', () => {
    const groups = groupResultFiles(
      [{ step: 's', files: ['final/x.png'], subfolder: 'final' }],
      [stepEnd('s', ['final/x.png'], '')],
    )
    expect(groups[0].subfolder).toBe('final')
  })
})

describe('sectionBySubfolder', () => {
  const group = (step: string, subfolder: string) => ({
    step,
    files: [`${subfolder ? subfolder + '/' : ''}${step}.png`],
    reused: false,
    subfolder,
  })

  it('returns one root section when no group has a subfolder', () => {
    const sections = sectionBySubfolder([group('a', ''), group('b', '')])
    expect(sections).toEqual([
      { subfolder: '', groups: [group('a', ''), group('b', '')] },
    ])
  })

  it('returns one root section for no groups at all', () => {
    expect(sectionBySubfolder([])).toEqual([{ subfolder: '', groups: [] }])
  })

  it('sections by subfolder in first-appearance order, final first', () => {
    const sections = sectionBySubfolder([
      group('draw', 'intermediate'),
      group('shot', 'intermediate'),
      group('episode', 'final'),
      group('notes', ''),
    ])
    expect(sections.map((s) => s.subfolder)).toEqual(['final', 'intermediate', ''])
    expect(sections[1].groups.map((g) => g.step)).toEqual(['draw', 'shot'])
  })

  it('keeps any name the engine accepted, not just the convention', () => {
    const sections = sectionBySubfolder([
      group('a', 'shots/act-1'),
      group('b', 'final'),
    ])
    expect(sections.map((s) => s.subfolder)).toEqual(['final', 'shots/act-1'])
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run from `ui/`: `npx vitest run src/lib/results.test.ts`
Expected: FAIL - `sectionBySubfolder` is not exported; the `groupResultFiles` equality fails on the missing `subfolder` key.

- [ ] **Step 3: Implement**

Replace `ui/src/lib/results.ts` with:

```ts
import type { JobEvent, ManifestEntry, StepEndEvent } from './types'

/** One step's output files, with where in the run directory they landed. */
export interface StepGroup {
  step: string
  files: string[]
  reused: boolean
  /** The step's `result.subfolder` - `''` for the run root. */
  subfolder: string
}

/** A run's output files, grouped by the step that produced them. The
 * step_end stream carries the association live; the manifest confirms it
 * at the end - merged here so the grouping never flattens (the old
 * behavior pooled every file into one bag). */
export function groupResultFiles(
  manifest: ManifestEntry[] | undefined,
  events: JobEvent[],
): StepGroup[] {
  const order: string[] = []
  const byStep = new Map<string, Set<string>>()
  // Only the manifest knows a step was served from the step cache; the
  // live step_end stream carries files but not that flag
  const reused = new Set<string>()
  // Placed by the live step_end, confirmed by the manifest - the manifest
  // is written last, so its value is the one that stands
  const subfolder = new Map<string, string>()
  const add = (step: string, files: string[], where: string | undefined) => {
    if (!byStep.has(step)) {
      byStep.set(step, new Set())
      order.push(step)
    }
    const set = byStep.get(step)!
    for (const file of files) set.add(file)
    if (where !== undefined) subfolder.set(step, where)
  }
  for (const event of events) {
    if (event.event === 'step_end') {
      const end = event as StepEndEvent
      add(end.step || '(unnamed)', end.files ?? [], end.subfolder)
    }
  }
  for (const entry of manifest ?? []) {
    add(entry.step, entry.files, entry.subfolder ?? '')
    if (entry.reused) reused.add(entry.step)
  }
  return order
    .map((step) => ({
      step,
      files: [...byStep.get(step)!],
      reused: reused.has(step),
      subfolder: subfolder.get(step) ?? '',
    }))
    .filter((group) => group.files.length > 0)
}

/** The step groups of one subfolder, under one heading on the job page. */
export interface SubfolderSection {
  subfolder: string
  groups: StepGroup[]
}

/** Step groups sectioned by the subfolder they landed in. A run where no
 * step chose one is a single root section, so the page renders as it did
 * before the field existed. Otherwise sections follow first appearance,
 * except `final` leads: by convention it is the deliverable, the thing the
 * page was opened for. The engine treats no name specially - any other
 * value is a section like the rest. */
export function sectionBySubfolder(groups: StepGroup[]): SubfolderSection[] {
  const order: string[] = []
  const bySubfolder = new Map<string, StepGroup[]>()
  for (const group of groups) {
    if (!bySubfolder.has(group.subfolder)) {
      bySubfolder.set(group.subfolder, [])
      order.push(group.subfolder)
    }
    bySubfolder.get(group.subfolder)!.push(group)
  }
  if (order.every((s) => s === '')) return [{ subfolder: '', groups }]
  const ordered = order.includes('final')
    ? ['final', ...order.filter((s) => s !== 'final')]
    : order
  return ordered.map((subfolder) => ({
    subfolder,
    groups: bySubfolder.get(subfolder)!,
  }))
}
```

Then update the four existing `groupResultFiles` tests' expected objects to include `subfolder: ''` (they use `toEqual` on whole objects).

- [ ] **Step 4: Run the tests to verify they pass**

Run from `ui/`: `npx vitest run src/lib/results.test.ts && npm run check && npm run lint`
Expected: PASS, 0 check errors. (`JobPage.svelte` still compiles: it reads `group.step`, `group.files`, `group.reused` only.)

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/results.ts ui/src/lib/results.test.ts
git commit -m "feat(ui): step groups carry subfolder; sectionBySubfolder puts final first

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: The job page renders results under subfolder headings

**Files:**
- Modify: `ui/src/lib/pages/JobPage.svelte:183-186` (derivations), `:346-395` (Results panel), `:526-533` (styles)
- Create: `ui/src/lib/pages/JobPage.test.ts`

**Interfaces:**
- Consumes: `sectionBySubfolder`, `StepGroup`, `SubfolderSection` from Task 2.

Background for the implementer: `JobPage.svelte` reads a job by id from the router, fetches `api.getJob(id)` and streams events; `fileGroups` (line ~184) is `groupResultFiles(job?.manifest, events)`. The Results panel (line ~346) iterates `fileGroups`, shows an `<h3 class="stephead muted">` per step when there is more than one group, and lists the files in a `.media` flex row, labelling each by `file.split('/').pop()` - which is why a `final/x.png` reads as `x.png`; the subfolder heading is what tells the reader where it is.

- [ ] **Step 1: Write the failing rendering test**

Create `ui/src/lib/pages/JobPage.test.ts`. The page's whole mount surface, read from `JobPage.svelte`'s `<script>`: it takes `jobId` as a **prop** (no router mock needed - `go` from `../router.svelte` is only called on rerun); on mount it calls `api.getJobWorkflow(jobId)` (the flow-view definition; the code reads `result.definition` and `result.seed_variable`, so resolve it to `{ definition: null, seed_variable: null }`), then `api.getJob(jobId)`, and unless `detail.historical` it opens `streamJobEvents(jobId, -1, onEvent, onEnd)` - a **named export** of `../api`, not a method on `api`. `outputUrl(path, jobId, workspace)` and `ApiError` are also named imports from `../api`; `FlowView` renders only when `definition` is non-null, so it needs no fixture.

```ts
import { cleanup, render, screen, waitFor, within } from '@testing-library/svelte'
import { afterEach, expect, it, vi } from 'vitest'
import JobPage from './JobPage.svelte'
import type { JobDetail, JobEvent } from '../types'

const detail = vi.hoisted(() => ({ job: null as JobDetail | null }))
// The live stream's callback, captured so a test can push step_end events
const stream = vi.hoisted(() => ({
  onEvent: null as ((event: JobEvent) => void) | null,
}))

vi.mock('../api', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    getJob: vi.fn(() => Promise.resolve(detail.job)),
    getJobWorkflow: vi.fn(() =>
      Promise.resolve({ definition: null, seed_variable: null }),
    ),
  },
  outputUrl: (path: string) => `/outputs/${path}`,
  streamJobEvents: (
    _jobId: string,
    _after: number,
    onEvent: (event: JobEvent) => void,
  ) => {
    stream.onEvent = onEvent
    return () => {}
  },
}))
vi.mock('../toast', () => ({ notify: { error: vi.fn(), success: vi.fn() } }))

const job = (manifest: JobDetail['manifest']): JobDetail => ({
  id: 'j1',
  workflow: 'templates/minimax/storyboard',
  status: 'succeeded',
  created_at: 1,
  started_at: 1,
  finished_at: 2,
  workspace: 'default',
  arguments: {},
  warnings: [],
  manifest,
  error: null,
  traceback: null,
  event_count: 0,
})

afterEach(() => {
  cleanup()
  stream.onEvent = null
})

it('groups a foldered run under final/ and intermediate/ headings, final first', async () => {
  detail.job = job([
    { step: 'board_1', files: ['intermediate/b1.png'], subfolder: 'intermediate' },
    { step: 'voyage', files: ['final/voyage.mp4'], subfolder: 'final' },
  ])
  render(JobPage, { jobId: 'j1' })
  const headings = await waitFor(() => {
    const found = screen.getAllByRole('heading', { level: 3 })
    expect(found.length).toBeGreaterThanOrEqual(2)
    return found
  })
  expect(headings.map((h) => h.textContent?.trim())).toEqual(['final/', 'intermediate/'])
  // Step names drop to h4 under a subfolder heading
  expect(
    screen.getAllByRole('heading', { level: 4 }).map((h) => h.textContent?.trim()),
  ).toEqual(['voyage', 'board_1'])
})

it('renders an unfoldered run exactly as before: step headings, no subfolder heading', async () => {
  detail.job = job([
    { step: 'generate', files: ['a.png'] },
    { step: 'upscale', files: ['a_big.png'], subfolder: '' },
  ])
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(screen.getByText('upscale')).toBeTruthy())
  expect(
    screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent?.trim()),
  ).toEqual(['generate', 'upscale'])
  expect(screen.queryByRole('heading', { level: 4 })).toBeNull()
  expect(screen.queryByText(/\/$/)).toBeNull()
})

it('places a live step_end under its subfolder before the manifest arrives', async () => {
  detail.job = { ...job([]), status: 'running', finished_at: null }
  render(JobPage, { jobId: 'j1' })
  await waitFor(() => expect(stream.onEvent).not.toBeNull())
  stream.onEvent!({
    seq: 1,
    event: 'step_end',
    step: 'episode',
    files: ['final/e.mp4'],
    subfolder: 'final',
  })
  await waitFor(() =>
    expect(screen.getByRole('heading', { level: 3, name: 'final/' })).toBeTruthy(),
  )
})
```

`jobId` is a prop, so each test mounts with `render(JobPage, { jobId: 'j1' })`. The Results panel's `<h3 class="stephead">` is the only `<h3>` in `JobPage.svelte` today (the *Progress*/*Workflow*/*Error*/*Log* panels use `<h2>`), so the unscoped heading queries are exact; if the `../toast` mock's shape is off, read `ui/src/lib/toast.ts` for the `notify` members and mock every one the page calls. `within` is imported for the case the reviewer asks the queries be scoped to the Results panel - drop the import if unused (lint fails on it).

- [ ] **Step 2: Run the test to verify it fails**

Run from `ui/`: `npx vitest run src/lib/pages/JobPage.test.ts`
Expected: the first and third tests FAIL (no `final/` heading, no h4); the second PASSES already (it pins the unchanged case).

- [ ] **Step 3: Implement the sectioned Results panel**

In the `<script>` of `JobPage.svelte`, after `fileGroups`:

```ts
  import { groupResultFiles, sectionBySubfolder } from '../results'
```
(extend the existing import on line 12), and

```ts
  // Sections by result.subfolder - one root section, no heading, when no
  // step chose one, so an older run renders as it always did
  const sections = $derived(sectionBySubfolder(fileGroups))
  const sectioned = $derived(
    sections.length > 1 || sections[0].subfolder !== '',
  )
```

Replace the `{#each fileGroups as group (group.step)} … {/each}` block inside the Results panel (keep the `<h2>Results</h2>` and the `allReused` note above it) with:

```svelte
      {#each sections as section (section.subfolder)}
        {#if sectioned}
          <h3 class="subhead">
            {section.subfolder === '' ? '(run root)' : `${section.subfolder}/`}
          </h3>
        {/if}
        {#each section.groups as group (group.step)}
          {#if fileGroups.length > 1}
            <svelte:element
              this={sectioned ? 'h4' : 'h3'}
              class="stephead muted"
            >
              {group.step}
              {#if group.reused && !allReused}
                <span
                  class="muted"
                  title="served from the step cache - an
                       earlier run's files, nothing generated for this step"
                  >· reused</span
                >
              {/if}
            </svelte:element>
          {/if}
          <div class="media">
            {#each group.files as file (file)}
              {#if isImage(file)}
                <a
                  class="frame plain"
                  href={fileUrl(file)}
                  target="_blank"
                  title={file.split('/').pop()}
                  ><img src={fileUrl(file)} alt={file.split('/').pop()} /></a
                >
              {:else if isVideo(file)}
                <span class="frame">
                  <!-- svelte-ignore a11y_media_has_caption -->
                  <video src={fileUrl(file)} controls loop></video>
                </span>
              {:else}
                <a class="filelink" href={fileUrl(file)} target="_blank"
                  >{file.split('/').pop()}</a
                >
              {/if}
            {/each}
          </div>
        {/each}
      {/each}
```

The inner file markup is the existing markup, moved one level in - do not change it. `'(run root)'` mirrors the prompt editor's `(root)` option and the gallery's `''` folder; it only ever shows beside a real subfolder heading.

Styles - add beside `.stephead`:

```css
  .subhead {
    font-size: var(--t-sm);
    text-transform: none;
    margin: var(--space-3) 0 var(--space-1);
  }
  .subhead:first-of-type {
    margin-top: 0;
  }
  .subhead + .stephead {
    margin-top: 0;
  }
```

and extend `.stephead` itself with the heading treatment `app.css` gives `h1, h2, h3` only - an `h4` would otherwise fall back to the browser's sans bold, and a step name is engine-resolved, so mono:

```css
  .stephead {
    font-family: var(--font-mono);
    font-weight: 600;
    line-height: 1.15;
    letter-spacing: -0.01em;
    font-size: 0.78rem;
    text-transform: none;
    margin: var(--space-3) 0 var(--space-2);
  }
```

(No change in the `h3` case - those values are what the global rule already gives it. Do not edit `app.css`.) `.subhead` is an `h3`, so `final/` is already mono. `.stephead:first-of-type` is per element type: in the sectioned case it matches the first `h4.stephead` in the panel and sets `margin-top: 0`, the same thing `.subhead + .stephead` sets - redundant there, not inert. Keep both rules: the second and later sections' first `h4` rely on `.subhead + .stephead`.

- [ ] **Step 4: Run the tests, check and lint**

Run from `ui/`: `npx vitest run src/lib/pages/JobPage.test.ts && npm test && npm run check && npm run lint`
Expected: all PASS, 0 errors. If `svelte-check` complains about `this={…}` on `svelte:element` with a class attribute, the attribute order is fine in Svelte 5 - re-read the message; it is more likely an unused-import or an a11y note.

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/pages/JobPage.svelte ui/src/lib/pages/JobPage.test.ts
git commit -m "feat(ui): the job page groups results under final/ and intermediate/ headings

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The gallery filters by subfolder

**Files:**
- Modify: `ui/src/lib/pages/GalleryPage.svelte:22-25` (state), `:87-93` (`visible`), `:276-281` (`.head`), `:305-319` (`Select all` label, `FolderGroups filterActive`), `:479-482` (styles)
- Test: `ui/src/lib/pages/GalleryPage.test.ts`

**Interfaces:**
- Consumes: `GalleryFile.subfolder` (Task 1); the `file(name, subfolder)` factory (Task 1).

- [ ] **Step 1: Write the failing tests**

Append to `ui/src/lib/pages/GalleryPage.test.ts`:

```ts
it('offers no subfolder control when nothing was written to one', async () => {
  await renderGallery()
  expect(screen.queryByRole('combobox', { name: 'subfolder' })).toBeNull()
})

it('lists the subfolders the outputs landed in and filters the grid by one', async () => {
  listing.files = [
    file('wf/run-1/final/deliverable.png', 'final'),
    file('wf/run-1/intermediate/scratch.png', 'intermediate'),
    file('wf/run-1/root.png', ''),
  ]
  await renderGallery('wf/run-1/final/deliverable.png')

  const pick = screen.getByRole('combobox', { name: 'subfolder' }) as HTMLSelectElement
  expect([...pick.options].map((o) => o.textContent?.trim())).toEqual([
    'all subfolders',
    '(run root)',
    'final/',
    'intermediate/',
  ])

  pick.value = 'final'
  pick.dispatchEvent(new Event('change', { bubbles: true }))
  await waitFor(() =>
    expect(screen.queryByLabelText('select wf/run-1/intermediate/scratch.png')).toBeNull(),
  )
  expect(screen.getByLabelText('select wf/run-1/final/deliverable.png')).toBeTruthy()
  expect(screen.queryByLabelText('select wf/run-1/root.png')).toBeNull()
  // Select all takes what the subfolder filter leaves showing
  screen.getByRole('button', { name: /select all matching \(1\)/i }).click()
  await waitFor(() => expect(screen.getByText('1 selected')).toBeTruthy())
})

it('intersects the subfolder pick with the text filter', async () => {
  listing.files = [
    file('wf/run-1/final/a.png', 'final'),
    file('wf/run-1/final/b.png', 'final'),
    file('wf/run-1/intermediate/a.png', 'intermediate'),
  ]
  await renderGallery('wf/run-1/final/a.png')

  const pick = screen.getByRole('combobox', { name: 'subfolder' }) as HTMLSelectElement
  pick.value = 'final'
  pick.dispatchEvent(new Event('change', { bubbles: true }))
  const filter = screen.getByPlaceholderText('filter…') as HTMLInputElement
  filter.value = '/a.'
  filter.dispatchEvent(new Event('input', { bubbles: true }))

  await waitFor(() =>
    expect(screen.queryByLabelText('select wf/run-1/final/b.png')).toBeNull(),
  )
  expect(screen.getByLabelText('select wf/run-1/final/a.png')).toBeTruthy()
  expect(screen.queryByLabelText('select wf/run-1/intermediate/a.png')).toBeNull()
})

it('falls back to every subfolder when the picked one empties out', async () => {
  listing.files = [
    file('wf/run-1/final/only.png', 'final'),
    file('wf/run-1/root.png', ''),
  ]
  await renderGallery('wf/run-1/final/only.png')
  const pick = screen.getByRole('combobox', { name: 'subfolder' }) as HTMLSelectElement
  pick.value = 'final'
  pick.dispatchEvent(new Event('change', { bubbles: true }))
  await waitFor(() =>
    expect(screen.queryByLabelText('select wf/run-1/root.png')).toBeNull(),
  )

  checkbox('wf/run-1/final/only.png').click()
  screen.getByRole('button', { name: /^delete/i }).click()
  await answerConfirm(true)

  // The last final/ file is gone: the control goes with it and the grid
  // shows everything again rather than an empty page pinned to a value
  // that no longer exists
  await waitFor(() =>
    expect(screen.getByLabelText('select wf/run-1/root.png')).toBeTruthy(),
  )
  expect(screen.queryByRole('combobox', { name: 'subfolder' })).toBeNull()
})
```

Note: the fourth test's "Delete" click is the bulk `Delete` in the `.picks` bar - the only button of that name before the dialog opens (the existing delete tests use the same `/^delete/i`), and `answerConfirm` is scoped to the dialog that follows.

- [ ] **Step 2: Run to verify they fail**

Run from `ui/`: `npx vitest run src/lib/pages/GalleryPage.test.ts`
Expected: the first PASSES (no control exists yet); the other three FAIL on the missing combobox.

- [ ] **Step 3: Implement**

State (beside `let filter = $state('')`):

```ts
  // The in-run subfolder to show - null is every one. Only offered when
  // some output landed in one; a workspace of unfoldered runs never sees
  // the control
  let subfolder = $state<string | null>(null)
```

Derivations - replace the `visible` derivation:

```ts
  // Every distinct subfolder in the listing, '' (the run root) included
  // once anything is nested - the same set the server's `subfolders`
  // reports, read off the entries so a delete updates it without a refetch
  const subfolders = $derived(
    [...new Set(files.map((f) => f.subfolder))].sort((a, b) =>
      a.localeCompare(b),
    ),
  )
  const subfolderOffered = $derived(subfolders.some((s) => s !== ''))
  const filterActive = $derived(filter !== '' || subfolder !== null)
  const visible = $derived(
    files.filter(
      (f) =>
        f.name.toLowerCase().includes(filter.toLowerCase()) &&
        (subfolder === null || f.subfolder === subfolder),
    ),
  )
  // A pick that no longer exists (its last file deleted) means everything,
  // not an empty grid pinned to a vanished value - and the control itself
  // is reset, since a <select> whose value matches no option shows blank
  $effect(() => {
    if (subfolder !== null && !subfolders.includes(subfolder)) subfolder = null
  })
```

Markup - in `.head`, after the filter input:

```svelte
  {#if subfolderOffered}
    <select
      class="subfolderpick"
      aria-label="subfolder"
      title="show only files a step saved to this subfolder of its run"
      bind:value={subfolder}
    >
      <option value={null}>all subfolders</option>
      {#each subfolders as s (s)}
        <option value={s}>{s === '' ? '(run root)' : `${s}/`}</option>
      {/each}
    </select>
  {/if}
```

Svelte 5 binds non-string option values (`null`) through `bind:value`; the test drives the control by setting `value = 'final'` and dispatching `change`, which reaches the same binding.

`Select all` label: change `{filter ? ' matching' : ''}` to `{filterActive ? ' matching' : ''}`. `FolderGroups`: change `filterActive={filter !== ''}` to `filterActive={filterActive}`.

Style, beside `.filter`:

```css
  .subfolderpick {
    /* The global select rule is width: 100% - here it must share the row */
    width: auto;
    max-width: 200px;
    font-family: var(--font-mono);
    font-size: var(--t-sm);
  }
```

(The option text is a path segment the engine wrote - mono, per the design system. Leave the global `select` colours alone; `width: auto` is what keeps the control beside the filter rather than wrapped onto a row of its own.) The `.filter` rule's `margin-left: auto` pushes the input to the right edge; the select lands after it and inherits the `.head` gap - check the layout reads "filter, then subfolder" and is not wrapped oddly at narrow widths; if the select should sit before the input, move it and keep `margin-left: auto` on `.filter`.

- [ ] **Step 4: Run tests, check, lint**

Run from `ui/`: `npm test && npm run check && npm run lint`
Expected: all PASS, 0 errors. `fetches the gallery listing exactly once on mount` must still pass - the new derivations make no request.

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/pages/GalleryPage.svelte ui/src/lib/pages/GalleryPage.test.ts
git commit -m "feat(ui): gallery filters by result subfolder

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Docs close the stage

**Files:**
- Modify: `docs/SERVER.md` (the *Jobs* bullet ~line 50 and the *Gallery* bullet ~line 79)
- Modify: `docs/proposals/output-folders.md` (status line near the top; *The web UI* Gallery bullet; *Phasing* item 4)
- Modify: `CLAUDE.md` (*Result subfolders* gotcha - one sentence)

- [ ] **Step 1: `docs/SERVER.md`**

In the *Jobs* bullet, the sentence "…and its result files as they land." wraps across two lines (~55-56, ending `they land.`); the next line begins "Jobs can be cancelled". Insert after `they land.` and before "Jobs can be cancelled", rewrapped to the bullet's width:

```
  A run whose steps chose a `result.subfolder` shows its results under
  `final/` and `intermediate/` headings, the deliverable first; one that
  chose none shows them as before.
```

In the *Gallery* bullet, after "…rather than listing each run separately," change the sentence so it reads:

```
  which the engine lays out as `<workflow>/<run id>/`. The folder filter groups a workflow's runs
  together rather than listing each run separately, and a **subfolder** pick
  beside the text filter - offered once any output landed in one - narrows
  the grid to `final/`, `intermediate/` or whatever a step's
  `result.subfolder` named; each run directory
  also holds a `manifest.json` describing what produced it (see
```

- [ ] **Step 2: `docs/proposals/output-folders.md`**

Status line: change `**stages 1-3 (engine, server/MCP, steering) implemented; stage 4 (UI) pending**` to `**all four stages (engine, server/MCP, steering, UI) implemented**`.

*The web UI*, Gallery bullet - replace with:

```
- Gallery: a second, smaller filter for `subfolder` beside the text filter,
  a `<select>` fed by the distinct `subfolder` values of the loaded entries
  (the same set the server's `subfolders` lists - the page already holds
  the whole listing and filters client-side, so no second request), shown
  only once some output landed in one, defaulting to everything. The two
  filters intersect. `grouping.ts` is unchanged - the server still supplies
  `folder`.
- Job page: the manifest it already renders is grouped under headings by
  `subfolder` when any entry has a non-empty one - `final/` first, then in
  order of appearance, `(run root)` for the empty value - with the step
  headings one level down; unchanged otherwise. The grouping is live from
  `step_end` and confirmed by the manifest (`sectionBySubfolder` in
  `ui/src/lib/results.ts`).
```

*Phasing* item 4 - change "`subfolder` on the `ManifestEntry` and `JobEvent` types" to "`subfolder` on the `ManifestEntry` and `GalleryFile` types and a `StepEndEvent` subtype of `JobEvent`" (the `JobEvent` type is an index signature, so the field lives on a typed subtype).

- [ ] **Step 3: `CLAUDE.md`**

In the *Result subfolders* gotcha, after "…MCP `list_gallery(subfolder=)` filter on it." insert before "`file_base_name` may not contain a separator":

```
  The web UI reads the field only: the gallery page offers a subfolder pick
  once any entry has one, and the job page sections results under `final/` /
  `intermediate/` headings (`sectionBySubfolder`, `ui/src/lib/results.ts`),
  unchanged for a run that chose none.
```

Keep the bullet's wrapping width (~90 columns) and make sure the inserted sentence joins the paragraph with a period on both sides.

- [ ] **Step 4: Check nothing else pins the old status**

Run: `grep -rn "stage 4 (UI) pending\|stages 1-3" docs/proposals docs/SERVER.md docs/MCP.md docs/WORKFLOW_GUIDE.md CLAUDE.md`
Expected: no matches. (The plans under `docs/superpowers/` legitimately mention both phrases - they are history, not status.)

- [ ] **Step 5: Commit**

```bash
git add docs/SERVER.md docs/proposals/output-folders.md CLAUDE.md
git commit -m "docs: the UI stage of output subfolders is in; design marks all four stages done

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage.** *The web UI* Gallery → Task 4; Job page → Tasks 2-3; Editor → none needed (schema-driven, stated in Architecture). *Phasing* 4: types → Task 1; gallery control → Task 4; grouping live/confirmed → Task 2 (`subfolder` map, manifest wins) + Task 3 third test. *Tests* UI bullet → `results.test.ts`, `JobPage.test.ts`, `GalleryPage.test.ts`. Docs → Task 5.
- **Placeholders.** None: Task 3's mock surface was read off `JobPage.svelte` (`getJobWorkflow`, `getJob`, named `streamJobEvents`/`outputUrl`/`ApiError`, `jobId` prop).
- **Type consistency.** `StepGroup.subfolder: string`, `SubfolderSection { subfolder, groups }`, `sectionBySubfolder(groups)` used identically in Tasks 2, 3 and 5. `file(name, subfolder = '')` from Task 1 is what Task 4's tests call. `GalleryFile.subfolder` required; `ManifestEntry.subfolder` optional and read with `?? ''` in Task 2 - matches Global Constraints.
