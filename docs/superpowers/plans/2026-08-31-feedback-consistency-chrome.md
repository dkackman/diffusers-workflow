# Feedback Layer, Consistency Sweep & Chrome Implementation Plan (UI/UX stages 3–5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the banner stack with toasts + a dismissible validation panel, unify the copy-forked library pages behind shared components, group job results by producing step, split the header into a clean nav row + status strip, and add a discoverable keyboard layer.

**Architecture:** A thin `notify` wrapper over svelte-sonner (battle-tested, Svelte 5-native) backs all transient feedback; a snippet-based `FolderGroups` component absorbs the duplicated grouping/collapse logic of the Workflows and Prompts pages; a pure `groupResultFiles` function feeds the run page's step-grouped results; App.svelte gains a second slim header row and a global `?` help overlay.

**Tech Stack:** Svelte 5 (runes + snippets), svelte-sonner, TypeScript, Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-08-31-ui-ux-optimization-design.md` (stages 3–5; stages 1–2 already merged).

## Global Constraints

- All changes under `ui/` only (plus this plan's own doc edits). No API, schema, or Python changes.
- Svelte 5 runes syntax; snippets for component composition. New CSS uses `--space-*`/`--radius-*` tokens.
- Native `window.confirm` stays for destructive actions (undo layer is deferred).
- Existing localStorage keys keep working: `storage.ts` auto-prefixes `dw-`, so pass `collapsed-folders`, `collapsed-prompt-folders`, `hint-dismissed`, `prompt-hint-dismissed` to hit the keys pages already wrote.
- Gates from `ui/`: `npm run check && npm run lint && npm test`, then `npm run build && npx playwright test` (the e2e fixture serves the BUILT SPA; a worktree needs `ln -s <main-repo>/venv venv` at the worktree root). All must pass per task.
- `npm run format` before every commit.
- Reuse over rewrite (user preference): svelte-sonner for toasts. The `?` help overlay stays hand-rolled — one static sheet doesn't justify a dialog library (a11y compliance is out of scope this pass).

---

### Task 1: Toast infrastructure

**Files:**
- Modify: `ui/package.json` (add dependency `svelte-sonner`)
- Create: `ui/src/lib/toast.ts`
- Modify: `ui/src/App.svelte` (mount `<Toaster>`)

**Interfaces:**
- Consumes: App.svelte's existing `theme` state (`'system' | 'light' | 'dark'`).
- Produces: `notify.success(message: string): void`, `notify.error(message: string, id?: string): void` (sticky until dismissed), `notify.dismiss(id: string): void`. Every later task calls these — never `toast` from svelte-sonner directly.

- [ ] **Step 1: Install svelte-sonner**

Run: `cd ui && npm install svelte-sonner`
Expected: added to `dependencies` in package.json (v1.x, Svelte 5 compatible).

- [ ] **Step 2: Write the wrapper**

```ts
// ui/src/lib/toast.ts
import { toast } from 'svelte-sonner'

/** All app feedback goes through here, not svelte-sonner directly - one
 * place owns durations and dismissal semantics. Successes announce and
 * leave; errors stay until the user dismisses them. An id makes an error
 * replace its previous occurrence instead of stacking (a re-fired JSON
 * parse error on every blur, say). */
export const notify = {
  success(message: string) {
    toast.success(message)
  },
  error(message: string, id?: string) {
    toast.error(message, { duration: Number.POSITIVE_INFINITY, id })
  },
  dismiss(id: string) {
    toast.dismiss(id)
  },
}
```

- [ ] **Step 3: Mount the Toaster in App.svelte**

In `ui/src/App.svelte`: add to the imports

```ts
import { Toaster } from 'svelte-sonner'
```

and directly after `</header>` (before `<main ...>`):

```svelte
<Toaster position="bottom-right" closeButton {theme} duration={4000} />
```

`theme` is App's existing state and already uses sonner's exact vocabulary (`'system' | 'light' | 'dark'`), so the toasts follow the app's theme cycle for free.

- [ ] **Step 4: Run gates**

Run: `cd ui && npm run check && npm run lint && npm test`
Expected: all pass (nothing calls `notify` yet).

- [ ] **Step 5: Commit**

```bash
cd ui && npm run format && cd ..
git add ui/package.json ui/package-lock.json ui/src/lib/toast.ts ui/src/App.svelte
git commit -m "Add toast layer: svelte-sonner behind a notify wrapper"
```

---

### Task 2: Editor feedback migration

**Files:**
- Modify: `ui/src/lib/pages/EditorPage.svelte` (status/error banners → toasts; validation panel dismissible)
- Modify: `ui/src/lib/pages/PromptEditorPage.svelte` (same treatment)

**Interfaces:**
- Consumes: `notify` from Task 1.
- Produces: no banner markup between the toolbar and the form except the (dismissible) validation panel. The e2e-visible strings `Saved to …` and `schema-valid` keep their exact wording — smoke tests assert them; a toast's text is in the DOM, so `getByText` still finds it.

- [ ] **Step 1: Migrate EditorPage statuses and errors**

In `ui/src/lib/pages/EditorPage.svelte`:

1. Import `notify`: `import { notify } from '../toast'`.
2. Delete the `status` and `error` state declarations and every assignment to them, replacing each assignment site:
   - `status = 'Imported from image metadata'` → `notify.success('Imported from image metadata')`
   - `status = \`Saved to ${result.path}\`` → `notify.success(\`Saved to ${result.path}\`)`
   - every `error = <expr>` in `validate()`, `save()`, `run()` and the load `$effect`'s `.catch` → `notify.error(<expr>)`
   - the `save()` guard errors (`'Give the workflow a file name first'` etc.) → `notify.error(...)` (keep the `fileOpen = true` line — the message names a field the user cannot see while collapsed)
   - bare resets like `status = ''` / `error = ''` are simply deleted.
3. `applyJson` is the id-deduped case — a parse error re-fires on every blur:

```ts
function applyJson(raw: string) {
  jsonDraft = raw
  try {
    workflow = JSON.parse(raw)
    jsonParseFailed = false
    notify.dismiss('json-parse')
  } catch (e) {
    jsonParseFailed = true
    notify.error(`JSON: ${e instanceof Error ? e.message : e}`, 'json-parse')
  }
}
```

4. Delete the banner markup `{#if error}<p class="error">{error}</p>{/if}` and the `{#if status}` block, plus the now-unused `.status` CSS rule (keep `.error` — the validation panel still uses the class).
5. Make the validation panel dismissible: add an X button to its top-right.

```svelte
{#if validation}
  <div
    class="panel validation"
    class:error-edge={!validation.valid}
    class:warn-edge={validation.valid && validation.warnings.length > 0}
    class:good-edge={validation.valid && validation.warnings.length === 0}
  >
    <button
      class="quiet icon dismiss"
      onclick={() => (validation = null)}
      title="dismiss"
      aria-label="dismiss validation results"
    >
      <X size={13} />
    </button>
    ... (existing inner content unchanged)
  </div>
{/if}
```

with `X` added to the lucide import and scoped CSS:

```css
.validation {
  position: relative;
  padding-right: 2.2rem;
}
.dismiss {
  position: absolute;
  top: var(--space-2);
  right: var(--space-2);
  border: 0;
  padding: 0.2rem 0.3rem;
}
```

- [ ] **Step 2: Migrate PromptEditorPage the same way**

In `ui/src/lib/pages/PromptEditorPage.svelte`: import `notify`; replace every `status = '...'` (e.g. `'Duplicated - save under a new name'`, `` `Saved to ${result.path}` ``) with `notify.success(...)` and every `error = <expr>` with `notify.error(<expr>)`; delete bare `status = ''` / `error = ''` resets, the two state declarations, the `{#if error}`/`{#if status}` markup and the `.status` CSS. Its `applyJson` equivalent (around line 347) gets the same `'json-parse'`-id treatment as EditorPage's. **Leave `enhanceError` alone** — it renders inside the Enhance panel next to its controls, which is already the inline pattern this pass wants.

- [ ] **Step 3: Run gates + affected e2e**

Run: `cd ui && npm run check && npm run lint && npm test && npm run build && npx playwright test e2e/smoke.spec.ts`
Expected: all pass. The `Saved to` and `schema-valid` assertions now match toast/panel text. If a toast auto-dismisses before an assertion with a long timeout resolves, the assertion still passes because Playwright matches on first render; if a test genuinely flakes on dismissal timing, raise that test's toast-dependent assertion to `{ timeout: 10_000 }` rather than changing app behavior.

- [ ] **Step 4: Commit**

```bash
cd ui && npm run format && cd ..
git add ui/src/lib/pages/EditorPage.svelte ui/src/lib/pages/PromptEditorPage.svelte
git commit -m "Editors: banner stack becomes toasts plus a dismissible validation panel"
```

---

### Task 3: Action errors become toasts on the remaining pages

**Files:**
- Modify: `ui/src/lib/pages/WorkflowPage.svelte`, `ui/src/lib/pages/GalleryPage.svelte`, `ui/src/lib/pages/ModelsPage.svelte`, `ui/src/lib/pages/JobsPage.svelte`

**Interfaces:**
- Consumes: `notify` from Task 1.
- Produces: page-level rule later work relies on — **load failures stay inline** (they describe the page's empty body), **action failures toast** (they describe a verb that just failed).

- [ ] **Step 1: Split load vs action errors per page**

In each file, import `notify` and change only the catch-sites of user actions:

- `WorkflowPage.svelte`: `remove()` and `run()` catches → `notify.error(...)`; the load `$effect` catch keeps setting `error` (inline "Could not load" stays).
- `GalleryPage.svelte`: the delete action's catch → `notify.error(...)`; the listing load error stays inline.
- `ModelsPage.svelte`: catches inside user-initiated actions (delete model/revision, start download, diffusers update — the functions wrapping `api.` calls fired from buttons) → `notify.error(...)`; the initial cache-listing load error stays inline.
- `JobsPage.svelte`: the queue-reorder/cancel action catch (around line 47) → `notify.error(...)`; the polling load error stays inline.

Where a page's `error` state remains only for its load path, leave the state and its inline markup untouched — do not restructure beyond the catch-sites.

- [ ] **Step 2: Run gates**

Run: `cd ui && npm run check && npm run lint && npm test && npm run build && npx playwright test e2e/smoke.spec.ts`
Expected: all pass (no e2e exercises these failure paths).

- [ ] **Step 3: Commit**

```bash
cd ui && npm run format && cd ..
git add ui/src/lib/pages/WorkflowPage.svelte ui/src/lib/pages/GalleryPage.svelte ui/src/lib/pages/ModelsPage.svelte ui/src/lib/pages/JobsPage.svelte
git commit -m "Action failures toast; load failures stay inline"
```

---

### Task 4: Step-grouped job results

**Files:**
- Create: `ui/src/lib/results.ts`
- Test: `ui/src/lib/results.test.ts`
- Modify: `ui/src/lib/pages/JobPage.svelte:99-106` (the `liveFiles` derived) and its Results panel markup

**Interfaces:**
- Consumes: `JobEvent`, `ManifestEntry` from `ui/src/lib/types.ts` (`ManifestEntry { step: string; files: string[] }`; `step_end` events carry `step` and `files`).
- Produces: `groupResultFiles(manifest: ManifestEntry[] | undefined, events: JobEvent[]): Array<{ step: string; files: string[] }>` — files grouped by producing step, deduped, in step-completion order.

- [ ] **Step 1: Write the failing tests**

```ts
// ui/src/lib/results.test.ts
import { describe, expect, it } from 'vitest'
import { groupResultFiles } from './results'
import type { JobEvent } from './types'

const stepEnd = (step: string, files: string[]): JobEvent =>
  ({ seq: 0, event: 'step_end', step, files }) as unknown as JobEvent

describe('groupResultFiles', () => {
  it('groups streamed files by producing step, in completion order', () => {
    const groups = groupResultFiles(undefined, [
      stepEnd('generate', ['a.png', 'b.png']),
      stepEnd('upscale', ['a_big.png']),
    ])
    expect(groups).toEqual([
      { step: 'generate', files: ['a.png', 'b.png'] },
      { step: 'upscale', files: ['a_big.png'] },
    ])
  })

  it('merges the manifest without duplicating streamed files', () => {
    const groups = groupResultFiles(
      [{ step: 'generate', files: ['a.png', 'c.png'] }],
      [stepEnd('generate', ['a.png'])],
    )
    expect(groups).toEqual([{ step: 'generate', files: ['a.png', 'c.png'] }])
  })

  it('drops steps with no files and handles a historical job (manifest only)', () => {
    const groups = groupResultFiles(
      [
        { step: 'load', files: [] },
        { step: 'generate', files: ['a.png'] },
      ],
      [],
    )
    expect(groups).toEqual([{ step: 'generate', files: ['a.png'] }])
  })
})
```

- [ ] **Step 2: Run to verify failure**

Run: `cd ui && npx vitest run src/lib/results.test.ts`
Expected: FAIL — cannot resolve `./results`

- [ ] **Step 3: Implement**

```ts
// ui/src/lib/results.ts
import type { JobEvent, ManifestEntry } from './types'

/** A run's output files, grouped by the step that produced them. The
 * step_end stream carries the association live; the manifest confirms it
 * at the end - merged here so the grouping never flattens (the old
 * behavior pooled every file into one bag). */
export function groupResultFiles(
  manifest: ManifestEntry[] | undefined,
  events: JobEvent[],
): Array<{ step: string; files: string[] }> {
  const order: string[] = []
  const byStep = new Map<string, Set<string>>()
  const add = (step: string, files: string[]) => {
    if (!byStep.has(step)) {
      byStep.set(step, new Set())
      order.push(step)
    }
    const set = byStep.get(step)!
    for (const file of files) set.add(file)
  }
  for (const event of events) {
    if (event.event === 'step_end') {
      add((event.step as string) ?? '', (event.files as string[]) ?? [])
    }
  }
  for (const entry of manifest ?? []) add(entry.step, entry.files)
  return order
    .map((step) => ({ step, files: [...byStep.get(step)!] }))
    .filter((group) => group.files.length > 0)
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd ui && npx vitest run src/lib/results.test.ts`
Expected: PASS (3 tests)

- [ ] **Step 5: Wire into JobPage**

In `ui/src/lib/pages/JobPage.svelte`: import `groupResultFiles`; replace the `liveFiles` derived with

```ts
const fileGroups = $derived(
  groupResultFiles(job?.manifest, events as JobEvent[]),
)
```

and the Results panel with a per-step rendering (heading suppressed when there is only one group — no noise on single-step runs):

```svelte
{#if fileGroups.length}
  <div class="panel">
    <h2>Results</h2>
    {#each fileGroups as group (group.step)}
      {#if fileGroups.length > 1}
        <h3 class="stephead muted">{group.step}</h3>
      {/if}
      <div class="media">
        {#each group.files as file (file)}
          {#if isImage(file)}
            <a href={fileUrl(file)} target="_blank"
              ><img src={fileUrl(file)} alt={file.split('/').pop()} /></a
            >
          {:else if isVideo(file)}
            <!-- svelte-ignore a11y_media_has_caption -->
            <video src={fileUrl(file)} controls loop></video>
          {:else}
            <a href={fileUrl(file)} target="_blank">{file.split('/').pop()}</a>
          {/if}
        {/each}
      </div>
    {/each}
  </div>
{/if}
```

with scoped CSS:

```css
.stephead {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin: var(--space-3) 0 var(--space-2);
}
.stephead:first-of-type {
  margin-top: 0;
}
```

(`JobEvent` may need adding to the type-only import in JobPage.)

- [ ] **Step 6: Run gates and commit**

Run: `cd ui && npm run check && npm run lint && npm test`
Expected: all pass.

```bash
cd ui && npm run format && cd ..
git add ui/src/lib/results.ts ui/src/lib/results.test.ts ui/src/lib/pages/JobPage.svelte
git commit -m "Job results group by producing step"
```

---

### Task 5: Shared library-page components

**Files:**
- Create: `ui/src/lib/FolderGroups.svelte`, `ui/src/lib/HintBar.svelte`, `ui/src/lib/Empty.svelte`
- Modify: `ui/src/lib/pages/WorkflowsPage.svelte`, `ui/src/lib/pages/PromptsPage.svelte` (refactor onto them), `ui/src/lib/pages/JobsPage.svelte` + `ui/src/lib/pages/GalleryPage.svelte` (adopt `Empty`)

**Interfaces:**
- Consumes: `storageGet`/`storageSet` from `ui/src/lib/storage.ts` (auto-prefix `dw-`).
- Produces:
  - `FolderGroups`: props `{ names: string[]; collapseKey: string; filterActive: boolean; newHref?: string; onnewingroup?: (group: string) => void; card: Snippet<[string]> }` — renders folder group rows (collapse chevron persisted under `collapseKey`, hover `+` new-in-folder link when `newHref` given) and a card grid, calling the `card` snippet per name. While `filterActive`, every group renders open.
  - `HintBar`: props `{ storageKey: string; children: Snippet }` — dismissible dashed hint bar; dismissal persists under `storageKey` (value `true`; the legacy `'1'` string parses truthy through storageGet, so previously dismissed hints stay dismissed).
  - `Empty`: props `{ icon: Snippet; children: Snippet }` — centered icon + message, the Jobs/Gallery visual style.

- [ ] **Step 1: Write the three components**

```svelte
<!-- ui/src/lib/HintBar.svelte -->
<script lang="ts">
  import type { Snippet } from 'svelte'
  import { X } from '@lucide/svelte'
  import { storageGet, storageSet } from './storage'

  let { storageKey, children }: { storageKey: string; children: Snippet } =
    $props()

  // The legacy pages stored the string '1' - any truthy stored value
  // counts as dismissed
  let show = $state(!storageGet(storageKey, false))

  function dismiss() {
    show = false
    storageSet(storageKey, true)
  }
</script>

{#if show}
  <div class="hintbar muted">
    <span>{@render children()}</span>
    <button
      class="quiet icon"
      onclick={dismiss}
      title="dismiss"
      aria-label="dismiss this hint"
    >
      <X size={13} />
    </button>
  </div>
{/if}

<style>
  .hintbar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    border: 1px dashed var(--line);
    border-radius: var(--radius-1);
    padding: 0.45rem 0.7rem;
    font-size: 0.85rem;
    margin-bottom: var(--space-4);
  }
  .hintbar span {
    flex: 1;
  }
  .hintbar .icon {
    display: inline-flex;
    padding: 0.2rem 0.3rem;
  }
</style>
```

```svelte
<!-- ui/src/lib/Empty.svelte -->
<script lang="ts">
  import type { Snippet } from 'svelte'

  let { icon, children }: { icon: Snippet; children: Snippet } = $props()
</script>

<div class="empty muted">
  {@render icon()}
  <p>{@render children()}</p>
</div>

<style>
  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-5) 0;
    text-align: center;
  }
</style>
```

```svelte
<!-- ui/src/lib/FolderGroups.svelte -->
<script lang="ts">
  import type { Snippet } from 'svelte'
  import { ChevronDown, ChevronRight, Plus } from '@lucide/svelte'
  import { storageGet, storageSet } from './storage'

  let {
    names,
    collapseKey,
    filterActive,
    newHref = undefined,
    onnewingroup = undefined,
    card,
  }: {
    names: string[]
    collapseKey: string
    filterActive: boolean
    newHref?: string
    onnewingroup?: (group: string) => void
    card: Snippet<[string]>
  } = $props()

  let collapsed = $state<Record<string, boolean>>(storageGet(collapseKey, {}))

  function toggle(group: string) {
    collapsed[group] = !collapsed[group]
    storageSet(collapseKey, $state.snapshot(collapsed))
  }

  const groupOf = (name: string) =>
    name.includes('/') ? name.split('/')[0] : ''
  const groups = $derived(
    [...new Set(names.map(groupOf))].sort((a, b) => a.localeCompare(b)),
  )
  const inGroup = (group: string) =>
    names.filter((name) => groupOf(name) === group)
  // While filtering, everything stays visible - a collapsed folder hiding
  // matches would make the filter look broken
  const isOpen = (group: string) => filterActive || !collapsed[group]
</script>

{#each groups as group (group)}
  {#if group}
    <div class="grouprow">
      <button
        class="group"
        onclick={() => toggle(group)}
        title={isOpen(group) ? 'collapse this folder' : 'expand this folder'}
      >
        {#if isOpen(group)}<ChevronDown size={14} />{:else}<ChevronRight
            size={14}
          />{/if}
        {group}/ <span class="muted">({inGroup(group).length})</span>
      </button>
      {#if newHref}
        <a
          class="groupnew"
          href={newHref}
          onclick={() => onnewingroup?.(group)}
          title="new in {group}/"
          aria-label="new in {group}/"
        >
          <Plus size={13} />
        </a>
      {/if}
    </div>
  {/if}
  {#if isOpen(group)}
    <div class="grid">
      {#each inGroup(group) as name (name)}
        {@render card(name)}
      {/each}
    </div>
  {/if}
{/each}

<style>
  .grouprow {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin: 1.2rem 0 var(--space-2);
  }
  .groupnew {
    display: inline-flex;
    align-items: center;
    padding: 0.15rem;
    color: var(--muted);
    border: 1px solid transparent;
    border-radius: 4px;
    opacity: 0;
    transition: opacity 0.15s ease;
  }
  .grouprow:hover .groupnew {
    opacity: 1;
  }
  .groupnew:hover {
    color: var(--accent);
    border-color: var(--line);
  }
  .group {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    background: none;
    border: none;
    color: var(--muted);
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0;
    margin: 0;
    cursor: pointer;
  }
  .group:hover {
    color: var(--ink);
    filter: none;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 0.6rem;
  }
</style>
```

- [ ] **Step 2: Refactor WorkflowsPage onto them**

In `ui/src/lib/pages/WorkflowsPage.svelte`: delete the collapse/hint state, `readCollapsed`, `dismissHint`, `toggle`, `groupOf`/`groups`/`inGroup`/`isOpen` and the group/hint markup + their styles (`.hintbar`, `.grouprow`, `.groupnew`, `.group`, `.grid`); keep the `visible` filter derived and the card markup/styles. Render:

```svelte
<HintBar storageKey="hint-dismissed">
  Pick a workflow → tweak its variables → Run. Every image saves its recipe —
  reopen it from the Gallery.
</HintBar>

<FolderGroups
  names={visible}
  collapseKey="collapsed-folders"
  filterActive={filter !== ''}
  newHref="#/edit"
  onnewingroup={(group) => sessionStorage.setItem('dw-editor-folder', group)}
>
  {#snippet card(name)}
    {@const detail = details[name]}
    {@const group = name.includes('/') ? name.split('/')[0] : ''}
    <a
      class="card panel"
      href={href(name)}
      title={detail?.description || undefined}
    >
      ... (existing card inner markup, unchanged)
    </a>
  {/snippet}
</FolderGroups>

{#if loaded && workflows.length === 0}
  <Empty>
    {#snippet icon()}<Layers size={36} strokeWidth={1.5} />{/snippet}
    No workflows yet — the + above creates the first one.
  </Empty>
{:else if loaded && visible.length === 0}
  <p class="muted">Nothing matches "{filter}".</p>
{/if}
```

(imports: `FolderGroups`, `HintBar`, `Empty` from `../`, `Layers` from lucide; drop now-unused `ChevronDown`/`ChevronRight`/`X` imports. `Empty`'s message children go via the implicit `children` snippet, `icon` explicitly.)

Note: the snippet's `card` markup is styled by the page's own `<style>` block — snippets carry the defining component's style scope — so the card CSS stays where it is.

- [ ] **Step 3: Refactor PromptsPage identically**

Same surgery in `ui/src/lib/pages/PromptsPage.svelte`, with `collapseKey="collapsed-prompt-folders"`, `storageKey="prompt-hint-dismissed"`, `newHref="#/prompt-edit"`, `onnewingroup` writing `dw-prompt-editor-folder`, its own card snippet (overlay-anchor card with model/tag chips, unchanged markup), and an `Empty` with `MessageSquareText` icon: `No prompts yet — the + above creates the first one.`

- [ ] **Step 4: Adopt Empty on Jobs and Gallery**

Replace the hand-rolled `.empty` divs in `JobsPage.svelte` (Inbox icon) and `GalleryPage.svelte` (ImageOff icon) with the `Empty` component, deleting each page's `.empty` CSS rule. Messages and icons unchanged.

- [ ] **Step 5: Run gates + smoke**

Run: `cd ui && npm run check && npm run lint && npm test && npm run build && npx playwright test e2e/smoke.spec.ts`
Expected: all pass — the smoke suite exercises folder collapse (`e2e-scratch/` visibility) and card filtering, which must behave identically. Verify by hand that a folder collapsed before this change is still collapsed after (the storage keys are unchanged).

- [ ] **Step 6: Commit**

```bash
cd ui && npm run format && cd ..
git add ui/src/lib/FolderGroups.svelte ui/src/lib/HintBar.svelte ui/src/lib/Empty.svelte ui/src/lib/pages/WorkflowsPage.svelte ui/src/lib/pages/PromptsPage.svelte ui/src/lib/pages/JobsPage.svelte ui/src/lib/pages/GalleryPage.svelte
git commit -m "Shared FolderGroups/HintBar/Empty components unify the library pages"
```

---

### Task 6: Status-bar header

**Files:**
- Modify: `ui/src/App.svelte` (two-row header per the approved mockup)
- Modify: `ui/src/lib/pages/EditorPage.svelte` (sticky offsets that assumed a one-row header)

**Interfaces:**
- Consumes: existing `memory`/`currentJob`/`vramPct`/`theme` state in App.svelte.
- Produces: `header` = clean nav row (brand + labeled nav + theme button); below it a slim persistent `.statusbar` strip (running-job pill or idle note, VRAM text + meter, docs links). Both sticky as one block. e2e hook: the strip has class `statusbar`.

- [ ] **Step 1: Restructure the header markup**

In `ui/src/App.svelte`, keep `<header>` as the sticky container but give it two rows: move the theme button into the nav row and everything else into the strip.

```svelte
<header>
  <div class="navrow">
    <span class="brand">diffusers<span class="accent">-workflow</span></span>
    <nav>... (existing seven links, unchanged)</nav>
    <button class="quiet icon themebtn" onclick={cycleTheme} title="theme: {theme} - click to change" aria-label="theme: {theme} - click to change">
      ... (existing icon logic)
    </button>
  </div>
  <div class="statusbar">
    {#if currentJob}
      <a class="runningnow" href={'#/jobs/' + currentJob} title="a job is running - click to watch">
        <span class="pulse-dot"></span>running
      </a>
    {:else}
      <span class="muted idle">idle</span>
    {/if}
    <span class="flex"></span>
    <span class="vram muted">
      {#if memory?.info?.gpu_available}
        {memory.info.gpu_device_name} · {gb(memory.info.gpu_memory_allocated_mb ?? 0)} / {gb(memory.info.gpu_memory_total_mb ?? 0)} GB
      {:else}
        worker idle
      {/if}
    </span>
    {#if vramPct !== null}
      <div class="meter" title="VRAM allocated">
        <div class="fill" class:hot={vramPct > 75} class:critical={vramPct > 92} style:width={vramPct + '%'}></div>
      </div>
    {/if}
    <a class="helplink" href="https://github.com/dkackman/diffusers-workflow#documentation" target="_blank" rel="noopener" title="documentation on GitHub" aria-label="documentation on GitHub"><BookOpen size={14} /></a>
    <a class="helplink" href="/docs" target="_blank" rel="noopener" title="interactive API reference (OpenAPI)" aria-label="interactive API reference (OpenAPI)"><Braces size={14} /></a>
  </div>
</header>
```

The `.tools` div is gone. CSS changes in App.svelte's style block:

```css
header {
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  position: sticky;
  top: 0;
  z-index: 10;
}
.navrow {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem 1.5rem;
  padding: 0.7rem 1.2rem 0.5rem;
}
.statusbar {
  display: flex;
  align-items: center;
  gap: 0.6rem 1rem;
  padding: 0.25rem 1.2rem;
  border-top: 1px solid var(--line);
  font-size: 0.8rem;
  min-height: 1.6rem;
}
.statusbar .flex {
  flex: 1;
}
.idle {
  font-size: 0.8rem;
}
```

Delete the old `header { display: flex; ... }` properties that moved to `.navrow`, and the `.tools` rule. Keep `.brand`, `nav`, `.helplink`, `.runningnow`, `.vram`, `.themebtn`, `.meter` rules (adjust `.runningnow`/`.vram` font sizes if they now double-apply — the strip sets 0.8rem). In the two media queries, the `nav { order: 3; flex-basis: 100% }` rule now applies inside `.navrow`; keep the 640px icon-only fallback exactly as is.

- [ ] **Step 2: Fix the editor's sticky offsets**

The header grew by the strip (~26px). In `ui/src/lib/pages/EditorPage.svelte`: `.head { top: 50px }` → `top: 76px`; `.jsoncol { top: 110px }` → `top: 136px`. Verify visually in the browser (`npm run dev` or the built app): scroll a long workflow in the editor and confirm the toolbar pins just below the strip with no gap or overlap; adjust ±2px if the strip's real height differs.

- [ ] **Step 3: Run gates + full e2e**

Run: `cd ui && npm run check && npm run lint && npm test && npm run build && npx playwright test`
Expected: all pass — the theme-toggle smoke test targets the theme button by accessible name (unchanged) and responsive specs assert no horizontal overflow (the strip must flex-wrap: it doesn't — `flex-wrap` is deliberately off; if 375px overflows, allow wrap at the 640px media query: `.statusbar { flex-wrap: wrap; }`).

- [ ] **Step 4: Commit**

```bash
cd ui && npm run format && cd ..
git add ui/src/App.svelte ui/src/lib/pages/EditorPage.svelte
git commit -m "Header splits into a clean nav row and a slim status strip"
```

---

### Task 7: Keyboard help overlay

**Files:**
- Create: `ui/src/lib/KeyboardHelp.svelte`
- Modify: `ui/src/App.svelte` (global `?` handler + mount)

**Interfaces:**
- Consumes: nothing new.
- Produces: pressing `?` anywhere outside a text field opens a centered shortcuts sheet; `Escape` or a click outside closes it. e2e hooks: `role="dialog"` with `aria-label="keyboard shortcuts"`.

- [ ] **Step 1: Write the overlay**

```svelte
<!-- ui/src/lib/KeyboardHelp.svelte -->
<script lang="ts">
  let { open = $bindable(false) }: { open?: boolean } = $props()
</script>

{#if open}
  <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
  <div class="scrim" onclick={() => (open = false)}>
    <div
      class="sheet panel"
      role="dialog"
      aria-label="keyboard shortcuts"
      onclick={(e) => e.stopPropagation()}
    >
      <h2>Keyboard shortcuts</h2>
      <dl>
        <dt><kbd>Ctrl/⌘</kbd> + <kbd>S</kbd></dt>
        <dd>save (workflow &amp; prompt editors)</dd>
        <dt><kbd>Ctrl/⌘</kbd> + <kbd>Enter</kbd></dt>
        <dd>validate &amp; run (workflow editor)</dd>
        <dt><kbd>Esc</kbd></dt>
        <dd>close this help, the gallery drawer, and other panels</dd>
        <dt><kbd>?</kbd></dt>
        <dd>show this help</dd>
      </dl>
      <button class="quiet" onclick={() => (open = false)}>close</button>
    </div>
  </div>
{/if}

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: color-mix(in srgb, var(--bg) 65%, transparent);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 50;
  }
  .sheet {
    min-width: min(420px, 92vw);
    max-width: 480px;
  }
  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: var(--space-2) var(--space-4);
    margin: var(--space-4) 0;
    align-items: baseline;
  }
  dt {
    white-space: nowrap;
  }
  dd {
    margin: 0;
    color: var(--muted);
  }
  kbd {
    font-family: ui-monospace, 'Cascadia Code', monospace;
    font-size: 0.8rem;
    border: 1px solid var(--line);
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 0.05rem 0.4rem;
    background: var(--panel-2);
  }
</style>
```

- [ ] **Step 2: Global handler in App.svelte**

```ts
import KeyboardHelp from './lib/KeyboardHelp.svelte'

let helpOpen = $state(false)

function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target.isContentEditable
  )
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === '?' && !isEditable(event.target)) {
    event.preventDefault()
    helpOpen = true
  } else if (event.key === 'Escape' && helpOpen) {
    helpOpen = false
  }
}
```

Markup: `<svelte:window onkeydown={onKeydown} />` at the top level and `<KeyboardHelp bind:open={helpOpen} />` after the `<Toaster>`. (Monaco swallows its own keystrokes, so `?` typed in the JSON editor never reaches window — no special case needed; the `isEditable` check covers inputs/textareas/selects.)

- [ ] **Step 3: Run gates and commit**

Run: `cd ui && npm run check && npm run lint && npm test`
Expected: all pass.

```bash
cd ui && npm run format && cd ..
git add ui/src/lib/KeyboardHelp.svelte ui/src/App.svelte
git commit -m "Add ? keyboard-shortcuts overlay"
```

---

### Task 8: E2E coverage and full gates

**Files:**
- Create: `ui/e2e/chrome.spec.ts`

**Interfaces:**
- Consumes: Task 6's `.statusbar`, Task 7's dialog, Task 2's toast strings, Task 5's components (exercised through existing smoke assertions).

- [ ] **Step 1: Write the spec**

```ts
// ui/e2e/chrome.spec.ts
import { expect, test } from '@playwright/test'

test('the status strip carries worker state and docs, off the nav row', async ({
  page,
}) => {
  await page.goto('/')
  const strip = page.locator('header .statusbar')
  await expect(strip).toBeVisible()
  // worker state renders in the strip (fixture worker is idle at start)
  await expect(strip).toContainText(/idle|GB/)
  // docs links moved down out of the nav row
  await expect(
    strip.getByRole('link', { name: 'documentation on GitHub' }),
  ).toBeVisible()
  await expect(
    page.locator('header .navrow').getByRole('link', { name: 'Workflows' }),
  ).toBeVisible()
})

test('? opens the shortcuts overlay; Escape closes it; typing ? in a field does not', async ({
  page,
}) => {
  await page.goto('/')
  await page.keyboard.press('?')
  const dialog = page.getByRole('dialog', { name: 'keyboard shortcuts' })
  await expect(dialog).toBeVisible()
  await expect(dialog).toContainText('validate & run')
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  // inside a text field the character types instead
  const filter = page.getByPlaceholder('filter…')
  await filter.click()
  await filter.press('?')
  await expect(dialog).toHaveCount(0)
  await expect(filter).toHaveValue('?')
})

test('tab order walks the nav row in reading order', async ({ page }) => {
  await page.goto('/')
  // From the top of the document, Tab lands on the nav links in their
  // visual order - the baseline "rational tab order" check on the chrome
  await page.keyboard.press('Tab')
  await expect(
    page.getByRole('link', { name: 'Workflows' }).first(),
  ).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(page.getByRole('link', { name: 'Prompts' })).toBeFocused()
})

test('saving surfaces a toast, not a pinned banner', async ({ page }) => {
  test.setTimeout(60_000)
  await page.goto('/#/edit/ZImage')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  // success text arrives as a toast...
  await expect(page.getByText(/Saved to .*ZImage\.json/)).toBeVisible({
    timeout: 30_000,
  })
  // ...and auto-dismisses instead of pinning the page down
  await expect(page.getByText(/Saved to .*ZImage\.json/)).toHaveCount(0, {
    timeout: 10_000,
  })
})
```

- [ ] **Step 2: Run the new spec**

Run: `cd ui && npm run build && npx playwright test e2e/chrome.spec.ts`
Expected: PASS (3 tests). The implementation is authoritative on selectors — adjust selectors, never an assertion's intent.

- [ ] **Step 3: Full gate**

Run: `cd ui && npm run check && npm run lint && npm test && npx playwright test`
Expected: everything green (smoke, responsive, step-model, expanding-text, chrome).

- [ ] **Step 4: Commit**

```bash
cd ui && npm run format && cd ..
git add ui/e2e/chrome.spec.ts
git commit -m "E2E coverage for the status strip, help overlay, and toast feedback"
```

---

## Deliberate scope rulings

- A standalone `Card` component (named in the spec) is not extracted: the two pages' cards genuinely differ (plain anchor vs overlay-anchor with filter chips), so each page supplies its card as a snippet to `FolderGroups`; the duplicated *logic* (grouping, collapse persistence, filter-open, grid) is what actually drifted, and that is what the shared component absorbs.
- `PageHead` extraction (named in the spec) is skipped: the two library pages' head markup is already identical and three lines long; extraction adds indirection without fixing an actual drift. Recorded here so the final review can weigh it.
- Filter placement on Jobs/Gallery/Schema stays as-is this pass — each is already in the page head; unifying their exact markup is cosmetic and deferred.
- The spec's "validation errors map to step/field paths where possible": dangling-reference warnings already render inline (stage 2); server validation output has no structured step/field mapping to parse reliably, so the validation panel (now dismissible, the spec's "single dismissible panel" escape hatch) is the mapping target. Recorded as a ruling.

## After this plan

This completes the UI/UX optimization pass's planned stages. Remaining ideas live in the deferred checklist in `docs/superpowers/scope/UI-UX.md`.
