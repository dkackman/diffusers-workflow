# UI Phase 2: Primitives on Bits UI - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The UI's overlays and suggestion inputs run on Bits UI behind one wrapper layer, so focus, Escape, outside clicks and scroll lock are handled once and correctly, and the hand-built focus trap, Escape chain, DOM sniffing and false `aria-modal` are gone.

**Architecture:** `bits-ui` (2.19.x, headless, Svelte 5) is imported only under `ui/src/lib/ui/`, and an ESLint rule enforces that. Four wrappers live there: `ConfirmDialog.svelte` (AlertDialog), `Modal.svelte` (Dialog), `Popover.svelte` (Popover, anchored with `customAnchor`) and `Suggest.svelte` (Combobox used as free text with suggestions). A small `layers.svelte.ts` counts open overlays and replaces `dialogOpen()`'s DOM query. Overlay styling moves to `app.css` (`.ui-overlay`, `.ui-sheet`, `.ui-popover`, `.ui-suggest`) on one z-index scale, because a component's scoped styles do not reach the elements Bits renders.

**Tech Stack:** Svelte 5 (runes), bits-ui 2.19, TypeScript, Vitest + @testing-library/svelte (jsdom), Playwright.

**Spec:** [ROADMAP.md](ROADMAP.md) (Phase 2 row; *Decisions*: Bits UI approved, pilot first, wrappers in `src/lib/ui/`), [ASSESSMENT.md](ASSESSMENT.md).

## Global Constraints

- Freeze holds until this gate: no new UI features. Behaviour each widget has today is kept unless a task says otherwise. Gate 2 lifts the freeze (ROADMAP *Decisions*).
- `bits-ui` is imported only in `ui/src/lib/ui/**`. Pages and other components import the wrappers.
- Accessible names the e2e suite selects by stay as they are: `alertdialog` for the confirm, `dialog` named `keyboard shortcuts` and `server status`, and the token popover's label.
- The UI ratchet must not rise. `a11y_suppressions` falls (the four scrim-click suppressions go); rewrite the baseline in that commit.
- Every new test fails first. Commands from `ui/`: `npx vitest run <file>`, `npm run check`, `npm run lint`, `npx prettier --write <files>`, `npm run metrics -- --check ../docs/stabilization/ui/baseline.json`, `npx playwright test` (needs `DW_E2E_PYTHON` set to the repo venv's python).
- Commits: conventional prefix; end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Branch `ui-stabilization/phase-2` from develop.

## Review Focus

1. Pressing Escape with the confirm open over the gallery closes the confirm and leaves the gallery's selection alone; a second Escape then clears the selection (Task 4).
2. Clicking a popover's trigger while it is open closes it and does not reopen it on the same click (Task 3).
3. Typing a value not in a `Suggest` list keeps that value, and choosing a suggestion replaces the typed text with it (Task 5).
4. A `Suggest` with an empty suggestion list behaves as a plain input: no empty popup (Task 5).
5. After a dialog closes, focus returns to the element that opened it (Tasks 1-3).

---

### Task 1: Install Bits UI; the confirm dialog on AlertDialog (pilot)

**Files:**
- Modify: `ui/package.json`, `ui/package-lock.json` (`bits-ui`)
- Create: `ui/src/lib/ui/layers.svelte.ts`, `ui/src/lib/ui/ConfirmDialog.svelte`
- Delete: `ui/src/lib/ConfirmDialog.svelte`
- Modify: `ui/src/App.svelte` (import path), `ui/src/app.css` (overlay classes, layer scale), `ui/eslint.config.js` (restricted import)
- Test: `ui/src/lib/ui/ConfirmDialog.test.ts`, `ui/src/lib/ui/layers.test.ts`
- Possibly modify: `ui/vitest.setup.ts` (jsdom stubs, only if a test shows Bits needs them)

**Interfaces:**
- Produces: `holdLayer(): () => void` (count up; the returned function counts down once) and `overlayOpen(): boolean` from `ui/src/lib/ui/layers.svelte.ts`.
- Produces: `ui/src/lib/ui/ConfirmDialog.svelte`, rendering `confirmState` from `ui/src/lib/confirm.svelte.ts` (unchanged API: `confirmDialog()`, `resolveConfirm()`).

- [ ] **Step 1: Install**

Run: `cd ui && npm install bits-ui@^2.19.4`
Expected: `bits-ui` appears under `dependencies`, and `@internationalized/date` arrives as its peer.

- [ ] **Step 2: Write the failing tests**

`ui/src/lib/ui/layers.test.ts`:

```ts
import { expect, it } from 'vitest'
import { holdLayer, overlayOpen } from './layers.svelte'

it('counts open overlays and releases each once', () => {
  expect(overlayOpen()).toBe(false)
  const releaseA = holdLayer()
  const releaseB = holdLayer()
  releaseA()
  releaseA()
  expect(overlayOpen()).toBe(true)
  releaseB()
  expect(overlayOpen()).toBe(false)
})
```

`ui/src/lib/ui/ConfirmDialog.test.ts`:

```ts
import { afterEach, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte'
import ConfirmDialog from './ConfirmDialog.svelte'
import { confirmDialog } from '../confirm.svelte'
import { overlayOpen } from './layers.svelte'

afterEach(() => cleanup())

it('asks, and answers true on the confirm button', async () => {
  render(ConfirmDialog)
  const answer = confirmDialog('Delete it?', { confirmLabel: 'Delete' })
  const dialog = await screen.findByRole('alertdialog')
  expect(dialog.textContent).toContain('Delete it?')
  expect(overlayOpen()).toBe(true)
  await fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
  expect(await answer).toBe(true)
  await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
  await waitFor(() => expect(overlayOpen()).toBe(false))
})

it('answers false on Cancel and on Escape', async () => {
  render(ConfirmDialog)
  const cancelled = confirmDialog('Delete it?')
  await fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }))
  expect(await cancelled).toBe(false)

  const escaped = confirmDialog('Delete it?')
  const dialog = await screen.findByRole('alertdialog')
  await fireEvent.keyDown(dialog, { key: 'Escape' })
  expect(await escaped).toBe(false)
})

it('moves focus into the dialog', async () => {
  render(ConfirmDialog)
  confirmDialog('Delete it?')
  const dialog = await screen.findByRole('alertdialog')
  await waitFor(() => expect(dialog.contains(document.activeElement)).toBe(true))
})
```

- [ ] **Step 3: Run them to see them fail**

Run: `cd ui && npx vitest run src/lib/ui`
Expected: FAIL (no modules).

- [ ] **Step 4: The layer count**

`ui/src/lib/ui/layers.svelte.ts`:

```ts
/** How many overlays are open. A page's own Escape handler (clearing a
 * selection, closing a detail) asks this first, so the Escape that closes
 * an overlay does not also act on the page beneath it. Each wrapper holds
 * a layer while it is open. */
let open = $state(0)

export function holdLayer(): () => void {
  open += 1
  let held = true
  return () => {
    if (held) {
      held = false
      open -= 1
    }
  }
}

export function overlayOpen(): boolean {
  return open > 0
}
```

- [ ] **Step 5: The confirm on AlertDialog**

`ui/src/lib/ui/ConfirmDialog.svelte`:

```svelte
<script lang="ts">
  import { AlertDialog } from 'bits-ui'
  import { confirmState, resolveConfirm } from '../confirm.svelte'
  import { holdLayer } from './layers.svelte'

  $effect(() => {
    if (confirmState.open) return holdLayer()
  })
</script>

<!-- Escape, an outside click and Cancel all answer false; Bits reports
     each as onOpenChange(false). The confirm button answers true. -->
<AlertDialog.Root
  open={confirmState.open}
  onOpenChange={(open) => {
    if (!open) resolveConfirm(false)
  }}
>
  <AlertDialog.Portal>
    <AlertDialog.Overlay class="ui-overlay" />
    <AlertDialog.Content
      class="ui-sheet panel"
      aria-label="confirm"
      interactOutsideBehavior="close"
    >
      <AlertDialog.Description class="ui-message">
        {confirmState.message}
      </AlertDialog.Description>
      <div class="ui-actions">
        <AlertDialog.Cancel class="quiet">{confirmState.cancelLabel}</AlertDialog.Cancel>
        <AlertDialog.Action onclick={() => resolveConfirm(true)}>
          {confirmState.confirmLabel}
        </AlertDialog.Action>
      </div>
    </AlertDialog.Content>
  </AlertDialog.Portal>
</AlertDialog.Root>
```

In `ui/src/App.svelte`, change the import to `import ConfirmDialog from './lib/ui/ConfirmDialog.svelte'` and delete `ui/src/lib/ConfirmDialog.svelte`.

- [ ] **Step 6: Overlay styles, one layer scale**

Append to `ui/src/app.css` (values taken from the old components; the scale replaces the scattered literals 30/50/60 the overlays used):

```css
/* One stacking scale for everything above the page. Bits renders overlays
   in a portal on <body>, where a component's scoped styles cannot reach,
   so their look lives here. */
:root {
  --layer-sticky: 10;
  --layer-drawer: 20;
  --layer-popover: 30;
  --layer-modal: 60;
}
.ui-overlay {
  position: fixed;
  inset: 0;
  background: color-mix(in srgb, var(--bg) 65%, transparent);
  z-index: var(--layer-modal);
}
.ui-sheet {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  min-width: min(360px, 92vw);
  max-width: 440px;
  z-index: var(--layer-modal);
}
.ui-message {
  margin: 0 0 var(--space-4);
  white-space: pre-line;
}
.ui-actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-2);
}
.ui-popover {
  z-index: var(--layer-popover);
}
```

- [ ] **Step 7: Bits only behind the wrappers**

In `ui/eslint.config.js`, add:

```js
  {
    files: ['src/**/*.{ts,svelte}'],
    ignores: ['src/lib/ui/**'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          paths: [
            {
              name: 'bits-ui',
              message: 'Import a wrapper from src/lib/ui/ - Bits UI stays behind it.',
            },
          ],
        },
      ],
    },
  },
```

- [ ] **Step 8: Run the tests; add jsdom stubs only if Bits needs them**

Run: `cd ui && npx vitest run src/lib/ui`
If a test fails because jsdom lacks an API Bits calls (for example `ResizeObserver`, `Element.prototype.scrollIntoView`, `hasPointerCapture`), add only the missing stubs to `ui/vitest.setup.ts`, each with a comment saying which Bits behaviour needs it:

```ts
// Bits UI measures and observes its floating layers; jsdom has neither
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver
Element.prototype.scrollIntoView ??= function () {}
Element.prototype.hasPointerCapture ??= () => false
```

Expected after any stubs: PASS.

- [ ] **Step 9: Whole suite, checks, ratchet, e2e**

Run: `cd ui && npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json && npx playwright test e2e/smoke.spec.ts e2e/workspaces.spec.ts`
Expected: all pass (the e2e specs select `alertdialog`). `a11y_suppressions` falls by one: rewrite the baseline.

- [ ] **Step 10: Commit**

```bash
git add ui/package.json ui/package-lock.json ui/src ui/eslint.config.js ui/vitest.setup.ts docs/stabilization/ui/baseline.json
git commit -m "feat(ui): Bits UI behind src/lib/ui; the confirm dialog on AlertDialog"
```

---

### Task 2: The keyboard help on Modal

**Files:**
- Create: `ui/src/lib/ui/Modal.svelte`
- Modify: `ui/src/lib/KeyboardHelp.svelte` (renders through `Modal`), `ui/src/App.svelte` (drop the help's Escape branch)
- Test: `ui/src/lib/ui/Modal.test.ts`

**Interfaces:**
- Produces: `Modal.svelte` with props `open` (bindable), `label: string` (the accessible name), `children` snippet. Escape, an outside click and a `close` inside call `open = false`; it holds a layer while open.

- [ ] **Step 1: Failing test** (`Modal.test.ts`): render a small harness component, or `KeyboardHelp` itself, with `open: true`. Assert:
  - `screen.getByRole('dialog', { name: 'keyboard shortcuts' })` exists;
  - focus is inside it;
  - `overlayOpen()` is true;
  - Escape on the dialog closes it (the role disappears; `overlayOpen()` goes false);
  - `aria-modal` is `"true"`, which is right for a modal.

Run: FAIL (no module).

- [ ] **Step 2: Implement** `Modal.svelte`:

```svelte
<script lang="ts">
  import type { Snippet } from 'svelte'
  import { Dialog } from 'bits-ui'
  import { holdLayer } from './layers.svelte'

  let {
    open = $bindable(false),
    label,
    children,
  }: { open?: boolean; label: string; children: Snippet } = $props()

  $effect(() => {
    if (open) return holdLayer()
  })
</script>

<Dialog.Root bind:open>
  <Dialog.Portal>
    <Dialog.Overlay class="ui-overlay" />
    <Dialog.Content class="ui-sheet panel" aria-label={label}>
      {@render children()}
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>
```

`KeyboardHelp.svelte` keeps its content (heading, `dl`, close button) inside `<Modal bind:open label="keyboard shortcuts">`, drops `focusTrap`, the scrim markup, its suppression comment and its scrim/sheet styles (keeping only the `dl`/`kbd` styles).

In `App.svelte`, delete the `else if (event.key === 'Escape' && helpOpen)` branch; `?` still opens the help.

- [ ] **Step 3:** Run `npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json && npx playwright test e2e/chrome.spec.ts`. PASS. Rewrite the baseline (`a11y_suppressions` falls). Commit `feat(ui): the keyboard help on the Modal wrapper`.

---

### Task 3: The status and token popovers on Popover

Both declare `aria-modal="true"` and trap focus while behaving as
non-modal popovers that close on an outside click. On Bits Popover they
are non-modal, anchored to their trigger and dismissed by Escape and an
outside click.

**Files:**
- Create: `ui/src/lib/ui/Popover.svelte`
- Modify: `ui/src/lib/StatusPopover.svelte`, `ui/src/lib/TokenPopover.svelte`, `ui/src/App.svelte` (anchor element refs; drop the two Escape branches; drop `stopPropagation` on the triggers if it was only there for the window-click close)
- Test: `ui/src/lib/TokenPopover.test.ts` (rewrite), `ui/src/lib/ui/Popover.test.ts`

**Interfaces:**
- Produces: `Popover.svelte` with props `open` (bindable), `label: string`, `anchor: HTMLElement | null` (passed as Bits `customAnchor`), `children` snippet. It is non-modal (no `aria-modal`), focus moves in on open and back to the trigger on close, and it holds a layer while open. The status popover has two triggers, so the triggers stay plain buttons in `App.svelte` that toggle `open`, and the anchor is the header element they sit in.

- [ ] **Step 1: Failing tests.**
  - Rewrite `TokenPopover.test.ts`: open → `getByRole('dialog', { name: <its label> })` exists; `getAttribute('aria-modal')` is not `"true"`; focus is inside; Escape closes it.
  - `Popover.test.ts`: an outside `pointerdown`/click on `document.body` closes it.
  - Run: FAIL (the `aria-modal` assertion, then the missing module).
- [ ] **Step 2: Implement** `Popover.svelte` on `Popover.Root bind:open` and `Popover.Portal` > `Popover.Content class="ui-popover panel" aria-label={label} customAnchor={anchor} side="bottom" align="end" sideOffset={6}`, with `$effect(() => { if (open) return holdLayer() })`. Rewrite both popovers' markup inside it. Delete their `svelte:window onclick`, `focusTrap`, `aria-modal`, suppression comments and positioning CSS (the anchoring replaces the absolute positioning). In `App.svelte`, bind the header element (`bind:this`) and pass it as `anchor`; delete the `statusOpen` and `tokenOpen` Escape branches.
- [ ] **Step 3:** Run `npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json && npx playwright test e2e/chrome.spec.ts`. PASS. Rewrite the baseline. Commit `fix(ui): the status and token popovers are non-modal popovers anchored to their trigger`.

---

### Task 4: Pages ask the layer count; the hand-built focus trap goes

**Files:**
- Modify: `ui/src/lib/picks.svelte.ts` (delete `dialogOpen`), `ui/src/lib/pages/GalleryPage.svelte`, `ui/src/lib/pages/AssetsPage.svelte` (use `overlayOpen` from `../ui/layers.svelte`)
- Delete: `ui/src/lib/focusTrap.ts`
- Test: `ui/src/lib/picks.test.ts` (drop the `dialogOpen` cases), `ui/src/lib/pages/GalleryPage.test.ts`

- [ ] **Step 1: Failing test** in `GalleryPage.test.ts`:
  - pick an item;
  - open the confirm (`confirmDialog(...)` with `ConfirmDialog` rendered beside the page);
  - press Escape on the dialog: the dialog closes and the pick remains;
  - press Escape on the window again: the pick clears.

  Run: FAIL, or pass for the wrong reason, which is then proven by deleting `dialogOpen`'s use. Record which.
- [ ] **Step 2: Implement:** replace `dialogOpen()` with `overlayOpen()` at both page call sites, delete `dialogOpen` and its tests, delete `focusTrap.ts` (no importer remains; `npm run check` proves it).
- [ ] **Step 3:** Run `npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json && npx playwright test`. PASS. Commit `refactor(ui): pages ask the overlay layer count; the hand-built focus trap goes`.

`App.svelte` keeps one Escape branch, for the mobile drawer: the drawer is a
layout region with correct `inert` handling, not an overlay, and is not
converted in this phase (ROADMAP: overlays and comboboxes).

---

### Task 5: `Suggest` replaces every `<datalist>`

**Files:**
- Create: `ui/src/lib/ui/Suggest.svelte`
- Modify: the 12 `list=` inputs and 13 `<datalist>` blocks in `ui/src/lib/pages/EditorPage.svelte`, `PromptEditorPage.svelte`, `WorkflowPage.svelte`, `ui/src/lib/editor/StepEditor.svelte`, `ComponentEditor.svelte`, `MappingEditor.svelte`, `ArgumentsEditor.svelte`, `ExpandingText.svelte`, `QuantizationEditor.svelte` (`grep -rn 'list=\|<datalist' ui/src` lists them)
- Test: `ui/src/lib/ui/Suggest.test.ts`

**Interfaces:**
- Produces: `Suggest.svelte` with props `value` (bindable string), `suggestions: string[]`, `id?`, `placeholder?`, `title?`, `onchange?: (value: string) => void`, and any other input attributes passed through. The typed text is the value. A suggestion is an offer: choosing one replaces the text. With no matching suggestions there is no popup. At most 50 suggestions show, filtered by case-insensitive substring.

- [ ] **Step 1: Failing tests** (`Suggest.test.ts`):
  - typing `Flux` with suggestions `['FluxPipeline', 'ZImagePipeline']` shows one option;
  - choosing it (click, or ArrowDown and Enter) sets the value to `FluxPipeline` and calls `onchange` with it;
  - typing `MyOwnPipeline` (no match) keeps that value, shows no listbox and calls `onchange` on change;
  - an empty suggestions list never renders a listbox;
  - the input keeps `id`, so a `<label for>` still names it (`getByLabelText`).

  Run: FAIL (no module).
- [ ] **Step 2: Implement** on `Combobox.Root type="single"` with `bind:open`, `inputValue` driven from `value`, `onValueChange` writing the chosen item into `value`, `Combobox.Input` writing typed text into `value` on input, and `Combobox.Portal` > `Combobox.Content class="ui-suggest"` listing the filtered items. `open` is true only while the filtered list is non-empty. Add `.ui-suggest` (panel look, `z-index: var(--layer-popover)`, `max-height` with scroll) to `app.css`.
- [ ] **Step 3: Convert each site,** one file per commit if the diff is large. Keep each site's existing binding (`bind:value` or `value` + `onchange`) and its suggestion source: the array the `<datalist>` iterated becomes `suggestions`. Delete each `<datalist>`. `PROMPT_LIST_ID` and `promptListId` in `prompts.ts` lose their reason: the prompt suggestions become `suggestions` on the inputs that offered them. Delete both, and their tests.
- [ ] **Step 4:** Run `npx vitest run && npm run check && npm run lint && npm run metrics -- --check ../docs/stabilization/ui/baseline.json && npx playwright test`. PASS. Confirm `grep -rn '<datalist\|list=' ui/src` finds nothing. Commit `feat(ui): Suggest replaces every datalist`.

---

### Task 6: Seam-map rows and the gate

**Files:** `docs/ARCHITECTURE.md` (UI rows), `docs/stabilization/ui/ROADMAP.md` (gate report, Phase 2 status, freeze lifted)

- [ ] **Step 1:** Add map rows:
  - **UI overlays and suggestions:** `ui/src/lib/ui/`; Bits UI is imported nowhere else (`ui/eslint.config.js` `no-restricted-imports`).
  - **Escape on a page under an overlay:** `ui/src/lib/ui/layers.svelte.ts`; enforced by `GalleryPage.test.ts`'s Escape case.
- [ ] **Step 2:** `scripts/preflight.sh` (pytest with `DW_DEVICE=cpu`, integration without it), both ratchets clean.
- [ ] **Step 3:** One fresh-reviewer pass over the branch, with this plan's Review Focus; Critical and Important fixed test-first.
- [ ] **Step 4:** With Don's word: merge, push develop, deploy to lem. Browser smoke in headless Chromium against lem:
  - `?` opens the help, and Escape closes it;
  - the status popover opens from both triggers and closes on an outside click;
  - a delete confirm on a scratch asset opens, and Cancel keeps the asset;
  - an editor class field offers suggestions and keeps typed text.
- [ ] **Step 5:** Gate report in `ROADMAP.md` (ratchet table column, what the smoke showed, deferred minors), Phase 2 status done, freeze lifted. Tag `ui-stabilization-gate-2` (annotated), and push the tag on Don's word.
