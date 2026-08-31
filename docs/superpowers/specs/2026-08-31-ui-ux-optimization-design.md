# UI/UX Optimization Pass — Design

**Date:** 2026-08-31
**Scope source:** `docs/superpowers/scope/UI-UX.md`
**Scope:** UI only (`ui/`). No API/backend changes, no schema changes, no new features.

## Goal

Resolve the three named pain points — binary information density, implicit step
sequencing, invisible output→input flow — with an editor-led rework, plus a lighter
consistency/navigation sweep across the rest of the app. Guiding phrase: *make the
simple easy and the more complex possible.*

## Decisions made during brainstorming

| Topic | Decision |
| --- | --- |
| Focus | Editor-led, app-wide consistency sweep |
| Foundations | Tokens + shared-component extraction in scope, each justified by a visible fix |
| Sequence/flow UI | Ordinal rail + producer/consumer chips with hover highlight ("B"); flow-strip minimap ("C") deferred |
| Flow shape | Execution is linear but data flow is a fan-in/fan-out DAG (e.g. 3 image steps → 1 video step). All flow UI must handle multiple inputs/consumers, never assume a chain |
| Density | Three-state steps: collapsed / compact (textual digest) / full |
| Knob discovery | Full view shows populated args only; "show all available (N)" reveals introspected unpopulated knobs with descriptions |
| Header | Slim status bar: clean nav row + persistent strip (job, VRAM, docs) |
| Feedback | Toasts for success/status, inline errors/validation; banner stack removed; native `confirm` stays (undo layer deferred) |
| Keyboard | Consistent core (Ctrl+S / Ctrl+Enter / Escape) + `?` help overlay + tab-order verification |

## Workstreams (priority order)

### P1 — Editor step model

1. **Rich collapsed summary.** A collapsed step's title bar shows: name, kind badge,
   and a one-line digest (pipeline class · model · arg count). Collapsing never loses
   all information.
2. **Three-state steps** — collapsed / compact / full, cycled per step, with
   expand-all/collapse-all controls; state persisted per workflow.
   - *Compact* is a read-mostly textual digest of what is **set**: model, dtype, key
     arguments, and section summaries ("transformer (nf4) · 2 LoRAs · UniPC ·
     offload: model"). Clicking any digest value switches the step to full and
     focuses the corresponding field.
   - *Full* is approximately today's everything-view, minus noise (see 4).
   - New steps open full; existing steps default to compact.
3. **Ordinal rail + producer/consumer chips.**
   - Numbered ordinal badges on a vertical connecting rail communicate "ordered
     sequence" explicitly.
   - Consumer steps show an input chip per producer: `← from gen1, gen2, gen3`.
   - Producer steps show `→ used by combine` (one chip per consumer).
   - Hovering a chip highlights the partner step(s).
   - Reordering that breaks a `previous_result:` reference warns inline on the
     affected step (replacing the distant prose banner).
   - A step with multiple `previous_result` references gets a cartesian-product
     note (e.g. "3 inputs × 4 images = 12 iterations").
4. **Knob discovery on demand.** The introspection-driven arguments editor shows
   populated arguments by default; a "show all available (N)" disclosure reveals
   unpopulated knobs with their descriptions for browsing/selection.

### P2 — Feedback layer

- Remove the stacked banner strip in `EditorPage.svelte` (error / dangling refs /
  status / validation).
- Success and transient status → auto-dismissing toasts (~4 s); errors sticky until
  dismissed.
- Validation errors map to step/field paths where possible and render inline at the
  offending step/field; unmappable errors go to one dismissible panel.
- Dangling-reference warnings render on the offending step's chip area.
- Native `window.confirm` remains for destructive actions this pass.

### P3 — Foundations in service of consistency

- **Tokens** in `app.css`: `--space-1..5`, `--radius-1..2`; named container-query
  breakpoint scale 400 / 640 / 900 (comment-enforced convention, since `@container`
  thresholds can't read custom properties). Pages migrate as touched — no big-bang.
- **Shared components**, each extracted only to fix a visible drift:
  - `Card` + `FolderGroup` — unify Workflows ↔ Prompts list pages (currently
    copy-forked and drifted).
  - `PageHead` (title/filter placement), `Empty` (empty states), toast outlet.
- **Jobs run page**: stop flattening per-step output files; group results by
  producing step (runtime mirror of the editor's flow story; uses data already in
  SSE `step_end` events).
- **Storage wrapper**: one namespaced module absorbing the scattered `dw-*`
  localStorage/sessionStorage keys.

### P4 — Chrome & keyboard

- **Header**: clean nav row (brand + labeled nav + theme) plus a slim persistent
  status strip below it (running job, VRAM readout + meter, docs links).
- **Keyboard**: Ctrl/Cmd+S and Ctrl/Cmd+Enter wherever save/run exist; Escape closes
  drawers/panels uniformly; `?` opens a shortcut-help overlay; tab order verified on
  the main flows.

## Key mechanics

- **`flowGraph(workflow)`** (in `ui/src/lib/editor.ts` or a sibling module): single
  source of truth returning per-step `{inputs, consumers}` edge lists, derived from
  `previous_result:` references (superseding the implicit parsing in
  `referenceSuggestions()` / `danglingReferences()`). Drives chips, hover
  highlighting, reorder warnings, cartesian notes, and (later) the minimap.
  Unit-tested against fan-in, fan-out, and suffix (`.frames`/`.audio`) cases.
- **`stepDigest(step)`**: pure function producing the compact view's text lines from
  step JSON; its first line doubles as the collapsed summary.
- **Toast store**: small Svelte store + `<Toasts>` outlet in `App.svelte`.
- **View-state persistence**: per-step state keyed by workflow name via the storage
  wrapper.

## Testing

- Vitest units: `flowGraph`, `stepDigest`, toast store.
- Playwright e2e additions: step state cycling, chip hover-highlight, toast
  lifecycle, keyboard shortcuts + help overlay.
- Existing gates stay green: `npm run check`, `npm run lint`, `npm test`,
  `npx playwright test` (incl. `responsive.spec.ts`).

## Build order

1. **Foundations-lite** — tokens, storage wrapper, toast store.
2. **Editor step model** — `flowGraph` + `stepDigest` (tests first) → collapsed
   summary → three-state steps → rail + chips + inline warnings → knob discovery.
3. **Feedback migration** — editor first, then WorkflowPage, PromptEditorPage,
   ModelsPage.
4. **Consistency sweep** — Card/FolderGroup/Empty extraction, Jobs step-grouped
   results, filter/empty-state alignment.
5. **Chrome & keyboard** — status-bar header, shortcut unification, `?` overlay,
   tab-order pass.

Each stage is independently shippable.

## Risks

- Stage 2 rewrites much of `StepEditor.svelte` (the largest single change). Bounded
  by the three-state model being additive: full view ≈ today's markup. Monaco/JSON
  view untouched.
- Strictly `ui/` — no API, schema, or execution-logic changes anywhere.

## Out of scope (recorded in the scope doc's deferred list)

Flow-strip minimap, undo layer + styled confirm dialogs, list-page keyboard
navigation, a11y compliance, mobile as a target.
