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
| 0 | UI ratchet and baseline; the gate reports count `.svelte`; fix the five rules the UI and the engine disagree on today (U1-U5) | `ui/scripts/arch-metrics.mjs --check` runs in CI and preflight against a committed baseline; U1-U5 fixed, each with a test that failed first; tagged `ui-stabilization-gate-0` | [phase-0.md](phase-0.md) | not started |
| 1 | One owner per rule: prefixes in `src/lib/references.ts` only; engine vocabularies read from the server or pinned; the server routes `dw_mcp` stage 2 needs, built once and used by both clients | `prefix_literals` 0; every UI copy of an engine rule is either gone or pinned by a test, and has a seam-map row | written at gate 0 | |
| 2 | Primitives: overlays and comboboxes on Bits UI, behind `src/lib/ui/` wrappers styled from `app.css` tokens | No hand-rolled focus trap, Escape chain or DOM sniffing for open dialogs; no false `aria-modal`; `<datalist>` gone; e2e green | written at gate 1 | |
| 3 | Structural moves: one editor shell under `EditorPage` and `PromptEditorPage`; `JobPage` split by job; one polling helper; `api.ts` de-duplicated and its cycle broken; shared layout styles | `files_over_size_ceiling` 0, `import_cycles` 0, `long_functions` 0; suite, e2e and a lem UI smoke green | written at gate 2 | |
| 4 | Contract and guardrails: response models on the routes the UI reads, generated TS types, e2e on PRs into develop, the harness ratchets the UI, `ui/CLAUDE.md` triaged | A server response change that breaks the UI fails CI; the harness refuses a UI ratchet rise without `arch-approved` | written at gate 3 | |

## Decisions

Recorded 2026-10-01; Don ruled on both open questions the same day.

- **Freeze (yes, Don 2026-10-01).** The engine pass froze features until its guardrails
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

(Filled at each gate.)

## Working rules for the duration

- A rule the engine owns is read from the server or pinned to its owner by a
  test; it is never re-derived in the UI.
- Every fix starts with a test that fails on today's code.
- No count-pinning assertions; baseline numbers live in `baseline.json`.
- Comments explain why the code is the way it is, not the ticket history.
- A ratchet only rises in a commit that names the rise and why, and the
  harness requires `arch-approved` for it, as for the engine.
