i like the direction the ui has been going and now i want to spend some time on optimizing ui/ux.

## process

before making changes, audit the current UI against the guiding principles below and propose a
prioritized plan (what to change, why, expected impact). don't refactor speculatively — flag
issues, then make changes deliberately once we've agreed on priority.

## scope

this pass is UI/UX only: layout, interaction patterns, information architecture, visual design,
responsiveness. no new features, no API/backend changes, no changes to underlying node execution
logic unless a UI change requires it.

## primary users

people put off by ComfyUI's complexity who want something easy to understand and straightforward.
they want to explore and adopt new models without needing complex branching workflows, but still
want access to the underlying configuration points when they need them.

catchphrase: **make the simple easy and the more complex possible.**

## guiding principles

- simplicity and ease of use
  - in tension with the space's inherent complexity and the total API surface area
  - minimize cognitive load: progressive disclosure over showing everything at once
- consistency across the platform — no surprises navigating between parts of the app
- ease of navigation and discoverability — users find what they need quickly and intuitively
- responsiveness and performance — the UI should feel fast, primarily as a desktop application.
  mobile/small screens are not a target, but the layout should degrade gracefully (best-effort
  responsiveness) rather than break outright. accessibility (a11y) is not a target for this pass.

## known pain points to solve

- information density is binary — collapsed shows almost nothing, expanded shows everything, with
  no middle ground. need intermediate disclosure levels.
- workflows are linear (step 1, 2, 3 — no branching, limited iteration, which is intentional), but
  the vertical top-to-bottom layout doesn't clearly communicate "this is an ordered sequence."
  users shouldn't have to infer the order from position alone.
- output-to-input flow between steps isn't visible. users can't easily see how a given step's
  output feeds into later steps.

## guardrails

- avoid clutter and elements that don't contribute to the user experience
- use ComfyUI as a comparison — not for what to copy, but for how to offer an easier alternative
  to learn and use. keep what works, avoid what doesn't.
- maintain standard UI/UX best practices throughout: keyboard shortcuts, rational tab order,
  focus states, and other baseline interaction conventions — even though full accessibility
  compliance is out of scope for now.

## innovation

explore new interaction patterns and design paradigms that enhance the experience, but always
weigh them against the guiding principles above.

## deferred / even better if

living list — items consciously deferred from the current pass, plus "even better if"
ideas that surface during implementation. append as they come up.

- [ ] flow-strip minimap above the editor's step list: clickable mini-DAG of the whole
      pipeline (must render fan-in/fan-out honestly, not a chain)
- [ ] undo layer: replace native `confirm` with app-styled dialogs; toast-with-Undo for
      step/argument/variable deletes
- [ ] list-page keyboard navigation (arrows/enter on cards, j/k on jobs) and step-level
      editor shortcuts (collapse all, move step)
- [ ] full a11y compliance pass
- [ ] mobile as a first-class target
- [ ] compact digest click focuses the exact field (currently opens the section only; main/arguments clicks just switch to full)
- [ ] cartesian note computes a best-effort multiplier when producers declare num_images_per_prompt (currently generic)
