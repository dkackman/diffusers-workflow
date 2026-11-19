# The run page validates and shows the plan before Run (#1446)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the ask on 2026-11-18, from a UI support assessment of
v0.19.0..develop. The workflow page's `run()` submitted straight to the queue, and
the plan a validate call answers with (estimate and basis, host-RAM projection,
required downloads, warnings) showed only in the editor. Several plan fields the
server returns weren't read by any UI code. Plan v1 is the plan comment on #1446,
and it still holds the full sweep. This file records what was built, what was cut
and what stays.

## Verdict: build smaller (two UI-only stages, no GPU)

- **Cut:** comparing the host-RAM need **against free RAM**. Free host RAM is in no
  API response (`HealthInfo` has none; `WorkerInfo.host_memory_rss_mb` is the worker
  process). The plan shows the projected need and its basis, and the server's own
  over-ceiling warning says when it's too much.
- **Cheaper alternatives rejected:** a one-shot "Check" button (easy to skip, and it
  still needs stage A), and extending `describePlan` alone (it leaves the run page
  blind). The button stays the fallback if the debounce proves noisy.

## Plan history

- **v1 (2026-11-18).** Batch lane, every question `[none]`. The advisor approved it on
  Don's behalf with three minor adjustments, folded into stage B's body without a new
  version:
  - the delayed `sizes=true` call falls under the same stale-drop;
  - acceptance 2 had to check that validate really refuses an out-of-range
    `num_images_per_prompt`. It doesn't (`99` is valid), so the cases use a wrong
    type (`"abc"`);
  - whichever of this and #1447 lands second keeps the other's `VariablesForm` prop.

**Rulings (advisor, all defaults):**
- Q1 [none]: validate 600 ms after the last change with `sizes=false`, plus one sized
  call after 3 s idle, so the hub isn't called per keystroke.
- Q2 [none]: show the plan tag and the server's warning both, filtering neither. Plan
  lines say *what*; warnings say *why*.
- Q3 [none]: Run stays enabled whatever validate says. The server stays the gate.

## What was built

### Stage A (#1467): `describePlan` reads the unread fields; the editor shows a plan whenever one is present

Merged to `develop` at `8a810c54` (branch `feat/1446-A-describeplan-fields`, `ce8d0ec1`),
then the bounce fix at `2cb71db8` (branch `feat/1446-A-priced-for-everywhere`,
`9d301f46`).

- **Host memory:** `host RAM ~N GB (observed | fit to N runs | declared | inherited
  from the template)`, plain, right after the figure line. N is `projected_mb / 1024`
  to one decimal.
- **Estimate qualifiers** (`ui/src/lib/plan.ts`):
  - `low_confidence`: the figure line ends `, low confidence` and goes `warn`, on
    every priced basis.
  - `tempered`: `, blended toward the curated M min from K runs at O min`.
  - `priced_for`: names the card when it differs from `device`. In the "on X" form it
    replaces `device`. On the inherited, withheld and unpriced lines it goes last, as
    ` - for <card>`.
  - withheld (basis `unknown` with `inherited_from`): `no figure - withheld (not
    priced from X)`, `warn`.
  - basis `inherited`: `~N min, inherited from X`, plain, or `warn` while
    `low_confidence`.
  - `unpriced_reason`: `no figure - <reason>`, `warn`, in place of "no measured cost".
- **Downloads:** `X (gated: this box's token can't reach it)` and `X (not found)`,
  after the size when it's known. The line is `warn`.
- **Types:** a `PlanHostMemory` alias in `ui/src/lib/types.ts`. No regen was needed.
- **Editor:** `ValidationPanel` shows the plan whenever one is present, under the
  errors when the answer is invalid (a `not_found` answer keeps its plan).
- **Tests:** new `plan.test.ts` cases per field and tone, with every earlier case
  unchanged; a new `ValidationPanel.test.ts` (three cases).
- **Docs:** `docs/SERVER.md`, the editor paragraph.
- **Things to know:** a live server sets `priced_for` on every plan, so figure lines
  name the card label wherever it differs from `device`. It sets `low_confidence` on
  every basis-`derived` figure, which was already `warn`.

**Bounces (1).** The first verify couldn't run: the tester's shell had no browser.
The advisor handed it back once Don had approved the tester's `ui-check` runner. The
second verify (develop @ `d687fa1c`) failed C-F1032 and C-F1030 (e): on basis
`inherited`, `priced_for` was dropped, and the editor's inline validate on mini-ai
does get basis `inherited`, so it was the line a user sees. The fix put the card on
every figure line. The third verify (develop @ `e838574d`) passed C-F1028 to C-F1032.

### Stage B (#1468): the run page validates as you edit and shows the plan and findings

Merged to `develop` at `96450721`, then `6a3c5514` (branch
`feat/1446-B-run-page-validate`: `96286f2e`, and `02952826`, which spells the
validate route as one literal so `tests/test_ui_twins.py` reads it as a POST).

- **`api.validate`** takes `{workflow?, workflow_path?, arguments?, sizes?}`, plus an
  optional `AbortSignal`. `sizes: false` becomes `?sizes=false`; the sized call is a
  plain `POST /api/validate`. The editor's inline call is unchanged.
- **One args builder:** `runArguments()` in `WorkflowPage.svelte` drops empty
  overrides. `run()` and the check both use it, so the check asks about exactly what
  Run would send.
- **Timing:** a check on load and 600 ms after the last change, with `sizes=false`,
  then one sized call after 3 s idle.
- **Stale drop:** each change bumps a generation and aborts in-flight requests. An
  answer for an older generation is ignored, the late sized one included.
- **The check box** (`aria-label="run check"`), under the header row that holds Run:
  - the `describePlan` lines;
  - a verdict: "a run would be admitted", "…, with warnings", "a run would be
    refused", or "could not check this run" on an HTTP error, whose text goes in the
    general list;
  - a general findings list, errors `--bad` and warnings `--warn`, text only.
- **Field findings:** `placeFindings` (`ui/src/lib/findings.ts`) puts a finding whose
  path is `arguments.<name>` or `variables.<name>` under that field, through a new
  `VariablesForm` `findings` prop, when the form shows that field. A trailing `[0]` or
  `.key` is kept as a label on the message. Everything else goes in the general list,
  unfiltered. #1447's `presets`/`constraints` props are untouched.
- **Run is never gated:** it stays enabled while a check is in flight and while one
  refuses. A refused Run still toasts.
- **Tests:** a new `ui/src/lib/pages/WorkflowPage.test.ts`, and a Playwright case in
  `ui/e2e/smoke.spec.ts` ("the workflow page checks a run as you edit, and places each
  finding") with a mocked `/api/validate`.
- **Docs:** `docs/SERVER.md`, the workflow page paragraph. `docs/ARCHITECTURE.md`, the
  UI row "The UI reads engine fields": the page places a finding by its path prefix
  alone and derives nothing else.

**Bounces (0).** The first verify (develop @ `6a3c5514`) passed C-F1033 to C-F1037
and SE-F069 with `ui-check`. It couldn't hold a mocked answer, so it skipped C-F1036
step 1 (Run enabled while a check is held) and C-F1037's stale-answer steps. Don
approved a `delay_ms` on `ui-check` (harnest#251). The follow-up verify (develop @
`6ef6ecf1`) ran those three steps, and all three passed: no stale answer and no stale
sized answer ever showed.

Stage costs aren't recorded: the stage comments name no `usage:` figures. The plan
estimated ~$2 and ~$4. The feature used no GPU time.

## Cut, and what brings each back

- **Host-RAM need against free RAM.** It comes back with a free-host-RAM field on
  `/api/health`, filed as its own issue. That is new REST surface, which is Don's call.
- **Gating Run on the plan** (Q3). It comes back if refused Runs turn out common
  enough to be worth moving a gate into the UI. That would break `smoke.spec.ts`'s
  "Run enabled" case.
- **Suppressing plan tags a warning restates** (Q2). It would mean text-matching in
  the UI.
- **"Run again" on the job page.** Out of scope; #1448.

## Known limits

- Placing a warning under a field depends on the server's `"path: message"` string
  format. If the format changes, placement degrades, but every warning still shows in
  the general list.
- A gated repo, a withheld figure or an over-ceiling host-RAM need can show twice, as
  a plan tag and as the server's warning. That is accepted under Q2.
- The plan's timings weren't measured on mini-ai, and CUDA hasn't seen either stage.
  Neither has anything accelerator-specific beyond the card label in `priced_for`.
