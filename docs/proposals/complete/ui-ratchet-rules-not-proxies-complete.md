# The UI ratchet gates on rules and warns on proxies (#1689)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-26). Plan v1 was approved as the delegate recommended ("approve with
delegate's recommendations", Don, 2026-11-25), with every question at its
default. Three stages, #1713, #1714 and #1715, were built, verified and
closed. #1666 (token discipline) was folded in as a duplicate and then cut.

## The idea

The UI refactor review of 2026-11-25 (b2e8ff73..f8af3f02) found that
`ui/scripts/arch-metrics.mjs` mixed two kinds of metric. Some are real rules:
no import cycles, no raw error toasts, no window dialogs, reads through
`resource`, mutations through `action`, banned terms. Others are proxies:
`files_over_size_ceiling`, `complex_functions`, `long_functions` and
`duplicate_blocks`. The proxies failed builds and pushed work toward the
number. Meanwhile a layering leak (#1669) and an amber that meant both
"running" and "cancelled" (#1665) passed every check.

## What the sweep found (plan v1, develop 5f90c6f5)

- **The size ceiling drew file boundaries.** `patterns.css` called itself
  "app.css's overflow". 189af586 trimmed `api.ts` back to 600 lines, and
  0b7bee23 moved code out of GalleryPage only to get it under 600. Ten files
  sat between 546 and 597 lines.
- **9a03172a** was a reshuffle whose only purpose was to bring
  `complex_functions` back to 11.
- **`duplicate_blocks`** (112) hashed six-line windows of markup. It found
  none of the duplicated behaviour the review found (#1667, #1668, #1656).
- **A second size gate** lived in `design-rules.test.ts`: `app.css` at most
  600 lines.
- **The proxies named nothing in their detail**, so a warning could not
  point a reviewer at anything.
- **The harness reads stdout.** `regressions()` prints `key: base -> cur`,
  and harnest's `check_ui` matches that form. Warnings had to stay off
  stdout.
- **Layering leaks.** ui → request/assets, top-level lib → request/editor,
  and an editor ↔ workflow cycle. A metric suited this better than ESLint,
  because flat-config `no-restricted-imports` options replace each other
  rather than merge.
- **Colour.** `--warn` equalled `--live` in all four theme blocks, and the
  `--live` rule checked token names, never values.

## Decisions

All at the plan's defaults, through Don's approval of the delegate's
recommendation:

- **Q1 [scope]:** the four proxies become advisory. They are still measured
  and recorded, warn on stderr, and never fail `--check`/`--compare`.
- **Q2 [scope]:** one enforced outer limit, `files_over_hard_ceiling` at 900
  lines. The `app.css` 600-line test is deleted.
- **Q3 [none]:** the default `LAYER_ALLOWS` direction.
- **Q4 [scope]:** #1665's `--warn` colour is included, as stage C.
- **Q5 [taste]:** ochre. Don accepted light `#7a6000` and dark `#d1b440` on
  the tester's screenshots (2026-11-26).
- **Q6, Q7, Q8 [scope]:** cut, and nothing filed: the token-discipline
  count, the CSS regroup and the backend ratchet.
- **Q9 [none]:** `ui/scripts/arch-metrics.mjs` joins
  `docs/stabilization/hot-zone.txt` after stage B.

## What was built

### Stage A, #1713: proxies become warnings (2572ad40, merged f009a2f0; docs fix 88c76d67, merged 7055ee40)

- `ADVISORY` holds the four proxy keys. `regressions()` skips them. The new
  `advisories()` and `warnAdvisories()` print each rise to **stderr** as
  `warning (advisory): key: base -> cur`, with one indented line per named
  item. stdout and the exit code are unchanged for enforced keys.
- Detail names what it flags. `complex_functions`/`long_functions` read
  `file:line name (value)`, `files_over_size_ceiling` reads
  `file (N lines)`, and `duplicate_blocks` reads `file: N lines (ranges)`.
  All are sorted.
- `files_over_hard_ceiling` counts files over `HARD_SIZE_CEILING = 900`,
  with baseline 0.
- The script's self-complexity test and the `app.css` 600-line test are
  removed. `patterns.css`'s header and ARCHITECTURE.md describe what each
  CSS file holds rather than calling one "overflow".
- The docs (ARCHITECTURE ratchet row, the ROADMAP metric paragraph and
  table, `stage-b-ratchet.md:57`, and the script header) say which keys are
  enforced and which are advisory.
- **The `check_ui` risk was settled.** harnest's `lib/arch_ratchet.py` runs
  `--compare` with `capture_output=True` and matches only stdout, so an
  advisory rise reads as a pass. harnest needed no change.
- **What `develop` showed at landing:** two advisory rises already present
  (`api.ts` at 614 lines; `Function 'resource'` at 63 lines). Before this
  stage they failed `--check`; now they print warnings.

### Stage B, #1714: the layering metric (ce2a4c95 and 1c80d192, merged 51fa57c2)

- `layering_violations` is enforced. It is counted over the import graph the
  cycles metric already builds, with type-only imports counted and tests
  excluded.
- `LAYER_ALLOWS`: job → gallery, request, workflow, editor; gallery →
  request; workflow → editor; editor, assets and request → none. It also
  declares `schema` and `test` as `[]`, two folders the plan didn't list.
  `SHARED_LAYER` is `ui` plus the top-level `src/lib/*` modules (shown as
  `lib`), and these may import only the shared layer. `TOP_LAYER` is
  `pages`. Files outside `src/lib` import anything. A folder in no table
  counts every edge out of it.
- `undeclaredFolders()` drives a completeness test on the real tree.
- **Baseline 0, not today's leak count.** #1669 landed first and removed the
  cross-layer edges, as the plan's risk section foresaw. Fixture tests show
  each rule counting and naming its edge.
- Docs: a *UI layering* ARCHITECTURE row, a *Rules for new UI* line in
  `ui/CLAUDE.md`, and a ROADMAP table row. The branch's last commit adds
  `arch-metrics.mjs` to `hot-zone.txt` (Q9).
- It counts one violation per import declaration.

### Stage C, #1715: `--warn` gets its own colour (0ef2be8d, merged 97e40a3b; docs fix 43bd04a6, merged 848e10c0)

- `--warn` is light `#7a6000` and dark `#d1b440` in all four theme blocks,
  kept clear of dark `--live` (`#e8a33d`) and `--finding-warn`.
- To meet the contrast gate, light `--live` moved from `#a85b00` to
  `#a35800`. A delegate ruling (record) accepted this inside Q5, and Don's
  acceptance of the screenshots covered it.
- `design-rules.test.ts` has two new cases. One fails when any token in a
  theme block takes `--live`'s value. The other checks `--warn` and `--live`
  at 4.5:1 or better on `--panel` and `--panel-2`.
- The wording was restated in the `app.css` header, `ui/CLAUDE.md:11`,
  ARCHITECTURE.md and the `StatusChip`/`FlowView` comments, and "amber" is
  gone. `ui/CLAUDE.md` names every colour carrier, including
  `--finding-*` and `wsColor()`.
- The cancelled chip, the stall note, the dirty dot and the required
  markers all moved with `var(--warn)`.

Acceptance cases C-F1364 to C-F1379 (`regression-suite-complete.md`):
stage A and B cases were checked against the run record and re-run in docs
review; stage C cases passed in a browser through `ui-check`.

## Deferred

- **#1666's token-discipline count** (spacing, font-size, radius literals).
  Cut. What would bring it back is a ruling on the spacing scale (a hairline
  step, or allowed literals), because 18 `gap` values against 5 steps put
  the scale in question first.
- **Regrouping the CSS by subject.** Not filed. The cap no longer forces the
  seam, and the seam's wording was fixed in stage A.
- **The backend ratchet's proxies** (`scripts/arch_metrics.py`). Not
  touched. It is in `hot-zone.txt`, so a follow-up would be Don's.
- **Stores in `unwatched_reads`/`unguarded_mutations`.** Left to #1663.
- **A trigger to re-enforce a proxy:** if the advisory numbers climb a lot
  between reviews (for example, files over 750 lines), move that key out of
  `ADVISORY`.

## Cost and bounces

Planned at about $8-11 over three sessions (A $3-4, B $3-4, C $2-3). The
stage comments name no `usage:` figures, so actual cost is not recorded.

- **#1713: two bounces.** The tester bounced the first hand-off because it
  had no command record; the cases required each command quoted with its
  exit code and separate stdout/stderr. Then docs review found a stale
  "`app.css` stays at 600 lines" claim. Each bounce was fixed in one pass.
- **#1714: none.** The tester sent it to docs review, which passed it first
  time.
- **#1715: one bounce**, from docs review: a stale "--warn is the --live hex"
  comment in `request/itemState.ts` and a short colour-carrier list in
  `ui/CLAUDE.md`. Separately, the stall case (C-F1377) couldn't be driven
  black-box until the builder documented the SSE frames to fake. That was a
  question, not a failure.
