# H3 estimates: same-bucket history rescaled by frames, and a first-pass-at-shape term (#1615)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-26). Plan v3 on #1615 is the full design, with v1 and v2 condensed
into it. This file records what shipped across all three plan versions, what
was left out, and why.

**Result:** dialogue-short's quote no longer drops same-template history when a
shot list's frame counts change. It rescales that history by frames instead
(stage 1). And a first run at a shape this card hasn't autotuned now quotes
the roughly 6 min it costs, named in `scaled` (stages 3 and 4). The cost was
measured on lem in stage 2.

## The problem

This came from #1523's field report ("Whitefish Point", lem, 0.23.0-alpha.1,
one 3090, items 3 and 10). Every H3 quote was `basis: derived,
low_confidence`, and the quotes ran low, worst at frame counts lem hadn't run.
shot-05 (L7s, 209 frames) was quoted 7.6 min and measured 17.0. The issue named
three gaps:

1. **First pass at a new shape:** 7-8 min that no estimate included.
2. **Same-session history unused:** a shot list at different frame counts
   matched no fingerprint, so the derived price ignored a near-neighbour run
   of the same template that had just been measured.
3. **Per-tier bias:** L5s ran high and B3s ran low.

## What shipped

### Stage 1, #1696: rescale same-bucket history by frames

- `dw/server/observed_cost.py`: each bucket's observed block reports
  `entry_frames`, the median of its cold runs' summed entry `num_frames`. It
  is read from the stored `arguments` (`effective_variables`), so it needs no
  migration.
- `dw/plan.py` `estimate`: where #796 used to discard a bucket whose
  list-entry cost field had shifted, `_frames_rescaled` now rescales it:
  `cold = fixed + (cold - fixed) × run_frames / bucket_frames` (and the same
  for warm), with `fixed` taken from the row's `fixed_share`.
  - `basis` is `derived`, and `scaled` names the source figure and both frame
    counts.
  - `low_confidence` is set only when the ratio falls outside 0.5-2.0.
  - A block with no `entry_frames` is still discarded.
- Thin buckets still temper toward the default-tier baseline (rescaled the
  same way) or toward the curated figure.
- No surface change.

### Stage 2, #1697: the first-pass probe on lem (measure only)

- B3s canvas (1344x768, `{int8}`, sparse, no LoRA, one reference, no voice).
- First at 226 frames: 351.1 s over a steady pass. First at 243: 373.8 s. A
  repeat at 226 after a full weight reload: 0.8 s.
  ([Report](https://github.com/dkackman/diffusers-workflow/issues/1615#issuecomment-7184453071).)
- So the penalty is per shape, sits in the box's Triton autotune cache rather
  than in the worker, and is nearly flat across sizes.
- Rulings on #1697 (the delegate's recommendation, approved by Don
  2026-11-26): the term is added to `minutes` and named in `scaled` (Q2), and
  the `first_at_shape` key goes ahead (Q3), priced flat at about 6 min.

### Stage 3, #1763: the `first_at_shape` cost-row key and term

- **Syntax** (Q4, default approved): `cost[].first_at_shape: {minutes, keys}`
  in `dw/workflow_schema.json`.
  - `additionalProperties: false`.
  - `minutes` is a number ≥ 0.
  - `keys` is a non-empty array of unique names of variables the workflow
    declares, scalar only.
  - A list driver's entry `num_frames` joins the shape through the existing
    `LIST_ENTRY_COST_FIELDS` convention. There is no path syntax, so
    `num_frames` in `keys` is refused as undeclared.
- **Seen shapes** (Q5, default approved): the server-side job rows
  `ObservedCosts` already loads. A shape counts as seen when a succeeded,
  non-partial run of the same workflow name (plus `inherited_from` for an
  inline workflow) ran it on a card with this card's name.
  - A shape is the tuple of the resolved `keys` values plus one entry frame
    count.
  - Nothing new is stored, and there is no REST, MCP or UI surface.
- **The term:** `first_at_shape.minutes` × the number of distinct unseen
  shapes. `dw/plan_pricing.py` `with_first_at_shape` applies it once to
  `_checked_estimate`'s result in `build_plan`, after the checked-chain floor.
  - `scaled` gains `plus N min first pass at K shape(s) this card hasn't run
    (...)`, naming up to three shapes and then "and N more".
  - `basis`, `low_confidence`, `warm_minutes` and `cached_minutes` don't move.
  - It doesn't feed `fixed_share`.
  - With no seen-shapes callable, no term is added.
- The compact `list_workflows` strips the key. `get_workflow` keeps it.

### Stage 4, #1764: dialogue-short wired, and cache clears forgotten

- `workflows/templates/minimax/dialogue-short.json`'s RTX 3090 row carries
  `first_at_shape: {minutes: 6.0, keys: [width, height, weights_dtype,
  attention_backend]}`.
- `dw/disk_caches.py` `clear()` writes `<root>/.dw-cleared` (epoch time), and
  `last_cleared(name)` reads it. `seen_shapes(..., since=)` only counts rows
  that finished after the newest `triton` clear. With no marker, every row
  counts.
- The `first_pass_slow` text (`dw/pipeline_processors/timing.py`) names
  autotune at a new shape as a possible cause.
- Docs and skills:
  - `docs/SERVER.md`.
  - The triton `cost_of_clearing` text.
  - `plugins/dw/skills/minimax-h3/references/watching-a-run.md` and
    `quality-and-speed.md`.
  - The version bump to `0.23.0-alpha.4`.
- Per Q6, the cache-clear rule is covered by unit tests only, and lem's
  Triton cache was never cleared.

## Verified on lem (validate only, no GPU)

For stage 4, the tester validated B3s settings with no voice arguments:
- Probe shapes 226 and 243, alone and as a pair: no term.
- Unseen 158: +6 min, on top of a stage-1 rescale within 5%.
- 158 + 141: +12.
- 158 twice: +6.
- 226 at `{int4}`: +6, because lem hasn't run that shape.
- The default shot list: still `observed`, with no term.

## Deferred, and why

- **The generic per-tier calibration ratio (gap 3)** was cut in v2.
  - Stage 1's rescale is what closed gap 2.
  - The trigger to revisit: a tier's first run still under-quoted by more
    than 25% in a field report.
- **`fixed_minutes` on dialogue-short** (Q1): ruled no on 2026-11-25. #1239
  stands.
- **Other H3 templates:** a data-only follow-up once the term has proven
  itself on dialogue-short.
- **A wider "seen" set** (other workflows, per box) waits for a field report.
  Today's proxy over-quotes when another template has warmed a shape.
- **Reference count and LoRA as shape keys** weren't probed, so they'll be
  added only on evidence. A changed `TRITON_CACHE_DIR` isn't detected.
- **The term inside a composing parent** (`_child_catalog_price`) isn't
  priced.

## Cost and bounces per stage

The stage comments carry no `usage:` figures, so no cost is recorded here.

| Stage | Bounces | Notes |
|---|---|---|
| 1, #1696 | 0 | Architecture review passed, verified on the first hand-off. |
| 2, #1697 | 0 | Probe passed. Closing it took two delegate rulings because of labels, not because of the work. |
| 3, #1763 | 1 (architecture) | Growth in `dw/plan.py` (over the size ceiling). The term moved into `plan_pricing.with_first_at_shape`, and the split was filed as #1767. |
| 4, #1764 | 1 (architecture) | The `docs/ARCHITECTURE.md` estimate row lacked the `since` filter. The fix was docs and a docstring only. |

The tester bounced nothing. Case defects found while verifying went to
harnest as amendments:
- C-F1536's listing shape should be `sequence`.
- C-F1537 should drop the voice references, which push 226 and above past
  the VRAM ceiling.
- C-F1537's unseen counts should be 158 and 141.

## Observation left open

While verifying stage 3, an inline copy of dialogue-short validated in
`default` was attributed to `templates/minimax/composable-references`, so it
was charged for shapes that `default` has run. The tester read this as an
inline-attribution artifact rather than a fault in the term. It isn't tracked
yet. File it if it shows up in a real quote.
