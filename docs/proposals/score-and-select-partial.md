# Remaining work: score and select - catalog template

Split from `score-and-select-complete.md` on 2026-09-20. The `select` and
`judge` tasks, their validation, and manifest/step_end recording (sections 1,
2, 4, and "No candidate passes" of the original proposal) are implemented and
tested — see the complete doc for that design and its evidence. What remains
is section 3 of the original proposal, unimplemented:

## One catalog template

`templates/flux/best-of-n-to-video` (name to taste): `still` fanned over a
`candidates` list of seeds, `judge` over the same list, `pick`, then the
LTX-2.5 image-to-video template as a `builtin:` sub-workflow on
`previous_result:pick`. Marks `still`/`judge` `intermediate` and the video
`final` per the subfolder rule. This gives the plugin skills something to name
and pins the shape in `tests/test_plugin_skills.py` the way every other
template's numbers are.

## Remaining steps

1. Author the template (`templates/flux/best-of-n-to-video.json` or
   equivalent), wiring `still` (for_each over seeds) → `judge` (same for_each)
   → `pick` (`select`) → an LTX-2.5 sub-workflow on `previous_result:pick`.
2. Mark `still`/`judge` `intermediate` and the video step `final`.
3. Add the template's numbers to `tests/test_plugin_skills.py`'s
   catalog-pinning sweep.
4. Consider whether a plugin skill (e.g. `dw:score-and-select` guidance inside
   an existing family skill) should name the new shape, per the original
   proposal's intent that this "gives the plugin skills something to name."

## Testing not yet covered

- Plugin: the template's numbers pinned like every other template's, once it
  exists.

## Not in scope (carried over from the original proposal)

- `when` / conditional steps / skipped-step semantics.
- Retry of any kind.
- A whole-list reference for a single step's artifacts (`all:<step>`).
- Semantic selection (stays with the agent).

## Later (carried over)

- `all:<step>` so a `num_images_per_prompt` fan-out can feed `select` without
  a `for_each`.
- A second scorer (aesthetic / CLIP).
- `select` returning the top *k* rather than one (`rule: top_k`).

## Open question for whoever picks this up (carried over)

Whether `judge` should be a new command or a mode of `image_to_text`
(`"parse": "number"`). Leaning towards a separate command that shares the
loader.
