# #801: per-entry `references` on `music-video`, falling back to `singer_reference`

Scope (Don's ruling on #801): item 3 only. Items 1 and 2 are #805.

## Why this is a proposal, not a template edit

The engine has no optional `item:` field. `dw/for_each.py::_item` raises
`names no field of entry ...` when an entry lacks the field, and nothing
substitutes a default. So "an entry may carry `references`, otherwise
`singer_reference`" cannot be written in the template today:

- `"references": "item:references"` (the `dialogue-short` pattern) makes the field
  required on every entry. A caller's existing `shots` lists, and the ones in
  skills and docs, would all fail validation: a breaking change.
- Keeping `["variable:singer_reference", <slice>]` in the step and adding an
  `item:` extra has the same problem, since the extra must exist.

Also, inside an `item:` the slice reference is not auto-paired
(`WORKFLOW_GUIDE.md`, *Authoring*), so a per-entry list has to spell
`slice@<name>` itself. An entry that overrides the picture has to restate the
audio reference too.

## Options

A. **Optional item field (engine).** A way to read an entry field with a
   fallback, e.g. `{"item": "singer", "default": "variable:singer_reference"}`,
   resolved in `_item`, with the field-list warning
   (`entry_field_warnings`) and the UI twin (`ui/src/lib/flow.ts`) told that the
   field is optional. Smallest change that gives the requested behaviour; adds
   one concept consumers learn. Narrow scope: the entry field would be
   `singer` (the image reference only), the template keeps building the list
   `[<picture>, <slice audio>]`, so the audio pairing is not restated.

B. **Required field, no engine change.** Entries carry `references` (as
   `dialogue-short`). Breaking for every existing `shots` list; no fallback.
   Not recommended.

Recommendation: A, with the entry field `singer` (not `references`), because
the audio slice is not the entry's to restate.

## Tests if approved

- `tests/test_for_each.py`: field present, field absent with default, absent
  with no default (still the current error).
- `tests/test_elision.py`: `draw_singer` stays elided when every entry
  overrides and `singer_reference` is supplied; kept when an entry falls back.
- `ui/src/lib/flow.test.ts` + `tests/test_ui_twins.py` case for the new form.
