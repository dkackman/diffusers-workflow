# Klein intents: start from a result

Part of the `flux3-klein` skill. Each row is one result someone wants, stated
as the claim that was tested, with where its pattern is written and the
record that settled it. Status is the record's verdict on that claim:
`confirmed` (it held), `refuted` (it didn't) or `untested` (no record here;
the issue or source is named). The numbers and prompt words live in the
pattern and the record, never in this table, so follow the pattern cell.
An empty template cell means the pattern is prompt-only and fits any Klein
template.

`tests/test_plugin_intents.py` pins the format: five columns, a status from
the three, a confirmed or refuted row citing a record that exists, and a
pattern heading that exists.

| intent | template | pattern | evidence | status |
|---|---|---|---|---|
| Keep two people on their sides of frame from the two-shot to each single, by writing each one's side and facing in every card | `templates/flux3/text-to-image` | `cinematography/references/continuity.md` § Sides and eyelines (FLUX 3 Klein) | `bench/records/continuity-grammar.md` (PRELIMINARY) | confirmed |
| Match eyelines across the singles: each looks toward the partner's named off-screen side | `templates/flux3/text-to-image` | `cinematography/references/continuity.md` § Sides and eyelines (FLUX 3 Klein) | `bench/records/continuity-grammar.md` (PRELIMINARY) | confirmed |
| Hold the key light on one side of the face across a wide, a medium and a close-up by naming its side | `templates/flux3/text-to-image` | `cinematography/references/continuity.md` § Probed and not a rule | `bench/records/continuity-grammar.md` (PRELIMINARY; one seed of two, Don's look pending) | refuted |
