# Klein intents: start from a result

Part of the `flux3-klein` skill. Each row is one result someone wants, stated
as the claim that was tested, with where its pattern is written. Status is
the verdict of the measured record behind the row: `confirmed` (it held),
`refuted` (it didn't) or `untested` (no measured record). The numbers and
prompt words live in the pattern, never in this table, so follow the pattern
cell.
An empty template cell means the pattern is prompt-only and fits any Klein
template.

The record behind each row stays in the dw repo, in
`tests/intents_evidence.json`.

| intent | template | pattern | status |
| --- | --- | --- | --- |
| Keep two people on their sides of frame from the two-shot to each single, by writing each one's side and facing in every card | `templates/flux3/text-to-image` | `cinematography/references/continuity.md` § Sides and eyelines (FLUX 3 Klein) | confirmed |
| Match eyelines across the singles: each looks toward the partner's named off-screen side | `templates/flux3/text-to-image` | `cinematography/references/continuity.md` § Sides and eyelines (FLUX 3 Klein) | confirmed |
