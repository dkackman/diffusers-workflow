# LTX-2.5 intents: start from a result

Part of the `ltx-2.5` skill. Each row is one result someone wants, stated
as the claim that was tested, with where its pattern is written and the
record that settled it. Status is the record's verdict on that claim:
`confirmed` (it held), `refuted` (it didn't) or `untested` (no record here;
the issue or source is named). The numbers and prompt words live in the
pattern and the record, never in this table, so follow the pattern cell.
An empty template cell means the pattern is prompt-only and fits any LTX-2.5
template.

`tests/test_plugin_intents.py` pins the format: five columns, a status from
the three, a confirmed or refuted row citing a record that exists, and a
pattern heading that exists. It also fails on a `bench/records/` file no
index row cites.

| intent | template | pattern | evidence | status |
|---|---|---|---|---|
| Chain past one clip on the last frame, capping a take and cutting to a fresh still of the subject | `templates/ltx2/chained-segments` | `SKILL.md` § Which shape is the request | `bench/records/ltx.md` | confirmed |
| One LTX chain held as a single take well past the cap | `templates/ltx2/chained-segments` | `long-form/references/quality-and-speed.md` § The tier dial | `bench/records/ltx.md` | refuted |
| Extend, conditioned on the whole opening clip, beats the last-frame chain as the default | `templates/ltx2/extend-clip` | `SKILL.md` § Which shape is the request | `bench/records/ltx.md` | refuted |
| `look_hold` smooths an LTX chain's seams without holding a camera move back | `templates/ltx2/chained-segments` | `long-form/references/quality-and-speed.md` § The tier dial | `bench/records/ltx.md` | confirmed |
| Carrying the sharpest last frame (`carry_pick`) improves an LTX chain | `templates/ltx2/chained-segments` | `long-form/references/levers.md` § Measured and cut: do not re-propose | `bench/records/ltx.md` | refuted |
| An fp32 audio decode brings a silent chain segment's sound back | `templates/ltx2/chained-segments` | `long-form/references/run-and-repair.md` § The per-segment check | `bench/records/ltx.md` | refuted |
| The caption steers a speaker's pitch, apparent age and gender | | `SKILL.md` § Prompts | measured on mps in commit ebe8d215, no record | untested |
| A whisper or a tempo written in the caption is delivered | | `SKILL.md` § Prompts | measured on mps in commit ebe8d215, no record | untested |
| Colorize colours a clip when the prompt names its colours; a low-rate or portrait source may stay grey | `templates/ltx2/restore-colorize` | `references/repair.md` § One defect each | `docs/proposals/complete/ltx2-ic-loras-complete.md` | confirmed |
| Deblur and decompression each invert one defect and no other | `templates/ltx2/restore-deblur` | `references/repair.md` § One defect each | vendor model cards, per `docs/proposals/audits/2026-10-02-ltx-2.5-audit.md` | untested |
| Upscale or refine the user's clip to exactly twice its size, soundtrack kept | `templates/ltx2/upscale-clip` | `references/repair.md` § Upscale or sharpen | `docs/proposals/complete/fit-to-model-restore-complete.md` | confirmed |
| Refine-in-place's strength runs from preserving the source to reinterpreting it | `templates/ltx2/refine-in-place` | `references/repair.md` § Upscale or sharpen | `docs/proposals/complete/ltx2-refine-in-place-complete.md` | confirmed |
| Face repair sharpens a small far face and leaves a near face untouched | `templates/ltx2/face-repair` | `references/repair.md` § A small face | `docs/proposals/complete/face-repair-complete.md` | confirmed |
| Restore footage longer than one window with no visible seam | `templates/ltx2/restore-long` | `references/repair.md` § Longer than one window | `docs/proposals/complete/overlap-windowing-complete.md` | confirmed |
| A hard cut in the source leaks its texture into the matte after it | `templates/ltx2/alpha-matte` | `SKILL.md` § Which shape is the request | `docs/proposals/complete/ltx2-ic-loras-complete.md` | confirmed |
| A prompt steers which subject the matte keeps | `templates/ltx2/alpha-matte` | `SKILL.md` § Which shape is the request | vendor model card, per `docs/proposals/audits/2026-10-02-ltx-2.5-audit.md` | untested |
| The two-stage refine adds the detail the latent upsample alone lacks | `templates/ltx2/two-stage` | `SKILL.md` § Which shape is the request | vendor recipe, per `docs/proposals/audits/2026-10-02-ltx-2.5-audit.md` | untested |
| Image-to-video reaches its longest clip in one pass on a 24 GB card | `templates/ltx2/image-to-video` | `SKILL.md` § Which shape is the request | issue only: #1052 | untested |
| What isn't on the reference sheet doesn't appear in the clip | `templates/ltx2/reference-sheet` | `SKILL.md` § Which shape is the request | vendor Ingredients card, per `docs/proposals/audits/2026-10-02-ltx-2.5-audit.md` | untested |
| Extend the user's own clip through extend-clip's `clip` input | `templates/ltx2/extend-clip` | `SKILL.md` § Which shape is the request | dw's own recipe, no measured run | untested |
