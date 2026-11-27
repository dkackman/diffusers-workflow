# LTX-2.5 intents: start from a result

Part of the `ltx-2.5` skill. Each row is one result someone wants, stated
as the claim that was tested, with where its pattern is written. Status is
the verdict of the measured record behind the row: `confirmed` (it held),
`refuted` (it didn't) or `untested` (no measured record). The numbers and
prompt words live in the pattern, never in this table, so follow the pattern
cell.
An empty template cell means the pattern is prompt-only and fits any LTX-2.5
template.

The record behind each row stays in the dw repo, in
`tests/intents_evidence.json`.

| intent | template | pattern | status |
| --- | --- | --- | --- |
| Chain past one clip on the last frame, capping a take and cutting to a fresh still of the subject | `templates/ltx2/chained-segments` | `SKILL.md` § Which shape is the request | confirmed |
| One LTX chain held as a single take well past the cap | `templates/ltx2/chained-segments` | `long-form/references/quality-and-speed.md` § The tier dial | refuted |
| Extend, conditioned on the whole opening clip, beats the last-frame chain as the default | `templates/ltx2/extend-clip` | `SKILL.md` § Which shape is the request | refuted |
| `look_hold` smooths an LTX chain's seams without holding a camera move back | `templates/ltx2/chained-segments` | `long-form/references/quality-and-speed.md` § The tier dial | confirmed |
| Carrying the sharpest last frame (`carry_pick`) improves an LTX chain | `templates/ltx2/chained-segments` | `long-form/references/levers.md` § Measured and cut: do not re-propose | refuted |
| An fp32 audio decode brings a silent chain segment's sound back | `templates/ltx2/chained-segments` | `long-form/references/run-and-repair.md` § The per-segment check | refuted |
| The caption steers a speaker's pitch, apparent age and gender |  | `SKILL.md` § Prompts | untested |
| A whisper or a tempo written in the caption is delivered |  | `SKILL.md` § Prompts | untested |
| Colorize colours a clip when the prompt names its colours; a low-rate or portrait source may stay grey | `templates/ltx2/restore-colorize` | `references/repair.md` § One defect each | confirmed |
| Deblur and decompression each invert one defect and no other | `templates/ltx2/restore-deblur` | `references/repair.md` § One defect each | untested |
| Upscale or refine the user's clip to exactly twice its size, soundtrack kept | `templates/ltx2/upscale-clip` | `references/repair.md` § Upscale or sharpen | confirmed |
| Refine-in-place's strength runs from preserving the source to reinterpreting it | `templates/ltx2/refine-in-place` | `references/repair.md` § Upscale or sharpen | confirmed |
| Face repair sharpens a small far face and leaves a near face untouched | `templates/ltx2/face-repair` | `references/repair.md` § A small face | confirmed |
| Restore footage longer than one window with no visible seam | `templates/ltx2/restore-long` | `references/repair.md` § Longer than one window | confirmed |
| A hard cut in the source leaks its texture into the matte after it | `templates/ltx2/alpha-matte` | `SKILL.md` § Which shape is the request | confirmed |
| A prompt steers which subject the matte keeps | `templates/ltx2/alpha-matte` | `SKILL.md` § Which shape is the request | untested |
| The two-stage refine adds the detail the latent upsample alone lacks | `templates/ltx2/two-stage` | `SKILL.md` § Which shape is the request | untested |
| Image-to-video reaches its longest clip in one pass on a 24 GB card | `templates/ltx2/image-to-video` | `SKILL.md` § Which shape is the request | untested |
| What isn't on the reference sheet doesn't appear in the clip | `templates/ltx2/reference-sheet` | `SKILL.md` § Which shape is the request | untested |
| Extend the user's own clip through extend-clip's `clip` input | `templates/ltx2/extend-clip` | `SKILL.md` § Which shape is the request | untested |
