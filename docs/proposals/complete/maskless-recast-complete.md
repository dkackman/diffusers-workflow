# Maskless whole-shot performer swap: a negative result (#1411)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-20). #1411 began as an idea from Don's owner:don walkthrough of
[Bambushu/h3-recast](https://github.com/Bambushu/h3-recast) (MIT; concepts only, no files
copied; credited in `docs/ATTRIBUTIONS.md`). Plan v2, the plan comment on #1411, was
approved on 2026-11-16 and built as stages #1412 (A), #1413 (B) and #1414 (C). Stage D
(#1415) was closed not planned on 2026-11-20, because C failed its gate. This file
records what was built, what was measured, and why the template never shipped.

## Verdict: build smaller, gated. The gate failed, so no template

h3-recast swaps people in one shot with no masks: the clip goes in as a guide latent on
ref2va, an edited frame of the clip is `<Picture 1>`, and a character-swap LoRA does the
rest. The plan built the engine opt-in (a whole-render guide on ref2va) and benched it
against #1194's masked `replace-performer`, with the template held back until the bench
earned it.

On seed 1194 only, the maskless arm held the face at SFace 0.543 against the masked
route's 0.747, and the untouched regions at 23.1 dB against 37.5 dB. It needs no seam
band, but it redraws the whole frame. Don ruled (2026-11-20): close as a negative result,
no re-plan. #1194's masked route stays the way to replace a performer.

## Plan history

- **v1 (approved 2026-11-16):**
  - Q1 [record] the BFS character-swap LoRA's licence: **dropped.** Its Civitai page is
    marked mature content, and Don won't use it. There was no upload and no catalog
    entry, and nothing in the plugin or docs names it.
  - Q2 [trust] how BFS reaches lem: moot.
  - Q3 [spend] lem for C: **one seed (1194) per arm**, not two.
  - Q4 [taste] Don's verdict on C's frames is part of D's gate: yes.
  - Q5 [scope] `guides` legal on ref2va, narrowly: yes. One entry, at frame 0, covering
    the whole render.
- **v2:** recorded the rulings. Stage C runs arms 1 and 3 (arm 2, BFS, is gone, and the
  numbering is kept). D was filed `parked` behind C.

## What was built

- **A (#1412): caption rules.** `plugins/dw/skills/minimax-h3/references/regional-edit.md`
  gives a performer swap a line for every person in the shot (a person left out gets
  re-dressed) and the form "A becomes B". Big changes land and subtle ones fall back to
  the clip. Credited to h3-recast. These hold on either route, so they stay.
- **B (#1413): ref2va takes one whole-render guide.** `guides` on ref2va accepts exactly
  one entry, at frame 0, as long as the render. `guide.audio` stays refused, and the
  audio is kept with `hold_audio`. `latent_anchor`, guide noise, selflift, windows and
  guide chains keep refusing ref2va with their old messages. The guide-family rule in
  `dw/guides.py` is split per argument. A ref2va twin of the dw H3 guide layout block
  appends the guide's latents after the target, at the target's rotary times, with clean
  timesteps. Merged as `230efd1b` and `8377a7b9`. The schema, `WORKFLOW_GUIDE.md`,
  `ARCHITECTURE.md` and `chains.md` state the new rule.
- **C (#1414): the bench.** `bench/arms/t16-recast-shot.json`,
  `bench/workflows/t16-recast-shot.json` and
  `bench/workflows/t16-recast-replace-performer.json`, plus the section "A mask-free head
  swap against replace-performer" in `bench/records/performer-replacement.md`. The
  `head-swap` catalog entry stays `rejected`, now on this evidence rather than "blocked,
  not judged", and is re-pinned to revision `61213345` (the old pin was lost to a
  history rewrite upstream).

## What was measured (#1414, seed 1194 only)

#1194's deflect cut (960x544, 124 frames, 24 fps), with one person swapped for the man in
`asset:bench/t26-stranger.jpg`.

| Arm | Face hold, SFace median (min) | PSNR more than 96 px outside | Seam band | Wall |
|---|---|---|---|---|
| 1: `replace-performer` (masked) | 0.747 (0.695) | 37.5 dB | yes, the mask edge | 4.2 min |
| 3: ref2va + whole-render guide + `head-swap`, scene reference | 0.543 (0.499) | 23.1 dB | none | 26.3 min |

- **By eye.** Arm 3 swaps the whole person from the scene reference and holds through the
  shot, but the face reads as a cousin of the portrait's man, and the room is redrawn at
  about 3-4x the VAE floor at every frame.
- **Gate.** Face hold within 0.03: fail (-0.204). Untouched regions within 2 dB: fail
  (-14.4). No seam band: pass. Don's preference was never needed: two measured failures
  decide it.
- **A supplementary run** with the face portrait itself as `<Picture 1>` (the LoRA
  author's documented input) scored 0.653 and 29.5 dB. It also fails both, so the verdict
  doesn't depend on the reference picture.
- **Caveat.** The risk named in the plan stands: with BFS gone, the bench can't tell a
  splice mismatch against ComfyUI's `MiniMaxH3AddGuide` from a LoRA that just doesn't
  hold this face. The evidence is single-seed.

## Deferred, and why

- **D, the `recast-shot` template and skill line (#1415):** closed not planned, since C
  failed its gate. Its pending cases C-F981 and C-F982 go to the curator for retirement.
- **The BFS LoRA:** dropped by Don (Q1), and not to be revived through the catalog.
- **The lip-repair pass** belongs to #1385. Multi-cut clips belong to #1194.

## What stays

Stage B's ref2va guide is in the engine, documented, and unit-tested. Nothing in the
catalog uses it today. Removing it would change workflow syntax, so that is Don's call
(`scope`), and nothing here asks for it. The caption rules from A stay in #1194's
guidance.

## Cost and bounces

The stage comments name no `usage:` figures, so cost is left out. The bench's lem time
was arm 1 at 4.2 min, arm 3 at 26.3 min (and a first run of it that failed on the lost
pin), and the supplementary portrait run at 19.6 min.

| Stage | Bounces |
|---|---|
| A (#1412) | 0 |
| B (#1413) | 1: an architecture-review bounce. A failed deploy then held the hand-off until Don ruled the server healthy |
| C (#1414) | 1: arm 3 first ran with the portrait instead of the plan's scene reference, and was re-run |
| D (#1415) | not built |
