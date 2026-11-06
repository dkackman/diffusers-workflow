---
name: long-form
description: Use when a dw MCP server is connected and the request is longer than one clip - a scene to block, a long take, a series, a music video, or music or narration longer than one generation. Splits the piece into scenes, picks per scene a single clip, cuts or a chain, sets the chain's shot preset and per-segment check, repairs one bad segment in place, joins long narration and music, and exports the cut for an editor. Every rule cites the bench record it came from.
---

# Long-form pieces on a dw server

A model renders one clip at a time: H3 up to 14.4 s, LTX-2.5 about 17 s at
768. A piece longer than that is planned per piece, then per scene, then per
seam, then per run. This skill is that plan. The family skills
(`minimax-h3`, `ltx-2.5`, `minimax-music3`) still own their templates'
rules and prompt formats, and their rules win. `script-to-video`,
`storyboard` and `series-episodes` own the shot list, the boards and the
episode cut.

Each rule names its bench record (`bench/records/<name>.md` in the repo;
`get_guide` serves the guides, not the records). A rule marked
*community* is another user's report, not a dw measurement. A line marked
*pending measurement* has no verdict yet. Don't quote a number for it.

## Before anything

1. `get_server_info`: the device. H3 is CUDA-first. On a 24 GB Mac,
   `templates/minimax/chained-segments` at its defaults is refused
   (about 18.3 GB projected against a 17.8 GB working set). Lower the
   canvas to 768x448, which fits under guide continuity. `num_frames`
   cannot drop below 124 (`docs/ACCELERATION.md`, *MPS*).
2. `list_workflows(shape="sequence")` and `list_workflows(traits="chained")`:
   the long-form templates with their `cost` rows.
3. Prompts: `list_prompts(tag=...)`, then `get_prompt`. A project's shot
   list lives at `<project>/shots` (`script-to-video`).

## 1. Per piece: scenes first

Split the piece into scenes the way a storyboard would. A scene is one place
and one continuous moment. Give each scene a target length from the story,
not from the model; the model ceiling only decides how the scene is rendered.
Record per scene `length_seconds` and `continuity` in the shot document's
`<project>/scenes` (`script-to-video/references/shot-document.md`).

## 2. Per scene: the rendering shape

| Shape | When | Templates |
| --- | --- | --- |
| One clip | at most 14.4 s (H3), about 17 s (LTX-2.5 at 768) | the family skills |
| Cuts | place, time or angle changes, or a beat allows a cut | `templates/minimax/dialogue-short`, `templates/minimax/music-video`, `templates/minimax/shots-batch`, then `templates/assemble-and-score` |
| Chain | one action, line or camera move must cross the ceiling unbroken | `templates/minimax/chained-segments`; LTX: `templates/ltx2/chained-segments`, `templates/ltx2/extend-clip` |
| Bridge | a cut must land on a fixed frame | `templates/minimax/first-and-last-frame`, `templates/minimax/last-frame-only` |

Prefer cuts. A cut resets drift and costs nothing at the seam. Chain only
what must be unbroken. A soundtrack-driven chain is
`templates/minimax/chain-matched-to-audio`; see `minimax-h3`.

## 3. Per chain: continuity and the shot preset

**Guide continuity is the default for every shot type**
(`bench/records/counters.md`, *Defaults by shot type*). Each segment's last
22 frames and their audio carry into the next as a guide.

Set `shot` on `templates/minimax/chained-segments` from the shot's
`cinematography` move:

- **`shot: static`** (locked-off camera, subject in front; `move: static`):
  guide, `anchor: segment_1`, `anchor_strength` 0.25. It beat the guide
  control on two seeds and at two minutes (`bench/records/counters.md`,
  *Round 3*). Don ruled it the static default (`bench/records/for-don.md`).
- **`shot: moving`** (the default; any camera move): guide, no counters.
  Every counter lost seam look on the arc.
- **`shot: dialogue`** (a speaking subject in motion): guide, no counters.
  Every counter was inside the floor or lost. Voice carries through the
  guide (`carry_audio`).

Two rules that bite:

- **Never use `last_frame` with `look_hold` or `carry_pick` on a camera
  move.** It froze the orbit on seed 1166 and held it back on seed 2. The
  "wins" were a camera that moved less (`bench/records/counters.md`,
  *Eye notes (S2)*).
- **Cut static takes longer than about 30 s into shots.** With anchor 0.25
  the backdrop crusts from about 42 s, the face softens from 52 s, and by
  83 s it is a mask. Without the anchor, the burn starts at 24 s
  (`bench/records/counters.md`, *Round 3*). No tested counter makes a
  two-minute static take usable.
- **A take of any shot type drifts past about 35 s.** A 90 s
  `shot: dialogue` walk was clean to 35 s; from 45 s the face changed
  and by 60 s the picture was oversaturated and painterly. The check
  passed every seam, because the drift is in the carry
  (`bench/records/drill.md`, ask 1). Before quoting a longer "single
  take", say it will not hold, and offer chains of 35 s or less joined
  by cuts (the *Cuts* row above; `minimax-h3` has the cut rules).

The classification of the vocabulary's other moves is #1200. Until then,
treat any move other than `static` as moving.

`references/levers.md` has every lever, the knobs that are not defaults, and
the ideas that were measured and cut. Read it before proposing a counter.

## 4. Per seam: the check, then the repair

Turn on the per-segment check for any chain you mean to keep. It is a
key on the chain block, not a template variable: `get_workflow`, add
`"check": true, "retries": 1` to `steps[0].pipeline.chain`, and
`save_workflow` the whole document under a project name. A merge patch
replaces `steps` whole.

- `jump_ratio`, `look_ratio` and `cut_ratio` re-roll a failing segment at a
  new seed, from the same carry.
- `texture_ratio` is warn-only: the drift is in the carry, so a re-roll
  inherits it (`bench/records/check.md`; ruling R38).
- A segment whose prompt asks for a cut inside it needs
  `"check": {"cut_ratio_max": null}`, or it re-rolls for nothing.

When a kept take still has one bad interior segment,
`templates/minimax/repair-segment` re-renders it between its neighbours
(`bench/records/repair.md`). It needs the chain run with `keep_segments:
true`. `references/run-and-repair.md` has both procedures, their costs, and
the one break a repair cannot fix.

## 5. Per piece: the soundtrack

- Under cut scenes, use one bed: `templates/assemble-and-score`. For room
  tone under dialogue gaps, `find_loop_bed` picks a window worth looping.
- Across chain seams, the voice carries in the guide.
- For music longer than one Music 3 track (300 s practical), use
  `templates/minimax/music-long`. The lyric, not the ceiling, sets a
  track's length, so a score past about 90 s is `music-long` sections
  too: tag-only instrumental bodies asked for 280 s and 300 s came back
  91 s (13 tags) and 62 s (9 tags) (`bench/records/drill.md`).
- For narration longer than one TTS take (about 40 s of speech), use
  `templates/narration-long`, or `split_takes` and then
  `templates/narration-from-takes` to edit the takes first.
- To put a track under a picture, use `pair_audio`.
- To give a finished silent picture new sound, use the H3 audio pass.

`references/audio.md` has each recipe.

## 6. Quality, speed and cost

Defaults favour quality. `references/quality-and-speed.md` has the measured
costs, what is still being measured, and the dial. In short, the chain's
`cost` row is 13.5 min on an RTX 3090 for 3 segments of 124 frames. The
bench's two-minute static take took 69 min, against 52 min for the plain
guide chain. Quote `plan.estimate` from `validate_workflow`, never these.

## Run and judge

1. `validate_workflow` with the arguments. Quote `plan.estimate`. A checked
   chain quotes its worst case: 1 + (N - 1)(1 + `retries`) segment renders.
   A saved copy has no `cost` row of its own (its estimate comes back
   `basis: unknown`), so validate the template with the same arguments and
   scale that figure by the worst case over N.
   Get the go-ahead, then `run_workflow` with `acknowledged_cost`.
2. `wait_for_job`, then `get_job`. Each entry carries `subfolder`: `final`
   is the take or the cut, and `intermediate` is the scratch. Kept segment
   files sit beside the take. Read the job's `warnings` for
   `chain_check_failed`. A warning names the segment, the measures and the
   kept attempt.
3. Judge seams with `assess_output(probe="analyze_seams")` and
   `get_output_frames(seams=true)`. Judge drift with `get_output_frames(at=...)`
   late in the take, and `compare_outputs(texture_bands=true)` on a patch
   the subject never enters, against segment 1. Listen with
   `get_output_audio`. Hand the user the gallery `url`.
4. For an editor, `export_timeline` writes the cut as an OTIO `.otioz` that
   DaVinci Resolve 18.5+ imports with one clip per shot
   (`references/run-and-repair.md`).

## Sources

The long-form design (`docs/proposals/long-form-media.md`, sections 1-2).
The bench records `counters.md`, `check.md`, `repair.md`, `structural.md`,
`noise-floor.md`, `determinism.md` and `for-don.md` under `bench/records/`.
`docs/WORKFLOW_GUIDE.md`, *Checking each segment*. `docs/TASKS.md`.
