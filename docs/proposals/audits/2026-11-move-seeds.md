# Move seeds: all 13 named moves, two seeds each

Stage A of #1260 (#1379). It measures H3 on all 13 move ids of the
`cinematography` vocabulary at two seeds. Before this, six moves had one
seed each (#1148, `2026-11-higgsfield/move-trial.md`) and seven were not
measured. It ran on lem (RTX 3090, `cuda:0`) on 2026-11-16, one H3 job at a
time. The verdicts are in the `measured (#1148, #1260)` column of
`plugins/dw/skills/cinematography/references/compile-h3.md`. Stage B
of #1260 adds LTX-2.5 to this doc and the skill's own table.

## Method

This follows #1148's protocol (`2026-11-higgsfield/move-trial.md:14-44`),
extended:

- **Subject:** the same old red cast-iron telephone box on a wet cobbled
  square at dusk, with a black iron lamp post beside it, a stone fountain
  to its left, brick shopfronts with lit windows behind, and low mist.
  Nothing in the frame moves by itself, so any change in framing is the
  camera.
  - `tracking` and `handheld` use a different subject, because their
    sentences follow someone: "a man in a dark coat walking across a
    cobbled square at dusk, past an old red cast-iron telephone box", with
    the rest of the scene unchanged (#1260 Q2). These two rows measure a
    follow of a walker, not a fixed object.
- **Template:** `templates/minimax/video-with-audio` at its defaults: 960x544,
  124 frames at 24 fps, the template's turbo LoRA. Only `prompt` and `seed`
  were passed.
- **Seeds:** 42 and 7.
  - The six moves #1148 measured were not rerun at seed 42; its jobs are
    cited.
  - Their seed-7 clips use #1148's exact prompts, so seed is the only
    variable.
  - The seven new moves were run at both seeds.
- **Prompt:** H3 Context-IR, the same frame for every clip:
  - It opens `integrated_multimodal_description: [Shot 1] Cinematic,
    live-action, a wide shot frames an old red cast-iron telephone box
    standing alone on a cobbled square at dusk.`
  - Then comes the scene description, unchanged from #1148.
  - Then comes the move sentence, verbatim from the compile table with its
    placeholders filled, ending "as a light mist drifts low across the
    cobbles."
  - It closes with `overall_soundscape: A quiet town square at dusk: a
    distant low hum of traffic, the soft steady trickle of the stone
    fountain, and an occasional faint gust of wind across the wet cobbles.`
  - For `crane-down` only, the opening reads "a wide high-angle shot frames",
    so the move has somewhere to come down from.
- **The move sentences, as run:**

| move | sentence in the prompt |
| --- | --- |
| `static` | "The camera holds a static shot as a light mist drifts low across the cobbles." |
| `dolly-in` | "The camera pushes in with large amplitude toward the telephone box as a light mist drifts low across the cobbles." |
| `dolly-out` | "The camera pulls out with large amplitude, revealing the whole cobbled square, as a light mist drifts low across the cobbles." |
| `truck-left` | "The camera trucks left alongside the telephone box as a light mist drifts low across the cobbles." |
| `truck-right` | "The camera trucks right alongside the telephone box as a light mist drifts low across the cobbles." |
| `crane-up` | "The camera pedestals up with large amplitude, rising above the telephone box, as a light mist drifts low across the cobbles." |
| `crane-down` | "The camera pedestals down to the telephone box's eye level as a light mist drifts low across the cobbles." |
| `orbit` | "The camera moves in an arc shot around the telephone box as a light mist drifts low across the cobbles." |
| `tracking` | "The camera holds a tracking shot, matching the man's pace, as a light mist drifts low across the cobbles." |
| `whip-pan` | "The camera pans left with large amplitude at fast speed, revealing the stone fountain, as a light mist drifts low across the cobbles." |
| `push-in` | "The camera pushes in with small amplitude at slow speed toward the telephone box's glass door as a light mist drifts low across the cobbles." |
| `handheld` | "The camera shakes slightly as it follows the man in the dark coat as a light mist drifts low across the cobbles." |
| `crash-zoom` | "The camera zooms in with large amplitude at fast speed on the telephone box as a light mist drifts low across the cobbles." |

- **Grading:** each clip was judged with `get_output_frames`.
  - The view was a 12-frame contact sheet or a few spaced frames, plus
    frame 0 against frame 123.
  - `whip-pan` and `crash-zoom` got a 24-frame sheet over the whole clip.
  - `handheld` seed 42 also got consecutive frames (60, 62, 64, 66) at
    480 px, to look for jitter.
- **Verdicts** (`move-trial.md:41-44`):
  - *honoured*: the move landed in the right direction at a visible size.
  - *partial*: motion of the right kind, but small or mixed with another
    move.
  - *ignored*: no motion of that kind.
- **Rerun rule:** a move ignored on both seeds gets one revised phrasing, run
  once at seed 42.

## Results

Each cell gives the verdict and the job id. The seed-42 cells marked † are
#1148's jobs.

| move | seed 42 | seed 7 |
| --- | --- | --- |
| `static` | honoured (`8261c82d7308`†) | honoured (`cf7ed7a0fd9c`) |
| `dolly-in` | honoured (`8d6786c0b6a2`†) | honoured (`1b3f16e691b6`) |
| `dolly-out` | honoured (`db5271519874`†) | honoured (`40e25fecb884`) |
| `truck-left` | partial (`5cf0b0af6608`†) | partial (`ee276ce6cf10`) |
| `truck-right` | partial (`0a48d566b711`) | partial (`06eb58105e52`) |
| `crane-up` | honoured (`a9644e35f0fb`†) | honoured (`9cbbf3e7bdc8`) |
| `crane-down` | honoured (`2084dbea20f1`) | honoured (`fbe227769c6d`) |
| `orbit` | honoured (`28d0f4b327e8`†) | honoured (`822aff4c185a`) |
| `tracking` | partial (`507c71743ec2`) | honoured (`f972c0634e6c`) |
| `whip-pan` | partial (`9465ac0ce2fb`) | partial (`bd245caa73c7`) |
| `push-in` | honoured (`e1cc95aeba7a`) | honoured (`47e3d0b21fb6`) |
| `handheld` | ignored (`370e55a59b02`) | partial (`e301c59a2d9c`) |
| `crash-zoom` | honoured (`5f91e4796acd`) | honoured (`2bc7ce4bf013`) |

- No move was ignored on both seeds, so there were no reruns, and every
  compile sentence stands as written.
- Seed 7 agreed with #1148's seed 42 on all six moves measured before.
- Three moves differ between seeds:
  - `tracking` is partial on 42 and honoured on 7;
  - `handheld` is ignored on 42 and partial on 7.
  - `whip-pan` is partial on both, but in the same way: the reveal lands
    and the pan does not.
- The lateral moves (both trucks and the whip-pan) are H3's weak spot. The
  axial and vertical moves (dolly, push-in, crash-zoom, both cranes) and the
  orbit landed on every clip.

### What the frames showed (new clips)

- **static, 7:** the framing is identical from frame 0 to frame 123; only
  the mist moves.
- **dolly-in, 7:** the box goes from about 40% of the frame height to filling
  it, its sign readable. The lamp post swings from right of the box to the
  right edge.
- **dolly-out, 7:** it opens on the box's door filling the frame height and
  ends on the whole terrace of shopfronts, the box small at centre-left.
- **truck-left, 7 (partial):** the box slides from left of centre to centre,
  about a ninth of the width. The fountain and shopfronts barely move. It is
  a small drift, not a truck, the same as seed 42.
- **truck-right, 42 (partial):** the box slides left from the right third
  toward centre, about an eighth of the width, and a new building edge
  enters at the right. The direction is right but the move is small, with
  only some parallax.
- **truck-right, 7 (partial):** the box slides left by about a ninth of the
  width and the fountain hardly moves. The same small drift as seed 42.
- **crane-up, 7:** it rises from eye level to a high view down on the
  square. By the end the box has left the bottom edge and the terrace fills
  the frame.
- **crane-down, 42:** it opens high, looking down on the box's roof and the
  fountain basin, and descends until the box's door is at eye level with
  the rooftops and sky back in frame.
- **crane-down, 7:** it opens high over the hexagonal fountain basin and
  ends at eye level, the box filling the frame height.
- **orbit, 7:** the box turns to show its left side panel, the fountain
  moves from beside it to behind it, and the shopfronts behind change. It
  closes in as it arcs.
- **tracking, 42 (partial):** the camera travels right with the man, but at
  about half his pace. The box goes from the right edge to centre-left,
  while he crosses from the left third to right of centre. It is a follow
  that lags, not a matched track.
- **tracking, 7:** the man stays right of centre while the square slides
  past. The box goes from right of centre to the left edge, the fountain
  leaves the frame at left, and new shopfronts come in from the right.
- **whip-pan, 42 (partial):** a fast swing, with heavy motion blur around
  frames 40-55, that reveals the fountain. But the camera arcs around the
  box instead of panning off it: the box stays near centre, the lamp moves
  from its left to its right, and the fountain from the left foreground to
  behind it. It is a fast arc, not a pan.
- **whip-pan, 7 (partial):** the same arc at a slower rate, with background
  blur around frame 60. The box holds centre while the fountain moves from
  the left edge to behind its right side. The reveal lands; the pan does
  not.
- **push-in, 42:** a slow, steady push. The box grows from about 60% of the
  frame height to filling it, the handset clear behind the panes, while the
  fountain stays in frame at left. The amplitude is small, as asked.
- **push-in, 7:** the box grows from about 70% of the frame height until its
  top leaves the frame, a gentler push than dolly-in.
- **handheld, 42 (ignored):** a locked-off frame. The box, lamp and fountain
  hold their place across all 12 sheet frames and on consecutive frames 60,
  62, 64 and 66 at 480 px. There is no jitter, and the man walks away into
  the background with no follow.
- **handheld, 7 (partial):** a slow follow. The box drifts from centre-right
  to centre-left as the camera turns with the man walking right. The follow
  half of the sentence lands, but there is no visible shake on the sheet.
- **crash-zoom, 42:** the frame holds wide to about 0.9 s, zooms fast until
  about 1.8 s, then holds. The box grows from about a quarter of the frame
  height to filling it, and the perspective stays flat: the fountain and
  lamp scale with it.
- **crash-zoom, 7:** the same shape. It is wide to about 1.1 s, zooms fast
  to about 1.8 s, then holds on the box, fountain and lamp.

## Limits

- **Two seeds are two samples, not a rate.**
  - A move honoured on both can still miss on a third seed.
  - A split cell (`partial, honoured`) says the move varies by seed, and
    nothing more.
- **The walker changes the subject for `tracking` and `handheld`.**
  - Those rows measure a follow of a person crossing the frame, which is
    not the fixed box the other eleven moves use.
- **The shake in `handheld` is fine motion.**
  - A 12-frame sheet samples every half second, so it can miss jitter.
    Seed 42 was checked on consecutive frames and showed none. Seed 7 was
    judged from the sheet, where the follow is plain and no shake shows.
- **Audio was not part of the trial.**
  - Most clips carried a quiet or near-silent audio warning: the prompts
    ask only for ambience.
- **No revised phrasing was tried.**
  - The rerun rule needs a move ignored on both seeds, and none was.
    `handheld` came closest.

## Notes on the run

- **`get_diffusers_state`**, before the first clip and after the last:
  - both read `status: idle`, version `0.44.0.dev0`, commit
    `43e30028af5120d8b0868cb6975b9fad39cc4fab`;
  - #1148 ran on develop @ 2c371a1d.
  - #1340's revert window was closed with no revert, so lem stayed on
    diffusers HEAD throughout.
- **Serial:** every clip ran alone on `cuda:0`, one H3 job at a time.
- **A download false positive:**
  - `validate_workflow`'s first plan of the session listed the template's
    turbo LoRA under `downloads_required`.
  - The job's timing showed it loaded from cache (nothing was downloaded),
    and every later validate listed none.
  - It is a possible validate false positive, noted here and not chased.
- **Timing:** each clip took about 78-92 s of job time, with weights cached
  after the first.
- **GPU total:** 20 new clips, 1615.7 s, or **26.9 GPU-min**, against a cap
  of 45.
