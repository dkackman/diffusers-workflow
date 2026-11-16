# Move seeds: all 13 named moves, two seeds each

Stage A of #1260 (#1379). It measures H3 on all 13 move ids of the
`cinematography` vocabulary at two seeds. Before this, six moves had one
seed each (#1148, `2026-11-higgsfield/move-trial.md`) and seven were not
measured. It ran on lem (RTX 3090, `cuda:0`) on 2026-11-16, one H3 job at a
time. The verdicts are in the `measured (#1148, #1260)` column of
`plugins/dw/skills/cinematography/references/compile-h3.md`. Stage B
of #1260 (#1380) measured LTX-2.5 the same way. Its section,
[LTX-2.5 (stage B)](#ltx-25-stage-b), is at the end, and its verdicts are in
`compile-ltx-2.5.md` and the skill's own table. Everything above that
section is H3.

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

## LTX-2.5 (stage B)

Stage B of #1260 (#1380), written by model `claude-opus-5-5` via provider
`anthropic`. The 13 move ids were run on LTX-2.5 at seeds 42 and 7 on lem on
2026-11-16, on develop @ 1b891f98.

### Method: what differs from H3

The subject, walker, seeds, grading views and verdicts are the H3 method's
above. These differ:

- **Template:** `templates/ltx2/text-to-video` at its defaults: 121 frames
  at 24 fps (5.0 s). Only `prompt` and `seed` were passed.
- **Cards:** lem's two RTX 3090s (`cuda:0`, `cuda:1`), one LTX job per card,
  so two clips could run at once.
- **Seeds:** #1148's six seed-42 LTX jobs are cited (†), and their seed-7
  clips reuse #1148's prompts. The seven new moves ran at both seeds.
- **Prompt:** LTX's flowing caption, not Context-IR.
  - It opens "A wide shot frames an old red cast-iron telephone box and the
    cobbled square around it at dusk, <move sentence>."
  - The scene follows: "The box stands still on the wet cobbles, its small
    glass panes glowing faintly from a bulb inside, a black iron lamp post
    beside it casting a warm pool of light, a stone fountain to its left and
    a row of brick shopfronts with lit windows behind it. A light mist is
    drifting low across the cobbles. Cool blue dusk light mixing with warm
    lamplight, photorealistic, fine detail on the wet stone and the chipped
    red paint."
  - `<viewpoint>` was "a front-facing angle" in every row.
  - `crane-down` opens "A wide shot from high above frames ...", so the move
    has somewhere to come down from.
  - `tracking` and `handheld` open on the walker: "A wide shot frames a man
    in a dark coat walking across a cobbled square at dusk, past an old red
    cast-iron telephone box, <move sentence>."
- **The move sentences, as run** (the compile row with its placeholders
  filled):

| move | sentence in the prompt |
| --- | --- |
| `static` | "captured from a front-facing angle, the camera remains static" |
| `dolly-in` | "seen from a front-facing angle, as the camera dollies in toward the telephone box, the black handset behind its glass panes comes into view" |
| `dolly-out` | "seen from a front-facing angle, as the camera dollies back from the telephone box, the whole square and its shopfronts come into view around it" |
| `truck-left` | "seen from a front-facing angle, as the camera trucks left alongside the telephone box, the stone fountain slides into view from the left" |
| `truck-right` | "seen from a front-facing angle, as the camera trucks right alongside the telephone box, the lamp post and the shopfronts to the right slide into view from the right" |
| `crane-up` | "seen from a front-facing angle, as the camera cranes upward above the telephone box, the rooftops and the whole cobbled square come into view below" |
| `crane-down` | "seen from a front-facing angle, as the camera cranes down to the telephone box's level, the shopfronts come into view behind it" |
| `orbit` | "seen from a front-facing angle, as the camera arcs around the telephone box, the narrow side street behind it comes into view behind it" |
| `tracking` | "seen from a front-facing angle, the camera tracks alongside the man at their pace, the telephone box and the lit shopfronts passing through the frame" |
| `whip-pan` | "seen from a front-facing angle, the camera whip-pans left in a blur, landing on the stone fountain" |
| `push-in` | "seen from a front-facing angle, the camera pushes in slowly toward the box's glass door, until its black handset fills the frame" |
| `handheld` | "seen from a front-facing angle, the handheld camera sways slightly as it follows the man in the dark coat, the stone fountain drifting into view" |
| `crash-zoom` | "seen from a front-facing angle, the camera crash-zooms in on the telephone box, until its glass panes fill the frame" |

### Results

Each cell gives the verdict and the job id. The seed-42 cells marked † are
#1148's jobs.

| move | seed 42 | seed 7 |
| --- | --- | --- |
| `static` | honoured (`417ce2c24c44`†) | honoured (`425b23694f8d`) |
| `dolly-in` | honoured (`9223e8eb9161`†) | honoured (`6f59682880a7`) |
| `dolly-out` | honoured (`0602c1400c3e`†) | honoured (`dab82ef05edd`) |
| `truck-left` | honoured (`6d46ed86d897`†) | honoured (`ce019eac6774`) |
| `truck-right` | partial (`a5b1e2fac29c`) | honoured (`24d392e88c10`) |
| `crane-up` | honoured (`7dda520e6667`†) | honoured (`c86f14f420bf`) |
| `crane-down` | ignored (`f08213e1b3ac`) | honoured (`e3aeae32dd3b`) |
| `orbit` | honoured (`5134c962d4aa`†) | honoured (`22608f463b7d`) |
| `tracking` | ignored (`67adebfa0d2a`) | partial (`b9675fc725a7`) |
| `whip-pan` | partial (`f606f271c878`) | partial (`f13d88c502a3`) |
| `push-in` | honoured (`f613438de363`) | honoured (`18cd7999d3ea`) |
| `handheld` | ignored (`be0528f18824`) | ignored (`10ffc93fd38b`) |
| `crash-zoom` | partial (`dc1adfad3fa4`) | partial (`a62565c64216`) |

- Seed 7 agreed with #1148's seed 42 on all six moves measured before.
- `handheld` was ignored on both seeds, so it got the one rerun (below). It
  was ignored again, and its compile cell reads `ignored, ignored: hint
  only, rarely honoured`.
- Three moves differ between seeds: `truck-right` (partial, honoured),
  `crane-down` (ignored, honoured) and `tracking` (ignored, partial).
- Against H3: LTX-2.5 trucks better (left honoured on both seeds) but is
  weaker on the fast moves. Its whip-pan and crash-zoom come out as a
  smooth reveal and a steady push, and it never follows the walker for
  long.

### What the frames showed (new clips)

- **static, 7:** the framing holds from frame 0 to 120; only the mist and
  the light move.
- **dolly-in, 7:** the box grows from about half the frame height to filling
  it, its sign readable, drifting left.
- **dolly-out, 7:** it opens on the door filling the frame and ends with the
  box at about a third of the height, the fountain and lamp beside it and
  the terrace behind.
- **truck-left, 7:** the box slides right by about an eighth of the width
  and the fountain slides in from the left, with a slight close-in.
- **truck-right, 42 (partial):** the box holds near left of centre while the
  background slides left: the fountain exits left and a terrace with shop
  signs enters right. It is more an arc than a truck, with a slight
  pull-back.
- **truck-right, 7:** the box slides left by about a sixth of the width,
  faster than the fountain (parallax), and closes in.
- **crane-up, 7:** it opens close on the box at eye level, then rises and
  pulls back, ending with the box small at the bottom edge and the square
  and terrace opening out. The rise is mixed with a pull-back, but the rise
  is plain.
- **crane-down, 42 (ignored):** reversed. It opens close on the box at eye
  level, then rises and pulls back to a high view down on the square, the
  box small.
- **crane-down, 7:** it opens high on the terrace with the box's top at the
  bottom edge, then descends and closes in until the box stands at eye
  level mid-frame.
- **orbit, 7:** the box holds centre while the background swings from the
  shopfront row to a side street with sky. The fountain moves from the left
  to behind the box.
- **tracking, 42 (ignored):** the man walks right and exits frame right by
  about 1.4 s. The camera drifts the other way, the box moving from the
  left edge to centre; he reappears small in the background.
- **tracking, 7 (partial):** it follows the man to about 1.4 s, holding him
  left of centre while the background slides. Then he falls out at the left
  and the camera settles on the box and fountain.
- **whip-pan, 42 (partial, 24-frame sheet):** a leftward move reveals the
  fountain within about 1.3 s, the box going from centre to the right. It is
  smooth, with no blur: not a whip.
- **whip-pan, 7 (partial, 24-frame sheet):** the box nearly holds centre,
  drifting right by about a tenth, and the lamps and fountain are revealed
  by a pull-back, with no blur. The reveal lands; the pan does not.
- **push-in, 42:** the box goes from about 90% of the frame height to
  filling it, the handset visible.
- **push-in, 7:** from about 60% of the height to filling the frame, the
  handset visible.
- **handheld, 42 (ignored):** the man walks right and exits by about 1.4 s.
  The camera swings the other way onto the box and fountain, then settles.
  There is no follow, and consecutive frames 60, 62, 64 and 66 at 480 px
  show the box fixed: no jitter.
- **handheld, 7 (ignored):** the man exits right by about 1.4 s. The camera
  drifts left and pushes toward the box until it fills most of the height.
  There is no follow; frames 60-66 show a smooth drift, not a shake.
- **crash-zoom, 42 (partial, 24-frame sheet):** a steady push over about
  0-3.9 s, the box going from about 70% of the height to filling the frame.
  The lamp shows parallax, so it is a dolly, not a zoom, and not fast.
- **crash-zoom, 7 (partial, 24-frame sheet):** the box is about a third of
  the height and grows slowly to about 1.1 s, then steadily until it fills
  the frame by about 4 s. It speeds up a little, but it is not a crash.

### The rerun

- **`handheld`, seed 42, job `2305175384ca`:** ignored. The revised
  sentence was "seen from a front-facing angle, the shot is handheld, the
  camera shaking with small jolts as it follows the man in the dark coat and
  keeps him in frame, the stone fountain drifting into view", with the same
  walker opening and scene.
- **What it showed:** the man walks right and leaves the frame by about
  1.4 s. The camera then arcs left onto the box and fountain and holds.
  Consecutive frames 60, 62, 64 and 66 at 480 px show no jitter.
- **Outcome:** the original compile sentence stays, and the cell reads
  `ignored, ignored: hint only, rarely honoured`. LTX-2.5's text-to-video
  has given no handheld shake on three clips.
- `crane-down` and `tracking` each had one seed that was not ignored, so
  neither got a rerun.

### Limits

- **Two seeds are two samples, not a rate**, as for H3.
- **The walker leaves the frame.** On every LTX `tracking` and `handheld`
  clip the man walks out within about 1.4 s. A follow that never starts and
  a subject who walks out of shot look alike on a sheet. A walker who stays
  longer might grade differently.
- **The shake is fine motion.** Both handheld seeds and the rerun were
  checked on consecutive frames, and none showed jitter.
- **Grading was done on 768 px and 1024 px contact sheets** (24 frames for
  `whip-pan` and `crash-zoom`), not on the full-size video.
- **Audio was not part of the trial.** Most clips warned of near-silent
  audio (about -41 to -56 dBFS); the prompts ask for no sound.

### Notes on the run

- **`get_diffusers_state`**, before the first clip and after the rerun: both
  read `status: idle`, version `0.44.0.dev0`, commit
  `43e30028af5120d8b0868cb6975b9fad39cc4fab`, `revert: false`.
- **Timing:** about 36-50 s of job time per clip with weights cached (for
  example, `a62565c64216`: denoise 31.4 s over 8 passes, decode 3.2 s). The
  first clips on each card ran longer (up to 152 s) while the weights
  loaded or while waiting on the other card's job.
- **GPU total:** 20 new clips at 1209.1 s plus the rerun at 49.1 s, so
  1258.2 s, or **21.0 GPU-min**, against a cap of 45. That is summed from
  each job's started-to-finished time, so it overcounts a little where the
  two cards overlapped.
