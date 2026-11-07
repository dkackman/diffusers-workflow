# Move trial: which named moves each video family honours

Stage G of #1141 (#1148), run on lem (2x RTX 3090, develop @ 2c371a1d) on
2026-11-02. Each of H3, LTX-2.5 and Kandinsky 6 got the same six moves from
the `cinematography` vocabulary, worded by that family's compile table, at
one seed on one fixed subject. Each clip was judged from `get_output_frames`
(a 4-frame contact sheet, then frame 0 against the last frame at 480 px where
the sheet was not conclusive). The verdicts are in the skill's table
(`plugins/dw/skills/cinematography/SKILL.md`, *A hint, not a promise*) and in
the `measured (#1148)` column of each `references/compile-*.md` move table.

## Method

- **Subject:** an old red cast-iron telephone box on a wet cobbled square at
  dusk, a black iron lamp post beside it, a stone fountain to its left, brick
  shopfronts with lit windows behind, low mist. Nothing in the frame moves on
  its own, so any change in framing is the camera.
- **Shot size:** `wide` for every clip, so a dolly-in or crane has room to
  show.
- **Seed:** 42 on every clip. Everything else is the template's default.
- **Templates:**
  - LTX-2.5: `templates/ltx2/text-to-video`, 960x544, 121 frames at 24 fps.
  - Kandinsky 6: `templates/kandinsky6/text-to-video`, 864x480, 121 frames.
  - H3: `templates/minimax/video-with-audio`, 960x544, 124 frames, the
    template's 5-step turbo LoRA.
- **Prompt per family:** that family's own format, with the move sentence
  taken verbatim from its compile table and only the placeholders filled.
  - LTX-2.5: one sentence, "A wide shot frames an old red cast-iron telephone
    box and the cobbled square around it at dusk, <move>." plus the scene
    description. Viewpoint "a front-facing angle". Reveals: dolly-in "the
    black handset behind its glass panes"; dolly-out "the whole square and
    its shopfronts"; truck-left "the stone fountain"; crane-up "the rooftops
    and the whole cobbled square"; orbit "the narrow side street behind it".
  - Kandinsky 6: "A wide shot of an old red cast-iron telephone box standing
    alone on a cobbled square at dusk." then exactly one camera sentence,
    then the scene, then `<AUDCAP>A quiet town square at dusk, distant
    traffic hum and the soft trickle of the fountain.<ENDAUDCAP>`.
  - H3: Context-IR, `integrated_multimodal_description: [Shot 1] Cinematic,
    live-action, a wide shot frames ...` with the move sentence in the shot,
    then `overall_soundscape:`.
- **Verdicts:** *honoured* - the move landed in the right direction at a
  visible size; *partial* - motion of the right kind, but small or mixed with
  another move; *ignored* - no motion of that kind (the skill would mark it
  "hint only, rarely honoured").

## Results

| move | H3 | LTX-2.5 | Kandinsky 6 |
| --- | --- | --- | --- |
| `static` | honoured (`8261c82d7308`) | honoured (`417ce2c24c44`) | honoured (`80b30f24fba1`) |
| `dolly-in` | honoured (`8d6786c0b6a2`) | honoured (`9223e8eb9161`) | honoured (`c52734282a38`) |
| `dolly-out` | honoured (`db5271519874`) | honoured (`0602c1400c3e`) | honoured (`26843efb6392`) |
| `truck-left` | partial (`5cf0b0af6608`) | honoured (`6d46ed86d897`) | partial (`2192d6ca5dcf`) |
| `crane-up` | honoured (`a9644e35f0fb`) | honoured (`7dda520e6667`) | partial (`99b30c30371e`) |
| `orbit` | honoured (`28d0f4b327e8`) | honoured (`5134c962d4aa`) | partial (`3361d11be518`) |

No cell was ignored. LTX-2.5 honoured all six; H3 missed only the truck;
Kandinsky 6 honoured the axial moves (static, dolly in, dolly out) and turned
each lateral or vertical one into a forward drift.

### What the frames showed

**LTX-2.5**
- static: the box, lamp and fountain hold their place over all 5 s.
- dolly-in: the box grows to fill most of the height; the fountain leaves the
  left edge.
- dolly-out: opens on the box's glass panes filling the frame and ends on the
  whole square with the fountain and shopfronts.
- truck-left: the box slides from centre to the right third; the fountain
  comes in from the left, with parallax against the shopfronts.
- crane-up: rises to a high angle looking down on the box, its paved circle
  and the whole square.
- orbit: the box stays central while the buildings behind it swing past; the
  last frame looks down a side street the first did not show.

**Kandinsky 6**
- static: unchanged framing over 5 s.
- dolly-in: a steady push from wide to the box filling the frame.
- dolly-out: opens close on the box's panes and ends wide on the square.
- truck-left (partial): the box moves right in frame but also grows; the
  fountain and lamp hardly move against the background, so it reads as a
  forward-right drift more than a truck.
- crane-up (partial): the camera lifts slightly (more of the fountain rim and
  paving shows, the box's base leaves the bottom edge) but never rises above
  the box or looks down.
- orbit (partial): mostly a push-in; the box's right side panel turns a
  little into view, the background does not swing.

**H3**
- static: unchanged framing; only the mist moves.
- dolly-in: from wide to the box's door filling the frame.
- dolly-out: from the box close at frame left to a very wide view of the
  square and the terrace.
- truck-left (partial): the box, lamp and fountain shift right by about a
  tenth of the frame width together, with little parallax against the
  buildings - a small drift, not a truck.
- crane-up: rises above the square until the box is at the bottom edge and
  the rooftops and street fill the frame.
- orbit: the fountain moves from beside the box to behind it, the lamp from
  its left to its right, and a different row of buildings is behind it.

## Limits

One seed per cell is one sample, not a rate: a cell marked honoured can miss
on another seed, and a partial can land. The subject was chosen to show
motion clearly (a fixed object with depth cues); a moving subject or a
close-up gives a move less to show. The seven moves not run (`truck-right`,
`crane-down`, `tracking`, `whip-pan`, `push-in`, `handheld`, `crash-zoom`)
stay unmeasured, and FLUX 3 Klein is a still with no move to measure.

## Notes on the run

- The Kandinsky 6 weights (`kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers`,
  24.9 GB) were not on lem and were downloaded by the first Kandinsky job.
  It is the catalog model already in use (run on lem 2026-10-06), and the
  approved K6 clips could not run without it, so it was taken as part of the
  approved spend.
- Every clip carried a quiet-audio warning; the prompts asked only for
  ambience. Audio was not part of this trial.
- GPU time: about 40 minutes of generation across both cards, plus the
  Kandinsky download.
