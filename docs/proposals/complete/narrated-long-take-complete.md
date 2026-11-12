# Narrated long take: one unbroken H3 shot driven by its narration

_Design, 2026-11-09. Written by `claude-opus-5-5` (anthropic) with Don, from the
`constitution-walk-2` assessment. It is the follow-on to `long-form-media.md` (#1199),
whose playbook this design changes for one shape: a speaking subject in a take
longer than about 35 s. The plan and its stages follow from it._

## Intent

A person asks for one continuous shot of a character speaking a long text, for
example a woman walking toward a backward-dollying camera reciting the Preamble and
the Bill of Rights, four to five minutes with no cut. An agent with only the dw MCP
server, the `dw` plugin's skills and the catalog should produce it without knowing
how dw works inside:

- it plans the take;
- it validates and quotes the cost;
- it runs a cheap probe and judges it with tools that name what failed;
- it carries on to the full length;
- it repairs what broke.

The agent need not be an Opus-class model. The test is a cold session on a smaller
model (Sonnet).

**Why now.** The constitution-walk-2 cut (lem workspace `constitution-walk-2`,
2026-11-08) followed today's playbook to the letter and shows the three limits of
that playbook. It has 15 chained takes, each restarted from a FLUX still, joined by
hard cuts, 6:28 long. Measured on the final cut:

1. **The face degrades.** Inside each take the face goes from the still's pore and
   hair detail to waxy, over-contoured, warmer skin by 16-31 s. Frame saturation
   rises in all 15 takes. She grows in frame although the dolly should hold her
   size.
2. **The picture resets.** All 14 cuts snap back to one of two stills. This is the
   playbook's own advice: "an H3 take of any other shot type drifts past about 35 s
   ... offer chains of 35 s or less joined by cuts" (`plugins/dw/skills/long-form`,
   from `bench/records/drill.md` ask 1).
3. **The delivery halts.** 39% of the runtime is silence (151 s of 388 s). She
   speaks at a natural 135 wpm but averages 83 wpm. Seam pauses average 1.2 s
   (15 over 1 s) and pauses at cuts 2.9 s. Two things cause this:
   - The 1.5 s opening cue each segment carries so H3 does not start its line
     inside the 22 trimmed guide frames (#1308).
   - One short phrase per 7.8 s segment (about 11 words). H3 fills spare time
     with silence before a short line rather than slowing it
     (`minimax-h3/references/dialogue.md`).

The evidence from outside dw has moved since #1199 closed. Two independent
community labs measured the same cause and the same cure:
- matlowai/h3-long-video-research, CC BY 4.0, 2026-11-07;
- roycho87/NerdNotes tests 16-17, CC BY-SA, 2026-11-07.

**The cause.** Decoding each segment and re-encoding its tail as the next guide is
the source of the saturation creep and the texture blow-up, visible by about 25 s.

**The cure.**
- Carry the sampler's own latent tail.
- Re-anchor to segment 1 with a latent-0 keystone and a capstone past the sequence's
  end that enters only late in the denoise.
- Do the re-anchor every segment, or only every 3rd to 6th.

**The result.** Face similarity to clip 1 (ArcFace) held 0.75-0.85 at two minutes
against 0.31 for a plain carry. Saturation drift fell from +44% to +7% over a
30-clip chain. A 2 min 6 s talking take held with the creator's own recorded voice
locked into the audio latent.

dw has every one of these pieces, but none in this combination:
- latent carry (#1150) and the late capstone (#1187) are dark scratch spikes;
- the keystone ships, but as a pixel keyframe or last-image slot;
- the pieces were only ever benched one at a time, never all together, and never
  on a speaking subject with held audio.

## Goals

1. **One unbroken take of a speaking subject, measured to four minutes or more.**
   The face scores within an agreed band of the cast still for the whole take. No
   seam fails `analyze_seams`. The picture never returns to the opening frame.
2. **Natural delivery.** Speech comes from one narration track planned before any
   video. Pauses follow the text: a comma is short, a sentence end longer, a section
   break longest. No pause comes from segment structure. The soundtrack has no
   seams, because the finished video carries the unsliced track.
3. **Lip sync to that track** in every segment, with a measure that catches a
   segment whose mouth does not follow its slice.
4. **Probe, then continue.** A cheap probe (60-90 s) proves the look, the identity
   and the sync. The full take then carries on from the probe's last kept segment
   instead of rendering the probe again.
5. **Find and fix inside the take.** Per-segment checks name a failing segment with
   a measure the agent can read (identity, seam, sync, words). Any segment can be
   re-rendered in place, including the last one, and the take keeps its narration.
6. **Discoverable.** A cold agent finds the shape from `list_workflows`, the
   `long-form` skill and the guides. It reaches a validated, cost-quoted plan
   without reading engine code or bench records. Every rule the agent must follow is
   stated where the agent will read it: a template description, a validate warning,
   a skill line or a check finding.
7. **Measured before it becomes a default.** Every new default and preset ships
   with a bench record, under the same evidence rules as #1199.

## Non-goals

- Training anything (LoRAs, distillation).
- Hybrid windows / one-latent rendering (#1188). It is the structural alternative;
  it is reconsidered only if this design's bench fails (see *If the bench fails*).
- Multi-person staging, cuts within the take, music beds. The narrated take is one
  person, one place and one continuous moment.
- LTX-2.5 and Kandinsky 6. H3 only.
- Re-cutting the existing constitution-walk-2 footage. The picture cannot be retimed
  without jump cuts, so it is regenerated.

## Anti-goals (failure even if it "works")

- **A take that silently degrades.** An agent that reports a four-minute single take
  that drifts at 60 s has failed, even if the take finishes. Drift must reach a
  finding the agent reads, and the skill must say when to stop and cut.
- **Regressing other shapes.** The defaults for `static`, `static_long`, `moving`
  and `dialogue` change only on bench evidence.
- **Knob sprawl.** The agent picks one preset for the shape and does not compose
  counters. New knobs stay knobs; the preset carries the measured combination.
- **Licence leakage.** Code is taken only from MIT, Apache and BSD sources, with
  credit; everything else is concepts only. GPL-3.0 (geiru), AGPL-3.0 (H3MD), the
  no-redistribution licence (Smite79) and unlicensed repos (Longshot) contribute
  concepts only. matlowai (CC BY 4.0) and NerdNotes (CC BY-SA) are research notes:
  cite them in `docs/ATTRIBUTIONS.md`.
- **Breaking the text budgets.** Skills stay inside their byte caps (the H3 skill is
  at its 12 KiB cap). The MCP surface stays inside `SURFACE_BUDGET`. Detail goes in
  references and guides.

## Constraints

- **Hardware.** lem is the only CUDA host (two 24 GB cards). int8 H3 needs about
  65 GB of host RAM, so one int8 job runs per host at a time (#1226, #1227). Don
  deploys lem and restarts its server.
- **H3 grid.**
  - `num_frames` is 17n+5, from 124 to 345, at 24 fps.
  - The canvas is a multiple of 32, at most 1344x768.
  - Guides sit on the 5/22/39/56... grid.
  - H3's audio latent runs at 40 steps/s, so only frame counts divisible by 3 land
    on the audio grid. 192 and 243 do; 209, today's constitution-walk segment
    length, does not, and neither does the 22-frame guide. NerdNotes test 18 saw
    held audio build about 17 dB over 10 clips at an off-grid length.
- **Engine refusals to lift deliberately.** `experimental_latent_carry` refuses
  `hold_audio` and `match_audio`. The reason is that the spike carries only
  generated audio latents (`chain_rules.py`, `chain_latent_carry_problem`). This
  design carries video latents and takes each segment's audio from the master track.
- **Fixed engine behaviour.** `for_each` expands before a run, so a list made by a
  step cannot drive a `for_each` in the same workflow. `narration-long` gets round
  this with a child workflow.
- **Repo rules.**
  - TDD.
  - The architecture ratchet: function length, module count, and the map row in
    `docs/ARCHITECTURE.md` for any new owner.
  - Ruff.
  - Plugin skills are pinned by `tests/test_plugin_skills.py`.
  - A commit headline never reads `fix: #N` for a staged parent (it auto-closes
    the parent).
- **Tier.** The bench never measured a chain at the constitution-walk settings
  (1344x768, int8, base model, 20 steps, no LoRA). The #1199 bench ran at 960x544,
  int4, on the turbo LoRA. Dialogue drops the turbo LoRA. The new bench runs at the
  shape's real tier.

## Design

### 1. Narration first

The take's soundtrack is one narration track, made and approved before any video.

**Two sources, both existing or nearly so:**

| Source | Use | What exists |
|---|---|---|
| Voice clone | Any new script. The voice comes from a short H3 clip of the character, so the voice matches the face. | `narration-long` / `narration-from-takes` (Audio8-TTS, sentence-bounded takes, `gap_seconds`); `check-narration` |
| Harvested H3 lines | A piece that already has verified H3 dialogue (constitution-walk-2's 48 kept phrases) | `check-script`, `split_takes`, `slice_audio`, `concat_videos` |

**New: a retime task** (working name `retime_narration`). It takes a speech track,
its text, and pause rules by punctuation class. Defaults to tune by ear:
- comma: 0.25-0.4 s;
- sentence end: 0.5-0.7 s;
- section break: about 1 s.

It finds the gaps from word timings, through the existing Whisper/ASR seam that the
words check and `check-narration` already use. It shortens or lengthens each gap to
its class, crossfading over room tone so the gap is never digital silence. It
returns the retimed track and a JSON word map: each word's start and end.

**New: layout onto the chain grid.** Given the word map, the segment length
(`num_frames`) and the guide length, place each segment boundary in a word gap
mid-flow, not at a sentence end. Nudge nearby gaps within their class bounds so each
boundary lands on the frame grid. Emit per segment:
- the words it carries, which become its `<d>` line and its words-check `lines`;
- its audio slice bounds.

Segments stay a fixed length (my call), so the chain engine needs no variable-length
segments. The layout absorbs the slack in the gaps. This may be the same task as the
retime or a second one; the builder decides.

### 2. A guide chain held to the narration

Each segment holds its slice of the master track (`hold_audio`, sliced by
`match_audio`), and the finished take is muxed with the unsliced track. The engine
already plans this combination (`tests/test_chain.py:893`). Nothing exposes it, and
it has never run at length.

Because H3 generates no speech:
- the #1308 head clip cannot happen;
- prompts carry no "first second and a half" cue and no "draws a breath";
- the `<d>` text is still given, so the mouth has the words.

The guide's carried audio is the held slice, re-encoded fresh from the master every
segment. Generated audio is never carried forward; both community labs measured
carried audio degrading.

Segment lengths default to the audio grid (192 or 243 frames, divisible by 3). The
validator warns when a held-audio chain's `num_frames` or guide length is off-grid.
Whether 39 guide frames serve better than 22 here is a bench question.

### 3. A drift counter for a moving, speaking subject (#1225)

The candidate preset (working name `shot: narrated`) combines four pieces:

- **Latent carry.** The next guide's video rows are the previous segment's own final
  latents, not a decode and re-encode. #1150's spike, made a real feature that
  accepts held audio.
- **Latent keystone.** Segment 1's clean frame-0 latent is locked into latent 0 of a
  later segment and restored after sampling, so it anchors detail without reaching
  the output. This is a new keystone mode beside today's pixel slots.
- **Late capstone.** Segment 1's late frame goes one latent past the segment's end
  and enters only below a sigma gate (#1187; spike `experimental_capstone`).
- **Cadence.** Keystone and capstone apply every Nth segment (N to be benched; the
  community found 3-6 enough). No anchor runs on the segments between.

Two existing pieces are arms, not assumptions:
- the per-channel latent stat anchor to segment 1 (`anchor`, at a low strength);
- the shot's camera wording ("the camera never pushes in; she stays the same
  size"), which NerdNotes found decides whether a walk-toward reads as a cut.

**Bench (gate before anything becomes a default).**
- **Shot.** A one-person dialogue walk toward a backward dolly at the shape's real
  tier, narrated by a held track: 90 s first, then 4 min.
- **Arms.**
  - Control: today's `dialogue` preset with held audio.
  - The candidate preset.
  - The candidate without the capstone.
  - The candidate with the stat anchor.
  - `static_long` (keystone slot `last`). This shot's framing is nearly constant,
    so it is the zero-build comparison.
- **Seeds.** Two per arm.
- **Measures.**
  - `analyze_face_drift` against the cast still, per segment;
  - `compare_outputs(texture_bands)` against segment 1;
  - `analyze_seams`;
  - frame saturation;
  - the lip-sync measure (§4).
- **Don judges by eye.**

A winner becomes the narrated preset and the long-form skill's rule for this shape.
If nothing beats the control, see *If the bench fails*.

**Fallback arm, if the candidate holds identity but drifts background or pose:**
per-segment keystones from a progression of clean stills along the walk (FLUX Klein
edits from the cast still). Each segment's last image is a fresh still, so drift is
capped at one segment. Its risk is identity flicker between stills. It needs a
`keystones: [images]` list on the chain.

### 4. Checks that name the failure

Each per-segment check either re-rolls the segment or warns:

- **Identity.** `analyze_face_drift` (SFace against the cast still) as a per-segment
  check measure. Warn-only or re-roll is a bench call: the drift is in the carry, so
  re-rolling may inherit it (R38). A finding names the segment and the score.
- **Sync.** A per-segment measure of whether the mouth follows the held slice. This
  does not exist today: `measure_sync` and `analyze_sync_drift` compare audio with
  audio, not mouth with sound. The builder picks the cheapest credible signal (for
  example, mouth-opening area from face landmarks cross-correlated with the slice's
  envelope) and calibrates it on known-good and known-bad segments. If no signal
  separates them, the measure is dropped and the skill says to judge sync by eye
  with `get_output_frames`.
- **Words.** The existing words check, with `lines` from the layout.
- **Seams.** The existing jump, look and cut ratios.

### 5. Probe, continue, repair

- **Continue a finished take.** Today `resume_from` is crash recovery only. A
  finished chain with every segment kept is refused. Extend it: a kept probe of k
  segments continues to N with the same plan, narration and seed lineage. Segments
  1..k are copied, not re-rendered. The narrated preset's anchors still come from
  segment 1. The quoted cost covers only the segments left to render.
- **Repair.**
  - `repair-segment` covers interior segments of a guide chain today. It must hold a
    narrated take's slice (`hold_audio`) for the segment it re-renders, and keep the
    take's master track.
  - Repairing the last segment (no tail neighbour) is in scope.
  - Repairing two adjacent segments (#1231) is in scope only if the bench shows
    breaks spilling into the next segment.

### 6. One template, one skill path

- **The template.** A template for the shape (working name
  `templates/minimax/narrated-take`), with trait `has-audio` and shape `sequence`,
  taking:
  - the cast still;
  - one scene description: framing, setting, wardrobe and the camera line, stated
    once and repeated into every segment's prompt by the template;
  - the narration track (`asset:`);
  - its layout JSON;
  - `segments`, or "all": probe by default, full on request.

  Prompts are built per segment from the scene description and the segment's words,
  so the agent writes one description, not 40 prompts. Where the engine's
  before-run `for_each` expansion bites, use the `narration-long` pattern (a child
  workflow).
- **A narration template.** Text plus a voice reference goes to a retimed track and
  its layout, or a harvested track goes to a retimed one. It probably composes
  `narration-long` and the new tasks.
- **Skills.**
  - **`long-form`.** Its 35-second rule gains this shape's exception, stated as
    measured: "a speaking subject in one place: narrated take; probe 90 s, then
    continue". It also states the procedure (narration → listen → probe → judge →
    continue → repair) with the tools for each step.
  - **`minimax-h3`.** Gets a one-line pointer only (byte cap).
  - **`script-to-video`.** Its shot list routes a long monologue to this shape.
- **Guides.** A `workflows` guide section, "A narrated long take", carries the
  procedure and the audio-grid rule. Validate warnings carry the rules the agent
  would otherwise have to know: an off-grid segment length, held audio without
  `match_audio`, a cue clause in a held-audio prompt, a probe size that cannot judge
  drift.

### Acceptance

1. **Unit and integration tests** for every new task, rule and template, and the
   plugin skill pins.
2. **The bench record** (`bench/records/narrated.md`) with the arms above, Don's
   verdict, and the preset's numbers.
3. **The constitution walk, regenerated** in a new lem workspace with the
   harvested-narration path, to four minutes or more. Don judges it against the
   three defects.
4. **A cold drill.** A fresh session (Sonnet, the `dw` plugin installed, a numbered
   `~/testing/<n>` directory) is given a different short monologue (60-120 s) and one
   cast still. It must reach, without help:
   - a validated, cost-quoted plan;
   - a probe it judges with the tools;
   - a continuation and the finished take.

   The transcript is the evidence. Any rule it had to guess becomes a fix.

## If the bench fails

If no arm holds identity past the control on a moving, speaking subject, the take
falls back to narration-first cuts that keep the audio:
- the narration track still plays unbroken under the cuts;
- each cut opens on a frame chosen to match the previous take's end (an FL2VA
  bridge), not on the cast still;
- the skill says so plainly.

#1188 (hybrid windows) then becomes the structural lever to revisit, with this bench
as its baseline. Goals 2, 3, 5 and 6 still ship.

## Calls made in the design (Claude's, open to Don's override)

- **Continue and repair accept one re-encoded seam.** Continuing a probe uses the
  existing `resume_from` path:
  - the refusal of a fully kept take is lifted;
  - the refusal against latent carry is lifted for the first resumed seam, which
    stays a decoded, re-encoded guide (Don's 2026-11-08 ruling in
    `chain_resume.py`);
  - anchors, keystone and capstone read segment 1 from its decoded kept file,
    re-encoded.

  The bench includes one continued arm to see whether that seam shows. Persisted
  latents (safetensors on disk) are built only if it does.
- **The identity band comes from the bench, relative to segment 1, advisory.**
  `analyze_face_drift` is SFace (published same-person cut-off 0.363), so the
  community's ArcFace 0.75-0.85 does not carry over. The bench records each arm's
  per-segment SFace score against the cast still and against segment 1. The finding
  fires when a segment falls below segment 1's score by a margin the bench sets. It
  warns and names the segment. It does not re-roll unless the bench shows a re-roll
  recovers (R38). Acceptance of the regenerated take is Don's eye, with the scores
  attached.
- **Repair keeps the layout.** A narrated repair must:
  - re-render at the take's own tier, read from the manifest, not the template's
    defaults;
  - hold the segment's narration slice;
  - preserve the frame count exactly, so the master track muxed over the take stays
    in sync. The splice's spill into k+1's head is fine as long as frames map 1:1.

  Pixel-guided repair is acceptable. A repaired take need not be repairable again.
  The last segment becomes repairable with a head guide only.
- **Bench cost is real.** The constitution-walk-2 runs at the shape's tier (1344x768,
  int8, base 20 steps, 209 frames, check on) averaged 16.5 min per segment: a 90 s
  probe is about 3.3 h and a 4 min take about 9 h, serial (one int8 job per host).
  The bench therefore screens at 90 s and takes only survivors to 4 min (see
  *Bench budget*).

## Bench budget (Don, 2026-11-09)

**Budget.** About 40 GPU-hours on lem: three arms screened, the winner confirmed. Don
evaluates at checkpoints rather than at the end.

**Where the defects show, and so where the checkpoint sits.** On the old cut:
- the face went waxy 16-31 s into each take;
- seam pauses showed from the first seam (about 8 s);
- the #1199 drill's dialogue walk changed the face from 45 s and was painterly by
  60 s.

Every defect is therefore obvious by about 60 s. On the retimed narration (about
126 wpm with pauses) Amendment 3 ends at about 77 s. The probe and the checkpoint
are the Preamble through Amendment 3, about 80 s, rounded up to whole segments
(about 85-90 s).

**The bench text is the constitution narration itself.** The winner's probe is then
the first stretch of the production take and is continued, not re-rendered (Goal 4).

| Checkpoint | After | GPU | Don sees |
|---|---|---|---|
| CP0 | the retimed narration (harvested H3 phrases) and its segment layout | none | the track to listen to, the pause rules, the layout |
| CP1 | control (`dialogue` + held narration) and candidate, seed 1, Preamble-Amendment 3 | ~7 h | both probes side by side; face score per segment against the still; seams; sync; texture and saturation curves |
| CP2 | the rest of the screen: `static_long` seed 1, then seed 2 of all three | ~10 h | the three arms ranked; the winner proposed |
| CP3 | the winner continued from its probe through Amendment 6 (~3:15) | ~4 h | the continued seam; drift at length |
| CP4 | the rest: Amendments 7-10, the full take (~4:15) | ~2.5 h | the finished constitution walk |

Total: about 24 h of GPU on the planned path, leaving headroom in the 40 h for re-rolls,
repairs and one second-line arm.

**Stopping a run early.** Kept segments land on disk as they finish. Each running
probe is checked at segment 4 (about 30 s, where the old cut was already waxy). An
arm already clearly worse than the control there is cancelled to save GPU time,
noted for Don at the next checkpoint.

**If the candidate fails.** If it fails at CP1 or CP2, the second-line arms (no
capstone, stat anchor, per-segment stills) are screened at 90 s, seed 1, with a
checkpoint after each pair. The cold drill (a 60-120 s monologue) and its runs are
outside this budget.

## Amendment 1 (CP1, Don, 2026-11-10): the moving keystone

**What CP1 showed** (`bench/records/narrated.md`, seed 1, 86 s):
- **Control:** degrades markedly by 40 s.
- **Candidate** (latent carry, latent keystone every 3, capstone): it removes the colour
  burn, but the glaze is distracting by 40 s (Don's eye; the measures put it later).
- **`static_long`:** holds the face to 86 s (`face_drop` within ±0.015), but she stands.
  Its last-image pin pulls every segment back to the opening still, so there is no walk.
- **Face-repair post-pass** on the candidate's segments 6 and 11: it softens the face and
  shifts the identity. It has no picture to aim at, so it is dropped.

**The finding.** The one thing measured to hold the face is ending every segment on a
clean picture of her. The opening still cannot be that picture for a walk: it pulls her
back to the start, which is a reset (defect 2).

**The change.** The remaining CP2 screen, and the second-line arms other than
per-segment stills, are replaced by per-segment stills: the *moving keystone*.
1. **Keyframes.** A series of clean stills of her along the walk, one per segment. Each
   is drawn on FLUX 3 Klein from the speaker still, which holds identity, scarf, coat
   and size in frame. The backdrop progresses: the camera tracks back at her pace, so
   she keeps her size and the avenue recedes behind her. Keyframe *k* is what segment
   *k*'s last frame should look like.
2. **The chain feature.** `keystone_images`: a list with one image per segment, or one
   `previous_result:` list. Under `keystone_slot: last`, segment *k* takes entry *k* as
   its fl2va `last_image` in place of segment 1's frame 0. It resolves like the chain's
   `prompts`/`lines` (`resolve_chain_lists`). It works with `anchor_every`: segments
   off the cadence get no last image. It works with probes and `resume_from`: entry *k*
   stays segment *k*'s at any start. It works with held narration. The list is refused
   if it is shorter than the planned segments, naming both counts.
3. **The template and skill.** `narrated-take` gains an optional `keyframes` variable,
   and the long-form skill gains the keyframe step: draw, review, then render.

**Bench.**
- **Keyframes first.** Don reviews the 12-keyframe series for the 86 s probe before any
  video is rendered. The risk to check: the backdrop and face must be consistent from
  one keyframe to the next, or the walk morphs.
- **Arm.** Moving keystone, guide carry, seed 1, `stop_after: 12`, judged against
  `static_long` (face) and the candidate (walk). Then seed 2, then CP3/CP4 on the winner.
- **Added GPU.** About 10-14 h, approved by Don. Daytime GPU work runs on cuda:0 only;
  cuda:1 is loud.

**What it costs if wrong.** If the keyframes are inconsistent, the face changes slightly
at every segment end, a pulse every 7 s. That shows at the keyframe review, before any
video is spent.

## Left to the builder

- Task names, module placement and the ARCHITECTURE map rows.
- The retime algorithm and its crossfade and room-tone handling.
- Whether retime and layout are one task or two.
- The lip-sync signal, or dropping it on evidence.
- How the latent keystone and capstone hook into `h3_guides` and the chain.
- Exact preset names and default N.
- Stage boundaries and issue filing.

## Sources

- **This repo.**
  - `bench/records/counters.md`, `structural.md`, `drill.md`, `check.md`,
    `tiers.md`, `words.md`.
  - `docs/proposals/complete/long-form-media-complete.md`.
  - `plugins/dw/skills/long-form`, `plugins/dw/skills/minimax-h3`.
  - Issues #1150, #1187, #1188, #1225, #1231, #1308.
- **The assessment.** Measured 2026-11-09 from local copies of the
  constitution-walk-2 final cut and its kept segments (ffmpeg silencedetect,
  per-take face crops, frame HSV).
- **Community.**
  - https://github.com/matlowai/h3-long-video-research (CC BY 4.0), study u46_extend.
  - https://github.com/roycho87/NerdNotes (CC BY-SA): 20261106 talking_clown_hybrid_16win,
    20261107 beach30/forest30 protection combos, 20261108 audio_hold_diagnostic.
  - https://github.com/roycho87/geiru_context_suite (GPL-3.0, concepts only).
  - https://github.com/1113800942/comfy-h3md-director (AGPL-3.0, concepts only).
  - https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream (MIT).
- **MiniMax H3.** The model card and the Ref2VA prompt guide (`fully_copy` audio,
  video continuation); no official long-video guidance as of 2026-11-09.
- **Adjacent research** (mechanism only, no training): Deep Forcing (arXiv
  2512.05081, training-free sink frames), InfiniteTalk (2508.14033, re-anchoring to
  near keyframes), LongCat-Video-Avatar 1.5 (2605.26486, a warning that always-on
  references repeat motion: the case for every-Nth cadence).

## Verdicts (2026-11-12)

Closed by Don on 2026-11-12: ship what the bench measured and stop the research. The
evidence is `bench/records/narrated.md`; every arm is seed 1 at 12 segments (about 86 s)
unless said.

**The three defects of constitution-walk-2.**

| Defect | Outcome |
| --- | --- |
| Halting pauses in the narration | Fixed. The narration is retimed once and held unbroken under the whole take. |
| Pose and progress resets | Fixed within a take: no arm reset. |
| Facial burn | Held down, not removed. No arm cleared face, lip sync and motion together at 86 s, and nothing is measured past 86 s. |

**What shipped.**

- `templates/narration-retime` and `templates/minimax/narrated-take`: the held
  narration, a segment per line, probe with `stop_after` and continue with
  `resume_from`, `face_drop` recorded per segment, `repair-segment` on a narrated take.
- Three `shot` presets:
  - `dialogue`: clean to about 30 s.
  - `standing` (static keystone): face and backdrop hold to 85 s, seed 1 only; she stands.
  - `walk_keyframed` (the moving keystone, Amendment 1): the face holds to 85 s on two
    seeds, at the cost of a hand-built keyframe per segment.
- A piece longer than one clean take is passages joined as cuts, one take each. This
  is the fallback of *If the bench fails*. It has not been run end to end.
- #1308 (b): a segment missing its opening words re-rolls, then reports.

**What did not ship.**

- The unbroken 5-minute constitution take (CP3, CP4) and the cold drill were not run.
- Latent carry and the capstone stay behind the experimental flag: they hold saturation
  flat and do not hold the face past about 60-70 s. The latent keystone and
  `anchor_every` showed no gain.
- A walk did not read at chest-up framing. A slow camera arc did, with keyframes.
- The lip-sync measure left the chain check (it did not separate good from bad takes);
  `measure_lip_sync` stays as a take-level task.

**Arms.**

| Arm | Face at 85 s | Colour | Notes |
| --- | --- | --- | --- |
| Control (guide carry) | `face_drop` 0.21, etched by 30 s | saturation 68 to 81 | |
| Candidate (latent carry, capstone, latent keystone every 3) | 0.21, natural to about 60 s | flat | |
| Static keystone | within 0.015 | 68 to 73 | she stands; two re-rolls |
| Moving keystone, v2 keyframes | at most 0.041, two seeds | 75 to 81 | lip sync drops at closed-mouth keyframes; sway, no walk |
| Arc, v3 keyframes (with and without latent carry) | at most 0.068 | 74 to 81 / 77 | the Klein mouth edits aged the face |
| Arc, v4 keyframes + latent carry | at most 0.074 | 73 to 76 | best lip sync; face over-sharpens from about 45 s |
| Kandinsky S2V 1.0 Lite distilled (spike, outside dw) | SFace 0.71 | 71 to 91 | fails on identity |

**Left open** (measured gaps, for follow-up issues): the multi-passage shape run end to
end with a cold agent; the cause of the over-sharpening under the arc keyframes (one
isolating run, v4 without latent carry, was in flight at close); split unsample on the
copied overlap (TrentHunter82, NerdNotes issue 3) as a burn lever; H3 context windows
(#1188); Kandinsky S2V Pro at 4-bit.
