# Long-form media: one approach, measured levers, quality-first defaults

_Design, 2026-11-03. Written by `claude-opus-5-5` (anthropic) with Don. This is the
design a plan is built from; the plan and its stage issues follow it._

## Intent

An agent asked for a long piece (a minute, five minutes, an episode, a music video,
a long narrated walk) treats the model's 15-second clip as a render detail, not a
limit. It blocks and storyboards the piece, gives each scene a length that suits the
story, picks how each scene gets past the clip ceiling, and handles what goes wrong
while it runs. The person asking never has to know what a seam, a guide or an anchor
is.

**Why now.** Don's own long takes (the constitution walk) worked, but smaller tests
and many community reports show the same struggles: drift that builds segment by
segment, visible seams, a bad segment that forces a full re-roll, and a soundtrack
that breaks at every join. Many ComfyUI packs each solve one slice of this
(roycho87/NerdNotes and geiru_context_suite, Smite79/MiniMax-H3-LongVideos, H3MD
Director, PrefixStream, Kinoloom, leonardofrito/Directors-Cut). dw should be the
best-in-breed version: those ideas folded into measured defaults and agent judgment,
not a shelf of switches.

**Philosophy (Don's):** common things made easy, while the uncommon remains possible.
Defaults change wherever the evidence supports it; every knob stays.

## Goals

1. **One playbook an agent reads before planning anything longer than one clip.** For
   each scene it says whether to cut, chain, bridge or render one long take; which
   continuity mode and drift counters fit which kind of shot (static, moving camera,
   dialogue, walk, montage); how the soundtrack stays continuous; what each lever
   costs; and the evidence behind each rule.
2. **Planning carries length and continuity.** The storyboard, script-to-video and
   series-episodes skills give every scene a target length and a continuity choice,
   so an agent's plan states its levers and its cost before it spends.
3. **Defaults that do the right thing.** A plain request for a long take gets the
   measured-best continuity, the drift counters that suit its shot type, and a
   quality-first tier, without the requester naming any of them. A chain can be asked
   for in seconds rather than segment counts.
4. **Handled during the run.** A chain checks each segment as it finishes, re-rolls a
   segment that fails its check within a retry budget quoted up front, and a finished
   chain can have one bad segment repaired in place without redrawing the rest.
5. **Quality against speed is one dial.** Defaults favour quality. The playbook says
   how to move the dial toward speed and what each step costs in quality. Any speedup
   that measures with no marked quality loss becomes the default.
6. **Every idea measured on one bench.** The eleven ideas filed 2026-11-03 (#1187 to
   #1197), the parked chain ideas (#1030, #1032, #1033, #1044, #1067, #1150, #1169),
   the open speed and LoRA candidates (#980, the anti-burn trial from #1151) and the
   shipped counters (#1166) are scored on the same long-take bench. Winners are built
   or become defaults; losers are recorded with their numbers and closed.
7. **Verifiable without code.** Every stage lands with acceptance intent a tester can
   check over MCP alone, plus a design record, so the harnest tester can verify after
   this work is declared done.

## Non-goals

- **Kandinsky 6.** Out of scope (Don: not impressed). Its skill keeps saying "cut".
- Training anything (LoRAs, lifters, distilled models).
- Code from licence-restricted sources. GPL, AGPL and no-licence repos contribute
  concepts only. MIT sources may contribute code with attribution in
  `docs/ATTRIBUTIONS.md`.
- A general video editor. Timeline export (#1197) hands a cut to Resolve; dw does
  not grow transitions, effects or ripple editing.
- New closed-model dependencies.

## Anti-goals (failure even if it technically works)

- **A pile of opt-in switches only an expert would know to set.** If an idea wins, it
  becomes default behaviour for the shots it suits, or it is chosen by the playbook
  from the shot type an agent already records. An idea that only helps when hand-tuned
  is documented as an uncommon knob, not promoted.
- A default that measures better on one scene and worse on another, shipped
  everywhere. Defaults are by shot type, or not at all.
- A long piece whose cost surprises the requester: every automatic re-roll and every
  quality-tier choice is inside the quoted estimate.
- Rules in the playbook that were never measured on our stack, stated as rules.
  Community claims are marked as such until the bench confirms them.

## Constraints

- **Families in scope:** MiniMax H3 (video and audio), LTX-2.5 (video and audio),
  MiniMax Music 3 (music), the TTS templates (`generate-speech`, `voice-clone`), FLUX 3
  Klein for boards. The domains around them: storyboarding and blocking,
  script-to-video, series-episodes, assembly and scoring, cinematography.
- **Model knowledge lives in `plugins/dw/` and the catalog, never in engine code**
  (CLAUDE.md). A per-shot-type default is a template default or a playbook rule, or an
  engine behaviour driven by something the workflow declares or the engine measures,
  never a hard-coded model fact.
- **Skill size caps.** The H3 skill is at its byte cap (12 KiB). The playbook is a new
  skill with its own budget, and the family skills point to it rather than growing.
  `tests/test_plugin_skills.py` pins skill names and budgets.
- **Security rules** in CLAUDE.md apply to every new argument that names media
  (`dw/locations.py` policy, validators).
- **Release:** PR #1198 (0.19.0, develop to master) is open. This work lives on
  `feat/long-form-media` and its stage branches, and merges to `develop` only after
  #1198 merges, so the release is not disturbed. Until then the bench runs on its own
  server instance (see *Decided by Don*).
- **Hardware:** lem (two RTX 3090, 24 GB each, shared host RAM). The bench numbers are
  lem numbers; MPS gets the ordering, not the minutes.

## Decided by Don

| Decision | Ruling |
|---|---|
| Who builds, deploys and bench-tests | This session and its subagents, on lem, including deploys. The harnest tester takes over after this work is declared done. |
| GPU budget | As many lem hours as needed, both cards. |
| Calls of judgement | Decide by measurement where the numbers settle it. Ask Don when unsure, unable to decide, or when a question is pure taste; bring a recommendation either way. |
| Defaults and templates | May change wherever the evidence supports it. Keep every knob. |
| Scope | H3, LTX-2.5, long audio (songs, narration, ambience beds) and the related domains. Not Kandinsky 6. |
| Run-time re-rolls | Allowed: a segment that fails its check may be re-rolled automatically, within a retry budget stated in the up-front quote. |
| Quality against speed | Defaults favour quality, with instructions for moving the dial. Speedups with no marked quality loss become defaults. |
| Cost ceiling for `standard` | Up to about 6x today's H3 chain default (4-step turbo, int4, 960x544, ~2 min per segment) per minute of output. That leaves room for the base model at 1344x768 (~12 min per segment), which the dialogue rule already needs. `best` has no ceiling. |
| Bench server during the 0.19.0 release window | A separate bench `dw.serve` on lem from a git worktree of this work's branch, on its own port and its own card. The main server stays on `develop` for the harnest tester and the release preflight. Both cards go to the bench once #1198 merges. |
| Approvals and labels | Approving this design and its plan covers every stage, engine and syntax changes and default moves included. Stage issues are filed at hand-off, so the loop cannot race the build. This session closes losing ideas itself, with their numbers. |
| Done | The playbook, the defaults and the run-time behaviours have shipped, and every idea has a measured verdict. A large build that passes its gate (for example #1188 hybrid windows or #1194 performer replacement) may land as its own follow-up feature after the hand-off. |

## The approach

### 1. The lever map (what the playbook teaches)

The playbook is organised as a decision per scene, then per seam, then per run.

**Per piece.** Split the piece into scenes the way a storyboard would (a scene is one
place and one continuous moment). Give each scene a target length from the story,
not from the model. The model ceiling only decides how a scene is rendered.

**Per scene, the rendering shape:**

| Shape | When | What exists today |
|---|---|---|
| One clip | ≤ 14.4 s (H3), ≤ ~17 s (LTX at 768) | the family templates |
| Cuts | the scene changes place, time or angle, or a story beat allows a cut | `dialogue-short`, `music-video`, `shots-batch` + `assemble-and-score` |
| Chain | one action, line or camera move must cross the ceiling unbroken | `minimax/chained-segments`, `chain-video-continuity`, `chain-matched-*`; `ltx2/chained-segments`, `ltx2/extend-clip` |
| Bridge / lead-in | a cut must land on a fixed frame | `first-and-last-frame`, `last-frame-only` (joins.md) |
| One long render | a continuous take with no seams at all | not built (#1188 is the candidate) |

**Per chain, by shot type** (the cell values are what the bench decides; the rows are
the shot types the playbook classifies):

| Shot type | Continuity | Drift counters | Evidence today |
|---|---|---|---|
| Static or locked-off | guide | latent anchor (full or partial) + ? | #1175 static: anchor best, B+C cleanest |
| Moving camera (arc, dolly, track) | guide | guide noise; anchor weak or off | #1175 arc: anchor fails, guide noise only arm that doesn't hurt |
| Dialogue / voice across seams | guide with `carry_audio`, or `last_segment` | ? | guide A/B held seams better (h3-guides-complete) |
| Walk / steady subject motion | guide | ? | NerdNotes walk (community) |
| Driven by a soundtrack | `match_audio` chains | ? | the matched templates |

The shot type comes from what the agent already records per shot (cinematography's
`move`: `static` against a camera move; whether the shot has dialogue). Where the
engine can measure it more reliably than the agent can declare it (segment 1's
motion), the bench decides whether "auto" means measured.

**Per seam, during the run:** the check each segment must pass, what a failure looks
like (`analyze_seams` jump and look ratios, texture-band drift against segment 1,
`face_drift` against the cast portrait, `check-script` for lines), and the repair
ladder: re-roll that segment, then repair it in place between its neighbours, then
make the seam a cut.

**Per piece, the soundtrack:** one bed under cut scenes (`assemble-and-score`,
`find_loop_bed`), voice carried across chain seams (`carry_audio`), music longer than
one Music 3 track (300 s practical ceiling; the join recipe), narration longer than
one TTS take (`voice-clone`'s ~95 s context; takes joined), and replacing or adding a
take's sound without redrawing its picture (#1192, if it wins).

### 2. The quality-against-speed rubric

Every recommendation the playbook makes carries a tier. The tiers are named once and
used by every family:

- **draft:** approve blocking, timing, head count and line delivery. Cheapest path
  whose staging carries to the final (the H3 *Ramp* rule: a draft only carries on its
  own path).
- **standard:** the default for a deliverable. Quality-first, at a cost the bench shows
  is reasonable for a long piece.
- **best:** the highest measured quality, cost no object.

Rules:

- Defaults are **standard**, chosen quality-first. Where today's default is a speed
  choice (for example the H3 chain template's 4-step turbo LoRA at int4 and 960x544,
  or the turbo LoRA's "digital" audio on dialogue, #984/#992), the bench decides
  whether it moves.
- One argument moves the dial on a template (a `tier`-like variable), with the
  existing per-argument knobs still exposed underneath. How a template maps one
  variable onto several arguments is the builder's to design within the workflow
  syntax.
- A speedup is "lossless" when the bench shows no marked loss: staging and identity
  held on at least 3 seeds, PSNR in the "changed numerics, same shot" band or better
  against the slow run (the workflow guide's bands), and no drift or seam measure
  worse by more than its rerun noise. Lossless speedups become defaults for the tiers
  they were measured at (sparse attention already is for H3). Lossy ones are listed as
  dial steps with their measured cost.
- The playbook states the dial in plain terms: what each step down saves per minute of
  output, and what it costs.
- **Audio under the lossless rule** (_builder's call, settled in the design review_).
  Claude cannot judge timbre, lip-sync feel or room sound by ear, so the bench measures
  what it can: level steps and continuity at seams, high-band energy against the slow
  run (the measure #991 used for the turbo "digital" sound), `pcm_sha256` identity
  where audio should be unchanged, and `check-script` for lines. A change that is
  lossless on these measures but may change timbre is not called lossless by Claude:
  it goes to Don's ear with a recommendation and the job ids to listen to. A small
  objective audio comparison measure may be added to the tooling if the bench needs
  one.
- **Skill budgets** (_builder's note_): `script-to-video` has almost no room left under
  its margin, so the per-scene length and continuity fields go in its `references/`
  files, not its `SKILL.md`.

### 3. Engine and template behaviour (what makes it easy)

These are the behaviours the bench may promote. Each one is built only if its
evaluation says so, and the mechanism is the builder's.

1. **Length in seconds.** A chain can be asked for by duration; segment count and
   frame snap follow (today a chain is `segments` or `match_audio`).
2. **Continuity default.** `minimax/chained-segments` defaults to the continuity the
   bench measures best (today `last_frame`, although guide measured better at ~40%
   more time; that default was left as Don's call and is now the bench's).
3. **Counters by shot type.** The chain picks its drift counters from the shot type,
   declared or measured, using the per-shot-type table above. Explicit settings win.
4. **Per-segment check and bounded re-roll.** After each segment, the chain measures it
   (seam and drift measures that need no human), and a segment that fails is re-rolled
   with a new seed up to a budget the validate estimate includes. The run record says
   which segments were re-rolled and why. The thresholds come from the bench's rerun
   noise, not from guesses.
   - _Builder's call (settled in the design review):_ on by default at `standard` and
     `best`, off at `draft`; a default budget of one re-roll per segment, quoted in the
     estimate as a worst case. When a segment's budget runs out, the chain keeps the
     best-measured attempt, finishes, and flags that segment in the run record and the
     job's warnings. It never restructures the run by itself: repair in place and
     turning the seam into a cut are the agent's next rungs, after the chain is done.
5. **Repair in place** (#1169): re-render one segment of a finished chain with its
   head held to the segment before and its tail held to the segment after.
6. **LTX parity.** The same playbook and, where the bench supports it, the same
   behaviours for LTX-2.5 chains (its conditioning is different; `extend-clip`
   conditions on a whole clip, not one frame).
7. **Audio continuity defaults.** The soundtrack of a long piece is continuous by
   default: one bed under cuts, held voice across chain seams, and long music and
   narration joined without audible seams.

### 4. The bench

A fixed set of long takes, run with fixed seeds through the same measures, kept as
stored workflows on lem so any later session (the regression agent included) can
rerun it. It replaces one-off probe protocols.

**Scenes:**

| Id | Scene | Family | Why |
|---|---|---|---|
| S1 | static locked-off backdrop (stripes and shapes) with a subject in front | H3 | the drift worst case; `probe-1166`'s static scene |
| S2 | moving-camera arc around a statue | H3 | where the anchor fails; `probe-1166`'s arc |
| S3 | a subject talking while walking, voice held across seams | H3 | dialogue and motion together, the constitution-walk shape |
| S4 | a scored montage of cut scenes with one music bed | H3 + Music 3 | the cut path and the soundtrack |
| S5 | silent action continuing across seams | LTX-2.5 | LTX chains and extend |
| S6 | a long narration under cut boards or shots | TTS + Music 3 | long audio joins |

Each chained scene runs at a length that exposes drift (about ten segments, ~40 s)
and, once a candidate wins, at a two-minute length to confirm it holds.

**Measures** (all reachable over MCP, so the tester can repeat them): texture bands on
a subject-free crop against segment 1 at the first, middle and last segments;
`analyze_seams` jump and look ratios and look steps; `face_drift` against the cast
portrait where a face is held; `check-script` on lines; seam level steps and
continuity of the soundtrack; wall time and peak VRAM per segment; frames at fixed
moments for eye judgement. Each measure's rerun noise is taken first (the same seed
twice) so a difference can be called real.

**Verdict rule:** an arm wins a scene when it beats the control on that scene's
primary measure by more than rerun noise and is not worse on any other measure by
more than noise, confirmed on a second seed. Taste-only differences go to Don with a
recommendation.

### 5. Evaluating the ideas

Every idea goes through the bench before it is built or promoted. Cheap mechanisms
first; a structural rewrite only when the cheap ones leave a measured gap.

| Group | Ideas | First gate |
|---|---|---|
| Counters on today's chains | #1190 anchor strength, #1187 keystone as keyframe (capstone second), #1189 hand-over back-off, #1032 sharpest of last N, #1033 and #1191 part 1 seam grade, #1191 parts 2-4, the anti-burn LoRA on continuation segments (#1151's trial), #1196 TST | S1, S2, S3 at ~40 s against today's default and today's best |
| Carry without the pixel round trip | #1150 latent tail, #1044 latent carry + crossfade, #1030 decode once | only if the counters leave a seam or drift gap on S1-S3 |
| One long render | #1188 hybrid windows | a spike to establish feasibility on the diffusers H3 pipeline, then the bench; built only if it beats the best chain |
| Running and repair | per-segment check + re-roll (new), #1169 repair in place, #1194 performer replacement | S3 with an injected bad segment; #1194 on a cut with a replaced character |
| Audio | #1192 audio-only pass, long music joins, long narration joins | S4, S6 |
| Speed | #1195 SelfLift, #980 FastH3 Omni, cache settings at long-take tiers, LTX speed levers | the lossless rule in section 2, on S1-S3 |
| Agent knowledge | #1193 skill probes, the playbook itself | two seeds per claim; a cold-session drill of the playbook |
| Delivery | #1197 timeline export | Resolve import of a bench cut |

#1067 (region colour-match, plan approved, parked) is folded into the seam-grade
evaluation where it overlaps. #1149 (cinematic add-ons) and #1186 (short-line split)
stay out of this design.

### 6. Agent knowledge (where the playbook lives)

- A new cross-cutting skill in `plugins/dw/` for long-form pieces (the playbook above),
  named in `plugins/dw/README.md` and CLAUDE.md and pinned by
  `tests/test_plugin_skills.py`, like the other composition skills.
- `storyboard`, `script-to-video` and `series-episodes` gain a per-scene target length
  and continuity choice and point to the playbook; `minimax-h3` and `ltx-2.5` replace
  their "longer" lines with a pointer and keep only family-specific rules;
  `minimax-music3` and the TTS templates point to the long-audio section.
- Every rule cites the bench record it came from. Community claims stay marked until
  measured.

### 7. Hand-off to the harnest tester

- A design record in `docs/proposals/complete/` when the work is done, in the house
  format (what shipped, what the bench found, what was cut and what would bring it
  back).
- A feature issue with the plan, and one stage issue per stage, each with: what was
  built, how to observe it over MCP, the acceptance intent (the tester writes cases
  from it), the bench job ids and stored workflow names to rerun, and any
  `breaking-change` note where a default moved.
- A short list for Don of the taste calls left open, each with a recommendation.

## Left to the builder

- The playbook skill's name and file layout within the plugin's conventions and
  budgets.
- How a template exposes the quality dial within the workflow syntax.
- The engine mechanisms of each behaviour in section 3, including how a chain measures
  a segment, how a re-roll budget enters the estimate, and how "auto" shot type is
  measured, if the bench says it should be.
- The bench's exact prompts, crops, seeds and stored-workflow names, kept stable once
  chosen.
- Stage order inside each group, and which ideas are closed on paper after the first
  gate shows them dominated.
- Which existing knobs get renamed or folded, provided every one stays reachable and a
  renamed one is marked `breaking-change`.
