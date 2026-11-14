# Cold-session drill (Task 28)

2026-11-06. Bench server lem:8766 at `6d923095`, MCP mounted at `/mcp`, both
cards (cuda:0, cuda:1). Each ask ran in a fresh headless Claude Code session
(`claude -p --model opus`) in its own empty directory (`~/testing/lf-drill-<n>`),
with only the dw MCP (`--strict-mcp-config`) and the plugin from this branch
(`--plugin-dir plugins/dw`). Turn 1 was the ask, verbatim. Turn 2 was "Looks
good, go ahead and run it. Let me know if anything needs a retry." A later
"Check on the job and finish up." let the session judge and finish.
Transcripts (stream-json) are in the drill agent's scratchpad, not the repo.

**Setup notes.**
- lem's firewall drops LAN connections to 8766 (and 8765) from this Mac;
  only ssh gets through. The sessions reached the server through an ssh
  tunnel (`-L 18766:127.0.0.1:8766`), and a cold agent's own `curl` to
  `lem:8766` timed out (ask 3, when it tried to download the cut).
- The three first sessions shared the `default` workspace with each other.
  The planning reruns after the fixes ran in their own empty workspaces
  (`lfdrill1c`, `lfdrill1d`, `lfdrill3c`) through the stdio `dw_mcp` client,
  because the first rerun of ask 1 and ask 3 in `default` found the earlier
  session's jobs and reused them.
- The user-level Mac skills (`h3-prompt-writing`, `music-caption-rewriter`)
  and Bash were available to the cold sessions, as they are in Don's own
  drills.

## Scorecard

| | Ask 1: 90 s walk-and-talk | Ask 2: 3 min, four scenes, score | Ask 3: 5 min narration over boards |
| --- | --- | --- | --- |
| Read `long-form` first | yes, first call | yes, first call | **no**: went to `storyboard`, then its `references/animatic.md` |
| Shape per scene | chain, `shot: dialogue` (what the playbook says for an unbroken take) | cuts (`dialogue-short`), the playbook's "prefer cuts" | boards + per-board takes + bed + `templates/animatic` |
| Cost quoted with retries | yes: 210 min, worst case about 6.5 h with `retries: 1` | no retries quoted; offered a draft first | guesses only; no retry budget |
| Tier dial | no (the playbook says the tiers are *pending measurement*) | improvised a draft (portraits + 2-3 shots) | no |
| Flagged segment | none: the check passed all 7 segments, and the take drifted anyway | not a chain; the short score and a doubled keeper in shot 4 were caught by eye and metadata and re-rendered alone | take 5 truncated, caught by length and transcript, re-spoken in halves |

## Ask 1: "Make me a 90-second single take of two friends on a walk-and-talk ..."

**Call order (turn 1).** `Skill dw:long-form` → `Skill dw:minimax-h3` →
`get_server_info` → `list_workflows(traits="chained")` →
`get_workflow(templates/minimax/chained-segments)` → `h3-prompt-writing` +
the H3 dialogue and chains references → `dw:flux3-klein` →
`validate_workflow` + **`run_workflow`** of the opening frame (FLUX 3 Klein,
`077795ecc576`, 0.4 min, run before asking) → `validate_workflow` on the
chain → `save_workflow(walk-and-talk/chained-dialogue-checked)` with
`check: true, retries: 1` → `validate_workflow` on the saved copy.

**What it planned.** Seven 345-frame segments, `duration_seconds: 90`,
`shot: dialogue`, `keep_segments: true`, turbo LoRA off at 12 steps (the H3
skill's voice rule), the check on. 26 scripted lines in `<d>` markup.

**Cost.** The template's `plan.estimate` was 210.3 min (`derived`, low
confidence, "scaled from the curated 13.5 min: steps 12 vs 5, frames 345 vs
124, segments 7 vs 3"). The saved checked copy came back `basis: unknown`.
The agent scaled the template figure by hand to about 6.5 h worst case.
**The run took 71.0 min** (`ab7f54f73b2f`): the scaled estimate was about 3x
high for this shape.

**Outcome.** All seven segments passed the check on the first attempt. There
were no warnings. `check-script` (`ad340963c34e`) heard all 26 lines (each
similarity at least 0.6), with one `line_clipped_at_end` at 27.8 s, seam 2.
By eye (frames at 2, 20, 35, 50, 65 and 88 s): clean to about 35 s. From
about 45 s, the woman's face and hair change and the trees go bare. From 60 s
on, the picture is oversaturated and painterly, with speckle on the man's
coat. The coats, props and left/right staging hold all the way. The cold
agent's own judge pass (`analyze_seams`, `get_output_frames` at late
moments, `levers.md`) found the same thing and named it carry drift that a
re-roll cannot fix. It offered: a reset at about 28 s and 55 s, two admitted
cuts, or keeping the first 35 s.

**Score.** It read the playbook first, picked the shape the playbook gives
an unbroken dialogue take, used the check and `keep_segments`, and judged
well. **The playbook misled it on one thing:** the 30 s rule was stated for
static takes only, so nothing told it that a 90 s dialogue take would not
hold. It quoted 3.5-6.5 h for a take that was unusable past 35 s.

**Waste.** It ran the FLUX still before asking (cheap, but unasked). It
tried to find the server token in `~/.claude.json` to poll by itself, and
the auto-mode classifier blocked it.

## Ask 2: "Make a 3-minute piece of four scenes - a lighthouse keeper's day ..."

**Call order (turn 1).** `Skill dw:long-form` → `get_server_info` →
`list_workflows(shape=sequence|shot|audio)` → `Skill dw:minimax-h3` →
`get_workflow(dialogue-short)`, `get_workflow(assemble-and-score)` →
`dw:minimax-music3`, `h3-prompt-writing`, `music-caption-rewriter` →
`validate_workflow(dialogue-short)` with 16 shots of 277 frames → 
`validate_workflow(templates/minimax/music)` with a 13-tag instrumental
body and `audio_duration` 280.

**Shape.** Four scenes of about 45 s, 16 cut shots on `dialogue-short`
(portrait-referenced keeper and lighthouse, no dialogue,
`non_diegetic_music: N/A`), one score under the cut with `assemble-and-score`.
That is the playbook's *Cuts* row and "one bed" rule.

**Cost.** Picture 147.5 min (`derived`), score 3.6 min (`catalog`, not
scaled by `audio_duration`), assembly 0.2 min. No retry budget. It offered
"render just the portraits and two or three shots (about 30 min)" first.

**Run.** Score `353751db0aac` (3.5 min) came back **91 s for 280 s asked**.
The agent caught it from `get_gallery_metadata`, read the playbook's
`music-long` row and re-ran it as four 52 s sections (`6ea608439d09`,
8.9 min, 195 s joined; `slice_audio` warned that three sections were still
playing at their cut). Picture `41692e04da2e`: 130.9 min, against
147.5 quoted. On "finish up", the agent looked at every shot
(`get_output_frames`) and found shot 4 with the keeper doubled on both sides
of the lens. It re-rendered that one shot (`dd8c149d3314`, 10.1 min). Then it
read `assemble-and-score` but wrote its own inline cut, to drop 7 run-up
frames per shot. Two attempts failed at run time:
`2cc8e119e2bf` ("Too many iterations generated: 7.9e38", a cartesian
product of `previous_result` lists) and `620dc103b35c` ("'Image' object is
not subscriptable"). The third, `8e75f5a1a33d`, cut the film: 3:00.3 at
960x544, peak -3.4 dBFS, with up to 8 dB level steps at exterior/interior
cuts, left in on purpose. By eye (a frame every 15.5 s): the keeper and the
red-banded tower hold across all 16 shots, and the night shots are dark but
read. The agent reported that the score's movements land 4-11 s after their
scene changes.

**Score.** It read the playbook first, picked cuts, and recovered the score
correctly. **The playbook misled it on the score:** "music longer than one
Music 3 track (300 s practical)" read as "a 180 s score is one track", and
a tag-only instrumental ends when its tags run out, not at the ceiling.

## Ask 3: "Make a 5-minute narration over storyboard boards: ..."

**Call order (turn 1).** `Skill dw:storyboard` → `animatic.md` (Bash `cat`)
→ `get_server_info` → `get_workflow(templates/animatic)`,
`get_workflow(templates/generate-speech)` → `list_workflows(image|audio|utility)`
→ `get_workflow(narration-from-takes)` → `list_assets`, `list_tasks`,
`get_task(crossfade_audio|mix_audio|generate_speech)` →
`get_workflow(templates/minimax/music)` → `get_guide(workflows, "Authoring a
workflow from an agent")` → `validate_workflow` of a hand-written
`CableNarration` (one Audio8 take per board, **no reference voice**), the
music template (**empty lyrics**, 300 s) and a hand-written `CableBoards`.

**It never loaded `long-form`.** `storyboard` and `animatic.md` did not point
a long read at it. It found `narration-from-takes`, read "There is no default
voice: `reference_audio` ... required", would not clone `bench/narrator.wav`
because it did not know whose voice that was, and so re-wrote the template
without the reference. That made each take Audio8's un-referenced voice,
with nothing tying the takes together. `animatic.md` also told it
"instrumental if the lyrics are empty", which the run refused.

**Run.** Boards `078e54d8f3c7` (1.6 min, 12 boards; the contact sheet is
on-style, no stray text). Narration `205ca638ed02` (5.5 min, 303.8 s). Take 5
came back 12.5 s for 62 words, and the agent caught it by length, confirmed it
by transcript (`8d4140f455d0`), and re-spoke it in halves (`36b5a74093df`;
transcript `45bb96d77fa1` has every word). Music `b176507ac918` **failed:
`lyrics` must be a non-empty string** (validation had passed). The retry with
a 9-tag body (`d00994e81284`) came back **62 s for 300 s asked**, and the
agent looped it with 3 s crossfades. The first animatic attempt (the rerun
session's `f4ff0c77d018`) **failed: stills_to_video would hold 7290 frames
of 1344x768 (21.0 GiB), over the 4 GiB limit**. Validation had passed. The
original session read `get_task(stills_to_video)` and built the animatic at
4 fps: `a5408f2f87d8`, 5:18, every board within 1/8 s of its line.

**Score.** No playbook read, so no shape from it. Per-board takes are a
reasonable shape for an animatic. Each take was 61-71 words, under the
111-word take limit. The voice, the empty lyrics and the frame budget were
all misled or missing in `storyboard`'s `animatic.md`. Cost was guessed
("not measured") with no retry budget. It caught and repaired its own
flagged take.

## Fixes (this commit)

| Where | Fix | Evidence |
| --- | --- | --- |
| `long-form/SKILL.md` §3 | A take of any shot type drifts past about 35 s. Say so and offer chains of at most 35 s joined by cuts before quoting a longer single take. | ask 1 |
| `long-form/SKILL.md` *Run and judge* | A saved copy has no cost row (`basis: unknown`). Validate the template with the same arguments and scale by the worst case. | ask 1 |
| `long-form/SKILL.md` §5 | The lyric, not the ceiling, sets a track's length. A score past about 90 s is `music-long` sections (91 s and 62 s back for 280 s and 300 s asked). | asks 2, 3 |
| `minimax-music3/SKILL.md` | The score bullet points a cut past about 90 s at `music-long`. | asks 2, 3 |
| `long-form/references/audio.md` | "There is no default voice" now reads as "no default reference". With no voice from the user, speak a 5-10 s Audio8 line, keep it, and clone every take from it (how `bench/narrator.wav` was made). Never speak takes without a reference. | ask 3 |
| `storyboard/references/animatic.md` | A read over about 40 s goes to `long-form`'s narration recipe, one take per board through `narration-from-takes`. Empty lyrics are refused, so use a tag-only body, and a bed past about 90 s is `music-long`. The 4 GiB frame budget is checked only at run time: about 58 s at 1344x768 and 24 fps, so lower `fps` for a long animatic. | ask 3 |

## Reruns after the fixes (turn 1 only, cold, own workspace)

- **Ask 1** (`lfdrill1c`, before the drift fix): validated the template
  itself, quoted 200 min, and quoted the checked worst case as about 6 h from
  the same figure. The cost fix holds. It still planned one 90 s chain.
- **Ask 1** (`lfdrill1d`, after the drift fix): `long-form` →
  `get_server_info` → `list_workflows(traits="chained")`, then stopped to
  advise. A 90 s single take will drift past about 35 s, so it recommended
  three continuous 30 s takes joined by two hard cuts on conversation beats,
  with one shared park bed. It offered the 90 s take only as "useful to see
  the drift". It asked about the cast before validating, so it gave no
  figure yet.
- **Ask 2** (`2b`, `default` workspace, after the music fix): `long-form`
  first, then cuts on `shots-batch` (four batches of four, after validation
  projected 270 GB of host RAM for one 16-shot batch), and **`music-long`
  in four sections from the start**. It ordered the score and scene 1 first
  as a look check. `music-long` has no estimate (`basis: unknown`), so the
  agent guessed about 15 min.
- **Ask 3** (`lfdrill3c`, after the narration fixes): `storyboard` →
  `animatic.md` → **`long-form/references/audio.md`** → a 6 s synthetic Audio8
  reference line, then 10 takes of 75-87 words cloned from it through
  `narration-from-takes`, a `music-long` bed in four 85 s sections, and the
  animatic. Every point the first session got wrong was right in the plan.
  It did not yet budget retries.

## For Don

- **Validation passes two run-time refusals:** `templates/minimax/music`
  with `lyrics: ""`, and `templates/animatic` (`stills_to_video`) over the
  4 GiB frame budget. When the boards are assets, their size can be read at
  validation. Each refusal cost a queued slot and a retry.
- **Estimates:** the scaled H3 chain estimate was 210 min against 71 min
  real (`ab7f54f73b2f`, 12 steps, 345-frame segments), and `music-long` has
  no cost row. `templates/minimax/music` quotes 3.6 min whatever the
  `audio_duration`.
- **The tier dial is not usable cold.** No session used it, because
  `quality-and-speed.md` says the tiers are *pending measurement* (Tasks
  19-20). Two sessions improvised a draft pass instead. Worth filling once
  the speed tiers land.
- **Bench firewall:** lem drops LAN traffic to 8766 from the Mac, so a
  session on the Mac cannot reach the server except over an ssh tunnel.
- **Voice without a reference** (ask 3's first animatic) has not been
  checked for drift between takes. The ECAPA probe in `bench/records/audio.md`
  could measure it on `205ca638ed02`.
- **Hand-rolled cuts fail at run time.** Ask 2's agent wrote its own cut
  instead of `assemble-and-score`, to trim H3 run-up frames, and two of its
  attempts failed at run time after validation passed: a `previous_result`
  cartesian product, and `'Image' object is not subscriptable` (`620dc103b35c`).
  The second looks like an engine error message worth a look.

## Job ids

| Ask | Jobs |
| --- | --- |
| 1 | `077795ecc576` (still), `ab7f54f73b2f` (90 s chain, 71.0 min), `ad340963c34e` (check-script) |
| 2 | `353751db0aac` (91 s score), `6ea608439d09` (music-long, 195 s), `41692e04da2e` (16 shots, 130.9 min), `dd8c149d3314` (shot 4 retry), `2cc8e119e2bf` / `620dc103b35c` (failed cuts), `8e75f5a1a33d` (final cut, 3:00.3) |
| 3 | `078e54d8f3c7` (boards), `205ca638ed02` (narration), `8d4140f455d0` / `45bb96d77fa1` (transcripts), `36b5a74093df` (take 5 retake), `b176507ac918` (music, failed), `d00994e81284` (62 s music), `f4ff0c77d018` (rerun session's animatic, failed), `a5408f2f87d8` (animatic, 5:18) |

# Cold drill: the Gettysburg Address as passages joined as cuts (#1311 stage 2, #1337)

2026-11-14. One headless Claude Code session (`claude -p --model sonnet`, which
ran as `claude-sonnet-5-5`, `--permission-mode auto`, `--max-budget-usd 15`),
with only the dw MCP (`--strict-mcp-config`) and the plugin snapshot at
`6316cc98`, which already holds stage 1's recipe (`run-and-repair.md`,
*Longer than one take: passages joined as cuts*). Workspace `drill-1311-1`,
holding only `speaker.jpg` (FLUX 3 Klein, `429b9fde735f`, seed 1311,
1344x768), made beforehand by the drill runner.

**Confound: the server was not stage 1's code.** lem's server had run
hand-started code at `595cee1f` (the 0.20.0 release code, reporting
`0.20.0-alpha.1`) since 00:48 UTC, so every job below ran on it, not on
`develop`. The plugin text the agent read was stage 1's, and
`templates/narration-retime` (which the agent said was missing) exists at
`595cee1f`.

**The ask** (turn 1, verbatim from `ask.txt`): workspace `drill-1311-1`, the
speaker `asset:speaker.jpg`, the Bliss text of the Gettysburg Address (about 270
words), "One continuous-looking speaker, about two minutes, natural delivery."
Turn 2 was "go ahead". Turns 3 and 4 were "check on the job and finish up".
`status.log` ends in `done`.

**Outcome: the attempt took the wrong shape.** The agent ran seven `dialogue`
passages on `templates/minimax/chained-segments`, with H3's own invented voice
and no narration track, joined as 6 cuts. Silence holes over 1.5 s sit at 5 of
the 6 cuts. The recipe's shape (one `narration-long` track, retimed once,
sliced into 2-3 `narrated-take` passages) was not run. The root causes are
two agent errors:
- **W1**, "no `narration-retime` template". Fix location: `run-and-repair.md`
  *Longer than one take* step 1 names `get_workflow("templates/narration-retime")`,
  and `workflows/templates/narration-retime.json` and `narration-long.json`
  declare `shape: "audio"`.
- **W2**, "no voice clip" taken as a blocker. Fix location:
  `run-and-repair.md` step 1 inlines the make-a-reference step
  (`templates/generate-speech`) instead of pointing at `audio.md`.

W3-W7 follow from these two (table below). The jobs ran on the stale
`595cee1f` server (`0.20.0-alpha.1`), not stage 1's deployed code (the
confound above).

## Scorecard

| | What the agent did | What the recipe says |
| --- | --- | --- |
| Preset | `shot: dialogue`, tier `standard` (960x544), `templates/minimax/chained-segments` with per-segment `lines` | `templates/minimax/narrated-take`; `standing` for a passage of about 175 words |
| Passage split | 7 passages of 2-6 segments (141 + 119 frames each), 30 segments, each "near or under the roughly 30 s the `dialogue` preset holds" | 2-3 passages at paragraph breaks |
| Narration path | none: H3 invented the voice per segment from `<d>` lines | one `narration-long` from one voice reference, retimed once, cut with `slice_audio` |
| Still per passage | the same still for all 7 | "a different still or angle for each" |
| Join | inline one-step `concat_videos`: `trim_frames: 0`, `match_levels: "lufs"`, `fps: 24`, 7 `output:` refs in order (`c89a2d8108e5`) | the same, as written |
| Cost quoted | "about 60-75 minutes of GPU time", "extrapolated from the probe's timing ... not from `validate_workflow`" (turn 2). Probe validated at 18.3 min, ran in about 7 | quote `validate_workflow`'s `plan.estimate` |
| Cost actual | probe about 7 min, then six passages and the join queued 01:16:42 and done by 01:58 on **both cards** (cuda:0 and cuda:1): about 42 min wall clock. Agent tokens $1.567 over 4 turns | |
| Result | 2:35.17 (3724 frames, 24 fps), -16.7 LUFS, true peak -0.93 dBFS, sync drift 0.03 ms max, 6 hard cuts | about 2:00 asked; the plan expected 1-2 cuts |

**The turn-2 choice, in its own words:** "Every passage stays near or under the
roughly 30 s the `dialogue` preset holds." and "I'd join them with hard cuts,
using the same still each time. They'd read as jump cuts on one framing, not
as a single unbroken take." In turn 1 it had already read that `standing`
holds about 85 s ("Measured limits are about 30 s for `dialogue` and about 85 s
for `standing`") and still chose `dialogue`, because it had dropped
`narrated-take` (W1, W2) and `chained-segments` is the path it knew for an
invented voice.

## Guesses and wrong turns (stage 3's rows)

Agent error, quoted from `turn-1.jsonl`..`turn-4.jsonl`. Each names where the
text that would have prevented it lives.

| # | What it did (quote) | Why it was wrong | Fix location |
| --- | --- | --- | --- |
| W1 | "The `narrated-take` template needs a finished narration track, and this server has no `narration-retime` template to produce one." (turn 1) | The template exists. The agent never called `get_workflow` on it; its `list_workflows(shape="sequence")` and `traits="has-audio,needs-input-media"` calls cannot return it, because `narration-retime` and `narration-long` declare no `shape`. | `run-and-repair.md` *Longer than one take* step 1: name the call (`get_workflow("templates/narration-retime")`). `workflows/templates/narration-retime.json`, `narration-long.json`: give them `shape: "audio"` so a shape listing finds them. |
| W2 | "The voice-cloning templates (`narration-long`, `voice-clone`) also require a reference clip of the voice ... I found no voice clip in the workspace." (turn 1) | `audio.md` *No voice from the user? Make one* answers this (a 5-10 s Audio8 line from `templates/generate-speech`), but step 1 only points at `audio.md` and the agent never opened it. | `run-and-repair.md` step 1: inline the make-a-reference step, not a pointer. |
| W3 | "(b) Let H3 invent the voice from the prompt, using `chained-segments` with per-segment `lines` ... The voice will probably differ a little from passage to passage." Recommended (b). (turn 1) | Follows from W1 and W2: it left the recipe's path, so nothing ties the passages' voices together. | Closed by W1 and W2; the recipe's opening line can also say `chained-segments` with invented voices is not this recipe. |
| W4 | `shot: dialogue`, 7 passages, "near or under the roughly 30 s the `dialogue` preset holds" (turn 2) | The recipe sizes passages for both presets and leaves the choice open. For one speaker held to camera, `standing` (about 85 s) gives 2 passages and 1 cut. | `run-and-repair.md` step 2: say `standing` for a monologue to camera, `dialogue` only for a passage under 30 s. |
| W5 | "using the same still each time. They'd read as jump cuts on one framing" (turn 2) | Recipe step 3 says a different still or angle, but not how to make one; the agent knew it would read as a reset and did it anyway. | `run-and-repair.md` step 3: name the way to make a second angle from the first still (`templates/image-edit`, or a FLUX 3 Klein portrait on the same seed and a changed framing). |
| W6 | "about 60–75 minutes ... extrapolated from the probe's timing ... not from `validate_workflow`" (turn 2); then queued six passages unvalidated in turn 3 | The loop says quote `plan.estimate`. The probe's own estimate (18.3 min) was 2.6x its 7 min run, which is likely why it trusted its own timing. | `long-form/SKILL.md` *Run and judge*: validate each passage and sum the estimates; say the H3 chain estimate runs high (record: 18.3 quoted, 7 real). |
| W7 | "I haven't listened to the audio or watched it end to end. I only looked at five frames." (turn 3) | It declared the cut finished with 6 level steps of 9-16 dB at the cuts and 1.75-2.75 s holes at 5 of 6 cuts, all of which `assess_output` reports. | `run-and-repair.md` step 4: after the join, run `assess_output(probe="analyze_seams")` and `get_output_frames(seams=...)` at the passage cuts, and listen across each with `get_output_audio`. |
| W8 | First join validate used `"name"` for the workflow id. | Caught by `validate_workflow` and fixed on the next call. No cost. | None needed. |

**Harness artifacts, not dw text** (not stage 3's rows):
- In turn 3 it waited with background `sleep 1500` and `sleep 780`, which
  ended the headless turn; the notifications re-invoked it twice.
- It read "check on the job and finish up" as the go-ahead for the six
  passages it had asked about. There was no job running.
- It used both cards. The plan named one.

## The constitution-walk-2 defects, measured

From `assess_output` (`analyze_seams`, `analyze_shots`, `analyze_sync_drift`)
and `get_gallery_metadata` (envelope at 0.25 s) on the final cut, and
`get_output_frames(seams=[3, 7, 11, 17, 22, 28])` at the six passage cuts.

**Silence and pauses at the cuts.** The span below -50 dBFS around each cut:

| Cut | Time | Silence | Level step at the cut | Picture `jump_ratio` (in-passage 0.45-1.36) |
| --- | --- | --- | --- | --- |
| P1 / P2a | 15.79 s | 15.50-17.50, **2.00 s** | 13.56 dB | 10.17 |
| P2a / P2b | 36.54 s | 36.00-38.75, **2.75 s** | 9.08 dB | 13.03 |
| P2b / P3a | 57.29 s | 56.75-58.00, 1.25 s | 16.15 dB | 17.73 |
| P3a / P3b | 87.96 s | 87.00-89.00, **2.00 s** | 9.65 dB | 6.22 |
| P3b / P3c | 113.67 s | 111.75-113.50, **1.75 s**, then a 0.25 s blip and 0.75 s more | 11.23 dB | 9.71 |
| P3c / P7 | 144.33 s | 144.25-145.00, 0.75 s | 12.45 dB | 7.75 |

Five of six cuts hold a hole over 1.5 s. The whole piece is **36% below
-50 dBFS** (55.75 of 155.25 s), and the two longest holes (2.75 s) are at
36.0 s (a cut) and 50.5 s (inside P2b). Each passage's first segment opens on
0.5-1.7 s of dead air (`shot_dead_air`), the H3 lead-in before the first line,
which a cut then puts in the middle of the speech. Inside passages the level
spread is 21.26 dB (`shot_level_spread`); `match_levels: "lufs"` evened the
passages, not the segments.

**Resets at the cuts.** Every cut is a reset. Each passage's last frame has
pushed in to a close-up of the face, and the next passage opens on the wide
still again. The frame difference at the cut grows from 35.3 (cut 1) to 51.1
(cut 6) as the passages lengthen, and `look_ratio` is 7.0-11.5 at the cuts
against 0.78-2.57 inside passages.

**`face_drop` per passage** (segment 1 of each is null):

| Passage | Segments | Seconds | `face_drop`, segments 2.. | Check |
| --- | --- | --- | --- | --- |
| P1 | 3 | 15.8 | 0.030, 0.056 | all passed first time |
| P2a | 4 | 20.8 | 0.074, 0.119, 0.137 | passed |
| P2b | 4 | 20.8 | 0.051, 0.046, 0.081 | segment 4 on its 2nd attempt (0.87) |
| P3a | 6 | 30.7 | 0.055, 0.055, 0.127, 0.174, **0.187** | segment 6 on its 2nd attempt |
| P3b | 5 | 25.7 | 0.019, 0.066, 0.093, 0.115 | segment 5 on its 2nd attempt |
| P3c | 6 | 30.7 | 0.035, 0.086, 0.110, 0.121, 0.154 | segment 2 failed (`line_clipped_at_end` on "gave", `line_clipped_at_head`) and was kept; segments 5 and 6 on their 2nd attempt |
| P7 | 2 | 10.8 | 0.021 | passed |

Past 0.146 is where an eye first calls a face etched (`run-and-repair.md`).
P3a and P3c cross it in their fifth and sixth segments, 21-31 s into a 30 s
`dialogue` passage, sooner than the record's 36 s. The agent saw the P3c face
go "oversaturated and distorted" near 2:20 and offered a 15 min re-render; it
was not run (no second attempt is approved).

## Open questions

- **Voice and room tone across the cuts:** not measured here. Every passage's
  voice was invented by H3 from its prompt, so nothing ties them together; the
  agent said so in turn 1. This is a listening verdict for Don (cuts at 15.8,
  36.5, 57.3, 88.0, 113.7 and 144.3 s).
- **`standing` on a new seed: unanswered.** The agent never ran `standing`, so
  the record still holds it on seed 1 only.
- **Model limits, apart from agent error:** the face drift inside 30 s
  `dialogue` passages (0.154-0.187 by the 6th segment), and the H3 lead-in
  silence at the head of each passage, are the model's. The 6 cuts, the
  invented voice, the same still and the missing seam check are the agent's
  (W1-W7).

## Paths and job ids

- Runner: `~/testing/drill-1311-runner/` (`ask.txt`, `run.py`, `status.log`,
  `turn-1.jsonl`..`turn-4.jsonl`, `plugin-dw/` at `6316cc98`). Session
  `6baef2e7-7d0c-40ef-86a4-1ec2b94c8991`, cwd `~/testing/drill-1311-1`.
  Not in the repo.
- Turns: 1 01:07:33-01:08:08 ($0.379), 2 01:08:08-01:16:28 ($0.774
  cumulative), 3 01:16:29-01:58:42 ($1.512), 4 01:58:42-01:58:52 ($1.567).

| Job | What |
| --- | --- |
| `429b9fde735f` | the speaker still (setup, FLUX 3 Klein) |
| `70931f6f96df` | P1, the probe, 3 segments (run `20261114-070848-fb884ea3`) |
| `862010673c76` | P2a, 4 segments |
| `8122b9e06bf0` | P2b, 4 segments |
| `34de8a1eaa50` | P3a, 6 segments |
| `569c0e68b774` | P3b, 5 segments |
| `b0ea5067290e` | P3c, 6 segments |
| `a15a5bd62cc9` | P7, the closing line, 2 segments |
| `c89a2d8108e5` | the join: `GettysburgJoin/20261114-075802-eb0c4c07/final/gettysburg-0.0.mp4` |
