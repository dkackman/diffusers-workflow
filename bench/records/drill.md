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
