# H3 multi-character band-video rules: four probes and an outside-reported list (#1528)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-22, on mini-ai). Plan v2 approved by Don on 2026-11-21 (every
question at its default) as "build smaller"; two stages, #1561 (probes) and
#1562 (skill text), both verified on lem. The probe record is
`docs/proposals/audits/2026-11-vrgdg-band-probes.md`; this doc is the design
record around it.

## The idea

vrgamegirl19/comfyui-vrgamedevgirl (read at 12adc26, pack 10.0.9) ships a
band-music-video agent skill written after two unattended 30-scene H3
Ref2VA videos. Its lessons list multi-character prompt rules (strangers,
clones, position maps, wet words, eyes, props, furniture, exits, crane
framing, final hits, static wides, closed lips), audio rules (vocal-free
slices, stems pick the players, time lyrics on the vocal stem) and one
review rule (count heads on full-size frames). The licence is custom
source-available, so only the claims were used; no wording or code.

## Decisions

- **Verdict: build smaller.** Probe the four cheap, testable claims on lem
  before any became a rule; write the rest in as an explicitly unprobed,
  outside-reported list. The source doesn't say whether its runs were base
  H3 or a finetune, and our `staging.md` contract is "every rule came from
  a render that went wrong on our server".
- **Probed (Q1):** P1 missing-reference stranger (rule a), P2 wet words
  near a face (d), P3 exit written last (j), P4 a silent player under a
  held sung slice (n + o). 16 turbo clips.
- **Outside-reported, unprobed (Q2):** b (clone wording: "exactly one
  person", "the only drummer", head and prop counts, always beside
  placement, never instead of it), c (a fixed position map; flat floor over
  a raised stage), e–m (eyes, small props, prop steps, fixed furniture,
  face touch, crane framing, final hit played through, no long static
  wide), and q's hand-check of the last lyric lines and held notes.
- **Frame size (Q3):** the `get_output_frames` default stays 512 px; one
  line says to count heads on wide shots with `max_dimension` raised or a
  `crop` on the group. `count=12` for the staging sheet stays.
- **Cut (Q4):** a vocal-free slice option in `music-video` (rule o) was
  new template surface, filed only if P4 held. It didn't, so nothing was
  filed. Rule p (stems pick who plays) was cut: no per-stem activity
  measure exists. *Comes back on* a field report with one.
- **Placement:** no new reference file (SKILL.md is near its byte cap);
  everything went into `staging.md`, `cuts.md`, `quality-and-speed.md` §4,
  and the #1519 H3 intents index, which had landed by stage B.

## What the probes found

960x544, 124 frames, 9 steps, turbo LoRA, seeds 1001 and 2002, workspace
`probe-1528`; held or refuted only when both seeds agree.

| # | Claim | Verdict |
| --- | --- | --- |
| P1 | a group member with no reference comes out as a stranger or clone | **refuted**: both seeds drew exactly three people; the described, unreferenced member came out as the description, no fourth head, no clone |
| P2 | wet words near a face streak the skin | **not settled, for Don (`taste`)**: stills show a mild sheen and droplets, no streak; a running streak is motion that stills can't rule out |
| P3 | an exit written last doesn't happen | **refuted**: the exit happens but takes the rest of the shot; put it in the first half if the shot must end on an empty frame |
| P4 | a silent player mouths a held sung slice unless closed lips is stated | **refuted**: closed at six frames inside the sung phrase on both seeds without the sentence; rule o not earned |

## What was built

**Stage A, #1561** (bench only): `732a0be2`, `b772d5c1` (develop
`494bd4f9`); bounce rework `eaa9e3d8` (develop `1db9fcbe`). The audit
record, and the arm arguments in `bench/records/probe-1528-arms.json`.

**Stage B, #1562** (plugin only): `c6573a54`, develop `b306d9f8`.
- `staging.md`: P1–P3 under *Probed and not a rule* (P2 as not settled:
  neither use nor avoid the wet wording on its account); b and c beside
  the placement rule; a new *Reported from outside, not probed here*
  section with its source line and items e–m, each "they report".
- `cuts.md`: the lyric hand-check (outside-reported) and *Probed, not a
  rule: a silent player under a held sung slice* for P4.
- `quality-and-speed.md` §4: the head-count line naming `max_dimension` /
  `crop`.
- `intents.md`: refuted rows for P1, P3, P4; `untested` rows for b, c,
  e–m and q.
- Four pins in `TestMiniMaxH3Skill`, including one that the `music-video`
  template has no vocal or stem variable. SKILL.md untouched; no
  `dialogue.md` cross-reference (it depended on P4 holding).

## Deferred

- **P2's playback verdict** is Don's. When he gives it, the `staging.md`
  line becomes a rule or a *Probed and not a rule* entry, and the intents
  row follows.
- The outside-reported list is promoted or removed line by line by a field
  report or a later probe.

## Cost and bounces

No stage comment states `usage:` figures, so cost is left out. The plan
estimated ~$7 and about 40–50 min of lem for the 16 clips.

| Stage | Bounces | Why |
| --- | --- | --- |
| #1561 A | 1 | the frames matched the record, but the hand-off quoted no prompts, gave P2 and P4 no per-job frames, and gave P2/P4 two verdicts each (ruled and "for Don"); reworked in `eaa9e3d8` |
| #1562 B | 0 | |
