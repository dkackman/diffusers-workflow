# VRGDG band-video claims: probes (#1561, feature #1528 stage A)

Probe date: 2026-11-22. Server `lem` (both RTX 3090s), `develop` @ 38eb2e5b. Agent: claude-opus-5-5
(anthropic). Workspace `probe-1528`. No engine file was changed.

The claims come from the outside band-video run that #1528 quotes (two unattended 30-scene
videos; base H3 or a finetune is not stated). Plan v2 on #1528 picked four of its rules to probe
before any becomes skill text: a (P1), d (P2), j (P3) and n + o (P4).

## Configuration

Every arm ran Ref2VA with the 768p turbo LoRA (`lightx2v/Minimax-h3-Turbo`,
`minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors`): 960x544, 124 frames, 24 fps,
9 steps, shift 12 / 3, int4 sdnq (VAE int8), sparse attention (tau 0.5, top-k 0.15). Each arm ran
on seeds 1001 and 2002. These are `templates/minimax/reference-to-video`'s own settings.

| Probe | Workflow (`bench/workflows/`) | Saved on lem as | What varies |
| --- | --- | --- | --- |
| P1-P3 | `probe-1528-shot.json` (`Probe1528Shot`): the template's `shot` step, with the whole `references` list a variable | `probe-1528-shot` | `prompt`, `references` |
| P4 | `probe-1528-held.json` (`Probe1528HeldShot`): `slice_audio` of the song, then one shot taking the performer's portrait plus that slice as an audio reference and holding it (`hold_audio: previous_result:slice`), as `templates/minimax/music-video`'s `shot` step does | `probe-1528-held` | `prompt` |

The exact prompts and references of every arm are in `bench/arms/probe-1528.json`. The prompts
were also saved to the server's library as `prompt:probe1528-*`. In each pair, the two prompts
differ only in the probed sentence.

A claim **held** only when both seeds of the arm that should show the effect show it.
It is **refuted** when neither seed shows it. It is **not settled** when the seeds disagree or
frames can't decide. All 16 jobs succeeded.

Frames were read with `get_output_frames`: 6-frame samples across the clip, frames chosen with
`at` around the moment in question, and `crop` at native size for faces and mouths. Claude can't
see motion or hear audio, so the perceptual verdicts (P2, P4) also go to Don as `taste` items
(below).

## Arms

| Arm | Seed | Job | Size | Frames | Steps |
| --- | --- | --- | --- | --- | --- |
| P1 three references | 1001 | e786c86ae4b1 | 960x544 | 124 | 9 |
| P1 three references | 2002 | 9c6007656a60 | 960x544 | 124 | 9 |
| P1 two references, third described | 1001 | c974b1fb8691 | 960x544 | 124 | 9 |
| P1 two references, third described | 2002 | dd83b3c5ea33 | 960x544 | 124 | 9 |
| P2 wet wording | 1001 | 95a195babbff | 960x544 | 124 | 9 |
| P2 wet wording | 2002 | b777e3a684f7 | 960x544 | 124 | 9 |
| P2 neutral wording | 1001 | 90783cbb39da | 960x544 | 124 | 9 |
| P2 neutral wording | 2002 | 262c9f57133f | 960x544 | 124 | 9 |
| P3 exit as the last action | 1001 | 7a140e59f431 | 960x544 | 124 | 9 |
| P3 exit as the last action | 2002 | fd01b6f3c4d2 | 960x544 | 124 | 9 |
| P3 exit in the first half | 1001 | 4c6ef3f68eee | 960x544 | 124 | 9 |
| P3 exit in the first half | 2002 | 0b9ab80e67b8 | 960x544 | 124 | 9 |
| P4 no lips wording | 1001 | 691da72c357b | 960x544 | 124 | 9 |
| P4 no lips wording | 2002 | 2bd90fac4541 | 960x544 | 124 | 9 |
| P4 "closed lips" stated | 1001 | d6c11391ae78 | 960x544 | 124 | 9 |
| P4 "closed lips" stated | 2002 | 6835bcd04e95 | 960x544 | 124 | 9 |

## Verdicts

| Probe | Claim (rule on #1528) | Verdict | Observation |
| --- | --- | --- | --- |
| P1 | a. A group member with no cast reference comes out as a stranger or a clone, so give every cast member a reference in every group shot. | **refuted** | **Three-reference arm, both seeds:** three people, each matching their reference: Pat (`asset:cast/pat.jpg`) left, Howie (`asset:cast/howie.jpg`) centre, Hal (`asset:qa-cast/hal-portrait.jpg`) right. **Two-reference arm:** Hal's reference was dropped and Subject 3 described only ("a bald, heavy-set man in his forties in a grey button overshirt over a navy T-shirt"). Both seeds show exactly three people. Pat and Howie match their references, and the third is a new man drawn from the description. There is no fourth person, no second Pat or Howie, and no head that is neither a reference nor the described man. On seed 1001 the described man looks much like Hal, because the description is Hal's own wardrobe and build. That is the description working, not a clone. |
| P2 | d. Wet or rain-soaked words near a face give streaks on the skin. | **refuted** on frames; `taste` item | **Wet arm, both seeds:** in a native-size `crop` of the face, the skin has a mild sheen, a few droplets and damp strands of hair on the forehead. There are no streaks, smears or banding. **Neutral arm, both seeds:** clean, dry skin. The wording gives a faint wet look and no artefact. Stills can't show a streak that only crawls in motion, so Don confirms it by watching (below). |
| P3 | j. An exit written as the shot's last action doesn't happen. | **refuted**: the exit happens, but it ends on the last frames | **Exit-last arm, both seeds:** Pat winds cables, turns at about frame 96 and walks out to the right. By frame 118 he is half out of frame, and at frame 123 only a sliver of his shoulder and arm is left at the right edge. The room is never seen empty. **Exit-early arm, both seeds:** he is gone by about frame 62, and frames 96-123 are the empty room. The exit isn't dropped. Written last, it simply takes the whole shot, so a shot that needs the room empty at the end has to put the exit in its first half. |
| P4 | n. A silent person in a shot holding a sung slice mouths the vocal unless "closed lips" is stated. o. So use a vocal-free slice. | **refuted** on frames; `taste` item | A drummer-only shot (`asset:qa-cast/hal-portrait.jpg`) holds `asset:song/la-vela.mp3` from 42.5 s (`start_frame` 1020) for 124 frames. **No-lips arm, both seeds:** he plays and nods, and his mouth stays closed in 8 frames sampled across the clip and in every frame from 60 to 75 (seed 1001, `crop` on the mouth). **Closed-lips arm:** seed 2002 is closed throughout. On seed 1001 the mouth is closed until about frame 118, then the lips part slightly for the last 0.2 s. That is not mouthing, and it happens in the arm that *states* closed lips. Without the sentence, the non-singer didn't mouth the vocal on either seed, so rule n isn't needed here and rule o (a vocal-free slice) isn't earned. Per Q4 on #1528, no vocal-free-slice idea is filed. Two limits: stills can't show a mouth moving in sync, and Claude can't hear whether the slice at 42.5 s carries a sung line. Both are Don's to check. |

None of the four claims held, and no pair split across seeds.

### What stage B takes from this

- **P1, P2 and P4** go under *Probed and not a rule* in `references/staging.md` (P4 in
  `references/cuts.md`).
- **P3** goes in as an observation, not a ban: an exit written as the last action is still
  under way on the last frame. Write it into the first half when the shot must end on an empty
  room.
- **No idea is filed** for the vocal-free slice.
- **P2 and P4** are recorded as refuted on frames, pending Don's `taste` check below. If he
  overturns either, stage B writes it as a rule instead.

## For Don (`taste`)

1. **P2: watch for wet-skin streaks in motion.** Play `95a195babbff` and `b777e3a684f7`
   (wet arm) against `90783cbb39da` and `262c9f57133f` (neutral arm). Do streaks crawl on the
   skin? Frames show only a sheen and droplets.
2. **P4: watch and listen for a mouthed vocal.** Play `691da72c357b` and `2bd90fac4541`
   (no lips wording). Is a vocal audible in the held slice, and does the drummer's mouth move
   with it? Also check whether the lip parting at the end of `d6c11391ae78` reads as singing.

## Deviations and notes

- **P1 reuses the house cast.** The three cast members are the existing `cast/` and `qa-cast/`
  assets, not new portraits. Subject 3's description is Hal's look, so the two-reference arm
  tests "described, not referenced" against a person the model could plausibly redraw. A
  description unlike any reference would test the stranger claim more sharply, but would no
  longer keep the arms identical outside the reference.
- **P4 uses one point in one song.** The slice starts at 42.5 s. Whether it
  carries a sung line wasn't checked by ear, so it's part of the `taste` item.
- **Quiet audio is expected.** Every P1-P3 job carries the server's near-silent audio warning
  (mean -45 to -74 dBFS). That is expected: those prompts ask for no speech and quiet room
  tone. The P4 jobs carried no warnings.
- **Two artifacts on the server.** The bench workflows were saved to lem (`save_workflow`)
  and the prompts to its library (`save_prompt`), so the jobs could be run by name. Neither
  changes a shipped default.
