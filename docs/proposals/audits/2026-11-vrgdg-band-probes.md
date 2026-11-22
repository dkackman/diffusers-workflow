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

Every arm's prompt and references are quoted verbatim under [Prompts](#prompts) at the end of
this record, copied from `bench/records/probe-1528-arms.json` (the arguments sent to
`run_workflow`, not a `bench/driver.py` arm file). The prompts were also saved to the server's
library as `prompt:probe1528-*`. In each pair, the two prompts differ only in the probed
sentence:

| Pair | Sentence in the arm that should show the effect | Sentence in the other arm |
| --- | --- | --- |
| P1 | `<Subject 3> is the man in <Picture 3>, whose face, bald head, build and grey button overshirt over a navy T-shirt define his appearance for the whole video.` (third reference given) | `<Subject 3> is a bald, heavy-set man in his forties in a grey button overshirt over a navy T-shirt; he has no reference picture.` (no third reference; retention `newly generated from the description above.`) |
| P2 | `His hair is rain-soaked and plastered to his forehead, rain-wet skin glistening, water running down his face from the downpour he has just come in out of.` | `His hair is dry and tousled, and his face is lit softly from the window.` |
| P3 | `He stands at the table winding a guitar cable into neat loops and setting it down, then sorting the rest of the cables one by one for most of the shot. As the last action of the shot he turns and walks out of frame to the right, leaving the room empty.` | `In the first half of the shot he turns from the table and walks out of frame to the right. For the rest of the shot the room is empty, the coiled cables lying on the table and the amplifier light glowing.` |
| P4 | (none: the no-lips arm) | `His lips stay closed the whole shot; he does not sing.` |

A claim **held** only when both seeds of the arm that should show the effect show it.
It is **refuted** when neither seed shows it. It is **not settled** when the seeds disagree or
frames can't decide. All 16 jobs succeeded.

Frames were read with `get_output_frames`: 6-frame samples across the clip, frames chosen with
`at` around the moment in question, and `crop` at native size for faces and mouths. Claude can't
see motion or hear audio. Where that leaves a claim undecided, the verdict says "not settled — for
Don (`taste`)" and the item is listed under [For Don](#for-don-taste).

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
| P2 | d. Wet or rain-soaked words near a face give streaks on the skin. | **not settled — for Don (`taste`)** | No streak in any frame read (per job, below): the wet arm shows a mild sheen and a few droplets, the neutral arm dry skin. The prompt asks for "water running down his face", and a running streak is motion, which stills can't rule out. Don decides by watching. The reference is a man, so this probe tested the wet wording only, not a description overriding a reference (see *Deviations*). |
| P3 | j. An exit written as the shot's last action doesn't happen. | **refuted**: the exit happens, but it ends on the last frames | **Exit-last arm, both seeds:** Pat winds cables, turns at about frame 96 and walks out to the right. By frame 118 he is half out of frame, and at frame 123 only a sliver of his shoulder and arm is left at the right edge. The room is never seen empty. **Exit-early arm, both seeds:** he is gone by about frame 62, and frames 96-123 are the empty room. The exit isn't dropped. Written last, it simply takes the whole shot, so a shot that needs the room empty at the end has to put the exit in its first half. |
| P4 | n. A silent person in a shot holding a sung slice mouths the vocal unless "closed lips" is stated. o. So use a vocal-free slice. | **refuted. Rule o is not needed.** | A drummer-only shot (`asset:qa-cast/hal-portrait.jpg`) holds `asset:song/la-vela.mp3` from 42.5 s (`start_frame` 1020) for 124 frames. The sung phrase sits in frames ~58-123 of that slice (below). **No-lips arm, both seeds:** the mouth is closed at all six frames read inside the sung phrase (59, 67, 79, 91, 110, 119), face `crop` at native size. Seed 1001 was also closed in every frame from 60 to 75. **Closed-lips arm:** closed at the same six frames on seed 2002, and at 59-110 on seed 1001. At 119 on seed 1001 the lips part with a red smear on the lower lip, in the arm that *states* closed lips. Without the sentence, the drummer didn't mouth the vocal on either seed, so rule n isn't needed and rule o (a vocal-free slice) isn't earned. Per Q4 on #1528, no vocal-free-slice idea is filed. |

None of the four claims held, and no pair split across seeds. P1, P3 and P4 are refuted; P2 is
not settled and waits on Don.

### P2 per job

Each wet job was read in a native-size `crop` on the face (`max_dimension` 400). The neutral
jobs were read on the full frame.

| Job | Arm, seed | Crop `[x, y, w, h]` | Frames | Seen |
| --- | --- | --- | --- | --- |
| 95a195babbff | wet, 1001 | [220, 40, 340, 400] | 30, 90 | Mild sheen on the forehead, nose and cheeks. One droplet or highlight near the chin at frame 30. No streak, smear or running water. The hair stays curly and isn't plastered to the forehead. |
| b777e3a684f7 | wet, 2002 | [250, 30, 340, 400] | 30, 90 | Mild sheen on the forehead and cheeks, faint droplets, the fringe a little damp. No streak. At frame 90 he glances down, as prompted. |
| 90783cbb39da | neutral, 1001 | full frame (512 wide) | 30 | Dry, matte skin, no sheen. |
| 262c9f57133f | neutral, 2002 | full frame (512 wide) | 30 | Dry, matte skin, no sheen. |

The wet wording gives a faint wet look and no artefact on these frames. It also doesn't give what
it asks for: no hair plastered down and no water running. Whether a streak crawls in motion is
Don's call.

### P4 per job

**Where the sung phrase sits.** The slice is la-vela.mp3 from 42.5 s, the same slice the earlier
record `docs/proposals/audits/2026-10-h3-wangp-probes.md` used. Whisper transcribed it as "Pass it
to your sister, pass it to your brother Yeah, no more silence, you'll let the whole heal ring".
That record's sung jobs (d2384deebb50, f27b7a7af223, 3d9a4e7377fa, 6d7f6faf8e60) show the
singer's mouth shut at 0.2 s and 1.1 s, open at 2.45 s, 2.8 s and 3.3 s, and making an "o" at
4.6 s. So the vocal runs from about 2.4 s to the slice's end: frames ~58-123 at 24 fps. The frames
below sit at 2.46, 2.79, 3.29, 3.79, 4.58 and 4.96 s, at the singer's open-mouth moments and
between them.

All read with `get_output_frames`, a native-size face `crop` (`max_dimension` 200). The no-lips
jobs were read with `hear` 1, which returned 1 s of the held audio around each frame. Claude can't
listen to it, so the sung window rests on the transcript and the earlier record, not on that
audio.

| Job | Arm, seed | Crop `[x, y, w, h]` | Frames | Mouth |
| --- | --- | --- | --- | --- |
| 691da72c357b | no lips, 1001 | [340, 0, 160, 200] | 59, 67, 79, 91, 110, 119 | Closed in all six. He faces the camera with a level head and his lips pressed together. |
| 2bd90fac4541 | no lips, 2002 | [390, 0, 160, 200] | 59, 67, 79, 91, 110, 119 | Closed in all six. His head bows with the beat, eyes down at 67, 79, 110 and 119. |
| d6c11391ae78 | closed lips, 1001 | [340, 0, 160, 200] | 59, 67, 79, 91, 110, 119 | Closed at 59-110. At 119 the lips part slightly, with a red smear on the lower lip. |
| 6835bcd04e95 | closed lips, 2002 | [390, 0, 160, 200] | 59, 67, 79, 91, 110, 119 | Closed in all six. |

### What stage B takes from this

- **P1 and P4** go under *Probed and not a rule* in `references/staging.md` (P4 in
  `references/cuts.md`). Rule o (a vocal-free slice) is not needed and isn't written.
- **P3** goes in as an observation, not a ban: an exit written as the last action is still
  under way on the last frame. Write it into the first half when the shot must end on an empty
  room.
- **No idea is filed** for the vocal-free slice.
- **P2 waits on Don's `taste` call** below. If he sees streaks crawl, stage B writes rule d as a
  rule. If he doesn't, P2 goes under *Probed and not a rule* with P1 and P4.

## For Don (`taste`)

1. **P2: watch for wet-skin streaks in motion.** Play `95a195babbff` and `b777e3a684f7`
   (wet arm) against `90783cbb39da` and `262c9f57133f` (neutral arm). Do streaks crawl on the
   skin? Frames show only a sheen and droplets.

P4 is refuted on frames and needs no call from Don. If he plays `691da72c357b` and
`2bd90fac4541` anyway and sees the drummer mouth the vocal, that reopens rule n and rule o.

## Deviations and notes

- **P1 reuses the house cast.** The three cast members are the existing `cast/` and `qa-cast/`
  assets, not new portraits. Subject 3's description is Hal's look, so the two-reference arm
  tests "described, not referenced" against a person the model could plausibly redraw. A
  description unlike any reference would test the stranger claim more sharply, but would no
  longer keep the arms identical outside the reference.
- **P2's reference is a man.** `asset:qa-cast/priya-portrait.jpg` is the man from its Z-Image
  prompt ("a wiry man in his early 30s with short curly dark hair... mustard-yellow sweater vest
  over a white button-down shirt"), whatever its name says. Both P2 prompts describe him that way,
  so P2 tested the wet wording on a face that matches its description. It says nothing about a
  description overriding a reference.
- **P4 uses one point in one song.** The slice starts at 42.5 s. Its sung phrase is placed from
  the earlier record's transcript and mouth timings, not by ear (see *P4 per job*).
- **Quiet audio is expected.** Every P1-P3 job carries the server's near-silent audio warning
  (mean -45 to -74 dBFS). That is expected: those prompts ask for no speech and quiet room
  tone. The P4 jobs carried no warnings.
- **Two artifacts on the server.** The bench workflows were saved to lem (`save_workflow`)
  and the prompts to its library (`save_prompt`), so the jobs could be run by name. Neither
  changes a shipped default.

## Prompts

Each arm's prompt exactly as sent, generated from `bench/records/probe-1528-arms.json`.

### P1-three-refs

Workflow `probe-1528-shot`. References, in `<Picture n>` order: `asset:cast/pat.jpg`, `asset:cast/howie.jpg`, `asset:qa-cast/hal-portrait.jpg`.

```text
subject_definitions:
<Subject 1> is the older man in <Picture 1>, whose face, grey hair, moustache and khaki short-sleeved shirt define his appearance for the whole video.
<Subject 2> is the young woman in <Picture 2>, whose face, glasses, dark hair tied up and blue hoodie define her appearance for the whole video.
<Subject 3> is the man in <Picture 3>, whose face, bald head, build and grey button overshirt over a navy T-shirt define his appearance for the whole video.

summary:
[reference generation] The target video shows the three members of a band, <Subject 1>, <Subject 2> and <Subject 3>, standing side by side in their rehearsal room, in one continuous shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the face, hair, moustache and shirt of <Picture 1> are kept exactly.
<Subject 2> (appears in [Shot 1]): fully_preserved - the face, glasses, hair and hoodie of <Picture 2> are kept exactly.
<Subject 3> (appears in [Shot 1]): fully_preserved - the face, bald head, build and clothing of <Picture 3> are kept exactly.

detailed_description:
The target video is in realistic photographic style. [Shot 1] A static medium-wide shot at eye level frames exactly three people standing side by side in a small rehearsal room with foam panels on the walls and a drum kit and amplifiers behind them. <Subject 1> stands on the left of frame, <Subject 2> in the centre, and <Subject 3> on the right of frame. All three face the camera. <Subject 1> folds his arms and gives a small nod, <Subject 2> pushes her glasses up and smiles, and <Subject 3> shifts his weight and rests his hands in his pockets. No one speaks. The camera holds a static medium-wide shot throughout, all three in frame from head to knee.

overall_soundscape:
The quiet room tone of a small rehearsal space, the low hum of an idle amplifier, and the soft rustle of clothing as they shift.

non_diegetic_music:
N/A
```

### P1-two-refs

Workflow `probe-1528-shot`. References, in `<Picture n>` order: `asset:cast/pat.jpg`, `asset:cast/howie.jpg`.

```text
subject_definitions:
<Subject 1> is the older man in <Picture 1>, whose face, grey hair, moustache and khaki short-sleeved shirt define his appearance for the whole video.
<Subject 2> is the young woman in <Picture 2>, whose face, glasses, dark hair tied up and blue hoodie define her appearance for the whole video.
<Subject 3> is a bald, heavy-set man in his forties in a grey button overshirt over a navy T-shirt; he has no reference picture.

summary:
[reference generation] The target video shows the three members of a band, <Subject 1>, <Subject 2> and <Subject 3>, standing side by side in their rehearsal room, in one continuous shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the face, hair, moustache and shirt of <Picture 1> are kept exactly.
<Subject 2> (appears in [Shot 1]): fully_preserved - the face, glasses, hair and hoodie of <Picture 2> are kept exactly.
<Subject 3> (appears in [Shot 1]): newly generated from the description above.

detailed_description:
The target video is in realistic photographic style. [Shot 1] A static medium-wide shot at eye level frames exactly three people standing side by side in a small rehearsal room with foam panels on the walls and a drum kit and amplifiers behind them. <Subject 1> stands on the left of frame, <Subject 2> in the centre, and <Subject 3> on the right of frame. All three face the camera. <Subject 1> folds his arms and gives a small nod, <Subject 2> pushes her glasses up and smiles, and <Subject 3> shifts his weight and rests his hands in his pockets. No one speaks. The camera holds a static medium-wide shot throughout, all three in frame from head to knee.

overall_soundscape:
The quiet room tone of a small rehearsal space, the low hum of an idle amplifier, and the soft rustle of clothing as they shift.

non_diegetic_music:
N/A
```

### P2-wet

Workflow `probe-1528-shot`. References, in `<Picture n>` order: `asset:qa-cast/priya-portrait.jpg`.

```text
subject_definitions:
<Subject 1> is the man in <Picture 1>, whose face, curly dark hair, white shirt and mustard sweater vest define his appearance for the whole video.

summary:
[reference generation] The target video shows <Subject 1> in a close-up, looking toward the camera in a quiet apartment, in one continuous shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the face, hair, shirt and vest of <Picture 1> are kept exactly, with the head newly animated.

detailed_description:
The target video is in realistic photographic style. [Shot 1] A close-up frames <Subject 1> from the shoulders up, centred and facing the camera, a brick wall soft and out of focus behind him. His hair is rain-soaked and plastered to his forehead, rain-wet skin glistening, water running down his face from the downpour he has just come in out of. He takes a slow breath, glances down, then lifts his eyes back to the lens with a faint, tired smile. No one speaks. The camera holds a static close-up throughout.

overall_soundscape:
The quiet room tone of an apartment, his slow breath, and the faint sound of traffic far outside.

non_diegetic_music:
N/A
```

### P2-neutral

Workflow `probe-1528-shot`. References, in `<Picture n>` order: `asset:qa-cast/priya-portrait.jpg`.

```text
subject_definitions:
<Subject 1> is the man in <Picture 1>, whose face, curly dark hair, white shirt and mustard sweater vest define his appearance for the whole video.

summary:
[reference generation] The target video shows <Subject 1> in a close-up, looking toward the camera in a quiet apartment, in one continuous shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the face, hair, shirt and vest of <Picture 1> are kept exactly, with the head newly animated.

detailed_description:
The target video is in realistic photographic style. [Shot 1] A close-up frames <Subject 1> from the shoulders up, centred and facing the camera, a brick wall soft and out of focus behind him. His hair is dry and tousled, and his face is lit softly from the window. He takes a slow breath, glances down, then lifts his eyes back to the lens with a faint, tired smile. No one speaks. The camera holds a static close-up throughout.

overall_soundscape:
The quiet room tone of an apartment, his slow breath, and the faint sound of traffic far outside.

non_diegetic_music:
N/A
```

### P3-exit-last

Workflow `probe-1528-shot`. References, in `<Picture n>` order: `asset:cast/pat.jpg`.

```text
subject_definitions:
<Subject 1> is the older man in <Picture 1>, whose face, grey hair, moustache and khaki short-sleeved shirt define his appearance for the whole video.

summary:
[reference generation] The target video shows <Subject 1> in a small rehearsal room, leaving the frame, in one continuous shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the face, hair, moustache and shirt of <Picture 1> are kept exactly.

detailed_description:
The target video is in realistic photographic style. [Shot 1] A static medium-wide shot at eye level frames a small rehearsal room with foam panels on the walls, an amplifier on the floor and a folding table in the centre of frame. <Subject 1> stands behind the table, facing the camera. He stands at the table winding a guitar cable into neat loops and setting it down, then sorting the rest of the cables one by one for most of the shot. As the last action of the shot he turns and walks out of frame to the right, leaving the room empty. No one speaks. The camera holds a static medium-wide shot throughout.

overall_soundscape:
The quiet room tone of a small rehearsal space, the rattle of cables on the table, and his footsteps on a hard floor.

non_diegetic_music:
N/A
```

### P3-exit-early

Workflow `probe-1528-shot`. References, in `<Picture n>` order: `asset:cast/pat.jpg`.

```text
subject_definitions:
<Subject 1> is the older man in <Picture 1>, whose face, grey hair, moustache and khaki short-sleeved shirt define his appearance for the whole video.

summary:
[reference generation] The target video shows <Subject 1> in a small rehearsal room, leaving the frame, in one continuous shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the face, hair, moustache and shirt of <Picture 1> are kept exactly.

detailed_description:
The target video is in realistic photographic style. [Shot 1] A static medium-wide shot at eye level frames a small rehearsal room with foam panels on the walls, an amplifier on the floor and a folding table in the centre of frame. <Subject 1> stands behind the table, facing the camera. In the first half of the shot he turns from the table and walks out of frame to the right. For the rest of the shot the room is empty, the coiled cables lying on the table and the amplifier light glowing. No one speaks. The camera holds a static medium-wide shot throughout.

overall_soundscape:
The quiet room tone of a small rehearsal space, the rattle of cables on the table, and his footsteps on a hard floor.

non_diegetic_music:
N/A
```

### P4-no-lips

Workflow `probe-1528-held`. `performer`: `asset:qa-cast/hal-portrait.jpg`, `start_frame`: 1020.

```text
subject_definitions:
<Subject 1> is the drummer in <Picture 1>, whose face, bald head, build and grey button overshirt over a navy T-shirt define his appearance for the whole video.
<Audio 1> is the song playing in the room: a band with a lead vocal.

summary:
[reference generation + audio reuse] The target video shows <Subject 1> playing the drums to the song carried by <Audio 1> in a rehearsal room, in one continuous shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the face, bald head, build and clothing of <Picture 1> are kept exactly, with the body newly animated to drum.
<Audio 1>: partially_copy - the piece of the song covering this part of the video is reused as the soundtrack.

detailed_description:
The target video is in realistic photographic style. [Shot 1] A medium shot frames <Subject 1> alone, seated behind a drum kit in a small rehearsal room with foam panels on the walls, his face and shoulders clear above the cymbals. He plays the beat of <Audio 1>, sticks striking the snare and hi-hat in time, his head nodding with the tempo. The camera holds a static medium shot throughout, his face in view the whole time.

overall_soundscape:
The song of <Audio 1> carries the shot, with the close crack of the snare and the hiss of the hi-hat in the room.

non_diegetic_music:
N/A
```

### P4-closed-lips

Workflow `probe-1528-held`. `performer`: `asset:qa-cast/hal-portrait.jpg`, `start_frame`: 1020.

```text
subject_definitions:
<Subject 1> is the drummer in <Picture 1>, whose face, bald head, build and grey button overshirt over a navy T-shirt define his appearance for the whole video.
<Audio 1> is the song playing in the room: a band with a lead vocal.

summary:
[reference generation + audio reuse] The target video shows <Subject 1> playing the drums to the song carried by <Audio 1> in a rehearsal room, in one continuous shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the face, bald head, build and clothing of <Picture 1> are kept exactly, with the body newly animated to drum.
<Audio 1>: partially_copy - the piece of the song covering this part of the video is reused as the soundtrack.

detailed_description:
The target video is in realistic photographic style. [Shot 1] A medium shot frames <Subject 1> alone, seated behind a drum kit in a small rehearsal room with foam panels on the walls, his face and shoulders clear above the cymbals. He plays the beat of <Audio 1>, sticks striking the snare and hi-hat in time, his head nodding with the tempo. His lips stay closed the whole shot; he does not sing. The camera holds a static medium shot throughout, his face in view the whole time.

overall_soundscape:
The song of <Audio 1> carries the shot, with the close crack of the snare and the hiss of the hi-hat in the room.

non_diegetic_music:
N/A
```
