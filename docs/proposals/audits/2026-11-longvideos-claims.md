# MiniMax-H3-LongVideos claims: probes (#1193, long-form plan Task 26, #1199)

Probe date: 2026-11-05 (lem's clock reads 2026-11-06 in the run ids). Server `lem`, the long-form
bench server (port 8766, both RTX 3090s), `feat/long-form-media` @ bc2fdc50 (the face check @
4be400db), dw 0.19.0-alpha.1, diffusers 0.42.1, torch 2.14.1+cu130. Agent: claude-opus-5-5
(anthropic). Workspace `default` under `~/bench-workspace`. No engine file was changed.

The claims come from Smite79/MiniMax-H3-LongVideos (README as of 2026-11-01), as #1193 lists
them. That repository's licence forbids redistribution, so nothing here copies its text; the
claims are the paraphrases on #1193.

## Configuration

Every probe ran at 960x544, 124 frames, 24 fps, int4 sdnq (VAE int8), sparse attention
(tau 0.5, top-k 0.15), on seeds 1001 and 2002.

| Claim | Workflow (`bench/workflows/`) | Pipeline | LoRA (`lightx2v/Minimax-h3-Turbo`) | Steps | Shift |
| --- | --- | --- | --- | --- | --- |
| 1, 5 | `t26-t2va.json`: `templates/minimax/video-with-audio` composed by name, only `prompt` and `seed` set | T2VA | `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` (the template's) | 5 | 6 / 3 |
| 2, 3 | `t26-ref2va.json`: `templates/minimax/reference-to-video`'s step, audio reference dropped, image references `picture_1` and an optional `picture_2` | Ref2VA | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` | 9 | 12 / 3 |
| 4 | `t26-continued.json`: `templates/minimax/chain-video-continuity`'s step, its seam built by hand (below) | Ref2VA | the same ref2v file | 9 | 12 / 3 |

The pictures came from `t26-portraits.json` (job af6dd086993b): `asset:bench/t26-stranger.jpg`,
a Z-Image head-and-shoulders portrait of a grey-bearded man in round glasses and a mustard
corduroy jacket, and `asset:bench/t26-walker-face.png`, a 512x512 `recenter_crop` of the bench
walker (`asset:bench/walker.png`, a woman in a long olive double-breasted coat, three-quarter
length) to her face and hair, with only a sliver of collar left.

Arms are in `bench/arms/t26-claims.json` (plus the first claim-1 run in `t26-prep.json` and the
face check in `t26-facecheck.json`); records are `bench/records/t26-*.json`. A claim **held**
only when both seeds of the arm that breaks it show the effect and neither seed of the arm that
follows it does; **refuted** when the breaking arm doesn't show the effect; **not settled**
when the seeds disagree or the measure can't decide. All 25 jobs succeeded.

Frames were read as 3x3 contact sheets (every 15th frame, 0 to 120) with lem's ffmpeg, plus
single frames where a sheet left a doubt. Faces were scored with `analyze_face_drift` (SFace,
16 samples, median cosine; OpenCV's same-person threshold is 0.363). Audio was decoded to
32 kHz mono and measured with numpy (below). No agent can listen, so the audio verdicts carry
a listening item.

## Verdicts

| # | Claim | Verdict | Jobs | Observation |
| --- | --- | --- | --- | --- |
| 1 | H3 acts out what a `<d>` line talks about; the countermeasure sentence stops it. | **refuted** | plain: 26c02d9ea5a4 (s1001), 2b93dfe5eb65 (s2002). Sentence: d45357868b54, 79a9969093de | The line names an absent red raincoat and an absent brother. On no seed of either arm did a raincoat, any red garment, a second person or an extra arm or hand appear: one woman in a white sleeveless T-shirt behind an empty counter, who speaks, shrugs and folds her arms. Same-seed clips of the two arms are near-identical, so the sentence changed nothing visible. |
| 2 | A reference picture the prompt doesn't mention is read as another person. | **refuted** | unmentioned: 6c6ebdc54886, ae1d84995e95. Control (no second picture): e1d5e3dcc86b, e05c79b91eda. Face check: 43d225620b94, 4fb8ff3ee82b, bf8c275d35d3, c00bec92b585 | With the stranger's portrait as `<Picture 2>` and never named, both seeds show only the walker on a park bench; the few blurred pedestrians far down the path are in both arms. His face didn't blend into hers: similarity to the stranger 0.116 / 0.053 against the control's 0.151 / 0.054. Side finding: her similarity to her own portrait fell, 0.390 / 0.356 against the control's 0.447 / 0.468. |
| 3 | A full-body identity picture copies its clothes back on, so crop to face and hair. | **refuted** | full picture: aea9c681dc2f, 0d5f4cafb7f2. Face-and-hair crop: 07e3855ea57c, fa445bdc6ba1 | The prompt dresses her in a red knitted turtleneck and jeans, "and no coat", in a morning kitchen. All four clips show exactly that, with no coat, no olive and nothing of the picture's street. Identity was no better from the crop: similarity 0.509 / 0.456 (full) against 0.448 / 0.566 (crop). |
| 4 | On a continued shot, leave the portrait out when the carried tail shows the person; with both, she is drawn twice. | **refuted**: dropping the portrait didn't stop the double and cost identity | 1b4869e2676a (s1001), a3930b5f3fb4 (s2002) | Segment 2 drew a second woman in the same olive coat walking behind her on both seeds, in *both* arms: with the portrait and the tail, and with the tail alone. Dropping the portrait lowered her similarity to the portrait on both seeds: 0.183 / 0.228 against 0.323 / 0.302 with it (segment 1: 0.309 / 0.291). Segment 1 itself already grew the follower on seed 2002 and a faint dark trailing shape on seed 1001, so the double comes in with the walking shot, not with the portrait. |
| 5a | Naming a steady hum (fridge, strip light) the scene doesn't need reads as a drone. | **not settled**: a low-band floor rise on both seeds, no tonal drone; listening item | hum: 313fe586f422, 412db5c1ed0e. No hum: 133cae62bfbd, c1b9be136407 | The hum arm raised the steady floor below 125 Hz on both seeds (31 Hz octave +14 / +10 dB, 63 Hz +9 / +4.5 dB over the no-hum arm) but no narrowband tone was found, and that floor sits 30-45 dB under the clip's level. Whether that is heard as a drone is an ear's call. |
| 5b | Say there is no background music, or H3 adds some. | **refuted** (by measure; listening item) | no music line: 3e449fbebbca, e0f10599f3d4, against the no-hum arm above (`non_diegetic_music: None.`) | Omitting the music section entirely added no tonal content on either seed: spectral flatness 0.31 / 0.39, no persistent spectral peak, the same noise-like rain-and-room sound as the arm that says "None." |

## Probe 1: what a line talks about

The prompt (arm "plain"):

```
summary:
[text generation] The target video shows a woman alone in a small kitchen, speaking to the camera.

detailed_description:
The target video is in realistic photographic style. [Shot 1] A static medium shot frames a woman in her thirties with short dark hair, in a plain white sleeveless T-shirt, standing alone behind a bare wooden kitchen counter and facing the camera. The counter is empty and the pale wall behind her is plain. She looks at the lens and, in a warm, clear female voice, says <d>[English] My brother Daniel borrowed my red raincoat last winter, and he still hasn't given it back.</d> Then she shrugs and folds her arms. The camera does not move.

overall_soundscape:
A quiet kitchen room tone.

non_diegetic_music:
None.
```

Arm "sentence" appends `Spoken words are only heard: the shot shows what it describes, not what
anyone talks about.` to the end of `detailed_description`. The line was rendered on all four
clips (speech-like spectra with 17-24 strong harmonic peaks a frame, -19 to -26 dBFS).

| Job | Arm, seed | Raincoat or red garment? | Second person, extra arms? |
| --- | --- | --- | --- |
| 26c02d9ea5a4 | plain, 1001 | no | no |
| 2b93dfe5eb65 | plain, 2002 | no | no |
| d45357868b54 | sentence, 1001 | no | no |
| 79a9969093de | sentence, 2002 | no | no |

This covers one easy case: one speaker, a garment she isn't wearing, a person who isn't there,
a static shot. Their own example, a garment taken off earlier in the story and drawn back on,
wasn't run; neither was a line on a moving camera or with other people in frame. Compare
`references/dialogue.md` row 10: a thing named in the prose *as off screen* does come on
screen. A thing named only inside `<d>` didn't, here.

## Probe 2: a picture the prompt doesn't name

`<Picture 1>` is the walker, `<Picture 2>` (unmentioned arm only) the stranger. The prompt
names only `<Picture 1>` and doesn't say she is alone:

```
subject_definitions:
<Subject 1> is the woman in <Picture 1>: early thirties, shoulder-length dark hair, a long olive double-breasted coat.

summary:
[reference generation] The target video shows <Subject 1> sitting on a wooden park bench in autumn, reading a paperback, then looking up and smiling.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - her face, hair, coat and proportions from <Picture 1> are kept, newly posed seated on the bench.

detailed_description:
The target video is in realistic photographic style. [Shot 1] A static medium-wide shot at eye level frames <Subject 1> sitting on a wooden park bench under trees with yellow and orange leaves, a paperback open in her hands. Soft overcast daylight. She turns a page, then looks up from the book toward the path in front of her and smiles. Leaves drift down behind her. The camera does not move.

overall_soundscape:
Light wind in the trees, dry leaves skittering across the path, a page turning, distant birds.

non_diegetic_music:
None.
```

| Job | Arm, seed | People | Walker similarity (median / min) | Stranger similarity (median / min) |
| --- | --- | --- | --- | --- |
| 6c6ebdc54886 | unmentioned, 1001 | the walker; distant blurred pedestrians | 0.390 / 0.195 | 0.116 / -0.005 (43d225620b94) |
| ae1d84995e95 | unmentioned, 2002 | the walker alone | 0.356 / 0.298 | 0.053 / 0.030 (4fb8ff3ee82b) |
| e1d5e3dcc86b | control, 1001 | the walker; distant blurred pedestrians | 0.447 / 0.393 | 0.151 / 0.085 (bf8c275d35d3) |
| e05c79b91eda | control, 2002 | the walker; distant blurred pedestrians | 0.468 / 0.304 | 0.054 / -0.041 (c00bec92b585) |

No grey beard, glasses or mustard jacket appears anywhere, on her or on anyone else. The side
finding, a 0.06-0.11 drop in her own similarity with the unused picture present, ran the same
way on both seeds, but there is no face-similarity floor for these clips to judge it against,
so it is recorded, not ruled on.

## Probe 3: clothes from the picture or the words

The same prompt for both arms; only `picture_1` changes (`asset:bench/walker.png` or
`asset:bench/t26-walker-face.png`):

```
subject_definitions:
<Subject 1> is the woman in <Picture 1>: early thirties, shoulder-length dark hair. In this video she wears a bright red knitted turtleneck sweater and light blue jeans, and no coat.

summary:
[reference generation] The target video shows <Subject 1> in a bright kitchen in the morning, pouring coffee from a glass carafe into a white mug and taking a sip.

retention_analysis:
<Subject 1> (appears in [Shot 1]): partially_preserved - her face and hair from <Picture 1> are kept. Her clothing is not taken from <Picture 1>: she wears the bright red knitted turtleneck sweater and light blue jeans described here, and no coat.

detailed_description:
The target video is in realistic photographic style. [Shot 1] A static medium shot at eye level frames <Subject 1> from the waist up, standing at a white kitchen counter in a bright kitchen with morning sun coming through a window to the left. She wears a bright red knitted turtleneck sweater and light blue jeans. She lifts a glass carafe of coffee, pours it into a white mug, sets the carafe down, and takes a slow sip, smiling. The camera does not move.

overall_soundscape:
Coffee pouring into the mug, the carafe set down on the counter, a quiet morning kitchen, birds faintly outside.

non_diegetic_music:
None.
```

| Job | Arm, seed | Clothes | Walker similarity (median / min) |
| --- | --- | --- | --- |
| aea9c681dc2f | full picture, 1001 | red turtleneck, jeans, no coat | 0.509 / 0.434 |
| 0d5f4cafb7f2 | full picture, 2002 | red turtleneck, jeans, no coat | 0.456 / 0.210 |
| 07e3855ea57c | face crop, 1001 | red cable-knit turtleneck, jeans, no coat | 0.448 / 0.368 |
| fa445bdc6ba1 | face crop, 2002 | red turtleneck, jeans, no coat | 0.566 / 0.328 |

The prompt here says the clothes are not taken from the picture, twice, and the retention line
says `partially_preserved`, which is how MiniMax's reference guide asks for a change of costume.
A prompt that dressed her differently without those words, or a picture with a bold print
(their "print and all"), wasn't run.

## Probe 4: the portrait on a continued shot

`t26-continued.json` (one job per seed) renders segment 1 from `[<Picture 1>]` (the walker),
cuts its last 48 frames with their audio as the carried tail, and renders segment 2 twice at the
same seed: **portrait** `[<Picture 1>, <Video 1>]` (what `chain-video-continuity` does on every
segment) and **carry only** `[<Video 1>]` (the claim's rule). Segment 1's prompt has her walking
alone toward a camera that tracks backward down an empty cobblestone street at golden hour;
segment 2's prompts continue the same take from `<Video 1>` and name her through `<Picture 1>`
or through `<Video 1>` respectively. The full prompts are the workflow's variables.

| Seed, segment | Women in frame | Similarity to the portrait (median / min) |
| --- | --- | --- |
| 1001, segment 1 | one; from about frame 15 a dark shape trails at her right side | 0.309 / 0.086 |
| 1001, segment 2, portrait | two: a second woman in the same coat a step behind her right shoulder, the whole clip | 0.323 / 0.202 |
| 1001, segment 2, carry only | two, the same | 0.183 / 0.102 |
| 2002, segment 1 | one, then from about frame 30 a second figure in the same coat following her | 0.291 / 0.111 |
| 2002, segment 2, portrait | two to three followers in matching coats | 0.302 / 0.123 |
| 2002, segment 2, carry only | two to three followers, the same | 0.228 / 0.103 |

The doubling is real, but leaving the portrait out didn't remove it on either seed, so the
claim's rule fails its own test. What the portrait did do, on both seeds, is hold her face:
without it segment 2 drifted 0.07 to 0.14 further from the portrait. That agrees with the
standing advice in `references/chains.md` (reference the subject picture in each segment on
the reference chains). The follower appears already in segment 1, a single Ref2VA turbo shot
of one person walking toward a backward-tracking camera from a three-quarter portrait, which
is the same family of failure as #1094's two astronauts; it wasn't probed further here.

## Probe 5: sound lines

The scene is a woman reading at a kitchen table at night with rain on the window. All three
arms share the picture prompt; `overall_soundscape` is:

- **no hum**: `Rain pattering steadily against the dark window, a paperback page turning now and then, a ceramic mug set down softly on the wooden table once.`
- **hum**: the same, prefixed with `The steady low hum of the refrigerator and the buzz of the fluorescent strip light overhead run under everything.`

Both end with `non_diegetic_music:` / `None.`; the **no music line** arm is the no-hum prompt
with the `non_diegetic_music` section left out.

Measures, per clip: RMS; the 10th-percentile (steady floor) and median power per octave band,
from a 4096-point STFT; spectral flatness over 80 Hz-4 kHz (near 0 for tones and voices, near
0.5 for noise); and narrowband peaks 12 dB over their neighbourhood.

| Job | Arm, seed | RMS dBFS | Floor 31 / 63 / 125 / 250 Hz octaves, dBFS | Flatness | Tonal peaks |
| --- | --- | --- | --- | --- | --- |
| 313fe586f422 | hum, 1001 | -35.1 | -76.2 / -73.2 / -75.4 / -73.6 | 0.391 | none |
| 412db5c1ed0e | hum, 2002 | -45.5 | -90.3 / -87.0 / -85.8 / -81.2 | 0.413 | none |
| 133cae62bfbd | no hum, 1001 | -39.9 | -90.4 / -82.5 / -81.4 / -75.6 | 0.447 | none |
| c1b9be136407 | no hum, 2002 | -42.5 | -100.1 / -91.5 / -88.1 / -81.8 | 0.448 | none |
| 3e449fbebbca | no music line, 1001 | -37.7 | -83.4 / -76.0 / -78.5 / -74.4 | 0.307 | none |
| e0f10599f3d4 | no music line, 2002 | -39.2 | -94.8 / -87.8 / -84.7 / -80.9 | 0.385 | none |

- **5a.** Naming the hum raised the steady low floor on both seeds, by about 14 and 10 dB in
  the 31 Hz octave and 9 and 4.5 dB at 63 Hz, against the no-hum arm. Leaving the music line
  out moved the same bands by 5-7 dB on its own, so seed 2002's 63-125 Hz change is inside
  arm-to-arm variation and only the 31 Hz octave clears it on both seeds. There was no tonal
  hum (no 50/60 Hz harmonic series, no persistent peak above 80 Hz), and the raised floor sits
  30-45 dB under each clip's RMS. A low rumble is measurable; a drone that a listener notices
  isn't shown. **Listening item for Don.**
- **5b.** No music on either seed without the music line: all six clips are noise-like, with
  no tonal peaks. The no-music-line clips are a little less flat (0.31 and 0.39 against 0.45),
  which a single strong rain transient can do. **Listening item for Don** to confirm.

## Method notes and deviations

- **Claim 4's seam is built by hand.** `templates/minimax/chain-video-continuity` passes the
  same `references` to every segment, so it can't drop the portrait after segment 1 without an
  engine change. `t26-continued.json` runs the template's ref2va step for segment 1, takes its
  last 48 frames with their audio (`trim_video`, the template's `carry_frames`) as a
  `MiniMaxH3VideoReference`, exactly as the chain's `last_segment` carry does, and renders
  segment 2 twice at the same seed: with `[portrait, tail]` (what the template does) and with
  `[tail]` alone. No audio reference: the claim is about the picture. The template's own
  prompt doesn't name `<Video 1>`; these prompts do.
- **Only the Ref2VA-trained LoRA was run for claim 4.** The claim says doubling is a property
  of FL2VA-family checkpoints and that Ref2VA-trained ones don't double. dw never puts an FL2VA
  LoRA on ref2va (`validate_workflow` refuses it), so the case dw can reach is the one tested.
- **Claim 1 and 5 ran the T2VA template's turbo defaults**, as #1193 asks, though the skill
  drops the turbo LoRA for dialogue shots: the claims are about what is drawn and about sound
  lines, not voice quality.
- **One H3 job ran at a time across the two cards.** The bench server's host-RAM admission kept the second H3
  job waiting; the two-face checks ran beside them.
- **Records.** `bench/records/t26-prep.json`, `t26-claims.json` and `t26-facecheck.json` are the
  driver's records. The portrait and face-check jobs carry `error: "... wrote no video"`: the
  driver looks for a video output to name, and those jobs write an image or JSON. Each job
  itself succeeded.
- **Scene label.** The arms ran with `"scene": "t26"`, which the job notes and records keep. The
  driver reads a scene only under `--measure`, which none of these used; the committed arms
  name `s3-dialogue-walk` (the walker scene) so the bench tests, which pin the six scenes,
  load them.
