# MiniMax-H3 AISkills claims: record-only probes (#1059 stage 2, #1071)

Probe date: 2026-10-30 (lem's clock reads 2026-10-31 in the run ids). Server `lem`, develop @ 7735379d,
dw 0.17.0-alpha.1, diffusers 0.42.0.dev0, torch 2.14.1+cu130, two RTX 3090s.
Agent: claude-opus-5-5 (anthropic). Workspace `qa-h3-aiskills-probes`.
No engine, skill or template file was changed. This document only records results.

The claims come from einhorn13/AISkills, as listed on #1059. Plan v1 there keeps three of them
(2, 3 and 5). Probes 1 and 4 are deferred, and 6 and 7 are dropped. Neither the plan nor this
document copies anything from that repository.

## Configuration

Every probe ran at 960x544 and 124 frames, int4 sdnq (VAE int8), 24 fps.

| Probe | Workflow | Pipeline | LoRA (`lightx2v/Minimax-h3-Turbo`) | Steps | Shift (video/audio) |
| --- | --- | --- | --- | --- | --- |
| P2 | `templates/minimax/voice-timbre-reference`, as shipped | Ref2VA, one audio reference | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors`, pinned by the template (plan: runs as shipped, despite the `<d>` line) | 9 | 12 / 3 |
| P3 | `p3-style-reference`, hand-authored and saved in the workspace | Ref2VA, one image reference | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` | 9 | 12 / 3 |
| P5 | `templates/minimax/video-with-audio`, template defaults | T2VA | `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` (the template's default) | 5 | 6 / 3 |

P3's workflow copies the `reference-to-video` step's loading (sparse attention, tau 0.5, top-k
0.15, block-level group offload) and drops the audio reference. It has one
`MiniMaxH3ImageReference` from the variable `subject`.

Each arm ran on seeds 1001 and 2002. A claim is **held** only when both seeds of the broken arm
show the effect and neither seed of the followed arm does. It is **refuted** when the broken arm
doesn't show the effect, and **not settled** when the seeds disagree. A job that failed would be
**not run**. None did: all 12 probe jobs and 4 check-script jobs succeeded.

## Verdicts

| # | Claim | Verdict | Jobs | Observation |
| --- | --- | --- | --- | --- |
| P2 | A timbre-only audio reference replays its source words. | **refuted** | clip A: 679d1647474e (s1001), 23951087cc0e (s2002). Clip B: 1ae32a340b81 (s1001), 8b5321bf4db3 (s2002). check-script: 558af206ad68, e247ffc86d1b, 58d17f8beea2, 94115820bfa5 | On all four, Whisper heard only the new line: "Bring the copper kettle inside before the storm reaches the harbor." `findings` was empty (no `line_mismatch`), the similarity was 0.9091 (only the harbour/harbor spelling differed), and nothing landed in `unmatched`. No source word from either clip was transcribed. See *Probe 2*. |
| P3 | A style image pulls in its person and composition. | **refuted**, but the reference also carried no style, so the probe says little about style references in general | arm a (old man at window): f90a041af26a (s1001), 1b751acfc1d4 (s2002). Arm b (still life): b1469a34cea9 (s1001), db3b5b47095b (s2002) | On no seed of either arm did the old man, his cap or beard, a seated pose, the window, its curtains or the reference's framing appear. Frames 0–5 show the beach shot, so the reference didn't leak into frame 0. All four clips are photographic, not watercolour. Same-seed clips of the two arms are near-identical, so the reference had no visible effect under this phrasing. See *Probe 3*. |
| P5 | "Screen right" plus the anatomical side holds handedness; the anatomical side alone misses. | **refuted** | arm a ("in her left hand"): a6c07856c701 (s1001), cb19f9007653 (s2002). Arm b ("in her left hand, on screen right"): e96170407504 (s1001), 7ce8ed6f90ab (s2002) | All four clips are right on both counts. She lifts the mug with her left hand, which shows on screen right because she faces the camera, and holds it at chest height there to the end. Her right hand stays flat on the counter, on screen left. See *Probe 5*. |

## Probe 2: timbre-only audio reference

The template's `voice` step makes the reference clip with Bark (`v2/en_speaker_6`) from
`voice_line`. Its Ref2VA step is prompted to take only the timbre of `<Audio 1>`. The prompt was
the template's default, with only the `<d>` line replaced:

```
<d>[English] Bring the copper kettle inside before the storm reaches the harbour.</d>
```

It shares no content word with either source:

| Clip | `voice_line` (the source speech) |
| --- | --- |
| A | the template's default, "The way ahead is longer still, and we are going anyway." |
| B | "Seven green apples rolled under the kitchen table." |

`templates/check-script` ran on each output, with `lines` set to the new line. Every run used
`openai/whisper-large-v3-turbo` with a `whole_cut` pass.

| Job | Clip, seed | check-script | Transcript | Line span | Findings |
| --- | --- | --- | --- | --- | --- |
| 679d1647474e | A, 1001 | 558af206ad68 | "bring the copper kettle inside before the storm reaches the harbor." | 0.66–4.82 s | none |
| 23951087cc0e | A, 2002 | e247ffc86d1b | "Bring the copper kettle inside before the storm reaches the harbor." | 0.78–4.68 s | none |
| 1ae32a340b81 | B, 1001 | 58d17f8beea2 | "Bring the copper kettle inside before the storm reaches the harbor." | 0.12–4.92 s | none |
| 8b5321bf4db3 | B, 2002 | 94115820bfa5 | "Bring the copper kettle inside before the storm reaches the harbor." | 0.64–4.82 s | none |

The rules applied were `line_mismatch`, `transcript_loop`, `tag_spoken`, `line_clipped_at_end`
and `speech_not_transcribed`. None fired. The transcript holds no word from either source line
("way", "ahead", "longer", "going", "seven", "apples", "kitchen", "table"). The effect, source
words spoken or the new line not spoken, appears on neither clip and neither seed.

Not measured: whether the timbre itself matched the reference. That needs a listener or a speaker
embedding, and the claim under test was about words.

## Probe 3: a style image and its person

The reference images were made once each with `templates/text-to-image` at 512x512, as
watercolours:

| Asset | Job | Content |
| --- | --- | --- |
| `asset:p3-style-person.jpg` | a120cca0fdbc | An old bearded man in a flat cap, seated in profile at a curtained window, with a misty landscape outside. Brown and grey wash on visible paper. |
| `asset:p3-style-stilllife.jpg` | e941f3bb3d9c | Three pears, a reddish vase and a sheet of paper on a table, against a blue wash. No person. |

A first person image (5f7d24bb04d7) lost the window and was not used.

The prompt is the same for both arms; only `subject` changes. Here it is exactly:

```
subject_definitions:
<Picture 1> is a style reference only. <Picture 1> sets only the medium and palette. Its subject and framing are not used.
<Subject 1> is a young woman cyclist in a yellow jersey and black shorts on a red road bicycle.

summary:
[reference generation] The target video shows <Subject 1> riding a bicycle along a sunny beach at the waterline, rendered in the medium and palette of <Picture 1>.

retention_analysis:
<Picture 1>: reference - only the medium (watercolour on paper, paper grain, wet-on-wet blooms, soft pigment edges) and the palette are taken. Its subject and framing are not used; nothing it depicts appears in the video.

detailed_description:
The target video is in the watercolour medium and palette of <Picture 1>. [Shot 1] A wide side-on tracking shot follows <Subject 1> as she rides from left to right along the wet sand at the edge of the sea, pedalling steadily, her jersey bright against the pale sand. Small waves wash in behind her and slide back, leaving a glossy sheen on the sand that reflects the bicycle. The horizon sits in the upper third, a flat band of sea under an open sky with a few soft clouds. The camera tracks alongside her at a matching medium speed across the whole shot.

overall_soundscape:
Open beach ambience: waves breaking softly and hissing back over the sand, a steady sea breeze across the microphone, the tyres' wet crackle on packed sand and the light tick of the freewheel, a gull calling once in the distance.

non_diegetic_music:
A light, bright acoustic guitar figure, mixed well beneath the ambience, steady through the whole clip.
```

Each output was read with `get_output_frames(count=12)` and frames 0–5.

| Job | Arm, seed | Person, pose or window? | Style carried? | Frame-0 leak? |
| --- | --- | --- | --- | --- |
| f90a041af26a | a, 1001 | no. One cyclist in yellow and black on a red bike, side-on, riding left to right along the waterline, with no window and no frame | no. Photographic, saturated blue sky and sea | no. Frames 0–5 are the beach shot |
| 1b751acfc1d4 | a, 2002 | no. Same subject and framing, with puffy clouds | no. Photographic | no |
| b1469a34cea9 | b, 1001 | no person beyond the cyclist, no pears or vase | no. Photographic | no |
| db3b5b47095b | b, 2002 | no | no. Photographic | no |

Arm a shows no sign of the claim: neither seed brings in the old man, his pose or the window
framing. Under the verdict rule that is **refuted**. But the followed arm didn't carry the medium
either, and the clips with the same seed are near-identical across arms. The image reference was
effectively ignored, which is the risk the plan names: "a role phrasing that H3 ignores entirely
would test the phrasing, not the claim." So this result says that a reference prompted
style-only, in these words, didn't leak its person on dw. It doesn't say that a style
reference H3 does follow keeps its person out. A second probe would need a phrasing that
carries the medium (for instance, a reference that isn't marked "not used" for everything except
medium) before the leak question can be asked of it.

## Probe 5: handedness

The `video-with-audio` defaults ran with this prompt (arm a):

```
summary:
[text generation] The target video shows a woman facing the camera in a kitchen who picks up a red mug and raises it.

detailed_description:
The target video is in realistic photographic style. [Shot 1] A static medium shot frames a woman in her thirties with short dark hair and a plain grey sweater, standing centred behind a white kitchen counter and facing the camera squarely. A single red mug sits on the counter directly in front of her, centred. Soft daylight falls from a window out of frame. She looks at the lens, then picks up the red mug in her left hand and raises it to chest height, holding it there steadily and smiling for the rest of the shot. The camera does not move.

overall_soundscape:
A quiet kitchen room tone, a soft ceramic clink as the mug leaves the counter, a faint fridge hum.

non_diegetic_music:
None.
```

Arm b changes only "in her left hand and raises" to "in her left hand, on screen right, and
raises". She faces the camera, so her left hand is on screen right, and the two arms ask for the
same picture.

Each clip was read with `get_output_frames(count=8)`, and then frames 35/53, 70/88 and 123,
around and after the raise.

| Job | Arm, seed | Screen side of the mug | Hand holding it | Other hand |
| --- | --- | --- | --- | --- |
| a6c07856c701 | a, 1001 | right (from about 2.9 s to the end) | her left | right hand flat on the counter, screen left |
| cb19f9007653 | a, 2002 | right (from about 3.7 s to the end) | her left | right hand flat on the counter, screen left |
| e96170407504 | b, 1001 | right | her left | right hand flat on the counter, screen left |
| 7ce8ed6f90ab | b, 2002 | right | her left | right hand flat on the counter, screen left |

The mug starts centred and moves to screen right as it is raised on every seed of both arms, so
the anatomical side alone was enough here. Treat this as one easy case: a single person facing
the camera, one prop, a static shot. It doesn't cover a person in profile, two people, or a
moving camera.

Side finding: the four P5 clips decode at -71 to -80 dBFS mean, near-silent. The engine flagged
each one, so the quiet soundscape was barely rendered. This doesn't bear on the claim.

## Method notes and deviations

- **Reference images from `templates/text-to-image`, not FLUX.** The FLUX 3 text-to-image
  template needed a 40.3 GB download on lem. `templates/text-to-image` was cached and makes a
  512px SD-class image, which was enough for a stylised picture. Both images are watercolour
  and carry the distinguishing content each arm needs.
- **`<Picture 1>`, not `<Image 1>`.** The plan's wording, "<Image 1> sets only the medium and
  palette. Its subject and framing are not used", was written with H3's image-reference token,
  `<Picture 1>`. The sentence is otherwise as the plan states it.
- **P2's second clip came from the template's own `voice` step.** That step is a
  `generate_speech` step, the plan's "or similar". `voice_line` was overridden with new words
  and kept the same Bark preset, so the second clip differs in words and not in voice.
- **P5 ran `video-with-audio`'s defaults:** the 4-step v1.2 fl2v turbo LoRA, 5 steps and shift
  6. The plan says "templates run their defaults". This differs from the 2026-10 dialogue
  probes' 8-step LoRA, 9 steps and shift 12.
- **Measures that read no audio.** No agent can listen. P2 is judged by Whisper transcripts
  through `check-script`, and P3 and P5 by frames only.
