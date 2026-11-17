# Visual memory across a cut: MiniMax-H3 ref2va probe (#1398 stage A, #1401)

**PRELIMINARY until Don views the arms.** The frame reads below are an
agent's (claude-opus-5-5, anthropic), from 12-frame contact sheets and face
crops, not a viewing or a listening. A whole-clip video reference (B1, S1)
does not fit the 24 GB card at 1344x768 (see *B1: out of memory*), so under
plan v2 the whole-clip arms ran at 960x544 against their own control, B0s.

## Setup

- lem, RTX 3090 24 GB, workspace `probe-1398`, 2026-11-17 (UTC). E to S2 ran
  on develop @ a2ce7201; B0s, B1s and S1s on develop @ 54c990c2 (no engine
  change between them for this path).
- Workflows: `bench/workflows/visual-memory-shot.json` (portrait plus an
  optional whole clip, `memory`) and `bench/workflows/visual-memory-window.json`
  (a `trim_video` window of an earlier output as the video reference).
  Both use `audio: null` on the video reference.
- Shared: ref2va, turbo LoRA (`minimax_h3_ref2v_turbo_8step_v1.0_768p`), int4
  SDNQ, sparse attention, 9 steps (8 passes), video_shift 12, audio_shift 3,
  124 frames (5.17 s), seed 1398, portrait `asset:cast/estela.jpg`. Size
  1344x768 unless the arm name ends in `s` (960x544).
- Scene markers: a red glass lamp, a tall wooden bookshelf, a tall window
  with blue evening sky, and a green coat. The prompts are in the run notes
  (`get_job`); B's prompt names the lamp, window and shelf but **not** their
  colours, and says "in her coat" without a colour. A reverse that keeps them
  has to carry them from a reference.

## Runs

| Arm | What | Size | Job | Run id | Wall | Denoise (pass) |
|---|---|---|---|---|---|---|
| E | establishing wide | 1344x768 | `4ab13c8b9c64` | `20261117-042440-80e9865c` | 394 s | 324.9 s (40.5)\* |
| A | silent medium | 1344x768 | `534601548d6d` | `20261117-043156-208e433a` | 359 s | 285.8 s (35.3) |
| A′ | A + line "I knew you would come back tonight." | 1344x768 | `309d8397f56f` | `20261117-043801-79f89a16` | 349 s | 285.0 s |
| B0 | reverse, portrait only | 1344x768 | `fdba91e57259` | `20261117-044359-71e32755` | 343 s | 284.8 s (~35.6) |
| B1 | B0 + A whole (124 frames) | 1344x768 | `e6ecb7df55ec` | `20261117-044953-62079bd3` | 395.7 s | **OOM**, no output |
| B2 | B0 + 5-frame window of E (68-72) | 1344x768 | `95b94eb50ea0` | `20261117-045832-c01e7f09` | 2.6 s | **refused**, see below |
| B2 | B0 + 22-frame window of E (51-72), the plan's fallback | 1344x768 | `4798c34efb07` | `20261117-045847-4a6b003a` | 723 s | 645.3 s (48.0; first pass 309.1)† |
| S1 | B + line + A′ whole | 1344x768 | not run | | | same shape as B1 |
| S2 | B + line "Then sit down, and tell me everything." + last 22 frames of A′ (102-123) | 1344x768 | `bd047339cf78` | `20261117-051107-8d656e60` | 465 s | 387.5 s (48.1) |
| B0s | B0 at 960x544, the control for B1s | 960x544 | `c868b5bf4372` | `20261117-132600-80625f4c` | 386 s | 326.8 s (22.0; first pass 173.0)‡ |
| B1s | B0s + A whole (124 frames) | 960x544 | `bc99717c18fd` | `20261117-133231-e8fe627d` | 1227.5 s | 1118.3 s (97.6; first pass 429.2)† |
| S1s | S2's prompt and line + A′ whole (124 frames) | 960x544 | `f0b6d61b8a84` | `20261117-135329-96799c68` | 890.9 s | 781.1 s (95.4; first pass 98.2) |

\* E ran first, on cuda:1, with a cold load (4.6 s) and its own warm-up.
The rest ran on cuda:0.
† The server flagged the first pass as a one-off warm-up of unknown cause.
‡ B0s loaded cold (5.2 s); its first pass carries that warm-up. B0s's audio
came back ambience-only (a server warning); it has no line, as B0 has none.

Reference encode (VAE) was 25.8 s for B0s (portrait only) and 85.5 s for
both whole-clip arms. Each output is
`<workflow>/<run id>/final/<WorkflowId>-shot.0-0.0.mp4`.

### B2 at 5 frames: refused (verbatim)

> Workflow execution error: A reference video is read at 2 fps and its sampled frames are merged in groups of 2, so it must run at least 13 frames at 24 fps (0.54 seconds), got 5.

The minimum is 13 frames, so the plan's 22-frame fallback was used for B2 and,
by the same rule, for S2 (the plan's "last 5 frames" can't run either).

### B1: out of memory (verbatim)

> Workflow execution error: CUDA out of memory. Tried to allocate 4.90 GiB. GPU 0 has a total capacity of 23.56 GiB of which 4.47 GiB is free. Process 1074355 has 260.00 MiB memory in use. Including non-PyTorch memory, this process has 18.82 GiB memory in use. Of the allocated memory 18.01 GiB is allocated by PyTorch, and 188.21 MiB is reserved by PyTorch but unallocated.

Text encode took 0.3 s, VAE encode of the reference reached 21.6 s and denoise
began at 85.5 s. No denoise pass completed, and the OOM landed at about 394 s.
A whole 124-frame reference at 1344x768 on top of a 124-frame 1344x768
generation does not fit this card with this offload. S1 at 1344x768 is the
same shape and was not run. At 960x544 both fit (B1s, S1s).

## Cost against the 3.4x claim

The skill's "~3.4x" is for a whole-clip reference. Measured at 960x544
against B0s:

- **per pass:** B1s 97.6 s and S1s 95.4 s against 22.0 s: **4.4x** and
  **4.3x**, above the claim;
- **wall:** S1s 891 s against 386 s: **2.3x**, below it, because text
  encode, decode and load don't scale. B1s 1228 s (3.2x) carries a 429 s
  first-pass warm-up; its steady passes match S1s's.

A 22-frame window, at 1344x768 against B0's ~35.6 s/pass:

- per pass, 48.0 s: **1.35x**;
- wall, S2 465 s against B0 343 s: **1.36x**. B2's 723 s (2.1x) includes its
  309 s first-pass warm-up.

## Frame reads (12-frame sheets)

| Arm | Lamp | Shelf | Window | Coat | Framing |
|---|---|---|---|---|---|
| E | red, on the desk centre-left | yes, left and back walls | tall, blue, right | green | wide, static |
| A | red glass, frame left | yes, behind | blue, right edge | green | medium, stable |
| A′ | as A | as A | as A | as A | as A |
| B0 | **white/brass desk lamp** | dark, at the edges | **orange sunset** | **purple** | seated medium shot of the reverse |
| B1 | - (OOM) | | | | |
| B2 | **red**, frame left | yes, behind | **blue night**, behind her | **green** | seated medium shot of the reverse, stable |
| S2 | red | yes | blue, behind | green | opens on a replay of A′, then a reverse (below) |
| B0s | **white/brass desk lamp** | dark, at the edges | **orange sunset** | **purple** | seated medium shot of the reverse |
| B1s | **red glass**, frame left | yes, right and behind | **tall, blue night**, left-centre | **purple** ✗ | pulled to A's composition (below) |
| S1s | **red**, left edge | yes, behind | **blue**, right, as in A′ | **purple** ✗ | close to A′'s, standing, window right |

- **B0 against B2 (1344x768).** Portrait only, the reverse loses three of
  four markers (lamp colour, window sky, coat colour). With a 22-frame window
  of the establishing wide, all four come back. B2 does not copy E's wide
  framing: it is a reverse-angle medium shot from frame 0.
- **B0s against B1s (960x544).** B0s loses the same three markers as B0, so
  the smaller size does not change the control. With A whole as reference,
  B1s gets back the lamp, the shelf and the blue window, but **not the coat**:
  it stays purple, as in B0s (A's is green). B1s is also pulled toward A's
  composition: she stands behind the desk with the lamp left and the shelf
  behind, close to A's angle, with the window mirrored to the left. That is
  not a clear reverse, which is the risk the plan named for a whole-clip
  reference.
- **S1s.** As B1s: lamp, shelf and window held, coat purple. Framing stays
  close to A′'s (standing, window right, shelf behind), not a reverse.

### Does a reference arm open on a replay of its reference?

| Arm | Reference | Replay at the opening |
|---|---|---|
| B2 | 22 frames from the middle of E | no: its own reverse from frame 0 |
| S2 | last 22 frames of A′ | **yes**: frame 0 is A′'s frames 118-123 again (same frontal medium, pose and closed-mouth smile), held about 1 s; at about 1.4 s it cuts inside the clip to a reverse (window and lamp in frame, her in three-quarter profile right) that holds all four markers. A cut built on S2 needs its first ~1.4 s trimmed |
| B1s | A whole | no: frame 0 is already its own shot (purple coat), though composed like A |
| S1s | A′ whole | no: frame 0 is its own shot, composed like A′ |

### Mouth copy

- **S2.** The mouth is closed in the replayed opening (frames 0-24), as at
  the end of A′ (frames 110-123). A′'s open mouth at frame 102 does not
  reappear, so no mouth copy is visible on frames.
- **S1s against A′.** On sampled frames S1s's mouth is open at about 0.9,
  1.4, 1.9, 3.2 and 4.2 s; A′'s at about 2.8, 3.2 and 4.2 s, closed early.
  S1s speaks from about 0.9 s, on its own timing, so there is no clear
  frame-level copy of A′'s mouth; 3.2 and 4.2 s coincide and could be either.
- Whether S1s's and S2's audio speak their own line, and whether the lips
  follow it, is for Don's viewing and listening.

## Sharpness and identity across the cut (`bench/cut_windows.py pair`)

Command, run locally on copies of the outputs fetched over HTTP:

    python -I bench/cut_windows.py <yunet.onnx> <sface.onnx> pair <A.mp4> <B.mp4>

with `opencv/face_detection_yunet` `face_detection_yunet_2023mar.onnx`
(revision 3cc26e7f) and `opencv/face_recognition_sface`
`face_recognition_sface_2021dec.onnx` (revision 3d708243). Every clip is
5.17 s with 21 face samples.

| Pair | A's last 1 s | B's first 1 s | Step | B reaches A's end level | SFace across |
|---|---|---|---|---|---|
| A–B0 | 376 | 199 | 0.53x (-47%) | never | 0.842 |
| A–B2 | 376 | 210 | 0.56x (-44%) | never | 0.699 |
| A–B0s | 376 | 514 | 1.37x (+37%) | at 0 s | 0.696 |
| A–B1s | 376 | 662 | 1.76x (+76%) | at 0 s | 0.581 |
| A′–S1s | 346 | 689 | 1.99x (+99%) | at 0 s | 0.803 |
| A′–S2 | 346 | 285 | 0.82x (-18%) | never | 0.910 |

Per-second face sharpness (median Laplacian variance, 0-1 s … 5-6 s):

| Clip | Size | Per second |
|---|---|---|
| A | 1344x768 | 376, 300, 310, 365, 396, 361 |
| A′ | 1344x768 | 396, 265, 295, 288, 345, 344 |
| B0 | 1344x768 | 199, 132, 116, 125, 150, 174 |
| B2 | 1344x768 | 210, 169, 219, 275, 276, 293 |
| S2 | 1344x768 | 285, 174, 168, 190, 180, 175 |
| B0s | 960x544 | 514, 341, 289, 328, 422, 418 |
| B1s | 960x544 | 662, 336, 330, 648, 491, 484 |
| S1s | 960x544 | 689, 461, 494, 532, 590, 592 |

**Caveat: sharpness is not comparable across sizes.** Laplacian variance
depends on resolution and on the face crop's size, so a 960x544 arm's step
against a 1344x768 A says little on its own; the B0s, B1s and S1s steps are
all "up" largely for that reason. Read the `s` arms against each other:
B1s against its control B0s is 662 against 514 in the first second (1.29x)
and higher in four of six seconds. S1s against S2 crosses sizes and carries
the same caveat. SFace (identity) is size-normalised by its aligned crop and
is comparable: B1s is the lowest across the cut (0.581, below B0s's 0.696),
and S2 the highest (0.910, from its replayed opening).

E was not paired: the plan pairs each B arm against A, and A′ against S1/S2.

## For Don to view

`probe-1398` workspace, all with notes naming the arm: E, A, A′, B0, B0s,
B1s, S1s (`visual-memory-shot`), B2 and S2 (`visual-memory-window`). The
questions for the viewing: whether B1s/S1s read as a reverse at all, the
coat in B1s/S1s, S2's ~1.4 s replay, and whether S1s and S2 speak their own
line with lips that follow it.

Lem time used: about 93 min (51 before v2, then 6.4 + 20.5 + 14.8) against
the ~86 approved, about 7 min over: a whole-clip pass cost 4.4x a control
pass, not the 3.4x the estimate assumed, and B1s had a 429 s warm-up.
