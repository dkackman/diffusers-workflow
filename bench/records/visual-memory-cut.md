# Visual memory across a cut: MiniMax-H3 ref2va probe (#1398 stage A, #1401)

**PRELIMINARY until Don views the arms.** Also **INCOMPLETE**: arms B1 and S1
(whole-clip video reference) do not fit the 24 GB card at the plan's
1344x768 (see *B1: out of memory*), so the plan went back to Don as v2. The
frame reads below are an agent's (claude-opus-5-5, anthropic), from 12-frame
contact sheets and face crops, not a viewing.

## Setup

- lem, RTX 3090 24 GB, develop @ a2ce7201, workspace `probe-1398`, 2026-11-17 (UTC).
- Workflows: `bench/workflows/visual-memory-shot.json` (portrait plus an
  optional whole clip, `memory`) and `bench/workflows/visual-memory-window.json`
  (a `trim_video` window of an earlier output as the video reference).
  Both use `audio: null` on the video reference.
- Shared: ref2va, turbo LoRA (`minimax_h3_ref2v_turbo_8step_v1.0_768p`), int4
  SDNQ, sparse attention, 9 steps (8 passes), video_shift 12, audio_shift 3,
  124 frames, 1344x768, seed 1398, portrait `asset:cast/estela.jpg`.
- Scene markers: a red glass lamp, a tall wooden bookshelf, a tall window
  with blue evening sky, and a green coat. The prompts are in the run notes
  (`get_job`); B's prompt names the lamp, window and shelf but **not** their
  colours, and says "in her coat" without a colour. A reverse that keeps them
  has to carry them from a reference.

## Runs

| Arm | What | Job | Run id | Wall | Denoise (pass) |
|---|---|---|---|---|---|
| E | establishing wide | `4ab13c8b9c64` | `20261117-042440-80e9865c` | 394 s | 324.9 s (40.5)\* |
| A | silent medium | `534601548d6d` | `20261117-043156-208e433a` | 359 s | 285.8 s (35.3) |
| A′ | A + line "I knew you would come back tonight." | `309d8397f56f` | `20261117-043801-79f89a16` | 349 s | 285.0 s |
| B0 | reverse, portrait only | `fdba91e57259` | `20261117-044359-71e32755` | 343 s | 284.8 s (~35.6) |
| B1 | B0 + A whole (124 frames) | `e6ecb7df55ec` | `20261117-044953-62079bd3` | 395.7 s | **OOM**, no output |
| B2 | B0 + 5-frame window of E (68-72) | `95b94eb50ea0` | `20261117-045832-c01e7f09` | 2.6 s | **refused**, see below |
| B2 | B0 + 22-frame window of E (51-72), the plan's fallback | `4798c34efb07` | `20261117-045847-4a6b003a` | 723 s | 645.3 s (48.0; first pass 309.1)† |
| S1 | B + line + A′ whole | not run | | | same shape as B1 |
| S2 | B + line "Then sit down, and tell me everything." + last 22 frames of A′ (102-123) | `bd047339cf78` | `20261117-051107-8d656e60` | 465 s | 387.5 s (48.1) |

\* E ran first, on cuda:1, with a cold load (4.6 s) and its own warm-up.
† The server flagged B2's first pass (309 s against 48 s) as a one-off warm-up of unknown cause.

### B2 at 5 frames: refused (verbatim)

> Workflow execution error: A reference video is read at 2 fps and its sampled frames are merged in groups of 2, so it must run at least 13 frames at 24 fps (0.54 seconds), got 5.

The minimum is 13 frames, so the plan's 22-frame fallback was used for B2 and,
by the same rule, for S2 (the plan's "last 5 frames" can't run either).

### B1: out of memory (verbatim)

> Workflow execution error: CUDA out of memory. Tried to allocate 4.90 GiB. GPU 0 has a total capacity of 23.56 GiB of which 4.47 GiB is free. Process 1074355 has 260.00 MiB memory in use. Including non-PyTorch memory, this process has 18.82 GiB memory in use. Of the allocated memory 18.01 GiB is allocated by PyTorch, and 188.21 MiB is reserved by PyTorch but unallocated.

Text encode took 0.3 s, VAE encode of the reference reached 21.6 s and denoise
began at 85.5 s. No denoise pass completed, and the OOM landed at about 394 s.
Earlier whole-clip video-reference runs (`docs/proposals/complete/chain-follow-video-complete.md`)
were at 960x544. A whole 124-frame reference at 1344x768 on top of a
124-frame 1344x768 generation does not fit this card with this offload. S1 is
the same shape (a whole 124-frame clip) and was not run, to save the ~7 min
it would take to reproduce the same OOM.

## Cost against the 3.4x claim

The skill's "~3.4x" is for a whole-clip reference, which did not run, so this
probe does not test it. A 22-frame window costs:

- per pass, 48.0 s against B0's ~35.6 s: **1.35x**;
- wall, S2 465 s against B0 343 s: **1.36x**. B2's 723 s (2.1x) includes the
  309 s first-pass warm-up; its steady passes match S2's.

## Frame reads (12-frame sheets)

| Arm | Lamp | Shelf | Window | Coat | Framing |
|---|---|---|---|---|---|
| E | red, on the desk centre-left | yes, left and back walls | tall, blue, right | green | wide, static |
| A | red glass, frame left | yes, behind | blue, right edge | green | medium, stable |
| A′ | as A | as A | as A | as A | as A |
| B0 | **white/brass desk lamp** | dark, at the edges | **orange sunset** | **purple** | seated medium shot of the reverse |
| B1 | - (OOM) | | | | |
| B2 | **red**, frame left | yes, behind | **blue night**, behind her | **green** | seated medium shot of the reverse, stable |
| S2 | red | yes | blue, behind | green | see below |

- **B0 against B2.** Portrait only, the reverse loses three of four markers
  (lamp colour, window sky, coat colour). With a 22-frame window of the
  establishing wide, all four come back. B2 does not copy E's wide framing: it
  is a reverse-angle medium shot from frame 0.
- **S2 opens on a replay of the window.** S2's frame 0 is A′'s frame
  118-123 again (same frontal medium framing, same pose, same closed-mouth
  smile), held for about 1 s. At about 1.4 s it cuts **inside the clip** to a
  reverse (window and lamp in frame, her in three-quarter profile on the
  right). The reverse holds all four markers. A trailing window used as a
  reference is continued before the shot turns to the prompt. A cut built on
  S2 would need its first ~1.4 s trimmed. B2's mid-clip window of a wide did
  not do this.
- **Mouth copy (S2).** The mouth is closed in the replayed opening
  (frames 0-24), as it is at the end of A′ (frames 110-123). A′'s open mouth
  at frame 102 does not reappear, so **no mouth copy is visible on frames**.
  Whether S2's audio speaks its own line, and whether lips follow it after the
  internal cut, is for Don's viewing and listening. S1 did not run.

## Not measured

- `bench/cut_windows.py pair` (A-B0, A-B2, A′-S2; A-B1 and A′-S1 have no
  output). It needs the YuNet/SFace ONNX models and local copies of the mp4s.
  This session had neither and ran no download. The measure is owed with the
  re-planned arms.

## For Don to view

`probe-1398` workspace: E, A, A′, B0 (`visual-memory-shot` / `VisualMemoryShot`
runs above), B2 and S2 (`visual-memory-window`). Lem time used: about 51 min of
the ~70 budgeted.
