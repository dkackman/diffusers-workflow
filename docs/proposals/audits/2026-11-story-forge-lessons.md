# Story-forge lessons: record-only probes (#1123 stage A, #1126)

Probe date: 2026-11-01. Server `lem` (2x RTX 3090), develop @ 2ca6ece8.
Agent: claude-opus-5-5 (anthropic). Workspace `probe-1123`.
No engine or skill file was changed. This document only records results.
Stage B (#1123) adds the sections for lessons 3, 4 and 6.

Every render ran on the LTX-2.5 catalog templates as they stand on develop:
`templates/ltx2/text-to-video` for t2v and `templates/ltx2/image-to-video` for
i2v, at 24 fps and the templates' default size, steps and offload. Each run's
`note` names its lesson, cell, seed and arm (`naive`, `control` or `remedy`),
for example `#1126 L1 cell t2v-day-241 seed 1 (naive: 241 frames)`.

**N = 2.** Each cell ran on seeds 1 and 2, and nothing else varied within a
cell. Verdicts hold at that N only.

**Verdict rule** (plan v2): a lesson **holds** when its failure shows in at
least half of the naive renders and its remedy removes it in at least half of
the remedy renders. Otherwise it is **not reproduced (on our pipeline at this
N)**.

**How it was judged.** `assess_output` measures audio, seams and A/V sync; it
has no probe for haze, contrast loss or camera motion. Every verdict below is
by eye, from `get_output_frames` stills at the times each row lists
(`fN` is frame N; the last frame of a 121-frame clip is f120). Haze was judged
as loss of contrast, saturation or black level against the same render's 1 s
frame. Camera motion was judged by registration of fixed background
features (horizon, window frame, shelving, bench) across the inspected frames.

## Verdicts

| # | Lesson | Verdict | Naive renders failing | Remedy renders clean |
| --- | --- | --- | --- | --- |
| L1 | LTX-2.5 renders haze over after 2–3.5 s; keep clips short (89 frames). | **not reproduced (on our pipeline at this N)** | 0 of 10 (6 t2v at 241, 2 i2v at 241, 2 i2v at 401) | 8 of 8 at 89 frames, but there was nothing to remove |
| L2 | i2v invents camera moves unless the prompt pins the camera; "camera remains static" alone is not enough, add "locked-off on a tripod, no pan, no zoom". | **not reproduced (on our pipeline at this N)** | 0 of 6 clear moves; 2 of 6 ambiguous (S3, see below) | 6 of 6 "static", 6 of 6 "locked-off" show no move |

L1's two questions:
- **(i) Does haze set in at a fixed time from the start?** No. No render, at
  89, 121, 241 or 401 frames, lost contrast or saturation at 2, 3 or 3.5 s,
  or at any later time inspected (5, 8, 12 s). Compare the 89-frame remedies
  (07cb2f669c68, be9d8afa66fa, a7775b2ff01a, bd676e2b1ccc, f39f603399a3,
  4e587078c06e, 40b83c33b23a, 5413b87108de at 2, 3, 3.5 s) with the 241-frame
  renders at the same times (a843d44d7963, 5eed612e3cb0, 22da8679d29b,
  1bdae5a5ab37, 3d897e7528ed, 7922a2310b26, 390102b12db4, 1f042b8c8b31) and
  the 401-frame renders (62d74005a35c, c6d94bb1edac at 3, 3.5 s): the same
  look at the same time, whatever the length.
- **(ii) Does the tail of any render degrade, whatever its length?** No. The
  last frame of every render (f88, f120, f240, f400) is as sharp and
  saturated as its 1 s frame, including both 401-frame i2v renders (f400 at
  16.7 s).

Incidental, for L2's reading: one naive L1 render moved its camera despite
"camera remains static" in the prompt (1f042b8c8b31, i2v-241 seed 2: a slow
push/arc from about 5 s), and 62d74005a35c (i2v-401 seed 1) drifts left by
12–16.7 s. Both are past the 5 s of an L2 cell. At 121 frames no render moved.

## L1: haze after 2–3.5 s

Prompts (prompt library, `probe-1123/`): `l1-day` (daylit street market),
`l1-dim` (low-key reading lamp interior), `l1-action` (boxing gym, dusty light
shafts), `l1-i2v-bench` (the S1 still's man on a park bench). All four say
"the camera remains static", so camera motion is not L1's variable.

The i2v still S1 is `asset:probe-1123-s1-bench.jpg` (z-image, seed 1, job
836e69c60940).

### t2v

| Cell | Arm | Frames | Seed 1 | Seed 2 | Frame times inspected | Verdict | Observation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t2v-day-241 | naive | 241 | a843d44d7963 | 5eed612e3cb0 | 1, 2, 3, 3.5, 5, 8 s, f240 | no haze | Contrast and colour steady to f240 on both. Seed 1's woman walks off left late; a slight reframe at 8–10 s. |
| t2v-dim-241 | naive | 241 | 22da8679d29b | 1bdae5a5ab37 | 1, 2, 3, 3.5, 5, 8 s, f240 | no haze | Blacks stay deep and the lamp highlight crisp to f240. |
| t2v-action-241 | naive | 241 | 3d897e7528ed | 7922a2310b26 | 1, 2, 3, 3.5, 5, 8 s, f240 | no haze | The dust shafts are prompted atmosphere and hold the same density at 1 s and f240. The blur on punches is per-frame motion blur, not a fade. |
| t2v-day-121 | control | 121 | d0b681f6d0a1 | 4ff547d7b280 | 1, 2, 3, 3.5 s, f120 | no haze | Steady to f120. |
| t2v-dim-121 | control | 121 | 9e47ba3de55f | 8e548bf95e8e | 1, 3 s, f120 | no haze | Low-key exposure constant. |
| t2v-action-121 | control | 121 | 92f361532f52 | d3a51b66db99 | 1, 3 s, f120 | no haze | Dust density constant; punch motion blur only. |
| t2v-day-89 | remedy | 89 | 07cb2f669c68 | be9d8afa66fa | 1, 2, 3, 3.5 s, f88 | no haze | Steady to f88. |
| t2v-dim-89 | remedy | 89 | a7775b2ff01a | bd676e2b1ccc | 1, 2, 3, 3.5 s, f88 | no haze | Steady to f88. |
| t2v-action-89 | remedy | 89 | f39f603399a3 | 4e587078c06e | 1, 2, 3, 3.5 s, f88 | no haze | Motion blur on punches at 1–2 s (seed 1). |

### i2v (still S1)

| Cell | Arm | Frames | Seed 1 | Seed 2 | Frame times inspected | Verdict | Observation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| i2v-bench-241 | naive | 241 | 390102b12db4 | 1f042b8c8b31 | 1, 2, 3, 3.5, 5, 8 s, f240 | no haze | Colour and foliage contrast steady to f240. Seed 1's camera is static; seed 2 pushes/arcs from about 5 s. |
| i2v-bench-401 | naive | 401 | 62d74005a35c | c6d94bb1edac | 1, 3, 3.5, 5, 8, 12 s, f400 | no haze | f400 as sharp and saturated as 1 s on both. Seed 1 drifts left by 12–16.7 s; seed 2 near-static. |
| i2v-bench-89 | remedy | 89 | 40b83c33b23a | 5413b87108de | 1, 2, 3, 3.5 s, f88 | no haze | Static. |
| (control) L2 s1-static | control | 121 | 8625f73f4beb | 31d1f51392c8 | 0 s, f120 | no haze | Borrowed from L2: the same still and a static-camera prompt at 121 frames. |

## L2: invented camera moves

Stills (z-image, seed 1): S1 `asset:probe-1123-s1-bench.jpg` (job
836e69c60940), S2 `asset:probe-1123-s2-kitchen.jpg` (bd8d69807e4c), S3
`asset:probe-1123-s3-lighthouse.jpg` (f3044c7d87ab). Every L2 render is i2v at
121 frames. Prompts `prompt:probe-1123/l2-s{1,2,3}-{none,static,locked}`: the
same scene sentence per still, then no camera sentence (`none`), "the camera
remains static" (`static`), or that plus "locked-off on a tripod, no pan, no
zoom" (`locked`).

| Cell | Arm | Seed 1 | Seed 2 | Frame times inspected | Verdict | Observation |
| --- | --- | --- | --- | --- | --- | --- |
| s1-none | naive | b37a61236086 | 303bc7dcdb40 | 0, 2.5 s, f120 | static | Bench and background registered; the man turns his head. |
| s1-static | remedy | 8625f73f4beb | 31d1f51392c8 | 0 s, f120 | static | Registered. |
| s1-locked | remedy | ba03832525f9 | 43ec4a08bc2d | 0 s, f120 | static | Registered. |
| s2-none | naive | 9650219d85a9 | 8e1147a306a7 | 0, 2.5 s, f120 | static | Window, shelves and plants registered; the woman pours or steps left inside a fixed frame. |
| s2-static | remedy | 807397eaacf3 | 696a1b7dbe65 | 0 s, f120 | static | Registered. |
| s2-locked | remedy | 2a62f1d3e46e | 6e7901458b80 | 0 s, f120 | static | Registered. |
| s3-none | naive | d3baf1e6ac33 | c07fa80ef144 | 0, 2.5 s, f120 | ambiguous | Horizon and lighthouse fixed, but the foreground grass recedes and (seed 2) the left rocks look larger by f120: wind sway or a very slight forward drift. Not a clear move. |
| s3-static | remedy | 96ca8ebd9196 | b56afa6a2875 | 0 s, f120 | static | Horizon and lighthouse fixed. The foreground grass changes the same way as in s3-none. |
| s3-locked | remedy | 2fb65dba5d13 | 1a89971de09e | 0 s, f120 | static | As s3-static, grass change included. |

The S3 grass change appears in all six S3 renders, with and without a camera
sentence. That points to the prompted wind in the grass rather than a camera
move, which is why s3-none is counted ambiguous and not as a failure. Even
counting both as failures, the naive arm fails 2 of 6, under half, and the
verdict stands.

## Deviations from plan v2

- **More L1 renders than the plan's minimum.** Each t2v prompt ran at 241
  (naive), 121 (control) and 89 (remedy) frames, and the i2v still at 241,
  401 and 89: 24 L1 renders in all.
- **The i2v 121-frame control is borrowed** from L2 s1-static (same still,
  static prompt, 121 frames) rather than run as a separate L1 cell; its note
  names the L2 cell.
- **Stills from z-image, not Klein.** The Klein route needed a 40 GB model
  download; `models/z-image` was already on lem.
- **Asset names are flat** (`probe-1123-s1-bench.jpg`), not in a folder.
- **The L1 prompts all say "the camera remains static"**, to keep camera
  motion out of the haze read.
- **Fewer frames on some cells**: the dim and action 121-frame controls at
  three times (1, 3 s, f120); the L2 remedy cells at two (0 s, f120), at
  160 px. The 401-frame renders added a 12 s frame.
- **Judged by eye**, since `assess_output` has no image or motion probe.

## Infrastructure notes (not lesson findings)

- The first denoise pass of a job on a cold worker takes about 50–64 s and
  emits a warm-up warning.
- A 401-frame i2v job spent about 330 s in decode and save.
- The quiet-audio warnings on these jobs are irrelevant to L1 and L2: the
  probes judge the picture only.
