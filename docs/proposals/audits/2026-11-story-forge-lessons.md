# Story-forge lessons: record-only probes (#1123 stages A and B, #1126, #1127)

Probe date: 2026-11-01. Server `lem` (2x RTX 3090), develop @ 2ca6ece8.
Agent: claude-opus-5-5 (anthropic). Workspace `probe-1123`.
No engine or skill file was changed. This document only records results.
Stage B (#1127) added the sections for lessons 3, 4 and 6, probed the same
day on develop @ f1020d8f by the same agent in the same workspace.

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

Stage B (L3, L4, L6) judged the same way. L3 and L6 are LTX
`templates/ltx2/text-to-video` at 121 frames, plus an H3
`templates/minimax/dialogue-short` cell for L3. Contact was judged on frames at
1, 2, 3, 4 s and f120 (f123 on H3's 124 frames), with hand crops where hands
met. Impacts were judged at 1, 2, 3, 4 s and f120, plus 2.5, 3, 3.5 and 4.5 s
around the strike. L4 is stills: each render's whole frame at 384 px, read for
any legible text.

## Verdicts

| # | Lesson | Verdict | Naive renders failing | Remedy renders clean |
| --- | --- | --- | --- | --- |
| L1 | LTX-2.5 renders haze over after 2–3.5 s; keep clips short (89 frames). | **not reproduced (on our pipeline at this N)** | 0 of 10 (6 t2v at 241, 2 i2v at 241, 2 i2v at 401) | 8 of 8 at 89 frames, but there was nothing to remove |
| L2 | i2v invents camera moves unless the prompt pins the camera; "camera remains static" alone is not enough, add "locked-off on a tripod, no pan, no zoom". | **not reproduced (on our pipeline at this N)** | 0 of 6 clear moves; 2 of 6 ambiguous (S3, see below) | 6 of 6 "static", 6 of 6 "locked-off" show no move |
| L3 | Bodies in contact (hugs, handshakes, a hand on a shoulder) fuse or grow limbs; stage the moment just before contact. | **not reproduced (on our pipeline at this N)** | LTX 2 of 6 (both handshakes, mild); H3 0 of 2 (1 brief ambiguous) | 6 of 6 clean, but 0 of 6 followed the restaging: every remedy render went on into contact |
| L4 | An ALL-CAPS word in an image prompt renders as written text; write it in lowercase. | **not reproduced (on our pipeline at this N)** | 0 of 6 z-image, 0 of 6 Klein | 12 of 12, but there was nothing to remove |
| L6 | Impacts (a strike, a slam) break physics; show the effort, not the impact. | **not reproduced (on our pipeline at this N)** | 1 of 4 (axe seed 1); both door renders never showed the slam | 3 of 4 clean, 1 ambiguous (door seed 2 closes softly) |

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

## L3: bodies in contact

Prompts `probe-1123/l3-{hug,handshake,shoulder}` (naive: the contact shown)
and `probe-1123/l3-{hug,handshake,shoulder}-before` (remedy: the same two
people staged in the moment just before contact). All say "the camera remains
static". LTX t2v, 121 frames.

| Cell | Arm | Seed 1 | Seed 2 | Frame times inspected | Verdict | Observation |
| --- | --- | --- | --- | --- | --- | --- |
| hug | naive | 2c8e2fd11a64 | a0bf15587a17 | 1, 2, 3, 4 s, f120; hand crop at 1, 3 s | clean | Clean hugs. Hands on backs are plausible, five fingers, no fusion. |
| hug-before | remedy | 2b5294666c7b | 8b1b00b412b0 | 1, 2, 3, 4 s, f120 | clean, restaging not followed | Both go on into a full hug by 4 s and hold it to f120. Clean, but in contact. |
| handshake | naive | 1e54ba7c709e | 33537ba0f0b5 | 1, 2, 3, 4 s, f120; hand crop at 1, 3 s | **fail (mild)** | Seed 1: the clasp at 3 s is a smeared, fused blob with no finger separation. Seed 2: fused or fanned fingers (an apparent extra) at first contact at 1 s, a plausible clasp by 3 s. Motion blur may contribute to both. |
| handshake-before | remedy | 35efb75713d0 | 030338233ca5 | 1, 2, 3, 4 s, f120; hand crop at 3 s, f120 (seed 1) | clean, restaging not followed | Both shake hands by about 2–3 s and hold to f120. Seed 1's clasp is plausible in the crop. |
| shoulder | naive | 486626742b05 | b50872edf631 | 1, 2, 3, 4 s, f120 | clean | Workshop, an old man and a girl. The hand settles on the shoulder by 3 s; an exchange of looks by f120. |
| shoulder-before | remedy | 66934ce13ed3 | 376441d1abca | 1, 2, 3, 4 s, f120 | clean, restaging not followed | Both rest the hand on the shoulder by 3 s. |

H3 cell, `templates/minimax/dialogue-short` with one shot (`hug`, 124
frames), the template's own drawn portraits, and the hug prompt stored as
`probe-1123/l3-h3-hug`. Naive arm only.

| Cell | Arm | Seed 1 | Seed 2 | Frame times inspected | Verdict | Observation |
| --- | --- | --- | --- | --- | --- | --- |
| h3-hug | naive | a793f12f9d8c | 134084669ce8 | 1, 2, 3, 4, 5 s; hug crop at 3, 5 s (seed 1), 2, 4 s (seed 2) | clean (seed 2 ambiguous at 2 s) | A station platform: the two step together and embrace by 2–3 s and hold to the end. Seed 1: hands on backs clean, rings and watch intact. Seed 2: at 2 s, as the arms close, four hands show at once with one motion-blurred over his shoulder, which is hard to assign to an arm. By 4 s it resolves into a clean embrace. Counted ambiguous, not a failure. |

Tally: the naive arm fails 2 of 6 on LTX (both handshakes, mild) and 0 of 2
on H3. That is under half, so the lesson is not reproduced. The remedy
cannot be credited either way: LTX did not follow "just before contact" in
any of 6 renders, and every remedy clip reached the contact it was staged to
avoid, cleanly. On this pipeline, if contact is to be avoided, the prompt
has to do more than stage the moment before it.

## L6: impacts

Prompts `probe-1123/l6-{axe,door}` (naive: the strike or slam) and
`probe-1123/l6-{axe,door}-effort` (remedy: the effort before the impact, for
example the axe raised overhead, or her gripping a door that stands wide open,
"shot holds before the door moves"). LTX t2v, 121 frames, "the camera remains
static".

| Cell | Arm | Seed 1 | Seed 2 | Frame times inspected | Verdict | Observation |
| --- | --- | --- | --- | --- | --- | --- |
| axe | naive | 6be90fe35d6b | 176f482ff752 | 1, 2, 3, 4 s, f120; crop at 2.5, 3, 3.5, 4.5 s | **fail (seed 1)** | Seed 1: a clean strike at 3 s and a half flying off right at 3.5 s, but by 4.5 s a whole log is back on the block with the axe in it, and pieces lie on the ground. The log re-forms or duplicates. Seed 2: the impact at 3 s is motion blur; by 4 s the block is empty with pieces on the ground. Plausible. |
| axe-effort | remedy | 90e028d7e9a4 | 1088ba878fa5 | 1, 3 s, f120 | clean | Axe raised overhead to f120, no strike, no artifacts. In seed 2 the axe head comes to rest on the top of the log at the end. |
| door | naive | 004112bab9e8 | 8d9afda0c95e | 1, 2, 3, 4 s, f120 | impact not depicted | The slam is never rendered. Seed 1: she presses on an already-closed door, with a reframe between 1 and 2 s. Seed 2: the door is near-shut at 1 s and she reaches for the handle at 3 s. No pass-through and no morph. The coat rack and picture are in the prompt, so they are not inventions. |
| door-effort | remedy | b658f4bd3d82 | f87c4fe72657 | 1, 3 s, f120 | seed 1 clean, seed 2 ambiguous | Seed 1: she holds the open door, with the street visible through it, to f120. Seed 2: the door is open at 1 s, but closed or near-closed by f120 with her gripping it. It moved against the prompt, but with no artifact. |

Tally: the naive arm fails 1 of 4 (axe seed 1), under half, so the lesson is
not reproduced. The door renders are not counted as failures: LTX never
showed the slam at all, so there was no impact to break. The remedy arm is
clean in 3 or 4 of 4.

## L4: ALL-CAPS words render as text

Prompts `probe-1123/l4-{proud,alone,hurry}-{caps,lower}`: a baker holding a
tray of croissants, "PROUD"/"proud" of them; a man alone in the rain at night,
"ALONE"/"alone"; a girl in a yellow raincoat running to a red door in a
"HURRY"/"hurry". The caps and lowercase prompts differ only in that one word's
case. Each still is judged on its whole image (`get_output_image`), which is
the "frame inspected". There are two families:
- `models/z-image` at its defaults (960x544);
- Klein (`black-forest-labs/flux-3-image-klein`, 1024x1024, 16 steps), run as
  `templates/flux3/text-to-image` with one change, the text encoder's
  placement (see Deviations).

### z-image

| Cell | Arm | Seed 1 | Seed 2 | Verdict | Observation |
| --- | --- | --- | --- | --- | --- |
| proud | naive (caps) | b8f4028b4aca | effa9b4d219c | no text | Baker smiling over a tray of croissants. No lettering anywhere: no sign, apron print or tray label. |
| proud | remedy (lower) | d4987b9329bc | 556194c9ed5b | no text | Near-identical compositions to the caps renders, seed for seed. |
| alone | naive (caps) | 06f277c06d19 | 64431b97bb07 | no text | Hooded man on a wet street at night. Shop fronts and signs stay blurred and illegible; "ALONE" appears nowhere. |
| alone | remedy (lower) | 2ecb218715f6 | e62c2c0baa8d | no text | Near-identical to the caps renders, seed for seed. |
| hurry | naive (caps) | 57fcba14cfcd | 137cc3206146 | no text | Girl running to a red door. No lettering on the door, wall or coat. |
| hurry | remedy (lower) | f47fcbf043b0 | 855955966ac3 | no text | Near-identical to the caps renders, seed for seed. |

Tally: the caps word appears as legible text in 0 of 6 caps renders and 0 of
6 lowercase ones, so the plan's "holds" test (at least half of the caps
renders, none of the lowercase) fails. Not reproduced on z-image. The case of
the word changed nearly nothing: each caps/lowercase pair at the same seed is
close to the same picture.

### Klein

Stage B's second session ran these after Don's `spend` ruling on #1127. Each
note reads `#1127 L4 cell klein-<word>-<caps|lower> seed <s>`.

| Cell | Arm | Seed 1 | Seed 2 | Verdict | Observation |
| --- | --- | --- | --- | --- | --- |
| proud | naive (caps) | 747a40ecd26e | 8dc1157be57f | no text | Grinning baker holding a tray of croissants in a sunlit bakery. No lettering on the apron, tray, shelves or wall. |
| proud | remedy (lower) | 37721ee6bee7 | 708bec1cc91a | no text | The same scene, with more croissants on the tray. No lettering. |
| alone | naive (caps) | 887c4be1f269 | baac48e8fcc8 | no text | Man in a wet coat, head bowed, alone on a rainy crosswalk at night. Shop fascias are blurred. One small illegible sign at the left edge (seed 2), not "ALONE". |
| alone | remedy (lower) | 408a20fe4a00 | 62422200260e | no text | The same scene. Fascias are illegible and "alone" appears nowhere. |
| hurry | naive (caps) | 946866bee92e | 3faf33a3c088 | no text | Girl in a yellow raincoat and boots running up a cobbled lane to a red door. No lettering on the door, wall or coat. |
| hurry | remedy (lower) | 898397bfb410 | 640ee4d1af35 | no text | The same scene against a brick wall. No lettering. |

Tally: the caps word appears as legible text in 0 of 6 Klein caps renders
and 0 of 6 lowercase ones, so the "holds" test fails on Klein as well. **L4
is not reproduced on either family (on our pipeline at this N).** As on
z-image, each caps/lowercase pair at the same seed is close to the same
picture. On Klein, the case of the word changes nothing visible.

## Deviations from plan v2

- **More L1 renders than the plan's minimum.** Each t2v prompt ran at 241
  (naive), 121 (control) and 89 (remedy) frames, and the i2v still at 241,
  401 and 89: 24 L1 renders in all.
- **The i2v 121-frame control is borrowed** from L2 s1-static (same still,
  static prompt, 121 frames) rather than run as a separate L1 cell; its note
  names the L2 cell.
- **Stills from z-image first, Klein later.** The Klein route needed a
  40 GB model download, and `models/z-image` was already on lem. The Klein
  cells followed once Don ruled the download in (#1127).
- **Asset names are flat** (`probe-1123-s1-bench.jpg`), not in a folder.
- **The L1 prompts all say "the camera remains static"**, to keep camera
  motion out of the haze read.
- **Fewer frames on some cells**: the dim and action 121-frame controls at
  three times (1, 3 s, f120); the L2 remedy cells at two (0 s, f120), at
  160 px. The 401-frame renders added a 12 s frame.
- **Judged by eye**, since `assess_output` has no image or motion probe.

Stage B (#1127):
- **More L3 LTX renders than the plan's 8**: three contact cells (hug,
  handshake, hand on shoulder), each in both arms × 2 seeds, 12 in all.
- **The H3 cell has no remedy arm**: one `hug` shot of 124 frames per seed,
  with the template's own drawn portraits. The shot prompt was written inline
  in the shots item; the same text is stored as `probe-1123/l3-h3-hug`.
- **The Klein L4 cells ran as an inline copy of
  `templates/flux3/text-to-image`.** Model, size, steps, dtype and seed
  handling are unchanged. The one difference is that the text encoder is
  leaf-level group-offloaded (`"group_offload": {"offload_type":
  "leaf_level"}`), not placed on `cuda`. As written, the template OOMs while
  loading on an empty 24 GB 3090 against the current `main` (job
  49459dc14c17), whatever its description says about "about 19 GB". The
  modular pipeline also has no `offload: "model"` (job 5ec3b2dcb9f5). Neither
  failed job counts as a probe. With the offload, the text encode takes about
  8–12 min per still, against about 9 s of denoise.
- **No old Klein snapshot to delete.** Don's ruling asked for the older,
  unreferenced 31.8 GB snapshot to be removed. It turned out to be the
  partial download of the same `main` revision, and the finished download
  completed it in place, so there was nothing separate to remove. Disk free
  on lem afterwards was about 262 GB.
- **The remedy restagings were not followed by LTX** in L3 (0 of 6) and only
  partly in L6 (the door seed 2 closed). The tables record what rendered, not
  what was asked.
- **The L6 door impact was never depicted** in either naive render, so that
  cell tests nothing about impact physics.
- **Fewer frames on some remedy cells** (1, 3 s, f120 on L6 remedies); hand
  and impact crops on the cells where contact or a strike happened.
- **Every L3 and L6 prompt says "the camera remains static"**, to keep camera
  motion out of the read.

## Infrastructure notes (not lesson findings)

- The first denoise pass of a job on a cold worker takes about 50–64 s and
  emits a warm-up warning.
- A 401-frame i2v job spent about 330 s in decode and save.
- The quiet-audio warnings on these jobs are irrelevant to L1 and L2: the
  probes judge the picture only.
- Stage B: the first two LTX decodes on a fresh worker took 93 s and 125 s
  (the warm-decode warning); later decodes took about 2 s, and warm LTX
  121-frame jobs about 35–50 s each.
- The first z-image job loaded in 17.4 s; warm z-image jobs ran about 14 s.
- An H3 single-shot `dialogue-short` job ran about 6–9 min on one 3090; the
  first denoise pass on a cold H3 load took 206 s, against about 30 s warm.
- The clipping and quiet-audio warnings on L6 jobs are irrelevant: the probes
  judge the picture only. `dialogue-short` warns that `audio_bleed_ms` has no
  effect with one shot, which is expected.
