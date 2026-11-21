# Performer replacement: recast an existing multi-cut clip (#1194)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-17). #1194 began as an idea from a review of
[leonardofrito/Directors-Cut](https://github.com/leonardofrito/Directors-Cut)'s
Second Take (MIT, credited in `docs/ATTRIBUTIONS.md`). It was folded into the
long-form bench (#1199, Task 16), where the gate passed on one cut. Don unparked it on
2026-11-15 for its own plan. Plan v3, the plan comment on #1194, was approved at its
defaults that day and built as three stages: #1373, #1374 and #1375. This file records
what was built and measured.

## Verdict: build, with per-person tracks last

At the #1199 gate, a ref2va regional edit with the cut as `source_video` and
`hold_audio`, plus a per-frame `segment "person"` mask, scored 0.759 / 0.728 face cosine
to the portrait. A fresh ref2va render scores 0.683. Outside the mask held at 37.5 dB,
and the audio correlation was 0.9988 (`bench/records/performer-replacement.md`). The
plan added what a multi-cut clip needs: cut detection, on-grid padding, a template
pair, and per-person masks.

## Plan history

- **v1 (approved 2026-11-15, at defaults):**
  - Q1 [scope]: build stage C, last.
  - Q2 [scope]: the task surface as named (`detect_cuts`, `track_people`, `trim_video`
    `pad_to`).
  - Q3 [default]: split an over-long cut into near-equal on-grid pieces flagged
    `split`.
  - Q4 [default]: the bench sets `mask_grow`.
  - Q5 [none]: `content_cuts` moves to `scene_cuts.py`.
- **v2:** recorded the rulings. It also recorded Don's grid ruling: every cut's
  `render_frames` sits on H3's 17n+5 grid inside 124-345 before anything is queued.
- **v3 (approved 2026-11-15, at defaults):** came from stage B's second bounce. Three
  v2 phrases had no form in the workflow language:
  - Q6 [loosen]: an unseeded cut uses the workflow `seed`, not `seed` + index. The
    language has no seed arithmetic.
  - Q7 [loosen]: v2's "6 references" had no source in the repo and was dropped.
    References are bounded by the ref2va class limits that `reference_limits` already
    refuses (9 images, 12 files). The 1-32 cut count stays with `for_each`.
  - Design 7: each limit has one owner, and the template doesn't restate it.

**Corrections the plan made to the proposal and the gate record:**
- `for_each` is static, so one run can't loop over a cut step's output. That makes
  two templates, a prep utility and the swap, following `music-video-cuts` →
  `music-video`.
- No PySceneDetect: the face track's `content_cuts` already finds cuts.
- Cuts are padded up to the grid, not trimmed down to it. Trimming would drop up to 16
  frames per cut and lose sync with the source audio.

## What was built

### Stage A (#1373): `detect_cuts` and `trim_video` `pad_to`

Merged at `dca1ee54`, with a fix at `9d356b61`.
- `dw/tasks/scene_cuts.py` holds the moved `content_cuts` and
  `detect_cuts(video, max_frames=345)`. Each cut comes back as a `shot_record` with
  these fields:
  - `render_frames`: the smallest 17n+5 count in 124-345 at or above the cut's length;
  - `split` pieces for a cut over `max_frames`;
  - `too_short` for a cut under 22 frames.
- `trim_video` `pad_to` holds the last frame and pads the audio with silence.
- Don ruled that `pad_to` takes a numeric string, as every whole-number task argument
  does (#692), and C-F880 was amended to match.
- `detect_cuts` on an `asset:` video now reports `fps`. It was added to
  `_LAZY_FRAME_COMMANDS`.

### Stage B (#1374): `replace-performer-cuts` and `replace-performer`, and the seam bench

Final merge `7ec07d03`, after `12e28608`, `48f6cc45`, `87261138` and `493549fe`.
- **`replace-performer-cuts` (utility):** runs `detect_cuts`, then `video_frames` →
  `segment` (`person`, threshold 0.3). It returns the cut list and one whole-clip mask.
- **`replace-performer` (sequence):** `for_each` over the literal `cuts`. Each cut runs
  four steps:
  1. trim the source and the mask with `pad_to` = `render_frames`;
  2. a ref2va regional edit at `refine_strength` 1.0 with `hold_audio`;
  3. trim the result back to the cut's own `num_frames`;
  4. `concat_videos` (hard cuts), then `pair_audio` with the whole source track.
- Per-cut `seed` and, as a deviation the build added, per-cut `prompt`. Both use
  `for_each`'s existing `{"item": ..., "default": "variable:..."}` form. A step-level
  `seed` accepts that form as a schema fix, with no new resolver.
- **Pre-run refusals, each owned by one module:**
  - `trim.span_problem`, which architecture review moved there from `hold_audio`;
  - `h3_rules.media_problems`;
  - the template's `render_frames` constraint;
  - the `pad_to` rule;
  - `for_each`;
  - `reference_limits`.
- `reference_limits` now refuses bare-string references. A bare string in an H3
  `references` list was slipping past the 9-image count (C-F915.1).
- **The seam bench (C-F890), on lem.** Arms `t16-swap-g0-s1194` / `-s2002`, jobs
  `b4a85961ac8c` and `15a5044427a6`:

  | mask_grow | step, s1194 | step, s2002 |
  |---|---|---|
  | 0 | 9.81 | 11.29 |
  | 1 | 4.37 | 4.45 |

  At 0 the change reaches about 32 px past the mask instead of 80. But the edge is
  harder and the couch back is still redrawn, so **`mask_grow` stays 1**. The eroded
  arm this triggered is #1382 (`complete/performer-eroded-mask-complete.md`), and it
  also failed.
- **The performance comes from the prompt.** At strength 1.0 nothing of the person
  inside the mask survives, so the pose and framing come from the prompt. On ep62, one
  workflow prompt couldn't keep a two-shot and a push-in alike. That is why the
  per-cut `prompt` exists. The skill tells the caller to give each cut a prompt naming
  its pose, action and framing, and to name every other masked person as kept.
- **Verified on lem** (job `0367dd7bb70a`, ep62, stranger portrait):
  - face drift per cut: median 0.491 / 0.641, against a 0.363 pass;
  - 248 frames at 24 fps, with the hard cut kept at 124;
  - levels within 0.03 dB of the source.
- **Re-rolling one cut** with `rerun_job` left the other cut pixel-identical (SSIM
  1.0).
- **Docs:** the `minimax-h3` SKILL.md and `references/regional-edit.md` route "replace
  a performer". They state the limits, the seed and prompt fallbacks, and the seam
  result.

### Stage C (#1375): `track_people`

Merged at `a1208a29`, then `ddc797bc`, `6ffb40e6` and `53f8172f`. Verified on lem at
`44370b83`.
- `dw/tasks/person_track.py`:
  `track_people(video, prompt="person", max_people=4, threshold=0.3)`.
  - Grounding-DINO finds boxes on each cut's first frame. Cuts come from
    `scene_cuts.content_cuts`.
  - Duplicate boxes are removed with `torchvision.ops.nms` (architecture review: buy,
    not build).
  - The detector has one owner, shared with `segment`.
  - `Sam2VideoModel` (`facebook/sam2-hiera-large`) propagates each person through the
    cut.
- Outputs:
  - `<base>-0.0-person-N.mp4` per person, largest first;
  - `-0.0-contact_sheet.png`, the people numbered as the masks are;
  - a `-0.0-people.json` record with a per-person `score`.
- The prep template gained a `track` option. `replace-performer` gained a two-person
  example where one person is replaced.
- **A match is not proof of a person.** A full-frame QR code scores 0.52 as "person".
  The `threshold` 0.3 default was kept, since `segment` and the template share it. The
  tasks guide says to check the contact sheet and the scores, and C-F895 was amended
  to threshold 0.6 (harnest#225).
- C-F898 (a one-person swap of a two-person shot leaves the other person unchanged)
  needs H3, so it ran on lem.

## Cost and bounces

No stage comment named a session `usage:` figure, so session cost is not recorded.
The plan estimated $15-23 plus about $2 for the v3 rework.

| Stage | Tester bounces | Architecture-review bounces | Notes |
|---|---|---|---|
| #1373 | 1 | 0 | C-F880: string `pad_to`. Don ruled the code right, and the case was amended |
| #1374 | 4 | 1 | Driver-parked after 4. Don ruled each bounce a real defect: unsourced 6-ref limit (re-plan v3), bare-string refs, skill text, unrun bench. Arch: second owner of the span rule |
| #1375 | 2, plus 1 incomplete on suite amendments | 1 | The MPS crash was mini-ai's torch 2.15.0 with the fp4-fp8 autoload. Don had the venv downgraded to 2.14.1, and the pin is #1395. Then the C-F895 fixture |

GPU: the seam bench took two H3 arms on lem. The C-F887 / C-F889 verify runs took
about 8.4 min per two-cut swap on lem.

## Deferred and open

- **Not built (non-goals):**
  - the VLM cast-listing prompt (the agent writes the swap line from what it sees);
  - environment replacement;
  - FastH3 Omni (#980);
  - guaranteed lip sync.
- **The couch-back step stays open** at `mask_grow` 1. Neither `mask_grow` 0 nor an
  eroded mask (#1382) removed it without a harder edge or a ghost.
- **Cut detection** misses dissolves and slow fades. The cut list is a literal, so the
  agent edits it by hand.
- **Padding risk:** a short cut padded to 124 holds its last frame longest. Those
  frames are trimmed off, but motion just before them may slow. No case failed on it.
- **Follow-ups filed:**
  - #1397 / #1404: an opt-in SAM 3 `backend` on `track_people`, merged.
  - #1411: a maskless whole-shot alternative, after Bambushu/h3-recast. The BFS LoRA
    was dropped (Don), and the ref2va guide with `head-swap` lost to this route on face
    hold and the untouched regions: `maskless-recast-complete.md`.
  - #1395: the darwin `torch<2.15` pin.
- **Suite:** C-F914-C-F917 hold the v3 wording. Per harnest#215, the contradicting
  v2-wording bullets of C-F882-C-F891 were not counted at verify.
