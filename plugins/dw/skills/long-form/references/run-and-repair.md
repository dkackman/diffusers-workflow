# Running a long take, and repairing one segment

Part of the `long-form` skill. Sources: `bench/records/check.md`,
`bench/records/repair.md`, `bench/records/noise-floor.md` (*Chain check
thresholds*), and `docs/WORKFLOW_GUIDE.md`, *Checking each segment*
(`get_guide("workflows", section="Checking each segment")`).

## The per-segment check

Add the check on the chain block of `templates/minimax/chained-segments`
(`steps[0].pipeline.chain`):

```json
"chain": { "...": "the template's keys", "check": true, "retries": 1 }
```

Each segment is measured before it is kept (segment 1 for `audio_dbfs` only):

| Measure | Against | On failure |
| --- | --- | --- |
| `jump_ratio` (ceiling 4) | the seam's frame step | re-roll |
| `look_ratio` (ceiling 5.5) | the seam's colour step | re-roll |
| `cut_ratio` (ceiling 3.5) | a cut inside the segment, which the seam windows miss | re-roll |
| `audio_dbfs` (floor -70 dBFS) | the segment's own generated audio level, segment 1 included; held audio is skipped | re-roll |
| `texture_ratio` (ceiling 2) | segment 1's fine texture, doubled or halved | warn-only, no re-roll |

- **Why `cut_ratio` exists.** The seam measures caught 1 of 3 broken renders,
  the one whose break sat at the seam. A cut 26 frames past the seam passed.
  Good segments read at most 2.98 on `cut_ratio`, and the harbour cut read
  4.16 (`check.md`).
- **Why texture is warn-only.** Texture drift is in the carry. A re-roll
  renders from the same carry and inherits it, so at `retries: 1` it only
  doubles the cost (ruling R38). A texture warning late in a static take is
  the 30 s cut rule talking: cut the scene into shots, or render it as
  `shot: static_long`.
- **No false positives.** Jump and look gave none on 22 good seams. The
  highest reads were about half the ceilings (`check.md`).
- **Turning on the check costs nothing extra.** With `retries: 0` the take
  reproduced the unchecked runs to the third decimal.
- **A cut asked for inside a segment.** Pass
  `"check": {"cut_ratio_max": null}`, or the segment fails `cut_ratio` and
  re-rolls.
- **When every attempt fails,** the chain keeps the attempt with the fewest
  failed measures and finishes. Its shot records `check_failed`, and the job
  carries a `chain_check_failed` warning.
- **A silent segment.** Dropout segments read -88 dBFS and the quietest real
  one -57, so the `audio_dbfs_min` floor of -70 sits between. A kept segment
  still under it carries a `chain_segment_silent` warning (segment, level,
  floor), with `check` on or off. Turn `check` on with `retries` to re-roll
  it, or set `"audio_dbfs_min": null` when the silence is intended. The
  re-roll draws fresh noise but is not guaranteed to bring sound back: on the
  S5 repro on mini-ai both seeds read about -89 dBFS and the re-roll cost
  about 3 minutes. A segment still silent after it is reported by
  `chain_segment_silent` and by its shot's `check_failed`.
- **Cost.** The quote is the worst case: N(1 + `retries`) segment renders
  for a checked chain that generates its audio (floor on, no `match_audio`,
  no step `hold_audio`; a guide chain carrying its audio counts), else
  1 + (N - 1)(1 + `retries`).

## Repair one segment in place

`templates/minimax/repair-segment` re-renders interior segment k between
k-1 and k+1, and splices it into the take.

1. Run the chain with `keep_segments: true`. Without it there are no
   segment files to repair from.
2. Pass `chain_take` (an `output:` reference to the take), `segment` (k,
   from 2 to N-1), that segment's own prompt, and the chain's settings:
   the canvas, steps, LoRA and `weights_dtype`. For a static chain, also
   pass `latent_anchor`, an `output:` reference to segment 1's file, and
   `latent_anchor_strength` 0.25.
3. Check that the two new seams are as good as the chain's own, with
   `assess_output(probe="analyze_seams")`.

**Measured** (`repair.md`): the S3 repair took 8.1 min on a 3090. Both new
seams tied or beat the chain's own seams, and nothing outside the span
moved. Peak VRAM was 18.92 GiB, under the template's 19.95 GiB estimate.

**A break that spilled into the next segment cannot be repaired.** Segment
k's broken tail became segment k+1's guide, so k+1 opens broken too. The
repair has to end on k+1's frames, so it re-enters the break. Repairing k
and k+1 together is not built. Re-run the take from segment k instead, or
cut at the seam. This is why the check, with `cut_ratio`, has to catch the
break at segment k.

A repaired take cannot be repaired again. Repair the original take, or
re-render it.

## A chain that died at segment k

Both chain templates write each finished segment beside the take as it
runs. A chain that fails at segment k leaves segments 1..k-1 on disk, and
the job's `chain_resume_hint` warning quotes the first file's `output:`
reference.

1. Rerun the same template with the same settings, and pass that
   reference as `resume_from`. Only segments k..N render, and the estimate
   prices only those.
2. A kept file whose frame count does not match the plan is refused,
   naming the segment. Changing `num_frames`, `guide_frames` or the length
   is a new take, not a resume.
3. Read the `chain_resumed` warning. On a guide chain the first resumed
   seam carries the decoded audio, so check that seam with
   `assess_output(probe="analyze_seams")`.

Unmeasured: no bench record covers a resumed take yet.

## The repair ladder

1. Re-roll the segment (the check does this during the run).
2. Repair it in place (`repair-segment`).
3. Make the seam a cut: the shot list's `join: cut`, rendered fresh
   (`script-to-video`, `references/joins.md` there).

## Export the cut for an editor

The `export_timeline` task writes a joined video, with its recorded shots,
as an OpenTimelineIO `.otioz` bundle. DaVinci Resolve 18.5+ imports it with
one clip per shot. Rates are exact rationals: 23.976 is written as
24000/1001.

```json
{ "task": { "command": "export_timeline",
            "arguments": { "video": "output:<identity>/latest/final/cut.mp4",
                           "name": "episode-1" } },
  "result": { "content_type": "application/x-otioz" } }
```

`sources` (one file per shot) points each clip at its own file instead of
the joined one. Checked in Resolve 21: the Mac App Store build is
sandboxed, so copy the bundle into `~/Movies/DaVinci Resolve/` before
File > Import > Timeline; from `~/Desktop` the import silently does nothing.
