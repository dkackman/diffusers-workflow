# Running a long take, and repairing one segment

Part of the `long-form` skill. Sources: `bench/records/check.md`,
`bench/records/repair.md`, `bench/records/noise-floor.md` (*Chain check
thresholds*), `bench/records/words.md`, `bench/records/narrated.md`, and `docs/WORKFLOW_GUIDE.md`, *Checking each segment*
(`get_guide("workflows", section="Checking each segment")`).

## The per-segment check

Pass the check as arguments to `templates/minimax/chained-segments`:

```json
{ "check": true, "retries": 1 }
```

Each segment is measured before it is kept (segment 1 for `audio_dbfs` and
`words` only):

| Measure | Against | On failure |
| --- | --- | --- |
| `jump_ratio` (ceiling 4) | the seam's frame step | re-roll |
| `look_ratio` (ceiling 5.5) | the seam's colour step | re-roll |
| `cut_ratio` (ceiling 3.5) | a cut inside the segment, which the seam windows miss | re-roll |
| `audio_dbfs` (floor -70 dBFS) | the segment's own generated audio level, segment 1 included; held audio is skipped | re-roll |
| `texture_ratio` (ceiling 2) | segment 1's fine texture, doubled or halved | warn-only, no re-roll |
| `words` (`words_similarity_min`, default `null` = off; 0.8 when set) | the segment's line (`lines`, else its prompt's `<d>`), against its kept audio transcribed by Whisper on CPU; segment 1 included | re-roll |

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
- **The words check, for a dialogue chain.** It is off by default:
  `bench/records/words.md` found no clean separation of the takes Don kept
  from the ones he re-rendered. Turn it on with an object:

  ```json
  { "check": { "words_similarity_min": 0.8 }, "retries": 1,
    "lines": ["We the people of the United States,",
              "in order to form a more perfect union,"] }
  ```

  0.8 (`chain_check.WORDS_SIMILARITY_MIN`) passed every good take seen and
  catches only gross failures: a wrong, missing or garbled line. It does
  not catch one word dropped at a segment's head, which is what Don's bad
  takes shared (similarities 0.94-0.96, above two good takes at 0.90). A segment
  also fails, at any similarity, on `line_clipped_at_end`,
  `transcript_loop`, `tag_spoken` or `unscripted_speech`.
  - **The line.** `lines` holds one string per segment, exactly as many as
    there are segments; unlike `prompts`, the last is not repeated. Without
    `lines`, segment k's line is the `<d>` dialogue of its prompt. A segment
    with no line (no `<d>`, or `""`) records `words: null` and is never
    re-rolled for it.
  - **The seam.** The guide's carried words are already cut by the head
    trim. A partial word from the previous line at the head of the kept
    span is dropped before scoring (`seam_head_dropped`); an invented word
    counts.
  - **When it fails.** The `chain_check_failed` warning carries `heard`,
    `similarity` and `findings` for the kept attempt (the higher
    similarity, among words-only failures). With
    `every_attempt_failed: true`, every seed spoke the line wrong, so the
    cause may be in the carry, which a fresh seed inherits: repair that
    segment with `repair-segment` (below) rather than raise `retries`.
    That needs `keep_segments: true` and an interior segment (2 to N-1);
    segment 1 or N can't be repaired in place, so re-render the take or
    split the line differently across `prompts`.
  - **`line_clipped_at_head`** in the findings means the line opened inside
    the guide's head trim, so its first words were cut from the take. A
    fresh seed starts the line at the same place: move the line later in
    the prompt rather than re-rolling (#1300).
  - **`line_opening_missing`** on a passed segment means its audio opens
    past the line's first words (`missing` names them): it re-rolls while
    `retries` remain, and warns when every attempt has it (#1308).
  - **Cost.** One Whisper pass on CPU per attempt, about 1.3 s for an 8 s
    segment on lem once loaded (`words.md`), counted in `plan.estimate`.
    No VRAM is added beside H3.
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
   `assess_output(probe="analyze_seams")`. The kept seams get a short
   declick fade, not `crossfade_ms`, because the blend material is gone.
4. A fan-out (three keyframes, three takes) resumes each take from its
   own files. A take with none left renders in full, and its
   `chain_resumed` warning says `kept: 0`.

Unmeasured: no bench record covers a resumed take yet.

## The repair ladder

1. Re-roll the segment (the check does this during the run).
2. Repair it in place (`repair-segment`). A words failure on every attempt
   (`every_attempt_failed`) skips straight here.
3. Make the seam a cut: the shot list's `join: cut`, rendered fresh
   (`script-to-video`, `references/joins.md` there).

## A narrated take: narration, probe, judge, continue

A speaking subject held to a narration is `templates/minimax/narrated-take`,
measured in `bench/records/narrated.md` (12 segments, about 86 s, tier `best`,
seed 1; about 18-19 min per segment on dialogue and walk_keyframed; standing took 282 min for 12 segments (about 23.5, with two re-rolls), on an RTX 3090).

1. **Narration.** `templates/narration-retime` makes the held track. Its
   `num_frames` and `guide_frames` must equal the take's (192 and 22), or
   the segment boundaries land in the wrong places. Listen to it with
   `get_output_audio` before spending GPU time on picture.
2. **Probe.** Validate the take with `stop_after` set: 4 segments for
   `shot: dialogue`, up to 12 for `shot: standing`. The estimate prices only
   those segments. Pick `shot` by length: `dialogue` is good for about 30 s;
   `standing` holds to about 85 s (she stands, and each segment eases back to
   the opening frame); `walk_keyframed` moves, and needs one hand-built
   keyframe per segment, each with an open, mid-word mouth.
3. **Judge.** Read each shot's `face_drop`, which is recorded and never
   failed. It is the face's drift from segment 1: `dialogue` read 0.003-0.057
   to 30 s, then 0.146 at 36 s, where an eye first calls the face etched;
   `standing` stayed within 0.015 to 85 s (seed 1 only); `walk_keyframed` stayed at or under
   0.041 on two seeds. A rising `face_drop` is the face, so the tool is
   `get_output_frames(at=...)` late in the take; seams are
   `assess_output(probe="analyze_seams")`. Lip sync is judged by eye
   (`measure_lip_sync` is a coarse take-level indicator, not a segment check).
4. **Continue.** Run again with `resume_from` set to the `output:` reference
   of the probe's last kept segment and `stop_after` cleared to `null`. Only
   the segments after it render.
5. **Repair.** One bad segment is re-rendered in place with
   `templates/minimax/repair-segment` (the take keeps its segments). A face
   that drifted on its own is not repaired this way:
   `templates/minimax/face-repair` had no reference to aim at and was dropped.

### Longer than one take: passages joined as cuts

A piece longer than one clean take is passages joined as cuts, one narrated
take each. Not yet run end to end; nothing is measured past 86 s.

1. **One narration for the whole script.** Speak it once with
   `templates/narration-long` from one voice reference (`audio.md`), then
   retime it once with `templates/narration-retime`. One track gives one
   voice, one room tone and one level across every cut; a narration made per
   passage gives none of those.
2. **Passages.** Split the script at paragraph or section breaks so each
   passage fits its preset. At the retimed pace (about 126 words a minute,
   `bench/records/narrated.md`) that is about 60 words for `shot: dialogue`
   (about 30 s) and about 175 words for `shot: standing` (about 85 s). Cut
   the retimed track at those section-break pauses with `slice_audio`
   (`start_seconds`/`duration_seconds`, the cut in the middle of the pause;
   the retime report's `words` give the times), and `keep_output` each slice.
   A slice's segment boundaries restart at its cut, so its
   `mid_word_boundaries` are not the whole track's: they stay advisory.
3. **Takes.** Run one `templates/minimax/narrated-take` per passage: that
   passage's slice as `narration`, that passage's text as `script`, and **a
   different still or angle** for each. A cut to the same still reads as a
   reset. Probe passage 1 first, as above.
4. **Join.** Author a one-step `concat_videos` workflow: the takes as
   `output:` refs in order, `trim_frames: 0` (a hard cut), `match_levels:
   "lufs"` (off by default), `fps: 24`. Every take must be the same frame
   size. Then `export_timeline` (below) if an editor is wanted.

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
