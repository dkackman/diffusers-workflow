# H3 video templates warn `audio_no_headroom` on their own defaults (#174)

Written by model `sonnet` via provider `anthropic`.

## The report

M-F008 (`templates/minimax/video-with-audio-768p`, stock defaults) asserts
`warnings: []` and got one: `audio_no_headroom` at `peak_dbfs: -0.14`, from
H3's own generated soundtrack, muxed with no `normalize_audio` step anywhere
in the template. `get_gallery_metadata` on the finished file reads
`peak_dbfs: -1.1158` — the AAC mux moved the level *down* 1.0 dB from the
source, not up. That is the third run in a row where this template family's
AAC mux overshoot went negative on this box (`M-F012.jsonl`: -0.92 dB and
-1.11 dB under target), against the +1.94 dB *over* that #161 measured
(on `music-video`'s mux, a different song, a different day). #174 asks
which of two fixes is right; this doc is that decision, plus one thing
neither #174 nor #159/#161 had: the current code already runs the
ground-truth check that would have caught this, and throws its answer away.

## Root cause, part one: scope

#159 gave `normalize_audio` to `templates/minimax/music` and `music-video`
only, because those were the templates the -0.94 dBFS-over-full-scale case
(#158) was filed against. Every H3 *video* template — `video-with-audio`,
`video-with-audio-768p`, `dialogue-short`, `storyboard`,
`chained-segments`, `chain-matched-and-aligned`, `chain-matched-to-audio`,
`chain-video-continuity`, `composable-references`, `first-and-last-frame`,
`generated-subject-reference`, `image-to-video`, `last-frame-only`,
`reference-to-video`, `voice-timbre-reference` — generates its own
soundtrack (`t2va`/`fl2va`/`ref2va`) and mixes it straight into the mux with
no gain stage. `#159` never claimed to cover them; it just happened to
close the two tickets that existed. All 15 templates carry the same
exposure #174 found in one of them.

## Root cause, part two: the warning suppresses its own correction

`dw/result.py`'s two checks are meant to compose: `warn_without_headroom`
(dw/result.py:77) reads the pre-encode waveform and predicts a clip;
`warn_if_written_above_full_scale` (dw/result.py:113) reads the encoded
file back and reports the truth, `already_warned=True` (the pre-encode
warning fired) suppressing the post-encode check outright
(dw/result.py:122-126, dw/result.py:132-133) —
because when #161 wrote it, the only encoder overshoot on record was
positive (mp3 +0.1 dB, AAC +1.9 dB): if the source was already over the
line, the encoded file surely would be too, so probing it again looked
redundant. Both call sites already run the same way for a video mux
(dw/result.py:762, dw/result.py:800) and for audio-only saves
(dw/result.py:641), and the post-encode call always runs, unconditionally,
right after the save (dw/result.py:677-682) — it just returns immediately
when told to.

M-F008 breaks that assumption: the AAC mux moved this soundtrack *down* a
full dB. The post-encode probe would have said so — it runs the same
`probe_media` call `get_gallery_metadata` used to read -1.1158 — but never
got the chance, because `already_warned` was `True` before it started.
The consumer got the wrong answer even though the code to get the right one
executes on every single run.

## Two fixes, not either/or

#174 poses these as alternatives. They're not — the second is a bug fix
underneath the first regardless of which way the loudness question goes.

**1. Stop suppressing the ground-truth check for a mux.** Drop
`already_warned` from the `warn_if_written_above_full_scale` call at
dw/result.py:680 for `content_type.startswith("video")` (keep the
suppression for a plain audio save, where #159's mp3 measurement — +0.1 dB,
consistently positive — still supports "the source being over the line
means the file will be too"). A video mux now always reports what it
actually wrote. If `audio_no_headroom` fired and the file came back clean,
nothing further is said — `warn_if_written_above_full_scale` only speaks
when the decoded peak is itself at or above 0 dBFS (dw/result.py:142). If
the mux is over, the caller gets `audio_clipped` with the real number
instead of (or, currently, never, since it can't fire when
`audio_no_headroom` already claimed the slot) a stale prediction. This
closes the immediate defect on its own: a clean mux stops being reported as
suspect, on any template, without adding a gain stage anywhere.

**2. Whether to also add `normalize_audio` to the H3 video templates** is
the part that's genuinely a product call, and where I think the case for
"yes" is weaker than #159's was. #159 added the step because a measured
run had *already* clipped end-to-end (#158's +0.94 dBFS case). Nothing here
has: every H3 video-template measurement on record (M-F008, and the two
`M-F012.jsonl` entries) landed *under* full scale after the mux. Baking a
fixed `peak_dbfs: -3` gain reduction into fifteen templates to guard
against a failure mode that fix (1) already reports accurately, and that
hasn't reproduced once on this family, trades a small amount of loudness on
every default run for a warning that (post fix 1) no longer misfires. I'd
hold off unless a future run actually shows an H3 video mux clipping — at
which point fix (1) is exactly the mechanism that will catch it and say so
correctly, with a real number to size the gain from instead of a guess.

## What I'm asking for

Approval to ship fix (1) (the suppression bug) now — it's a pure
correctness fix to a warning that currently lies when the encoder happens
to undershoot instead of overshoot, no template changes, no gain applied to
any deliverable. Fix (2) (gain stage on the fifteen H3 video templates) I'd
rather leave undone until there's a clipped-in-practice case to size it
against, the way #158 gave #159 one; happy to be overruled if the
preference is defense-in-depth over waiting for evidence.

## M-F008's `warnings: []` assertion

Once fix (1) ships, M-F008 as currently worded should pass on a fresh run
of `video-with-audio-768p`'s stock defaults (H3's soundtrack is
under-scale after the mux, same as the other two measurements), without
touching the case text — #173 (parked, owner:don) is where any wording
change for the case itself belongs; this issue only concerns the engine
behavior.

## Regression case proposed (for the tester to add once verified)

- Suite: `regression-suite-model-specific.md` (H3-specific mux behavior).
- Call: `run_workflow("templates/minimax/video-with-audio-768p")`, stock
  defaults.
- Expected once fix (1) ships: `status: "succeeded"`, and no
  `audio_no_headroom` warning unless `get_gallery_metadata`'s `peak_dbfs`
  on the written file is itself at or above 0 dBFS (in which case
  `audio_clipped` should appear instead, with a `peak_dbfs` matching what
  `get_gallery_metadata` reports for the same file).

## Amendment (2026-09-16): fix (1) as approved and shipped is insufficient

Written by model `sonnet` via provider `anthropic`.

Fix (1) was approved and shipped as `ce82f06` (merged `aab4ef5`, deployed to
`lem`): `warn_if_written_above_full_scale`'s `already_warned` suppression is
now passed as the real prior state only for `content_type.startswith("audio")`;
for `"video"` it is always `False`, so the post-encode ground-truth probe
always runs and always speaks for a video mux.

That is correct as far as it goes, but the tester's verification run against
`lem` (job `46f4d0f2faf1`, run `20260916-092932-7e47a764`,
`video-with-audio-768p` stock defaults) shows it doesn't reach the acceptance
criterion:

- Written file measured `peak_dbfs: -1.1158844…` via `get_gallery_metadata` —
  well under 0 dBFS, a clean mux.
- `job.warnings` still contains `audio_no_headroom` at `peak_dbfs: -0.14`.
- `get_job_events` shows why: that warning is emitted at seq 32, `at: 571.7`,
  *before* the write (seq 33, `at: 574.1`). It is the **pre-encode**
  `warn_without_headroom` check (dw/result.py:77), which fix (1) never
  touched. No warning event of any kind follows the write in this run — the
  post-encode probe ran (per the code path) but had nothing to report,
  because the file came back clean, exactly as fix (1) predicts. It just
  doesn't cancel or replace the pre-encode warning that already fired.

The approval's acceptance criterion was "no `audio_no_headroom` … unless the
written file is ≥0 dBFS, in which case `audio_clipped` should appear
**instead**." What ships today instead produces *both* checks running
independently: the pre-encode one always fires on this family's stock
defaults (H3's own soundtrack sits close enough to the line that
`warn_without_headroom`'s threshold catches it before any encoding happens),
and the post-encode one only *adds* a second warning when the mux is
genuinely bad — it never suppresses the first one when the mux turns out
fine. So the stock M-F008 run still asserts `warnings: []` and still fails
that assertion, unchanged from the original report.

### The undershoot record now has four points, all negative

| Run | Template / mux | Predicted (pre-encode) | Written (post-encode, `get_gallery_metadata`) | Delta |
|---|---|---|---|---|
| #161's `music-video` case | AAC, different song | — | — | **+1.94 dB** (the one positive measurement on record, a different template family) |
| `M-F012.jsonl` #1 | H3 video mux | target | measured | **-0.92 dB** |
| `M-F012.jsonl` #2 | H3 video mux | target | measured | **-1.11 dB** |
| M-F008 original report | `video-with-audio-768p` | -0.14 dBFS | -1.1158 dBFS | **-0.98 dB** |
| Tester's 2026-09-16 re-verify | `video-with-audio-768p` | -0.14 dBFS | -1.1158844… dBFS | **-0.97 dB** |

Four consecutive H3-family AAC muxes, on two different templates and two
different sessions, all undershoot by roughly 1 dB. The one overshoot on
record is a different template (`music-video`) on a different day. This
doesn't prove H3's mux never overshoots — it's still a small sample — but it
directly contradicts treating "encoder overshoot is always positive" as safe
to assume for this family, which is the flag Don raised in the approval
comment for the audio-only suppression. The same caution now applies with
actual evidence behind it on the video side.

### What changing would require

To satisfy the approved criterion literally — `audio_no_headroom` never
reaching the caller for a video mux that measures clean — the pre-encode
`warn_without_headroom` result would have to be held back rather than
emitted immediately, and either dropped or upgraded to `audio_clipped`
once the post-encode probe runs ~2.4 s later. That's a real behavior change
beyond fix (1) as approved: it changes *when* a warning reaches the caller
(after the write completes, not at the point the risk is detected) and
*which* checks a video mux can ever surface (post-encode ground truth only,
never the prediction) — for every content type that already goes through
`warn_without_headroom`, not just H3's video family, since the suppression
logic is generic to `dw/result.py` and not H3-specific.

I'm not implementing this. Per the triage note that queued this issue, the
approval rested on the proposal's premise that fix (1) alone would clear
M-F008 on its own — the 2026-09-16 verification run disproves that premise,
so this needs a fresh decision rather than a unilateral extension of what
was already approved.

### Asking

Approve or decline holding the pre-encode `audio_no_headroom` warning for a
video mux until the post-encode probe has run, replacing it with
`audio_clipped` (measured value) when the file is genuinely over, and
emitting nothing when it's clean — scoped to `content_type.startswith("video")`
only, audio-only saves unaffected. Fix (2) (a `normalize_audio` gain step on
the fifteen H3 video templates) remains held, unchanged from the original
decision — no run on record has an H3 video mux actually clipping.
