# Remaining work: H3 video mux headroom warning (#174) - pre-encode/post-encode reconciliation

Split from `h3-video-mux-headroom-warning-complete.md` on 2026-09-20. Fix (1)
— stop suppressing the post-encode ground-truth probe for a video mux — is
implemented (`dw/result.py`, shipped `ce82f06`/merged `aab4ef5`, deployed to
`lem`). The tester's 2026-09-16 verification run showed it is insufficient on
its own: the pre-encode `audio_no_headroom` warning still fires unconditionally
before the write, independent of what the post-encode probe finds, so
`warnings: []` still fails on `video-with-audio-768p` stock defaults. What
remains is the amendment's open decision and its implementation, unresolved:

## The remaining decision, "Asking" (2026-09-16 amendment)

Approve or decline holding the pre-encode `audio_no_headroom` warning for a
video mux until the post-encode probe has run, replacing it with
`audio_clipped` (measured value) when the file is genuinely over, and
emitting nothing when it's clean — scoped to `content_type.startswith("video")`
only, audio-only saves unaffected.

This is a real behavior change beyond fix (1): it changes *when* a warning
reaches the caller (after the write completes, not at the point the risk is
detected) and *which* checks a video mux can ever surface (post-encode ground
truth only, never the prediction) — for every content type that already goes
through `warn_without_headroom`, not just H3's video family, since the
suppression logic in `dw/result.py` is generic.

## Remaining steps, if approved

1. Hold the pre-encode `warn_without_headroom` result (`dw/result.py:77`)
   rather than emitting it immediately via `emit_warning`, for
   `content_type.startswith("video")`.
2. Run the post-encode `warn_if_written_above_full_scale` probe as today.
3. Drop the held warning if the file measures clean; upgrade it to
   `audio_clipped` with the measured value if the file is genuinely over.
4. Leave audio-only saves unaffected (unconditional, unheld, as today).
5. Update the M-F008 regression case wording once this ships (it should then
   pass `warnings: []` on `video-with-audio-768p` stock defaults for real,
   not just as originally predicted by fix (1) alone).

## Fix (2) — remains explicitly deferred, no decision needed yet

Whether to add a `normalize_audio` gain stage to the 15 H3 video templates.
Held per the original recommendation: no run on record has an H3 video mux
actually clipping (four consecutive measurements all undershoot by roughly
1 dB); wait for a real clipped-in-practice case to size the gain against,
the way #158 gave #159 one.

## Evidence this decision rests on (undershoot record, all four points negative)

| Run | Template / mux | Predicted (pre-encode) | Written (post-encode) | Delta |
|---|---|---|---|---|
| #161's `music-video` case | AAC, different song | — | — | +1.94 dB (the one positive measurement on record, a different template family) |
| `M-F012.jsonl` #1 | H3 video mux | target | measured | -0.92 dB |
| `M-F012.jsonl` #2 | H3 video mux | target | measured | -1.11 dB |
| M-F008 original report | `video-with-audio-768p` | -0.14 dBFS | -1.1158 dBFS | -0.98 dB |
| Tester's 2026-09-16 re-verify | `video-with-audio-768p` | -0.14 dBFS | -1.1158844… dBFS | -0.97 dB |
