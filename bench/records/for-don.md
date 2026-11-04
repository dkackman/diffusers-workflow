# For Don

Things only you can check.

- [ ] Import one exported `.otioz` (`export_timeline`, #1197) in DaVinci Resolve 18.5+ once a bench cut exists: File > Import > Timeline, then check one clip per shot, the rate (23.976 as 24000/1001) and that each clip plays its span. Nothing here can run Resolve.
- Compact catalog listing budget raised to 13_250 (tests) for the new split-takes, narration-long, narration-from-takes and music-long templates. Recommendation: keep (discovery).
- Arch ratchet module count grows on feat/long-form-media (presets, chain_pick, chain_look, chain_check, timeline_export, text_takes, ...). Re-baseline at merge needs your ok. Recommendation: approve.
- Compact catalog listing budget raised again to 13_450 (tests) for `minimax/repair-segment` (#1169, task 14), measured 13_432. Recommendation: keep (discovery).
- `minimax/repair-segment` declares its own VRAM formula (folded guide frames, 42.84 bytes/voxel = 28.71 * (124 + 61) / 124, an upper bound with head_frames held at 22 and tail_frames at most 39) and is listed in `TEMPLATE_SPECIFIC` in `tests/test_vram_inheritance.py`; it is derived, not measured, and over-projects a long render (about 0.9 GB at 192 frames on 1344x768). The Step 3 bench run should record its real peak. Recommendation: replace with the measured figure after the bench.
- Compact catalog listing budget raised to 13_500 (tests) for the #1169 review fixes (repair-segment's `chain_take`, `keep_segments` on four H3 chain templates), measured 13_476. Recommendation: keep (discovery).
- `save_segments`/`keep_segments` in the workflow schema now accept a `variable:` reference (they were boolean only). Recommendation: keep.
