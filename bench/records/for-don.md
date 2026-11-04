# For Don

Things only you can check.

- [ ] Import one exported `.otioz` (`export_timeline`, #1197) in DaVinci Resolve 18.5+ once a bench cut exists: File > Import > Timeline, then check one clip per shot, the rate (23.976 as 24000/1001) and that each clip plays its span. Nothing here can run Resolve.
- Compact catalog listing budget raised to 13_250 (tests) for the new split-takes, narration-long, narration-from-takes and music-long templates. Recommendation: keep (discovery).
- Arch ratchet module count grows on feat/long-form-media (presets, chain_pick, chain_look, chain_check, timeline_export, text_takes, ...). Re-baseline at merge needs your ok. Recommendation: approve.
- Compact catalog listing budget raised again to 13_450 (tests) for `minimax/repair-segment` (#1169, task 14), measured 13_432. Recommendation: keep (discovery).
- `minimax/repair-segment` declares its own VRAM formula (folded guide frames, 38.72 bytes/voxel) and is listed in `TEMPLATE_SPECIFIC` in `tests/test_vram_inheritance.py`; it is derived, not measured. The Step 3 bench run should record its real peak. Recommendation: replace with the measured figure after the bench.
