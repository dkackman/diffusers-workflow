# todos

## chaining

- Keyframe pre-planning: since fl2va takes first and last keyframes, generate a storyboard of keyframes first, then fill each segment between consecutive pairs. Segments become independent — no cumulative drift, and parallelizable. This is arguably a better long-video strategy than sequential chaining
- Anti-drift correction: histogram/color matching each segment's frames back to segment 0 — cheap, addresses the best-known failure mode of autoregressive video chaining
- Per-segment prompt scheduling: "prompt": ["intro shot...", "then the camera...", ...] — one prompt per segment, for narrative arcs (falls out of the chain loop almost for free)
- Chained prompt-embed reuse — LTX2I2VChained only. chain.py:255 re-runs the full prompt through the 14 GB Gemma once per segment; at 3 segments that's two redundant encodes per run. Engine change, not config.

## performance

- Save a pre-quantized checkpoint. The 45 s SDNQ pass re-quantizes identical weights on every cold start. Save once locally, point model_name at it, and cold starts drop to plain weight loading. Also speeds the REPL's first load. This is the one real remaining structural win for non-REPL use.
- save compiled checkpoint like above
- torch.compile with repeated_blocks. Attacks the 25 s denoise across 48 repeated blocks. It only became viable when the transformer went resident — compile and group-offload hooks fight each other, and that's gone now. But first-run compilation costs more than it saves, so it only pays off paired with #1, where the graph survives between runs.
