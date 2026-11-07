# H3 chain estimate: a fixed term in the scaled cost (#1224)

Status: proposal, needs Don's approval.

## Finding

The drill's 90 s chain (`ab7f54f73b2f`: 7 segments of 345 frames at 12 steps) ran in
71.0 min. `chained-segments.json` is curated at 13.5 min for 3 segments of 124 frames
at 5 steps (turbo LoRA). `scale_curated` plus the segment ratio is purely proportional:
13.5 x (12/5) x (345/124) x (7/3) = 210 min.

No single axis is wrong. Run the other way, the measured job scaled back to the
defaults is 71.0 / 15.58 = 4.6 min, not 13.5. The curated figure therefore carries
about 9 min that does not scale with steps, frames or segments (model load,
text encode, VAE decode, the turbo LoRA fuse). Proportional scaling multiplies that
fixed share by 15.6 along with the real work, so the over-quote grows with the
distance from the defaults. `scale_curated`'s docstring already says it "over-quotes a
little when a value rises"; at 15x it is 3x.

One measured point cannot split the curated 13.5 into fixed and per-unit parts
(one equation, two unknowns). The other H3 entries do not isolate it either.

## Proposal

Add an optional `fixed_minutes` to a `cost` entry (like `per_entry`, a measured part
of the figure): the share that does not scale. The scaled quote becomes
`fixed + (minutes - fixed) x factor`. It is declared by the workflow, so the engine
holds no H3 knowledge. Set `fixed_minutes` on the H3 chain templates once measured
(about 9 from the drill, to be confirmed by a second run, e.g. 3 segments at 345
frames / 12 steps, which separates the two unknowns).

## Why this is parked

A new key on the `cost` entry is a concept authors would have to learn, and it needs
a GPU measurement to set. Both are Don's call. The alternative is to re-curate
`chained-segments` at a measured point nearer the long-form shape and leave the
engine alone; that fixes this template only.

Default if approved: `fixed_minutes` on `cost` entries, schema + `_scaled` change,
tests with the drill's figures (expected quote within ~1.3x of 71 min), and a
re-measure of one chain run before the template value is committed.
