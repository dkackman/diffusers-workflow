# Texture-band drift measurement on `compare_outputs` (#1152)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-02). Don filed the idea on 2026-11-01 from a review of
[knoic/ComfyUI-MiniMaxH3-PrefixStream](https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream)
(MIT). Plan v1 was designed on #1152
(https://github.com/dkackman/diffusers-workflow/issues/1152#issuecomment-6569250864),
and Don approved it as written. It was built as one stage, #1160. The plan
comment is still the full design; this file records what shipped and what is
still open.

## The problem

H3 chain texture drift (blotches, stripes, coarsening, or the over-smoothing a
fix can cause) had no number, so drift A/Bs such as #1151 rested on the eye
alone. `minimax-h3/references/cuts.md` measured drift as lightness, saturation
and contrast only.

## Verdict: build smaller

One opt-in parameter on `compare_outputs`. No new tool and no assess probe.

**Cut**, with what brings each back:
- **`region` + bands on `assess_output`.** That would be a fourth name for the
  crop concept, a second full-resolution decode, and a break in the "exactly
  three probes" invariant. *Comes back* if agents need bands inside an
  unattended assess pass (for example a seam rule), shown by a field report.
- **Pre-encode PNG export of chain segments** (a hook in
  `dw/pipeline_processors/chain.py`). Chains only ever write MP4, so every
  measurement includes encoder texture, mostly in the 2-8 px band. *Comes back*
  if an A/B under this tool lands on a fine-band difference that encoding could
  explain.
- **Any rule, threshold or verdict on the bands.** The output stays a
  measurement.

**The plan corrected the issue on three points:**
- `compare_outputs` metrics were all pairwise against the baseline. Bands are
  per output, so the baseline gets values too, plus a ratio to the baseline per
  band on every other output.
- `compare_outputs` needed 2+ names. Within-chain drift (segments 1, 3, 5 of one
  video) needs one, so `texture_bands=true` lowers the minimum to 1 (Don's Q2).
- "Measure before encode" isn't possible in dw today (see Cut). The guide says
  so.

## What shipped (stage #1160)

- **`dw/texture_bands.py`**: a pure function on an RGB array. For each channel
  (Y with ITU-601 weights, R−G and B−G, on the 0-255 scale) it removes a
  least-squares plane, applies a 2D Hann window, runs `rfft2`, and reports RMS
  over radial annuli. The bands are `fine` 2-8 px, `mid` 8-32 and `coarse`
  32-128. Values are rescaled for the window, so a sine of amplitude A reads
  about A/√2. A band longer than the box's short side is `null`.
- **`compare_outputs(texture_bands=true)`** on the MCP tool, the REST compare
  body (`CompareRequest.texture_bands`) and `dw/media_compare.py`. It is off by
  default, and the payload and summary are byte-identical without it.
  - One name is accepted with it. Without it, 2-8 names are still required.
  - A per-name crop list with bands is refused (bands need one shared box). A
    shared crop under 16 px a side is refused.
  - The reply adds a `texture_bands` header (band → px range) and a top-level
    `texture` list with one entry per output, `{name, row, baseline, cells:
    [{texture, texture_ratio, reason}]}`. `texture_ratio` (output ÷ baseline,
    `null` where the baseline band is 0) is set on non-baseline outputs. Both
    are also copied into the existing pairwise `metrics` cells.
  - The MCP summary adds a "texture bands (RMS, a measurement not a score)"
    block.
- **Docs:** a `#### Texture bands` subsection under `### Comparing outputs` in
  `docs/WORKFLOW_GUIDE.md` (the caveats, measuring after encode, the A/B
  method). Also `docs/MCP.md`, `docs/SERVER.md`, `docs/ARCHITECTURE.md` (Output
  comparison row) and `docs/ATTRIBUTIONS.md` (the PrefixStream credit). Pointers
  in `plugins/dw/skills/minimax-h3/SKILL.md` and `references/cuts.md`. The
  OpenAPI document and UI schema were regenerated. The tool docstring grew by one
  clause, and the 16,000-token surface budget held without being raised.
- **Guide lookup:** `dw/server/guides.py` `_extract_subsection` now tries `####`
  headings in a second pass, after no `###` heading matched, so
  `get_guide("workflows", section="Texture bands")` resolves. Names that resolved
  before still resolve to the same place. A side effect: existing `####`
  headings such as "On-demand components" now resolve too.

**Deviations from the plan**, all recorded on #1160:
1. The baseline has no `metrics` entry, so per-output band numbers live in the
   top-level `texture` list as well as in the non-baseline `metrics` cells.
2. "Comparing outputs" and "Remediation" in the workflows guide were bold
   paragraphs that `get_guide` couldn't reach. Both became `###` headings.
3. No separate plugin version bump: the plugin's version follows the engine's
   and the release script bumps both.

## Acceptance

Eight cases, C-F662 to C-F669 in `regression-suite-complete.md`, written by the
tester from the plan. All eight pass on `develop` @ `768400da` on lem, and they
are now ordinary regression cases.

## Bounces

- **#1160: 1 bounce.** C-F669 step 2: `get_guide("workflows",
  section="Texture bands")` was refused because *Texture bands* was a bold
  paragraph. The fix made it a `####` heading and taught the guide matcher a
  `####` pass. It verified on the second pass.

## Cost

The stage comments give no `usage:` figures, so no cost is recorded. The plan
estimated $4-6.

## Still open

- No consumer has used it yet. #1151's taste verdict (its Q4) and #1150 (parked)
  are the expected first users.
- The three cuts above, each with its trigger.
