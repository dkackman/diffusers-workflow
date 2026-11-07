# H3 Anti-Burn LoRA on chain continuation segments (#1151)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-02). Don filed the idea on 2026-11-01 from a review of
[knoic/ComfyUI-MiniMaxH3-PrefixStream](https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream).
Plan v1 was designed on #1151
(https://github.com/dkackman/diffusers-workflow/issues/1151#issuecomment-6569178311),
and Don approved it as written. It was built as one probe stage, #1159. The plan
comment is still the full design; this file records what shipped and what is
still open.

## The problem

Texture coarsens with each seam of an H3 chain: skin goes blotchy and
over-sharpened, fabric turns crunchy, flat regions block up
(`plugins/dw/skills/minimax-h3/SKILL.md`, "Drift compounds per seam").
Imagineer13's [H3-Anti-Burn-LoRA](https://huggingface.co/Imagineer13/H3-Anti-Burn-LoRA)
is trained to prevent that, applied to continuation segments only. The question
was whether it loads and helps on our stack: diffusers' H3 loader, guide-row
`fl2va` continuity, the 4-step turbo at 5 steps, SDNQ int4 and top-k sparse
attention. Its author trained it on masked-latent `ref2va` continuations, lightx2v
8-step, and int8.

## Verdict: build smaller

Probe only: a load check, an A/B on lem, and one catalog entry. **Cut:
`chain.continuation_loras`**, a chain option that would apply a LoRA from
segment 2 on. It would be new workflow syntax.
*What would bring it back:* Don's taste verdict (Q4) that arm B beats arm A on
skin and fabric without going waxy. It would then come back as plan v2 on #1151.
The plan's "Follow-up" section sketches it: the schema's chain block,
`ChainConfig`, a `set_adapter_scales` call before `_run_once`, `FakePipeline`
adapter tracking, template variables, a SKILL.md line, and the guide. Before
that, someone has to show that scale 0 switches the adapter off.

**The plan corrected the issue's premise.** A chained template can't attach a
LoRA to segments 2+ only without engine code. The chain loop runs one loaded
pipeline whose LoRA state is set once at load, and `_segment_arguments` never
touches adapters. The only way today is a hand-built chain: one `fl2va` step
per segment, each later segment guided by the previous clip's tail via
`trim_video`, with the guide heads dropped and the clips joined with
`concat_videos`.

Non-goals that stayed out: engine, schema, MCP or REST changes; a custom key
converter; changes to any shipped template; a `ref2va` chain; a wider strength
sweep.

## Plan history

- **v1 (approved, 2026-11-01).** There was one version and no revisions.
  - Q1 [spend]: yes. The 310 MB download and about an hour of lem, with arm C
    (0.4) allowed if B looked waxy.
  - Q2 [trust]: yes. The Imagineer13 safetensors may be loaded under the
    MiniMax H3 Community License.
  - Q3 [none]: `guide_frames` 39, the LoRA's training context, rather than
    the issue's 22.
  - Q4 [taste]: **still open.** Don judges the comparison comment on #1151
    (https://github.com/dkackman/diffusers-workflow/issues/1151#issuecomment-6572404958).

## What was built

### Stage 1 (#1159): load check, continuation A/B/C, catalog entry

Commit `dece403d`, merged to `develop` as `adff4958`, and deployed to lem on
2026-11-02. No code changed.

- **`loras/minimax-h3/h3-antiburn.json`**, `status: trial`. It has
  `workflow` `["fl2va"]`, `scale` 0.6 with a range of [0.4, 0.8], and
  `stacks_with` `minimax-h3/turbo-4step-keyframe`. Its `license` is
  `minimax-h3-community`, the id the catalog already uses, and its
  `revision` is pinned (`b98eb1d3…`). `use_when` is "continuation segments
  only, never segment 1; needs a hand-built chain until a chain option
  exists". `evidence` lists the four jobs below.
- **Probe workflows** saved in workspace `probe-1151`: `antiburn-load-check`,
  and `antiburn-chain-ab`, a hand-built 5-segment chain of 18 explicit steps.

**Load check: passed.** The anti-burn file uses ComfyUI `diffusion_model.*`
keys, but it loads through the stock `load_lora_weights` alongside turbo with
no unmatched-key warning. `3dc3f60fb217` (turbo + anti-burn 0.6) differs from
`e82ea2430448` (turbo only). The lead measured PSNR 22.75 dB and SSIM 0.78 on
one frame set; the tester measured 25.88 dB and 0.828 at frames 0, 61 and 123.

**A/B/C:** seed 42, 960×544, `guide_frames` 39 with the guide carrying audio,
turbo-4step-keyframe at 5 steps. The join is 464 frames. Segments 1, 3 and 5
were judged at frames 62, 251 and 421, cropped to skin, cable knit and a flat
wall.

| Arm | Job | Anti-burn on segments 2-5 | Segment 5 |
|---|---|---|---|
| A | `e0c2ff0dd654` | none | Burnt: blotchy, posterised skin, haloed edges, crunchy knit, blocky wall |
| B | `bf477d3089dd` | 0.6 | Holds segment 1's look. Pores and lines intact, fine knit slightly softened |
| C | `f8287f133e1c` | 0.4 | In between: blotching and edge halo partly back |

`assess_output` has no burn metric, so the reading rests on the frames.
The lead's reading is that 0.6 is the right default and that 0.4
under-corrects over 5 segments. That reading is not the taste verdict.

**Deviations recorded at build:**
- `license` uses the catalog id `minimax-h3-community`, not the prose name.
- `stacks_with` names the catalog entry id, not the bare word "turbo".
- Validate and runtime both warn `adapter_unrecognized`, because the file
  name names neither fl2v nor ref2v. The entry's description notes this.
  The tester saw a cosmetic mismatch: `validate_workflow` matches the file to
  the catalog and doesn't warn on fl2va, but the runtime warning still fires.

**Verify:** the tester verified all seven cases (M-F298 to M-F304) on the
first pass. M-F300, the `rejected` branch, did not apply. `recommend_loras`
does not offer the entry for `ref2va` or `t2va` requests. No shipped template
or skill changed.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1159 | 0 | Not recorded on the stage. Lem time was about the plan's estimate: the load check plus three arms, with C run |

## Deferred and open

- **Q4, Don's taste verdict** on the comparison comment. "Yes" moves the entry
  to `proven`, adds a line to the H3 skill, and brings back
  `chain.continuation_loras` as plan v2. "No" moves the entry to `rejected`
  with his note. Until he rules, the entry stays `trial` and the skill doesn't
  recommend it.
- **One seed, one scene.** The card reports ±15 points of variance per chain.
  The probe shows a large effect, but it can't establish `proven` by itself.
- **The result comes from a hand-built chain.** It applies to the engine's
  `chain` only by inference.
- Untried: `ref2va` continuations, 22-frame guides, other turbo variants.
- Related: #1150 (latent-tail continuity) attacks the same drift at its
  source. #1152 (texture-drift measure) has since added `compare_outputs`
  `texture_bands` (stage #1160, after this probe ran). It could put a number
  on a rerun.
