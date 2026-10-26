# FLUX 3 Image Klein: catalog onboarding design

Date: 2026-10-15. Audit: [2026-10-15-flux-3-klein-audit.md](../../proposals/audits/2026-10-15-flux-3-klein-audit.md).
Status: draft, probes on lem in progress.

## Decisions taken (Don, 2026-10-15)

- **Scope: the distilled core.** Text-to-image, upsample-then-generate, single edit,
  multi-reference, bounding-box layout, and a 24 GB recipe. Base + CFG (needs modular
  `guider` support), LoRA and typography are dated follow-ups.
- **The upsampler is a prior step** on `black-forest-labs/flux-3-image-klein-pup`
  (Qwen3.5-9B, Apache-2.0, ungated), released before the generation step loads. BFL:
  "do not use the model without a prompt upsampler".
- **lem moved after the loop finished:** diffusers 7ecb1e1b (PR #15046) installed via
  `update_diffusers` and `dw-serve` restarted, 2026-10-15 18:58 CDT.

## Verified on lem

- `modular_model_index.json` (read with lem's token): `Qwen3VLForConditionalGeneration`,
  `Qwen3VLProcessor`, `Flux3Transformer2DModel`, `AutoencoderKLFlux2`. No `.py` in the
  repo, so no `trust_remote_code`. The catalog structure test's refusal does not bite.
- A `ModularPipeline` step on `black-forest-labs/flux-3-image-klein` validates on lem.
- `update_diffusers` alone leaves a running `dw.serve` with a half-old diffusers: the lazy
  `modular_pipeline` import raises `RuntimeError` in `component_types`/`component_names`
  validation until the server is restarted (the known admission gap from the
  Kandinsky 6 onboarding).
- The first in-worker download looked hung (3.0 GB, no traffic for a 5 s sample) and P1
  was cancelled, but that blob had completed a minute earlier, so it was probably
  between files: not a confirmed hang. The weights were then fetched with `hf download`
  on the box (~34 MB/s).

## Probes (results to fill in)

| Probe | Job | Result |
|---|---|---|
| P1a as below, text encoder + VAE `residency: on_demand` | 8c88debc90ab | Failed: `index is on cpu` in the text encoder |
| P1b transformer and text encoder unplaced, VAE on demand | 19ece0b9522a | Failed: `Cannot generate a cpu tensor from a generator of type cuda` |
| P1d distilled t2i bf16, 1024², 16 steps, vendor `man_dog` caption, seed 7, transformer + text encoder + VAE `device: cuda` | 9cc1ce441555 | **Passed.** 24 s total: load 10.1 s, encode 0.4 s, denoise 12.1 s (0.8 s/step), decode 0.4 s. 19.2 GB allocated, 22.1 GB reserved peak at decode (1.7 GB free). The image follows the caption element by element |
| P2 pup via `text_generation` as-is, `t2i` system prompt, `man_dog` request | a5471561de18 | Ran (22 s, cuda:1); vendor-style caption, but it opens mid-phrase in lowercase ("medium-wide outdoor photograph ...") |
| P2b chat template check | (script) | The template's default opens `<think>\n` (thinking on); `enable_thinking=False` gives `<think>\n\n</think>\n\n`. P2 ran in thinking mode |
| P3 multi-reference edit: `image` a list of two `{"location": url}` refs (BFL examples/inputs 2.png mug, 3.png dog, commit bbfe317d), BFL upsampled mugprint caption, seed 23, 1024², 16 steps | cc7fb1b29091 | **Passed.** The list arrives as PIL images; the output printed the dog on the mug and kept the scene. 45 s total: encode 0.5 s, denoise 41.0 s (2.6 s/step, references add tokens), decode 0.3 s. 19.2 GB allocated, 22.1 GB reserved peak |

What P1 settled:

- `ModularPipeline._execution_device` is the first model component's device unless a
  group-offload or accelerate hook says otherwise. An unplaced component stays on the
  CPU, so the pipeline encodes and draws noise on the CPU. **Every Klein component needs
  an explicit `device`.** `residency: on_demand` on the text encoder breaks for the same
  reason: the encoder step moves `input_ids` to the execution device before the wrapper
  moves the weights. On the VAE it is fine once the transformer is placed.
- bf16 with everything resident fits a 24 GB card at 1024², just. Larger outputs and
  edits (references add tokens) want the VAE on demand or the components manager.

## Templates (`workflows/templates/flux3/`)

One idea each, in reading order; every setting from the audit's *Vendor recommended
settings*:

1. `text-to-image.json` - distilled, 16 steps, no guidance, 1024², `width`/`height`
   modulus-16 constraints (the pipeline raises after load), prompt is a full caption
   (stored prompt = the vendor's `man_dog` upsampled caption).
2. `upsample-then-generate.json` - pup `t2i` step -> `previous_result:` prompt;
   `release_models` between them.
3. `image-edit.json` - one reference; the `i2i` upsampler sees the same image; output
   defaults to 1 MP at the reference's aspect ratio (diffusers' rule, not the vendor
   code's - stated in the description).
4. `multi-reference.json` - 2-10 references named `<ref_image_N>` in the caption. The
   10 cap is stated in the description and the skill, not enforced: diffusers does not
   enforce it, and `variable_constraints` are numeric only, so they cannot check a
   list's length. An enforcing list-length check is a follow-up.
5. `layout-boxes.json` - `<id>` caption + 0-1000 box table, written in the box format
   directly as the HF card shows. Upsampled boxes (`t2i_bbox`) need BFL's
   `bbox.plan`/`bbox.reconcile` scaffolding, so they're a follow-up.

No separate 24 GB recipe: P1 showed bf16 with every component placed fits the card at
1024², so the templates above are the 24 GB recipe. SDNQ int8 (the H3 pattern) waits
until a larger size or an edit is measured not to fit.

The upsampler builtins (`builtin:flux3_klein_upsample.json`,
`builtin:flux3_klein_upsample_edit.json`) carry the pinned pup settings once. The edit
one wraps the instruction in BFL's three-line `I2I_SCAFFOLD` (`src/flux3/pup.py` at
bbfe317d) with `compose_text`. BFL's presence penalty (1.5 for text-to-image) has no
transformers `generate()` equivalent and is left out.

Plus `workflows/models/flux3-klein.json` and the family README.

## Engine work (generic, no per-model Python)

`text_generation` (`dw/tasks/text_generation.py`) loads and runs the pup as it is (P2). It
needs two things to run it as BFL specifies:

1. **`chat_template_kwargs`** passed to the chat template (`enable_thinking: false`).
   Without it the pup runs in thinking mode (P2b).
2. **A list of images** - an edit sends the upsampler every reference, in order.

`repetition_penalty` is already overridable (1.15 is only the vision default), and
`generate_kwargs` already passes through, so the vendor's greedy decoding needs no
change: the edit templates set `repetition_penalty: 1.0`. Built later: `generate_kwargs`
is filtered like `chat_template_kwargs` (scalars or lists of them, none of the call's own
keys), and `custom_generate`/`trust_remote_code` need trusted workflows, since
transformers runs a `custom_generate` repo's code.

## Decision: the upsampler's system prompts are read from the Hub (Don, 2026-10-15)

BFL's four system prompts (`t2i`, `i2i`, `t2i_bbox`, `i2i_bbox`) are in the pup repo's
`system_prompts.json` and still changing (pup head 3f3edb52; diffusers PR #15058, open,
copies an older 8b069751). The skill rule is to point at vendor text, not copy it, so
nothing is transcribed. Considered and not taken: a pinned copy as stored prompts (drifts),
and waiting for #15058 (no upsampler templates until it merges).

`text_generation` takes `system_prompt` as today, or a generic reference to a JSON file in
a Hub repo:

```json
"system_prompt": {"from_hub": {"repo": "black-forest-labs/flux-3-image-klein-pup",
                               "file": "system_prompts.json",
                               "revision": "3f3edb52b2c55de0d49feab28598b68f7f8e4830",
                               "key": "t2i"}}
```

- Not "one download, then offline": before every use the Hub is asked, anonymously
  (`get_hf_file_metadata`, `token=False`), whether it serves the file publicly at that
  commit, and any error refuses the run rather than falling back to the cache. So it
  fails closed offline or under `HF_HUB_OFFLINE`. Without that ask, a full commit would
  be answered from the local cache, where files the server fetched with its own login
  (gated or private repos) would be readable by any workflow.
- The download is public-only (`hf_hub_download(..., token=False)`) and capped at
  1 MiB: the Hub's reported size is checked first (no reported size is a refusal),
  and the downloaded file again before it is read.
- `revision` is required and must be a full 40-hex commit, so the text a template runs
  can't move under it.
- `repo` matches the Hub id pattern, and `file` is a relative path with no `..` or
  absolute part. `key` is optional; the value it selects (or the file, without one) must
  be a string.
- Model-agnostic: any vendor that ships prompts as a repo file can use it. When #15058
  merges, the skill can cite the diffusers constant beside it. A test compares the
  constant with the file only if it pins the same revision.

## Follow-ups (dated 2026-10-15)

- Base + CFG: modular `guider` support (`pipe.guider.new(guidance_scale=...)`).
- Vendor fp8r/NVFP4 via `Flux3Transformer2DModel.from_single_file` + `BFLQuantizationConfig`;
  fused only on Ada+/Blackwell, so of little use on the 3090s.
- LoRA on distilled (scale 0.75), typography template, MPS measurement on mini-ai.
- Re-pin the system prompts when #15058 merges.
- dw gap: `validate_workflow`'s `plan.downloads_required` was empty for both uncached
  repos (the modular Klein repo, ~20 GB, and the pup as a `text_generation`
  `model_name`, 18.8 GB), so the cost quote misses the download. Once a partial
  cache existed it swung the other way: 40.3 GB, the whole repo including the
  `variants/` the pipeline never loads.
- Enforce the multi-reference 10 cap with a list-length check (`variable_constraints`
  are numeric only).
- `generate_kwargs` sweep of the other tasks: `text_generation` now refuses
  `custom_generate`/`trust_remote_code` untrusted and filters the call's own keys and
  non-scalar values. `speech_generation` still passes a workflow's `generate_kwargs`
  (`dw/tasks/speech_generation.py:219`) and `forward_params` (`:190`, `:218`) to the
  transformers pipeline unfiltered, and the controlnet_aux detector handlers splat
  workflow kwargs into the detector call (`dw/tasks/image_utils.py:103`, `:114`, `:124`).
  The repo owner decides that fix.
- `revision` validation parity (M7, deferred): `text_generation`'s model `revision` is
  passed to the loader as given, while `system_prompt.from_hub.revision` must be a full
  commit; whether the model revision gets the same pin rule is open.
