# `remove_object`: prompt-free LaMa object removal (#956)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-10-22). The idea came from #946: on lem an agent asked to erase an
object hand-composited it in PIL, and `models/z-image-inpaint` failed
removal in all four runs. Plan v1 was designed on #956. Don's review made
v2, written after #946 closed `wontfix`, and Don approved v2. It was built
as two stages, #973 and #974. The plan comment on #956 stays the full
design. This doc records what shipped.

## The problem

dw had no "erase this and fill it from the surroundings" operation. Every
masked route was a prompted diffusion pipeline, and those add or replace
content. With #946 declined, FLUX.1 Fill was never verified for removal on
lem, so dw had no working removal route at all.

## Verdict: build smaller

Two small stages: the task, then the routing docs and templates. No new
dependency was needed, because `spandrel` (already a core dependency)
loads LaMa as a `MaskedImageModelDescriptor`.

Cut, with what would bring each back:
- **A temporally consistent video removal model** (ProPainter-style).
  *Trigger:* a field report where per-frame LaMa flickers visibly.
- **A per-frame mask video.** It needs a two-input `per_frame`. *Trigger:*
  a field report that needs a moving mask. The task applies one still mask
  to every frame.
- **A general mask utility task** (dilate/feather/union). *Trigger:* a
  second consumer. The task dilates internally.
- **A second, larger removal model for big regions.** *Trigger:* a field
  report where LaMa smears a large removal.

Non-goals stayed out: prompting, outpainting, temporal consistency,
per-frame masks, and any change to the Fill templates beyond their removal
wording.

## Plan history

- **v1:** Fill stayed the prompted fallback for large regions and the
  outpaint route. The signature had no `weight_name`. It used bf16 where
  the device supported it, and left the weights source open.
- **v2 (approved):** #946 is `wontfix`, so `remove_object` is dw's only
  removal route. Fill was dropped from the fallback and overlap, and
  `templates/outpaint` was left alone. `weight_name` was added, so the
  safetensors-only gate has a key to check. fp32 runs on every device
  (LaMa is ~51M params, and bf16 is an untested quality risk). Q1, ruled
  by Don: pin `michaelgold/big-lama@98ab7cface89a594b381f32fc7a9f2cae3f91ccc`,
  file `big-lama.safetensors` (204 MB), with the byte-identical
  `biali/big-lama` as the named fallback. LaMa (advimman) is credited as
  Apache-2.0, not under the repo card's MIT. Q2 kept the 8 px `dilate`
  default. Q3 accepted the `downloads_required` gap and called for a
  follow-up issue.

Corrections the plan made to the issue: the dependency question went away
(spandrel); the common `big-lama.pt` is TorchScript and must not be loaded;
the H3 backdrop leak in `minimax-h3`'s `references/staging.md` is fixed by
`subject_definitions`, not by a mask; `WORKFLOW_GUIDE.md` had no
masked-editing section to edit, so one was added.

## What was built

### Stage A (#973): the `remove_object` task

Commits 89e75acf (the task) and 8bc5b884 (tests and TASKS.md), merged as
c7a694db.
- `remove_object(image, mask_image, dilate=8, model_name="michaelgold/big-lama",
  weight_name="big-lama.safetensors")`. The implementation is in
  `dw/tasks/remove_object.py`, and the handler is in
  `dw/tasks/model_handlers.py` (lazy import, `per_frame`,
  `consumes_device=True`).
- **Mask:** converted to L, binarized at 128 (white means fill, the same as
  `segment`'s output, so no `invert`), resized to the image with nearest if
  it differs, then dilated. The model's output is composited only inside
  the dilated mask, so pixels outside it are the input's own.
- **All-black mask:** the input comes back unchanged with a job `warning`
  event, and no model is loaded. For frames the check runs once, so a video
  warns once.
- **Weights:** `REMOVER_REVISION` is applied only to the default repo and
  file. The loader runs `validate_hub_repo_id` and `validate_weight_name`,
  then `hf_hub_download`, `safetensors.torch.load_file`, and spandrel
  `load_from_state_dict`. It refuses anything that isn't a
  `MaskedImageModelDescriptor`. The model is cached through `cached_model`
  under `(repo, file, revision, device)`.
- **fp32 everywhere:** no `.half()` and no device branch.
- **Trust gate:** `remove_object` is in `SAFETENSORS_ONLY_COMMANDS`, so a
  `.pt`/`.ckpt`/`.bin` `weight_name` or a local `model_name` is refused at
  validate, and again in the loader.
- `tests/test_remove_object.py` (42 cases). `docs/TASKS.md` gained an
  *Object Removal* section: parameters, polarity, `dilate` advice, the
  weights source and fallback, the Apache-2.0 credit, the large-region
  limit and the first-run download.
- **Deviations:** `image`/`mask_image` are not declared as
  `media_arguments`, because the media-key naming convention already covers
  both and a test pins the traversal refusal. With frames and an all-black
  mask, the frames come back re-wrapped at the same count and fps.
- Measured on mps with the real weights: about 10 s cold including the
  download, and 0.08 s warm.

### Stage B (#974): routing and templates

Commit 4844c1f7 (merged as d1141a66), and the bounce fix in bfb4e222 and
941b5fd7 (merged as 1a6d8b5c).
- `workflows/templates/remove-object.json` (`image`, `mask_image`,
  `dilate`) and `workflows/templates/segment-and-remove.json` (`image`,
  `object`, `dilate`). Both declare `"shape": "image-edit"`, because the
  shape rules would otherwise read a task-only workflow as `utility`. Their
  default image is the diffusers docs `cup.png`. `remove-object`'s default
  mask is `docs/img/cup-object-mask.png`, served from this repo pinned at
  bfb4e222.
- **Routing rule** ("erase → `remove_object`; put something new in a mask →
  a prompted inpainter"): a new *Erasing versus repainting a masked region*
  section in `WORKFLOW_GUIDE.md`'s *Authoring a workflow from an agent*,
  and TASKS.md *Object Segmentation*. The `models/z-image-inpaint`
  description was rewritten so it no longer names Fill for removal. The
  `templates/inpaint` and `segment-and-inpaint` descriptions each gained
  one clause.
- The compact listing budget in `tests/test_catalog_structure.py` went from
  11_400 to 11_650.
- **Outside the plan:** `dw/tasks/segment.py` now keeps each box's
  best-rated SAM2 candidate (`iou_scores` argmax), not the first. On
  `cup.png` the first candidate was the cup's body without its lid, with a
  hole at the logo. This changes the mask every `segment` caller gets
  (including `segment-and-inpaint`). It is a bug fix with no change to the
  MCP shape.

## Verification and bounces

| Stage | Bounces | Cases |
|---|---|---|
| A (#973) | 0 | C-F510–C-F515, SE-F050 passed on lem first time |
| B (#974) | 1 | C-F517 failed: the cup survived. Fixed by the `segment` best-mask change and the new default mask. C-F516–C-F518 then passed |

On lem the warm `segment` → `remove_object` job took 1.6 s, and its
`remove_object` phase 0.3 s. The stage comments name no `usage:` figures,
so cost per stage is not recorded here. The plan estimated $3–5 for A and
$2–3 for B.

## Found along the way

- **segment's mask quality** (above): every `segment` caller used to get
  SAM2's first proposal, which can be a partial object.
- In C-F511, `segment("picture frame")` covered only part of the frame's
  matte, so a faint remnant was left. That is a segment setup shortfall,
  not a `remove_object` finding.

## Follow-up

- **#975** (shipped 2026-10-23): `plan.downloads_required` now lists task
  weights, including `remove_object`'s pinned big-lama file. Record:
  `task-weights-downloads-complete.md`.
- The templates' `vram_gb` figures (1 and 4) are estimates from model
  sizes, not measurements. Replace them when one is measured.
