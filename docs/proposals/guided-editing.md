# Proposal: guided image editing (masks, boxes, sketches, regions)

Status: draft / options only — no implementation. Written 2026-10-16.

## Problem

Today a dw edit is steered by text, by a mask file the user already has, or by
a mask a step derives (`segment`, `add_border_and_mask`, a detector). A user
cannot point at the picture: paint the area to repaint, drag a box where a
thing should go, extend the canvas, sketch an outline for the model to
finish, or give one region its own prompt.

The ask is to let them do that, in Krita, in InvokeAI, or in a simple editor
of our own, and have dw run the matching inpaint, outpaint, ControlNet,
sketch-completion or regional edit.

Every client option below needs the same engine work underneath, so this doc
separates that shared layer from the choice of client.

## What the models accept

What the models accept decides what any client can usefully send. Sources are
listed at the end; "main" means diffusers git main, not v0.41.0.

| Family | Hard mask | Control image | Boxes / layout | Sketch | Notes |
|---|---|---|---|---|---|
| FLUX 3 Klein (main, modular only) | no | no | yes, in the caption (0-1000 grid; soft: they guide placement and scale, not pixels) | only as a reference image the caption names | up to 10 references; box edits move/remove/anchor elements |
| FLUX.2 Klein | `Flux2KleinInpaintPipeline` | no | no | via `image_reference` | white = repaint |
| FLUX.1 | Fill, Inpaint, Control-Inpaint, ControlNet-Inpainting, Kontext-Inpaint | Canny/Depth (Control), ControlNet | no | via a scribble/lineart control | dw ships Fill and Canny today |
| Qwen-Image-2.1 | no `mask_image`; masks, circles and painted marks go in as extra images the prompt names | no | no | yes, as `image` (Turbo card) | the exact mask convention is unpublished |
| Qwen-Image (orig.) / Edit-2509/2511 | `QwenImageInpaint`, `QwenImageEditInpaint`, `QwenImageControlNetInpaint` | `QwenImageControlNet`; Edit-2509 takes depth/edge/pose as an input image | no | Edit-2509: yes | |
| Z-Image | `ZImageInpaint`, `ZImageControlNetInpaint` | `ZImageControlNet` | no | via control | |
| SDXL | Inpaint, ControlNet-Inpaint, ControlNet-Union-Inpaint | ControlNet-Union (many modes) | no | scribble/lineart union modes | the most complete control matrix |

Three conclusions:

- **Mask means white = repaint, black = keep** in every diffusers docstring.
  Outpainting is padding plus a white border mask; no pipeline has a mode for it.
- **Masks are not universal.** The newest families (FLUX 3, Qwen-2.1) take
  marks as *images described in the prompt*, not as a `mask_image`. A client
  that only knows "image + mask" cannot reach them well.
- **Regional prompting is not in any of these diffusers pipelines.** Krita AI
  Diffusion gets it from ComfyUI attention-coupling nodes. dw would have to
  build it, or settle for the cheap approximation (one inpaint pass per region,
  each with its own prompt). The one exception is FLUX 3's per-box `desc`,
  which is genuinely regional, but soft.

## The shared engine layer (needed by every option)

What exists: Fill inpaint and outpaint templates, Canny control, SD1.5
ControlNet, `segment` (text → mask), the `controlnet_aux` detectors,
`add_border_and_mask`, and `previous_result:step.key` addressing. Gaps:

1. **An edit bundle.** One description of "this base image, these masks,
   these control layers, these boxes/regions with prompts", so templates do
   not each grow a variable per layer. Proposed: a JSON sidecar plus PNGs under
   one asset folder (`asset:edits/<name>/`), e.g.
   ```json
   {"base": "base.png",
    "mask": "mask.png",
    "controls": [{"kind": "scribble", "image": "sketch.png", "strength": 0.8}],
    "regions": [{"mask": "r1.png", "bbox": [120, 80, 560, 400], "prompt": "a red kite"}]}
   ```
   A task reads the bundle and exposes each part as a keyed result
   (`previous_result:bundle.mask`), so existing templates take bundle parts
   with no new reference syntax.
2. **Mask tasks:** alpha → mask, colour → mask (strokes in a known colour),
   boxes → mask, grow / feather / invert as their own step, crop to the mask's
   bounding box plus padding, and paste back with a feathered blend (the
   "only the masked area" pattern every serious inpaint UI uses).
   `fetch_image` flattens to RGB unless `keep_alpha`, so alpha → mask needs
   the alpha-keeping loader.
3. **Per-family adapters.** One builtin per family turns a bundle into that
   family's inputs: `mask_image` for Fill, Z-Image or SDXL; `control_image`
   plus `controlnet_conditioning_scale` for the ControlNets; reference images
   plus a caption that names them for FLUX 3 and Qwen-2.1; and the FLUX 3 box
   JSON written from the bundle's regions.
4. **Templates.** Inpaint, outpaint, sketch-to-image, ControlNet-inpaint and
   region edit, for at least Z-Image, SDXL and FLUX.2 Klein (hard masks),
   plus a FLUX 3 box-layout edit from bundle regions (the template exists;
   it needs the bundle input). Regional prompting starts as sequential
   per-region inpaint; real attention-coupled regions are a separate,
   later proposal.
5. **Catalog and security.** `catalog_shape.py` should count `control_image`
   as `image-edit` and add a `needs-mask` trait. The upload allowlist
   (`dw/security.py:29`) needs `.ora` and `.invk` if D is chosen, unpacked
   through the path validators with zip-slip and size caps.

**Effort:** M, about 1-2 weeks of agent work including lem GPU checks per
template. **Risk:** low. It is ordinary task and template work in established
patterns, and it pays off without any client: an agent over MCP can already
make a mask from boxes or a segment and run the edit.

## Client options

### A. Pose as Krita AI Diffusion's cloud backend

Krita AI Diffusion (Acly, v1.53) has an abstract `Client`. Its `CloudClient`
posts a backend-neutral, typed `WorkflowInput` to `{INTERSTICE_URL}/generate`
and polls `status/{id}`. That input carries the base image, the selection
mask, inpaint parameters (grow, feather, blend, target bounds), control layers
(17 modes: scribble, line art, depth, pose and more), and regions, each with
its own mask, prompt, controls and LoRAs. dw would implement those routes plus
the auth handshake, and translate `WorkflowInput` into a dw workflow.

- **Pros:** the best painting UI available, with layers, selections, control
  layers, regions and live preview, for no UI work. The payload is already the
  edit bundle, typed.
- **Cons:** the contract is undocumented and meant for one vendor's service;
  any plugin release can change it. Models are chosen from the plugin's `Arch`
  enum (flux2_4b/9b, qwen, zimage, …), so a dw family the plugin does not know
  (FLUX 3 today) cannot be picked. The auth flow has to be faked. Users must
  point an env var at dw.
- **Effort:** L. **Risk:** high, both breakage and maintenance.

(Emulating ComfyUI itself, its `/prompt` graph, websocket and the ETN custom
nodes, is the other way into this plugin. It is XL and is not considered further.)

### B. Our own Krita plugin

A small Python plugin, a Krita docker panel. You paint in ordinary layers
named by convention (`mask`, `control:scribble`, `region: a red kite`); the
selection can serve as the mask. Pick a dw workflow, then Run: the plugin
exports the layers as a bundle, uploads it with `POST /api/uploads`, runs the
workflow, polls the job and inserts the result as a new layer.

- **Pros:** Krita's full brush engine (pressure, selections, layers), our
  contract, so every dw template and family is reachable, including FLUX 3
  boxes and Qwen-2.1 annotations. Works against a remote server through the
  existing HTTP API and REMOTE.md auth.
- **Cons:** a second codebase in another runtime (Krita's PyQt, with Qt 5 to 6
  churn in Krita 6). Distribution and installs are on us. Testing needs Krita
  or a stubbed `krita` module. A thinner UI than A: no live preview, no region
  editor beyond layer names.
- **Effort:** M (plugin about 1 week on top of the engine layer). **Risk:**
  medium, mainly Krita API and version maintenance.

### C. OpenAI-compatible images endpoint for InvokeAI

InvokeAI (v6.14) has built-in "external" providers. Its OpenAI adapter posts
to `{base_url}/v1/images/generations` and a multipart `/v1/images/edits` with
`image`/`image[]` and a `mask` PNG. dw would add those two routes and map an
external model id onto a dw template.

- **Pros:** the smallest option. Stock InvokeAI with no fork, and any other
  OpenAI-images client (scripts, other apps) gets it too.
- **Cons:** only base image, mask and references cross the boundary. InvokeAI's
  control layers, regional guidance and boxes stay behind. The mask convention
  needs checking: OpenAI's is "transparent = edit", diffusers' is "white =
  edit", and whether InvokeAI converts is untested. Synchronous request
  semantics fit dw's queued jobs badly; long runs mean long-held HTTP requests.
- **Effort:** S (a few days). **Risk:** low to build, but capped in value.

### D. File round-trip (`.invk` / `.ora` import)

The user exports from the painting app and uploads the file; a dw task unpacks
it into a bundle and the result goes back as a file. InvokeAI canvas projects
(`.invk` v1) are zips holding `canvas_state.json` (raster layers, control
layers, inpaint masks, regional guidance, bbox) plus images. Krita exports
OpenRaster (`.ora`: a zip of PNGs plus `stack.xml`), a stable standard.

- **Pros:** carries everything both apps know, regions included, with no live
  integration to maintain. `.ora` is a stable format. A good way to prove the
  engine layer on real painted input.
- **Cons:** no loop. Export, upload, run, download and re-import by hand gets
  tedious fast. `.invk` is versioned (v2 is already the Workbench format), so
  expect churn. Layer meaning has to come from naming conventions in `.ora`.
- **Effort:** S-M. **Risk:** low (`.ora`), medium (`.invk`).

### E. Our own simple editor in the dw web UI

A canvas in `ui/`, on the template/run page wherever a template takes a
bundle. Load the base image (an upload, an asset or a gallery output), then:
a mask brush and eraser with size and feather, a box tool with a prompt per
box, outpaint handles that drag the canvas edges out, a sketch layer, and
pick-a-detector to make a control layer from the base image. Saving writes
the bundle through the existing upload route, and Run feeds it to the
template. The result can be opened straight back in the editor for the next
pass.

- **Pros:** no install. It works from any browser against lem or mini-ai,
  including from a tablet with a pen (Pointer Events carry pressure). One
  codebase, our contract, every family reachable. Fits the existing
  run → gallery → keep loop, and edits stay in the workspace's history. An
  agent can produce the same bundle over MCP, so UI and agent share a format.
- **Cons:** basic tools next to Krita: no brush engine, limited layers, no
  selections beyond what we build. The UI ratchet (600-line files) pushes it
  into several components. Touch/pen ergonomics need care. It can creep toward
  "rebuild Krita"; scope has to stay at mask, box, sketch and extend.
- **Effort:** M (about 1-1.5 weeks for brush/eraser, boxes with prompts,
  outpaint extend, sketch layer and bundle save). **Risk:** low to medium,
  mostly UX and scope rather than technical.

## Comparison

| | Engine layer | A Krita AI Diff. | B Krita plugin | C InvokeAI OpenAI | D file import | E dw editor |
|---|---|---|---|---|---|---|
| Masks | — | yes | yes | yes | yes | yes |
| Control layers | — | yes | yes | no | yes | detector + sketch |
| Boxes / regions | — | yes | layer names | no | yes | yes |
| All dw families | — | no (`Arch` enum) | yes | via model-id map | yes | yes |
| Live loop | — | yes | yes | yes | no | yes |
| Install for user | — | env var | plugin zip | provider config | none | none |
| We maintain | tasks, templates | an emulated API | a Krita plugin | 2 routes | 1-2 parsers | UI components |
| Effort | M | L | M | S | S-M | M |
| Risk | low | high | medium | low (capped value) | low/medium | low-medium |

## Recommendation (for when this is picked up)

1. **The engine layer first.** Every option needs it, and it is useful by
   itself through MCP. Prove it with `.ora` import (the cheap half of D),
   which gives real painted input without building a client.
2. **Then E, the dw editor,** as the default way in. It is the only option that
   works from anywhere with nothing to install, and it keeps one contract.
3. **B if E's tools prove too thin** for serious painting. It reuses the
   bundle and the API unchanged.
4. **C as an opportunistic extra**, because it is cheap and helps any
   OpenAI-images client. **A only if** Krita AI Diffusion ever documents its
   backend API for third parties.

## Open questions

- Regional prompting: is sequential per-region inpaint good enough, or is
  attention coupling (a custom attention processor per family) worth a
  proposal of its own?
- Qwen-Image-2.1's mask convention (colour? an extra reference? how the
  prompt names it?) is unpublished; it needs a probe on lem.
- Will BFL ship FLUX 3 Fill, inpaint or control models? If so, FLUX 3 moves
  from soft boxes to the hard-mask column.
- Does InvokeAI's OpenAI adapter convert its mask to OpenAI's
  transparent-means-edit convention? This decides C's mask handling.
- Where do bundle results land: a new image per pass, or a versioned "edit
  session" asset that keeps every pass's bundle?

## Sources

- FLUX 3 Klein card: https://huggingface.co/black-forest-labs/flux-3-image-klein ;
  box format: https://docs.bfl.ml/flux_3/flux3_image_bounding_boxes ; local audit
  `docs/proposals/audits/2026-10-15-flux-3-klein-audit.md`
- Qwen-Image-2.1: https://huggingface.co/Qwen/Qwen-Image-2.1 ,
  https://github.com/QwenLM/Qwen-Image-2.1 ; Qwen-Image-Edit-2509 card (ControlNet-style inputs)
- diffusers v0.41.0 pipelines: `Flux2KleinInpaintPipeline`, `QwenImage*Inpaint*`,
  `QwenImageControlNet*`, `ZImage{Inpaint,ControlNet,ControlNetInpaint}Pipeline`,
  `StableDiffusionXLControlNetUnion*`, FLUX.1 Fill/Control/ControlNet/Kontext-Inpaint
- Krita AI Diffusion: https://github.com/Acly/krita-ai-diffusion (`ai_diffusion/backend/client.py`,
  `backend/api.py` `WorkflowInput`, `CloudClient`, `INTERSTICE_URL`)
- InvokeAI: https://github.com/invoke-ai/InvokeAI (`services/external_generation/providers`,
  `docs/src/content/docs/development/Front End/canvas-projects.mdx`)
- dw code today: `workflows/templates/{inpaint,outpaint,segment-and-inpaint,controlnet}.json`,
  `dw/tasks/{image_utils,borders,segment}.py`, `dw/argument_media.py:126` (`fetch_image`),
  `dw/server/routes/assets.py:138` (uploads), `dw/server/catalog_shape.py:54-59,233-235`
