# Cinematic add-ons (tier 2): re-shoot, angles, speed, relight, dubbing, swap

Source: [audits/2026-11-higgsfield](audits/2026-11-higgsfield/README.md), the section "Add-ons that unlock re-shoot, angles and dubbing".

**Follows:** the tier 1 plan in [../superpowers/plans/2026-11-01-cinematic-controls.md](../superpowers/plans/2026-11-01-cinematic-controls.md).

Every stage below opens with a probe on lem. A stage becomes a template only after a 24 GB proof and a licence read. Each probe is a `spend` decision for Don, and some also need a `trust` decision.

## Stages

**A. LTX-2.5 Slow-Motion-Control and Cinemagraph LoRAs.**
- **Probe:** get the sampling recipe from the cards, then run 2 clips each.
- **If they pass:** two templates under `workflows/templates/ltx2/`, plus `speed` and `cinemagraph` rows in the `cinematography` vocabulary.
- **Decisions:** spend (2 LoRA downloads, about 30 min on lem).

**B. Qwen-Image-Edit-2511 + fal Multiple-Angles LoRA ("Shots").**
- **Checks first:** read the LoRA's licence. Choose a quantization that fits 24 GB, against the 57.7 GB base.
- **If it fits:** a `shots-9-angles` template with `for_each` over 9 fixed angle prompts, whose output feeds H3 FL2VA or LTX keyframes.
- **Decisions:** spend (about 58 GB download), trust (a new family).

**C. LTX-2.5 Layout-To-Render IC-LoRA.**
- **Probe:** does it need the guided dev-transformer route that kept Clean-Plate and Day-To-Night out (`complete/ltx2-ic-loras-complete.md`)?
- **If not:** a blocking-to-shot template (depth video → render).
- **Re-shoot (ReAngle and Comfy-Dial):** depth estimation, warping along a camera path, then guiding the render. This adds a `reproject_depth` task on Video Depth Anything **Small** only, because Base and Large are CC-BY-NC.
- **Decisions:** spend, scope (new task).

**D. LTX-2.3 Relight / Motion-Track / DubIt on 2.5 (probe only).**
- **Plan:** one afternoon of probes, and a record of each verdict. Nothing becomes a template unless a probe passes.
- **Decisions:** spend.

**E. H3 swap recipe (OPENjutsu idea, existing tools).**
- **Recipe:** SAM-style `segment` track, the mask grown 40 px, a grey-mannequin motion reference, then Ref2VA with the new cast.
- **Output:** a skill section plus one template. The BFS, CrossView and DMAD LoRAs only if they load in diffusers.
- **Decisions:** spend.

**F. LatentSync 1.6 dubbing task.**
- **Licence:** Apache-2.0 code, OpenRAIL++ weights, about 18 GB.
- **Output:** a `lip_sync` utility task on an existing clip plus an audio track.
- **Decisions:** trust (new dependency), spend.

**G. 360 location references.**
- **Output:** an `equirect_views` image task that cuts a panorama into overlapping perspective references plus a manifest. No model. The idea comes from 360 Scout (MIT).
- **Decisions:** scope (new task).

## Deferred

**Wan 2.x** as a family: Animate swap, VACE, effect LoRAs, ReCamMaster, InfiniteTalk. Revisit only on a field report asking for an actor swap that stage E can't do.

**The Lightricks camera LoRAs** target LTX-2 19B, not 2.5, so they count only as vocabulary.
