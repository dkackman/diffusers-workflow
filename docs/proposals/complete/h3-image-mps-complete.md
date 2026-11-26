# MiniMax-H3-Image on a 64 GB Mac: the mps probe and the corrected record (#1680)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-26). Plan v4 on #1680 is the full design. This file records what
shipped, what was cut, and why. The measured numbers live in
`bench/records/h3-image.md`, section *mini-ai (mps)*.

**Result:** H3-Image runs on mini-ai (Apple M5 Pro, 64 GB, torch 2.14.1), but
only from pre-quantized int8 folders and in two steps. A 2048x2048 still takes
3.2 min including the load (job `b99e189e1059`). The peak is 34.2 GB of MPS
memory and 40.4 GB of job RSS, leaving 20.5 GB of host memory free. The
`minimax-h3-image` skill, `docs/ACCELERATION.md` and the template README now
state this measured route in place of the unmeasured "CUDA only, ~70 GB" rule.
The catalog templates are unchanged and still CUDA only: validate refuses them
on mini-ai at `host_ram_gb`.

## The problem

The skill said H3-Image is CUDA only ("on `mps` or `cpu`, stop") and needs
~70 GB of host RAM. Nobody had measured that on mps. janishar/helmstudio
reported 112-145 s at 4 MP and a 38.9 GB peak on a 64 GB Mac (torch 2.15, int8,
one model at a time). Those are their figures, not ours.

## Verdict: build smaller

The plan dropped a shipped mps template and kept two stages: a bench-only
probe, then a correction to the record.

**The plan corrected the issue:**
- **"As H3 video on mini-ai already does" had no record in the repo.** It was
  reported in `ACCELERATION.md` and two proposal docs, but there was no
  workflow, bench record or job. The shipped H3 templates use hub ids.
- **The quantized cache can't carry a lem entry over.** Its key includes the
  torch, diffusers, transformers and sdnq versions, and the startup sweep deletes
  entries that don't match. The folders are loaded by path instead: a plain
  `from_pretrained` that the cache never keys.
- **Cold quantize on mini-ai is out.** It peaked at 117 GB on lem. So the
  folders were built on lem (job `3eec967fe939`) and copied over (Q2).
- **One model at a time means two steps.** An `encode` step loads only the text
  encoder and sets `release_pipeline: true`. A `generate` step loads the
  transformer, VAE and small parts, and takes `prompt_embeds` and the modality
  tags by `previous_result:`. dw handed those tensors across without an engine
  change.

## Questions and rulings

- **Q1 [spend]:** about 1 h of mini-ai time. Approved, with a time stop: if the
  cold arm runs past ~20 min, record it and skip the other arms.
- **Q2 [record]:** the folders are lem's int8 entries. The text encoder is
  `MiniMaxAI/MiniMax-H3` `0919fb68` (35.6 GB). The transformer is
  `MiniMaxAI/MiniMax-H3-Image` `transformer_diffusers` `8ced6006` (32 GB).
- **Q3 [none]:** the files are `bench/workflows/h3-image-mps-*.json` and the
  record is `bench/records/h3-image.md`.
- **Q4 [trust]** (added in v3): mini-ai's server runs with
  `trust_workflows: false`, and dw's trust gate (`dw/locations.py`) refused the
  folders on `/Volumes/NVME`. A symlink doesn't help, because the check is on the
  realpath. Don chose the "otherwise" option. He copied both folders (~68 GB)
  into the `bench` workspace's asset library,
  `/Users/don/diffusers-workspace/bench/assets/h3-prequantized-int8/`. Trust
  stayed off and the server was not restarted. The original copy is at
  `/Volumes/NVME/dw-bench/assets/h3-prequantized-int8/`, and keeping or
  deleting it is Don's call.

## What shipped

### Stage A: mps probe, bench only (#1729)

- `bench/workflows/h3-image-mps-t2i.json` and
  `bench/workflows/h3-image-mps-edit3.json`, both saved in mini-ai's `bench`
  workspace. The edit file's three references are `cast/estela.jpg`,
  `cast/aurelio.jpg` and `bench/runner.png`, all of which resolve on mini-ai.
- The `## mini-ai (mps)` section of `bench/records/h3-image.md`.
- **What the probe found.** The memory plan held: the released encode step frees
  its 26.6 GB before the transformer loads, so the two are never resident
  together. Speed failed, though. The cold t2i arm (`872ea59cbdcb`) never
  finished its first denoise step. The transformer folder had been saved on lem
  with `use_quantized_matmul: true`, and a folder's own config bypassed dw's mps
  adaptation, so every int8 linear ran `torch._int_mm` on the CPU. That was filed
  and fixed as **#1744**. The job also never honoured its cancel, because cancel
  waits for a step boundary. #1746 fixes that, but it only took effect once Don
  restarted mini-ai's server.
- The time stop skipped the warm, edit and 16:9 arms.
- **Bounces: 1.** The tester found that the workflows weren't saved in `bench`.
  `save_workflow` refused the asset-folder path until #1739's fix was live, and
  the stuck job blocked the deploy that would have made it live. The tester also
  found that the M-F504 record row disagreed with the job. The lead corrected the
  record (`071e48be`). The delegate passed the restart to Don, who restarted the
  server at 15:38 UTC. After that both workflows saved, and the stage verified at
  develop `bd48f5b7`.

### Stage B: correct the record (#1730)

- Built from the "probe ran" branch, using #1744's verify job `b99e189e1059`.
  That job is the same t2i arm at develop `11bff16d`, after the fix: encode load
  12.0 s and run 2.5 s; generate load 92.2 s, denoise 77.9 s over 6 passes and
  decode 4.0 s; 190.7 s wall. The lead added that job to the bench record. No new
  GPU run was made for this stage.
- `plugins/dw/skills/minimax-h3-image/SKILL.md`, *Before anything*, *Cost* and
  *Quote, then wait*: an mps host under ~70 GB runs H3-Image only through the
  bench workflow, copied into the workspace with its folder paths set, and never
  through the catalog template. The measured numbers are stated, and so is the
  fact that the edit was not measured on mps.
- `docs/ACCELERATION.md`: the H3-Image Apple Silicon cell now reads "runs slowly"
  with the numbers. CPU reads "no record". Both gaps are listed under *Cells with
  no record*.
- `workflows/templates/minimax-h3-image/README.md`: a new *On mps* section.
- `tests/test_plugin_skills.py`:
  `test_mps_runs_from_the_prequantized_bench_workflow_as_measured`. The
  `host_ram_gb == 64` pins are unchanged.
- **Bounces: 0.** Verified at develop `6c1fb691`.

Cost: the stage comments give no `usage:` figures, so none is recorded here.
Stage A used about 30 min of mini-ai time on the failed cold arm. The
successful run was #1744's verify, not this feature's.

## Deferred, with what would bring each back

1. **A shipped mps template.** A catalog template can't name a host-specific
   local folder. It would need a Hub-published pre-quantized repo or a
   quantized-cache import task (new surface). It comes back if Don wants
   H3-Image in mini-ai's catalog.
2. **The two-step split on the CUDA templates**, to lower lem's ~64 GB host
   floor. It comes back with a field report of H3-Image being held for host RAM
   on lem.
3. **int8 quantized matmul on MPS.** This is #1678's work. With matmul off,
   mini-ai dequantizes on use, which gives the 12.7 s per pass above.
4. **The unmeasured mps arms:** a warm rerun, the edit with three references (so
   there is no reference-count limit on mps), and a 16:9 size. Each is a rerun of
   the saved bench workflows, and the edit file is already in place.
5. **An mps rung for H3-Image in the `text_to_image` preference ladder (#1625).**
   It needs a catalog route first (item 1).
