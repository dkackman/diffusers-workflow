# Skipping Klein's prompt-upsampler block (#1168)

## Problem

`Flux3DistilledAutoBlocks` starts with a `prompt_upsampler` block
(`Flux3AutoPromptUpsampleStep`, an `AutoPipelineBlocks` with triggers `["image", None]`). The
`None` branch is the default, so it runs on every call: no pipeline input makes it a no-op.
Its only documented skip is `blocks.sub_blocks.pop("prompt_upsampler")`.

Commit 3de1506c stopped loading the 18.8 GB upsampler (it overflowed a 24 GB card), so
`components.prompt_upsampler` and its processor are `None` and every Klein text-to-image
call fails with `'NoneType' object has no attribute 'apply_chat_template'`. Loading it again
would reintroduce the OOM, so the fix has to drop the block.

dw has no syntax for that (`load_components`, `configs`, `components` and `arguments` cannot
touch the block graph), so this is a new concept and is parked.

## Options

1. **Pipeline `configuration.remove_blocks: ["prompt_upsampler"]`** (recommended). After
   the pipeline is built, pop each named top-level `sub_blocks` entry; an unknown name is a
   validation error. Declared in the Klein workflows, so model knowledge stays out of engine
   code. Touches the schema, `dw/pipeline_processors/pipeline.py` and docs.
2. A Klein-specific pipeline processor, like the H3 ones. Smaller surface but puts model
   knowledge in engine code, against CLAUDE.md.
3. Keep loading the upsampler and offload it (CPU offload or quantize). Fits only with
   slower runs, and the upsampler would still rewrite the already-long captions the
   templates pass.

## Affected templates

`flux3/text-to-image`, `image-edit`, `multi-reference`, `layout-boxes`,
`upsample-then-generate` and the `flux3-klein` model entry. The edit templates and
`upsample-then-generate` already upsample in a separate step, so they also double-upsampled
before; popping the block fixes that too.

## Caveat

`layout-boxes` passes the caption through as written; `boxes` as an input for the
in-pipeline upsampler is not used by any template, so popping loses nothing.

## Outcome

Option 1 shipped (#1168): `configuration.remove_blocks`. Validate cannot check a name against the block graph (it does not build the pipeline), so an unknown name is refused when the pipeline loads, before any weights are pulled for the step to use.
