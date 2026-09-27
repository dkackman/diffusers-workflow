# H3 Ref2VA VRAM ceiling: references, `for_each` members and hand-built workflows (#479)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-09-27). Plan v1 was designed on #479 and approved with its defaults. It was
built as stages #501 (A) and #502 (B).

## The report

This was a field report from about 7 h of H3 Ref2VA shots on lem (RTX 3090,
24 GB, 64 GB host). Two CUDA OOMs and one host SIGKILL all got past
`validate_workflow`. At 1344x768, int4, with the 768p turbo LoRA and 9 steps:

| frames | refs | result |
|---|---|---|
| 175 | 2 | ok |
| 209 | 2 | ok |
| 175 | 3 | ok |
| 209 | 3 | OOM in reference encode (+5.1 GB refused) |
| 260 | 1 | OOM at denoise step 1 (+4.8 GB refused) |

Two gaps let these through:
- The old formula (`16.3 GB + 28.71 B x w x h x frames`) had no reference term.
- The check ran only on the raw top-level definition. A `for_each` list made
  `required_gb` return `None`, so the check was skipped silently. A workflow
  that declared no `vram_estimate` (the report's inline shots) was never
  checked at all.

## Verdict: build smaller

The design went ahead with the per-reference term, a per-step check after
expansion, and inheritance by pipeline identity. The host-memory half was cut.

There were two corrections to the triage:
- **No engine registry.** An engine-side registry of pipeline costs would have
  put model knowledge in engine code. Instead, the index is derived from the
  catalog's own declarations, and the engine holds no numbers.
- **~29 GB is a lower bound, not a measured peak.** The report's "+5.1 GB" is
  the size of the allocation that was refused.

## What was built

### Stage A (#501): a per-step, post-expansion ceiling with a reference term

- **Where the check runs.** `vram_estimate_errors` runs on the expanded
  definition in `validation_errors`. The run-time backstop `apply_vram_estimate`
  runs after `expand_for_each` in `_prepare_definition`, including for a
  workflow with no `variables` block.
- **Voxel values** are read from each pipeline step's own arguments first, then
  from the workflow's variables (the caller's arguments over the defaults).
- **New optional schema field `vram_estimate.gb_per_reference`.** The term
  counts each `references` entry whose `from_file`/`from_previous_result` is
  non-null (#478's null rule). Every kind of reference counts once.
- **One error per exceeded cost entry, for the largest step only.** For a
  `for_each` member, the error sits at `arguments.shots[i]` when the caller
  supplied `shots`, `variables.shots[i]` otherwise, and `steps[k].for_each[i]`
  for a literal list. The message names the member (`Member 'shot@<name>':`)
  and its reference count.
- **Calibration** of every Ref2VA template: `base_gb` 16.0,
  `bytes_per_voxel` 28.71, `gb_per_reference` 1.0, against the 24 GB entry.
  At 1344x768:

  | shot | projected | verdict |
  |---|---|---|
  | 209x2 | 23.77 GB | valid |
  | 175x3 | 23.83 GB | valid |
  | 209x3 | 24.77 GB | refused |
  | 260x1 | 24.18 GB | refused |

  This puts the 1344x768 ceiling at 243/209/175/141 frames for 1/2/3/4
  references. `minimax-h3` SKILL.md states these figures, pinned from the
  template block in `tests/test_plugin_skills.py`.
- **Templates.**
  - Recalibrated: `reference-to-video` and `storyboard`.
  - Newly declared: `dialogue-short`, `music-video`, `composable-references`,
    `generated-subject-reference`, `voice-timbre-reference` and the three
    `chain-*` Ref2VA templates.
  - Unchanged: T2VA and FL2VA.
- **Tests:** `tests/test_vram_estimate.py` and `tests/test_h3_vram_ceiling.py`.
- **Docs:**
  - the minimax README;
  - the SKILL;
  - the CLAUDE.md gotcha;
  - `docs/RECIPES_24GB.md`.

### Stage B (#502): inherit the ceiling by pipeline identity, as a warning

- **The index.** `dw/vram_inheritance.py` `build_index` maps
  `(component_type, model_name, workflow)` to
  `{vram_estimate, cost, template}`. The server's `_ceiling_index(ws)` caches
  it against the listing's mtimes. `POST /api/validate` and pre-queue
  (`POST /api/jobs`) use it.
- **Matching.** A workflow with no `vram_estimate` has each expanded pipeline
  step matched by identity and projected with stage A's code. Over the ceiling,
  it gets a `vram_projection_inherited` warning that names the source template
  and says the offload/quantization config may differ. It is never a refusal.
  A workflow's own declaration always wins.
- **Sources today:**
  - `ref2va`: `templates/minimax/reference-to-video`;
  - `t2va`: `templates/minimax/shots-batch`;
  - LTX-2.5: `templates/ltx2/text-to-video`.
- **Agreement test.** `tests/test_vram_inheritance.py` pins every template
  declaring one identity to the same numbers.
- **Docs:**
  - the MCP instructions (now naming the inherited source and that config may
    differ);
  - `docs/WORKFLOW_GUIDE.md`'s authoring section;
  - CLAUDE.md.

## Deviations from the plan

- The `reason` strings cite the report's table rather than job ids, since the
  report carried none.
- Six Ref2VA templates declare `vram_estimate` but have no `cost` entry, so
  their block is inert: the three `chain-*` templates,
  `composable-references`, `generated-subject-reference` and
  `voice-timbre-reference`. `cost` is never derived.
- Validation projects an off-grid frame count at its raw value, before
  `snap: "up"` rounding.
- Only single-identity templates with a non-empty `cost` feed the index. Ties
  go to the fewest steps, then the name.
- The warning kind is embedded in the warning string.
- `POST /api/jobs/{id}/rerun` is not wired to inheritance, since a rerun
  re-submits a job that already passed pre-queue.

## Deferred

- **The host-memory half** (the 311x3 SIGKILL at 960x544). #243's host
  projection is observed-only and keyed on the catalog name, so an inline
  workflow has no history to read.
  - **Reopen trigger:** a second host SIGKILL on an inline workflow, or #243
    gaining a pipeline-keyed source.
- **Re-checking a composed child** at the composing step's arguments. This is
  a separate gap and was not filed.
- **Calibration** rests on one box and four points, with about 0.2 GB of
  margin either side. Measurement runs that bracket the line would sharpen it.
  A video reference may cost more than 1.0 GB, but nothing measured it.
- **MPS:** the numbers stay CUDA-measured.

## Stages, cost and bounces

| stage | shipped | bounces |
|---|---|---|
| A #501 | `develop` @ `bb23147` | 0 (M-F043 to M-F048 passed first time) |
| B #502 | `develop` @ `1a94551`, fix @ `156a948` | 1: M-F054, the MCP instructions sentence named neither the source template nor that config may differ; docs-only fix |

The stage comments name no `usage:` figures, so cost is not recorded. The
plan estimated $9 to $13.

Acceptance cases: M-F043 to M-F054 in `regression-suite-model-specific.md`.
The tester found an arithmetic error in M-F049's 260-frame step (the text said
1 ref, but the inline copy carries 3) and filed an amendment on harnest.
