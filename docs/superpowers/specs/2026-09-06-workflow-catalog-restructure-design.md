# Workflow catalog restructure: design

Date: 2026-09-06. Status: approved for planning. Grew out of the MCP capability
discovery discussion that followed
[docs/proposals/scripted-dialogue-and-tts-complete.md](../../proposals/scripted-dialogue-and-tts-complete.md),
whose `generate_speech` half is already implemented.

## Goal

Make the workflow catalog answerable to an open-ended request. An agent handed
"a lego movie trailer set in the marvel universe" has to decide the *shape* of
the deliverable - a multi-shot cut sequence - and find the workflows that make
that shape. Today it faces 121 undifferentiated entries in which the 8 that
matter are indistinguishable from 100 iterations on a checkpoint.

The fix is not to prune. It is to separate two artifacts that have been sharing
one flat list, and to let the catalog say which is which.

## Decisions

| Question | Decision |
| --- | --- |
| Structure | Separate trees: `workflows/templates/` and `workflows/models/`. |
| Pruned workflows | Deleted. Git history is the archive; `archive/` stops being a graveyard. |
| Collapsing model variation | Into variables where the pipeline class matches; stop where component configs genuinely differ. |
| `archive/` | Pruned to the generalizable few, which graduate into `templates/`. The rest is deleted. |
| Workflow `id` | Stays stable across a move, so the step cache survives. |
| Backward compatibility | Not maintained. Existing outputs and any reference into them may break. |
| Acceptance criterion | The post-state is internally consistent: every reference in the repo resolves. |
| Sequencing | The keep/collapse/delete inventory is agreed with the user *before* anything is deleted. |
| Out of scope | The derived shape index and measured runtimes (a separate piece, built on top of this - see "What this unblocks"). |

## What is wrong today

Revised 2026-09-06 after reading all 121 workflows. The first version of this
section argued from a crude feature signature and overstated the problem; what
follows is what reading found.

1. **`archive/` (37 files) is a model gallery and nothing else.** Almost every
   entry is one line - "Text-to-video with Allegro", "Text-to-image with
   Kolors", "Text-to-video with CogVideoX 5B". Four CogVideoX, six Wan, three
   Kandinsky, three Hunyuan. This is where the redundancy actually is.
2. **The `minimax/` (18) and `ltx2/` (8) families are not redundant at all.**
   They are a curriculum: each file demonstrates a conditioning mode or
   mechanism nothing else does - first-frame, last-frame and both-ends
   conditioning; `match_audio` sizing a chain from a supplied track;
   `last_segment` handing over a video tail rather than one frame; composable
   reference types. Under this spec's own criteria they are all templates, and
   they stay (user's decision, 2026-09-06: these are current models and of
   interest to people).
3. The genuine non-archive redundancy is narrower and specific: `flux/`
   acceleration variants (`FluxDevTeaCache`, `FluxDevFirstBlockCache`,
   `FluxDevFast` - three ways to speed up one model), `FluxGGUF`/`FluxTorchAO`
   (two quantization backends), and the one-step text-to-image set spread across
   `FluxDev`, `Krea2`, `ZImage`, `Flux2Dev` and `sd15`.
4. **`gyre/` (8) is a test project's production history, not examples.**
   `GyreStillsFix2` is described as "Third attempt at GYRE's second shot". The
   mechanisms in it that generalize - cross-dissolves, an edit-only assembly
   pass, `recenter_crop` registration - are worth templates; the project-specific
   passes and one-off repairs are not (user's decision, 2026-09-06).
5. Nothing in a catalog entry distinguishes "this teaches a pattern" from "this
   records a hardware fact", so an agent cannot weight them differently and a
   person browsing the repo cannot either. This is the problem the restructure
   is actually for, and it is unaffected by the correction above.

**Two ids are used twice**, which collides in the step cache (keyed on
`(workflow id, step name)`): `sd35` for both `archive/bnb_quant.json` and
`archive/sd35.json`, and `test_job` for both `archive/sd15.json` and
`workflows/sd15.json`. Emptying `archive/` resolves both - the restructure fixes
this rather than causing it.

### Expected scale

Roughly 45-50 deletions, most of them in `archive/`, leaving around 70-75
workflows. Not the halving the first draft of this spec implied. The change is
worth making for the structure it creates, not for the volume it removes.

## The distinction the structure encodes

**A template teaches a pattern.** Its value is in being read and copied, so it
wants to be general, commented, and *few* - one per capability.

**A model config records a fact.** The quantization, `group_offload` and
`residency: on_demand` block that makes MiniMax H3 fit in 24 GB is not
derivable, not generalizable, and worth keeping runnable. But nobody learns
anything from the fifth one.

Pruning alone would fix the noise and throw away the facts. Separating keeps
both and makes each findable as what it is.

## Structure

```
workflows/
  templates/   one file per capability; model as a variable where the
               pipeline class allows; commented
  models/      tuned per-checkpoint configurations; reference data, each
               naming the template it configures
```

`archive/` is emptied. Its generalizable few - `txt2img2vid` (two-stage chain),
`sdxl_refiner` (the refiner pattern), `ip-adapter`, `qr_code` (still linked from
TASKS.md) - graduate into `templates/`. Everything else is deleted.

### What earns a slot

A **template** demonstrates a feature nothing else demonstrates:

- a shape - multi-shot cut sequence, chained conditioning, image-to-video
- a mechanism - shared components, sub-workflows, `pipeline_reference`,
  `references`, `release_pipeline`/`release_models`
- a reference convention - `prompt:`, `asset:`, `output:`, `previous_result:`

A **model config** carries settings that make a specific checkpoint fit real
hardware: quantization, offload, component placement, truncation, cache.

Anything that is a third checkpoint through an existing pattern is deleted.

### Collapsing

Where the pipeline class matches, one template takes the model as a variable
plus one argument set per checkpoint, written into the template's own
`description` so it travels with the file and reaches `list_workflows` - not
into a doc that drifts from it. `FluxDev`, `FluxSchnell` and `Krea2` become one
text-to-image template.

It stops where component configs genuinely differ. The MiniMax H3 files share a
pipeline class but not their quantization blocks, so they keep separate model
configs under one template. Collapsing those would produce a file that is
harder to read than the ones it replaced, which is the opposite of the point.

## Compatibility, and what replaces it

Backward compatibility is explicitly not a goal (user's decision, 2026-09-06).
Existing outputs may be orphaned and references into them may break. No alias
mechanism, no migration shim, no old->new mapping document - each would be a
permanent feature carried for a one-time move.

What replaces it is a stricter obligation on the *end* state: **when this lands,
every reference in the repo resolves.** Not "mostly", and not "the ones a test
happens to cover" - the point of the restructure is a catalog that can be
trusted, and a catalog with a dangling reference in it cannot be.

### The reference classes, and what enforces each

Counted on 2026-09-06:

| Class | Count | Enforced by |
| --- | --- | --- |
| Markdown links into `workflows/` (docs/ + README) | 66 distinct paths | `tests/test_docs_links.py`, which walks every `.md` in the repo |
| Workflow naming another workflow by `path` | 6 files | `tests/test_examples.py::test_example_workflow_references_resolve` |
| `prompt:` references into the prompt library | 18 distinct names across 20 files | **nothing - a gap this work closes** |
| `asset:` and `output:` references | none in `workflows/` | moot |
| UI source referencing a workflow path | none | moot |

The `prompt:` gap matters here specifically because collapsing rewrites and
merges the files that carry those references, and a `prompt:` naming something
that no longer exists fails only when the workflow is actually run. A test
asserting every `prompt:` reference in the tree resolves against `prompts/` is
part of this work, and is worth having independently of the restructure.

### The one thing that must not change

The workflow `id`. The step cache is keyed on it (`dw/step_cache.py`), not on
the path, so every surviving workflow keeps the `id` it has even where its
filename changes. This is a constraint on the restructure, not an observation:
renaming ids would silently invalidate the cache for every workflow at once.

Run-directory identity, by contrast, is derived from the file's path under a
`workflows/` tree (`workflow_identity`, `dw/runs.py:246`) and therefore does
change. That is the accepted break.

### Relative paths inside workflows

A `path` resolves against the referencing file's own directory, so a move has to
be applied to both files together. Schema validation does not follow those
references and nothing else does until the workflow runs - which is why the test
above is the thing standing between a move and a silent break.

## Verification

- every surviving file validates - `tests/test_examples.py` already walks the
  tree and loads each one
- every doc link resolves - `tests/test_docs_links.py`
- a new test for the invariant this exists to create: every template carries a
  description, and every model config names the template it configures
- a new test that every `prompt:` reference in `workflows/` resolves against
  `prompts/` - the one reference class nothing currently checks

The listing needs no change to distinguish the two: a catalog name already
carries its path, so `templates/...` and `models/...` are self-describing.
Anything beyond that - grouping, filtering, ranking templates first - belongs to
the follow-on piece.

## Sequence

1. **Inventory.** Read all 121 workflows and classify each: template, model
   config, collapse-into, or delete. Produce the list.
2. **Agree the list with the user.** Nothing is deleted before this. The
   inventory is a judgement call across 121 files and is the step most likely to
   throw away something worth keeping.
3. **Move and collapse.** Apply the agreed list, keeping `id`s stable and moving
   path-referencing pairs together.
4. **Fix links and docs.** All 66, plus any prose describing the old layout.
5. **Prove consistency.** Every reference class in the table above resolves,
   including the `prompt:` one under its new test. This is the acceptance
   criterion, and the point at which the work is done.

## What this unblocks

The derived shape index and measured runtimes - the follow-on piece - should be
built *after* this, not before. Two reasons: classifying a catalog about to be
halved is wasted work, and the derivation gets materially more accurate once
"template or model config" is a distinction it can read rather than infer.

That piece is now specified in [docs/proposals/catalog-shape-index.md](../../proposals/catalog-shape-index.md). In outline, as agreed here: shape derived from the definition
in `workflow_details` (`dw/server/app.py:154`, which already parses and caches
per file) with an optional declared override where the derivation is wrong; and
median observed runtime per workflow from the `jobs` table, which carries
`workflow`, `status`, `started_at` and `finished_at`. One wrinkle to resolve
then: that column stores the loaded workflow name, which may not match the
catalog name the listing keys on.
