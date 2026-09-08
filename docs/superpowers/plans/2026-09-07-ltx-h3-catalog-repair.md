# LTX-2.5 and MiniMax H3 catalog repair — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the LTX-2.5 two-stage template, the LTX prompt library, the H3 Context-IR builtin and both model READMEs against the two vendor-source audits, with a test holding each repair in place.

**Architecture:** Every change is catalog data: template JSON, stored prompts, one builtin's system prompt, two READMEs and one recipe doc. The two-stage repair uses the latent handoff the engine already has (`previous_result:<step>.<field>` reads attributes off a pipeline output object), so no engine code changes unless Task 1's proof fails. Each task lands with its test and with a line in the proposal's ledger row.

**Tech Stack:** Python 3.10+, pytest, diffusers (installed; `diffusers.pipelines.ltx2.utils` carries the constants tests compare against), the dw catalog under `workflows/` and `prompts/`.

**Spec:** [docs/superpowers/specs/2026-09-07-ltx-h3-catalog-repair-design.md](../specs/2026-09-07-ltx-h3-catalog-repair-design.md). Evidence: [docs/proposals/audits/2026-09-07-ltx-2.5-audit.md](../../proposals/audits/2026-09-07-ltx-2.5-audit.md) and [docs/proposals/audits/2026-09-07-minimax-h3-audit.md](../../proposals/audits/2026-09-07-minimax-h3-audit.md).

## Global Constraints

- No model-specific prompting knowledge in Python: guide prose, template JSON, stored prompts only (proposal, "Principle").
- No `trust_remote_code` or `custom_pipeline` in any bundled entry (`tests/test_catalog_structure.py::test_no_catalog_entry_needs_trust_workflows`).
- Schema validation runs before variable substitution; JSON defaults must be the right JSON type.
- A template's `description` may not mention a variable or step in single quotes that the workflow does not have (`test_catalog_structure.py` drift check).
- Every test file docstring and every test name states the behaviour, not the mechanism, matching the existing suite's style.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Each task appends to the Part 4 row of the ledger table in `docs/proposals/agent-catalog-legibility.md` (the row beginning `| Part 4 packaging |`), one sentence per task, in the same commit as the change.
- Work on branch `catalog-repair`, cut from `master` after PR #49 has merged. Do not start it from `agent-legibility`.

---

## File map

| File | Responsibility | Task |
| --- | --- | --- |
| `tests/test_result.py` | proves `previous_result:step.frames` / `.audio` reach tensors on a latent output | 1 |
| `workflows/templates/ltx2/two-stage.json` | the three-move flow | 2 |
| `tests/test_catalog_structure.py` | `noise_scale` literal equals the library constant; README links resolve | 2, 5 |
| `docs/RECIPES_24GB.md` | two-stage paragraph rewritten, sharpness claim withdrawn | 2 |
| `prompts/ltx2/*.json` (6 files) | trained-format captions | 3 |
| `workflows/templates/ltx2/{chained-segments,enhance-prompt,extend-clip,keyframes}.json` | `summary` says LTX-2.5 | 3 |
| `tests/test_ltx_prompt_library.py` | caption-format invariants | 3 |
| `dw/workflows/h3_context_ir.json` | system prompt corrections | 4 |
| `tests/test_h3_context_ir.py` | the corrections are present, the removed lines absent | 4 |
| `workflows/templates/minimax/README.md`, `workflows/templates/ltx2/README.md` | links, vendor sources, model facts | 5 |
| `docs/proposals/agent-catalog-legibility.md` | ledger row and dated follow-ups | every task, 6 |

---

### Task 1: Prove the latent handoff reaches both tensors

**Recommended model:** sonnet — one test, one file, the behaviour is already there.

**Files:**
- Test: `tests/test_result.py`

**Interfaces:**
- Consumes: `dw.result.Result.get_artifact_properties(name)`, `dw.previous_results.get_previous_results(previous_results, "step.field")`.
- Produces: the guarantee Task 2 relies on: for an `LTX2PipelineOutput`-shaped object (a dataclass with `frames` and `audio` tensors), `previous_result:base.frames` yields the frames tensor and `previous_result:base.audio` the audio tensor, untouched.

- [ ] **Step 1: Write the failing-or-passing test**

Append to `tests/test_result.py` (inside the existing test class that holds `test_get_artifact_properties`, or as a new class beside it):

```python
from dataclasses import dataclass

import torch

from dw.previous_results import get_previous_results


@dataclass
class _LatentOutput:
    """The shape LTX2PipelineOutput has when output_type is 'latent': two
    tensors, video latents under 'frames' and audio latents under 'audio'."""

    frames: torch.Tensor
    audio: torch.Tensor


class TestLatentHandoff:
    """A step that returns latents hands each tensor to the next step by name,
    which is what a two-stage flow needs: the upsampler takes the video latents
    and the refinement pass takes the audio latents the base step made."""

    def _base_result(self):
        result = Result({"content_type": "video/mp4", "save": False})
        result.add_result(
            _LatentOutput(frames=torch.zeros(1, 128, 16, 14, 24), audio=torch.zeros(1, 8, 50, 16))
        )
        return result

    def test_the_video_latents_are_reached_by_name(self):
        values = get_previous_results({"base": self._base_result()}, "base.frames")

        assert len(values) == 1
        assert values[0].shape == (1, 128, 16, 14, 24)

    def test_the_audio_latents_are_reached_by_name(self):
        values = get_previous_results({"base": self._base_result()}, "base.audio")

        assert len(values) == 1
        assert values[0].shape == (1, 8, 50, 16)
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_result.py -k LatentHandoff -v`
Expected: PASS. `get_artifact_properties` reads attributes with `getattr`, so the property route already works.

If it FAILS: the failure names what the engine cannot do. Do not patch around it in the template. Report BLOCKED with the assertion text; the controller rules on the smallest change to `dw/result.py` or `dw/previous_results.py`, which then lands in this task with this test as its cover.

- [ ] **Step 3: Ledger and commit**

Append to the Part 4 ledger row: `Catalog repair task 1: the latent handoff a two-stage flow needs is proven by test (tests/test_result.py::TestLatentHandoff), no engine change.`

```bash
git add tests/test_result.py docs/proposals/agent-catalog-legibility.md
git commit -m "test: a latent pipeline output hands each tensor to the next step by name

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The three-move two-stage template

**Recommended model:** sonnet — the JSON is given in full below; the work is transcription plus validation.

**Files:**
- Modify: `workflows/templates/ltx2/two-stage.json`
- Modify: `docs/RECIPES_24GB.md` (the paragraph starting `Spend headroom on the two-stage flow` and the `**Examples:**` line's `two-stage.json` parenthetical)
- Test: `tests/test_catalog_structure.py`

**Interfaces:**
- Consumes: Task 1's guarantee. `LTX2Pipeline.__call__` accepts `latents` (5-D, denormalized; the pipeline normalizes and packs them), `audio_latents` (4-D, denormalized; same), `noise_scale`, `sigmas`. `LTX2LatentUpsamplePipeline.__call__` accepts `latents` (5-D, denormalized) and returns denormalized latents when `output_type` is latent. The base pipeline's latent output denormalizes both tensors before returning them, so the handoff needs no flags.
- Produces: a template whose refine step is the pattern the LTX-2.5 skill will describe as "the three-move flow".

- [ ] **Step 1: Write the failing test**

Append to `tests/test_catalog_structure.py`:

```python
def _step(definition, name):
    return next(step for step in definition["steps"] if step["name"] == name)


class TestLtxTwoStage:
    """LTX-2.5's distilled two-stage flow is three moves: eight sigmas at half
    size, a 2x latent upsample, then renoise and three more sigmas at full size.
    The renoise scale is the first stage-two sigma, which no reference syntax can
    name, so the template carries the literal and this test ties it to the library."""

    def _definition(self):
        path = os.path.join(REPO_ROOT, "workflows", "templates", "ltx2", "two-stage.json")
        return json.load(open(path, encoding="utf-8"))

    def test_the_renoise_scale_is_the_first_stage_two_sigma(self):
        from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

        refine = _step(self._definition(), "refine")

        assert refine["pipeline"]["arguments"]["noise_scale"] == STAGE_2_DISTILLED_SIGMA_VALUES[0]

    def test_the_refine_pass_runs_the_stage_two_schedule_on_the_upsampled_latents(self):
        refine = _step(self._definition(), "refine")
        arguments = refine["pipeline"]["arguments"]

        assert arguments["sigmas"] == "constant:diffusers.pipelines.ltx2.utils.STAGE_2_DISTILLED_SIGMA_VALUES"
        assert arguments["latents"] == "previous_result:upscale.frames"
        assert arguments["audio_latents"] == "previous_result:base.audio"

    def test_the_base_and_refine_passes_share_one_pipeline(self):
        definition = self._definition()
        base = _step(definition, "base")["pipeline"]
        refine = _step(definition, "refine")["pipeline"]

        assert base["configuration"] == refine["configuration"]
        assert base["from_pretrained_arguments"] == refine["from_pretrained_arguments"]
        assert not _step(definition, "base").get("release_pipeline", False)
```

- [ ] **Step 2: Run it to see it fail**

Run: `pytest tests/test_catalog_structure.py -k LtxTwoStage -v`
Expected: FAIL with `StopIteration` (no step named `refine`).

- [ ] **Step 3: Rewrite the template**

Replace the whole of `workflows/templates/ltx2/two-stage.json` with the following. The `configuration`, `from_pretrained_arguments`, `transformer` and `text_encoder` blocks of the `refine` step are byte-for-byte copies of the `base` step's, which is what lets the pipeline cache serve the second pass without a second load.

```json
{
    "id": "LTX2TwoStage",
    "description": "LTX-2.5's distilled two-stage flow in its three moves: render at half size on the eight distilled sigmas, double the video latents with the latent upsampler, then renoise them and run the three stage-two sigmas at full size through the same pipeline. The soundtrack is generated in the base pass and its latents are carried through the refine pass unchanged, so the finished clip has the audio the base pass made. Latents pass between the steps directly - nothing is decoded to pixels and re-encoded. The renoise scale 0.909375 is the first value of the library's STAGE_2_DISTILLED_SIGMA_VALUES, which a reference cannot name, so it is written out and a test holds it to the constant. The base pass keeps its pipeline loaded for the refine pass (identical configurations load once); the refine pass releases it. The upsampler borrows the base pass's VAE.",
    "summary": "LTX-2.5 video with audio at 1536x896 by the distilled two-stage flow: half-size render, 2x latent upsample, three-sigma refine.",
    "variables": {
        "transformer_weights_dtype": "{uint4}",
        "text_encoder_weights_dtype": "{int8}",
        "prompt": "prompt:ltx2/fox_dawn_choir",
        "negative_prompt": "constant:diffusers.pipelines.ltx2.utils.DEFAULT_NEGATIVE_PROMPT",
        "width": 768,
        "height": 448,
        "full_width": 1536,
        "full_height": 896,
        "num_frames": 121,
        "frame_rate": 24.0
    },
    "seed": 42,
    "steps": [
        {
            "name": "base",
            "pipeline": {
                "configuration": {
                    "component_type": "LTX2Pipeline",
                    "pre_load_modules": [
                        "sdnq"
                    ],
                    "shared_components": [
                        "vae"
                    ],
                    "components": {
                        "transformer": {
                            "device": "cuda"
                        },
                        "text_encoder": {
                            "group_offload": {
                                "offload_type": "leaf_level",
                                "use_stream": true
                            }
                        },
                        "connectors": {
                            "group_offload": {
                                "offload_type": "leaf_level",
                                "use_stream": true
                            }
                        },
                        "vae": {
                            "device": "cuda"
                        },
                        "audio_vae": {
                            "device": "cuda"
                        },
                        "vocoder": {
                            "device": "cuda"
                        },
                        "duration_head": {
                            "device": "cuda"
                        }
                    },
                    "vae": {
                        "enable_tiling": true
                    }
                },
                "from_pretrained_arguments": {
                    "model_name": "Lightricks/LTX-2.5-Diffusers",
                    "torch_dtype": "torch.bfloat16"
                },
                "transformer": {
                    "configuration": {
                        "component_type": "LTX2VideoTransformer3DModel",
                        "preserve_device_placement": true
                    },
                    "quantization_config": {
                        "configuration": {
                            "config_type": "sdnq.SDNQConfig"
                        },
                        "arguments": {
                            "weights_dtype": "variable:transformer_weights_dtype",
                            "quantization_device": "cuda",
                            "return_device": "cuda",
                            "use_quantized_matmul": true,
                            "dequantize_fp32": false
                        }
                    },
                    "from_pretrained_arguments": {
                        "model_name": "Lightricks/LTX-2.5-Diffusers",
                        "subfolder": "transformer",
                        "torch_dtype": "torch.bfloat16"
                    }
                },
                "text_encoder": {
                    "configuration": {
                        "component_type": "transformers.Gemma4UnifiedForConditionalGeneration",
                        "preserve_device_placement": true
                    },
                    "quantization_config": {
                        "configuration": {
                            "config_type": "sdnq.SDNQConfig"
                        },
                        "arguments": {
                            "weights_dtype": "variable:text_encoder_weights_dtype",
                            "quantization_device": "cuda",
                            "return_device": "cpu",
                            "dequantize_fp32": false,
                            "modules_to_not_convert": [
                                "embed_vision",
                                "embed_audio",
                                "lm_head"
                            ]
                        }
                    },
                    "from_pretrained_arguments": {
                        "model_name": "Lightricks/LTX-2.5-Diffusers",
                        "subfolder": "text_encoder",
                        "torch_dtype": "torch.bfloat16"
                    }
                },
                "arguments": {
                    "prompt": "variable:prompt",
                    "negative_prompt": "variable:negative_prompt",
                    "width": "variable:width",
                    "height": "variable:height",
                    "num_frames": "variable:num_frames",
                    "frame_rate": "variable:frame_rate",
                    "sigmas": "constant:diffusers.pipelines.ltx2.utils.DISTILLED_SIGMA_VALUES",
                    "guidance_scale": 1.0,
                    "audio_guidance_scale": 1.0,
                    "stg_scale": 0.0,
                    "audio_stg_scale": 0.0,
                    "modality_scale": 1.0,
                    "audio_modality_scale": 1.0,
                    "output_type": "{latent}"
                }
            },
            "result": {
                "content_type": "video/mp4",
                "save": false,
                "fps": 24
            }
        },
        {
            "name": "upscale",
            "pipeline": {
                "configuration": {
                    "component_type": "LTX2LatentUpsamplePipeline",
                    "reused_components": [
                        "vae"
                    ]
                },
                "from_pretrained_arguments": {},
                "latent_upsampler": {
                    "configuration": {
                        "component_type": "diffusers.pipelines.ltx2.latent_upsampler.LTX2LatentUpsamplerModel"
                    },
                    "from_pretrained_arguments": {
                        "model_name": "Lightricks/LTX-2.5-Diffusers",
                        "subfolder": "latent_upsampler",
                        "torch_dtype": "torch.bfloat16"
                    }
                },
                "arguments": {
                    "latents": "previous_result:base.frames",
                    "width": "variable:width",
                    "height": "variable:height",
                    "adain_factor": 0.0,
                    "tone_map_compression_ratio": 0.0,
                    "output_type": "{latent}"
                }
            },
            "result": {
                "content_type": "video/mp4",
                "save": false,
                "fps": 24
            }
        },
        {
            "name": "refine",
            "release_pipeline": true,
            "pipeline": {
                "configuration": {
                    "component_type": "LTX2Pipeline",
                    "pre_load_modules": [
                        "sdnq"
                    ],
                    "shared_components": [
                        "vae"
                    ],
                    "components": {
                        "transformer": {
                            "device": "cuda"
                        },
                        "text_encoder": {
                            "group_offload": {
                                "offload_type": "leaf_level",
                                "use_stream": true
                            }
                        },
                        "connectors": {
                            "group_offload": {
                                "offload_type": "leaf_level",
                                "use_stream": true
                            }
                        },
                        "vae": {
                            "device": "cuda"
                        },
                        "audio_vae": {
                            "device": "cuda"
                        },
                        "vocoder": {
                            "device": "cuda"
                        },
                        "duration_head": {
                            "device": "cuda"
                        }
                    },
                    "vae": {
                        "enable_tiling": true
                    }
                },
                "from_pretrained_arguments": {
                    "model_name": "Lightricks/LTX-2.5-Diffusers",
                    "torch_dtype": "torch.bfloat16"
                },
                "transformer": {
                    "configuration": {
                        "component_type": "LTX2VideoTransformer3DModel",
                        "preserve_device_placement": true
                    },
                    "quantization_config": {
                        "configuration": {
                            "config_type": "sdnq.SDNQConfig"
                        },
                        "arguments": {
                            "weights_dtype": "variable:transformer_weights_dtype",
                            "quantization_device": "cuda",
                            "return_device": "cuda",
                            "use_quantized_matmul": true,
                            "dequantize_fp32": false
                        }
                    },
                    "from_pretrained_arguments": {
                        "model_name": "Lightricks/LTX-2.5-Diffusers",
                        "subfolder": "transformer",
                        "torch_dtype": "torch.bfloat16"
                    }
                },
                "text_encoder": {
                    "configuration": {
                        "component_type": "transformers.Gemma4UnifiedForConditionalGeneration",
                        "preserve_device_placement": true
                    },
                    "quantization_config": {
                        "configuration": {
                            "config_type": "sdnq.SDNQConfig"
                        },
                        "arguments": {
                            "weights_dtype": "variable:text_encoder_weights_dtype",
                            "quantization_device": "cuda",
                            "return_device": "cpu",
                            "dequantize_fp32": false,
                            "modules_to_not_convert": [
                                "embed_vision",
                                "embed_audio",
                                "lm_head"
                            ]
                        }
                    },
                    "from_pretrained_arguments": {
                        "model_name": "Lightricks/LTX-2.5-Diffusers",
                        "subfolder": "text_encoder",
                        "torch_dtype": "torch.bfloat16"
                    }
                },
                "arguments": {
                    "prompt": "variable:prompt",
                    "negative_prompt": "variable:negative_prompt",
                    "latents": "previous_result:upscale.frames",
                    "audio_latents": "previous_result:base.audio",
                    "noise_scale": 0.909375,
                    "sigmas": "constant:diffusers.pipelines.ltx2.utils.STAGE_2_DISTILLED_SIGMA_VALUES",
                    "width": "variable:full_width",
                    "height": "variable:full_height",
                    "num_frames": "variable:num_frames",
                    "frame_rate": "variable:frame_rate",
                    "guidance_scale": 1.0,
                    "audio_guidance_scale": 1.0,
                    "stg_scale": 0.0,
                    "audio_stg_scale": 0.0,
                    "modality_scale": 1.0,
                    "audio_modality_scale": 1.0,
                    "output_type": "{np}"
                }
            },
            "result": {
                "content_type": "video/mp4",
                "fps": 24
            }
        }
    ]
}
```

Notes for the implementer:
- `"vae": {"enable_tiling": true}` sits inside `configuration` of both `LTX2Pipeline` steps, where the old upscale step carried it; the engine reads it there. The base pass decodes nothing (latent output), so tiling costs it nothing, and keeping the two configurations identical is what makes the refine pass a cache hit.
- The upscale step no longer has the tiling block: it decodes nothing now.
- The description mentions `STAGE_2_DISTILLED_SIGMA_VALUES` without single quotes on purpose: the drift check treats a single-quoted name as a variable or step the workflow must have.

- [ ] **Step 4: Validate and test**

Run:
```bash
python -m dw.validate workflows/templates/ltx2/two-stage.json
pytest tests/test_catalog_structure.py tests/test_examples.py -k "LtxTwoStage or two-stage or two_stage" -v
pytest tests/test_catalog_structure.py -q
```
Expected: validate prints no errors; the three new tests PASS; the description-drift check passes (the description quotes no `'name'` the workflow lacks: it names `STAGE_2_DISTILLED_SIGMA_VALUES` without quotes).

- [ ] **Step 5: Rewrite the recipe paragraph**

In `docs/RECIPES_24GB.md`, replace the paragraph beginning `Spend headroom on the two-stage flow` with:

```markdown
Spend headroom on the two-stage flow rather than on base resolution: render at 768x448
on the eight distilled sigmas, double the video latents with the latent upsampler, then
renoise them and run the three stage-two sigmas at 1536x896 through the same pipeline.
That refine pass is what puts the detail back - the upsampler alone gives a soft 2x -
and it is the flow the model card, Lightricks' pipeline notes and the diffusers docs all
describe. The base pass keeps its pipeline loaded so the refine pass is served from the
cache; the refine pass releases it. Since 2026-08 Lightricks route production quality
through their DFR pipeline instead, which diffusers ships and nothing here uses yet.
```

and in the `**Examples:**` line change `(base -> latent upsample -> mux)` to `(base -> latent upsample -> refine)`.

- [ ] **Step 6: Ledger and commit**

Append to the Part 4 ledger row: `Catalog repair task 2: two-stage.json is the three-move flow (8 sigmas at 768x448, 2x latent upsample, renoise + 3 stage-two sigmas at 1536x896, audio latents carried); latents pass by name; a test holds noise_scale to STAGE_2_DISTILLED_SIGMA_VALUES[0]. Cost and the sharpness comparison await the lem run (task 7).`

```bash
git add workflows/templates/ltx2/two-stage.json tests/test_catalog_structure.py docs/RECIPES_24GB.md docs/proposals/agent-catalog-legibility.md
git commit -m "LTX-2.5 two-stage template runs the stage-two refine pass it was missing

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The LTX-2.5 prompt library in the trained format

**Recommended model:** opus — six captions written to a spec is judgment work, and the test only catches the gross failures.

**Files:**
- Modify: `prompts/ltx2/fox_dawn_choir.json`, `prompts/ltx2/hummingbird_garden.json`, `prompts/ltx2/lighthouse_keeper.json`, `prompts/ltx2/lighthouse_keeper_gallery.json`, `prompts/ltx2/marmot_robot_overlords.json`, `prompts/ltx2/polaroid_lighthouse.json`
- Modify: `workflows/templates/ltx2/chained-segments.json`, `enhance-prompt.json`, `extend-clip.json`, `keyframes.json` (the `summary` field only)
- Create: `tests/test_ltx_prompt_library.py`

**Interfaces:**
- Consumes: `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` and `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT` from `diffusers.pipelines.ltx2.utils`, which are the spec. Read both in full before writing a word.
- Produces: six prompts other templates reference by the same names; nothing else depends on their text.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ltx_prompt_library.py`:

```python
"""The stored LTX-2.5 prompts are in the genre the model was trained on.

Lightricks' training captions are one paragraph of roughly 150-220 words in
the present progressive, opening on the action, carrying a shot type, a
camera motion and a viewpoint in prose, with the soundscape interleaved
rather than appended. A stored prompt is what every template runs by
default and what an agent copies, so one in image-generation tag style
teaches the wrong thing twice.
"""

import glob
import json
import os

import pytest

from tests.test_examples import REPO_ROOT

PROMPTS = sorted(glob.glob(os.path.join(REPO_ROOT, "prompts", "ltx2", "*.json")))

# Tag-style and preamble phrases the training spec rules out
FORBIDDEN = (
    "8k",
    "ultra-detailed",
    "photorealistic",
    "vibrant colors",
    "highly detailed",
    "The scene opens",
    "We see",
    "The image is",
)


def _prompt(path):
    return json.load(open(path, encoding="utf-8"))


def test_there_are_ltx_prompts():
    assert len(PROMPTS) == 6


@pytest.mark.parametrize("path", PROMPTS, ids=os.path.basename)
def test_a_prompt_is_one_paragraph_of_caption_length(path):
    text = _prompt(path)["text"]

    assert "\n" not in text.strip(), f"{path} is more than one paragraph"
    words = len(text.split())
    assert 140 <= words <= 240, f"{path} is {words} words; the trained caption is 150-220"


@pytest.mark.parametrize("path", PROMPTS, ids=os.path.basename)
def test_a_prompt_carries_no_tag_style_phrase(path):
    text = _prompt(path)["text"]

    for phrase in FORBIDDEN:
        assert phrase.lower() not in text.lower(), f"{path} contains {phrase!r}"


@pytest.mark.parametrize("path", PROMPTS, ids=os.path.basename)
def test_a_prompt_names_the_model_it_is_for(path):
    assert _prompt(path)["intended_model"] == "ltx-2.5"


def test_no_ltx_template_summary_names_the_older_model():
    templates = glob.glob(os.path.join(REPO_ROOT, "workflows", "templates", "ltx2", "*.json"))
    for path in templates:
        summary = json.load(open(path, encoding="utf-8")).get("summary", "")
        assert "LTX-2 " not in summary and not summary.endswith("LTX-2"), path
```

- [ ] **Step 2: Run it to see it fail**

Run: `pytest tests/test_ltx_prompt_library.py -v`
Expected: FAIL on length for all six, on forbidden phrases for `fox_dawn_choir` and `marmot_robot_overlords`, on `intended_model` for all six, and on four summaries.

- [ ] **Step 3: Read the spec, then rewrite the six prompts**

Print the spec once and keep it open:

```bash
python -c "from diffusers.pipelines.ltx2.utils import LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT as t, LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT as i; print(t); print('=====I2V'); print(i)"
```

Rewrite the `text` of each file. Keep each file's `description` and `tags` as they are (the gallery and the templates reference them), and set `"intended_model": "ltx-2.5"`. Rules, from the spec:

- 150 to 220 words, one paragraph, present-progressive verbs, real time, chronological connectors ("Initially", "A moment later", "Simultaneously").
- Open on the action. Never "The scene opens", "We see", "There is".
- Every shot states its shot type (wide, medium, close-up, extreme close-up, over-the-shoulder, or POV), its camera motion (say "the camera holds static" when there is none), and its viewpoint relative to the subject, woven into the prose.
- Soundscape interleaved with the action as it happens, not appended at the end. Dialogue and sung lines quoted exactly as before: `'l t x'`, `'i for one welcome our robot overloards'`, `'still burning'`.
- Observable detail only: materials, textures, light, colour in plain words ("red dress", not "vibrant"). No inferred emotion, no quality tags, no artist names, no "image".
- The two I2V prompts, `marmot_robot_overlords` (used by `image-to-video.json` and `chained-segments.json`) and `polaroid_lighthouse` (used by `keyframes.json`), follow the I2V spec instead: describe what changes from the reference image, its motion, camera and sound, and do not restate what the image already establishes. They still meet the length and framing rules.
- `lighthouse_keeper_gallery` stays a continuation of `lighthouse_keeper`: same keeper, same oilskin, and it ends on the gallery with the line `'still burning'`.

The subject of each stays what its `description` says. A worked example for `hummingbird_garden`, to calibrate length and shape (write your own; do not paste this):

> A hummingbird is hovering at a red trumpet flower in a sunlit garden, its wings a grey blur either side of a green-throated body, in an extreme close-up from slightly below as the camera holds static on the bloom. Its needle bill is sliding into the flower's throat while the tongue flicks in and out, and the wingbeats are producing a low steady hum over the drone of unseen insects. Initially the flower is swaying a few centimetres under the bird's weight, petals catching the light in flat orange-red, with dew on the leaves behind it throwing small white highlights. A moment later the bird is backing off, pivoting in the air, and the camera is pulling out slowly to a medium shot from the same low angle, showing a bed of the same flowers and a wooden fence behind, weathered grey, while a light breeze is moving the leaves with a soft dry rustle. Simultaneously the bird is darting to a second flower on the left of frame and the hum is rising in pitch as it brakes, then settling as it hovers again, bill in, wings blurred, the insects still droning under it.

- [ ] **Step 4: Update the four summaries**

In `workflows/templates/ltx2/chained-segments.json`, `enhance-prompt.json`, `extend-clip.json` and `keyframes.json`, change `LTX-2 ` to `LTX-2.5 ` in the `summary` field only. Check the summary is still under 120 characters (`pytest tests/test_catalog_shape.py -q` covers the limit).

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_ltx_prompt_library.py tests/test_catalog_shape.py tests/test_catalog_structure.py -q`
Expected: all PASS.

- [ ] **Step 6: Ledger and commit**

Append to the Part 4 ledger row: `Catalog repair task 3: the six prompts/ltx2 captions are rewritten to the trained format (one paragraph, 150-220 words, shot type/camera motion/viewpoint in prose, sound interleaved; the two I2V ones describe only what changes), intended_model ltx-2.5, four summaries say LTX-2.5; tests/test_ltx_prompt_library.py holds the shape.`

```bash
git add prompts/ltx2 workflows/templates/ltx2 tests/test_ltx_prompt_library.py docs/proposals/agent-catalog-legibility.md
git commit -m "LTX-2.5 prompt library rewritten in the caption format the model was trained on

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The H3 Context-IR builtin, corrected against the guides

**Recommended model:** opus — five surgical edits inside a thousand-word system prompt that must stay coherent.

**Files:**
- Modify: `dw/workflows/h3_context_ir.json` (the `variables.system_prompt` string only)
- Create: `tests/test_h3_context_ir.py`

**Interfaces:**
- Consumes: the audit's claim-by-claim section for `h3_context_ir.json` (read `docs/proposals/audits/2026-09-07-minimax-h3-audit.md` sections "Continuity modes", "Speakers and dialogue", "Ref2VA six sections", and "Missing knowledge" items 6, 7, 9).
- Produces: a system prompt the enhance-prompt templates use unchanged; their `prompt` variables and the `Continuity:` convention in the user message are unaffected.

A correction to the spec: nothing in the engine or any template emits `Continuity: continuation` today; the phrase exists only in this system prompt, where the user message "may state" it. So the block is a convention a workflow author uses when writing a chained segment's prompt through the enhancer, not something the chain feature injects. The relabelling below says exactly that.

- [ ] **Step 1: Write the failing test**

Create `tests/test_h3_context_ir.py`:

```python
"""The H3 Context-IR builtin teaches what MiniMax's prompt-writing guides say.

The system prompt is a compression of the two guides on the model card
(docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md and _ref_en.md). The 2026-09-07
audit found three things it got wrong and one it left out; these tests keep
the corrections in place and the unsourced lines out.
"""

import json
import os

from tests.test_examples import BUILTIN_DIR


def _system_prompt():
    path = os.path.join(BUILTIN_DIR, "h3_context_ir.json")
    return json.load(open(path, encoding="utf-8"))["variables"]["system_prompt"]


def test_the_sources_are_named():
    prompt = _system_prompt()

    assert "VIDEO_PROMPT_WRITING_GUIDE_base_en.md" in prompt
    assert "VIDEO_PROMPT_WRITING_GUIDE_ref_en.md" in prompt


def test_silent_audio_fields_are_written_as_not_applicable():
    """Without this the enhancer invents a soundscape for a silent brief."""
    prompt = _system_prompt()

    assert "N/A" in prompt
    assert "overall_soundscape" in prompt[prompt.index("N/A") - 600 : prompt.index("N/A") + 600]


def test_video_and_audio_references_are_numbered_within_their_own_category():
    prompt = _system_prompt()

    assert "numbered independently" in prompt
    assert "does not by itself" in prompt


def test_continuity_modes_are_labelled_as_this_engine_convention():
    prompt = _system_prompt()

    assert "not part of Context-IR" in prompt


def test_the_dialogue_fidelity_rules_are_present():
    prompt = _system_prompt()

    assert "[unclear]" in prompt
    assert "retention_analysis" in prompt and "(Sx)" in prompt


def test_the_unsourced_lines_are_gone():
    prompt = _system_prompt()

    assert "degrades on anything else" not in prompt
    assert "8k" not in prompt
    assert "artist names" not in prompt
    # The confirmed part of that line stays
    assert "no negative prompt" in prompt
```

- [ ] **Step 2: Run it to see it fail**

Run: `pytest tests/test_h3_context_ir.py -v`
Expected: five of six FAIL (`the_dialogue_fidelity_rules` may partially pass on `retention_analysis`; the `(Sx)` assertion fails).

- [ ] **Step 3: Edit the system prompt**

Edit the `system_prompt` string in `dw/workflows/h3_context_ir.json`. It is one JSON string with `\n` line breaks; edit it with a small Python script that loads the JSON, applies string replacements, asserts each `old` occurred exactly once, and dumps with `indent=4, ensure_ascii=False` so the file's formatting is preserved. Make these six changes:

1. Replace the opening sentence `You rewrite a user's video idea into a MiniMax-H3 Context-IR prompt. H3-Base is trained to consume this exact format and degrades on anything else. ` with:

   `You rewrite a user's video idea into a MiniMax-H3 Context-IR prompt, the format MiniMax's two prompt-writing guides define (docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md and docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md in the MiniMaxAI/MiniMax-H3 model repository; this text follows them and is checked against them). `

2. In the CORE FIELDS block, after the sentence defining `non_diegetic_music: background score the audience hears and the characters cannot.` append: ` When the brief has no ambient sound, write overall_soundscape as N/A; when it has no score, write non_diegetic_music as N/A. Never invent audio a silent brief did not ask for.`

3. In the `subject_definitions` sentence of the REF2VA block, replace `The references appear in the order the request passes them, and that order is what the labels number.` with: `The references appear in the order the request passes them, and each label type is numbered independently within its own category: the first video is <Video 1> whether or not pictures precede it, and a video reference does not by itself produce an <Audio N> - an audio label exists only for an audio asset the request passes.`

4. Replace the heading line `CONTINUITY MODES, which decide how the clip opens:` with: `CONTINUITY MODES. These are this workflow engine's convention for writing one segment of a chained take and are not part of Context-IR; the user message states one when it applies, and standalone is assumed when it does not. They decide how the clip opens:`

5. In the RULES block, after the bullet that begins `- Put spoken and sung content inside <d>[Language] ...</d>` append a new bullet: `\n- Dialogue fidelity: write [unclear] for a span that cannot be made out; standardise punctuation and end every <d> block with terminal punctuation before </d>; never write speaker IDs such as (Sx) inside retention_analysis, which describes references, not speech; when an <Audio N> is referenced for a speaker's timbre only, do not carry its original words over - the dialogue is what the brief supplies.`

6. Replace the final bullet `- Every detail must be something visible or audible. No camera hardware specs, no artist names, no quality tags such as 8k or ultra-detailed, and no negative phrasing - H3 is guidance-distilled and has no negative prompt.` with: `- Every detail must be something visible or audible, in plain words. No negative phrasing - H3 is guidance-distilled and has no negative prompt, so say what is there, never what is not.`

The script, to be run from the repo root and then deleted (do not commit it):

```python
import json

PATH = "dw/workflows/h3_context_ir.json"
EDITS = [
    (
        "You rewrite a user's video idea into a MiniMax-H3 Context-IR prompt. H3-Base is trained to consume this exact format and degrades on anything else. ",
        "You rewrite a user's video idea into a MiniMax-H3 Context-IR prompt, the format MiniMax's two prompt-writing guides define (docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md and docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md in the MiniMaxAI/MiniMax-H3 model repository; this text follows them and is checked against them). ",
    ),
    (
        "non_diegetic_music: background score the audience hears and the characters cannot.",
        "non_diegetic_music: background score the audience hears and the characters cannot. When the brief has no ambient sound, write overall_soundscape as N/A; when it has no score, write non_diegetic_music as N/A. Never invent audio a silent brief did not ask for.",
    ),
    (
        "The references appear in the order the request passes them, and that order is what the labels number.",
        "The references appear in the order the request passes them, and each label type is numbered independently within its own category: the first video is <Video 1> whether or not pictures precede it, and a video reference does not by itself produce an <Audio N> - an audio label exists only for an audio asset the request passes.",
    ),
    (
        "CONTINUITY MODES, which decide how the clip opens:",
        "CONTINUITY MODES. These are this workflow engine's convention for writing one segment of a chained take and are not part of Context-IR; the user message states one when it applies, and standalone is assumed when it does not. They decide how the clip opens:",
    ),
    (
        "- Voiceover uses the exact phrase",
        "- Dialogue fidelity: write [unclear] for a span that cannot be made out; standardise punctuation and end every <d> block with terminal punctuation before </d>; never write speaker IDs such as (Sx) inside retention_analysis, which describes references, not speech; when an <Audio N> is referenced for a speaker's timbre only, do not carry its original words over - the dialogue is what the brief supplies.\n- Voiceover uses the exact phrase",
    ),
    (
        "- Every detail must be something visible or audible. No camera hardware specs, no artist names, no quality tags such as 8k or ultra-detailed, and no negative phrasing - H3 is guidance-distilled and has no negative prompt.",
        "- Every detail must be something visible or audible, in plain words. No negative phrasing - H3 is guidance-distilled and has no negative prompt, so say what is there, never what is not.",
    ),
]

with open(PATH, encoding="utf-8") as handle:
    definition = json.load(handle)

prompt = definition["variables"]["system_prompt"]
for old, new in EDITS:
    assert prompt.count(old) == 1, f"expected exactly one occurrence of: {old[:60]!r}"
    prompt = prompt.replace(old, new)
definition["variables"]["system_prompt"] = prompt

with open(PATH, "w", encoding="utf-8") as handle:
    json.dump(definition, handle, indent=4, ensure_ascii=False)
    handle.write("\n")
```

Note the `non_diegetic_music` replacement: the phrase occurs once in the CORE FIELDS block. The REF2VA block says `non_diegetic_music: as above.`, which is a different string, so the count assertion holds. If any assertion fails, stop and report the exact text found rather than loosening the match.

- [ ] **Step 4: Check nothing else moved**

Run:
```bash
git diff --stat dw/workflows/h3_context_ir.json
python -m dw.validate workflows/templates/minimax/enhance-prompt.json
pytest tests/test_h3_context_ir.py tests/test_server.py -k "h3 or context_ir" -q
pytest tests/test_catalog_structure.py -q
```
Expected: the diff touches only the `system_prompt` line; validate passes; all tests PASS.

- [ ] **Step 5: Ledger and commit**

Append to the Part 4 ledger row: `Catalog repair task 4: h3_context_ir names its two source guides, writes N/A for silent audio fields, numbers <Video N>/<Audio N> within their category, labels continuity modes as this engine's chaining convention (nothing in the engine emits the phrase; it is a user-message convention), adds the ref guide's dialogue-fidelity rules, drops the two unsourced lines; tests/test_h3_context_ir.py.`

```bash
git add dw/workflows/h3_context_ir.json tests/test_h3_context_ir.py docs/proposals/agent-catalog-legibility.md
git commit -m "H3 Context-IR builtin corrected against MiniMax's prompt-writing guides

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The two READMEs, with links that resolve and sources named

**Recommended model:** sonnet — the mapping and the new paragraphs are given; the test is short.

**Files:**
- Modify: `workflows/templates/minimax/README.md`, `workflows/templates/ltx2/README.md`
- Test: `tests/test_catalog_structure.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: README link targets Task 6 and Package B's skills will quote.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_catalog_structure.py`:

```python
LINK_PATTERN = re.compile(r"\]\(([^)]+)\)")
READMES = sorted(
    os.path.relpath(path, REPO_ROOT)
    for path in glob.glob(os.path.join(REPO_ROOT, "workflows", "templates", "**", "README.md"), recursive=True)
)


@pytest.mark.parametrize("path", READMES)
def test_every_readme_link_resolves(path):
    """A README is a reading-order map over the templates beside it. The
    templates were renamed once and every link in the map broke; this keeps the
    map pointing at files that exist."""
    text = open(os.path.join(REPO_ROOT, path), encoding="utf-8").read()
    base = os.path.dirname(os.path.join(REPO_ROOT, path))

    for target in LINK_PATTERN.findall(text):
        if target.startswith(("http://", "https://", "#")):
            continue
        target = target.split("#", 1)[0]
        assert os.path.exists(os.path.join(base, target)), f"{path} links to {target}, which does not exist"
```

and add `import glob` to the file's imports.

- [ ] **Step 2: Run it to see it fail**

Run: `pytest tests/test_catalog_structure.py -k readme_link -v`
Expected: FAIL for both READMEs.

- [ ] **Step 3: Fix the MiniMax README**

Apply this mapping to every link in `workflows/templates/minimax/README.md`, changing both the link text and the target (the text becomes the file name, since that is what a reader opens):

| old | new |
| --- | --- |
| `MiniMaxMusic.json` | `music.json` |
| `MiniMaxH3.json` | `video-with-audio.json` |
| `MiniMaxH3I2V.json` | `image-to-video.json` |
| `MiniMaxH3FL2VA.json` | `first-and-last-frame.json` |
| `MiniMaxH3L2V.json` | `last-frame-only.json` |
| `MiniMaxH3EnhancePrompt.json` | `enhance-prompt.json` |
| `MiniMaxH3I2VEnhancePrompt.json` | `enhance-prompt-with-image.json` |
| `MiniMaxH3Ref2VA.json` | `reference-to-video.json` |
| `MiniMaxH3Ref2VAVideo.json` | `composable-references.json` |
| `MiniMaxH3Ref2VAGeneratedSubject.json` | `generated-subject-reference.json` |
| `MiniMaxH3Storyboard.json` | `storyboard.json` |
| `MiniMaxH3I2VChained.json` | `chained-segments.json` |
| `MiniMaxH3Ref2VAChained.json` | `chain-matched-to-audio.json` |
| `MiniMaxH3Ref2VAChainedVideo.json` | `chain-video-continuity.json` |
| `MiniMaxH3Ref2VAChainedAligned.json` | `chain-matched-and-aligned.json` |
| `MiniMaxH3SitcomShort.json` | `dialogue-short.json` |
| `MiniMaxH3MusicVideo.json` | `music-video.json` |

Add `voice-timbre-reference.json` (id `MiniMaxH3GeneratedVoice`) as a row in the "Conditioning on identity" table, after the `reference-to-video.json` row: `| [voice-timbre-reference.json](voice-timbre-reference.json) | A generated voice clip as the audio reference, so a speaker's timbre is fixed without a recording |` (read the template's description first and adjust the wording to what it actually does).

Replace the first paragraph (the one ending `is what the [built-in enhancer](...) produces.`) with:

```markdown
Joint video-and-audio generation with [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
and music generation with [MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
fitted onto a single 24GB consumer GPU. Every example here runs on an RTX 3090;
the memory configuration they share is explained in
[docs/RECIPES_24GB.md](../../docs/RECIPES_24GB.md).

The prompt format is MiniMax's own. Two guides on the model card define it -
`docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` for text- and frame-conditioned
generation and `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` for reference-conditioned -
and MiniMax publishes them as an agent skill, `skills/h3-prompt-writing`, in the
[MiniMax-H3 GitHub repository](https://github.com/MiniMax-AI/MiniMax-H3). The
[built-in enhancer](enhance-prompt.json) writes the same format from those guides;
for hand-written prompts, read them rather than reverse-engineering the examples.
Audited against those sources on 2026-09-07
([the audit](../../../docs/proposals/audits/2026-09-07-minimax-h3-audit.md)).
```

After the paragraph beginning `A note on length:` add:

```markdown
A note on the canvas: H3 draws on a 768-pixel short edge (960x544 in these
examples is the speed choice, coupled to the 544p turbo LoRA and its nine steps -
change one and change the others), dimensions are multiples of 32, and aspect ratios
run from 1:4 to 4:1. The 5-second floor is diffusers' constraint; the model card and
the hosted API accept 4. Output audio is 32 kHz stereo.
```

- [ ] **Step 4: Fix the LTX README**

Apply this mapping in `workflows/templates/ltx2/README.md`:

| old | new |
| --- | --- |
| `LTX2.json` | `text-to-video.json` |
| `LTX2I2V.json` | `image-to-video.json` |
| `LTX2Keyframes.json` | `keyframes.json` |
| `LTX2I2VEnhancePrompt.json` | `enhance-prompt.json` |
| `LTX2TwoStage.json` | `two-stage.json` |
| `LTX2ICLora.json` | `generative-upscale.json` |
| `LTX2Extend.json` | `extend-clip.json` |
| `LTX2I2VChained.json` | `chained-segments.json` |

Change the model link from `https://huggingface.co/Lightricks/LTX-Video` to `https://huggingface.co/Lightricks/LTX-2.5-Diffusers`.

Replace the first paragraph with:

```markdown
Text- and image-to-video with a generated soundtrack, using
[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) fitted onto a single
24GB consumer GPU. The memory configuration the examples share is explained in
[docs/RECIPES_24GB.md](../../docs/RECIPES_24GB.md).

The prompt format is Lightricks' own: the model was trained on one-paragraph
captions of roughly 150-220 words that carry a shot type, a camera motion and a
viewpoint in prose, with the soundscape interleaved with the action. That spec ships
inside diffusers as `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` and
`LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT` in `diffusers.pipelines.ltx2.utils`, which is what
the [enhancer](enhance-prompt.json) runs; the stored prompts under `prompts/ltx2/` are
written to it. Settings come from the
[model card](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) and Lightricks'
`ltx-pipelines` notes. Audited against those sources on 2026-09-07
([the audit](../../../docs/proposals/audits/2026-09-07-ltx-2.5-audit.md)).
```

In the "Quality and scale" table, change the `two-stage.json` row's text to: `The distilled two-stage flow in its three moves: render at half size, double the latents, renoise and refine at full size. Lightricks' newer DFR pipeline is the follow-up`.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_catalog_structure.py -q`
Expected: PASS, including `test_every_readme_link_resolves` for both files. Also confirm the three `../../../docs/proposals/audits/...` links resolve from each README's directory (the test checks this).

- [ ] **Step 6: Ledger and commit**

Append to the Part 4 ledger row: `Catalog repair task 5: both template READMEs link the files that exist, name the vendor sources and the audit, the H3 one states the canvas rules and the 5-second diffusers floor; a test resolves every README link.`

```bash
git add workflows/templates/minimax/README.md workflows/templates/ltx2/README.md tests/test_catalog_structure.py docs/proposals/agent-catalog-legibility.md
git commit -m "Template READMEs: links that resolve, vendor sources named, H3 canvas rules stated

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Dated follow-ups in the ledger

**Recommended model:** sonnet — a transcription from the two audits' "Missing knowledge" sections.

**Files:**
- Modify: `docs/proposals/agent-catalog-legibility.md`

**Interfaces:**
- Consumes: both audits' "Missing knowledge" sections.
- Produces: the list the next model-family pass starts from.

- [ ] **Step 1: Add the follow-up section**

Below the `### Cold-session probe, 2026-09-07` section at the end of the proposal, add:

```markdown
### Model-knowledge follow-ups, 2026-09-07

What the two vendor audits found that the catalog does not carry, dated by the
source, and deliberately not done in the repair pass. Each is a template or a
guide sentence when it lands, never engine code.

LTX-2.5 ([audit](audits/2026-09-07-ltx-2.5-audit.md)):

- DFR pipeline as the production-quality path (LTX-2 1.2.0, 2026-08-11; refined
  1.3.0, 2026-08-25): `LTX2DFRPipeline` ships in diffusers, unused; needs
  64-divisible dimensions.
- Temporal upscaling to 48/96 fps (2026-08-11), and the RoPE fps trap: condition
  at 60 for high frame rates, never 120.
- Generated keyframe slots for fast motion (2026-08-11).
- Image conditioning is re-compressed at CRF 18 and needs a PIL image; undocumented.
- Keyframe strength below 1.0 for smooth interpolation (`ANCHOR_KEYFRAME_STRENGTH`).
- IC-LoRA trade-off (fewer steps, closer to reference) and the clean-reference rule.
- fp8 / NVFP4 / CUDA-graph capture / `AUTO_TILING` (2026-08); HDR and retake pipelines;
  native multishot prompting.
- The dev transformer's bf16 size in RECIPES_24GB (stated ~38GB; 22B is ~44GB).

MiniMax H3 ([audit](audits/2026-09-07-minimax-h3-audit.md)):

- A Ref2VA turbo LoRA exists (4-step v0.1; 8-step v1.0 768p, HF 2026-09-04); every
  Ref2VA template runs 20 unaccelerated steps.
- Newer FL2VA LoRAs (4-step v1.1/v1.2 768p, 8-step v1.0 768p) and the scheduler-shift
  contract they carry (12/3 at 544p, 6/3 at 768p); nothing here mentions shifts.
- Reference-image resize policy: ModelTC recommend `match`; diffusers' fixed 2048 short
  edge is the `diffusers` policy.
- Four templates load the FL2VA LoRA on reference-bearing requests (`storyboard`,
  `dialogue-short`, `music-video`, `chain-matched-and-aligned`); check against the
  Ref2VA LoRA.
- Cut-verb and audio-continuity vocabularies (base guide §4.2, §4.4).
- Step-count note: 20 default, ~25 for motion (ComfyUI).
- Ref2VA input limits (≤9 images, ≤3 videos, ≤3 audio, ≤12) and that audio can never
  be the only reference; H3-Regenerate-2K is API-only.
- Music3 `audio_duration` cap: 9000 frames, six minutes.
```

- [ ] **Step 2: Commit**

```bash
git add docs/proposals/agent-catalog-legibility.md
git commit -m "Ledger: dated model-knowledge follow-ups from the H3 and LTX-2.5 audits

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Verify on lem and write the costs

**Recommended model:** this is run by the controlling session (it needs the dw MCP connection to lem and a human go-ahead for GPU minutes), not a subagent. Opus.

**Files:**
- Modify: `workflows/templates/ltx2/two-stage.json` (`cost`), `docs/RECIPES_24GB.md` (only if the comparison supports a sharpness sentence), `docs/proposals/agent-catalog-legibility.md`

**Interfaces:**
- Consumes: Tasks 2 to 5 merged into the branch; lem checked out on the branch and `dw.serve` restarted (the builtin and the JSON are read at start or per run respectively, and a restart makes both certain).

- [ ] **Step 1: Restart lem on the branch**

On lem: `cd ~/diffusers-workflow && git fetch && git checkout catalog-repair && git pull`, then restart `dw.serve` with its usual arguments (`--token ... --workspace ~/diffusers-workspace --examples-dir ~/diffusers-workflow/workflows --mcp`).

- [ ] **Step 2: Two-stage, and the comparison**

Through MCP: `validate_workflow` on `templates/ltx2/two-stage`, then `run_workflow` with `acknowledged_cost=true` after a go-ahead, `wait_for_job`, `get_job`. Record the warm wall time of the refine step and the whole run. Then run the old flow for comparison: submit the pre-repair template inline (from `git show master:workflows/templates/ltx2/two-stage.json`) with the same seed, and look at a frame of each with `get_output_image` on a `video_frame` or via the gallery URL. Decide: is the refined output visibly sharper than the upsample-only one? Write the answer in the ledger either way.

Expected: the run completes at 1536x896; if the refine step OOMs, the fallback to try first is `full_width` 1280 / `full_height` 736 (both divisible by 32, the aspect kept), and the ledger says the full size did not fit on 24 GB.

- [ ] **Step 3: Two prompts through text-to-video**

Run `templates/ltx2/text-to-video` with `prompt=prompt:ltx2/fox_dawn_choir` (the default) and again with `prompt=prompt:ltx2/hummingbird_garden`. Look at each. Pass: the fox clip has a tracking shot and the choir line is audible; the hummingbird clip holds on the flower, then pulls out, with the hum and rustle present.

- [ ] **Step 4: The silent brief through the H3 enhancer**

Run `templates/minimax/enhance-prompt` with `prompt="A single lit candle on a bare wooden table in a dark room, no sound at all, no music, 5 seconds"` and read the enhancer's text output with `get_output_text`. Pass: `overall_soundscape: N/A` and `non_diegetic_music: N/A` appear. If the model writes audio anyway, strengthen the N/A sentence in the system prompt (Task 4's script pattern) and rerun; record what it took.

- [ ] **Step 5: Write the costs**

Set `cost` on `two-stage.json` to the measured warm minutes, one decimal rounded up:

```json
"cost": [
    {
        "device": "cuda",
        "name": "RTX 3090",
        "vram_gb": 24,
        "minutes": <measured>
    }
]
```

If the comparison in Step 2 showed the refine pass sharper, add to the RECIPES paragraph from Task 2 the sentence: `Measured on an RTX 3090 the refined clip is sharper than the 2x upsample alone at the same seed.` Otherwise leave the paragraph as Task 2 wrote it.

- [ ] **Step 6: Ledger, commit, PR**

Append to the Part 4 ledger row: `Catalog repair task 7: verified on lem <date>: two-stage <minutes> warm min at 1536x896 (<sharper / not sharper> than upsample-only at seed 42); fox and hummingbird captions run as written; the silent-candle brief produced N/A for both audio fields<, after N tries>.`

```bash
git add workflows/templates/ltx2/two-stage.json docs/RECIPES_24GB.md docs/proposals/agent-catalog-legibility.md
git commit -m "Two-stage cost measured on lem; lem verification of the catalog repair recorded

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push -u origin catalog-repair
```

Open the PR against `master` with the six commits' subjects as its bullets and end the body with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
