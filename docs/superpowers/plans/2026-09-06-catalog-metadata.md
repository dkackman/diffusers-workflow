# Catalog Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every workflow in a listing carries a derived `shape`, `traits`, `summary` and hand-authored `cost`; `/api/workflows` can filter on them and project a compact view; the MCP `list_workflows` tool uses that view; jobs record the catalog name they were run from.

**Architecture:** A pure module `dw/server/catalog_shape.py` derives metadata from a raw workflow definition and projects/filters a listing; `workflow_details` in `dw/server/app.py` calls it per file (already cached by mtime) and resolves model-config inheritance; the route applies the projection. No engine code changes — derivation reads JSON structure only.

**Tech Stack:** Python 3.10+, FastAPI, jsonschema, pytest. Tests run with `python -m pytest tests/<file> -v` from the repo root with the venv active (`source ./activate`).

**Spec:** `docs/superpowers/specs/2026-09-06-agent-catalog-legibility-design.md` (plan 1, sections 1.1–1.8). Read it first.

**Agent assignment:** each task is tagged `[model: haiku|sonnet|opus]`. Haiku for mechanical edits with exact code given, sonnet for integration work with judgement about existing code, opus for tasks that require reading real templates and deciding. The orchestrator passes the tag as the `model` parameter when dispatching.

**Ledger:** `docs/proposals/agent-catalog-legibility-complete.md` ends with a `## Ledger` table. The last step of every task updates the row(s) it lands, changing `designed` to `done (task N)` and adding anything learned to the notes column. Keep the row on one line.

## Global Constraints

- Never `eval`/`exec`/`shell=True`; all paths through `dw/security.py` (spec: Security Rules).
- Derivation reads the raw definition only — no variable substitution, no type loading (spec §1.2).
- Shape vocabulary is closed: `image`, `image-set`, `image-edit`, `shot`, `sequence`, `audio`, `text`, `utility` (spec §1.1).
- Trait vocabulary is closed: `speech`, `chained`, `image-conditioned`, `identity-referenced`, `needs-input-media`, `composes-workflows` (spec §1.1).
- `summary` ≤ 120 characters (spec §1.1).
- The no-param `/api/workflows` response is today's response plus four new fields per entry — nothing removed (spec §1.4).
- No model-family names anywhere in derivation code (spec "Principle").
- `tests/test_schema.py::test_every_key_the_code_reads_is_declared` scans code for definition keys; any new top-level key read must be declared in `dw/workflow_schema.json`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File map

| file | responsibility |
|---|---|
| `dw/server/catalog_shape.py` (new) | vocabularies; `derive_catalog_metadata(definition)`; `project_listing(details, ...)` |
| `dw/server/app.py` | `workflow_details` calls derivation and resolves inheritance; `/api/workflows` takes query params; `PUT` returns derived metadata; `submit_job` passes the catalog name |
| `dw/workflow_schema.json` | four optional top-level properties |
| `dw/server/jobs.py` | `workflow_name` column, `catalog_name` on spec/Job, in summaries |
| `dw_mcp/catalog.py`, `dw_mcp/server.py` | `list_workflows` params, compact view, instructions sentence |
| `tests/test_catalog_shape.py` (new) | derivation and projection unit tests |
| `tests/test_catalog_structure.py` | catalog-wide invariants |
| `tests/test_server.py`, `tests/test_mcp_catalog.py`, `tests/test_mcp_server.py` | route and tool tests |

---

### Task 1: derivation module `[model: opus]`

**Files:**
- Create: `dw/server/catalog_shape.py`
- Test: `tests/test_catalog_shape.py`

**Interfaces:**
- Produces: `SHAPES: tuple[str]`, `TRAITS: tuple[str]`, `GENERATIVE_TASKS: frozenset[str]`, `SUMMARY_LIMIT = 120`, `derive_catalog_metadata(definition: dict) -> dict` with keys `shape: str`, `traits: list[str]` (sorted), `summary: str`, `summary_truncated: bool`, `declared: set[str]` (which of `shape`/`traits`/`summary` came from a declaration). Later tasks import all of these.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_catalog_shape.py
"""Shape, trait and summary derivation over raw workflow definitions.

Every rule in spec §1.2 gets a minimal definition that exercises only it.
"""

import pytest

from dw.server.catalog_shape import (
    GENERATIVE_TASKS,
    SHAPES,
    SUMMARY_LIMIT,
    TRAITS,
    derive_catalog_metadata,
)


def pipeline_step(name, content_type, component_type="{Fake}", arguments=None, chain=None):
    pipeline = {
        "configuration": {"component_type": component_type},
        "from_pretrained_arguments": {"model_name": "m"},
        "arguments": arguments or {"prompt": "variable:prompt"},
    }
    if chain is not None:
        pipeline["chain"] = chain
    return {"name": name, "pipeline": pipeline, "result": {"content_type": content_type}}


def task_step(name, command, arguments, content_type=None):
    step = {"name": name, "task": {"command": command, "arguments": arguments}}
    if content_type:
        step["result"] = {"content_type": content_type}
    return step


def definition(*steps, **top):
    return {"id": "t", "steps": list(steps), **top}


def test_vocabularies_are_closed_and_stable():
    assert SHAPES == ("image", "image-set", "image-edit", "shot", "sequence", "audio", "text", "utility")
    assert TRAITS == (
        "speech", "chained", "image-conditioned", "identity-referenced",
        "needs-input-media", "composes-workflows",
    )
    assert GENERATIVE_TASKS == frozenset(
        {"generate_speech", "text_generation", "image_to_text", "diffusion_upscale", "interpolate_frames"}
    )


def test_a_single_still_is_image():
    meta = derive_catalog_metadata(definition(pipeline_step("gen", "image/jpeg")))
    assert meta["shape"] == "image"
    assert meta["traits"] == []


def test_two_image_steps_are_an_image_set():
    meta = derive_catalog_metadata(
        definition(pipeline_step("a", "image/jpeg"), pipeline_step("b", "image/jpeg"))
    )
    assert meta["shape"] == "image-set"


def test_a_workflow_step_emitting_images_is_an_image_set():
    step = {
        "name": "sub",
        "workflow": {"path": "x.json", "arguments": {"prompt": "variable:prompt"}},
        "result": {"content_type": "image/jpeg"},
    }
    meta = derive_catalog_metadata(definition(step))
    assert meta["shape"] == "image-set"
    assert "composes-workflows" in meta["traits"]


@pytest.mark.parametrize(
    "component_type", ["FluxImg2ImgPipeline", "StableDiffusionInpaintPipeline", "QwenImageEditPipeline", "FluxKontextPipeline", "StableDiffusionUpscalePipeline"]
)
def test_an_editing_pipeline_is_image_edit(component_type):
    meta = derive_catalog_metadata(definition(pipeline_step("gen", "image/jpeg", component_type)))
    assert meta["shape"] == "image-edit"


def test_an_image_argument_on_an_image_pipeline_is_image_edit():
    step = pipeline_step("gen", "image/jpeg", arguments={"prompt": "p", "image": "variable:image"})
    meta = derive_catalog_metadata(definition(step))
    assert meta["shape"] == "image-edit"
    assert "needs-input-media" in meta["traits"]


def test_one_clip_is_a_shot():
    assert derive_catalog_metadata(definition(pipeline_step("v", "video/mp4")))["shape"] == "shot"


def test_a_concat_fed_by_two_steps_is_a_sequence():
    meta = derive_catalog_metadata(
        definition(
            pipeline_step("a", "video/mp4"),
            pipeline_step("b", "video/mp4"),
            task_step("cut", "concat_videos", {"videos": ["previous_result:a", "previous_result:b"]}, "video/mp4"),
        )
    )
    assert meta["shape"] == "sequence"


def test_a_dissolve_fed_by_two_steps_is_a_sequence():
    meta = derive_catalog_metadata(
        definition(
            pipeline_step("a", "video/mp4"),
            pipeline_step("b", "video/mp4"),
            task_step("cut", "dissolve_videos", {"videos": ["previous_result:a", "previous_result:b"]}, "video/mp4"),
        )
    )
    assert meta["shape"] == "sequence"


def test_a_concat_fed_by_one_step_is_still_a_shot():
    meta = derive_catalog_metadata(
        definition(
            pipeline_step("a", "video/mp4"),
            task_step("cut", "concat_videos", {"videos": ["previous_result:a", "previous_result:a"]}, "video/mp4"),
        )
    )
    assert meta["shape"] == "shot"


def test_a_chain_is_a_chained_shot_not_a_sequence():
    step = pipeline_step("v", "video/mp4", chain={"segments": 3})
    meta = derive_catalog_metadata(definition(step))
    assert meta["shape"] == "shot"
    assert "chained" in meta["traits"]


@pytest.mark.parametrize("name", ["last_frame", "last_segment", "last_image", "match_audio"])
def test_continuation_arguments_are_chained(name):
    step = pipeline_step("v", "video/mp4", arguments={"prompt": "p", name: "previous_result:x"})
    assert "chained" in derive_catalog_metadata(definition(step))["traits"]


def test_video_outranks_image_when_both_are_produced():
    meta = derive_catalog_metadata(
        definition(pipeline_step("board", "image/jpeg"), pipeline_step("v", "video/mp4"))
    )
    assert meta["shape"] == "shot"


def test_audio_only_is_audio():
    step = task_step("speak", "generate_speech", {"text": "variable:text"}, "audio/wav")
    meta = derive_catalog_metadata(definition(step))
    assert meta["shape"] == "audio"
    assert "speech" in meta["traits"]


def test_text_only_is_text():
    step = task_step("expand", "text_generation", {"prompt": "variable:prompt"}, "text/plain")
    assert derive_catalog_metadata(definition(step))["shape"] == "text"


def test_processing_tasks_alone_are_utility():
    step = task_step("crop", "crop_square", {"image": "asset:a.png"}, "image/jpeg")
    meta = derive_catalog_metadata(definition(step))
    assert meta["shape"] == "utility"
    assert "needs-input-media" in meta["traits"]


def test_a_generative_task_is_not_utility():
    step = task_step("up", "diffusion_upscale", {"image": "asset:a.png"}, "image/jpeg")
    assert derive_catalog_metadata(definition(step))["shape"] != "utility"


def test_a_video_pipeline_emitting_audio_speaks():
    step = pipeline_step("v", "video/mp4", arguments={"prompt": "p", "output": ["videos", "audio"]})
    assert "speech" in derive_catalog_metadata(definition(step))["traits"]


def test_a_video_pipeline_with_an_image_argument_is_image_conditioned():
    step = pipeline_step("v", "video/mp4", arguments={"prompt": "p", "image": "previous_result:still"})
    meta = derive_catalog_metadata(definition(pipeline_step("still", "image/jpeg"), step))
    assert "image-conditioned" in meta["traits"]
    # previous_result is not supplied media
    assert "needs-input-media" not in meta["traits"]


def test_an_image_to_video_component_is_image_conditioned():
    step = pipeline_step("v", "video/mp4", "LTX2ImageToVideoPipeline")
    assert "image-conditioned" in derive_catalog_metadata(definition(step))["traits"]


def test_references_are_identity_referenced():
    step = pipeline_step("v", "video/mp4", arguments={"prompt": "p", "references": ["previous_result:face"]})
    assert "identity-referenced" in derive_catalog_metadata(definition(step))["traits"]


def test_a_location_argument_needs_input_media():
    step = pipeline_step("v", "video/mp4", arguments={"prompt": "p", "image": {"location": "https://x/y.png"}})
    assert "needs-input-media" in derive_catalog_metadata(definition(step))["traits"]


def test_a_pipeline_reference_step_counts_as_generation():
    ref = {
        "name": "shot_2",
        "pipeline_reference": {"reference_name": "shot_1", "arguments": {"prompt": "p", "references": ["asset:face.png"]}},
        "result": {"content_type": "video/mp4"},
    }
    meta = derive_catalog_metadata(
        definition(
            pipeline_step("shot_1", "video/mp4"),
            ref,
            task_step("cut", "concat_videos", {"videos": ["previous_result:shot_1", "previous_result:shot_2"]}, "video/mp4"),
        )
    )
    assert meta["shape"] == "sequence"
    assert "identity-referenced" in meta["traits"]
    assert "needs-input-media" in meta["traits"]


def test_summary_is_the_first_sentence():
    meta = derive_catalog_metadata(definition(pipeline_step("g", "image/jpeg"), description="Makes a cat. Then more."))
    assert meta["summary"] == "Makes a cat."
    assert meta["summary_truncated"] is False


def test_summary_splits_on_newline_too():
    meta = derive_catalog_metadata(definition(pipeline_step("g", "image/jpeg"), description="Line one\nLine two."))
    assert meta["summary"] == "Line one"


def test_a_long_first_sentence_is_truncated_at_a_word_boundary():
    words = " ".join(["word"] * 40) + "."
    meta = derive_catalog_metadata(definition(pipeline_step("g", "image/jpeg"), description=words))
    assert len(meta["summary"]) <= SUMMARY_LIMIT
    assert meta["summary"].endswith("…")
    assert not meta["summary"][:-1].endswith(" ")
    assert meta["summary_truncated"] is True


def test_no_description_means_empty_summary():
    meta = derive_catalog_metadata(definition(pipeline_step("g", "image/jpeg")))
    assert meta["summary"] == ""


def test_declarations_override_and_are_reported():
    meta = derive_catalog_metadata(
        definition(
            pipeline_step("g", "image/jpeg"),
            description="A still.",
            shape="image-set",
            traits=["chained"],
            summary="Declared.",
        )
    )
    assert meta["shape"] == "image-set"
    assert meta["traits"] == ["chained"]
    assert meta["summary"] == "Declared."
    assert meta["declared"] == {"shape", "traits", "summary"}


def test_an_empty_or_malformed_definition_derives_something():
    assert derive_catalog_metadata({})["shape"] == "utility"
    assert derive_catalog_metadata({"steps": "nope"})["shape"] == "utility"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_catalog_shape.py -q`
Expected: ImportError on `dw.server.catalog_shape`.

- [ ] **Step 3: Implement the module**

```python
# dw/server/catalog_shape.py
"""What a workflow makes, read off its definition.

A catalog entry is chosen by shape before anything else - one still, a set,
one shot, a cut sequence - and the definition already says which, in the
steps it has and what they feed each other. Deriving it here means the
vocabulary cannot go stale against the file and needs no backfill; a
declared `shape`/`traits`/`summary` overrides for the odd workflow the
rules misread.

Reads the raw JSON only: no variable substitution, no type loading, so it
costs a dict walk and behaves the same on a repo template and a file an
agent saved a second ago. Nothing here names a model family - the rules
read structure (a concat step, a `references` argument), never checkpoints.
"""

import re

SHAPES = ("image", "image-set", "image-edit", "shot", "sequence", "audio", "text", "utility")
TRAITS = (
    "speech",
    "chained",
    "image-conditioned",
    "identity-referenced",
    "needs-input-media",
    "composes-workflows",
)
# Tasks that create content rather than process it. A workflow made only
# of processing tasks is a utility.
GENERATIVE_TASKS = frozenset(
    {"generate_speech", "text_generation", "image_to_text", "diffusion_upscale", "interpolate_frames"}
)
SUMMARY_LIMIT = 120

_KIND_PRECEDENCE = ("video", "audio", "image", "text")
_EDIT_PIPELINE = re.compile(r"inpaint|img2img|edit|upscale|outpaint|kontext", re.I)
_CHAIN_ARGUMENTS = frozenset({"last_frame", "last_segment", "last_image", "match_audio"})
_MEDIA_ARGUMENTS = frozenset({"image", "video", "audio", "mask_image"})
_CUT_TASKS = frozenset({"concat_videos", "dissolve_videos"})
_SENTENCE_END = re.compile(r"(?<=[.!?])\s|\n")


def _steps(definition):
    steps = definition.get("steps") if isinstance(definition, dict) else None
    return [step for step in steps if isinstance(step, dict)] if isinstance(steps, list) else []


def _block(step):
    """The step's one body: pipeline, pipeline_reference, task or workflow."""
    for key in ("pipeline", "pipeline_reference", "task", "workflow"):
        body = step.get(key)
        if isinstance(body, dict):
            return key, body
    return None, {}


def _arguments(step):
    _kind, body = _block(step)
    arguments = body.get("arguments")
    return arguments if isinstance(arguments, dict) else {}


def _kind(step):
    result = step.get("result")
    if isinstance(result, dict) and isinstance(result.get("content_type"), str):
        return result["content_type"].split("/")[0]
    return None


def _generates(step):
    """Whether the step creates content: a pipeline, a reference to one, a
    sub-workflow, or a generative task."""
    key, body = _block(step)
    if key in ("pipeline", "pipeline_reference", "workflow"):
        return True
    return key == "task" and body.get("command") in GENERATIVE_TASKS


def _component_type(step):
    key, body = _block(step)
    if key != "pipeline":
        return ""
    configuration = body.get("configuration")
    if not isinstance(configuration, dict):
        return ""
    return str(configuration.get("component_type", ""))


def _fed_by(value):
    """Distinct step names a value's previous_result references name."""
    found = set()
    if isinstance(value, str) and value.startswith("previous_result:"):
        found.add(value.split(":", 1)[1].split(".", 1)[0])
    elif isinstance(value, list):
        for item in value:
            found |= _fed_by(item)
    elif isinstance(value, dict):
        for item in value.values():
            found |= _fed_by(item)
    return found


def _walk(value):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _needs_input_media(steps):
    for step in steps:
        arguments = _arguments(step)
        for name, value in arguments.items():
            if name in _MEDIA_ARGUMENTS and isinstance(value, str) and value.startswith("variable:"):
                return True
        for value in _walk(arguments):
            if isinstance(value, str) and value.startswith("asset:"):
                return True
            if isinstance(value, dict) and "location" in value:
                return True
    return False


def _derive_shape(steps, kind):
    if kind is None or not any(_generates(step) for step in steps):
        return "utility"
    if kind == "text":
        return "text"
    if kind == "audio":
        return "audio"
    if kind == "video":
        for step in steps:
            key, body = _block(step)
            if key == "task" and body.get("command") in _CUT_TASKS:
                if len(_fed_by(_arguments(step).get("videos"))) >= 2:
                    return "sequence"
        return "shot"
    image_steps = [step for step in steps if _generates(step) and _kind(step) == "image"]
    for step in image_steps:
        if _EDIT_PIPELINE.search(_component_type(step)):
            return "image-edit"
        if _MEDIA_ARGUMENTS & set(_arguments(step)) & {"image", "mask_image"}:
            return "image-edit"
    if len(image_steps) >= 2 or any(_block(step)[0] == "workflow" for step in image_steps):
        return "image-set"
    return "image"


def _derive_traits(steps):
    traits = set()
    for step in steps:
        key, body = _block(step)
        arguments = _arguments(step)
        if key == "task" and body.get("command") == "generate_speech":
            traits.add("speech")
        output = arguments.get("output")
        if _kind(step) == "video" and isinstance(output, list) and "audio" in output:
            traits.add("speech")
        if key == "pipeline" and "chain" in body:
            traits.add("chained")
        if _CHAIN_ARGUMENTS & set(arguments):
            traits.add("chained")
        if _kind(step) == "video" and ("image" in arguments or "ImageToVideo" in _component_type(step)):
            traits.add("image-conditioned")
        if "references" in arguments:
            traits.add("identity-referenced")
        if key == "workflow":
            traits.add("composes-workflows")
    if _needs_input_media(steps):
        traits.add("needs-input-media")
    return sorted(traits)


def _derive_summary(description):
    text = str(description or "").strip()
    if not text:
        return "", False
    first = _SENTENCE_END.split(text, maxsplit=1)[0].strip()
    if len(first) <= SUMMARY_LIMIT:
        return first, False
    cut = first[: SUMMARY_LIMIT - 1]
    cut = cut[: cut.rfind(" ")] if " " in cut else cut
    return cut.rstrip() + "…", True


def derive_catalog_metadata(definition):
    """shape, traits and summary for one definition, declarations honoured.

    Returns {shape, traits, summary, summary_truncated, declared}; `declared`
    names which of the three came from the file rather than the rules, so a
    test can refuse a declaration that merely repeats the derivation.
    """
    if not isinstance(definition, dict):
        definition = {}
    steps = _steps(definition)
    kinds = {_kind(step) for step in steps} - {None}
    kind = next((k for k in _KIND_PRECEDENCE if k in kinds), None)

    declared = set()
    shape = _derive_shape(steps, kind)
    if definition.get("shape") in SHAPES:
        shape = definition["shape"]
        declared.add("shape")
    traits = _derive_traits(steps)
    if isinstance(definition.get("traits"), list):
        traits = sorted(t for t in definition["traits"] if t in TRAITS)
        declared.add("traits")
    summary, truncated = _derive_summary(definition.get("description"))
    if isinstance(definition.get("summary"), str) and definition["summary"].strip():
        summary, truncated = definition["summary"].strip(), False
        declared.add("summary")
    return {
        "shape": shape,
        "traits": traits,
        "summary": summary,
        "summary_truncated": truncated,
        "declared": declared,
    }
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_catalog_shape.py -q`
Expected: all pass. If `test_a_long_first_sentence_is_truncated_at_a_word_boundary` fails on the trailing-space assertion, check `_derive_summary` strips before appending `…`.

- [ ] **Step 5: Run the schema key scan**

Run: `python -m pytest tests/test_schema.py -q`
Expected: `test_every_key_the_code_reads_is_declared` FAILS for `shape`, `traits`, `summary` — that is Task 2's job. If it passes, the scan did not see the new module; check `tests/test_schema.py` for which files it scans and note it for Task 2.

- [ ] **Step 6: Commit**

```bash
git add dw/server/catalog_shape.py tests/test_catalog_shape.py
git commit -m "Derive catalog shape, traits and summary from a workflow definition

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 7: Ledger** — no row is complete yet; skip.

---

### Task 2: schema fields `[model: haiku]`

**Files:**
- Modify: `dw/workflow_schema.json` (top-level `properties`, after `configures`)
- Test: `tests/test_schema.py`, `tests/test_catalog_shape.py`

**Interfaces:**
- Produces: schema properties `shape`, `traits`, `summary`, `cost` — the names Task 1 reads and Task 8's `cost` test checks.

- [ ] **Step 1: Write the failing test** (append to `tests/test_catalog_shape.py`)

```python
from dw.schema import load_schema, validate_data


def test_the_schema_declares_the_vocabulary():
    schema = load_schema("workflow")
    props = schema["properties"]
    assert tuple(props["shape"]["enum"]) == SHAPES
    assert tuple(props["traits"]["items"]["enum"]) == TRAITS
    assert props["summary"]["maxLength"] == SUMMARY_LIMIT
    cost_item = props["cost"]["items"]
    assert set(cost_item["required"]) == {"device", "vram_gb", "minutes"}
    assert cost_item["properties"]["device"]["enum"] == ["cuda", "mps", "cpu"]


def test_a_declared_cost_validates_and_a_bad_one_does_not():
    schema = load_schema("workflow")
    base = definition(pipeline_step("g", "image/jpeg"))
    ok, _ = validate_data({**base, "cost": [{"device": "cuda", "name": "RTX 4090", "vram_gb": 22, "minutes": 3}]}, schema)
    assert ok
    bad, message = validate_data({**base, "cost": [{"device": "tpu", "vram_gb": 1, "minutes": 1}]}, schema)
    assert not bad and "cost" in message
    bad, _ = validate_data({**base, "shape": "cinematic"}, schema)
    assert not bad
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_catalog_shape.py -q -k schema`
Expected: KeyError `'shape'`.

- [ ] **Step 3: Add the properties** — in `dw/workflow_schema.json`, immediately after the `"configures": {...}` property, insert:

```json
        "shape": {
            "type": "string",
            "enum": ["image", "image-set", "image-edit", "shot", "sequence", "audio", "text", "utility"],
            "description": "What the workflow produces. Derived from the steps by the server; declare it only where the derivation is wrong. Closed vocabulary - adding a value is additive, renaming one breaks every client that filtered on it."
        },
        "traits": {
            "type": "array",
            "uniqueItems": true,
            "items": {
                "type": "string",
                "enum": ["speech", "chained", "image-conditioned", "identity-referenced", "needs-input-media", "composes-workflows"]
            },
            "description": "How the output is made or what it needs supplied. Derived by the server; declare only to override."
        },
        "summary": {
            "type": "string",
            "maxLength": 120,
            "description": "One line saying what the workflow is for, carried in catalog listings. Defaults to the first sentence of 'description'."
        },
        "cost": {
            "type": "array",
            "description": "Measured runs, one per device the maintainer measured on. Never derived; absent means unknown.",
            "items": {
                "type": "object",
                "required": ["device", "vram_gb", "minutes"],
                "properties": {
                    "device": {"type": "string", "enum": ["cuda", "mps", "cpu"]},
                    "name": {"type": "string", "description": "The accelerator, for a person: 'RTX 4090', 'M2 Ultra'"},
                    "vram_gb": {"type": "number", "minimum": 0},
                    "minutes": {"type": "number", "minimum": 0}
                },
                "additionalProperties": false
            }
        },
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_catalog_shape.py tests/test_schema.py tests/test_examples.py -q`
Expected: all pass, including the key-scan test that failed in Task 1 step 5.

- [ ] **Step 5: Commit**

```bash
git add dw/workflow_schema.json tests/test_catalog_shape.py
git commit -m "Declare shape, traits, summary and cost in the workflow schema

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 6: Ledger** — no row is complete yet; skip.

---

### Task 3: listing carries the metadata `[model: sonnet]`

**Files:**
- Modify: `dw/server/app.py` — `workflow_details` (around lines 159–237)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `derive_catalog_metadata` from Task 1.
- Produces: every entry in `details` has `shape`, `traits`, `summary`, `cost`; model configs inherit `shape`/`traits` from the template named by `configures`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_server.py`, beside `test_configures_resolves_against_the_listing`)

```python
def video_workflow(job_id, with_cost=False):
    workflow = {
        "id": job_id,
        "description": "One clip from a prompt. It runs a while.",
        "variables": {"prompt": "d"},
        "steps": [
            {
                "name": "gen",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}", "no_generator": True},
                    "from_pretrained_arguments": {"model_name": "m"},
                    "arguments": {"prompt": "variable:prompt", "output": ["videos", "audio"]},
                },
                "result": {"content_type": "video/mp4"},
            }
        ],
    }
    if with_cost:
        workflow["cost"] = [{"device": "cuda", "name": "RTX 4090", "vram_gb": 20, "minutes": 2}]
    return workflow


def test_the_listing_carries_derived_metadata(server):
    with server(success_script) as client:
        client.put("/api/workflows/templates/clip", json={"workflow": video_workflow("clip", with_cost=True)})
        tuned = video_workflow("tuned")
        tuned["configures"] = "templates/clip"
        tuned["description"] = "The same clip on a bigger checkpoint."
        client.put("/api/workflows/models/tuned", json={"workflow": tuned})

        details = client.get("/api/workflows").json()["details"]

        clip = details["templates/clip"]
        assert clip["shape"] == "shot"
        assert clip["traits"] == ["speech"]
        assert clip["summary"] == "One clip from a prompt."
        assert clip["cost"] == [{"device": "cuda", "name": "RTX 4090", "vram_gb": 20, "minutes": 2}]

        tuned = details["models/tuned"]
        assert tuned["shape"] == "shot" and tuned["traits"] == ["speech"]
        assert tuned["summary"] == "The same clip on a bigger checkpoint."
        assert tuned["cost"] is None

        basic = details["Basic"]
        assert basic["shape"] == "utility"  # no result block, so no kind
        assert basic["summary"] == "" and basic["cost"] is None
        # nothing the UI reads went away
        assert {"kinds", "steps", "variables", "variable_names", "description", "configures", "prompt_refs", "origin", "writable"} <= set(basic)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_server.py -q -k derived_metadata`
Expected: KeyError `'shape'`.

- [ ] **Step 3: Implement** — in `dw/server/app.py`:

Add the import near the other local imports (after `from .updater import DiffusersUpdater`):

```python
from .catalog_shape import derive_catalog_metadata
```

In `workflow_details`, inside the `try:` that builds `detail`, after `variables = ...` and before `detail = {`, add:

```python
            metadata = derive_catalog_metadata(definition)
            cost = definition.get("cost")
```

and add to the `detail` dict:

```python
                "shape": metadata["shape"],
                "traits": metadata["traits"],
                "summary": metadata["summary"],
                "cost": cost if isinstance(cost, list) and cost else None,
```

In the `except Exception:` fallback dict add:

```python
                "shape": "utility",
                "traits": [],
                "summary": "",
                "cost": None,
```

Replace the trailing `configures` resolution loop with one that also inherits:

```python
    # A model config names its template as a catalog name. Resolve it here,
    # where the whole listing is in hand, so a badge is a link to a real card
    # rather than a string - and say which name did not resolve. A config
    # also takes its shape and traits from the template: what it makes is
    # the template's business, what it costs is its own. Entries can be the
    # very dict cached above (a cache hit skips the copy at the origin
    # merge), so copy before mutating - otherwise a stale "not found yet"
    # verdict would stick in the cache and outlive the typo once the
    # template it names is added.
    for name, detail in details.items():
        named = detail.get("configures", "")
        if not named:
            continue
        detail = dict(detail)
        template = details.get(named)
        if template is None:
            detail["configures_missing"] = named
            detail["configures"] = ""
        else:
            detail["shape"] = template["shape"]
            detail["traits"] = list(template["traits"])
        details[name] = detail
    return details
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_server.py tests/test_server_workspaces.py tests/test_library_sources.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add dw/server/app.py tests/test_server.py
git commit -m "Carry shape, traits, summary and cost in the workflow listing

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 6: Ledger** — no row is complete yet; skip.

---

### Task 4: filter and compact projection `[model: sonnet]`

**Files:**
- Modify: `dw/server/catalog_shape.py` (add `project_listing`), `dw/server/app.py` (`/api/workflows` route, ~line 1109)
- Test: `tests/test_catalog_shape.py`, `tests/test_server.py`

**Interfaces:**
- Produces: `project_listing(details: dict, *, shape=None, traits=None, configures=None, include_models=False, view=None) -> dict`; raises `ValueError` with a message naming the vocabulary on an unknown shape/trait. `COMPACT_FIELDS: tuple[str]`. Route query params `shape`, `traits` (comma-separated), `configures`, `include_models` (bool), `view` (`compact` or omitted).

- [ ] **Step 1: Write the failing unit tests** (append to `tests/test_catalog_shape.py`)

```python
from dw.server.catalog_shape import COMPACT_FIELDS, project_listing


def entry(shape, traits=(), configures="", **extra):
    return {
        "kinds": [], "steps": 1, "variables": 0, "variable_names": [],
        "description": "long text", "configures": configures, "prompt_refs": [],
        "origin": "workspace", "writable": True,
        "shape": shape, "traits": sorted(traits), "summary": "short", "cost": None,
        **extra,
    }


LISTING = {
    "templates/tti": entry("image"),
    "templates/talk": entry("sequence", ["speech", "identity-referenced"]),
    "templates/clip": entry("shot", ["speech"]),
    "models/flux": entry("image", configures="templates/tti"),
    "mine": entry("image"),
}


def test_no_options_returns_the_listing_untouched():
    assert project_listing(LISTING) == LISTING


def test_shape_filters():
    assert set(project_listing(LISTING, shape="shot")) == {"templates/clip"}


def test_traits_must_all_match():
    assert set(project_listing(LISTING, traits=["speech"])) == {"templates/talk", "templates/clip"}
    assert set(project_listing(LISTING, traits=["speech", "identity-referenced"])) == {"templates/talk"}


def test_configures_filters_to_a_templates_configs():
    assert set(project_listing(LISTING, configures="templates/tti")) == {"models/flux"}


def test_compact_drops_prose_and_model_configs_and_keeps_user_workflows():
    compact = project_listing(LISTING, view="compact")
    assert set(compact) == {"templates/tti", "templates/talk", "templates/clip", "mine"}
    assert set(compact["templates/tti"]) == set(COMPACT_FIELDS)
    assert "description" not in compact["templates/tti"]


def test_compact_with_include_models_keeps_them():
    assert "models/flux" in project_listing(LISTING, view="compact", include_models=True)


def test_compact_with_configures_implies_models():
    assert set(project_listing(LISTING, view="compact", configures="templates/tti")) == {"models/flux"}


def test_compact_keeps_configures_missing_when_set():
    listing = {"models/typo": entry("image", configures="", configures_missing="templates/nope")}
    compact = project_listing(listing, view="compact", include_models=True)
    assert compact["models/typo"]["configures_missing"] == "templates/nope"


def test_unknown_shape_or_trait_names_the_vocabulary():
    with pytest.raises(ValueError) as caught:
        project_listing(LISTING, shape="cinematic")
    assert "sequence" in str(caught.value)
    with pytest.raises(ValueError) as caught:
        project_listing(LISTING, traits=["fast"])
    assert "speech" in str(caught.value)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_catalog_shape.py -q -k "project or compact or filters"`
Expected: ImportError on `project_listing`.

- [ ] **Step 3: Implement `project_listing`** (append to `dw/server/catalog_shape.py`)

```python
COMPACT_FIELDS = (
    "summary",
    "shape",
    "traits",
    "cost",
    "kinds",
    "steps",
    "variables",
    "variable_names",
    "configures",
)


def project_listing(details, *, shape=None, traits=None, configures=None, include_models=False, view=None):
    """The listing an agent asked for: filtered by shape and traits, and in
    the compact view stripped to what choosing a template needs.

    Compact is templates-only unless `include_models` or `configures` says
    otherwise - nine checkpoint variants of text-to-image are the noise the
    two-tree split removed. A workflow with no `configures` is a template
    for this purpose, whichever directory it sits in: a user wrote it to be
    found. The full view never drops entries or fields.
    """
    if shape is not None and shape not in SHAPES:
        raise ValueError(f"Unknown shape {shape!r}. The shapes are: {', '.join(SHAPES)}.")
    traits = list(traits or [])
    unknown = [t for t in traits if t not in TRAITS]
    if unknown:
        raise ValueError(f"Unknown trait(s) {', '.join(unknown)}. The traits are: {', '.join(TRAITS)}.")
    if view not in (None, "compact"):
        raise ValueError("view must be 'compact' or omitted")

    compact = view == "compact"
    keep_models = include_models or configures is not None or not compact
    projected = {}
    for name, detail in details.items():
        is_model = bool(detail.get("configures") or detail.get("configures_missing"))
        if shape is not None and detail.get("shape") != shape:
            continue
        if traits and not set(traits) <= set(detail.get("traits", [])):
            continue
        if configures is not None and detail.get("configures") != configures:
            continue
        if is_model and not keep_models:
            continue
        if compact:
            slim = {key: detail.get(key) for key in COMPACT_FIELDS}
            if detail.get("configures_missing"):
                slim["configures_missing"] = detail["configures_missing"]
            projected[name] = slim
        else:
            projected[name] = detail
    return projected
```

- [ ] **Step 4: Run unit tests**

Run: `python -m pytest tests/test_catalog_shape.py -q`
Expected: all pass.

- [ ] **Step 5: Write the failing route test** (append to `tests/test_server.py`)

```python
def test_the_listing_filters_and_compacts(server):
    with server(success_script) as client:
        client.put("/api/workflows/templates/clip", json={"workflow": video_workflow("clip")})
        tuned = video_workflow("tuned")
        tuned["configures"] = "templates/clip"
        client.put("/api/workflows/models/tuned", json={"workflow": tuned})

        full = client.get("/api/workflows").json()
        assert set(full["details"]) == {"Basic", "templates/clip", "models/tuned"}

        by_shape = client.get("/api/workflows", params={"shape": "shot"}).json()
        assert set(by_shape["details"]) == {"templates/clip", "models/tuned"}
        assert by_shape["workflows"] == sorted(by_shape["details"])

        compact = client.get("/api/workflows", params={"view": "compact"}).json()
        assert set(compact["details"]) == {"Basic", "templates/clip"}
        assert "description" not in compact["details"]["templates/clip"]
        assert compact["details"]["templates/clip"]["summary"] == "One clip from a prompt."

        with_models = client.get("/api/workflows", params={"view": "compact", "include_models": "true"}).json()
        assert "models/tuned" in with_models["details"]

        configs = client.get("/api/workflows", params={"configures": "templates/clip"}).json()
        assert set(configs["details"]) == {"models/tuned"}

        by_trait = client.get("/api/workflows", params={"traits": "speech"}).json()
        assert "Basic" not in by_trait["details"]

        bad = client.get("/api/workflows", params={"shape": "cinematic"})
        assert bad.status_code == 400 and "sequence" in bad.json()["detail"]
        bad = client.get("/api/workflows", params={"traits": "speech,fast"})
        assert bad.status_code == 400 and "fast" in bad.json()["detail"]
```

- [ ] **Step 6: Run to verify failure**

Run: `python -m pytest tests/test_server.py -q -k filters_and_compacts`
Expected: FAIL — `by_shape["details"]` still holds `Basic`.

- [ ] **Step 7: Implement the route** — replace the `/api/workflows` GET handler in `dw/server/app.py`:

```python
    @app.get("/api/workflows")
    def list_workflows(
        ws: Workspace = Depends(selected_workspace),
        shape: Optional[str] = None,
        traits: Optional[str] = None,
        configures: Optional[str] = None,
        include_models: bool = False,
        view: Optional[str] = None,
    ):
        """Every workflow the search path offers, each detail saying which
        source it came from and whether it can be written to. 'workflow_dir'
        stays the writable one - what a save targets.

        `shape`, `traits` (comma-separated, all must match) and `configures`
        narrow the listing; `view=compact` is the agent's view - summaries
        rather than descriptions, templates rather than model configs
        unless `include_models` asks for them. `workflows` always names
        exactly the entries `details` holds.
        """
        sources = _sources_for(ws)
        found = listing(sources)
        try:
            details = project_listing(
                workflow_details(found),
                shape=shape,
                traits=[t for t in (traits or "").split(",") if t],
                configures=configures,
                include_models=include_models,
                view=view,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "workspace": ws.name,
            "workflow_dir": ws.workflows,
            "sources": [source.to_dict() for source in sources],
            "workflows": sorted(details),
            "details": details,
        }
```

and extend the import: `from .catalog_shape import derive_catalog_metadata, project_listing`.

Check `test_workflow_browsing_and_confinement` still passes — `sorted(details)` must equal `list(found)` for the no-param case; if `listing()` returned names in a different order, change `sorted(details)` to `[name for name in found if name in details]`.

- [ ] **Step 8: Run tests**

Run: `python -m pytest tests/test_server.py tests/test_server_workspaces.py tests/test_library_sources.py tests/test_catalog_shape.py -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add dw/server/catalog_shape.py dw/server/app.py tests/test_catalog_shape.py tests/test_server.py
git commit -m "Filter /api/workflows by shape, traits and configures; add the compact view

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 10: Ledger** — in `docs/proposals/agent-catalog-legibility-complete.md`, row `1 shape vocabulary`: change `designed` to `done (tasks 1–4)`. Row `2 \`summary\``: change to `done (tasks 1, 3)`. Commit: `git commit -am "Ledger: shape and summary landed"` with the co-author trailer.

---

### Task 5: MCP `list_workflows` and instructions `[model: sonnet]`

**Files:**
- Modify: `dw_mcp/catalog.py` (`list_workflows`), `dw_mcp/server.py` (tool at ~line 143; instructions ~line 74–80)
- Test: `tests/test_mcp_catalog.py`, `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: the route params from Task 4.
- Produces: `catalog.list_workflows(client, shape=None, traits=None, configures=None, include_models=False)`; MCP tool `list_workflows(shape: Optional[str] = None, traits: Optional[str] = None, configures: Optional[str] = None, include_models: bool = False)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_mcp_catalog.py`, replace the `(lambda c: catalog.list_workflows(c), "/api/workflows")` entry's expectations by adding, after `test_a_pass_through_tool_returns_the_body_unchanged`:

```python
def test_list_workflows_always_asks_for_the_compact_view():
    client, seen = recording_client({"workflows": [], "details": {}})

    catalog.list_workflows(client)

    assert seen["path"] == "/api/workflows"
    assert seen["params"] == {"view": "compact"}


def test_list_workflows_passes_its_filters_through():
    client, seen = recording_client({"workflows": [], "details": {}})

    catalog.list_workflows(
        client, shape="sequence", traits=["speech", "chained"], configures="templates/x", include_models=True
    )

    assert seen["params"] == {
        "view": "compact",
        "shape": "sequence",
        "traits": "speech,chained",
        "configures": "templates/x",
        "include_models": "true",
    }
```

In `tests/test_mcp_server.py`, add:

```python
@pytest.mark.asyncio
async def test_list_workflows_takes_shape_and_traits():
    tools = await tools_of(server_over(ok({})))
    schema = tools["list_workflows"].inputSchema
    assert {"shape", "traits", "configures", "include_models"} <= set(schema["properties"])
    assert "shape" in tools["list_workflows"].description


@pytest.mark.asyncio
async def test_the_instructions_name_the_vocabulary():
    server = server_over(ok({}))
    for word in ("image-set", "sequence", "speech", "list_workflows(shape="):
        assert word in server.instructions
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mcp_catalog.py tests/test_mcp_server.py -q -k "list_workflows or instructions"`
Expected: `seen["params"]` is `{}`; the schema lacks `shape`.

- [ ] **Step 3: Implement**

`dw_mcp/catalog.py`:

```python
def list_workflows(client, shape=None, traits=None, configures=None, include_models=False):
    """Workflow names the server can reach, in the compact view: summary,
    shape, traits, cost, output kinds and variable names per workflow -
    what choosing one needs and nothing that reading one needs. Templates
    only unless `include_models` or `configures` asks for the model configs
    of one template. `get_workflow` has the full definition."""
    params = {"view": "compact"}
    if shape:
        params["shape"] = shape
    if traits:
        params["traits"] = ",".join(traits) if isinstance(traits, (list, tuple)) else traits
    if configures:
        params["configures"] = configures
    if include_models:
        params["include_models"] = "true"
    return client.get_json("/api/workflows", params=params)
```

`dw_mcp/server.py` tool (keep it in the same place, same `tool(fn, READ_ONLY)` registration):

```python
    def list_workflows(
        shape: Optional[str] = None,
        traits: Optional[str] = None,
        configures: Optional[str] = None,
        include_models: bool = False,
    ) -> dict:
        """List the workflows stored on the server, compactly. Decide the
        deliverable's shape first and pass it: one of image, image-set,
        image-edit, shot, sequence, audio, text, utility. `traits` narrows
        further (comma-separated, all must match): speech, chained,
        image-conditioned, identity-referenced, needs-input-media,
        composes-workflows. Each entry carries a one-line `summary`, its
        `shape` and `traits` (what it needs supplied), `cost` (measured
        runs per device; null means unknown - call `get_memory` and say
        so), output kinds and variable names. Templates only by default;
        `configures=<template>` lists the checkpoint configs tuned for
        one, `include_models=true` lists them all. `get_workflow` has the
        full description and definition."""
        return catalog.list_workflows(
            client,
            shape=shape,
            traits=[t for t in (traits or "").split(",") if t],
            configures=configures,
            include_models=include_models,
        )
```

Add `from typing import Optional` at the top of `dw_mcp/server.py` if it is not already imported.

Instructions: replace the paragraph beginning `"Start from \`list_workflows\`: ..."` with:

```python
            "Start from `list_workflows(shape=...)`: the server keeps a "
            "large catalog, and its compact listing carries each "
            "workflow's summary, shape, traits, cost and variable names - "
            "run what is already there, with `arguments` overriding its "
            "variables, rather than authoring a new workflow for a "
            "request an existing one covers. Shapes: image, image-set, "
            "image-edit, shot, sequence, audio, text, utility. Traits: "
            "speech, chained, image-conditioned, identity-referenced, "
            "needs-input-media, composes-workflows.\n"
```

and in the next paragraph change `"Decide which shape the deliverable is first, then match the catalog against that; "` to `"Decide which shape the deliverable is first, then call `list_workflows` with it; "`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_mcp_catalog.py tests/test_mcp_server.py tests/test_server_mcp.py -q`
Expected: all pass. `TOOL_WIRING`'s `("list_workflows", {}, "GET", "/api/workflows")` still matches on path; if it asserts empty params, update that expectation to `{"view": "compact"}`.

- [ ] **Step 5: Measure**

Run against the repo catalog, from a checkout with the venv active:

```bash
python - <<'EOF'
import json, os
from dw.workflow_sources import WorkflowSource, listing
from dw.server.app import workflow_details
from dw.server.catalog_shape import project_listing
found = listing([WorkflowSource("workflows", "workspace", True)])
details = workflow_details(found)
full = len(json.dumps({"workflows": sorted(details), "details": details})) / 4
compact = project_listing(details, view="compact")
seq = project_listing(details, view="compact", shape="sequence")
print("full", int(full), "compact", int(len(json.dumps(compact)) / 4), "sequence", int(len(json.dumps(seq)) / 4))
EOF
```

Record the three numbers in the commit message. If `compact` exceeds 5,500 or `sequence` exceeds 1,500, note it — Task 8's summary sweep is expected to bring it under; Task 10 enforces it.

- [ ] **Step 6: Commit**

```bash
git add dw_mcp/catalog.py dw_mcp/server.py tests/test_mcp_catalog.py tests/test_mcp_server.py
git commit -m "MCP list_workflows takes shape/traits and reads the compact view

Measured over the repo catalog: full <n>, compact <n>, sequence <n> tokens.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 7: Ledger** — no new row completes; append to row 1's notes: `MCP tool and instructions updated (task 5)`. Commit with the co-author trailer.

---

### Task 6: save returns derived metadata `[model: haiku]`

**Files:**
- Modify: `dw/server/app.py` — `save_workflow` PUT handler (~line 1124–1158)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `derive_catalog_metadata`.
- Produces: PUT response gains `shape`, `traits`, `summary`; `warnings` gains a string when the summary is empty.

- [ ] **Step 1: Write the failing test** (append to `tests/test_server.py`)

```python
def test_saving_reports_how_the_workflow_will_be_matched(server):
    with server(success_script) as client:
        saved = client.put("/api/workflows/clip", json={"workflow": video_workflow("clip")}).json()
        assert saved["shape"] == "shot"
        assert saved["traits"] == ["speech"]
        assert saved["summary"] == "One clip from a prompt."
        assert not any("summary" in w for w in saved["warnings"])

        bare = valid_workflow("bare")
        saved = client.put("/api/workflows/bare", json={"workflow": bare}).json()
        assert saved["summary"] == ""
        assert any("summary" in w and "description" in w for w in saved["warnings"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_server.py -q -k how_the_workflow_will_be_matched`
Expected: KeyError `'shape'`.

- [ ] **Step 3: Implement** — replace the `return {...}` at the end of `save_workflow` with:

```python
        # What the catalog will say about it, so the author sees the match
        # it just created. An empty summary is a warning, never a refusal:
        # a workflow with no description still runs, it is just invisible
        # to shape-first discovery
        metadata = derive_catalog_metadata(request.workflow)
        warnings = list(workflow_argument_warnings(request.workflow))
        if not metadata["summary"]:
            warnings.append(
                "No summary: add a 'description' (its first sentence becomes "
                "the catalog summary) or a 'summary' so the listing can say "
                "what this workflow is for"
            )
        return {
            "name": name,
            "path": path,
            "warnings": warnings,
            "shape": metadata["shape"],
            "traits": metadata["traits"],
            "summary": metadata["summary"],
        }
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_server.py tests/test_mcp_authoring.py -q`
Expected: all pass (the MCP `save_workflow` tool passes the body through unchanged).

- [ ] **Step 5: Commit**

```bash
git add dw/server/app.py tests/test_server.py
git commit -m "Saving a workflow reports its derived shape, traits and summary

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 6: Ledger** — row `10 save-time metadata`: `done (task 6)`. Commit with the co-author trailer.

---

### Task 7: jobs record the catalog name `[model: sonnet]`

**Files:**
- Modify: `dw/server/jobs.py` (schema/migration ~line 70–98, `record` ~104, `recent_summaries` ~134, `get` ~160, `_to_detail` ~259, `Job.__init__` ~288, `Job.summary` ~330, `RERUN_SPEC_KEYS` ~36, `submit` ~394), `dw/server/app.py` (`submit_job` ~735)
- Test: `tests/test_server.py`

**Interfaces:**
- Produces: `JobManager.submit(..., catalog_name=None)`; `spec["catalog_name"]`; `Job.catalog_name`; job summaries/details carry `"workflow_name"` (the catalog name, or `None`); sqlite column `workflow_name`.

Note the existing attribute `Job.workflow_name` holds the definition's **`id`** and feeds the `workflow` field — do not rename it. The new value is `catalog_name` internally and `workflow_name` on the wire, per spec §1.7.

- [ ] **Step 1: Write the failing test** (append to `tests/test_server.py`)

```python
def test_a_job_remembers_the_catalog_name_it_ran_from(server, tmp_path):
    with server(success_script) as client:
        job = client.post("/api/jobs", json={"workflow_path": "Basic"}).json()
        assert job["workflow"] == "basic"          # the definition's id, as before
        assert job["workflow_name"] == "Basic"     # the catalog name
        wait_for_status(client, job["id"], TERMINAL_STATES)

        listed = {j["id"]: j for j in client.get("/api/jobs").json()["jobs"]}
        assert listed[job["id"]]["workflow_name"] == "Basic"

        inline = client.post("/api/jobs", json={"workflow": valid_workflow("inline")}).json()
        assert inline["workflow_name"] is None
        wait_for_status(client, inline["id"], TERMINAL_STATES)

    # history, read back by a fresh manager over the same database
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    detail = manager.get(job["id"])
    assert detail["workflow_name"] == "Basic"
    assert manager.get(inline["id"])["workflow_name"] is None


def test_an_old_history_database_gains_the_column(tmp_path):
    import sqlite3
    db = tmp_path / "old.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, workflow TEXT, status TEXT, created_at REAL,"
            " started_at REAL, finished_at REAL, arguments TEXT, spec TEXT, manifest TEXT,"
            " warnings TEXT, error TEXT)"
        )
        connection.execute("INSERT INTO jobs (id, workflow, status) VALUES ('old1', 'sd', 'finished')")
    manager = JobManager(str(tmp_path / "outputs"), worker_manager=ScriptedWorkerManager(success_script), history_path=str(db))
    assert manager.get("old1")["workflow_name"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_server.py -q -k "catalog_name or gains_the_column"`
Expected: KeyError `'workflow_name'`.

- [ ] **Step 3: Implement** in `dw/server/jobs.py`:

Migration — after the `workspace` block in `JobHistory.__init__`:

```python
            # The catalog name the job was run from, beside `workflow` (the
            # definition's id). Ids are not unique across a catalog forever;
            # names are, and a later runtime-by-workflow join wants the exact
            # one. Rows before this column stay NULL: old history is
            # unjoinable, new history is exact
            if "workflow_name" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN workflow_name TEXT")
```

`RERUN_SPEC_KEYS` — add `"catalog_name",` after `"asset_dir",` with the comment `# so a rerun is attributed to the same catalog entry`.

`record` — add `workflow_name` to the column list and `job.catalog_name` to the values tuple (14 placeholders).

`recent_summaries` — add `workflow_name` to the SELECT (after `workspace`) and `"workflow_name": row[7],` to the dict.

`get` — add `workflow_name` to the SELECT (after `workspace`); in `_to_detail` add `"workflow_name": row[12],`.

`Job.__init__` — add `self.catalog_name = spec.get("catalog_name")`.

`Job.summary` — add `"workflow_name": self.catalog_name,` after `"workflow"`.

`submit` — add the parameter `catalog_name=None` after `workspace=None`, document it in the docstring (`\`catalog_name\` is the listing name the caller resolved \`workflow_path\` from, kept for history; None for an inline definition`), and after `spec["workspace"] = workspace` add `spec["catalog_name"] = catalog_name`.

`dw/server/app.py` `submit_job` — add to the `manager.submit(` call:

```python
                # The listing name, when the request came as one - what a
                # later runtime-by-workflow report joins on
                catalog_name=request.workflow_path if source else None,
```

Check `rerun` passes `spec` through to `submit(**spec, ...)`; if it enumerates keyword arguments instead, add `catalog_name=spec.get("catalog_name")`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_server.py tests/test_server_workspaces.py tests/test_jobs_reused.py tests/test_mcp_diagnose.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add dw/server/jobs.py dw/server/app.py tests/test_server.py
git commit -m "Record the catalog name a job ran from, beside the workflow id

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 6: Ledger** — row `3 \`cost\``: note `workflow_name on jobs landed (task 7)`; status stays `designed` until Task 9. Commit with the co-author trailer.

---

### Task 8: catalog invariants and the description sweep `[model: opus]`

**Files:**
- Modify: `tests/test_catalog_structure.py`, and whichever `workflows/templates/**/*.json` the tests point at
- Consumes: `derive_catalog_metadata`, `SHAPES`, `SUMMARY_LIMIT`.

This task edits real templates. Editing a `description` changes what a person reads, so keep every edit to the **first sentence** and keep the rest of the paragraph intact; add a `summary` field only when the first sentence cannot be made to serve as one. Add `shape`/`traits` only where derivation is genuinely wrong for that file — and if a rule is wrong for several files, fix the rule in `catalog_shape.py` (with a unit test) instead.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_catalog_structure.py`)

```python
from dw.server.catalog_shape import SHAPES, SUMMARY_LIMIT, derive_catalog_metadata


def load(path):
    return json.load(open(os.path.join(REPO_ROOT, path), encoding="utf-8"))


# The regression net for the derivation rules: real templates whose shape
# is subtle enough that a rule change should have to answer to them.
EXPECTED_SHAPES = {
    "workflows/templates/text-to-image.json": ("image", []),
    "workflows/templates/minimax/storyboard.json": ("shot", ["identity-referenced"]),
    "workflows/templates/minimax/dialogue-short.json": ("sequence", ["identity-referenced", "speech"]),
    "workflows/templates/minimax/music-video.json": ("sequence", ["identity-referenced", "speech"]),
    "workflows/templates/minimax/chained-segments.json": ("shot", ["chained", "image-conditioned", "needs-input-media", "speech"]),
    "workflows/templates/ltx2/chained-segments.json": ("shot", ["chained", "image-conditioned", "needs-input-media", "speech"]),
    "workflows/templates/image-variation.json": ("image-edit", ["needs-input-media"]),
    "workflows/templates/segment-and-inpaint.json": ("image-edit", ["needs-input-media"]),
    "workflows/templates/describe-and-regenerate.json": ("image-set", ["composes-workflows", "needs-input-media"]),
    "workflows/templates/compose-workflows.json": ("image-set", ["composes-workflows"]),
    "workflows/templates/generate-speech.json": ("audio", ["speech"]),
    "workflows/templates/assemble-and-score.json": ("sequence", ["needs-input-media"]),
    "workflows/templates/image-processors.json": ("utility", ["needs-input-media"]),
}

# Templates that genuinely are utilities - processing with no generation.
# Anything else that derives 'utility' is a rule that missed.
UTILITIES = {
    "workflows/templates/audio-trim-fade.json",
    "workflows/templates/embed-metadata.json",
    "workflows/templates/image-processors.json",
    "workflows/templates/recenter-crop.json",
    "workflows/templates/segment.json",
    "workflows/templates/upscale-spandrel.json",
}


@pytest.mark.parametrize("path, expected", sorted(EXPECTED_SHAPES.items()))
def test_the_rules_read_these_templates_as_expected(path, expected):
    meta = derive_catalog_metadata(load(path))
    assert (meta["shape"], meta["traits"]) == expected


@pytest.mark.parametrize("path", TEMPLATES)
def test_no_template_falls_through_to_utility(path):
    meta = derive_catalog_metadata(load(path))
    if meta["shape"] == "utility":
        assert path in UTILITIES, f"{path} derived 'utility' - a rule missed it, or add it to UTILITIES"
    else:
        assert path not in UTILITIES, f"{path} is listed as a utility but derives {meta['shape']}"


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_declaration_must_differ_from_the_derivation(path):
    """An override that repeats the rules is noise that will rot when the
    rules or the file change. Declare only what derivation gets wrong."""
    definition = load(path)
    meta = derive_catalog_metadata(definition)
    stripped = {k: v for k, v in definition.items() if k not in ("shape", "traits", "summary")}
    derived = derive_catalog_metadata(stripped)
    for key in meta["declared"]:
        assert meta[key] != derived[key], f"{path} declares {key}={meta[key]!r}, which derivation already produces"


@pytest.mark.parametrize("path", TEMPLATES)
def test_every_template_has_a_summary_that_fits(path):
    meta = derive_catalog_metadata(load(path))
    assert meta["summary"], f"{path}: description has no first sentence"
    assert len(meta["summary"]) <= SUMMARY_LIMIT
    assert not meta["summary_truncated"], (
        f"{path}: first sentence runs past {SUMMARY_LIMIT} chars - shorten it or declare 'summary': {meta['summary']!r}"
    )
```

Note `EXPECTED_SHAPES` values are the author's best reading before running. The step below is where they get checked against reality.

- [ ] **Step 2: Run and read the failures**

Run: `python -m pytest tests/test_catalog_structure.py -q 2>&1 | tail -60`

For each `EXPECTED_SHAPES` failure, open the template and decide: is the **rule** wrong (fix in `catalog_shape.py` + unit test in `tests/test_catalog_shape.py`), the **expectation** wrong (fix the table), or the **file** an odd one (declare `shape`/`traits` on it)? Record the decision in the commit message. For each `UTILITIES` failure, same three-way choice. For each summary failure, rewrite the first sentence of that description (≤ 120 chars, says what it is for) or add a `summary`.

- [ ] **Step 3: Run until green**

Run: `python -m pytest tests/test_catalog_structure.py tests/test_catalog_shape.py tests/test_examples.py -q`
Expected: all pass.

- [ ] **Step 4: Re-measure** — run the measurement script from Task 5 step 5 and note the numbers.

- [ ] **Step 5: Commit**

```bash
git add tests/test_catalog_structure.py dw/server/catalog_shape.py tests/test_catalog_shape.py workflows/
git commit -m "Hold every template to a derived shape and a fitting summary

<one line per rule/expectation/file decision made in step 2>
Compact listing now <n> tokens; shape=sequence <n>.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 6: Ledger** — row `5 drift checks`: `partly done (task 8) - shape and summary invariants; description drift in task 9`. Add any rule that had to change to row 1's notes. Commit with the co-author trailer.

---

### Task 9: unique ids, cost shape, description drift `[model: sonnet]`

**Files:**
- Modify: `tests/test_catalog_structure.py`
- Consumes: `get_example_files`, `BUILTIN_DIR` from `tests/test_examples.py`.

- [ ] **Step 1: Write the tests** (append to `tests/test_catalog_structure.py`)

```python
import re

from tests.test_examples import BUILTIN_DIR

BUILTINS = sorted(
    os.path.relpath(os.path.join(BUILTIN_DIR, name), REPO_ROOT)
    for name in os.listdir(BUILTIN_DIR)
    if name.endswith(".json")
)


def test_workflow_ids_are_unique_across_the_catalog():
    """A duplicate id is a step-cache collision waiting to happen, and it
    makes job history ambiguous about which workflow ran."""
    seen = {}
    for path in TEMPLATES + MODEL_CONFIGS + BUILTINS:
        identity = load(path).get("id")
        assert identity, f"{path} has no id"
        assert identity not in seen, f"{path} and {seen[identity]} share id {identity!r}"
        seen[identity] = path


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_declared_cost_is_well_formed(path):
    cost = load(path).get("cost")
    if cost is None:
        return
    assert isinstance(cost, list) and cost, f"{path}: cost must be a non-empty list or absent"
    for entry in cost:
        assert entry["device"] in ("cuda", "mps", "cpu"), path
        assert isinstance(entry["vram_gb"], (int, float)) and entry["vram_gb"] >= 0, path
        assert isinstance(entry["minutes"], (int, float)) and entry["minutes"] >= 0, path


BACKTICKED = re.compile(r"`([a-z_][a-z0-9_]*)`")


def _variable_names_in_catalog():
    names = set()
    for path in TEMPLATES + MODEL_CONFIGS:
        names |= set((load(path).get("variables") or {}).keys())
    return names


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_description_names_only_variables_the_workflow_declares(path):
    """A description that says `num_frames` for a workflow with no such
    variable sends an agent to pass an argument nothing reads. Restricted
    to identifiers that are variable names somewhere in the catalog, so a
    backticked task or type name is not a false positive."""
    definition = load(path)
    declared = set((definition.get("variables") or {}).keys())
    catalog_variables = _variable_names_in_catalog()
    mentioned = set(BACKTICKED.findall(definition.get("description", "")))
    undeclared = (mentioned & catalog_variables) - declared
    assert not undeclared, f"{path} describes {sorted(undeclared)} but declares no such variable"
```

- [ ] **Step 2: Run and resolve**

Run: `python -m pytest tests/test_catalog_structure.py -q -k "unique or cost or names_only"`
Expected: unique-id and cost pass immediately (no duplicates, no costs yet). Description-drift failures are real drift: fix each description (change the name to the one the workflow declares, or drop the backticks if it was a task name) — do not add variables to satisfy the test.

- [ ] **Step 3: Author the first `cost` entries** — for `workflows/templates/text-to-image.json` and `workflows/models/flux-dev.json` only, if you have measured runs to hand; otherwise skip and say so in the commit. Never invent a number.

- [ ] **Step 4: Run the catalog suite**

Run: `python -m pytest tests/test_catalog_structure.py tests/test_examples.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_catalog_structure.py workflows/
git commit -m "Catalog invariants: unique ids, well-formed cost, descriptions name real variables

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 6: Ledger** — row `3 \`cost\``: `done (tasks 7, 9)`; row `5 drift checks`: `done (tasks 8, 9)`. Commit with the co-author trailer.

---

### Task 10: token-budget test `[model: haiku]`

**Files:**
- Modify: `tests/test_catalog_structure.py`
- Consumes: `WorkflowSource`, `listing` from `dw/workflow_sources.py`; `workflow_details` from `dw/server/app.py`; `project_listing`.

- [ ] **Step 1: Write the test** (append to `tests/test_catalog_structure.py`)

```python
from dw.server.app import workflow_details
from dw.server.catalog_shape import project_listing
from dw.workflow_sources import WorkflowSource, listing

# Spec targets, as chars / 4. The listing is the first thing an agent reads;
# these are the ceilings that keep it readable rather than skimmed.
COMPACT_BUDGET = 5_500
FILTERED_BUDGET = 1_500


def _tokens(payload):
    return len(json.dumps(payload)) / 4


def test_the_compact_listing_fits_the_budget():
    found = listing([WorkflowSource(os.path.join(REPO_ROOT, "workflows"), "workspace", True)])
    details = workflow_details(found)

    compact = project_listing(details, view="compact")
    assert _tokens(compact) <= COMPACT_BUDGET, f"compact listing is {_tokens(compact):.0f} tokens"

    sequences = project_listing(details, view="compact", shape="sequence")
    assert sequences, "no template derives 'sequence'"
    assert _tokens(sequences) <= FILTERED_BUDGET, f"shape=sequence is {_tokens(sequences):.0f} tokens"
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_catalog_structure.py -q -k budget`
Expected: PASS. If it fails, report the numbers — do not raise the budgets; the fix is shorter summaries (Task 8's territory) or fewer fields in `COMPACT_FIELDS`, and either is a decision for the orchestrator.

- [ ] **Step 3: Full suite**

Run: `python -m pytest tests -q -x --ignore=tests/test_integration.py`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_catalog_structure.py
git commit -m "Hold the compact listing to its token budget

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 5: Ledger** — row `2 \`summary\``: append `budget test (task 10)`. Update the proposal's *What it costs today* table with a new row `list_workflows (compact)` carrying the measured number. Commit with the co-author trailer.

---

## Self-review

**Spec coverage.** §1.1 data model → Tasks 1, 3. §1.2 derivation → Task 1 (rules), Task 8 (real templates). §1.3 schema → Task 2. §1.4 route → Task 4. §1.5 MCP → Task 5. §1.6 save → Task 6. §1.7 jobs → Task 7. §1.8 tests → Tasks 1, 4, 7, 8, 9, 10. The `configures_missing` fallback (a config whose template is missing derives from its own definition) is what Task 3's loop does by leaving `shape`/`traits` alone — covered.

**Interfaces.** `derive_catalog_metadata` returns `declared` as a set; Task 8 iterates it — consistent. `project_listing` raises `ValueError`; Task 4's route maps to 400 — consistent. `Job.catalog_name` vs wire `workflow_name` is called out in Task 7 so nobody renames the existing attribute.

**Known judgement points, flagged for the orchestrator.** Task 8's `EXPECTED_SHAPES` are predictions; the task's step 2 is where they meet the files. Task 5 step 5 and Task 10 measure the budget; if the sweep does not bring compact under 5.5k, that comes back as a decision rather than a raised number.
