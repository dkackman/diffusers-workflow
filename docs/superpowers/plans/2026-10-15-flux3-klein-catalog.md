# FLUX 3 Image Klein Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Onboard FLUX 3 Image Klein (distilled) into the dw catalog: upsampler support in `text_generation`, two upsampler builtins, five templates, a model entry, a family README, GPU-measured costs, and the `flux3-klein` composition skill.

**Architecture:** Model knowledge stays in JSON, prompts and the skill. The engine gains three model-agnostic `text_generation` options (`chat_template_kwargs`, `revision`, a list for `image`) and a `system_prompt` that can name a JSON file in a Hub repo at a pinned commit. Two builtins (`flux3_klein_upsample`, `flux3_klein_upsample_edit`) carry BFL's upsampler settings once, and every template that upsamples composes one of them.

**Tech Stack:** Python 3.12+, transformers 5.19 pipelines, huggingface_hub `hf_hub_download`, diffusers `ModularPipeline` at or after 7ecb1e1b (PR #15046), pytest.

**Spec:** [docs/superpowers/specs/2026-10-15-flux-3-klein-catalog-design.md](../specs/2026-10-15-flux-3-klein-catalog-design.md), audit [docs/proposals/audits/2026-10-15-flux-3-klein-audit.md](../../proposals/audits/2026-10-15-flux-3-klein-audit.md).

## Global Constraints

- Branch `feat/flux3-klein`. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No per-model Python. Nothing in `dw/` names Flux3, Klein or BFL.
- diffusers must be at or after `7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941` (the Flux3 merge). Install it locally in Task 0; lem already has it.
- Klein repo `black-forest-labs/flux-3-image-klein`; upsampler repo `black-forest-labs/flux-3-image-klein-pup` pinned at `3f3edb52b2c55de0d49feab28598b68f7f8e4830` for both its weights and its `system_prompts.json`.
- Distilled settings: `num_inference_steps` 16, no guidance argument, no `negative_prompt`, `width`/`height` multiples of 16, default 1024x1024, at most 10 references.
- Every Klein pipeline step places every component explicitly: `"components": {"transformer": {"device": "cuda"}, "text_encoder": {"device": "cuda"}, "vae": {"device": "cuda"}}`. An unplaced component leaves `_execution_device` on the CPU (spec, P1a/P1b). Never put `residency: on_demand` on the text encoder.
- Upsampler decoding: greedy (the task's default), `max_new_tokens` 3584, `chat_template_kwargs: {"enable_thinking": false}`, `repetition_penalty: 1.0` on the image path.
- No vendor prompt text copied into the repo. The system prompts are read with `from_hub`. The edit scaffold (three short lines from BFL's `src/flux3/pup.py` at `bbfe317dd81c825c30d76090a67d3d0db51da3cc`) is the one quoted vendor string, and the builtin's description cites that file and commit.
- Descriptions name rules without single quotes (the drift check reads a single-quoted name as a variable or step).
- Security: validate the Hub repo id with `dw.locations.validate_hub_repo_id`; the `file` is a relative POSIX path with no `..`, no absolute part and no backslash; the `revision` is exactly 40 lowercase hex characters. No `eval`, `exec` or `shell=True`.
- Unit tests: `venv/bin/python -m pytest tests/<file> -v` locally (the venv has torch). The full suite runs with `-n auto` from `pytest.ini`.
- `tests/test_catalog_structure.py::test_the_compact_listing_fits_the_budget` may fail once the templates land. Raising that budget needs Don's OK: stop and ask, don't edit the number.

## Review Focus

1. **`from_hub` naming a key that isn't in the file, or a value that isn't text.** Expect a `ValueError` that lists the keys the file has, raised before any model loads, not a `KeyError` traceback. Test in Task 2.
2. **`image` given as `[]` or `[null]`.** That's an optional reference list left empty by a template. Expect the text-only path, exactly as with no image. Test in Task 1.
3. **A short `revision` (`3f3edb52`) or a branch name (`main`) in `from_hub`.** Expect a refusal naming the 40-hex rule. Without one, the prompt text could move under a pinned template. Test in Task 2.
4. **The Klein step with a component left unplaced**, as a cold agent copying the template might write it. Expect a catalog test to refuse it: `test_every_flux3_pipeline_step_places_every_component` in Task 4.
5. **More than 10 references.** Diffusers doesn't refuse it, and the model was trained on 10. Expect the multi-reference description and the skill to state the cap; an enforcing check is a dated follow-up. Test that the text says it in Task 6.

---

### Task 0: Local diffusers with Flux3

**Files:** none (environment)

- [ ] **Step 1: Install diffusers at the Flux3 merge or later**

Run: `venv/bin/pip install -q --force-reinstall --no-deps git+https://github.com/huggingface/diffusers@7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941`

- [ ] **Step 2: Confirm the modular package imports**

Run: `venv/bin/python -c "from diffusers.modular_pipelines.flux3 import modular_blocks_flux3_distilled as m; print(m.__file__)"`
Expected: a path ending in `modular_pipelines/flux3/modular_blocks_flux3_distilled.py`.

---

### Task 1: `text_generation` takes chat-template settings, a pinned revision and a list of images

**Files:**
- Modify: `dw/tasks/text_generation.py` (`_image_part`, `_build_messages`, `generate_text`, `_generate`)
- Test: `tests/test_text_generation.py`

**Interfaces:**
- Produces: `generate_text(prompt, device="cpu", **kwargs)` now also reads `chat_template_kwargs: dict | None`, `revision: str | None`, and `image: PIL | str | list[PIL | str] | None`. The return value doesn't change.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_text_generation.py`, inside `TestTextGeneration`)

```python
    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_chat_template_kwargs_reach_the_text_template(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        generate_text(
            "a cat", device="cpu", model_name="m/text-chat-kwargs",
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertEqual(
            pipe.call_args[1]["tokenizer_encode_kwargs"], {"enable_thinking": False}
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_chat_template_kwargs_reach_the_vision_template(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        generate_text(
            "a cat", device="cpu", model_name="m/vision-chat-kwargs",
            image="https://example.com/a.png",
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertIs(pipe.call_args[1]["enable_thinking"], False)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_no_chat_template_kwargs_passes_nothing_extra(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        generate_text("a cat", device="cpu", model_name="m/no-chat-kwargs")

        self.assertNotIn("tokenizer_encode_kwargs", pipe.call_args[1])

    def test_chat_template_kwargs_must_be_a_mapping(self):
        with self.assertRaises(ValueError):
            generate_text("a cat", device="cpu", chat_template_kwargs=["enable_thinking"])

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_revision_reaches_the_loader(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        generate_text("a cat", device="cpu", model_name="m/revision", revision="a" * 40)

        self.assertEqual(mock_pipeline.call_args[1]["revision"], "a" * 40)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_a_list_of_images_becomes_one_part_each_in_order(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        generate_text(
            "edit", device="cpu", model_name="m/image-list",
            image=["https://example.com/1.png", "https://example.com/2.png"],
        )

        user = pipe.call_args[1]["text"][-1]["content"]
        self.assertEqual(
            [part.get("url") for part in user[:2]],
            ["https://example.com/1.png", "https://example.com/2.png"],
        )
        self.assertEqual(user[2], {"type": "text", "text": "edit"})
        self.assertEqual(mock_pipeline.call_args[0][0], "image-text-to-text")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_an_empty_or_all_null_image_list_is_the_text_path(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        for empty in ([], [None]):
            generate_text("a cat", device="cpu", model_name=f"m/empty-{len(empty)}", image=empty)
            self.assertEqual(mock_pipeline.call_args[0][0], "text-generation")
```

Each test uses its own `model_name` because `cached_model` keys on it and a shared name would return an earlier test's mock.

- [ ] **Step 2: Run them and watch them fail**

Run: `venv/bin/python -m pytest tests/test_text_generation.py -v -n0`
Expected: the seven new tests FAIL (`KeyError: 'tokenizer_encode_kwargs'`, `'enable_thinking'`, `'revision'`, no `ValueError` raised, and the list tests on the message shape or pipeline task).

- [ ] **Step 3: Implement**

In `dw/tasks/text_generation.py`, replace `_build_messages` with the version below (keep `_image_part` as it is):

```python
def _images(image):
    """The images a call carries, as a list in order, or None for none.

    A template's optional reference list arrives as [] or [None] when the
    caller supplied nothing, and that is the same statement as no image.
    """
    if image is None:
        return None
    images = list(image) if isinstance(image, (list, tuple)) else [image]
    images = [i for i in images if i is not None and i != ""]
    return images or None


def _build_messages(prompt, system_prompt, images):
    """Chat messages in the shape the chosen pipeline expects.

    Text-generation models take plain string content. Vision models take a
    list of typed parts, because the images are part of the message rather
    than something alongside it - every image first, in order, then the text.
    """
    messages = []
    if images is None:
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
    else:
        if system_prompt is not None:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [_image_part(i) for i in images]
                + [{"type": "text", "text": prompt}],
            }
        )
    return messages
```

In `generate_text`, document the three new kwargs in the docstring:

```
            image: Optional PIL image, URL or path, or a list of them - every
                one goes to the model, in order. Its presence is what selects
                the vision pipeline; an empty list is no image.
            chat_template_kwargs: Settings for the model's chat template,
                such as {"enable_thinking": false} for a Qwen3 model that
                must not reason before answering.
            revision: The model repo's commit, branch or tag to load.
```

Then change the body:

```python
    images = _images(kwargs.get("image", None) or None)
    system_prompt = kwargs.get("system_prompt", None)
    max_new_tokens = int(kwargs.get("max_new_tokens", 500))
    revision = kwargs.get("revision", None)
    chat_template_kwargs = kwargs.get("chat_template_kwargs") or {}
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError(
            "text_generation: chat_template_kwargs must be an object of settings "
            f"for the chat template, got {type(chat_template_kwargs).__name__}"
        )

    if images is None:
        pipeline_task = "text-generation"
        model_name = kwargs.get("model_name", _DEFAULT_MODEL)
    else:
        pipeline_task = "image-text-to-text"
        model_name = kwargs.get("model_name", DEFAULT_VISION_MODEL)
```

`load_pipe` passes the revision only when one is set:

```python
    def load_pipe():
        logger.info(f"Generating text with {model_name} on {device}")
        placement = hf_pipeline_placement(device)
        if revision is not None:
            placement = {**placement, "revision": revision}
        return hf_pipeline(
            pipeline_task,
            model=model_name,
            torch_dtype=dtype,
            **placement,
        )
```

The cache key gains the revision, so two pins of one repo don't share a model:

```python
    pipe = cached_model(
        ("text_generation", pipeline_task, model_name, str(revision), str(device), str(dtype)),
        load_pipe,
```

Pass `images` where the body used `image`: `_build_messages(prompt, system_prompt, images)`, `if images is not None:` around the repetition penalty, and `_generate(pipe, messages, images, max_new_tokens, generation, chat_template_kwargs)`.

`_generate` hands the template settings to each path the way that pipeline accepts them:

```python
def _generate(pipe, messages, images, max_new_tokens, generation, chat_template_kwargs):
    """Run the pipeline, which each path calls differently."""
    if images is None:
        # This pipeline collects anything it does not name into the arguments it
        # forwards to generate(), so settings go in as plain keywords. Chat
        # template settings have their own argument, which it hands to
        # apply_chat_template
        extra = {"tokenizer_encode_kwargs": chat_template_kwargs} if chat_template_kwargs else {}
        results = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            **generation,
            **extra,
        )
    else:
        # (keep the existing comment) ... Keywords this pipeline does not name
        # go to the processor's apply_chat_template, which is where the chat
        # template settings belong
        results = pipe(
            text=messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            generate_kwargs=generation,
            **chat_template_kwargs,
        )
```

- [ ] **Step 4: Run the file and the callers' tests**

Run: `venv/bin/python -m pytest tests/test_text_generation.py tests/test_image_to_text.py tests/test_judge.py -v -n0` (skip any of those names that don't exist; `ls tests | grep -i "image_to_text\|judge"`)
Expected: all PASS, including the existing tests, which call `_build_messages`'s callers with a single image.

- [ ] **Step 5: Commit**

```bash
git add dw/tasks/text_generation.py tests/test_text_generation.py
git commit -m "feat(text_generation): chat_template_kwargs, revision and a list of images

A Qwen3.5 upsampler must run with enable_thinking off, an edit sends every
reference, and a pinned prompt wants pinned weights.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `system_prompt` can name a JSON file in a Hub repo at a pinned commit

**Files:**
- Modify: `dw/tasks/text_generation.py` (new `resolve_system_prompt`, called at the top of `generate_text`)
- Modify: `docs/TASKS.md` (the `text_generation` section)
- Test: `tests/test_text_generation.py` (new class `TestSystemPromptFromHub`)

**Interfaces:**
- Consumes: `dw.locations.validate_hub_repo_id(name, what=...)`.
- Produces: `resolve_system_prompt(value) -> str | None`. `value` is `None`, a `str`, or `{"from_hub": {"repo": str, "file": str, "revision": str, "key": str (optional)}}`. Tasks 3-4 write the dict form in JSON.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_text_generation.py`)

```python
import json as _json

from dw.tasks.text_generation import resolve_system_prompt

PIN = "3f3edb52b2c55de0d49feab28598b68f7f8e4830"


def _spec(**over):
    spec = {"repo": "black-forest-labs/flux-3-image-klein-pup",
            "file": "system_prompts.json", "revision": PIN, "key": "t2i"}
    spec.update(over)
    return {"from_hub": spec}


class TestSystemPromptFromHub(unittest.TestCase):
    def setUp(self):
        import tempfile, os
        self.dir = tempfile.mkdtemp()
        self.json_path = os.path.join(self.dir, "system_prompts.json")
        with open(self.json_path, "w") as f:
            _json.dump({"t2i": "expand the request", "i2i": "edit it", "n": 3}, f)

    def test_a_string_or_none_passes_through(self):
        self.assertEqual(resolve_system_prompt("plain"), "plain")
        self.assertIsNone(resolve_system_prompt(None))

    @patch("dw.tasks.text_generation.hf_hub_download")
    def test_the_key_s_text_is_returned_from_the_pinned_file(self, download):
        download.return_value = self.json_path
        self.assertEqual(resolve_system_prompt(_spec()), "expand the request")
        download.assert_called_once_with(
            "black-forest-labs/flux-3-image-klein-pup", "system_prompts.json", revision=PIN
        )

    @patch("dw.tasks.text_generation.hf_hub_download")
    def test_a_missing_key_names_the_keys_the_file_has(self, download):
        download.return_value = self.json_path
        with self.assertRaises(ValueError) as caught:
            resolve_system_prompt(_spec(key="t2v"))
        self.assertIn("i2i", str(caught.exception))
        self.assertIn("t2i", str(caught.exception))

    @patch("dw.tasks.text_generation.hf_hub_download")
    def test_a_value_that_is_not_text_is_refused(self, download):
        download.return_value = self.json_path
        with self.assertRaises(ValueError):
            resolve_system_prompt(_spec(key="n"))

    def test_a_short_hash_or_branch_is_refused(self):
        for revision in ("3f3edb52", "main", PIN.upper()):
            with self.assertRaises(ValueError) as caught:
                resolve_system_prompt(_spec(revision=revision))
            self.assertIn("40", str(caught.exception))

    def test_a_path_that_escapes_the_repo_is_refused(self):
        for file in ("../secrets.json", "/etc/passwd", "a\\b.json", "", "x/../../y.json"):
            with self.assertRaises(ValueError):
                resolve_system_prompt(_spec(file=file))

    def test_a_bad_repo_id_is_refused(self):
        from dw.security import InvalidInputError
        with self.assertRaises((ValueError, InvalidInputError)):
            resolve_system_prompt(_spec(repo="https://evil.example/x"))

    def test_an_unknown_field_or_shape_is_refused(self):
        for bad in ({"from_hub": {**_spec()["from_hub"], "branch": "main"}},
                    {"from_url": "x"}, ["t2i"]):
            with self.assertRaises(ValueError):
                resolve_system_prompt(bad)

    @patch("dw.tasks.text_generation.hf_pipeline")
    @patch("dw.tasks.text_generation.hf_hub_download")
    def test_generate_text_sends_the_resolved_text(self, download, mock_pipeline):
        download.return_value = self.json_path
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        generate_text("a dog", device="cpu", model_name="m/from-hub", system_prompt=_spec())

        messages = pipe.call_args[0][0]
        self.assertEqual(messages[0], {"role": "system", "content": "expand the request"})
```

- [ ] **Step 2: Run them and watch them fail**

Run: `venv/bin/python -m pytest tests/test_text_generation.py -v -n0 -k FromHub`
Expected: ImportError on `resolve_system_prompt`.

- [ ] **Step 3: Implement**

At the top of `dw/tasks/text_generation.py`, add module-level imports (a module-level `hf_hub_download` is what the tests patch):

```python
import json
import re
from pathlib import PurePosixPath

from huggingface_hub import hf_hub_download

from ..locations import validate_hub_repo_id
```

If importing `..locations` here creates an import cycle (run the tests to see), import `validate_hub_repo_id` inside `resolve_system_prompt` instead.

Add, above `generate_text`:

```python
# A pinned prompt has to stay pinned: a short hash can become ambiguous and a
# branch moves, so only a full commit is accepted
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_FROM_HUB_FIELDS = {"repo", "file", "revision", "key"}


def _hub_file(file):
    """A file path inside a repo, refused if it could name anything outside it."""
    if not isinstance(file, str) or not file or "\\" in file:
        raise ValueError(f"system_prompt.from_hub.file must be a path inside the repo, got {file!r}")
    path = PurePosixPath(file)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"system_prompt.from_hub.file must be a path inside the repo, got {file!r}")
    return file


def resolve_system_prompt(value):
    """The system prompt text, reading it from a Hub repo when one is named.

    Some vendors ship the instructions their model was trained against as a
    file in its repo, and change them. Naming the file and a full commit runs
    exactly that text without copying it into a workflow, where it would
    drift from the source.

    Args:
        value: None, the text itself, or {"from_hub": {"repo", "file",
            "revision", "key"}}. `revision` is a full 40-character commit.
            A .json file takes `key` to pick one string out of a top-level
            object; any other file is read whole as text

    Raises:
        ValueError: On any other shape, a revision that is not a full commit,
            a file path that leaves the repo, a missing key, or a value that
            is not text
    """
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, dict) or set(value) != {"from_hub"} or not isinstance(value["from_hub"], dict):
        raise ValueError(
            'system_prompt must be text or {"from_hub": {"repo", "file", "revision", "key"}}'
        )
    spec = value["from_hub"]
    unknown = set(spec) - _FROM_HUB_FIELDS
    if unknown:
        raise ValueError(f"system_prompt.from_hub does not take {sorted(unknown)}")
    repo = validate_hub_repo_id(spec.get("repo"), what="system_prompt.from_hub.repo")
    file = _hub_file(spec.get("file"))
    revision = spec.get("revision")
    if not isinstance(revision, str) or not _FULL_COMMIT.match(revision):
        raise ValueError(
            "system_prompt.from_hub.revision must be a full 40-character lowercase "
            f"commit hash, so the text cannot change under the workflow; got {revision!r}"
        )
    key = spec.get("key")

    path = hf_hub_download(repo, file, revision=revision)
    with open(path, encoding="utf-8") as f:
        if file.endswith(".json"):
            data = json.load(f)
        else:
            data = f.read()
    if key is not None:
        if not isinstance(data, dict) or key not in data:
            keys = sorted(data) if isinstance(data, dict) else []
            raise ValueError(
                f"system_prompt.from_hub: {file} at {revision[:8]} has no key {key!r}; "
                f"it has {keys}"
            )
        data = data[key]
    if not isinstance(data, str) or not data.strip():
        raise ValueError(
            f"system_prompt.from_hub: {file}{'[' + repr(key) + ']' if key else ''} "
            "is not text"
        )
    return data
```

In `generate_text`, replace `system_prompt = kwargs.get("system_prompt", None)` with:

```python
    # Resolved before anything loads, so a bad reference costs nothing
    system_prompt = resolve_system_prompt(kwargs.get("system_prompt", None))
```

In `docs/TASKS.md`'s `text_generation` section, add the three Task 1 arguments and this paragraph with an example:

```markdown
`system_prompt` may name a file in a Hugging Face repo instead of carrying the text,
for a vendor that ships its prompt instructions there:
`{"from_hub": {"repo": "owner/name", "file": "system_prompts.json", "revision": "<40-hex commit>", "key": "t2i"}}`.
The revision must be a full commit, so the text can't move under the workflow. A `.json`
file takes `key` to pick one string; any other file is read whole.
```

- [ ] **Step 4: Run the tests**

Run: `venv/bin/python -m pytest tests/test_text_generation.py -v -n0`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/tasks/text_generation.py tests/test_text_generation.py docs/TASKS.md
git commit -m "feat(text_generation): system_prompt from a Hub repo file at a pinned commit

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The upsampler builtins

**Files:**
- Create: `dw/workflows/flux3_klein_upsample.json`
- Create: `dw/workflows/flux3_klein_upsample_edit.json`
- Test: `tests/test_flux3_catalog.py` (new)

**Interfaces:**
- Consumes: `text_generation` arguments from Tasks 1-2; the `compose_text` task (`parts`, `separator`).
- Produces: `builtin:flux3_klein_upsample.json` (variables `prompt`, `device`) and `builtin:flux3_klein_upsample_edit.json` (variables `instruction`, `images`, `device`). Each returns the caption text from its last step.

- [ ] **Step 1: Write the failing test** (`tests/test_flux3_catalog.py`)

```python
"""The FLUX 3 Image Klein catalog: the upsampler builtins, the templates' settings
and their placement. The vendor values here are the ones the design states
(docs/superpowers/specs/2026-10-15-flux-3-klein-catalog-design.md)."""

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILTINS = os.path.join(ROOT, "dw", "workflows")
PUP = "black-forest-labs/flux-3-image-klein-pup"
PUP_PIN = "3f3edb52b2c55de0d49feab28598b68f7f8e4830"


def load(path):
    with open(path) as f:
        return json.load(f)


def generation_steps(definition):
    return [s for s in definition["steps"] if s.get("task", {}).get("command") == "text_generation"]


@pytest.mark.parametrize("name,key", [
    ("flux3_klein_upsample.json", "t2i"),
    ("flux3_klein_upsample_edit.json", "i2i"),
])
def test_each_upsampler_runs_the_pinned_pup_as_bfl_does(name, key):
    definition = load(os.path.join(BUILTINS, name))
    (step,) = generation_steps(definition)
    args = step["task"]["arguments"]
    assert args["model_name"] == PUP
    assert args["revision"] == PUP_PIN
    assert args["system_prompt"] == {"from_hub": {
        "repo": PUP, "file": "system_prompts.json", "revision": PUP_PIN, "key": key}}
    assert args["chat_template_kwargs"] == {"enable_thinking": False}
    assert args["max_new_tokens"] == 3584


def test_the_edit_upsampler_wraps_the_instruction_in_bfl_s_scaffold():
    definition = load(os.path.join(BUILTINS, "flux3_klein_upsample_edit.json"))
    compose = next(s for s in definition["steps"] if s["task"]["command"] == "compose_text")
    parts = compose["task"]["arguments"]["parts"]
    assert parts == [
        "Reference images follow in order (the first image, the second image, ...).\nUser request: ",
        "variable:instruction",
        "\nOutput the caption.",
    ]
    assert compose["task"]["arguments"]["separator"] == ""
    (step,) = generation_steps(definition)
    assert step["task"]["arguments"]["image"] == "variable:images"
    assert step["task"]["arguments"]["repetition_penalty"] == 1.0
    assert "bbfe317d" in definition["description"]
```

- [ ] **Step 2: Run it and watch it fail**

Run: `venv/bin/python -m pytest tests/test_flux3_catalog.py -v -n0`
Expected: FAIL, `FileNotFoundError`.

- [ ] **Step 3: Write the two builtins**

`dw/workflows/flux3_klein_upsample.json`:

```json
{
    "id": "flux3_klein_upsample",
    "description": "Expands a short request into the caption FLUX 3 Image Klein was trained on, with Black Forest Labs' own prompt upsampler, flux-3-image-klein-pup, a Qwen3.5-9B fine-tune. BFL says not to use Klein without it. The weights and the text-to-image system prompt are both pinned to pup commit 3f3edb52, and the prompt is read from the repo's system_prompts.json rather than copied. Thinking is switched off and decoding is greedy, as BFL runs it. BFL also applies a presence penalty of 1.5 for text-to-image, which transformers' generate() has no equivalent for, so it is left out. About 18 GB in bf16.",
    "variables": {
        "prompt": "A man walking with his dog, afternoon sun",
        "device": "cuda"
    },
    "steps": [
        {
            "name": "upsample",
            "task": {
                "command": "text_generation",
                "arguments": {
                    "prompt": "variable:prompt",
                    "model_name": "black-forest-labs/flux-3-image-klein-pup",
                    "revision": "3f3edb52b2c55de0d49feab28598b68f7f8e4830",
                    "system_prompt": {"from_hub": {
                        "repo": "black-forest-labs/flux-3-image-klein-pup",
                        "file": "system_prompts.json",
                        "revision": "3f3edb52b2c55de0d49feab28598b68f7f8e4830",
                        "key": "t2i"
                    }},
                    "chat_template_kwargs": {"enable_thinking": false},
                    "max_new_tokens": 3584,
                    "device": "variable:device"
                }
            },
            "result": {"content_type": "text/plain", "save": false}
        }
    ]
}
```

`dw/workflows/flux3_klein_upsample_edit.json` has the same shape, with these differences:
- `id` `flux3_klein_upsample_edit`.
- `variables`: `{"instruction": "put a red bandana around the dog's neck", "images": [], "device": "cuda"}`.
- A first step `scaffold` with `"command": "compose_text"`, `"arguments": {"parts": [<the three parts from the test>], "separator": ""}`, and `"result": {"content_type": "text/plain", "save": false}`.
- The `upsample` step uses `"prompt": "previous_result:scaffold"`, `"image": "variable:images"`, `"repetition_penalty": 1.0` and `"key": "i2i"`.
- The description says the upsampler sees every reference in order, as the pipeline will, and that the scaffold's three lines are BFL's `I2I_SCAFFOLD` from `src/flux3/pup.py` at commit bbfe317d in github.com/black-forest-labs/flux-3-image-klein. It also says BFL resizes each reference to a pixel budget (1 MP for one reference, 724x724 for two, 512x512 for three or more) before the upsampler sees it, while transformers' processor applies its own sizing, so the two can describe small details differently.

- [ ] **Step 4: Run the test and the library tests**

Run: `venv/bin/python -m pytest tests/test_flux3_catalog.py tests/test_library_path.py tests/test_library_sources.py -v -n0`
Expected: PASS.

Then validate each builtin against the schema: `venv/bin/python -m dw.validate dw/workflows/flux3_klein_upsample.json` and the same for `flux3_klein_upsample_edit.json`. Expected: valid.

- [ ] **Step 5: Commit**

```bash
git add dw/workflows/flux3_klein_upsample.json dw/workflows/flux3_klein_upsample_edit.json tests/test_flux3_catalog.py
git commit -m "feat(flux3): Klein prompt-upsampler builtins on the pinned pup

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Text-to-image template, model entry and stored prompt

**Files:**
- Create: `prompts/flux3/man_dog.json`
- Create: `workflows/templates/flux3/text-to-image.json`
- Create: `workflows/models/flux3-klein.json`
- Test: `tests/test_flux3_catalog.py`

**Interfaces:**
- Produces: the Klein pipeline step shape every later template copies:

```json
{
    "name": "text_to_image",
    "pipeline": {
        "configuration": {
            "component_type": "ModularPipeline",
            "load_components": {"dtype": "torch.bfloat16"},
            "components": {
                "transformer": {"device": "cuda"},
                "text_encoder": {"device": "cuda"},
                "vae": {"device": "cuda"}
            }
        },
        "from_pretrained_arguments": {"model_name": "black-forest-labs/flux-3-image-klein"},
        "arguments": {
            "prompt": "variable:prompt",
            "width": "variable:width",
            "height": "variable:height",
            "num_inference_steps": 16,
            "output": "images"
        }
    },
    "result": {"content_type": "image/jpeg", "embed_metadata": true, "subfolder": "final"}
}
```

- [ ] **Step 1: Write the failing tests** (append to `tests/test_flux3_catalog.py`)

```python
TEMPLATES = os.path.join(ROOT, "workflows", "templates", "flux3")
KLEIN = "black-forest-labs/flux-3-image-klein"


def klein_steps(definition):
    return [s for s in definition["steps"]
            if s.get("pipeline", {}).get("from_pretrained_arguments", {}).get("model_name") == KLEIN]


def flux3_definitions():
    paths = [os.path.join(TEMPLATES, n) for n in sorted(os.listdir(TEMPLATES)) if n.endswith(".json")]
    paths.append(os.path.join(ROOT, "workflows", "models", "flux3-klein.json"))
    return paths


@pytest.mark.parametrize("path", flux3_definitions())
def test_every_flux3_pipeline_step_places_every_component(path):
    steps = klein_steps(load(path))
    assert steps, f"{path} runs no Klein step"
    for step in steps:
        components = step["pipeline"]["configuration"]["components"]
        for name in ("transformer", "text_encoder", "vae"):
            assert components.get(name, {}).get("device") == "cuda", (path, name)
        assert components["text_encoder"].get("residency") is None


@pytest.mark.parametrize("path", flux3_definitions())
def test_every_klein_step_runs_the_distilled_schedule_without_guidance(path):
    for step in klein_steps(load(path)):
        args = step["pipeline"]["arguments"]
        assert args["num_inference_steps"] == 16
        assert "guidance_scale" not in args and "negative_prompt" not in args
        assert args["output"] == "images"


@pytest.mark.parametrize("path", flux3_definitions())
def test_size_variables_are_held_to_the_16_pixel_grid(path):
    definition = load(path)
    constraints = definition.get("variable_constraints", {})
    for name in ("width", "height"):
        if name in definition.get("variables", {}):
            assert constraints[name]["modulus"] == 16 and constraints[name].get("remainder", 0) == 0


def test_the_16_pixel_grid_and_16_step_default_are_the_pipeline_s():
    import inspect
    from diffusers.modular_pipelines.flux3 import before_denoise, modular_blocks_flux3_distilled
    assert "num_inference_steps" in inspect.getsource(modular_blocks_flux3_distilled)
    assert "16" in inspect.getsource(modular_blocks_flux3_distilled)
    assert "vae_scale_factor * 2" in inspect.getsource(before_denoise)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `venv/bin/python -m pytest tests/test_flux3_catalog.py -v -n0`
Expected: collection error (the `flux3` templates folder doesn't exist).

- [ ] **Step 3: Write the files**

`prompts/flux3/man_dog.json`: `{"text": "<BFL's upsampled man_dog caption>"}`. Copy the `prompt` field of `examples/distilled/man_dog.json` in github.com/black-forest-labs/flux-3-image-klein at bbfe317d byte for byte (`gh api "repos/black-forest-labs/flux-3-image-klein/contents/examples/distilled/man_dog.json?ref=bbfe317dd81c825c30d76090a67d3d0db51da3cc" --jq .content | base64 -d`). It's the vendor's example caption, already the length and form the model was trained on. Match the other files' shape in `prompts/` (`cat prompts/kandinsky6/mountain_steam_train.json`).

`workflows/templates/flux3/text-to-image.json`:

```json
{
    "id": "Flux3KleinT2I",
    "description": "FLUX 3 Image Klein (distilled, 5.6B) text to image: the family's baseline. The prompt here is already a full caption - BFL's own upsampled caption for their man walking a dog example - because Klein was trained on long upsampler captions and turns short prompts into generic pictures; upsample-then-generate shows how to get one from a short request. Sixteen steps with no guidance: the distilled model is a flow map, so it takes no guidance scale and no negative prompt. Width and height must be multiples of 16, up to 4096. Every component is placed on the GPU explicitly, about 19 GB of bf16 weights, which fits a 24 GB card at 1024x1024 with under 2 GB to spare. The weights are gated under the FLUX Kommunity License: accept it on the model page and log in with hf auth login first.",
    "summary": "<one line under the summary cap; see test_every_template_has_a_summary_that_fits>",
    "cost_drivers": ["width", "height"],
    "variables": {
        "prompt": "prompt:flux3/man_dog",
        "width": 1024,
        "height": 1024,
        "seed": 7
    },
    "variable_constraints": {
        "width": {"modulus": 16, "remainder": 0, "reason": "FLUX 3 Image Klein's pipeline refuses a width not divisible by 16 (the VAE's 8x compression times the 2x2 latent packing), and only after the weights have loaded, so it is checked here first"},
        "height": {"modulus": 16, "remainder": 0, "reason": "FLUX 3 Image Klein's pipeline refuses a height not divisible by 16 (the VAE's 8x compression times the 2x2 latent packing), and only after the weights have loaded, so it is checked here first"}
    },
    "seed": "variable:seed",
    "steps": ["<the Klein step from Interfaces>"]
}
```

`workflows/models/flux3-klein.json`: the same Klein step with `"configures": "templates/text-to-image"`. Copy `variables` and `cost_drivers` from `workflows/templates/text-to-image.json` (`prompt` there is a short prompt, and the description says so: generic output without the upsampler). Add a `width`/`height` pair and their constraints only if that template declares them. Its description names the gated license and points at `templates/flux3/upsample-then-generate` for full captions.

Fill `summary` to pass `test_every_template_has_a_summary_that_fits`.

- [ ] **Step 4: Run the catalog tests**

Run: `venv/bin/python -m pytest tests/test_flux3_catalog.py tests/test_catalog_structure.py tests/test_catalog_shape.py tests/test_template_defaults.py tests/test_template_subfolders.py tests/test_prompt_library.py tests/test_prompt_references.py -v -n0`
Expected: all PASS. If only `test_the_compact_listing_fits_the_budget` fails, stop and report its numbers (Global Constraints).

Then: `venv/bin/python -m dw.validate workflows/templates/flux3/text-to-image.json` and the same for `workflows/models/flux3-klein.json`. Expected: valid.

- [ ] **Step 5: Commit**

```bash
git add prompts/flux3/man_dog.json workflows/templates/flux3/text-to-image.json workflows/models/flux3-klein.json tests/test_flux3_catalog.py
git commit -m "feat(flux3): Klein text-to-image template and model entry

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Upsample-then-generate template

**Files:**
- Create: `workflows/templates/flux3/upsample-then-generate.json`
- Test: `tests/test_flux3_catalog.py`

**Interfaces:**
- Consumes: `builtin:flux3_klein_upsample.json` (Task 3) and the Klein step (Task 4).

- [ ] **Step 1: Write the failing test**

```python
def test_upsample_then_generate_feeds_the_builtin_caption_to_klein():
    definition = load(os.path.join(TEMPLATES, "upsample-then-generate.json"))
    first, second = definition["steps"]
    assert first["workflow"]["path"] == "builtin:flux3_klein_upsample.json"
    assert first["release_models"] is True
    assert second["pipeline"]["arguments"]["prompt"] == f"previous_result:{first['name']}"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `venv/bin/python -m pytest tests/test_flux3_catalog.py -v -n0 -k upsample_then`
Expected: FAIL, file not found.

- [ ] **Step 3: Write the template**

Steps: `upsample` (`"workflow": {"path": "builtin:flux3_klein_upsample.json", "arguments": {"prompt": "variable:request"}}`, `"release_models": true`, result `text/plain`, `subfolder` `intermediate`), then the Klein step (Task 4 Interfaces) with `"prompt": "previous_result:upsample"`. Variables: `request` ("A man walking with his dog, afternoon sun"), `width`, `height`, `seed` 7, with the Task 4 constraints. The description says why `release_models` is there: the 18 GB upsampler and the 19 GB generator don't fit a 24 GB card together, so the upsampler is released before Klein loads. It also says the caption is saved as an intermediate so it can be reused.

- [ ] **Step 4: Run the catalog tests and validate**

Run: `venv/bin/python -m pytest tests/test_flux3_catalog.py tests/test_catalog_structure.py -v -n0 && venv/bin/python -m dw.validate workflows/templates/flux3/upsample-then-generate.json`
Expected: PASS and valid.

- [ ] **Step 5: Commit**

```bash
git add workflows/templates/flux3/upsample-then-generate.json tests/test_flux3_catalog.py
git commit -m "feat(flux3): upsample-then-generate template

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Probe P3, then the edit and multi-reference templates

**Files:**
- Create: `workflows/templates/flux3/image-edit.json`
- Create: `workflows/templates/flux3/multi-reference.json`
- Modify: `docs/superpowers/specs/2026-10-15-flux-3-klein-catalog-design.md` (P3 row)
- Test: `tests/test_flux3_catalog.py`

**Interfaces:**
- Consumes: `builtin:flux3_klein_upsample_edit.json` (Task 3) and the Klein step (Task 4).

- [ ] **Step 1: Probe how a list of images reaches the pipeline (lem, through the dw MCP)**

The Klein pipeline's `image` takes a PIL image or a list of them (`encoders.py:84` refuses anything else). Before writing the templates, find out whether a variable holding `[{"location": "<url>"}, {"location": "<url>"}]` arrives as a list of PIL images. Validate and run, on lem, the Task 4 step with `"image": "variable:images"` added, `images` set to two image URLs (any two public JPEGs), and a prompt naming `<ref_image_0>` and `<ref_image_1>`. Quote the plan's estimate and get Don's go-ahead first. Record the job id and the result in the spec's P3 row, along with peak memory from `get_job_events(kinds=["memory"])`. If the list doesn't load, stop and report: the fix is engine work outside this plan.

- [ ] **Step 2: Write the failing tests**

```python
@pytest.mark.parametrize("name", ["image-edit.json", "multi-reference.json"])
def test_the_edit_templates_show_the_upsampler_the_pipeline_s_references(name):
    definition = load(os.path.join(TEMPLATES, name))
    upsample = next(s for s in definition["steps"] if "workflow" in s)
    assert upsample["workflow"]["path"] == "builtin:flux3_klein_upsample_edit.json"
    (klein,) = klein_steps(definition)
    assert upsample["workflow"]["arguments"]["images"] == klein["pipeline"]["arguments"]["image"]
    assert klein["pipeline"]["arguments"]["prompt"] == f"previous_result:{upsample['name']}"


def test_multi_reference_states_the_ten_reference_cap_and_the_naming():
    description = load(os.path.join(TEMPLATES, "multi-reference.json"))["description"]
    assert "10 references" in description
    assert "<ref_image_0>" in description
```

- [ ] **Step 3: Run them and watch them fail**

Run: `venv/bin/python -m pytest tests/test_flux3_catalog.py -v -n0 -k "edit_templates or ten_reference"`
Expected: FAIL, files not found.

- [ ] **Step 4: Write the templates**

`image-edit.json`: variables `instruction` ("put a red bandana around the dog's neck", BFL's example), `image` (a `{"location": ...}` reference; use the same public dog photo P3 used), `seed` 22, no width/height (diffusers defaults the output to 1 MP at the reference's aspect ratio, rounded to 16, and the description says so). Steps: `upsample` → `builtin:flux3_klein_upsample_edit.json` with `"images": ["variable:image"]` and `"instruction": "variable:instruction"`, `release_models: true`; then the Klein step with `"image": ["variable:image"]` and `"prompt": "previous_result:upsample"`.

`multi-reference.json`: the same, with `images` a list variable of two references, `instruction` "print the dog from the second image onto the mug" (BFL's example), seed 23. Both the builtin and the Klein step read `"variable:images"`. The description states: references are named `<ref_image_0>`, `<ref_image_1>`, … (zero-based) in a caption written by hand, while the upsampler's own wording is "the first image, the second image"; the first reference sets the output's aspect ratio; and BFL trained Klein on up to 10 references, a cap the pipeline doesn't enforce, so more than 10 runs but is outside what the model learned.

Add `"cost_drivers"` naming the image variable(s).

- [ ] **Step 5: Run the catalog tests and validate both**

Run: `venv/bin/python -m pytest tests/test_flux3_catalog.py tests/test_catalog_structure.py -v -n0 && venv/bin/python -m dw.validate workflows/templates/flux3/image-edit.json && venv/bin/python -m dw.validate workflows/templates/flux3/multi-reference.json`
Expected: PASS and valid.

- [ ] **Step 6: Commit**

```bash
git add workflows/templates/flux3/image-edit.json workflows/templates/flux3/multi-reference.json tests/test_flux3_catalog.py docs/superpowers/specs/2026-10-15-flux-3-klein-catalog-design.md
git commit -m "feat(flux3): image-edit and multi-reference templates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Layout-with-boxes template

**Files:**
- Create: `workflows/templates/flux3/layout-boxes.json`
- Create: `prompts/flux3/balloons_layout.json`
- Test: `tests/test_flux3_catalog.py`

Scope note: BFL's box upsampling (`t2i_bbox`) goes through extra scaffolding in their `bbox.plan`/`bbox.reconcile` code. This template takes a caption written in the box format directly, as the Hugging Face card shows. Upsampled boxes are a dated follow-up.

- [ ] **Step 1: Write the failing test**

```python
def test_the_layout_caption_carries_tags_and_a_0_to_1000_box_table():
    import re
    with open(os.path.join(ROOT, "prompts", "flux3", "balloons_layout.json")) as f:
        text = json.load(f)["text"]
    table = json.loads(text[text.index("["):])
    assert table and all({"id", "bbox", "desc"} <= set(row) for row in table)
    for row in table:
        assert f"<{row['id']}>" in text
        top, left, bottom, right = row["bbox"]
        assert 0 <= top < bottom <= 1000 and 0 <= left < right <= 1000
    assert klein_steps(load(os.path.join(TEMPLATES, "layout-boxes.json")))
```

- [ ] **Step 2: Run it and watch it fail**

Run: `venv/bin/python -m pytest tests/test_flux3_catalog.py -v -n0 -k layout`
Expected: FAIL, file not found.

- [ ] **Step 3: Write the prompt and template**

`prompts/flux3/balloons_layout.json`: a caption in BFL's box format (see the audit's *Prompt format → Layout / JSON* and https://docs.bfl.ml/flux_3/flux3_image_bounding_boxes). That's prose naming each boxed element as `<id>`, then a JSON array of `{"id", "bbox": [top, left, bottom, right] on 0-1000, "desc"}`. Base it on BFL's `examples/inputs/balloons_boxes.json` at bbfe317d, converting its 0-1 floats to 0-1000 integers. Write the prose yourself: the example file holds boxes, not the caption.

`layout-boxes.json`: the Task 4 template shape with `"prompt": "prompt:flux3/balloons_layout"` and seed 7. The description says boxes are a hint, not a mask: small boxed elements may not appear, and elements can run past their box.

- [ ] **Step 4: Run the catalog tests and validate**

Run: `venv/bin/python -m pytest tests/test_flux3_catalog.py tests/test_catalog_structure.py tests/test_prompt_library.py -v -n0 && venv/bin/python -m dw.validate workflows/templates/flux3/layout-boxes.json`
Expected: PASS and valid.

- [ ] **Step 5: Commit**

```bash
git add workflows/templates/flux3/layout-boxes.json prompts/flux3/balloons_layout.json tests/test_flux3_catalog.py
git commit -m "feat(flux3): layout-with-boxes template

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Family README

**Files:**
- Create: `workflows/templates/flux3/README.md`
- Test: the existing README link checks (`grep -rln "README" tests/ | head` to find them; `tests/test_plugin_skills.py` and `tests/test_catalog_structure.py` are likely)

- [ ] **Step 1: Write the README**

Follow `workflows/templates/kandinsky6/README.md`:
- **First paragraph:** the model (5.6B distilled flow map, 16 steps, no guidance; the FLUX.2 VAE; the Qwen3-VL-4B text encoder), the sources (the Hugging Face card, github.com/black-forest-labs/flux-3-image-klein, docs.bfl.ml, diffusers `modular_pipelines/flux3`, merged 2026-10-15 and not yet released, so diffusers must be installed from git), the gated FLUX Kommunity License, and a link to `../../../docs/proposals/audits/2026-10-15-flux-3-klein-audit.md`.
- **A prompt paragraph:** BFL says not to use Klein without its upsampler, and the upsampler builtins read BFL's system prompts from the pup repo at a pinned commit. For writing captions by hand, link BFL's prompting guide (https://docs.bfl.ml/guides/prompting_summary) and note it's written for the API model, which upsamples on the server side.
- **A reading-order table:** text-to-image, upsample-then-generate, image-edit, multi-reference, layout-boxes, one idea each.
- **A placement note** marked as dw craft: every component placed explicitly, and why.
- **The agent pointer** to the `flux3-klein` skill of the dw plugin.

- [ ] **Step 2: Run the link and catalog tests**

Run: `venv/bin/python -m pytest tests/test_catalog_structure.py tests/test_plugin_skills.py -v -n0`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add workflows/templates/flux3/README.md
git commit -m "docs(flux3): Klein templates README

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: GPU runs and costs (lem)

**Files:**
- Modify: every `workflows/templates/flux3/*.json` and `workflows/models/flux3-klein.json` (`cost`)
- Modify: the spec's probe table

- [ ] **Step 1: Deploy the branch to lem**

Ask Don to deploy `feat/flux3-klein` to lem (the auto-mode classifier refuses `deploy.sh` over ssh). Then confirm with `get_diffusers_state` that the commit is at or after 7ecb1e1b. If `update_diffusers` was used, `dw-serve` must be restarted before validation sees the new classes.

- [ ] **Step 2: Run each template once warm**

For each template: `validate_workflow` with `workflow_path` set to the catalog name, quote the estimate, get Don's go-ahead (one batch approval for all five is fine), then `run_workflow`, `wait_for_job`, `get_job`, and look at each output (`get_output_image`, `get_output_text` for captions). Run each one a second time to get a warm timing, since the first run loads the models.

- [ ] **Step 3: Write `cost`**

Add `"cost": [{"device": "cuda", "name": "RTX 3090", "vram_gb": 24, "minutes": <warm minutes, one decimal, rounded up>}]` to each template and the model entry. Record each job id, warm minutes and peak reserved memory in the spec's probe table.

- [ ] **Step 4: Run the catalog tests, then commit**

Run: `venv/bin/python -m pytest tests/test_catalog_structure.py tests/test_flux3_catalog.py -v -n0`

```bash
git add workflows/templates/flux3 workflows/models/flux3-klein.json docs/superpowers/specs/2026-10-15-flux-3-klein-catalog-design.md
git commit -m "feat(flux3): measured RTX 3090 costs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: The `flux3-klein` composition skill

**Files:**
- Create: `plugins/dw/skills/flux3-klein/SKILL.md`
- Modify: `plugins/dw/README.md` (name the skill)
- Modify: `CLAUDE.md` (the plugin paragraph lists every skill)
- Modify: `tests/test_plugin_skills.py` (new `TestFlux3KleinSkill`)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_plugin_skills.py`)

```python
FLUX3_SKILL = os.path.join(PLUGIN_DIR, "skills", "flux3-klein", "SKILL.md")


class TestFlux3KleinSkill:
    """The Klein skill's numbers come from the diffusers modular blocks that
    enforce them; the 10-reference cap is the vendor's, which the pipeline
    does not enforce, and the skill says so."""

    def test_the_step_count_and_grid_are_the_pipeline_s(self):
        import inspect
        from diffusers.modular_pipelines.flux3 import before_denoise, modular_blocks_flux3_distilled
        text = skill_text(FLUX3_SKILL)
        assert "16" in inspect.getsource(modular_blocks_flux3_distilled)
        assert "vae_scale_factor * 2" in inspect.getsource(before_denoise)
        assert "16 steps" in text
        assert "multiples of 16" in text

    def test_the_distilled_model_takes_no_guidance(self):
        assert "no guidance" in skill_text(FLUX3_SKILL)

    def test_the_reference_cap_is_stated_as_the_vendor_s(self):
        text = skill_text(FLUX3_SKILL)
        assert "10 references" in text and "<ref_image_0>" in text

    def test_the_skill_defers_prompt_format_to_bfl(self):
        text = skill_text(FLUX3_SKILL)
        assert "docs.bfl.ml" in text
        assert "flux3_klein_upsample" in text

    def test_the_skill_starts_with_the_server(self):
        text = skill_text(FLUX3_SKILL)
        assert text.index("get_server_info") < text.index("list_workflows")
```

Read the existing `TestKandinsky6Skill` and `skill_text` helper first and match their style. If the parametrized skill tests (size cap, links, catalog names) already pick up every `SKILL.md` under `plugins/dw/skills/`, they cover the new file with no change.

- [ ] **Step 2: Run them and watch them fail**

Run: `venv/bin/python -m pytest tests/test_plugin_skills.py -v -n0 -k Flux3`
Expected: FAIL, file not found.

- [ ] **Step 3: Write the skill**

Start from a copy of `plugins/dw/skills/kandinsky-6/SKILL.md` and replace every section:
- **Frontmatter `description`:** written for triggering ("Use when a dw MCP server is connected and the user wants an image from FLUX 3 Image Klein - a picture from a request, an edit of their photo, several references combined, or a layout with boxes…").
- **Start with the server:** `get_server_info`, then `list_workflows(shape="image")` / `shape="image-edit"`.
- **The shape decision:** short request → upsample-then-generate; full caption → text-to-image; one photo to change → image-edit; several → multi-reference; positions → layout-boxes.
- **Hard rules:** 16 steps, no guidance, multiples of 16, up to 4096, 10 references named `<ref_image_N>`, gated license and `hf auth login`, explicit placement.
- **Prompts:** use the upsampler builtins (`flux3_klein_upsample`, `flux3_klein_upsample_edit`), and for hand-written captions BFL's guide at docs.bfl.ml. Nothing transcribed.
- **Validate, quote cost, run, look.**
- **Failure modes:** generic output from short prompts without the upsampler; boxes are hints; edits can change content outside the requested region; small reference details can be misdescribed.
- **Sources with dates.**

Keep it under the 12 KB cap.

Add the skill to `plugins/dw/README.md`'s list and to `CLAUDE.md`'s plugin paragraph (one composition skill per model family: add `flux3-klein`).

- [ ] **Step 4: Run the plugin tests**

Run: `venv/bin/python -m pytest tests/test_plugin_skills.py -v -n0`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/dw/skills/flux3-klein plugins/dw/README.md CLAUDE.md tests/test_plugin_skills.py
git commit -m "feat(plugin): flux3-klein composition skill

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Full suite, cold drill, close

- [ ] **Step 1: Full suite and lint**

Run: `venv/bin/python -m pytest tests/ -q` and `venv/bin/ruff check dw tests`
Expected: all pass, no lint errors.

- [ ] **Step 2: Cold drill**

Follow `.claude/skills/model-family-onboarding/references/cold-drill-example.md`. Use lem on this branch with `dw-serve` restarted, a fresh Claude Code session in an empty directory with the plugin installed, and the request "make me a picture of a lighthouse keeper's cat on a stormy night". Run a control without the plugin. Record both transcripts' paths and the verdict (the skill fired, upsample-then-generate was chosen, cost was quoted before the run, the image was looked at) in the spec.

- [ ] **Step 3: Follow-ups in the spec, dated**

The spec's *Follow-ups* section holds these; add any new ones:
- Base + CFG (`guider`)
- fp8r/NVFP4
- LoRA at 0.75
- typography
- MPS on mini-ai
- upsampled boxes (`t2i_bbox`)
- a list-length constraint for the 10-reference cap
- the `downloads_required` gaps
- re-pinning the system prompts and the scaffold when diffusers PR #15058 merges

- [ ] **Step 4: Hand off**

Use superpowers:finishing-a-development-branch.
