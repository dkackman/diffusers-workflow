# LoRA Catalog and Recommender Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `loras` library of proven/trial/rejected LoRAs keyed by exact base repo, served over HTTP and MCP (`list_loras`, `save_lora`), plus an opt-in `recommend_loras` that adds filtered, inspected Hugging Face Hub candidates.

**Architecture:** A fourth `dw/library.py` kind (`loras`) reuses `LibraryPath`; the server's writable LoRA library is `<workspace root>/loras`, shared by every workspace, with the shipped top-level `loras/` read-only behind it. `dw/lora_catalog.py` (no network) owns the schema, base-model resolution, matching and ranking; `dw/lora_hub.py` is the only module that searches the Hub. Routes live in `dw/server/routes/loras.py`; MCP tools in `dw_mcp/loras.py` + a `LoraTools` class.

**Tech Stack:** Python 3.12+, FastAPI, `huggingface_hub` 1.33 (`HfApi.list_models(expand=...)`, `HfApi.parse_safetensors_file_metadata`), jsonschema via `dw.schema.validate_data`, the `mcp` SDK, pytest (+xdist).

**Spec:** `docs/superpowers/specs/2026-10-03-lora-catalog-design.md`

## Global Constraints

- Matching is **exact** on `base_models` - no aliases, no prefix matching.
- Every path into a library goes through a named validator in `dw/security.py` (`validate_lora_path`, `validate_lora_name`), each added to the CodeQL models (`.github/codeql/extensions/dw-models/models/dw-security.model.yml`, `.github/codeql/dw-security/DwPathSanitizers.qll`).
- Never `eval`/`exec`/`shell=True`; no URL is built from Hub data; Hub repo ids are checked against the repo-id pattern before reuse.
- No network in the test suite: every Hub call goes through an injectable `api` object and is faked.
- `recommend_loras` spends no GPU and downloads no weights; it takes no `acknowledged_cost`.
- Hub results are `status: "candidate"`, never "recommended"; a catalog-`rejected` repo comes back `status: "rejected"`.
- `SURFACE_BUDGET` (`tests/test_mcp_server.py`) is raised by the measured cost of the three new tools, with a dated comment.
- Run tests with `venv/bin/python -m pytest` (local venv has torch); do not ship tests to lem.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A workflow whose `model_name` is a `variable:` reference (every H3 template's LoRA, many bases) - resolution substitutes the variable's default, and skips it when the default is not a repo id. Pinned in Task 2.
2. An existing named workspace called `loras` - `loras` becomes a reserved workspace name; `create_workspace("loras")` is refused rather than colliding with the library directory. Pinned in Task 1.
3. A query of only stop words (`"style"`, `"lora"`, `""`) - the Hub search still runs, unfiltered, ordered by downloads, rather than returning nothing or erroring. Pinned in Task 5.
4. A Hub repo with no card (`card_data is None`), no `lastModified`, or `instance_prompt` given as a list - a candidate with warnings, never a crash. Pinned in Task 5.
5. Saving an entry named `recommend` - refused, since `GET /api/loras/recommend` would make it unreadable by name. Pinned in Task 4.

---

### Task 1: The `loras` library kind

**Files:**
- Modify: `dw/security.py` (after `validate_prompt_reference`, ~line 417; after `validate_prompt_path`, ~line 205)
- Modify: `dw/library.py:158-170` (kinds), `dw/library.py:210-225` (`path_in`), `dw/library.py:364-365` (`_front_directory`), `dw/library.py:368-383` (`read_only_candidates`)
- Modify: `dw/workspace.py:58-96` (subdir constants, reserved names), `dw/workspace.py:288-313` (`example_libraries`)
- Modify: `.github/codeql/extensions/dw-models/models/dw-security.model.yml`, `.github/codeql/dw-security/DwPathSanitizers.qll:47`
- Test: `tests/test_lora_library.py`

**Interfaces:**
- Produces: `dw.library.LORAS_KIND = "loras"`; `dw.workspace.LORAS_SUBDIR = "loras"`; `dw.security.validate_lora_path(path, lora_dir) -> str`; `dw.security.validate_lora_name(name) -> str` (raises `InvalidInputError`); `library_path(LORAS_KIND, workspace, examples_dirs, primary=...)` works, including on a `ConfiguredWorkspace` with `primary=None` (no front).

- [ ] **Step 1: Write the failing tests**

```python
"""The `loras` library kind: a JSON library like prompts, whose writable front
the server names, behind which the shipped loras/ sits read-only."""

import json
import os

import pytest

from dw.library import (
    EXAMPLES_ORIGIN,
    LORAS_KIND,
    WORKSPACE_ORIGIN,
    ReadOnlyLibraryError,
    library_path,
)
from dw.security import (
    InvalidInputError,
    SecurityError,
    validate_lora_name,
    validate_lora_path,
)
from dw.workspace import (
    LORAS_SUBDIR,
    ConfiguredWorkspace,
    Workspace,
    create_workspace,
    example_libraries,
)


def write_entry(root, name, body=None):
    path = os.path.join(root, f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(body or {"model_name": "a/b"}, file)
    return path


@pytest.fixture
def libraries(tmp_path):
    checkout = tmp_path / "repo"
    (checkout / "workflows").mkdir(parents=True)
    shipped = checkout / LORAS_SUBDIR
    own = tmp_path / "studio" / LORAS_SUBDIR
    write_entry(str(shipped), "qwen-image/voxel", {"model_name": "shipped/voxel"})
    write_entry(str(shipped), "flux/realism", {"model_name": "shipped/realism"})
    write_entry(str(own), "qwen-image/voxel", {"model_name": "own/voxel"})
    workspace = Workspace(tmp_path / "studio", "flag")
    path = library_path(
        LORAS_KIND, workspace, [str(checkout / "workflows")], primary=str(own)
    )
    return path, str(own), str(shipped)


class TestTheKind:
    def test_the_examples_tree_brings_its_loras(self, tmp_path):
        checkout = tmp_path / "repo"
        (checkout / "workflows").mkdir(parents=True)
        (checkout / LORAS_SUBDIR).mkdir()
        found = example_libraries([str(checkout / "workflows")])
        assert found[LORAS_SUBDIR] == [str(checkout / LORAS_SUBDIR)]

    def test_the_workspace_copy_shadows_the_shipped_one(self, libraries):
        path, own, shipped = libraries
        winners, hidden = path.entries()
        assert winners["qwen-image/voxel"].root == own
        assert winners["flux/realism"].root == shipped
        assert [(name, root.root) for name, root, _ in hidden] == [
            ("qwen-image/voxel", shipped)
        ]

    def test_find_resolves_front_to_back(self, libraries):
        path, own, _ = libraries
        found, root = path.find("qwen-image/voxel")
        assert root.origin == WORKSPACE_ORIGIN
        assert found == os.path.join(own, "qwen-image", "voxel.json")

    def test_the_shipped_root_is_read_only(self, libraries):
        path, _, _ = libraries
        _, root = path.find("flux/realism")
        assert root.origin == EXAMPLES_ORIGIN
        with pytest.raises(ReadOnlyLibraryError):
            path.require_writable(root, "flux/realism")

    def test_a_configured_workspace_with_no_front_has_no_writable_root(self, tmp_path):
        configured = ConfiguredWorkspace(
            workflows=tmp_path / "w", assets=None, outputs=tmp_path / "o", prompts=None
        )
        path = library_path(LORAS_KIND, configured, [], primary=None)
        assert path.writable_root() is None


class TestValidators:
    def test_a_name_one_folder_deep_is_accepted(self):
        assert validate_lora_name("qwen-image/voxel-style") == "qwen-image/voxel-style"

    @pytest.mark.parametrize("name", ["../x", "a/b/c", "/abs", "", ".hidden"])
    def test_a_name_that_could_leave_the_library_is_refused(self, name):
        with pytest.raises(InvalidInputError):
            validate_lora_name(name)

    def test_the_path_validator_refuses_traversal(self, tmp_path):
        with pytest.raises(SecurityError):
            validate_lora_path(str(tmp_path / ".." / "x.json"), str(tmp_path))

    def test_the_path_validator_wants_json(self, tmp_path):
        (tmp_path / "x.txt").write_text("{}")
        with pytest.raises(InvalidInputError):
            validate_lora_path(str(tmp_path / "x.txt"), str(tmp_path))


def test_loras_is_a_reserved_workspace_name(tmp_path):
    root = Workspace(tmp_path, "flag").ensure()
    with pytest.raises(InvalidInputError):
        create_workspace(root, LORAS_SUBDIR)
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_lora_library.py -v -p no:xdist`
Expected: FAIL at import (`cannot import name 'LORAS_KIND'`).

- [ ] **Step 3: Implement**

`dw/security.py`, next to `validate_prompt_path`:

```python
def validate_lora_path(path: str, lora_dir: str) -> str:
    """Validate LoRA catalog entry paths, confined to the LoRA library."""
    validated = validate_path(path, lora_dir, allow_create=False)
    return validate_file_extension(validated, ALLOWED_JSON_EXTENSIONS)
```

next to `validate_prompt_reference` (same pattern: a catalog entry is named like a prompt, at most one folder deep - the family folder):

```python
def validate_lora_name(name: str) -> str:
    """Validate a LoRA catalog entry's name before it is joined onto a
    library root: a plain name or one family folder deep."""
    return _validate_name(
        name,
        PROMPT_REFERENCE_PATTERN,
        MAX_PROMPT_REFERENCE_LENGTH,
        "LoRA name",
        "a catalog entry is named by its file under the LoRA library, at most "
        "one folder deep, like 'qwen-image/voxel-style'",
        allowed=PROMPT_REFERENCE_CHARACTERS,
    )
```

`dw/workspace.py`: add `LORAS_SUBDIR = "loras"` after `ASSETS_SUBDIR`, and change
`RESERVED_WORKSPACE_NAMES = SUBDIRS + (EXPORTS_SUBDIR, COMMON_SUBDIR)` to
`RESERVED_WORKSPACE_NAMES = SUBDIRS + (EXPORTS_SUBDIR, COMMON_SUBDIR, LORAS_SUBDIR)` with a comment:
`# loras/ is the root's LoRA library, shared like prompts/ - not created by ensure()`.
In `example_libraries`, change `found = {PROMPTS_SUBDIR: [], ASSETS_SUBDIR: []}` to
`found = {PROMPTS_SUBDIR: [], ASSETS_SUBDIR: [], LORAS_SUBDIR: []}` and add `loras` to the docstring's Returns line.

`dw/library.py`:

```python
LORAS_KIND = "loras"
_JSON_KINDS = {WORKFLOWS_KIND, PROMPTS_KIND, LORAS_KIND}
LIBRARY_KINDS = (WORKFLOWS_KIND, PROMPTS_KIND, ASSETS_KIND, LORAS_KIND)
```

(`LIBRARY_PATH_ENV_VARS` is unchanged: the worker never reads the LoRA catalog, so nothing pins it.) Import `validate_lora_path` and in `path_in` add, after the prompts branch:

```python
            if self.kind == LORAS_KIND and not allow_create:
                return validate_lora_path(candidate, root.root)
```

`_front_directory` becomes `return getattr(workspace, kind, None)` - a `Workspace` has no `loras` attribute, and the server always passes `primary`, so a missing attribute means "no front". Update the `library_path` docstring with a `LoRAs:` paragraph: the server's LoRA directory (writable, `primary`), then each examples tree's `loras/` (read-only).

CodeQL: add `- ["dw.security", "Member[validate_lora_path].ReturnValue", "path-injection"]` after the `validate_prompt_path` line and `- ["dw.security", "Member[validate_lora_name].ReturnValue", "path-injection"]` after `validate_prompt_reference`; in `DwPathSanitizers.qll:47` add `"validate_lora_path", "validate_lora_name"` to the name list.

- [ ] **Step 4: Run to verify they pass, and the neighbours still do**

Run: `venv/bin/python -m pytest tests/test_lora_library.py tests/test_library_path.py tests/test_library_sources.py tests/test_security.py tests/test_workspace*.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/security.py dw/library.py dw/workspace.py .github/codeql tests/test_lora_library.py
git commit -m "feat(library): the loras library kind and its validators

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Catalog core - schema, resolution, matching, ranking

**Files:**
- Create: `dw/lora_catalog_schema.json`
- Create: `dw/lora_catalog.py`
- Test: `tests/test_lora_catalog.py`

**Interfaces:**
- Consumes: `dw.schema.load_schema("lora_catalog")`, `dw.schema.validate_data`, `dw.references.VARIABLE`.
- Produces (all in `dw.lora_catalog`):
  - `SCHEMA_NAME = "lora_catalog"`, `RESERVED_NAMES = frozenset({"recommend"})`, `STATUS_ORDER = {"proven": 0, "trial": 1, "rejected": 2}`
  - `entry_errors(entry: dict) -> str | None`
  - `is_repo_id(value) -> bool`
  - `workflow_bases(definition: dict) -> list[tuple[str, str | None]]` - `(repo, partition)` pairs, in step order, de-duplicated
  - `matches(entry: dict, bases: list[tuple[str, str | None]]) -> bool`
  - `query_terms(query: str | None) -> list[str]`
  - `ranked(entries: dict[str, dict], terms: list[str]) -> list[dict]` - rows `{"name": ..., **entry}`
  - `rejection_reasons(entries: dict[str, dict]) -> dict[str, str]` - `model_name -> reason` for `rejected` entries

- [ ] **Step 1: Write the failing tests**

```python
"""dw/lora_catalog.py: the entry schema, which bases a workflow loads, and
which entries fit them - exactly, never by family."""

import pytest

from dw.lora_catalog import (
    entry_errors,
    is_repo_id,
    matches,
    query_terms,
    ranked,
    rejection_reasons,
    workflow_bases,
)


def entry(**overrides):
    base = {
        "model_name": "prithivMLmods/Qwen-Image-2.1-Voxel-Style",
        "base_models": ["Qwen/Qwen-Image-2.1"],
        "description": "Blocky voxel look",
        "use_when": "voxel or blocky 3D requests",
        "status": "trial",
    }
    base.update(overrides)
    return base


class TestSchema:
    def test_a_minimal_trial_entry_is_valid(self):
        assert entry_errors(entry()) is None

    def test_proven_needs_evidence(self):
        assert entry_errors(entry(status="proven")) is not None
        assert (
            entry_errors(
                entry(status="proven", evidence=[{"issue": 585, "note": "best arm"}])
            )
            is None
        )

    def test_base_models_may_not_be_empty(self):
        assert entry_errors(entry(base_models=[])) is not None

    def test_an_unknown_status_is_refused(self):
        assert entry_errors(entry(status="recommended")) is not None

    def test_an_unknown_field_is_refused(self):
        assert entry_errors(entry(colour="blue")) is not None

    def test_the_partition_vocabulary_is_the_adapter_checks(self):
        assert entry_errors(entry(workflow="ref2va")) is None
        assert entry_errors(entry(workflow="i2v")) is not None


class TestRepoIds:
    @pytest.mark.parametrize("value", ["Qwen/Qwen-Image-2.1", "a/b.c_d-e"])
    def test_a_repo_id(self, value):
        assert is_repo_id(value)

    @pytest.mark.parametrize(
        "value", ["noslash", "a/b/c", "../x", "a/..", None, 3, "variable:x"]
    )
    def test_not_a_repo_id(self, value):
        assert not is_repo_id(value)


class TestWorkflowBases:
    def test_each_pipeline_step_names_its_base_and_partition(self):
        definition = {
            "steps": [
                {
                    "pipeline": {
                        "from_pretrained_arguments": {
                            "model_name": "MiniMaxAI/MiniMax-H3",
                            "workflow": "ref2va",
                        }
                    }
                },
                {
                    "pipeline": {
                        "from_pretrained_arguments": {
                            "model_name": "Qwen/Qwen-Image-2.1"
                        }
                    }
                },
                {"task": {"command": "gather_images"}},
            ]
        }
        assert workflow_bases(definition) == [
            ("MiniMaxAI/MiniMax-H3", "ref2va"),
            ("Qwen/Qwen-Image-2.1", None),
        ]

    def test_a_variable_reference_takes_the_variables_default(self):
        definition = {
            "variables": {"model": "Qwen/Qwen-Image-2.1"},
            "steps": [
                {
                    "pipeline": {
                        "from_pretrained_arguments": {"model_name": "variable:model"}
                    }
                }
            ],
        }
        assert workflow_bases(definition) == [("Qwen/Qwen-Image-2.1", None)]

    def test_a_variable_with_no_repo_default_is_skipped(self):
        definition = {
            "variables": {"model": None},
            "steps": [
                {
                    "pipeline": {
                        "from_pretrained_arguments": {"model_name": "variable:model"}
                    }
                }
            ],
        }
        assert workflow_bases(definition) == []

    def test_a_repeated_base_is_listed_once(self):
        step = {"pipeline": {"from_pretrained_arguments": {"model_name": "a/b"}}}
        assert workflow_bases({"steps": [step, step]}) == [("a/b", None)]


class TestMatching:
    def test_the_base_must_match_exactly(self):
        assert matches(entry(), [("Qwen/Qwen-Image-2.1", None)])
        assert not matches(entry(), [("Qwen/Qwen-Image-2.1-2509", None)])
        assert not matches(entry(), [("qwen/qwen-image-2.1", None)])

    def test_a_partitioned_entry_fits_only_its_partition(self):
        h3 = entry(base_models=["MiniMaxAI/MiniMax-H3"], workflow="t2va")
        assert matches(h3, [("MiniMaxAI/MiniMax-H3", "t2va")])
        assert not matches(h3, [("MiniMaxAI/MiniMax-H3", "ref2va")])

    def test_a_bare_repo_lists_every_partition(self):
        h3 = entry(base_models=["MiniMaxAI/MiniMax-H3"], workflow="t2va")
        assert matches(h3, [("MiniMaxAI/MiniMax-H3", None)])


class TestRanking:
    def test_stop_words_and_short_words_are_dropped(self):
        assert query_terms("a Voxel style LoRA, isometric!") == ["voxel", "isometric"]
        assert query_terms("style") == []
        assert query_terms(None) == []

    def test_query_hits_rank_first_then_status_then_name(self):
        entries = {
            "b": entry(use_when="anything", status="proven", evidence=[{"note": "x"}]),
            "a": entry(use_when="voxel art"),
            "c": entry(use_when="voxel art", status="proven", evidence=[{"note": "x"}]),
        }
        assert [row["name"] for row in ranked(entries, ["voxel"])] == ["c", "a", "b"]

    def test_tags_count_toward_the_score(self):
        entries = {"x": entry(use_when="", tags=["voxel"]), "y": entry(use_when="")}
        assert ranked(entries, ["voxel"])[0]["name"] == "x"

    def test_rejection_reasons_come_from_the_first_evidence_note(self):
        entries = {
            "fast": entry(
                model_name="drozbay/FastH3",
                status="rejected",
                evidence=[{"note": ".diff keys"}],
            ),
            "ok": entry(),
        }
        assert rejection_reasons(entries) == {"drozbay/FastH3": ".diff keys"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_lora_catalog.py -v -p no:xdist`
Expected: FAIL at import (`No module named 'dw.lora_catalog'`).

- [ ] **Step 3: Write the schema** - `dw/lora_catalog_schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "LoRA catalog entry",
  "type": "object",
  "additionalProperties": false,
  "required": ["model_name", "base_models", "description", "use_when", "status"],
  "properties": {
    "model_name": { "type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$" },
    "weight_name": { "type": "string", "pattern": "\\.safetensors$" },
    "revision": { "type": "string", "minLength": 1 },
    "base_models": {
      "type": "array", "minItems": 1, "uniqueItems": true,
      "items": { "type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$" }
    },
    "workflow": { "type": ["string", "null"], "enum": ["t2va", "fl2va", "ref2va", null] },
    "description": { "type": "string", "minLength": 1 },
    "use_when": { "type": "string", "minLength": 1 },
    "trigger": { "type": "string" },
    "scale": {
      "type": "object", "additionalProperties": false, "required": ["default"],
      "properties": {
        "default": { "type": "number" },
        "range": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 }
      }
    },
    "stacks_with": { "type": "array", "items": { "type": "string" } },
    "status": { "enum": ["proven", "trial", "rejected"] },
    "evidence": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false, "required": ["note"],
        "properties": {
          "job": { "type": "string" },
          "issue": { "type": "integer" },
          "note": { "type": "string", "minLength": 1 }
        }
      }
    },
    "license": { "type": "string" },
    "tags": { "type": "array", "items": { "type": "string" } }
  },
  "allOf": [
    {
      "if": { "properties": { "status": { "enum": ["proven", "rejected"] } } },
      "then": { "required": ["evidence"], "properties": { "evidence": { "minItems": 1 } } }
    }
  ]
}
```

(`rejected` needs evidence too - its first note is the reason Task 6 hands back.)

- [ ] **Step 4: Write `dw/lora_catalog.py`**

```python
"""The LoRA catalog: which adapters fit a base model, and in what order to
offer them.

An entry (`dw/lora_catalog_schema.json`) records a LoRA that was tried on a
base - proven, still a trial, or rejected with the reason. Matching is exact
on the base repo id and, for MiniMax-H3, on the partition a step denoises
against (the `workflow=` value `dw/adapter_compatibility.py` reads): a LoRA
on the wrong base usually loads without complaint and is quietly worse, so a
family prefix or an alias would list exactly the entries that fail silently.

No network here - `dw/lora_hub.py` is the only module that searches the Hub.
"""

import re

from . import references
from .schema import load_schema, validate_data

SCHEMA_NAME = "lora_catalog"
# A catalog entry by this name would be shadowed by GET /api/loras/recommend
RESERVED_NAMES = frozenset({"recommend"})
STATUS_ORDER = {"proven": 0, "trial": 1, "rejected": 2}

REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
# Words that say nothing about which LoRA: every style LoRA is a "style lora"
STOP_WORDS = frozenset(
    {"style", "lora", "loras", "the", "and", "with", "for", "image", "images"}
)
MIN_TERM_LENGTH = 3


def entry_errors(entry):
    """The schema's complaint about an entry, or None when it is valid."""
    valid, message = validate_data(entry, load_schema(SCHEMA_NAME))
    return None if valid else message


def is_repo_id(value):
    """Whether `value` is a Hub repo id - `owner/name`, no traversal."""
    return isinstance(value, str) and bool(REPO_ID.match(value)) and ".." not in value


def _resolved(value, variables):
    """A `variable:` reference replaced by the variable's default; anything
    else unchanged."""
    if isinstance(value, str) and value.startswith(references.VARIABLE):
        return variables.get(value[len(references.VARIABLE) :])
    return value


def workflow_bases(definition):
    """The `(repo, partition)` each pipeline step loads, in step order, once
    each. A base that is not a repo id once its variable is substituted (a
    null default, a local path) is skipped: nothing on the Hub is keyed by it."""
    variables = definition.get("variables") or {}
    bases = []
    for step in definition.get("steps") or []:
        pipeline = step.get("pipeline") if isinstance(step, dict) else None
        if not isinstance(pipeline, dict):
            continue
        arguments = pipeline.get("from_pretrained_arguments") or {}
        repo = _resolved(arguments.get("model_name"), variables)
        if not is_repo_id(repo):
            continue
        partition = _resolved(arguments.get("workflow"), variables)
        pair = (repo, partition if isinstance(partition, str) else None)
        if pair not in bases:
            bases.append(pair)
    return bases


def matches(entry, bases):
    """Whether an entry fits any of `bases`. A partitioned entry fits a step
    of that partition, or a bare repo (no partition asked for)."""
    constraint = entry.get("workflow")
    for repo, partition in bases:
        if repo not in entry.get("base_models", []):
            continue
        if constraint is None or partition is None or constraint == partition:
            return True
    return False


def query_terms(query):
    """The words of a request worth matching on, lower-cased, in order."""
    words = re.findall(r"[a-z0-9]+", (query or "").lower())
    terms = []
    for word in words:
        if (
            len(word) >= MIN_TERM_LENGTH
            and word not in STOP_WORDS
            and word not in terms
        ):
            terms.append(word)
    return terms


def _score(entry, terms):
    text = " ".join(
        [entry.get("use_when", ""), entry.get("description", "")]
        + list(entry.get("tags", []))
    ).lower()
    return sum(1 for term in terms if term in text)


def ranked(entries, terms):
    """Entries as rows (`{"name", **entry}`): most query terms first, then
    proven before trial before rejected, then by name."""
    rows = [{"name": name, **entry} for name, entry in entries.items()]
    rows.sort(
        key=lambda row: (
            -_score(row, terms),
            STATUS_ORDER.get(row.get("status"), len(STATUS_ORDER)),
            row["name"],
        )
    )
    return rows


def rejection_reasons(entries):
    """`model_name -> reason` for every rejected entry; the reason is its
    first evidence note, which the schema requires."""
    return {
        entry["model_name"]: entry["evidence"][0]["note"]
        for entry in entries.values()
        if entry.get("status") == "rejected" and entry.get("evidence")
    }
```

- [ ] **Step 5: Run to verify they pass**

Run: `venv/bin/python -m pytest tests/test_lora_catalog.py -v -p no:xdist`
Expected: PASS. If `validate_data` reports the `enum` with `null` oddly under `type: ["string","null"]`, keep both keywords - jsonschema accepts that form.

- [ ] **Step 6: Commit**

```bash
git add dw/lora_catalog.py dw/lora_catalog_schema.json tests/test_lora_catalog.py
git commit -m "feat(loras): catalog entry schema, base resolution and exact matching

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Seed the shipped catalog

**Files:**
- Create: `loras/minimax-h3/realism-people.json`, `loras/minimax-h3/turbo-keyframe.json`, `loras/minimax-h3/turbo-reference.json`, `loras/minimax-h3/turbo-4step-keyframe.json`, `loras/minimax-h3/fasth3-preview.json`, `loras/minimax-h3/acc-pdd.json`, `loras/minimax-h3/hyperflow.json`, `loras/ltx-2.5/ic-pixel-spatial-upscaler.json`, `loras/ltx-2.5/ic-decompression.json`, `loras/ltx-2.5/ic-deblur.json`, `loras/ltx-2.5/ic-ingredients.json`, `loras/flux/in-context.json`, `loras/flux/realism.json`
- Test: `tests/test_lora_catalog.py` (append)

**Interfaces:**
- Consumes: `entry_errors`, `is_repo_id` from Task 2; `workflow_bases` to cross-check.
- Produces: the shipped corpus under top-level `loras/`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_lora_catalog.py`)

```python
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED = sorted(glob.glob(os.path.join(REPO, "loras", "**", "*.json"), recursive=True))


def test_the_catalog_ships_entries():
    assert len(SHIPPED) >= 13


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: os.path.relpath(p, REPO))
def test_every_shipped_entry_is_valid(path):
    with open(path) as file:
        assert entry_errors(json.load(file)) is None


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: os.path.relpath(p, REPO))
def test_every_shipped_entry_pins_a_revision(path):
    with open(path) as file:
        assert json.load(file).get("revision"), (
            "shipped entries pin the commit they were tried at"
        )


def test_every_proven_base_is_one_a_catalog_workflow_loads():
    """A proven entry for a base no shipped workflow loads is knowledge
    nobody can reach from list_workflows."""
    loaded = set()
    for path in glob.glob(
        os.path.join(REPO, "workflows", "**", "*.json"), recursive=True
    ):
        with open(path) as file:
            try:
                definition = json.load(file)
            except ValueError:
                continue
        loaded |= {repo for repo, _ in workflow_bases(definition)}
    for path in SHIPPED:
        with open(path) as file:
            data = json.load(file)
        if data["status"] == "proven":
            assert set(data["base_models"]) & loaded, path
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_lora_catalog.py -v -p no:xdist -k shipped`
Expected: FAIL (`test_the_catalog_ships_entries`: 0 >= 13).

- [ ] **Step 3: Look up the pinned revisions and the rejected repos' ids**

Run (network, once, at authoring time - not in tests):

```bash
venv/bin/python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
for repo in [
    "fal/MiniMax-H3-Realism-People-LoRA", "lightx2v/Minimax-h3-Turbo",
    "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler",
    "Lightricks/LTX-2.5-22b-IC-LoRA-Decompression", "Lightricks/LTX-2.5-22b-IC-LoRA-Deblur",
    "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients", "ali-vilab/In-Context-LoRA",
    "XLabs-AI/flux-RealismLora",
]:
    info = api.model_info(repo)
    print(repo, info.sha, getattr(info.card_data, "license", None) if info.card_data else None)
for author, search in [("drozbay", "H3"), ("alibaba-pai", "H3"), ("videorebirth", "HyperFlow")]:
    print(author, [m.id for m in api.list_models(author=author, search=search, limit=10)])
EOF
```

Use each printed `sha` as `revision` and the card's license as `license`. For the three rejected entries pick the repo whose id names the H3 adapter the #585 eval tried (FastH3 preview rank128; Acc PDD; HyperFlow); if a list is ambiguous, open the repo pages listed and match against #585's description. A gated repo you cannot read: write its entry without `license`, keep `revision` from `list_models(..., expand=["sha"])`.

- [ ] **Step 4: Write the entries.** Exact contents (fill `revision`/`license` from Step 3's output; every other value is given):

`loras/minimax-h3/realism-people.json`:
```json
{
  "model_name": "fal/MiniMax-H3-Realism-People-LoRA",
  "weight_name": "h3-realism-people-t2v-i2v-r2v.safetensors",
  "revision": "<sha from step 3>",
  "base_models": ["MiniMaxAI/MiniMax-H3"],
  "workflow": "t2va",
  "description": "Realistic people and faces; stacks as a second adapter on the 8-step turbo",
  "use_when": "People, crowds or faces should look photoreal; tried on T2VA only. It tightens framing against a prompt's wide shot",
  "scale": { "default": 0.7, "range": [0.7, 0.7] },
  "stacks_with": ["minimax-h3/turbo-keyframe"],
  "status": "proven",
  "evidence": [{ "issue": 585, "job": "aaa061f40bc5", "note": "Best arm of the 2026-10-03 eval, about 6 s over the 9.5 min 768p baseline; 1.0 and a second prompt untested" }],
  "tags": ["realism", "faces", "people", "crowd"]
}
```
(If Step 3 shows the job id `aaa061f40bc5` is not the Realism arm, take the right one from `h3-lora-eval` jobs `1559f5dac1b6, aaa061f40bc5, 59db5a9f79bd, 4d909dc0d10d, a8126062195c, b6c4564e067c` via `get_job`, or drop `job` and keep `issue`.)

`loras/minimax-h3/turbo-keyframe.json`:
```json
{
  "model_name": "lightx2v/Minimax-h3-Turbo",
  "weight_name": "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors",
  "revision": "<sha>",
  "base_models": ["MiniMaxAI/MiniMax-H3"],
  "workflow": "t2va",
  "description": "8-step turbo for the keyframe partition (t2va/fl2va), 768p",
  "use_when": "Any T2VA or FL2VA render; the shipped templates already load it through lora_model_name/lora_weight_name",
  "scale": { "default": 1.0 },
  "status": "proven",
  "evidence": [{ "issue": 585, "note": "Shipped default of the 768p templates; 9.5 min warm baseline" }],
  "tags": ["turbo", "speed"]
}
```

`loras/minimax-h3/turbo-reference.json`: as above with `"weight_name": "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"`, `"workflow": "ref2va"`, `"description": "8-step turbo for the reference partition (ref2va), 768p"`, `"use_when": "Any Ref2VA render; never on a t2va/fl2va step (#149, #155)"`, `"evidence": [{ "issue": 155, "note": "The partition rule adapter_compatibility.py enforces" }]`.

`loras/minimax-h3/turbo-4step-keyframe.json`: `"weight_name": "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"`, `"workflow": "t2va"`, `"description": "4-step v1.2 turbo, keyframe partition, 768p - a draft path"`, `"use_when": "A faster draft render (~25% faster); set num_inference_steps 5. Keep 8-step for finals"`, `"status": "proven"`, `"evidence": [{ "issue": 585, "note": "7.1 min vs 9.5 min, faces good; 8-step possibly slightly better" }]`, `"tags": ["turbo", "speed", "draft"]`.

`loras/minimax-h3/fasth3-preview.json`, `acc-pdd.json`, `hyperflow.json`: `"status": "rejected"`, `"base_models": ["MiniMaxAI/MiniMax-H3"]`, `use_when` "Do not use under dw", and the `evidence[0].note`:
- FastH3: `"diffusers' MiniMax-H3 converter rejects its .diff/.diff_b full-weight keys (state_dict should be empty at this point)"`
- Acc PDD: `"Carries a 32-copy output-head bank and needs its own apply_pdd_lora loader; not a plain LoRA"`
- HyperFlow: `"Needs load_hyperflow_lora for its time-embedding changes; its license excludes the US, EU, UK and South Korea"`
each with `"issue": 585`.

`loras/ltx-2.5/*.json` - `"base_models": ["Lightricks/LTX-2.5-Diffusers"]`, `"status": "proven"`, `"scale": {"default": 1.0}`, model and weight names exactly as the templates name them:
- `ic-pixel-spatial-upscaler`: `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` / `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors`; use_when "A 2x re-render of a clip (the upscale templates)"
- `ic-decompression`: `Lightricks/LTX-2.5-22b-IC-LoRA-Decompression` / `ltx-2.5-22b-ic-lora-decompression-0.9.safetensors`; use_when "Restore a compressed clip"
- `ic-deblur`: `Lightricks/LTX-2.5-22b-IC-LoRA-Deblur` / `ltx-2.5-22b-ic-lora-deblur-0.9.safetensors`; use_when "Restore a blurry clip"
- `ic-ingredients`: `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients` / `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors`; use_when "Hold a subject across a clip from a reference sheet"
each with `"evidence": [{"note": "Used by the shipped workflows/templates/ltx2 template of that name"}]` and tags from the use_when.

`loras/flux/in-context.json`: `ali-vilab/In-Context-LoRA`, no `weight_name` at the entry level is invalid for a multi-file repo, so set `"weight_name": "film-storyboard.safetensors"`, `"base_models": ["black-forest-labs/FLUX.1-dev"]`, `"description": "In-context multi-panel layouts; the repo holds ten (font-design, home-decoration, portrait-photography, ppt-templates, sandstorm/sparklers visual effects, visual-identity-design, portrait-illustration, couple-profile)"`, `"use_when": "Storyboards, multi-panel sheets, identity/design boards; swap weight_name for the other layouts"`, `"trigger": "[MOVIE-SHOTS]"`, `"status": "proven"`, evidence note "workflows/templates/lora-styles.json".

`loras/flux/realism.json`: `XLabs-AI/flux-RealismLora`, `"base_models": ["black-forest-labs/FLUX.1-dev"]`, `"use_when": "Photoreal Flux images"`, `"status": "proven"`, evidence note "workflows/templates/lora.json". Read the repo's file list in Step 3 output if it holds more than one `.safetensors` and set `weight_name` accordingly.

- [ ] **Step 5: Run to verify they pass**

Run: `venv/bin/python -m pytest tests/test_lora_catalog.py -v -p no:xdist`
Expected: PASS. A failing `test_every_proven_base_is_one_a_catalog_workflow_loads` means a base id was mistyped - check it against `grep -rh '"model_name"' workflows`.

- [ ] **Step 6: Commit**

```bash
git add loras tests/test_lora_catalog.py
git commit -m "feat(loras): seed the shipped catalog from the templates and #585

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: HTTP routes - list, get, save, delete, schema

**Files:**
- Create: `dw/server/routes/loras.py`
- Modify: `dw/server/routes/__init__.py` (import `loras`, add `loras.router` after `library.router`)
- Modify: `dw/server/app.py:63-105` (`_store_directories`: `state.lora_dir`)
- Modify: `dw/server/deps.py` (add `server_lora_library`, `lora_library_missing`)
- Test: `tests/test_server_loras.py`

**Interfaces:**
- Consumes: Task 1 (`LORAS_KIND`, `validate_lora_name`, `LORAS_SUBDIR`), Task 2 (everything).
- Produces:
  - `dw.server.deps.server_lora_library(state) -> LibraryPath`
  - `dw.server.routes.loras.catalog_entries(state) -> tuple[dict[str, dict], dict[str, LibraryRoot], LibraryPath]` - valid entries by name, their roots, the path
  - `dw.server.routes.loras.resolve_model(state, ws, model) -> list[tuple[str, str | None]]` (raises 400)
  - routes `GET /api/loras`, `GET /api/lora-schema`, `GET|PUT|DELETE /api/loras/{name:path}`; response of `GET /api/loras`: `{"libraries": [...], "loras": [row...], "resolved"?: [{"repo", "workflow"}]}` with each row `{"name", "origin", "writable", **entry}`.

- [ ] **Step 1: Write the failing tests**

```python
"""The LoRA catalog over HTTP: the server's own library at <root>/loras,
shared by every workspace, with an examples tree's loras/ behind it."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.workspace import Workspace, create_workspace

from .test_server import ScriptedWorkerManager, success_script

QWEN = "Qwen/Qwen-Image-2.1"


def entry(**overrides):
    body = {
        "model_name": "prithivMLmods/Qwen-Image-2.1-Voxel-Style",
        "base_models": [QWEN],
        "description": "Blocky voxel look",
        "use_when": "voxel or blocky 3D requests",
        "status": "trial",
    }
    body.update(overrides)
    return body


def write(root, name, body):
    path = os.path.join(root, f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(body, file)


@pytest.fixture
def server(tmp_path, monkeypatch):
    for variable in (
        "DW_PROMPT_PATH",
        "DW_ASSET_PATH",
        "DW_WORKFLOW_PATH",
        "DW_PROMPT_DIR",
        "DW_ASSET_DIR",
    ):
        monkeypatch.delenv(variable, raising=False)
    checkout = tmp_path / "repo"
    (checkout / "workflows" / "models").mkdir(parents=True)
    write(
        str(checkout / "workflows"),
        "models/qwen",
        {
            "steps": [
                {
                    "name": "s",
                    "pipeline": {"from_pretrained_arguments": {"model_name": QWEN}},
                }
            ]
        },
    )
    write(
        str(checkout / "workflows"),
        "models/h3-ref",
        {
            "steps": [
                {
                    "name": "s",
                    "pipeline": {
                        "from_pretrained_arguments": {
                            "model_name": "MiniMaxAI/MiniMax-H3",
                            "workflow": "ref2va",
                        }
                    },
                }
            ]
        },
    )
    write(str(checkout / "loras"), "qwen-image/voxel", entry())
    write(
        str(checkout / "loras"),
        "minimax-h3/realism",
        entry(
            model_name="fal/R",
            base_models=["MiniMaxAI/MiniMax-H3"],
            workflow="t2va",
            status="proven",
            evidence=[{"note": "best"}],
        ),
    )
    write(str(checkout / "loras"), "broken", {"model_name": "not valid"})
    workspace = Workspace(tmp_path / "studio", "flag").ensure()
    manager = JobManager(
        workspace.outputs,
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=workspace.workflows,
    )
    app = create_app(
        workflow_dir=workspace.workflows,
        output_dir=workspace.outputs,
        job_manager=manager,
        prompt_dir=workspace.prompts,
        asset_dir=workspace.assets,
        examples_dirs=[str(checkout / "workflows")],
        workspace=workspace.root,
    )
    with TestClient(app, base_url="http://localhost") as client:
        client.workspace = workspace
        client.checkout = checkout
        yield client


class TestListing:
    def test_lists_valid_entries_and_skips_an_invalid_file(self, server):
        body = server.get("/api/loras").json()
        assert [row["name"] for row in body["loras"]] == [
            "minimax-h3/realism",
            "qwen-image/voxel",
        ]
        assert body["loras"][0]["origin"] == "examples"
        assert body["loras"][0]["writable"] is False

    def test_a_repo_id_filters_exactly(self, server):
        body = server.get("/api/loras", params={"model": QWEN}).json()
        assert [row["name"] for row in body["loras"]] == ["qwen-image/voxel"]
        assert body["resolved"] == [{"repo": QWEN, "workflow": None}]
        assert (
            server.get(
                "/api/loras", params={"model": "Qwen/Qwen-Image-2.1-2509"}
            ).json()["loras"]
            == []
        )

    def test_a_workflow_name_resolves_to_its_bases(self, server):
        body = server.get("/api/loras", params={"model": "models/qwen"}).json()
        assert [row["name"] for row in body["loras"]] == ["qwen-image/voxel"]

    def test_a_reference_workflow_does_not_list_a_t2va_entry(self, server):
        body = server.get("/api/loras", params={"model": "models/h3-ref"}).json()
        assert body["loras"] == []
        assert body["resolved"] == [
            {"repo": "MiniMaxAI/MiniMax-H3", "workflow": "ref2va"}
        ]

    def test_the_workflow_parameter_narrows_a_repo(self, server):
        params = {"model": "MiniMaxAI/MiniMax-H3", "workflow": "ref2va"}
        assert server.get("/api/loras", params=params).json()["loras"] == []

    def test_status_and_tag_filter(self, server):
        assert [
            r["name"]
            for r in server.get("/api/loras", params={"status": "proven"}).json()[
                "loras"
            ]
        ] == ["minimax-h3/realism"]

    def test_neither_a_workflow_nor_a_repo_is_a_400(self, server):
        response = server.get("/api/loras", params={"model": "no-such-thing"})
        assert response.status_code == 400
        assert "no-such-thing" in response.json()["detail"]


class TestWrites:
    def test_save_lands_in_the_roots_library_and_shadows_the_shipped_one(self, server):
        response = server.put(
            "/api/loras/qwen-image/voxel", json={"entry": entry(scale={"default": 0.8})}
        )
        assert response.status_code == 200
        saved = os.path.join(server.workspace.root, "loras", "qwen-image", "voxel.json")
        assert os.path.isfile(saved)
        got = server.get("/api/loras/qwen-image/voxel")
        assert got.json()["scale"] == {"default": 0.8}
        assert got.headers["X-Lora-Origin"] == "workspace"

    def test_an_invalid_entry_is_a_400(self, server):
        response = server.put("/api/loras/x", json={"entry": entry(status="proven")})
        assert response.status_code == 400

    def test_a_traversing_name_is_a_400(self, server):
        # the client may normalise the '..' away, which lands on no route at all
        assert server.put("/api/loras/../x", json={"entry": entry()}).status_code in (
            400,
            404,
            405,
        )

    def test_the_name_recommend_is_refused(self, server):
        response = server.put("/api/loras/recommend", json={"entry": entry()})
        assert response.status_code == 400

    def test_a_named_workspace_sees_the_same_library(self, server):
        server.put("/api/loras/mine", json={"entry": entry()})
        create_workspace(server.workspace, "other")
        names = [
            r["name"]
            for r in server.get("/api/loras", params={"workspace": "other"}).json()[
                "loras"
            ]
        ]
        assert "mine" in names

    def test_deleting_a_shipped_entry_is_a_403(self, server):
        assert server.delete("/api/loras/qwen-image/voxel").status_code == 403

    def test_deleting_an_own_entry(self, server):
        server.put("/api/loras/mine", json={"entry": entry()})
        assert server.delete("/api/loras/mine").json() == {
            "name": "mine",
            "deleted": True,
        }
        assert server.get("/api/loras/mine").status_code == 404

    def test_the_schema_has_its_own_route(self, server):
        assert server.get("/api/lora-schema").json()["title"] == "LoRA catalog entry"
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_server_loras.py -v -p no:xdist`
Expected: FAIL (404s - no routes).

- [ ] **Step 3: Implement**

`dw/server/app.py`, in `_store_directories` after `state.workspace = ...`:

```python
    # The LoRA catalog this server writes to: the root's loras/, shared by
    # every workspace like the prompt library. None when there is no root (a
    # server configured from loose directories) - the shipped catalog is
    # still read, and a save answers 409
    state.lora_dir = (
        os.path.join(state.workspace, LORAS_SUBDIR) if state.workspace else None
    )
```

(import `LORAS_SUBDIR` from `..workspace`).

`dw/server/deps.py`:

```python
def lora_library_missing():
    """The 409 a LoRA save answers when the server has no LoRA library."""
    return HTTPException(
        status_code=409,
        detail="This server has no LoRA library - it was configured from "
        "loose directories rather than a workspace root",
    )


def server_lora_library(state):
    """The LoRA catalog's search path: the root's loras/ (writable), then
    each examples tree's loras/ (read-only). Shared by every workspace."""
    return library_path(
        LORAS_KIND,
        state.default_workspace,
        state.examples_dirs,
        primary=getattr(state, "lora_dir", None),
    )
```

`dw/server/routes/loras.py`:

```python
"""The LoRA catalog routes (`dw/lora_catalog.py` holds the rules).

`GET /api/loras/recommend` is registered before the greedy
`{name:path}` routes, and `recommend` is a reserved entry name, so the two
cannot shadow each other.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...library import ReadOnlyLibraryError, shadowed_listing, workflow_names
from ...lora_catalog import (
    RESERVED_NAMES,
    SCHEMA_NAME,
    entry_errors,
    is_repo_id,
    matches,
    ranked,
)
from ...schema import load_schema
from ...security import (
    InvalidInputError,
    SecurityError,
    validate_lora_name,
    validate_path,
)
from ...workspace import Workspace, forget_workspace_usage
from ..deps import (
    lora_library_missing,
    selected_workspace,
    server_lora_library,
    sources_for,
)
from ...lora_catalog import workflow_bases

logger = logging.getLogger("dw")
router = APIRouter()


class LoraRequest(BaseModel):
    entry: Dict[str, Any] = Field(description="The catalog entry to save")


def _named(name):
    """A request's entry name, validated; 404 for a name no entry can have."""
    bare = name.removesuffix(".json")
    try:
        return validate_lora_name(bare)
    except InvalidInputError as error:
        raise HTTPException(status_code=404, detail=str(error))


def catalog_entries(state):
    """(entries, roots, library): every valid entry on the search path by
    name, the root each came from, and the path. An unreadable or invalid
    file is logged and skipped - one bad file must not hide the catalog."""
    library = server_lora_library(state).existing()
    winners, _hidden = library.entries()
    entries, roots = {}, {}
    for name, root in winners.items():
        path = library.path_in(root, name)
        if path is None:
            continue
        try:
            with open(path, "r") as file:
                entry = json.load(file)
        except (OSError, ValueError) as error:
            logger.warning(f"Skipping unreadable LoRA entry {name}: {error}")
            continue
        problem = entry_errors(entry) if isinstance(entry, dict) else "not an object"
        if problem:
            logger.warning(f"Skipping invalid LoRA entry {name}: {problem}")
            continue
        entries[name] = entry
        roots[name] = root
    return entries, roots, library


def resolve_model(state, ws, model, partition=None):
    """`model` as `(repo, partition)` pairs: a catalog workflow's bases, or
    a Hub repo id taken as given. A workflow is tried first - a workflow
    name like `models/qwen` has a repo id's shape too."""
    found = sources_for(state, ws).find(model.removesuffix(".json"))
    if found is not None:
        try:
            with open(found[0], "r") as file:
                bases = workflow_bases(json.load(file))
        except (OSError, ValueError) as error:
            raise HTTPException(
                status_code=400, detail=f"Workflow '{model}' cannot be read: {error}"
            )
    elif is_repo_id(model):
        bases = [(model, None)]
    else:
        raise HTTPException(
            status_code=400,
            detail=f"'{model}' is neither a workflow on this server nor a Hub repo id (owner/name)",
        )
    if partition:
        bases = [(repo, partition) for repo, _ in bases]
    return bases


def described_bases(bases):
    return [{"repo": repo, "workflow": partition} for repo, partition in bases]


@router.get("/api/lora-schema")
def get_lora_schema():
    """The JSON schema a catalog entry must satisfy - its own path, so an
    entry named 'schema' cannot shadow it."""
    return JSONResponse(load_schema(SCHEMA_NAME))


@router.get("/api/loras")
def list_loras(
    request: Request,
    model: Optional[str] = None,
    workflow: Optional[str] = None,
    status: Optional[str] = None,
    tag: Optional[str] = None,
    ws: Workspace = Depends(selected_workspace),
):
    state = request.app.state
    entries, roots, library = catalog_entries(state)
    body = {"libraries": library.describe()}
    if model:
        bases = resolve_model(state, ws, model, workflow)
        entries = {n: e for n, e in entries.items() if matches(e, bases)}
        body["resolved"] = described_bases(bases)
    if status:
        entries = {n: e for n, e in entries.items() if e.get("status") == status}
    if tag:
        entries = {n: e for n, e in entries.items() if tag in e.get("tags", [])}
    body["loras"] = [
        {
            **row,
            "origin": roots[row["name"]].origin,
            "writable": roots[row["name"]].writable,
        }
        for row in sorted(ranked(entries, []), key=lambda row: row["name"])
    ]
    return body


# GET /api/loras/recommend is added here by Task 6, before the greedy routes


@router.put("/api/loras/{name:path}")
def save_lora(http_request: Request, name: str, request: LoraRequest):
    """Write an entry into the server's LoRA library, schema-checked first.
    Saving over a shipped entry writes a copy that shadows it."""
    state = http_request.app.state
    bare = name.removesuffix(".json")
    try:
        bare = validate_lora_name(bare)
    except InvalidInputError as error:
        raise HTTPException(status_code=400, detail=str(error))
    if bare in RESERVED_NAMES:
        raise HTTPException(
            status_code=400, detail=f"'{bare}' is reserved by the LoRA routes"
        )
    problem = entry_errors(request.entry)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    root = server_lora_library(state).writable_root()
    if root is None:
        raise lora_library_missing()
    try:
        path = validate_path(
            os.path.join(root.root, f"{bare}.json"), root.root, allow_create=True
        )
    except SecurityError as error:
        raise HTTPException(status_code=400, detail=str(error))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(request.entry, file, indent=2)
        file.write("\n")
    logger.info(f"Saved LoRA entry {bare} to {path}")
    return {"name": bare}


@router.delete("/api/loras/{name:path}")
def delete_lora(request: Request, name: str):
    """Remove one of the server's own entries; a shipped one is a 403."""
    state = request.app.state
    library = server_lora_library(state)
    found = library.find(_named(name))
    if found is None:
        raise HTTPException(status_code=404, detail=f"Unknown LoRA entry: {name}")
    path, root = found
    try:
        library.require_writable(root, name)
    except ReadOnlyLibraryError as refusal:
        raise HTTPException(status_code=403, detail=str(refusal))
    os.remove(path)
    forget_workspace_usage()
    return {"name": name.removesuffix(".json"), "deleted": True}


@router.get("/api/loras/{name:path}")
def get_lora(request: Request, name: str):
    found = server_lora_library(request.app.state).find(_named(name))
    if found is None:
        raise HTTPException(status_code=404, detail=f"Unknown LoRA entry: {name}")
    path, root = found
    with open(path, "r") as file:
        return JSONResponse(
            json.load(file),
            headers={
                "X-Lora-Origin": root.origin,
                "X-Lora-Writable": "true" if root.writable else "false",
            },
        )
```

Tidy the imports (merge the two `lora_catalog` imports, drop unused `shadowed_listing`/`workflow_names` if the linter flags them). In `dw/server/routes/__init__.py` import `loras` and put `loras.router` after `library.router` in `ROUTERS`.

- [ ] **Step 4: Run to verify they pass, plus the route-table tests**

Run: `venv/bin/python -m pytest tests/test_server_loras.py tests/test_api_contract.py tests/test_security_auth.py tests/test_server.py -q`
Expected: PASS. If `tests/test_api_contract.py` pins the OpenAPI route list (`constraints-openapi.txt`), add the new routes there - that file is the pin, not a bug.

- [ ] **Step 5: Commit**

```bash
git add dw/server/routes/loras.py dw/server/routes/__init__.py dw/server/app.py dw/server/deps.py tests/test_server_loras.py
git commit -m "feat(server): LoRA catalog routes - list by base or workflow, save, delete

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Hub search and inspection (`dw/lora_hub.py`)

**Files:**
- Create: `dw/lora_hub.py`
- Test: `tests/test_lora_hub.py`

**Interfaces:**
- Consumes: `is_repo_id` (Task 2).
- Produces (in `dw.lora_hub`):
  - `HUB_TIMEOUT = 20.0`, `HEADER_TIMEOUT = 5.0`, `SEARCH_EXPAND: list[str]`
  - `classify_format(keys) -> str` - `"diffusers" | "kohya" | "full_weight" | "unknown"`
  - `hub_candidates(bases: list[str], query: str, terms: list[str], limit: int, rejected: dict[str, str], api) -> list[dict]`
  - `search_hub(bases, query, terms, limit, rejected, api=None, timeout=HUB_TIMEOUT) -> tuple[list[dict], str | None]` - `(results, hub_error)`; never raises

- [ ] **Step 1: Write the failing tests**

```python
"""dw/lora_hub.py against a fake HfApi: exact-base search, safetensors-only,
format classification from the header, warnings rather than crashes."""

import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from dw.lora_hub import classify_format, hub_candidates, search_hub

OLD = datetime(2025, 1, 1, tzinfo=timezone.utc)
NEW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def sibling(name):
    return SimpleNamespace(rfilename=name)


def info(
    repo,
    files=("lora.safetensors",),
    downloads=10,
    card=None,
    gated=False,
    modified=NEW,
):
    return SimpleNamespace(
        id=repo,
        downloads=downloads,
        likes=1,
        last_modified=modified,
        gated=gated,
        sha=f"{repo}-sha",
        siblings=[sibling(f) for f in files],
        card_data=card,
    )


class FakeApi:
    def __init__(self, listings, headers=None, base_modified=OLD, fail_header=()):
        self.listings = listings  # {(filter, search): [info]}
        self.headers = headers or {}
        self.base_modified = base_modified
        self.fail_header = set(fail_header)
        self.searches = []

    def list_models(self, *, filter, search=None, sort=None, limit=None, expand=None):
        self.searches.append((filter, search))
        return list(self.listings.get((filter, search), []))

    def model_info(self, repo):
        return SimpleNamespace(last_modified=self.base_modified)

    def parse_safetensors_file_metadata(
        self, repo, filename, *, revision=None, timeout=None
    ):
        if repo in self.fail_header:
            raise RuntimeError("range read refused")
        return SimpleNamespace(
            tensors=dict.fromkeys(self.headers.get(repo, ["x.lora_A.weight"]))
        )


F = "base_model:adapter:Qwen/Qwen-Image-2.1"


class TestFormat:
    def test_diffusers_peft_keys(self):
        assert (
            classify_format(["transformer.blocks.0.attn.to_q.lora_A.weight"])
            == "diffusers"
        )

    def test_kohya_keys(self):
        assert classify_format(["lora_unet_blocks_0_attn.lora_down.weight"]) == "kohya"

    def test_full_weight_keys_win(self):
        assert (
            classify_format(["a.lora_A.weight", "b.diff", "c.diff_b"]) == "full_weight"
        )

    def test_anything_else(self):
        assert classify_format(["model.weight"]) == "unknown"


class TestCandidates:
    def test_searches_the_exact_base_per_word_and_merges(self):
        api = FakeApi(
            {
                (F, "voxel style"): [info("a/voxel")],
                (F, "voxel"): [info("a/voxel"), info("b/voxel2", downloads=99)],
            }
        )
        results = hub_candidates(
            ["Qwen/Qwen-Image-2.1"], "voxel style", ["voxel"], 8, {}, api
        )
        assert (F, "voxel style") in api.searches and (F, "voxel") in api.searches
        assert [r["model_name"] for r in results] == ["b/voxel2", "a/voxel"]

    def test_a_stop_word_query_searches_unfiltered(self):
        api = FakeApi({(F, None): [info("a/x")]})
        results = hub_candidates(["Qwen/Qwen-Image-2.1"], "style", [], 8, {}, api)
        assert api.searches == [(F, None)]
        assert results[0]["model_name"] == "a/x"

    def test_a_pickle_only_repo_is_dropped(self):
        api = FakeApi({(F, None): [info("a/bin", files=("lora.bin",))]})
        assert hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api) == []

    def test_a_candidate_carries_a_step_ready_lora(self):
        api = FakeApi({(F, None): [info("a/x")]})
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert result["source"] == "hub" and result["status"] == "candidate"
        assert result["as_lora"] == {
            "model_name": "a/x",
            "weight_name": "lora.safetensors",
            "revision": "a/x-sha",
            "scale": 1.0,
        }
        assert result["format"] == "diffusers"

    def test_multiple_weights_leave_weight_name_unset(self):
        api = FakeApi(
            {(F, None): [info("a/x", files=("one.safetensors", "two.safetensors"))]}
        )
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert "as_lora" not in result
        assert result["weights"] == ["one.safetensors", "two.safetensors"]
        assert "multiple_weights" in result["warnings"]

    def test_full_weight_keys_warn_will_not_load(self):
        api = FakeApi({(F, None): [info("a/x")]}, headers={"a/x": ["b.diff"]})
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert "will_not_load" in result["warnings"]

    def test_no_card_is_warnings_not_a_crash(self):
        api = FakeApi({(F, None): [info("a/x", card=None, modified=None)]})
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert "no_license" in result["warnings"]
        assert result.get("trigger") is None

    def test_a_list_instance_prompt_takes_its_first(self):
        card = {"license": "apache-2.0", "instance_prompt": ["Voxel Style", "other"]}
        api = FakeApi({(F, None): [info("a/x", card=card)]})
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert result["trigger"] == "Voxel Style"
        assert result["license"] == "apache-2.0"

    def test_older_than_its_base_is_stale(self):
        api = FakeApi({(F, None): [info("a/x", modified=OLD)]}, base_modified=NEW)
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert "stale" in result["warnings"]

    def test_gated_is_flagged(self):
        api = FakeApi({(F, None): [info("a/x", gated="manual")]})
        assert (
            "gated"
            in hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)[0][
                "warnings"
            ]
        )

    def test_an_unreadable_header_is_a_warning(self):
        api = FakeApi({(F, None): [info("a/x")]}, fail_header=["a/x"])
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert "header_unreadable" in result["warnings"]

    def test_a_catalog_rejected_repo_comes_back_rejected(self):
        api = FakeApi({(F, None): [info("drozbay/FastH3")]})
        [result] = hub_candidates(
            ["Qwen/Qwen-Image-2.1"], "", [], 8, {"drozbay/FastH3": ".diff keys"}, api
        )
        assert result == {
            "source": "hub",
            "status": "rejected",
            "model_name": "drozbay/FastH3",
            "reason": ".diff keys",
        }

    def test_a_malformed_repo_id_from_the_hub_is_ignored(self):
        api = FakeApi({(F, None): [info("../evil")]})
        assert hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api) == []

    def test_limit_caps_the_candidates(self):
        api = FakeApi({(F, None): [info(f"a/x{i}", downloads=i) for i in range(5)]})
        assert len(hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 2, {}, api)) == 2


class TestFailure:
    def test_a_hub_error_is_returned_not_raised(self):
        class Down(FakeApi):
            def list_models(self, **kwargs):
                raise ConnectionError("hub unreachable")

        results, error = search_hub(
            ["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api=Down({})
        )
        assert results == [] and "hub unreachable" in error

    def test_a_slow_hub_times_out(self):
        class Slow(FakeApi):
            def list_models(self, **kwargs):
                time.sleep(2)
                return []

        results, error = search_hub(
            ["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api=Slow({}), timeout=0.2
        )
        assert results == [] and "timed out" in error
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_lora_hub.py -v -p no:xdist`
Expected: FAIL at import.

- [ ] **Step 3: Write `dw/lora_hub.py`**

```python
"""LoRA candidates from the Hugging Face Hub - the one place dw searches the
Hub on its own, and only when `recommend_loras` asks.

Exact base: the search is `list_models(filter="base_model:adapter:<repo>")`,
so an adapter declared for another revision of a family never appears.
Nothing is downloaded: the file list and card come with the listing
(`expand=`), and the weight file's safetensors header is a range read, which
is enough to tell a `load_lora_weights` layout from a full-weight diff that
diffusers will refuse (the FastH3 failure, #585). A repo holding only pickle
`.bin` weights is dropped - dw does not offer to load pickles.

What comes back is a candidate to trial, never a recommendation: download
counts are the only quality signal the Hub has.
"""

import concurrent.futures
import logging

from .lora_catalog import is_repo_id

logger = logging.getLogger("dw")

HUB_TIMEOUT = 20.0
HEADER_TIMEOUT = 5.0
SEARCH_EXPAND = [
    "downloads",
    "likes",
    "lastModified",
    "gated",
    "cardData",
    "sha",
    "siblings",
]
SAFETENSORS = ".safetensors"
KOHYA_PREFIXES = ("lora_unet_", "lora_te")
FULL_WEIGHT_SUFFIXES = (".diff", ".diff_b")
LORA_MARKERS = ("lora_A", "lora_B", "lora_down", "lora_up")


def classify_format(keys):
    """Which layout a LoRA's tensor names are in. A full-weight diff is
    checked first: a file carrying any is refused by diffusers' converters
    whatever else it holds."""
    keys = list(keys)
    if any(key.endswith(FULL_WEIGHT_SUFFIXES) for key in keys):
        return "full_weight"
    if any(key.startswith(KOHYA_PREFIXES) for key in keys):
        return "kohya"
    if any(marker in key for key in keys for marker in LORA_MARKERS):
        return "diffusers"
    return "unknown"


def _card_value(card, key):
    if card is None:
        return None
    value = card.get(key) if hasattr(card, "get") else getattr(card, key, None)
    if isinstance(value, list):
        value = value[0] if value else None
    return value


def _searches(query, terms):
    """The search strings to run: the request as typed, then each term.
    No terms (a stop-word or empty query) is one unfiltered search."""
    if not terms:
        return [None]
    searches = []
    for search in [(query or "").strip()] + terms:
        if search and search not in searches:
            searches.append(search)
    return searches


def _inspect(api, info, base_modified):
    """One listing row as a candidate, or None for a repo with no
    safetensors weights."""
    weights = sorted(
        s.rfilename for s in info.siblings or [] if s.rfilename.endswith(SAFETENSORS)
    )
    if not weights:
        return None
    warnings = []
    weight = weights[0] if len(weights) == 1 else None
    if weight is None:
        warnings.append("multiple_weights")
    form = "unknown"
    if weight is not None:
        try:
            header = api.parse_safetensors_file_metadata(
                info.id, weight, revision=info.sha, timeout=HEADER_TIMEOUT
            )
            form = classify_format(header.tensors.keys())
        except Exception as error:  # one candidate's failure is its warning
            logger.info(f"LoRA header for {info.id} unreadable: {error}")
            warnings.append("header_unreadable")
    if form == "full_weight":
        warnings.append("will_not_load")
    elif form == "unknown" and "header_unreadable" not in warnings and weight:
        warnings.append("unknown_format")
    card = info.card_data
    license_id = _card_value(card, "license")
    if not license_id:
        warnings.append("no_license")
    if info.gated:
        warnings.append("gated")
    if base_modified and info.last_modified and info.last_modified < base_modified:
        warnings.append("stale")
    candidate = {
        "source": "hub",
        "status": "candidate",
        "model_name": info.id,
        "trigger": _card_value(card, "instance_prompt"),
        "license": license_id,
        "gated": bool(info.gated),
        "downloads": info.downloads or 0,
        "likes": getattr(info, "likes", 0) or 0,
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "format": form,
        "warnings": warnings,
    }
    if weight is None:
        candidate["weights"] = weights
    else:
        candidate["as_lora"] = {
            "model_name": info.id,
            "weight_name": weight,
            "revision": info.sha,
            "scale": 1.0,
        }
    return candidate


def hub_candidates(bases, query, terms, limit, rejected, api):
    """Up to `limit` candidates for `bases`, most downloaded first, plus a
    `rejected` row for each catalog-rejected repo the search turned up."""
    found, base_of = {}, {}
    for base in bases:
        for search in _searches(query, terms):
            for info in api.list_models(
                filter=f"base_model:adapter:{base}",
                search=search,
                sort="downloads",
                limit=limit * 3,
                expand=SEARCH_EXPAND,
            ):
                if not is_repo_id(info.id) or info.id in found:
                    continue
                found[info.id] = info
                base_of[info.id] = base
    base_modified = {}
    for base in bases:
        try:
            base_modified[base] = api.model_info(base).last_modified
        except Exception as error:
            logger.info(f"Base {base} last-modified unreadable: {error}")
    results, offered = [], 0
    for info in sorted(found.values(), key=lambda i: -(i.downloads or 0)):
        if offered >= limit:
            break
        if info.id in rejected:
            results.append(
                {
                    "source": "hub",
                    "status": "rejected",
                    "model_name": info.id,
                    "reason": rejected[info.id],
                }
            )
            continue
        candidate = _inspect(api, info, base_modified.get(base_of[info.id]))
        if candidate is not None:
            results.append(candidate)
            offered += 1
    return results


def search_hub(bases, query, terms, limit, rejected, api=None, timeout=HUB_TIMEOUT):
    """`(results, hub_error)`. Never raises: an unreachable, rate-limited or
    slow Hub is reported, and the caller still answers with the catalog."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(hub_candidates, bases, query, terms, limit, rejected, api)
    try:
        return future.result(timeout=timeout), None
    except concurrent.futures.TimeoutError:
        return [], f"Hub search timed out after {timeout:g} s"
    except Exception as error:
        return [], f"Hub search failed: {type(error).__name__}: {error}"
    finally:
        # Not waiting: a timed-out search finishes in the background
        pool.shutdown(wait=False)
```

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/python -m pytest tests/test_lora_hub.py -v -p no:xdist`
Expected: PASS.

- [ ] **Step 5: Smoke against the real Hub once (manual, not a test)**

```bash
venv/bin/python -c "
from dw.lora_hub import search_hub
from dw.lora_catalog import query_terms
r, e = search_hub(['Qwen/Qwen-Image-2.1'], 'voxel style', query_terms('voxel style'), 5, {})
print(e); [print(c['model_name'], c.get('format'), c['warnings']) for c in r]"
```

Expected: no error and a handful of rows; record what came back in the commit body. If the real `ModelInfo.card_data` is a `ModelCardData` without `.get`, `_card_value`'s `getattr` branch covers it - confirm `license` is populated for at least one row.

- [ ] **Step 6: Commit**

```bash
git add dw/lora_hub.py tests/test_lora_hub.py
git commit -m "feat(loras): Hub candidate search with header-based format check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The recommend route

**Files:**
- Modify: `dw/server/routes/loras.py` (add the route where Task 4 left the marker comment)
- Test: `tests/test_server_loras.py` (append)

**Interfaces:**
- Consumes: Task 4 (`catalog_entries`, `resolve_model`, `described_bases`), Task 2 (`query_terms`, `ranked`, `rejection_reasons`, `matches`), Task 5 (`search_hub`).
- Produces: `GET /api/loras/recommend?model=&query=&limit=&workspace=` -> `{"resolved": [...], "catalog": [row...], "hub": [candidate...], "note": str, "hub_error"?: str}`; `dw.server.routes.loras.TRIAL_NOTE`.

- [ ] **Step 1: Write the failing tests** (append; reuses the `server` fixture)

```python
from dw.server.routes import loras as lora_routes


@pytest.fixture
def hub(monkeypatch):
    calls = {}

    def fake(bases, query, terms, limit, rejected, api=None, timeout=None):
        calls.update(
            bases=bases, query=query, terms=terms, limit=limit, rejected=rejected
        )
        return calls.get("results", []), calls.get("error")

    monkeypatch.setattr(lora_routes, "search_hub", fake)
    return calls


class TestRecommend:
    def test_catalog_first_then_hub(self, server, hub):
        hub["results"] = [{"source": "hub", "status": "candidate", "model_name": "x/y"}]
        body = server.get(
            "/api/loras/recommend",
            params={"model": "models/qwen", "query": "voxel style"},
        ).json()
        assert [row["name"] for row in body["catalog"]] == ["qwen-image/voxel"]
        assert body["catalog"][0]["source"] == "catalog"
        assert body["hub"] == hub["results"]
        assert hub["bases"] == ["Qwen/Qwen-Image-2.1"]
        assert hub["terms"] == ["voxel"]
        assert "trial" in body["note"]

    def test_rejected_entries_are_not_offered_but_suppress_the_hub(self, server, hub):
        server.put(
            "/api/loras/qwen-image/bad",
            json={
                "entry": entry(
                    model_name="bad/one",
                    status="rejected",
                    evidence=[{"note": "noise"}],
                )
            },
        )
        body = server.get(
            "/api/loras/recommend", params={"model": QWEN, "query": ""}
        ).json()
        assert "qwen-image/bad" not in [row["name"] for row in body["catalog"]]
        assert hub["rejected"] == {"bad/one": "noise"}

    def test_a_hub_failure_still_returns_the_catalog(self, server, hub):
        hub["error"] = "Hub search timed out after 20 s"
        body = server.get(
            "/api/loras/recommend", params={"model": QWEN, "query": "voxel"}
        ).json()
        assert body["hub_error"] == "Hub search timed out after 20 s"
        assert body["catalog"]

    def test_recommend_is_not_read_as_an_entry_name(self, server, hub):
        response = server.get("/api/loras/recommend", params={"model": QWEN})
        assert response.status_code == 200
        assert "catalog" in response.json()

    def test_limit_is_bounded(self, server, hub):
        assert (
            server.get(
                "/api/loras/recommend", params={"model": QWEN, "limit": 0}
            ).status_code
            == 422
        )
        assert (
            server.get(
                "/api/loras/recommend", params={"model": QWEN, "limit": 26}
            ).status_code
            == 422
        )

    def test_a_model_with_no_repo_bases_skips_the_hub(self, server, hub):
        write(
            str(server.checkout / "workflows"),
            "models/local",
            {
                "steps": [
                    {
                        "name": "s",
                        "pipeline": {
                            "from_pretrained_arguments": {"model_name": "./weights"}
                        },
                    }
                ]
            },
        )
        body = server.get(
            "/api/loras/recommend", params={"model": "models/local"}
        ).json()
        assert body["resolved"] == [] and body["hub"] == [] and "bases" not in hub
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_server_loras.py -v -p no:xdist -k Recommend`
Expected: FAIL (`AttributeError: module ... has no attribute 'search_hub'`).

- [ ] **Step 3: Implement** - in `dw/server/routes/loras.py` add `from fastapi import Query`, `from ...lora_catalog import query_terms, rejection_reasons`, `from ...lora_hub import search_hub`, and replace the marker comment with:

```python
TRIAL_NOTE = (
    "Catalog rows are LoRAs tried on this base. Hub rows are candidates to "
    "trial, ranked by downloads only - run one beside a no-LoRA render at the "
    "same seed, and save_lora a trial that works (status proven, its job in "
    "evidence)."
)


@router.get("/api/loras/recommend")
def recommend_loras(
    request: Request,
    model: str,
    query: str = "",
    limit: int = Query(8, ge=1, le=25),
    ws: Workspace = Depends(selected_workspace),
):
    """Catalog entries for `model` ranked against `query`, then Hub
    candidates for the same exact bases. The Hub is searched only here."""
    state = request.app.state
    bases = resolve_model(state, ws, model)
    entries, _roots, _library = catalog_entries(state)
    fitting = {n: e for n, e in entries.items() if matches(e, bases)}
    terms = query_terms(query)
    catalog = [
        {"source": "catalog", **row}
        for row in ranked(fitting, terms)
        if row.get("status") != "rejected"
    ]
    body = {
        "resolved": described_bases(bases),
        "catalog": catalog,
        "hub": [],
        "note": TRIAL_NOTE,
    }
    repos = sorted({repo for repo, _ in bases})
    if repos:
        hub, hub_error = search_hub(
            repos, query, terms, limit, rejection_reasons(fitting)
        )
        body["hub"] = hub
        if hub_error:
            body["hub_error"] = hub_error
    return body
```

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/python -m pytest tests/test_server_loras.py -v -p no:xdist`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/server/routes/loras.py tests/test_server_loras.py
git commit -m "feat(server): GET /api/loras/recommend - catalog first, then Hub candidates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: MCP tools

**Files:**
- Create: `dw_mcp/loras.py`
- Modify: `dw_mcp/tools_authoring.py` (add `LoraTools` after `PromptTools`)
- Modify: `dw_mcp/server.py` (import, `READ_ONLY_OPEN` annotation, registration after the prompt tools)
- Modify: `tests/test_mcp_server.py` (`EXPECTED_TOOLS`, `READ_ONLY_TOOLS` exclusions, `SURFACE_BUDGET` + comment)
- Test: `tests/test_mcp_loras.py`

**Interfaces:**
- Consumes: the HTTP routes of Tasks 4 and 6; `dw_mcp.client.api_path`, `coerce_json_object`.
- Produces: MCP tools `list_loras(model?, workflow?, status?, tag?)`, `save_lora(name, entry)`, `recommend_loras(model, query, limit=8)`.

- [ ] **Step 1: Write the failing tests** - `tests/test_mcp_loras.py`:

```python
"""LoRA catalog tools: what each sends to the server."""

import json

import httpx

from dw_mcp import loras
from dw_mcp.client import DwClient


def recording(body=None):
    seen = []

    def handler(request):
        raw = request.read()
        seen.append(
            (
                request.method,
                request.url.path,
                dict(request.url.params),
                json.loads(raw) if raw else None,
            )
        )
        return httpx.Response(200, json=body if body is not None else {})

    return DwClient(transport=httpx.MockTransport(handler)), seen


def test_list_loras_sends_only_the_filters_given():
    client, seen = recording({"loras": []})
    loras.list_loras(client, model="models/qwen-image-2.1")
    assert seen == [("GET", "/api/loras", {"model": "models/qwen-image-2.1"}, None)]


def test_save_lora_puts_the_entry():
    client, seen = recording({"name": "qwen-image/voxel"})
    loras.save_lora(client, "qwen-image/voxel", {"model_name": "a/b"})
    assert seen[0][:2] == ("PUT", "/api/loras/qwen-image/voxel")
    assert seen[0][3] == {"entry": {"model_name": "a/b"}}


def test_save_lora_accepts_a_json_string():
    client, seen = recording({"name": "x"})
    loras.save_lora(client, "x", '{"model_name": "a/b"}')
    assert seen[0][3] == {"entry": {"model_name": "a/b"}}


def test_recommend_loras_sends_model_query_and_limit():
    client, seen = recording({"catalog": [], "hub": []})
    loras.recommend_loras(client, "Qwen/Qwen-Image-2.1", "voxel style", limit=5)
    assert seen == [
        (
            "GET",
            "/api/loras/recommend",
            {"model": "Qwen/Qwen-Image-2.1", "query": "voxel style", "limit": "5"},
            None,
        )
    ]
```

and in `tests/test_mcp_server.py` add `"list_loras", "save_lora", "recommend_loras"` to `EXPECTED_TOOLS`, add `"save_lora"` to the set subtracted for `READ_ONLY_TOOLS`, plus:

```python
@pytest.mark.asyncio
async def test_recommend_loras_says_it_reaches_the_hub():
    tools = await tools_of(server_over(ok({})))
    assert tools["recommend_loras"].annotations.open_world_hint is True
    assert tools["list_loras"].annotations.open_world_hint is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_mcp_loras.py tests/test_mcp_server.py -q -p no:xdist`
Expected: FAIL (import error; missing tools).

- [ ] **Step 3: Implement**

`dw_mcp/loras.py`:

```python
"""The LoRA catalog over the HTTP API: list by base or workflow, save a
promoted trial, and the opt-in Hub search."""

from dw_mcp.client import api_path, coerce_json_object


def list_loras(client, model=None, workflow=None, status=None, tag=None):
    params = {
        key: value
        for key, value in (
            ("model", model),
            ("workflow", workflow),
            ("status", status),
            ("tag", tag),
        )
        if value is not None
    }
    return client.get_json("/api/loras", params=params)


def save_lora(client, name, entry):
    entry = coerce_json_object(entry, "entry")
    return client.put_json(api_path("api", "loras", name), {"entry": entry})


def recommend_loras(client, model, query, limit=8):
    return client.get_json(
        "/api/loras/recommend", params={"model": model, "query": query, "limit": limit}
    )
```

`dw_mcp/tools_authoring.py`, after `PromptTools` (import `from dw_mcp import loras` beside the `prompts` import):

```python
class LoraTools:
    """The LoRA catalog and the opt-in Hub search."""

    def __init__(self, client):
        self.client = client

    def list_loras(
        self,
        model: Optional[str] = None,
        workflow: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> dict:
        """LoRAs tried on a base model - proven, trial, or rejected with the
        reason. `model` is a workflow name or a Hub repo id; matching is
        exact on the base (and the H3 partition). `use_when` says when to
        reach for one; `trigger` and `scale` say how. Guide: loras."""
        return loras.list_loras(
            self.client, model=model, workflow=workflow, status=status, tag=tag
        )

    def save_lora(self, name: str, entry: dict | str) -> dict:
        """Save a catalog entry, e.g. promote a trial that worked to
        `proven` with its job in `evidence`. get_guide("loras") has the
        entry format."""
        return loras.save_lora(self.client, name, entry)

    def recommend_loras(self, model: str, query: str, limit: int = 8) -> dict:
        """Opt-in: catalog LoRAs for `model` ranked against a style request,
        then Hugging Face Hub adapters of that exact base (queries the Hub;
        no download, no GPU). Hub rows are candidates to trial, not
        recommendations - mind each one's `warnings`."""
        return loras.recommend_loras(self.client, model, query, limit=limit)
```

`dw_mcp/server.py`: `READ_ONLY_OPEN = ToolAnnotations(read_only_hint=True, open_world_hint=True)` beside `READ_ONLY`; import `LoraTools`; in `build_server` add `lor = LoraTools(client)` and after `tool(prm.enhance_prompt, WRITES)`:

```python
    tool(lor.list_loras, READ_ONLY)
    tool(lor.save_lora, OVERWRITES)
    tool(lor.recommend_loras, READ_ONLY_OPEN)
```

If `list_guides`/`get_guide` index guides by a fixed name map (`dw/server/guides.py`) and `"loras"` is not a guide name there, change the two `Guide: loras` / `get_guide("loras")` pointers to the name that file uses for `docs/LORAS.md`.

- [ ] **Step 4: Measure the surface and raise the budget**

Run: `venv/bin/python -m pytest tests/test_mcp_server.py -q -p no:xdist -k budget`
The failure message prints the new total. Set `SURFACE_BUDGET` to `ceil(total) + 10` and append to the comment block above it:

```python
# 2026-10-03: the LoRA catalog (#<issue>) added list_loras, save_lora and
# recommend_loras - measured at <total> (<descriptions> / <schemas> /
# <instructions>); the ceiling moves to <new> with 10 of headroom.
```

- [ ] **Step 5: Run to verify they pass**

Run: `venv/bin/python -m pytest tests/test_mcp_loras.py tests/test_mcp_server.py tests/test_mcp_prompts.py -q`
Expected: PASS (including the no-indentation and description-length tests).

- [ ] **Step 6: Commit**

```bash
git add dw_mcp/loras.py dw_mcp/tools_authoring.py dw_mcp/server.py tests/test_mcp_loras.py tests/test_mcp_server.py
git commit -m "feat(mcp): list_loras, save_lora and the opt-in recommend_loras

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Docs, skills, architecture map, full suite

**Files:**
- Modify: `docs/LORAS.md` (append two sections), `docs/MCP.md` (tool table rows next to the prompt tools, ~line 293), `docs/ARCHITECTURE.md` (two rows), `docs/SECURITY.md` (the validator list), `docs/WORKSPACES.md` (the root's `loras/`)
- Modify: `plugins/dw/skills/minimax-h3/SKILL.md`, `plugins/dw/skills/ltx-2.5/SKILL.md` (one-line pointer each)
- Test: `tests/test_plugin_skills.py`, `tests/test_docs*.py` (existing), whole suite

**Interfaces:**
- Consumes: everything above. Produces: no code.

- [ ] **Step 1: `docs/LORAS.md`** - append:

````markdown
## LoRA catalog

The catalog records LoRAs that were tried on a base model: `proven`, `trial`,
or `rejected` with the reason. It is a library like the prompt library - the
server's own `loras/` at the workspace root (writable, shared by every
workspace), then the `loras/` an examples tree brings (read-only). One JSON
file per LoRA, under a family folder: `loras/qwen-image/voxel-style.json`.

```json
{
  "model_name": "fal/MiniMax-H3-Realism-People-LoRA",
  "weight_name": "h3-realism-people-t2v-i2v-r2v.safetensors",
  "revision": "<sha>",
  "base_models": ["MiniMaxAI/MiniMax-H3"],
  "workflow": "t2va",
  "description": "Realistic people and faces",
  "use_when": "People, crowds or faces should look photoreal",
  "scale": { "default": 0.7, "range": [0.7, 0.7] },
  "status": "proven",
  "evidence": [{ "issue": 585, "note": "Best arm of the 2026-10-03 eval" }]
}
```

`model_name`, `weight_name`, `revision` and `scale.default` drop straight into
a step's `loras` entry. `base_models` holds exact repo ids, and matching is
exact - an adapter on the wrong base usually loads and is quietly worse.
`workflow` constrains a MiniMax-H3 entry to one partition (`t2va`, `fl2va`,
`ref2va`). `proven` and `rejected` entries need `evidence`; a rejected
entry's first note is why. The schema is `GET /api/lora-schema`.

Promotion: a trial that worked is saved with `save_lora` (or
`PUT /api/loras/{name}`) as `proven`, its job id in `evidence`.

## Finding LoRAs on the Hub

`recommend_loras(model, query)` (`GET /api/loras/recommend`) is the only
place dw searches the Hub, and only when called. It returns the catalog's
entries for the model first, ranked against the query, then Hub adapters
whose card declares that exact base (`base_model:adapter:<repo>`), most
downloaded first. Nothing is downloaded; the weight file's header is read to
check its layout. Hub rows are candidates to trial, never recommendations,
and carry `warnings`:

| Warning | Meaning |
| --- | --- |
| `will_not_load` | Full-weight `.diff` keys diffusers' converters refuse |
| `unknown_format` | The tensor names match no known LoRA layout |
| `header_unreadable` | The header range read failed; format unchecked |
| `multiple_weights` | Several `.safetensors`; pick one from `weights` |
| `gated` | Needs the server's HF token to have accepted the gate |
| `no_license` | The card declares none |
| `stale` | Last changed before its base was - trained on an older revision |

A repo holding only pickle `.bin` weights is never offered. A repo the
catalog marks `rejected` comes back as `rejected` with the reason. When the
Hub is unreachable the catalog rows still come back, with `hub_error`.
````

- [ ] **Step 2: `docs/MCP.md`** - after the prompt-tool rows add:

```markdown
| `list_loras(model=None, workflow=None, status=None, tag=None)` | optional `model` (workflow name or repo id), `workflow` (H3 partition), `status`, `tag` | Catalog LoRAs for a base, matched exactly; each row carries `origin` and `writable`, and `resolved` names the bases looked up |
| `save_lora(name, entry)` | `name`, `entry` | Save a catalog entry (a promoted trial); schema-checked, `recommend` is reserved |
| `recommend_loras(model, query, limit=8)` | `model`, `query`, optional `limit` (1-25) | Opt-in: catalog rows ranked against `query`, then Hub candidates for the exact base, with `warnings`; queries the Hub, downloads nothing |
```

- [ ] **Step 3: `docs/ARCHITECTURE.md`** - add two rows in the library area (after the *Library writes* row):

```markdown
| LoRA catalog matching | `dw/lora_catalog.py`: `matches`, `workflow_bases` | An entry fits a base only by exact repo id (and H3 partition); no alias or family match. | `tests/test_lora_catalog.py::TestMatching::test_the_base_must_match_exactly` |
| Hub search | `dw/lora_hub.py`: `search_hub` | The Hub is searched only by `GET /api/loras/recommend`, never raises, and offers no pickle-only repo. | `tests/test_lora_hub.py::TestCandidates::test_a_pickle_only_repo_is_dropped`, `tests/test_lora_hub.py::TestFailure::test_a_hub_error_is_returned_not_raised` |
```

- [ ] **Step 4: `docs/SECURITY.md` / `docs/WORKSPACES.md`** - add `validate_lora_path` and `validate_lora_name` wherever `validate_prompt_path` / `validate_prompt_reference` are listed; in `WORKSPACES.md` add `loras/` to the root's folders with "the LoRA catalog, shared by every workspace like `prompts/`; `loras` is a reserved workspace name".

- [ ] **Step 5: Skills** - in each of `plugins/dw/skills/minimax-h3/SKILL.md` and `plugins/dw/skills/ltx-2.5/SKILL.md`, add one line in the LoRA area: ``Tried LoRAs for this family (proven, rejected and why): `list_loras(model=<workflow>)`.`` In the H3 skill, replace an existing inline LoRA note of equal or greater length that the catalog now carries (the Realism/turbo-file notes, if present) so the file stays under its byte cap; keep the partition rule and step-count rules - they are not catalog facts.

Run: `venv/bin/python -m pytest tests/test_plugin_skills.py -q`
Expected: PASS (byte caps hold).

- [ ] **Step 6: Full suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: PASS. Investigate any failure in a test that enumerates library kinds, route tables or doc links - those pins are meant to be updated, not skipped.

- [ ] **Step 7: Commit**

```bash
git add docs plugins/dw/skills
git commit -m "docs(loras): catalog and Hub-candidate guide, MCP table, architecture rows, skill pointers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
