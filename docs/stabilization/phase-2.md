# Phase 2 Implementation Plan: seams in place

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each concept that is spread across the engine one owner: the reference prefixes, validation, the rules the checks and tasks share, the step cache's dependencies, and the worker protocol.

**Architecture:** Phase 2 runs as four stages, each merged to `develop` on its own. Each stage has its own hot zone, so the harness is kept out of only the files the current stage restructures. A stage's detailed tasks are written when the previous stage merges, on the code that stage left, the same rule the phases follow. Stage 2a is detailed below.

**Tech Stack:** Python 3.10+, pytest; metrics from `scripts/arch_metrics.py` and `scripts/arch_report.py`.

**Spec:** [ROADMAP.md](ROADMAP.md) (Phase 2 row: "`validation_errors` is a registry loop; no prefix literals outside `references.py`") and [ASSESSMENT.md](ASSESSMENT.md). Items carried from gate 1 are listed there under "Carried to Phase 2".

## Stages

| Stage | Scope | Done when |
| --- | --- | --- |
| 2a | `dw/references.py`: the prefixes, the three prefix sets, `is_ref` / `ref_name` / `make_ref`, and `author_index` (the source-index idiom copied in 20 checkers) | `prefix_literals` is 0 and ratcheted there; no private copy of a prefix or a prefix set remains |
| 2b | Validation plumbing: a `ValidationContext` (one expansion, one memoized metadata-only media probe that understands `{"location"}` dicts, which fixes B9), a `Finding(severity, kind, path, message)` type serialized only at the route, and a check registry with one exception policy (fixes B10). The ~25 checkers and 9 warning sources migrate one at a time, keeping their public names. The four rules written in both a check and a task (dissolve overlap, frame size, slice region, select rules) get one home each. | `validation_errors` is a loop over the registry; a crashing check is an internal-error finding |
| 2c | Typed worker protocol: message dataclasses, and one reply dispatcher on `WorkerManager`. The worker trusts an admitted snapshot: the job carries the admitted definition, and run-directory identity travels separately, so the worker stops re-reading the file and re-validating. | The worker has no `validate()` call; one dispatcher handles replies |
| 2d | Step cache: dependency tracking covers every cross-step reference. `pipeline_cache_key` moves into `step_cache` (the hidden `step_cache` → `pipeline` import); `Pipeline.load` copies its definition on entry; `_run_lock_path` becomes public. Fix import cycles where the stage's own moves allow. | `modules_in_import_cycles` below 26; B7-class keys come from one function |

Gate 2 follows 2d, with the same checks as gate 1:
- the full metrics report, with a Gate 2 column;
- a real-model timing on lem (the B2 cached rerun, plus one run exercising a `for_each` template);
- deploy, tag and re-baseline.

The smaller items carried from gate 1 go to the stage whose files they touch:
- the `recorded_variables` media copy → 2d (it is run-path code);
- `dw.run` paging a truncated event tail → 2c;
- expansion memo invalidation → dropped (the definition is not mutated after construction; 2b Task 3 pins that);
- the loose `JobManager` name fallbacks → 2c.

## Release notes collected (for gate 2)

- **2a:** reference prefixes are spelled in backticks instead of quotes in nine MCP tool descriptions and in `compose_text`'s `parts` description (served by `GET /api/tasks/compose_text` and MCP `get_task`). The wording is unchanged.
- **2a:** `modules` 132 → 133 (`dw/references.py`, accepted by Don 2026-09-28).
- **2a follow-ups:**
  - `dw/workflow.py:747` could use `author_index`.
  - Prefix tests are spelled three ways.
  - Four import styles for `references`.
  - Prefixes inside f-strings, which the metric does not count.

## Global Constraints (all stages)

- Hard freeze: no net-new features or functionality. Surface may change only where the consolidation requires it, and every change is listed for the gate's release notes.
- `scripts/arch_metrics.py --check docs/stabilization/baseline.json` passes at the end of every task.
- Every new test fails before its fix. No count-pinning tests. Add no string `patch("dw...")` targets.
- Behavior-preserving tasks prove preservation with the existing suite: it passes unchanged, apart from tests the task names.
- No module moves beyond what a task names (Phase 3 moves modules).
- Tests: `venv/bin/python -m pytest -q -x -p no:cacheprovider` from the worktree root, plus `ruff check` and `ruff format --check` on `dw dw_mcp tests`. The worktree's `venv` is shared with the main checkout, so never run `pip install -e .` from the worktree.

## Decisions (rulings, 2026-09-28)

- **Stages instead of one Phase 2 plan.**
  - Why: the five items share almost no files, and one hot zone covering all of them would lock the harness out of most of the engine for the whole phase.
  - Each stage merges on its own, re-baselines, and hands the next stage's hot zone to the harness.
- **`dw/references.py` raises `modules` by one** (132 → 133).
  - Why: it is the module the ROADMAP names for this. No existing module folds into it and keeps it a leaf: `reference_names.py` and `probe_paths.py` both import other `dw` modules.
  - Task 1 re-baselines `modules` to 133 in the same commit and records why.
  - This is a loosened limit. Nothing in Phase 2 is planned to pay it back: 2b keeps each checker's public name, and so its module. Expect 133 to stand until Phase 3's structural moves. Don accepts or rejects this before Task 1.
- **Three prefix sets, not one.** The nine prefix tuples in the code are three real sets. Collapsing them would silently weaken checks:
  - `SUBSTITUTED` = `variable:` and `item:`, used by 4 checkers that must still check `previous_result:` values;
  - `UNRESOLVED` = `SUBSTITUTED` + `previous_result:` + `gather:` (5 modules);
  - `DEFERRED` = `UNRESOLVED` + `asset:`, `output:`, `prompt:`, `constant:` and `builtin:` (`locations._deferred`).
  - The prompt library's `RESERVED_TEXT_PREFIXES` stays its own 6-member tuple, built from the constants.

---

## Stage 2a: `dw/references.py`

Work on branch `stabilization/phase-2a` in the worktree `/Users/don/src/dkackman/dw-stabilization`, from `develop`.

### Review Focus (2a)

1. **Membership drift.** A tuple replaced by the wrong set changes what a check skips. For example, a `SUBSTITUTED` site given `UNRESOLVED` would stop checking `previous_result:` values in `content_types`, `kernel_availability`, `result_fps` or `subfolders`, and every check would still pass. Task 2's test pins each migrated site's set.
2. **`removeprefix` vs `ref_name`.** `str.removeprefix` returns the value unchanged when the prefix is absent; `ref_name` returns None. Only replace a strip that is already guarded by the matching prefix test. Anything else keeps its exact semantics.
3. **Built references.** `f"asset:{name}"` in `dw/server/app.py` becomes `make_ref(ASSET, name)`, and the string must be byte-identical. The asset routes' tests cover this.
4. **Public names other modules import.** Aliases such as `ASSET_PREFIX` (`dw/assets.py`), `OUTPUT_PREFIX` (`dw/runs.py`), `PROMPT_PREFIX` (`dw/prompts.py`), `PREVIOUS_RESULT_PREFIX` / `CONSTANT_PREFIX` (`dw/arguments.py`), `ITEM_PREFIX` / `GATHER_PREFIX` (`dw/for_each.py`), `CONSTRAINT_PREFIX`, `SHOT_REFERENCE_PREFIX` and `probe_paths.UNRESOLVED_PREFIXES` must keep working for their importers. Remove one only when a grep shows no importer left.
5. **Import cycles.** `references.py` imports nothing from `dw`, and `import_cycles` / `modules_in_import_cycles` must not rise.

### Task 1: `dw/references.py` and the metric that owns it

**Files:**
- Create: `dw/references.py`, `tests/test_references.py`
- Modify: `scripts/arch_metrics.py` (`PREFIX_OWNERS`), `tests/test_arch_metrics.py`, `docs/stabilization/baseline.json` (`modules` 133)
- The stage 2a hot zone is already on `develop`, committed with this plan, so the harness saw it before the stage began.

**Interfaces:**
- Produces, from `dw.references`:
  - the constants `ASSET`, `OUTPUT`, `PROMPT`, `VARIABLE`, `PREVIOUS_RESULT`, `CONSTANT`, `ITEM`, `GATHER`, `BUILTIN` and `CONSTRAINT`;
  - the tuples `SUBSTITUTED`, `UNRESOLVED` and `DEFERRED`;
  - `is_ref(kind, value) -> bool` (`kind` is one prefix or a tuple);
  - `ref_name(kind, value) -> str | None`;
  - `make_ref(kind, name) -> str`;
  - `author_index(source_indices, index) -> int`.

- [ ] **Step 1: Write the failing tests** (`tests/test_references.py`, verified against the module below: 18 passed):

```python
"""dw/references.py - the one place a reference prefix is spelled."""

import ast
import pathlib

import pytest

from dw import references
from dw.references import (
    ASSET,
    DEFERRED,
    GATHER,
    ITEM,
    PREVIOUS_RESULT,
    SUBSTITUTED,
    UNRESOLVED,
    VARIABLE,
    author_index,
    is_ref,
    make_ref,
    ref_name,
)


@pytest.mark.parametrize(
    "kind, value, expected",
    [
        (ASSET, "asset:iris.png", True),
        (ASSET, "output:x/latest/a.png", False),
        (ASSET, "an asset: in prose", False),
        (ASSET, None, False),
        (ASSET, 3, False),
        (UNRESOLVED, "gather:shot", True),
        (UNRESOLVED, "asset:iris.png", False),
    ],
)
def test_is_ref_is_a_prefix_test_on_strings_only(kind, value, expected):
    assert is_ref(kind, value) is expected


def test_ref_name_strips_its_own_prefix_only():
    assert ref_name(VARIABLE, "variable:num_frames") == "num_frames"
    assert ref_name(VARIABLE, "item:len") is None
    assert ref_name(VARIABLE, 7) is None


@pytest.mark.parametrize("helper", [ref_name, make_ref])
def test_a_prefix_set_is_refused_where_one_prefix_is_needed(helper):
    with pytest.raises(TypeError):
        helper(UNRESOLVED, "variable:x")


def test_make_ref_round_trips_through_ref_name():
    assert ref_name(ASSET, make_ref(ASSET, "gyre/frames/web.mp4")) == (
        "gyre/frames/web.mp4"
    )


def test_the_three_sets_nest():
    # a check that skips DEFERRED values skips every UNRESOLVED one, and an
    # UNRESOLVED skip covers every SUBSTITUTED one
    assert set(SUBSTITUTED) < set(UNRESOLVED) < set(DEFERRED)
    assert set(UNRESOLVED) - set(SUBSTITUTED) == {PREVIOUS_RESULT, GATHER}
    assert ITEM in SUBSTITUTED


@pytest.mark.parametrize(
    "source_indices, index, expected",
    [([0, 0, 1], 1, 0), ([0, 0, 1], 2, 1), (None, 4, 4), ([], 2, 2), ([5], 3, 3)],
)
def test_author_index_maps_an_expanded_step_back_to_its_template(
    source_indices, index, expected
):
    assert author_index(source_indices, index) == expected


def test_references_imports_nothing_from_dw():
    tree = ast.parse(pathlib.Path(references.__file__).read_text())
    imported = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert imported == []
```

Append to `tests/test_arch_metrics.py`:

```python
def test_the_prefix_owner_may_spell_a_prefix(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {"dw/references.py": 'ASSET = "asset:"\n', "dw/a.py": 'X = "asset:"\n'},
        )
    )
    assert metrics["prefix_literals"] == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/test_references.py tests/test_arch_metrics.py`
Expected: `ModuleNotFoundError: No module named 'dw.references'`, and the arch-metrics test fails with `2 == 1`.

- [ ] **Step 3: Create `dw/references.py`** (verified: ruff clean, 18 tests pass):

```python
"""The workflow reference prefixes, spelled once.

A string value in a workflow definition that starts with one of these is a
reference, not a literal: something to substitute (`variable:`, `item:`),
an earlier step's result (`previous_result:`, `gather:`), a file or text to
fetch (`asset:`, `output:`, `prompt:`), a python value (`constant:`), a
packaged sub-workflow (`builtin:`) or a declared rule (`constraint:`).
Every module that tests for, strips or builds one does it through this
module, and no other module spells a prefix
(`scripts/arch_metrics.py` counts any that do).

This module imports nothing from dw, so every module can import it.
"""

ASSET = "asset:"
OUTPUT = "output:"
PROMPT = "prompt:"
VARIABLE = "variable:"
PREVIOUS_RESULT = "previous_result:"
CONSTANT = "constant:"
ITEM = "item:"
GATHER = "gather:"
BUILTIN = "builtin:"
CONSTRAINT = "constraint:"

# Still to be substituted: what the variable and for_each passes replace.
# A check that runs on the substituted definition skips a value still
# spelled this way - it is an entry field or a variable those passes left
SUBSTITUTED = (VARIABLE, ITEM)
# Not a literal until the run reaches the step: substitution plus the
# results of earlier steps
UNRESOLVED = (VARIABLE, ITEM, PREVIOUS_RESULT, GATHER)
# Anything a value may still hold before realization: the unresolved
# prefixes plus the ones realize_args fetches or looks up
DEFERRED = UNRESOLVED + (ASSET, OUTPUT, PROMPT, CONSTANT, BUILTIN)


def is_ref(kind, value):
    """Whether `value` is a reference of `kind` - one prefix, or a tuple of
    them (any of). False for anything that is not a string."""
    return isinstance(value, str) and value.startswith(kind)


def _single(kind):
    # A prefix set has no one name to strip or build with - is_ref is the
    # only helper that takes one
    if not isinstance(kind, str):
        raise TypeError(f"expected one reference prefix, got {kind!r}")


def ref_name(kind, value):
    """What follows the `kind` prefix in `value`, or None when `value` is
    not a reference of that kind. `kind` is a single prefix."""
    _single(kind)
    if not is_ref(kind, value):
        return None
    return value[len(kind) :]


def make_ref(kind, name):
    """The reference of `kind` naming `name`: make_ref(ASSET, "iris.png")
    is "asset:iris.png"."""
    _single(kind)
    return f"{kind}{name}"


def author_index(source_indices, index):
    """The index, in the steps the author wrote, of expanded step `index`.

    `expand_for_each` records one source index per expanded step, so an
    error in a for_each member is reported at the template step it came
    from. Without a record (no expansion, or a list too short) the step is
    its own source."""
    if source_indices is not None and index < len(source_indices):
        return source_indices[index]
    return index
```

In `scripts/arch_metrics.py`, replace the `PREFIX_OWNERS` comment and value:

```python
# Modules allowed to spell a reference prefix: the one owner (Phase 2a)
PREFIX_OWNERS = frozenset({"dw/references.py"})
```

- [ ] **Step 4: Run the tests, then re-baseline**
  - Run the Step 2 command. Expected: all pass.
  - Run `venv/bin/python scripts/arch_metrics.py`. Expected: `modules` 133, `prefix_literals` 93 (the new module's own literals are exempt), everything else unchanged.
  - Edit `docs/stabilization/baseline.json` by hand: set only `"modules": 133`, and leave every other value as it is.
  - Run `--check`. Expected: exit 0.
- [ ] **Step 5: Commit** `feat(references): one module owns the reference prefixes`, including the baseline change and its reason in the commit body.

### Task 2: Every prefix constant and prefix set comes from `references`

**Files:** Modify every module that defines a prefix constant or a prefix tuple. From the survey:
- constants:
  - `dw/assets.py:25`, `dw/runs.py:55`, `dw/prompts.py:23`;
  - `dw/arguments.py:63,68`, `dw/for_each.py:37,38`, `dw/realize.py:41,42`;
  - `dw/adapter_compatibility.py:55`, `dw/elision.py:50`, `dw/vram_estimate.py:55`, `dw/vram_inheritance.py:34`;
  - `dw/variable_constraints.py:48`, `dw/shots.py:42` (built from the constant; keep it as is).
- tuples:
  - `UNRESOLVED` in `dw/adapter_compatibility.py:58`, `dw/reference_names.py:34`, `dw/probe_paths.py:20`, `dw/reference_limits.py:48` and `dw/video_extensions.py:33`;
  - `SUBSTITUTED` in `dw/content_types.py:37`, `dw/kernel_availability.py:35`, `dw/result_fps.py:26` and `dw/subfolders.py:30`;
  - `DEFERRED` in `dw/locations.py:530-540` (`_deferred`);
  - `dw/prompts.py:28-34` `RESERVED_TEXT_PREFIXES`, rebuilt from the constants as the same six members.

**Interfaces:**
- A public name another module or a test imports stays importable from its module as an alias, for example `ASSET_PREFIX = references.ASSET`.
- A private single-prefix constant (a leading underscore, or no importer) is deleted, and its uses read `references.<NAME>`.
- The nine prefix tuples keep their names as aliases of the shared set (`_UNRESOLVED_PREFIXES = references.SUBSTITUTED` and so on), so each module still names the set it skips and the identity test below can pin it.
- The `is_*_reference` helpers (`dw/arguments.py:237,269,274,348`, `dw/assets.py:70`, `dw/runs.py:116`) keep their names and signatures. A helper whose whole body is the prefix test returns `is_ref(...)` with the matching kind. One with more logic, such as `is_media_reference` or `is_path_reference`, only swaps its literal for the constant.

- [ ] **Step 1: Write the failing test** `tests/test_reference_sets.py`. It pins each migrated site to its set by identity:

```python
"""Each checker skips exactly the references its pass cannot see yet - the
set it uses is references.<SET> itself, not a copy that can drift."""

import pytest

from dw import (
    adapter_compatibility,
    content_types,
    kernel_availability,
    probe_paths,
    reference_limits,
    reference_names,
    references,
    result_fps,
    subfolders,
    video_extensions,
)


@pytest.mark.parametrize(
    "module", [adapter_compatibility, reference_names, reference_limits, video_extensions]
)
def test_the_unresolved_checkers_share_one_set(module):
    assert module._UNRESOLVED_PREFIXES is references.UNRESOLVED


@pytest.mark.parametrize("module", [content_types, kernel_availability, result_fps, subfolders])
def test_the_substituted_checkers_still_check_step_results(module):
    assert module._UNRESOLVED_PREFIXES is references.SUBSTITUTED


def test_probe_paths_public_name_is_the_shared_set():
    assert probe_paths.UNRESOLVED_PREFIXES is references.UNRESOLVED
```

- [ ] **Step 2: Run it to verify it fails.** Run `venv/bin/python -m pytest -q -p no:cacheprovider tests/test_reference_sets.py`. Expected: `AssertionError` on each `is`, because the tuples are separate objects today.
- [ ] **Step 3: Migrate** the constants, tuples and helpers above. Check for each tuple, in the diff, that its member set equals the target set, whatever its order.
- [ ] **Step 4: Run** the new test and the full suite. Expected: all pass, with the suite otherwise unchanged. `prefix_literals` drops; report the number.
- [ ] **Step 5: Commit** `refactor(references): prefix constants and sets come from dw.references`

### Task 3: No inline prefix literal outside `references`

**Files:** the remaining literal sites in the 26 files the survey lists. The largest are:
- `dw/arguments.py` (the `startswith` pairs at :511, :518, :1003, :1150 and :1215);
- `dw/server/catalog_shape.py` (:120, :151, :156, :172, :181);
- `dw/server/app.py` (the `f"asset:{name}"` builders at :3772, :3862, :3876, :4020 and :4098);
- `dw/for_each.py`, `dw/variables.py`, `dw/introspection.py`, `dw/previous_results.py`, `dw/select_validation.py`, `dw/step_cache.py`, `dw/elision.py`, `dw/vram_estimate.py` and `dw/vram_inheritance.py`.

**Interfaces:** Rewrite each site with the matching helper:
- `value.startswith("x:")` → `is_ref(X, value)`, where `value` may be a non-string only if the site already checked; `is_ref` returns False for non-strings, so a guarded site simplifies;
- a guarded `value[len("x:"):]` / `removeprefix` → `ref_name(X, value)`;
- `f"x:{name}"` → `make_ref(X, name)`.

  A site whose semantics do not map one-to-one (Review Focus 2) keeps its logic and only swaps the literal for the constant.

- [ ] **Step 1: The failing check is the ratchet.** Run `venv/bin/python scripts/arch_metrics.py | grep prefix_literals` and record the number, above 0 after Task 2.
- [ ] **Step 2: Migrate** every site. Where a docstring or comment names a prefix inside longer text, leave it; the metric counts only exact-constant strings.
- [ ] **Step 3: Verify.**
  - `prefix_literals` is 0.
  - The full suite passes unchanged.
  - `grep -rnE '"(asset|output|prompt|variable|previous_result|constant|item|gather|builtin|constraint):"' dw dw_mcp --include='*.py' | grep -v "dw/references.py"` prints nothing.
  - `grep -rn 'f"asset:' dw` prints nothing.
- [ ] **Step 4: Ratchet it.**
  - Add `"builtin:"` and `"constraint:"` to `REFERENCE_PREFIXES` in `scripts/arch_metrics.py`, so the metric counts all ten prefixes that `references` owns. Do this only now, at 0; adding them earlier raises the count above the baseline.
  - Set `"prefix_literals": 0` in `docs/stabilization/baseline.json` by hand, then run `--check` (exit 0).
- [ ] **Step 5: Commit** `refactor(references): no module but dw.references spells a prefix`

### Task 4: One `author_index`

**Files:** the 20 copies of the source-index idiom:
- `dw/content_types.py:113`, `dw/kernel_availability.py:183`, `dw/reference_names.py:71`, `dw/result_fps.py:55`;
- `dw/reference_limits.py:191`, `dw/subfolders.py:75`, `dw/video_extensions.py:118`, `dw/locations.py:595`;
- `dw/previous_results.py:367`, `dw/adapter_compatibility.py:122,226`, `dw/dissolve_frame_errors.py:100`, `dw/null_media.py:74`;
- `dw/scalar_result_validation.py:120`, `dw/select_validation.py:61`, `dw/shot_span_preflight.py:97`, `dw/slice_preflight.py:125`;
- `dw/task_domains.py:306`, `dw/video_size_errors.py:98`, `dw/introspection.py:722,942,1019`;
- `dw/vram_estimate.py:249`, which keeps its own `_entry_position`.

**Interfaces:** each `source = (source_indices[index] if ... else index)` becomes `source = author_index(source_indices, index)`. `locations.py`'s variant (`if source_indices and ...`) is equivalent, because an empty list falls through to `index` in both.

- [ ] **Step 1: The failing check.** Run `grep -rn "index < len(source_indices)" dw | wc -l` and record the number (20 or more).
- [ ] **Step 2: Replace** every copy.
- [ ] **Step 3: Verify.** The grep prints 0 hits outside `dw/references.py`. The full suite passes unchanged. Report `duplicate_blocks`: it may drop, and if it does, lower it in `baseline.json` by hand.
- [ ] **Step 4: Commit** `refactor(references): one author_index for every checker`

### Task 5: Stage 2a merge

- [ ] **Step 1.** Run the full suite and `arch_metrics --check`. Both must pass.
- [ ] **Step 2.** The final whole-branch review over the stage, with its single fix pass.
- [ ] **Step 3.** Merge `stabilization/phase-2a` to `develop` and push. Do not deploy: lem is Don's during the phase, and a stage merge is behavior-preserving. lem is deployed and timed once, at gate 2.
- [ ] **Step 4.** Write stage 2b's detailed tasks in this file and set `hot-zone.txt` to stage 2b's files, in one commit, before 2b starts.

---

## Stage 2b: validation plumbing

Work on branch `stabilization/phase-2b` in the worktree, from `develop` after stage 2a (`39a97862` or later).

**What exists (survey, 2026-09-28).**

- **Errors.** `Workflow.validation_errors` (`dw/workflow.py:757-943`) works in three steps:
  1. It runs its gates: schema, then `constraint_reference_errors`, then the expansion with its five caught exceptions. Each returns early.
  2. It sums 23 checks and `sub_workflow_errors` in a fixed order. Each check is a pure function returning `[{path, message}]`, with an extra `"variable"` key on some entries.
  3. Nothing guards a single check, so one that raises loses the whole verdict. `admit()` turns that into `ValidatorFailure`, a 400.
- **Warnings.** There are ten warning sources, all returning `list[str]` in the form `"path: message"`.
  - Six are `Workflow` methods that each re-expand, the expansion memoized since Phase 1, inside `except Exception: return []`.
  - `admit()`'s `_warnings` wraps each of the ten in its own `except Exception`: a failure is logged, and its warnings silently vanish. That is **B10**.
- **Media.** Four probes run `probe_media`, which decodes whole files, fully and uncached: `dissolve_frame_errors._frame_count`, `video_size_errors._frame_size`, `slice_preflight._source_seconds` and `shot_span_preflight._frame_count`. One file referenced by two checks is decoded twice. That is **B9**.
- **The four rules.** Two of them have already drifted apart:
  - Frame size: validation lists every mismatched video, while the run stops at the first and always calls the reference "video 0".
  - Slice region: validation lacks #557's end-frame rounding.
  - Dissolve overlap is identical in both copies.
  - Select: the rule table is duplicated, and the requires-threshold and requires-index messages are identical literals.
  - `dw/task_domains.py` is the model: each domain is declared once, and `domain_violation` builds its message once. The run calls `check_arguments` and validation calls `task_argument_errors`. Tasks import it, and it imports no task, so there is no cycle risk.

### Decisions (2b)

- **`dw/validation.py` is the one new module.** It holds `Finding`, `ValidationContext`, the two registries and the runner with its exception policy.
  - It pays for itself: three checker modules whose only importers are `dw/workflow.py` and their own test fold into it, keeping their function names.
    - Task 2 folds `result_fps.py` (`fps_errors`) and `null_media.py` (`null_media_errors`) as it creates the module, so `modules` goes 133 → 132 in the same task and the ratchet holds after every task.
    - Task 6 folds `select_validation.py` (`select_errors`) once Task 5 has moved its rules out, taking `modules` to 131.
  - This is a small module move in a phase that otherwise forbids them, accepted by Don as part of 2b.
  - If importing the check modules from `dw/validation.py` creates an import cycle, the two registry lists live in `dw/workflow.py`, which already imports every check. `dw/validation.py` then keeps only `Finding`, `ValidationContext`, `run_checks` and the serializers.
  - `dw/validation.py` must not import `dw.workflow`. `ValidationContext` takes the `Workflow` instance as a value.
- **Each of the four rules gets one home in `dw/task_domains.py`.** Both the check and the task call the same function, with the same message. Where the two copies differ today, the text says deliberately which one wins:
  - **Slice:** the run's arithmetic wins, including #557's end rounding, because it is what executes.
  - **Frame size:** validation's fuller report wins. Every mismatch is named with each video's real index, because it names every fix the caller must make. The run's error text gains the other mismatches, which goes in the release notes.
- **One context per request.** `Workflow.validation_context(arguments=None, composing=())` builds a `ValidationContext`.
  - `validation_errors(arguments=None, composing=None, *, context=None)` builds one only when none is passed.
  - `admit()` calls the factory once and passes that context to the error pass and the warning pass, so there is one expansion and one probe cache per request.
  - The six `Workflow.*_warnings` methods build their own when called directly.
  - The context is never stored on the `Workflow`.
- **The exception policy (B10).**
  - A check that raises becomes one error finding: `path` None (the editor renders a null path as `root`; `ui/src/lib/pages/EditorPage.svelte:649`), `kind` `"internal"`, message `check '<name>' failed (<ExcType>: <msg>) - the server log has the traceback`. The traceback is logged at ERROR. Every other check still runs.
  - The verdict is `valid: false`. An internal failure never admits a job that went unchecked, and it no longer turns into a 400 `ValidatorFailure`.
  - A warning source that raises becomes one warning, `internal: warning check '<name>' failed (<ExcType>: <msg>)`, and is logged. It never refuses.
  - Surface changes for the release notes: `/api/validate` answers `valid: false` with an internal finding where it used to 400, and a failed warning pass now shows as a warning.
- **The response shapes are unchanged.** Errors serialize to `{path, message}` plus any extra keys they carry today. Warnings serialize to `"path: message"` strings, or the bare message when `path` is None. Serialization happens in one place, the runner's `to_errors()` and `to_warnings()`.
- **Order is preserved.** The error registry lists the checks in today's concatenation order, and the warning registry follows `admit()`'s helper order. A test pins the order by name.
- **B9 metadata probe.** `probe_metadata(path)` goes in `dw/media_info.py`. It reads the header values: kind, width, height, fps, sample rate, channels and duration. When the header lacks a video frame count, it counts that count by **demuxing packets without decoding**, which is exact for the codecs dw writes. It never runs the audio loudness analysis.
  - `ValidationContext.probe(path)` memoizes it by real path for one validation.
  - The four probing checks gain a keyword `probe=probe_metadata`, so their public call signatures keep working.
- **Out of scope for 2b, moved to 2d:** the `recorded_variables` media copy, which is run-path code. Expansion-memo invalidation is dropped: the definition is not mutated after construction, and a test in Task 3 pins that.

### Review Focus (2b)

1. **Verdict parity.**
   - For every catalog workflow in `workflows/templates/**`, validated with its defaults, `validation_errors` returns the same list before and after the migration, in the same order.
   - A committed fixture would be machine-dependent (VRAM capacity, kernel availability and local assets differ between the Mac and lem). So Task 3 checks parity in one process before deleting the `+` chain: it runs the old chain and the registry loop over every template and asserts equality. That is a throwaway script whose output goes in the report.
   - The committed net is the 27 existing validation test files.
2. **A crashing check never admits a job.** The internal finding makes the verdict `valid: false`. `admit()` refuses the submit with 400 and the finding's message, not a `ValidatorFailure` 500 or 400.
3. **The probe cache is per validation.** A cache that outlived the request would serve a replaced asset's old frame count. `ValidationContext` is built per call and never stored on the `Workflow`.
4. **`probe_metadata` frame counts match `probe_media`'s decode counts** on the test fixtures, the video files under `tests/` and the assets the probing tests use. Any codec where they differ falls back to `probe_media`.
5. **Rule messages.** A test calls the shared rule function and checks that the validation finding and the task's `ValueError` or warning carry the same sentence for the same inputs.

### Task 1: `probe_metadata` and a per-validation probe cache (B9)

**Files:** `dw/media_info.py` (add `probe_metadata`); `dw/dissolve_frame_errors.py`, `dw/video_size_errors.py`, `dw/slice_preflight.py` and `dw/shot_span_preflight.py` (each takes `probe=probe_metadata`); tests in `tests/test_media_info.py` (or the existing probe tests) and each checker's test file.

**Interfaces:**
- `probe_metadata(path) -> dict | None`, with the same keys `probe_media` returns for kind, width, height, fps, `frame_count`, `duration_seconds`, `sample_rate` and channels, and no loudness keys.
- The checkers take `(..., probe=probe_metadata)`.

**Tests, each RED first:**
- **Parity.** `probe_metadata(p)["frame_count"] == probe_media(p)["frame_count"]` and the same `width`/`height`/`duration_seconds`, for every video fixture the checker tests already use (parametrize over them).
- **No decode.** `probe_metadata` does not call `container.decode`. Assert it with a `patch.object` on the opened container, or by timing an injected fake container that raises on `decode`.
- **Shared cache.** A checker called twice through one cache probes once: pass a counting `probe`.

**Commit:** `perf(media): a metadata-only probe for validation (B9)`

### Task 2: `dw/validation.py`: `Finding`, `ValidationContext` and the runner

**Files:**
- Create `dw/validation.py` and `tests/test_validation.py`.
- Fold `fps_errors` (with `FPS_KEY`) and `null_media_errors` (with its helpers) into it, and delete `dw/result_fps.py` and `dw/null_media.py`.
- Update `dw/workflow.py`'s imports, and the import line in `tests/test_result_fps.py` and `tests/test_null_media.py`.
- Update `tests/test_reference_sets.py`, which pins `result_fps._UNRESOLVED_PREFIXES`, to pin the folded name.
- After this task `modules` is 132; lower it in `baseline.json` by hand.

**Interfaces (the implementer writes the bodies):**

```python
@dataclass(frozen=True)
class Finding:
    severity: str            # "error" | "warning"
    kind: str                # the check's registry name, or "internal"
    path: str | None
    message: str
    extra: dict = field(default_factory=dict)   # e.g. {"variable": ...} today

@dataclass
class ValidationContext:
    workflow: object         # the Workflow; never imported here
    arguments: dict | None
    expanded: dict           # from workflow.expanded_definition (memoized)
    source_indices: list
    base_dir: str | None
    composing: tuple = ()
    # probe(path) -> memoized probe_metadata for this context only

@dataclass(frozen=True)
class Check:
    name: str
    run: Callable[[ValidationContext], list]   # returns legacy dicts or strings

def run_checks(context, checks, severity) -> list[Finding]
    # per-check try/except Exception -> internal Finding (see Decisions),
    # logger.exception(...); legacy {path, message, **extra} dicts and
    # "path: message" strings convert to Findings
def to_errors(findings) -> list[dict]      # today's error shape
def to_warnings(findings) -> list[str]     # today's warning shape
```

**Tests, each RED first:**
- A check that raises yields exactly one internal finding and the other checks still run.
- Legacy dicts, with `"variable"` kept in `extra`, and `"path: message"` strings round-trip through `to_errors` / `to_warnings` unchanged.
- `ValidationContext.probe` memoizes per context: two contexts do not share entries.
- `dw/validation.py` does not import `dw.workflow`: check its AST imports, the same way `test_references` checks its module.

**Commit:** `feat(validation): Finding, ValidationContext and one exception policy`

### Task 3: `validation_errors` is a registry loop

**Files:** `dw/validation.py` (the `ERROR_CHECKS` registry), `dw/workflow.py` (`validation_errors`), and `tests/test_validation.py`, plus the template baseline fixture.

- **Step 1:** the registry, holding the 23 checks plus `task_errors`, with its `arguments is None` variable filter, and `sub_workflow_errors`, in today's order, each a `Check(name, lambda ctx: <existing call>)`. `validation_errors` keeps its gates exactly and replaces the `+` chain with `to_errors(run_checks(ctx, ERROR_CHECKS, "error"))`.
- **Step 2, before removing the `+` chain:** run the throwaway parity script (Review Focus 1). Put its output in the report, then remove the chain.
- **Tests:**
  - registry order pinned by name, with the docstring line "error order is part of the /api/validate response; a reorder changes what the editor shows first";
  - a monkeypatched check raising → `valid: false` with the internal message, through `POST /api/validate`;
  - the same through `POST /api/jobs` → 400 with that message, and nothing queued;
  - the expansion memo: the definition object is unchanged after validation (the dropped invalidation item).
- **Commit:** `refactor(validation): validation_errors is a loop over the check registry (B10 for errors)`

### Task 4: One warning registry for `admit()` and the `Workflow` methods

**Files:** `dw/validation.py` (`WARNING_CHECKS`), `dw/workflow.py` (the six `*_warnings` methods delegate and keep their names and signatures), and `dw/server/admission.py` (`_warnings` becomes one `run_checks` over one context).

**Expected test changes:** tests that call `adapter_warnings` or `slice_past_end_warnings` directly on a broken input and assert `[]` now get an `internal:` warning. Change each such assertion deliberately and list it in the report.

- The context is built once per admission and shared with the error pass, so there is one expansion and one probe cache per request.
- **Tests:**
  - A warning source that raises → an `internal:` warning in the validate response, and the job still queues.
  - A workflow whose dissolve step and analyze step name the same asset probes it once per validate request (a counting `probe`).
  - The existing admission fold-count tests still pass: one `_fold` per request.
- **Commit:** `refactor(validation): one warning registry; a failed warning pass is visible (B10)`

### Task 5: The four rules, one home each

**Files:**
- `dw/task_domains.py`, which gains the four rule functions;
- `dw/dissolve_frame_errors.py`, `dw/video_size_errors.py`, `dw/slice_preflight.py` and `dw/select_validation.py`;
- `dw/tasks/dissolve_videos.py`, `dw/tasks/video_utils.py` (`check_same_frame_size`), `dw/tasks/audio_utils.py` (`slice_audio` / `_warn_on_slice_past_end`) and `dw/tasks/select.py`.

**Interfaces (names final, bodies per the survey):**
- `dissolve_shortfalls(frame_counts, dissolve_frames) -> list[str]`: one sentence per short video, in today's wording.
- `frame_size_mismatches(sizes) -> str | None`: `sizes` maps index to `(w, h)`, with a missing video absent. It returns the one sentence naming every mismatch against the first known size. Validation appends its remedy text; the run raises it as it stands.
- `slice_padding(total_samples, start, length, sample_rate) -> float | None`: the padded seconds, when at or over `SLICE_PAD_WARN_MS`. `SLICE_PAD_WARN_MS` moves here. `audio_utils` re-exports it for its importers, and validation computes `start`/`length` in samples with the #557 end rounding: `frames_to_samples` moves or is imported so both sides call it.
- `SELECT_RULES`, `SELECT_THRESHOLD_RULES`, `select_rule_problems(rule, threshold, index) -> list[str]`: the unknown-rule, requires-threshold and requires-index sentences. Validation's forbid-extraneous check and the gather-shape check stay validation-only.

**Tests:** one parity test per rule. Its inputs go through both the checker and the task, and the rule's sentence appears in both, RED first where the copies differ today (frame size, slice rounding).

**Commit:** `refactor(task_domains): the four rules written twice have one home each`

### Task 6: Fold `select_validation` into `dw/validation.py`

**Files:** move `select_errors`, which is validation-only once Task 5 has moved the rules to `task_domains`, into `dw/validation.py`; delete `dw/select_validation.py`; update `dw/workflow.py`'s import and the import line in `tests/test_select_validation.py`.

- **The failing check is the ratchet:** `modules` 132. Afterwards it is 131; lower it in `baseline.json` by hand.
- **Commit:** `refactor(validation): fold select_validation into dw/validation.py`

### Task 7: Stage 2b merge

The same steps as 2a's Task 5:
1. The full suite and `--check`.
2. The final review and its one fix pass.
3. Merge and push, with no lem deploy.
4. Write stage 2c's detail and its hot zone in one commit.

Add 2b's surface changes to "Release notes collected".
