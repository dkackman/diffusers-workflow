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
- the `recorded_variables` media copy → 2b (Task 4's area);
- `dw.run` paging a truncated event tail → 2c;
- expansion memo invalidation → 2b;
- the loose `JobManager` name fallbacks → 2c.

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
