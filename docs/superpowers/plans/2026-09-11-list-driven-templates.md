# List-driven templates (for_each stage 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `workflows/templates/minimax/music-video.json` and `workflows/templates/minimax/dialogue-short.json` onto one `shots` list each, so a six-shot episode is an argument rather than a different file, and move the plugin skill, docs and tests with them.

**Architecture:** Stage 1 shipped `expand_for_each` (`dw/for_each.py`): a step carrying `for_each` becomes one ordinary step per entry, `item:field` splices an entry's field, `gather:<step>` reads the group. This stage (1) lets a list-valued variable's entries reference *other* variables (`"from_file": "variable:character_a_voice"`), resolved once before `realize_args`, because that is how dialogue-short's optional voices survive the move into entries; (2) teaches the catalog's shape derivation that `"videos": "gather:shot"` is a cut over a list; (3) rewrites the two templates, whose shot steps become full `pipeline` blocks (the identity-keyed pipeline cache means every member after the first reuses the loaded H3 exactly as `pipeline_reference` did); (4) rewrites the tests that hand-expanded the *old* templates, the voices test, the plugin skill and the docs.

**Tech Stack:** Python 3.10, pytest, JSON workflow templates, markdown skill/docs.

**Spec:** `docs/proposals/list-driven-steps.md` — "Phasing" item 2, "Decided at review", and "Notes for stage 2".

## Global Constraints

- Every test in the suite passes: `pytest -q -x tests/` (the baseline on `develop` at faa27e1 is 3492 passed, 5 skipped). Run the full suite before every commit that touches `dw/`.
- No engine behaviour changes beyond the two named here: nested `variable:` resolution inside list/dict variable values, and `gather:` recognised by `_cuts_together`. Nothing else in `dw/for_each.py`, `dw/variables.py` or `dw/workflow.py` changes semantics.
- The rewritten templates must expand (`Workflow.expanded_definition()`) to the same *number* of shot steps with the same prompts, references (same order, same `from_previous_result` targets), `num_frames` and `start_frame` values the hand-written steps carried on `develop` at faa27e1 — `git show faa27e1:workflows/templates/minimax/<name>.json` is the reference.
- Member names are `slice@<key>` / `shot@<key>` with keys `wide_open`, `closeup`, `room`, `finale` (music-video) and `cold_open`, `deflect`, `react`, `button`, `tag` (dialogue-short) — the entry names stage 1's tests already chose.
- The catalog pins hold unchanged: `tests/test_catalog_structure.py` `EXPECTED_SHAPES` (`sequence`, `["has-audio", "identity-referenced"]`) and `COSTED` (35 / 42 minutes) for both templates.
- `plugins/dw/skills/minimax-h3/SKILL.md` stays at or under 12288 bytes (`SKILL_SIZE_LIMIT`); it is at 11535 now, so every sentence added is paid for by one removed or tightened.
- Security rules from CLAUDE.md apply: entry names are already validated by `validate_variable_name` in the expansion; nothing new touches the filesystem.
- Commit trailer on every commit: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- No version bump in this branch. The proposal calls for the two templates to land together, and they do; the engine version (`pyproject.toml`, `plugins/dw/.claude-plugin/plugin.json`, both `0.4.0-beta.3`) is bumped by `scripts/release.sh` from master, which is the user's call. The breaking change (the `shot_N_*` argument names are gone) is recorded in the proposal's status and in the CLAUDE.md gotcha so the next release note can quote it.

---

### Task 1: A list-valued variable's entries may reference other variables

**Files:**
- Modify: `dw/variables.py` (add `resolve_variable_values`; extend `undeclared_variable_references`)
- Modify: `dw/workflow.py:300-312` (`expanded_definition`), `dw/workflow.py:337-354` (`_undeclared_variable_errors`), `dw/workflow.py:430-446` (`Workflow.run` substitution block)
- Test: `tests/test_variables.py`, `tests/test_workflow.py`

**Interfaces:**
- Consumes: `replace_variables(data, variables)`, `VariableNotFoundError`, `set_variables`, `argument_errors` (all in `dw/variables.py`).
- Produces: `resolve_variable_values(variables) -> dict` — a new dict in which every `"variable:<name>"` string found *inside* a list- or dict-valued variable is replaced by that variable's (already resolved) value. Scalar variable values are returned as they are, even a string that begins with `variable:` (that has always been passed through verbatim and stays so). Raises `VariableNotFoundError` for an undeclared name and `ValueError("Variable 'a' references itself through: a -> b -> a")` for a cycle. `undeclared_variable_references(definition)` now also returns references found inside list/dict values of `variables`, with paths like `variables.shots[0].references[2].from_file`. `Workflow._undeclared_variable_errors(arguments=None)` folds good caller arguments in before walking, and reports a reference inside a caller-supplied value at `arguments.<name>...` rather than `variables.<name>...`.

- [ ] **Step 1: Write the failing tests for `resolve_variable_values`**

Append to `tests/test_variables.py`:

```python
from dw.variables import resolve_variable_values, undeclared_variable_references


class TestResolveVariableValues:
    """A list-valued variable's entries may name other variables - a shot
    entry says "from_file": "variable:character_a_voice" and one variable
    sets the voice in every shot it speaks in."""

    def test_a_reference_inside_a_list_value_is_replaced(self):
        variables = {
            "voice": "cast/priya.wav",
            "shots": [{"name": "a", "references": [{"from_file": "variable:voice"}]}],
        }
        resolved = resolve_variable_values(variables)
        assert resolved["shots"][0]["references"][0]["from_file"] == "cast/priya.wav"

    def test_a_reference_inside_a_dict_value_is_replaced(self):
        variables = {"n": 124, "shape": {"num_frames": "variable:n"}}
        assert resolve_variable_values(variables)["shape"] == {"num_frames": 124}

    def test_a_null_variable_resolves_to_null(self):
        variables = {
            "voice": None,
            "shots": [{"references": [{"from_file": "variable:voice"}]}],
        }
        resolved = resolve_variable_values(variables)
        assert resolved["shots"][0]["references"][0]["from_file"] is None

    def test_a_scalar_value_that_looks_like_a_reference_is_left_alone(self):
        variables = {"x": "variable:y", "y": 1}
        assert resolve_variable_values(variables)["x"] == "variable:y"

    def test_a_chain_resolves_through_a_referenced_list(self):
        variables = {
            "voice": "a.wav",
            "refs": [{"from_file": "variable:voice"}],
            "shots": [{"references": "variable:refs"}],
        }
        resolved = resolve_variable_values(variables)
        assert resolved["shots"][0]["references"] == [{"from_file": "a.wav"}]

    def test_the_input_is_not_mutated(self):
        variables = {"voice": "a.wav", "shots": [{"from_file": "variable:voice"}]}
        before = copy.deepcopy(variables)
        resolve_variable_values(variables)
        assert variables == before

    def test_an_undeclared_name_is_the_usual_error(self):
        with pytest.raises(VariableNotFoundError, match="nope"):
            resolve_variable_values({"shots": [{"x": "variable:nope"}]})

    def test_a_cycle_is_an_error_that_names_the_loop(self):
        variables = {"a": [{"x": "variable:b"}], "b": [{"y": "variable:a"}]}
        with pytest.raises(ValueError, match="a -> b -> a"):
            resolve_variable_values(variables)

    def test_a_self_reference_is_a_cycle(self):
        with pytest.raises(ValueError, match="a -> a"):
            resolve_variable_values({"a": [{"x": "variable:a"}]})


class TestUndeclaredReferencesInsideVariableValues:
    def test_a_reference_inside_a_list_value_is_found_with_its_path(self):
        definition = {
            "variables": {"shots": [{"references": [{}, {"from_file": "variable:nope"}]}]},
            "steps": [],
        }
        assert undeclared_variable_references(definition) == [
            ("variables.shots[0].references[1].from_file", "nope")
        ]

    def test_a_declared_reference_inside_a_value_is_not_reported(self):
        definition = {
            "variables": {"voice": None, "shots": [{"from_file": "variable:voice"}]},
            "steps": [],
        }
        assert undeclared_variable_references(definition) == []

    def test_a_scalar_value_beginning_with_the_prefix_is_not_a_reference(self):
        definition = {"variables": {"x": "variable:nope"}, "steps": []}
        assert undeclared_variable_references(definition) == []
```

Add `import copy` and `import pytest` at the top of the file if they are not already there, and make sure `VariableNotFoundError` is imported from `dw.variables`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_variables.py -q`
Expected: ImportError on `resolve_variable_values`.

- [ ] **Step 3: Implement `resolve_variable_values` and extend the undeclared walk**

In `dw/variables.py`, after `replace_variables`:

```python
def resolve_variable_values(variables):
    """A copy of `variables` in which every "variable:name" inside a list-
    or dict-valued variable is replaced by that variable's value.

    A list-driven step reads its entries from a variable, and an entry that
    says "from_file": "variable:character_a_voice" is how one variable sets
    a voice in every shot the character speaks in. replace_variables only
    walks the definition, so those references would reach the step as the
    literal strings; this resolves them once, before realize_args, so a
    reference type inside an entry is a type name by the time it is loaded.

    Only list and dict values are walked. A scalar value that begins with
    "variable:" is passed through as it always was.

    Raises:
        VariableNotFoundError: a reference names nothing declared
        ValueError: a value references itself, directly or through others
    """
    resolved = {}

    def resolve(name, chain):
        if name in resolved:
            return resolved[name]
        if name in chain:
            loop = " -> ".join(chain[chain.index(name) :] + [name])
            raise ValueError(f"Variable '{name}' references itself through: {loop}")
        value = variables[name]
        if isinstance(value, (list, dict)):
            value = walk(value, chain + [name])
        else:
            value = copy.deepcopy(value)
        resolved[name] = value
        return value

    def walk(node, chain):
        if isinstance(node, str) and node.startswith("variable:"):
            target = node.removeprefix("variable:")
            if target not in variables:
                available = ", ".join(sorted(variables.keys())) or "<none>"
                raise VariableNotFoundError(
                    f"Variable <{target}> not found; available variables: {available}"
                )
            return resolve(target, chain)
        if isinstance(node, list):
            return [walk(item, chain) for item in node]
        if isinstance(node, dict):
            return {key: walk(item, chain) for key, item in node.items()}
        return copy.deepcopy(node)

    for name in variables:
        resolve(name, [])
    return resolved
```

In `undeclared_variable_references`, replace the final loop so list/dict variable values are walked too:

```python
    for key, value in definition.items():
        if key != "variables":
            walk(value, key)
    if isinstance(declared, dict):
        for name, value in declared.items():
            if isinstance(value, (list, dict)):
                walk(value, f"variables.{name}")
    return found
```

and update its docstring: "Walks everything but `variables` itself, the way resolution does" becomes "Walks everything but `variables` itself, plus the inside of every list- or dict-valued variable, the way `resolve_variable_values` and `replace_variables` together do."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_variables.py -q`
Expected: PASS.

- [ ] **Step 5: Write the failing workflow-level tests**

Append to `tests/test_workflow.py` (it already has `_for_each_workflow(**overrides)` and `_workflow_from(definition, tmp_path)`; reuse them):

```python
def test_an_entry_may_reference_another_variable(tmp_path):
    """A shot entry's "from_file": "variable:voice" is the voice variable's
    value by the time the member exists."""
    definition = _for_each_workflow()
    definition["variables"]["voice"] = "cast/priya.wav"
    definition["variables"]["shots"] = [
        {"name": "a", "text": "one", "voice": "variable:voice"}
    ]
    definition["steps"][0]["task"]["arguments"]["voice"] = "item:voice"
    workflow = _workflow_from(definition, tmp_path)

    expanded = workflow.expanded_definition()

    assert expanded["steps"][0]["task"]["arguments"]["voice"] == "cast/priya.wav"


def test_an_undeclared_reference_inside_a_default_entry_is_a_validation_error(
    tmp_path,
):
    definition = _for_each_workflow()
    definition["variables"]["shots"] = [{"name": "a", "text": "variable:nope"}]
    workflow = _workflow_from(definition, tmp_path)

    errors = workflow.validation_errors()

    assert [e["path"] for e in errors] == ["variables.shots[0].text"]
    assert "names no declared variable" in errors[0]["message"]


def test_an_undeclared_reference_inside_a_caller_s_entry_is_reported_under_arguments(
    tmp_path,
):
    workflow = _workflow_from(_for_each_workflow(), tmp_path)

    errors = workflow.validation_errors(
        arguments={"shots": [{"name": "a", "text": "variable:nope"}]}
    )

    assert [e["path"] for e in errors] == ["arguments.shots[0].text"]


def test_a_caller_s_entry_may_reference_a_declared_variable(tmp_path):
    definition = _for_each_workflow()
    definition["variables"]["voice"] = None
    workflow = _workflow_from(definition, tmp_path)

    errors = workflow.validation_errors(
        arguments={"shots": [{"name": "a", "text": "variable:voice"}]}
    )

    assert errors == []
```

Read `_for_each_workflow` first (`tests/test_workflow.py:622`) to confirm the step is a `task` step whose arguments include `"text": "item:text"`; if its field is named differently, use that name in place of `text` above.

- [ ] **Step 6: Run them to verify they fail**

Run: `pytest tests/test_workflow.py -q -k "entry"`
Expected: the first fails with the literal `variable:voice` string surviving; the undeclared ones fail with an empty error list or the wrong path.

- [ ] **Step 7: Wire resolution into the workflow**

In `dw/workflow.py`, import `resolve_variable_values` alongside the other `.variables` imports.

`expanded_definition` — replace the substitution block:

```python
        definition = copy.deepcopy(self.workflow_definition)
        variables = definition.get("variables")
        if isinstance(variables, dict):
            if arguments and not argument_errors(definition, arguments):
                set_variables(arguments, variables)
            variables = resolve_variable_values(variables)
            definition = replace_variables(definition, variables)
        return expand_for_each(definition, source_indices)
```

`validation_errors` — pass the arguments through: `return self._undeclared_variable_errors(arguments)`.

`_undeclared_variable_errors` — take and fold the arguments:

```python
    def _undeclared_variable_errors(self, arguments=None):
        """Every 'variable:' reference naming nothing the workflow declares.

        Fatal rather than a warning: once a workflow has a 'variables'
        block, replace_variables refuses an undeclared reference, so this is
        a run that cannot start. Good caller `arguments` are folded in first,
        and a reference inside one of them is reported under `arguments.`,
        where the caller wrote it.
        """
        definition = copy.deepcopy(self.workflow_definition)
        variables = definition.get("variables")
        supplied = set()
        if isinstance(variables, dict) and arguments:
            if not argument_errors(definition, arguments):
                set_variables(arguments, variables)
                supplied = set(arguments)
        declared = sorted(variables or {})

        def where(path):
            head, _, rest = path.partition(".")
            if head == "variables":
                name = rest.split(".", 1)[0].split("[", 1)[0]
                if name in supplied:
                    return "arguments." + rest
            return path

        return [
            {
                "path": where(path),
                "message": (
                    f"'variable:{name}' names no declared variable; "
                    f"declared: {', '.join(declared) or '<none>'}"
                ),
            }
            for path, name in undeclared_variable_references(definition)
        ]
```

`Workflow.run` — resolve before `realize_args`, so a `reference_type` inside an entry is a type name when the realizer reaches it:

```python
                set_variables(arguments, variables)
                # an entry of a list-valued variable may name another
                # variable; resolve those before anything inside it is
                # realized, so a reference type in an entry is a type name
                variables = resolve_variable_values(variables)
                # realize the variables, initialiting downloads of images etc
                realize_args(variables, base_dir)
```

Check the two later uses of `variables` in that function (the realized-workflow write and the seed) still read the resolved dict — `realize_workflow` takes `arguments` and re-folds them itself, so nothing else changes. Also open `dw/realize.py:75-93` and confirm it does not need resolution: it writes the variables as declared, with the run's arguments folded, and a nested `variable:` string there is right — a rerun resolves it again.

- [ ] **Step 8: Run the tests and the full suite**

Run: `pytest tests/test_workflow.py tests/test_variables.py -q` then `pytest -q -x tests/`
Expected: PASS; full suite green.

- [ ] **Step 9: Commit**

```bash
git add dw/variables.py dw/workflow.py tests/test_variables.py tests/test_workflow.py
git commit -m "feat(variables): an entry of a list-valued variable may reference another variable

Resolved once before realize_args, reported as undeclared at the entry's
path, cycles refused. This is how a for_each template's optional voice
variable reaches every shot the character speaks in.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The catalog sees a `gather:` concat as a cut over a list

**Files:**
- Modify: `dw/server/catalog_shape.py:160-183` (`_cuts_together`)
- Test: `tests/test_catalog_shape.py`

**Interfaces:**
- Consumes: `derive_catalog_metadata(definition)`, the test file's `definition`, `pipeline_step`, `task_step` helpers.
- Produces: `_cuts_together` returns True for `"videos": "gather:<step>"` exactly as it does for `"videos": "variable:<name>"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_catalog_shape.py`, next to `test_a_concat_fed_by_two_steps_is_a_sequence`:

```python
def test_a_concat_over_a_gathered_for_each_group_is_a_sequence():
    """A list-driven template's editor says "videos": "gather:shot" - a
    cut over as many shots as the list holds, which the file cannot count."""
    meta = derive_catalog_metadata(
        definition(
            {
                "name": "shot",
                "for_each": "variable:shots",
                "pipeline": {
                    "arguments": {"prompt": "item:prompt"},
                },
                "result": {"content_type": "video/mp4"},
            },
            task_step("cut", "concat_videos", {"videos": "gather:shot"}, "video/mp4"),
        )
    )
    assert meta["shape"] == "sequence"
```

Read the file's `pipeline_step` helper first; if it can produce a step with a `for_each` key, use it instead of the literal dict, so the test reads like its neighbours.

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_catalog_shape.py -q -k gathered`
Expected: FAIL, shape is `shot`.

- [ ] **Step 3: Recognise the prefix**

In `_cuts_together`, change the string check and its docstring:

```python
    """A concat or dissolve fed by two or more distinct steps, or by a list
    of shots handed in whole - one `variable:` reference is a supplied list
    whose length only the caller knows, one `gather:` reference is every
    member of a for_each group, and a cut over either is still an edit."""
    ...
            if isinstance(videos, str) and videos.startswith(("variable:", "gather:")):
                return True
```

- [ ] **Step 4: Run the catalog tests**

Run: `pytest tests/test_catalog_shape.py tests/test_catalog_structure.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/server/catalog_shape.py tests/test_catalog_shape.py
git commit -m "fix(catalog): a concat over gather:<step> derives sequence

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `music-video` on a `shots` list

**Files:**
- Modify: `workflows/templates/minimax/music-video.json`
- Modify: `tests/test_for_each.py` (`TestMusicVideoTemplate`, lines 630-712)
- Modify: `workflows/templates/minimax/README.md:136` (the table row)

**Interfaces:**
- Consumes: `expand_for_each`, `Workflow.expanded_definition()` (Task 1), `resolve_variable_values`.
- Produces: the template's `variables.shots`, a list of four `{name, prompt, start_frame}` entries; steps `draw_singer`, `write_song`, `slice` (for_each), `soundtrack`, `shot` (for_each), `edit`, `music_video`.

- [ ] **Step 1: Rewrite the template**

Start from the file as it is. Changes, in order:

1. In `variables`: delete `shot_1_wide_open`, `shot_2_closeup`, `shot_3_room`, `shot_4_finale`. In their place (after `audio_duration`) add:

```json
        "shots": [
            {
                "name": "wide_open",
                "start_frame": 0,
                "prompt": "<the exact string that was variables.shot_1_wide_open>"
            },
            {
                "name": "closeup",
                "start_frame": 124,
                "prompt": "<the exact string that was variables.shot_2_closeup>"
            },
            {
                "name": "room",
                "start_frame": 248,
                "prompt": "<the exact string that was variables.shot_3_room>"
            },
            {
                "name": "finale",
                "start_frame": 372,
                "prompt": "<the exact string that was variables.shot_4_finale>"
            }
        ],
```

Move each prompt string byte-for-byte (copy the JSON string literal, escapes and all); do not retype it.

2. Replace the four `slice_N` steps with one:

```json
        {
            "name": "slice",
            "for_each": "variable:shots",
            "task": {
                "command": "slice_audio",
                "arguments": {
                    "audio": "previous_result:write_song",
                    "sample_rate": "variable:sample_rate",
                    "start_frame": "item:start_frame",
                    "num_frames": "variable:num_frames",
                    "fps": "variable:fps"
                }
            }
        },
```

3. Keep `soundtrack` as it is.

4. Replace `shot_1_wide_open` and the three `pipeline_reference` steps with one step: the full `shot_1_wide_open` step, renamed `shot`, with `"for_each": "variable:shots"` inserted directly after `"name"`, `"prompt": "item:prompt"`, and the audio reference's `"from_previous_result": "slice_1"` changed to `"from_previous_result": "slice"`. Everything else in the block (configuration, quantization, components, loras, `num_frames`, `width`, `height`, `num_inference_steps`, `output`, `result`) stays exactly as it was.

5. In `edit`, replace the four-element `videos` list with `"videos": "gather:shot"`.

6. Rewrite `description` (one string; keep the catalog `summary`-less form the file has — it has no `summary`, so the first sentence is the summary):

```
A music video built from cuts, sung to a soundtrack that never touches a chain. The long-take way to film a song - one chained generation lip-synced end to end - degrades with every carried segment and can let the sync slip. This builds the video the way music television does instead: MiniMax-Music3 writes the song, one 'slice_audio' step per entry of 'shots' cuts it into frame-exact pieces (124 frames at 24 fps each, from each entry's 'start_frame'), and one shot per entry is generated fresh from the same Z-Image portrait plus its own slice, lip-synced to just those five seconds. 'shots' is a list, so the cut is an argument: add an entry and there is one more slice and one more shot, named for it ('slice@closeup', 'shot@closeup'), and the loaded model is reused across every shot - identical pipeline definitions share one model. No shot conditions on another shot's output, so the last cut is as clean as the first. 'concat_videos' gathers the shots in list order, and because each one covered exactly its slice's frames, the edit is sample-accurate by construction: 'pair_audio' drops the original, unbroken song over the whole cut and the mouths line up in every shot. The generation models pass through one at a time - each is released before the next loads - so the workflow peaks no higher than its largest single model.
```

- [ ] **Step 2: Validate the file the way the CLI does**

Run: `python -m dw.validate workflows/templates/minimax/music-video.json`
Expected: valid, no errors. Then `python -c "import json; json.load(open('workflows/templates/minimax/music-video.json'))"` to be sure the edit left valid JSON.

- [ ] **Step 3: Replace `TestMusicVideoTemplate`**

Delete the class at `tests/test_for_each.py:630-712` and write in its place:

```python
class TestMusicVideoTemplate:
    """music-video's slices and shots are two for_each groups over one
    'shots' list, paired by entry name: shot@closeup reads slice@closeup."""

    KEYS = ["wide_open", "closeup", "room", "finale"]

    def expanded(self):
        from dw.workflow import Workflow

        return Workflow(load_template("music-video.json"), TEMPLATES).expanded_definition()

    def test_the_template_validates_as_it_will_run(self):
        from dw.workflow import Workflow

        assert Workflow(load_template("music-video.json"), TEMPLATES).validation_errors() == []

    def test_one_slice_and_one_shot_per_entry_in_list_order(self):
        names = [s["name"] for s in self.expanded()["steps"]]
        assert names == (
            ["draw_singer", "write_song"]
            + [f"slice@{k}" for k in self.KEYS]
            + ["soundtrack"]
            + [f"shot@{k}" for k in self.KEYS]
            + ["edit", "music_video"]
        )

    def test_each_slice_starts_where_its_entry_says(self):
        got = steps_by_name(self.expanded())
        starts = [got[f"slice@{k}"]["task"]["arguments"]["start_frame"] for k in self.KEYS]
        assert starts == [0, 124, 248, 372]

    def test_each_shot_reads_its_own_slice_and_the_one_portrait(self):
        got = steps_by_name(self.expanded())
        for key in self.KEYS:
            references = got[f"shot@{key}"]["pipeline"]["arguments"]["references"]
            assert [r["from_previous_result"] for r in references] == [
                "draw_singer",
                f"slice@{key}",
            ]

    def test_each_shot_carries_its_entry_s_prompt(self):
        template = load_template("music-video.json")
        got = steps_by_name(self.expanded())
        for entry in template["variables"]["shots"]:
            prompt = got[f"shot@{entry['name']}"]["pipeline"]["arguments"]["prompt"]
            assert prompt == entry["prompt"]
            assert prompt.startswith("subject_definitions:")

    def test_every_shot_is_the_same_pipeline(self):
        """Full pipeline blocks rather than pipeline_reference: the identity
        cache reuses the loaded model, so this costs no reload."""
        from dw.workflow import pipeline_cache_key

        got = steps_by_name(self.expanded())
        keys = {pipeline_cache_key(got[f"shot@{k}"]["pipeline"]) for k in self.KEYS}
        assert len(keys) == 1

    def test_the_edit_gathers_the_shots_in_order(self):
        got = steps_by_name(self.expanded())
        assert got["edit"]["task"]["arguments"]["videos"] == [
            f"previous_result:shot@{k}" for k in self.KEYS
        ]

    def test_the_realized_file_keeps_the_list(self):
        """workflow.json beside a run is the source form: 'for_each' and the
        'shots' variable, not the expanded members."""
        template = load_template("music-video.json")
        assert "shots" in template["variables"]
        assert [s["name"] for s in template["steps"] if "for_each" in s] == ["slice", "shot"]
```

Check `Workflow.__init__`'s signature (`dw/workflow.py`, `class Workflow`) before writing: if it takes `(workflow_definition, base_dir, ...)` in a different order or by keyword, match it. If the file's `without_pipeline_reference` helper is now unused by any test, delete it.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_for_each.py tests/test_examples.py tests/test_catalog_structure.py tests/test_plugin_skills.py -q`
Expected: PASS. If `test_example_type_references_resolve` complains about a `*_type` inside `shots`, read its skip rules (`tests/test_examples.py:135-160`) — a `variable:`-prefixed value is skipped; the music-video entries carry no `*_type` keys, so this should not fire.

- [ ] **Step 5: Update the README row**

`workflows/templates/minimax/README.md:136` becomes:

```
| [music-video.json](music-video.json) | A music video cut to a generated song, one `shots` list driving both `for_each` groups: `slice_audio` deals each entry its frame-exact piece, the shot lip-syncs to it, and `pair_audio` lays the unbroken track over the finished edit |
```

- [ ] **Step 6: Full suite and commit**

Run: `pytest -q -x tests/`
Expected: green.

```bash
git add workflows/templates/minimax/music-video.json tests/test_for_each.py workflows/templates/minimax/README.md
git commit -m "feat(templates): music-video's slices and shots are two for_each groups over one shots list

Breaking for scripted callers: shot_1_wide_open .. shot_4_finale are
entries of 'shots' now, each {name, prompt, start_frame}.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `dialogue-short` on a `shots` list

**Files:**
- Modify: `workflows/templates/minimax/dialogue-short.json`
- Modify: `tests/test_dialogue_short_voices.py`
- Modify: `tests/test_for_each.py` (`TestDialogueShortTemplate`, the class after `TestMusicVideoTemplate`)
- Modify: `workflows/templates/minimax/README.md:135`
- Modify: `tests/test_catalog_structure.py` only if a description-mentions test fails (see Step 4)

**Interfaces:**
- Consumes: Task 1's nested resolution (entries say `"from_file": "variable:character_a_voice"` and `"reference_type": "variable:subject_reference_type"`), `Workflow.expanded_definition()`, `realize_args`.
- Produces: `variables.shots`, five `{name, prompt, references, num_frames}` entries; steps `draw_character_a`, `draw_character_b`, `shot` (for_each), `episode`. The variables `shot_1_cold_open` … `shot_5_tag`, `num_frames` and `tag_num_frames` are gone; `character_a_voice` / `character_b_voice`, the two `*_reference_type` variables and everything else stay.

- [ ] **Step 1: Rewrite the template**

1. In `variables`, delete `shot_1_cold_open` … `shot_5_tag`, `num_frames` and `tag_num_frames`. After `character_b_portrait_prompt` add `shots`. Each entry's `references` is the reference list its hand-written step carried, verbatim; each `prompt` is the string moved byte-for-byte from the deleted variable:

```json
        "shots": [
            {
                "name": "cold_open",
                "num_frames": 124,
                "references": [
                    {"reference_type": "variable:subject_reference_type", "from_previous_result": "draw_character_a"},
                    {"reference_type": "variable:subject_reference_type", "from_previous_result": "draw_character_b"},
                    {"reference_type": "variable:voice_reference_type", "from_file": "variable:character_a_voice"},
                    {"reference_type": "variable:voice_reference_type", "from_file": "variable:character_b_voice"}
                ],
                "prompt": "<was variables.shot_1_cold_open>"
            },
            {
                "name": "deflect",
                "num_frames": 124,
                "references": [
                    {"reference_type": "variable:subject_reference_type", "from_previous_result": "draw_character_b"},
                    {"reference_type": "variable:voice_reference_type", "from_file": "variable:character_b_voice"}
                ],
                "prompt": "<was variables.shot_2_deflect>"
            },
            {
                "name": "react",
                "num_frames": 124,
                "references": [
                    {"reference_type": "variable:subject_reference_type", "from_previous_result": "draw_character_a"},
                    {"reference_type": "variable:voice_reference_type", "from_file": "variable:character_a_voice"}
                ],
                "prompt": "<was variables.shot_3_react>"
            },
            {
                "name": "button",
                "num_frames": 124,
                "references": [
                    {"reference_type": "variable:subject_reference_type", "from_previous_result": "draw_character_b"},
                    {"reference_type": "variable:voice_reference_type", "from_file": "variable:character_b_voice"}
                ],
                "prompt": "<was variables.shot_4_button>"
            },
            {
                "name": "tag",
                "num_frames": 141,
                "references": [
                    {"reference_type": "variable:subject_reference_type", "from_previous_result": "draw_character_a"},
                    {"reference_type": "variable:subject_reference_type", "from_previous_result": "draw_character_b"},
                    {"reference_type": "variable:voice_reference_type", "from_file": "variable:character_a_voice"},
                    {"reference_type": "variable:voice_reference_type", "from_file": "variable:character_b_voice"}
                ],
                "prompt": "<was variables.shot_5_tag>"
            }
        ],
```

Format the reference objects one key per line like the rest of the file (four-space indent); the compact form above is for reading.

2. Replace `shot_1_cold_open` and the four `pipeline_reference` steps with one step: the full `shot_1_cold_open` block renamed `shot`, `"for_each": "variable:shots"` after `"name"`, and in `arguments`: `"prompt": "item:prompt"`, `"references": "item:references"`, `"num_frames": "item:num_frames"`. Everything else unchanged.

3. In `episode`, `"videos": "gather:shot"`.

4. Rewrite `description`:

```
A digital short built the way television is built: from cuts, not from one long take. Chained generation degrades with length - every segment conditions on the previous segment's output, so artifacts compound and identity drifts. A scene cut resets that completely: each shot here is generated fresh from the same two character portraits, so shot five is exactly as clean as shot one and the scene can run as long as the script does. Two Z-Image steps draw the cast (the second reuses the first's loaded pipeline - identical configurations share one model - and 'release_pipeline' frees it before the video model loads). The shots are one 'for_each' step over the 'shots' list: one entry per shot, carrying its 'name', its 'prompt', its 'references' and its 'num_frames' - the tag entry runs 141 frames where the others run 124, since length is per-shot. The list is an argument, so a six-shot scene is one more entry, not another file, and the members are named for their entries ('shot@react'); the loaded MiniMax-H3 is reused across all of them. Character consistency across cuts comes from referencing the same portraits in every shot; voice consistency comes from repeating each character's voice description verbatim in every prompt, and, when 'character_a_voice' / 'character_b_voice' name a clip ('asset:cast/priya.wav'), from the audio reference each entry lists for whoever speaks in it - an entry's reference says 'variable:character_a_voice', so one variable sets the voice in every shot that character has. Both default to null, and a reference whose file is null is left out of the list, so a run that names no voice generates exactly what it generated before the variables existed. A shot where both speak lists both; if that reads worse than one, name one voice and leave the other null. The variables are named for roles rather than for the cast of this example - 'character_a', the 'react' entry - because the beats are the reusable part and the sketch is not. The soundscape writes the laugh track. A final 'concat_videos' task is the editor, gathering the shots in list order into one episode - hard cuts, no trims, no seams to hide, because nothing was carried between them. Only the picture cuts hard: 'audio_bleed_ms' rings each shot's laugh track on over the silent opening of the next, the way a live audience carries across a cut. It and 'seam_fade_ms' are variables, so a seam is re-tuned with an argument rather than a copy of the workflow; 1800 ms is where a five-shot cut measured best, since a generated shot opens on more silence than it looks.
```

Keep `summary` as it is.

- [ ] **Step 2: Validate**

Run: `python -m dw.validate workflows/templates/minimax/dialogue-short.json`
Expected: valid.

- [ ] **Step 3: Rewrite the voices test**

Replace the body of `tests/test_dialogue_short_voices.py` from `SPEAKERS` down (keep the module docstring, add one sentence to it: "The shots are entries of a list now, and an entry's voice reference names the variable, so the same one variable still sets the voice everywhere.") with:

```python
# Which character speaks in which shot - each entry lists a voice
# reference per speaker, and this is the mapping it is asserted against
SPEAKERS = {
    "shot@cold_open": ("a", "b"),
    "shot@deflect": ("b",),
    "shot@react": ("a",),
    "shot@button": ("b",),
    "shot@tag": ("a", "b"),
}


@pytest.fixture
def definition():
    with open(TEMPLATE, encoding="utf-8") as file:
        return json.load(file)


def shot_references(definition, arguments):
    """Every shot member's references, as the engine realizes them for a
    run: the caller's arguments folded, entries resolved, the list expanded,
    then each member's arguments realized as its step would."""
    from dw.workflow import Workflow

    expanded = Workflow(definition, os.path.dirname(TEMPLATE)).expanded_definition(
        arguments
    )
    references = {}
    for step in expanded["steps"]:
        if step["name"] not in SPEAKERS:
            continue
        arguments = step["pipeline"]["arguments"]
        realize_args(arguments)
        references[step["name"]] = arguments["references"]
    assert set(references) == set(SPEAKERS)
    return references


class TestOptionalVoices:
    def test_the_voices_default_to_null(self, definition):
        variables = definition["variables"]
        assert variables["character_a_voice"] is None
        assert variables["character_b_voice"] is None

    def test_every_entry_names_the_voice_variables_rather_than_a_file(self, definition):
        for entry in definition["variables"]["shots"]:
            voices = [
                r["from_file"] for r in entry["references"] if "from_file" in r
            ]
            assert voices, entry["name"]
            assert all(v.startswith("variable:character_") for v in voices), entry["name"]

    def test_no_voice_named_leaves_only_the_portraits(self, definition):
        for name, references in shot_references(definition, {}).items():
            assert len(references) == len(set(SPEAKERS[name])), name
            assert all(
                reference["from_previous_result"].startswith("draw_character")
                for reference in references
            ), name

    def test_a_named_voice_is_referenced_in_the_shots_it_speaks_in(
        self, definition, tmp_path
    ):
        from tests.test_media_info import write_wav

        voice = tmp_path / "priya.wav"
        write_wav(voice, seconds=1.0)

        for name, references in shot_references(
            definition, {"character_a_voice": str(voice)}
        ).items():
            built = [r for r in references if not isinstance(r, dict)]
            assert len(built) == (1 if "a" in SPEAKERS[name] else 0), name

    def test_the_tag_runs_longer(self, definition):
        frames = {e["name"]: e["num_frames"] for e in definition["variables"]["shots"]}
        assert frames == {
            "cold_open": 124, "deflect": 124, "react": 124, "button": 124, "tag": 141
        }

    def test_the_variable_names_are_roles_rather_than_a_cast(self, definition):
        """Every run carried howie_portrait_prompt and shot_3_howie_incredulous
        through its arguments, manifest and export whatever the cast was."""
        names = " ".join(definition["variables"]) + " ".join(
            step["name"] for step in definition["steps"]
        ) + " ".join(e["name"] for e in definition["variables"]["shots"])
        assert "howie" not in names.lower()
        assert "pat_" not in names.lower()
        assert "character_a_portrait_prompt" in definition["variables"]
        assert [e["name"] for e in definition["variables"]["shots"]] == [
            "cold_open", "deflect", "react", "button", "tag"
        ]
```

Note the realized `from_file` reference is built at variable-realization time in a real run (before expansion) and at member-realization time here; both go through `realize_object`, and a built object passes through `realize_args` unchanged, so the count is the same either way. If `realize_args` in `shot_references` fails on an already-loaded `reference_type` (a `type`, not a string), that is a bug to report as a concern, not to work around.

- [ ] **Step 4: Replace `TestDialogueShortTemplate`**

```python
class TestDialogueShortTemplate:
    """dialogue-short's five shots are one for_each group whose entries
    carry everything that differs between shots: prompt, references and
    length."""

    KEYS = ["cold_open", "deflect", "react", "button", "tag"]

    def expanded(self):
        from dw.workflow import Workflow

        return Workflow(load_template("dialogue-short.json"), TEMPLATES).expanded_definition()

    def test_the_template_validates_as_it_will_run(self):
        from dw.workflow import Workflow

        assert Workflow(load_template("dialogue-short.json"), TEMPLATES).validation_errors() == []

    def test_one_shot_per_entry_between_the_cast_and_the_edit(self):
        names = [s["name"] for s in self.expanded()["steps"]]
        assert names == (
            ["draw_character_a", "draw_character_b"]
            + [f"shot@{k}" for k in self.KEYS]
            + ["episode"]
        )

    def test_each_shot_references_the_portraits_its_entry_lists(self):
        got = steps_by_name(self.expanded())
        portraits = {
            key: [
                r["from_previous_result"]
                for r in got[f"shot@{key}"]["pipeline"]["arguments"]["references"]
                if "from_previous_result" in r
            ]
            for key in self.KEYS
        }
        assert portraits == {
            "cold_open": ["draw_character_a", "draw_character_b"],
            "deflect": ["draw_character_b"],
            "react": ["draw_character_a"],
            "button": ["draw_character_b"],
            "tag": ["draw_character_a", "draw_character_b"],
        }

    def test_the_reference_types_are_resolved_inside_the_entries(self):
        """An entry's "variable:subject_reference_type" is the dotted type
        name by the time the member exists - never the literal reference."""
        got = steps_by_name(self.expanded())
        for key in self.KEYS:
            for r in got[f"shot@{key}"]["pipeline"]["arguments"]["references"]:
                assert r["reference_type"].startswith("diffusers.modular_pipelines")

    def test_the_tag_runs_longer(self):
        got = steps_by_name(self.expanded())
        frames = [got[f"shot@{k}"]["pipeline"]["arguments"]["num_frames"] for k in self.KEYS]
        assert frames == [124, 124, 124, 124, 141]

    def test_every_shot_is_the_same_pipeline(self):
        from dw.workflow import pipeline_cache_key

        got = steps_by_name(self.expanded())
        assert len({pipeline_cache_key(got[f"shot@{k}"]["pipeline"]) for k in self.KEYS}) == 1

    def test_the_episode_gathers_the_shots_in_order(self):
        got = steps_by_name(self.expanded())
        assert got["episode"]["task"]["arguments"]["videos"] == [
            f"previous_result:shot@{k}" for k in self.KEYS
        ]
```

- [ ] **Step 5: Run the affected tests**

Run: `pytest tests/test_for_each.py tests/test_dialogue_short_voices.py tests/test_examples.py tests/test_catalog_structure.py tests/test_plugin_skills.py -q`
Expected: PASS. If `test_a_description_names_only_variables_the_workflow_declares` fails because the description quotes `'shots'`-entry words that another workflow declares as a variable, read its allow-list at `tests/test_catalog_structure.py:210` and either reword the description or extend the list with a comment saying why; do not weaken the test.

- [ ] **Step 6: Update the README row**

`workflows/templates/minimax/README.md:135`:

```
| [dialogue-short.json](dialogue-short.json) | A five-shot sitcom scene: Z-Image draws the cast, one `for_each` step over a `shots` list generates a shot per entry - its prompt, its references, its length - on one loaded model, and `concat_videos` gathers the episode |
```

- [ ] **Step 7: Full suite and commit**

Run: `pytest -q -x tests/`

```bash
git add workflows/templates/minimax/dialogue-short.json tests/test_dialogue_short_voices.py tests/test_for_each.py workflows/templates/minimax/README.md tests/test_catalog_structure.py
git commit -m "feat(templates): dialogue-short's five shots are one for_each group over a shots list

Breaking for scripted callers: shot_1_cold_open .. shot_5_tag, num_frames
and tag_num_frames are entries of 'shots' now, each {name, prompt,
references, num_frames}. An entry's voice reference names
character_a_voice / character_b_voice, so one variable still sets a voice
in every shot the character speaks in.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

(Drop `tests/test_catalog_structure.py` from `git add` if it was not touched.)

---

### Task 5: The skill, the guide, CLAUDE.md and the proposal say so

**Files:**
- Modify: `plugins/dw/skills/minimax-h3/SKILL.md` (the "A piece with cuts" bullet; size cap 12288 bytes)
- Modify: `tests/test_plugin_skills.py` (`TestMiniMaxH3Skill`)
- Modify: `docs/WORKFLOW_GUIDE.md` (the `### One step per entry: for_each` section, ~line 340-410)
- Modify: `CLAUDE.md` (the `for_each` bullet in Type System, ~line 157; the `for_each` gotcha in Critical Gotchas)
- Modify: `docs/proposals/list-driven-steps.md` (status line, lines 3-5; "Notes for stage 2")

**Interfaces:**
- Consumes: the templates' new `shots` entry shapes from Tasks 3 and 4.
- Produces: skill text an agent reads before composing; tests that pin it.

- [ ] **Step 1: Write the failing skill tests**

Add to `TestMiniMaxH3Skill` in `tests/test_plugin_skills.py`:

```python
    def test_the_cuts_templates_are_described_as_list_driven(self):
        """Both cut templates take one 'shots' list; the skill says what an
        entry carries, so an agent writes entries rather than the shot_N_*
        arguments T005 and this rewrite removed."""
        import json

        text = skill_text(H3_SKILL)
        assert "`shots`" in text
        assert "shot_1_" not in text and "shot_2_" not in text
        for name, fields in (
            ("dialogue-short", {"name", "prompt", "references", "num_frames"}),
            ("music-video", {"name", "prompt", "start_frame"}),
        ):
            path = os.path.join(
                REPO_ROOT, "workflows", "templates", "minimax", name + ".json"
            )
            spec = json.load(open(path, encoding="utf-8"))
            entries = spec["variables"]["shots"]
            assert all(set(entry) == fields for entry in entries), name
            for field in fields:
                assert f"`{field}`" in text, f"the skill does not name {field}"
        # cost scales with the list, and the listing's figure is the default's
        assert "per shot" in text or "per entry" in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_plugin_skills.py -q -k list_driven`
Expected: FAIL on `` `shots` ``.

- [ ] **Step 3: Edit the skill**

In `plugins/dw/skills/minimax-h3/SKILL.md`, the bullet beginning `- **A piece with cuts**:`. Replace its first two sentences

```
  `templates/minimax/dialogue-short` (Z-Image draws the cast, one loaded model
  per shot, `concat_videos` splices) and `templates/minimax/music-video`
  (shots cut to a generated song, lip-synced slices). A cut erases drift; the
  last shot is as clean as the first. Write shots, not takes.
```

with

```
  `templates/minimax/dialogue-short` (Z-Image draws the cast, one shot per
  entry of its `shots` list on one loaded model, `concat_videos` splices) and
  `templates/minimax/music-video` (a song, one slice and one lip-synced shot
  per entry). `shots` is one list argument: a dialogue entry is `name`,
  `prompt`, `references` (which portraits and voices this shot uses) and
  `num_frames`; a music-video entry is `name`, `prompt` and `start_frame`.
  A six-shot piece is one more entry, not another file; the listing's `cost`
  is the default list's, so quote it per shot times the entries you write.
  A cut erases drift; the last shot is as clean as the first. Write shots,
  not takes.
```

and later in the same bullet, change

```
  Each shot's `num_frames` is its own, so pace the
  cut - a trailer builds by varying shot length.
```

to

```
  Each entry's `num_frames` is its own, so pace the cut.
```

Then measure: `wc -c plugins/dw/skills/minimax-h3/SKILL.md`. If over 12288, tighten elsewhere in the same bullet (the Bark/narrator sentences are the longest and can lose "Bark's presets are conversational;") until under. Do not cut a hard rule or a source.

- [ ] **Step 4: Run the plugin tests**

Run: `pytest tests/test_plugin_skills.py -q`
Expected: PASS, including the size cap.

- [ ] **Step 5: The guide**

In `docs/WORKFLOW_GUIDE.md`, in `### One step per entry: for_each`, after the paragraph beginning "`item:field` is the whole value of that field", add:

```
An entry may name another variable: `"from_file": "variable:character_a_voice"`
inside a `references` entry is that variable's value by the time the member
exists, so one variable sets a voice in every shot the character speaks in
and a caller who supplies the list still writes `variable:` for the parts the
template fixes. Those references are resolved before anything in the entry is
loaded, and an undeclared one is a validation error at the entry's path
(`arguments.shots[2].references[1].from_file` when the list is yours,
`variables.shots[...]` when it is the template's). A value may not reference
itself, directly or through another variable.
```

And in the closing "Limits" paragraph, after "quote `cost × len(list)`", add nothing new (it already says it), but change "`validate_workflow` with the `arguments` you will run with: it expands your list, not the template's default" to "`validate_workflow` with the `arguments` you will run with: it expands your list, not the template's default, resolves the variables your entries name". Then at the end of the section, before `### The loop`, add:

```
`templates/minimax/dialogue-short` and `templates/minimax/music-video` are
this shape: each takes one `shots` list, and `get_workflow` on either shows
the entry an item needs.
```

- [ ] **Step 6: CLAUDE.md**

In the Type System `for_each` bullet, after "two groups over the same list pair by key (`slice` inside `shot@x` is `slice@x`)." insert: "An entry of a list-valued variable may reference another variable (`"from_file": "variable:character_a_voice"`); `resolve_variable_values` (`dw/variables.py`) replaces those once, before `realize_args`, refusing a cycle, and `undeclared_variable_references` walks inside list/dict variable values too."

In Critical Gotchas, add a bullet after the `for_each` one:

```
- **The two MiniMax cut templates take one `shots` list** — since the stage-2
  rewrite (2026-09-11) `templates/minimax/dialogue-short` and `music-video`
  have no `shot_N_*` variables; a scripted caller passes `shots` (entries
  `{name, prompt, references, num_frames}` and `{name, prompt, start_frame}`).
  The members are `shot@<name>` in the manifest and the gallery. This is the
  breaking change the next release note should name
```

- [ ] **Step 7: The proposal**

`docs/proposals/list-driven-steps.md` lines 3-5 become:

```
Status: **stages 1 and 2 implemented** (expansion pass, validation, schema,
docs - `dw/for_each.py`; `music-video` and `dialogue-short` on a `shots`
list, 2026-09-11); stage 3 (catalog per-entry cost and entry shape) not
started. Written for MCP feedback ticket T003.
```

Under "## Notes for stage 2", add a closing paragraph:

```
Resolved in stage 2: the `pipeline_reference` question went away - every
member is the full pipeline block, and the identity-keyed pipeline cache
reuses the loaded model exactly as the reference did (a test holds every
member to one `pipeline_cache_key`). A group's `pipeline_reference` still
gets no directed error; nothing bundled uses one now. Entries that name
other variables (`variable:character_a_voice`) are resolved by
`resolve_variable_values` before `realize_args`, which is what let the
optional voices move into the entries. The empty-list and
`realize_constants` questions are still open and belong to stage 3 with the
catalog work.
```

- [ ] **Step 8: Full suite, then commit**

Run: `pytest -q -x tests/`

```bash
git add plugins/dw/skills/minimax-h3/SKILL.md tests/test_plugin_skills.py docs/WORKFLOW_GUIDE.md CLAUDE.md docs/proposals/list-driven-steps.md
git commit -m "docs(for_each): the cut templates take one shots list; entries may name variables

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
