import json
import pytest
import torch
import tempfile
from unittest.mock import MagicMock
from dw.workflow import (
    Workflow,
    workflow_from_file,
    pipeline_cache_key,
    referenced_result_names,
    release_unreferenced_results,
    workflow_output_subfolder,
)
from dw.pipeline_processors.pipeline import Pipeline
import os

# Referenced by test_validation_realizes_a_constant_default_list via
# "constant:tests.test_workflow.CONSTANT_SHOTS" - a module-level value a
# workflow's variable default can point at instead of a literal list.
CONSTANT_SHOTS = [{"name": "a", "text": "1"}, {"name": "b", "text": "2"}]


def test_workflow_validation_valid(valid_workflow_json, tmp_path):
    workflow = Workflow(valid_workflow_json, str(tmp_path), "")
    workflow.validate()  # Should not raise exception


def test_workflow_validation_invalid(invalid_workflow_json, tmp_path):
    workflow = Workflow(invalid_workflow_json, str(tmp_path), "")
    with pytest.raises(Exception) as exc_info:
        workflow.validate()
    assert "Validation error" in str(exc_info.value)


def test_workflow_name(valid_workflow_json, tmp_path):
    workflow = Workflow(valid_workflow_json, str(tmp_path), "")
    assert workflow.name == "test_workflow"


def test_workflow_from_file(test_data_dir, tmp_path):
    workflow_path = os.path.join(test_data_dir, "workflows", "valid_workflow.json")
    workflow = workflow_from_file(workflow_path, str(tmp_path))
    assert isinstance(workflow, Workflow)


def test_workflow_variables_property(valid_workflow_json, tmp_path):
    workflow = Workflow(valid_workflow_json, str(tmp_path), "")
    assert "prompt" in workflow.variables
    assert workflow.variables["prompt"] == "test prompt"


def test_workflow_argument_template(valid_workflow_json, tmp_path):
    workflow = Workflow(valid_workflow_json, str(tmp_path), "")
    # Should return empty dict if no argument_template
    assert workflow.argument_template == {}


def test_workflow_security_validation(tmp_path):
    from dw.security import SecurityError

    # Test path traversal protection
    with pytest.raises(SecurityError):
        workflow_from_file("../../../etc/passwd", str(tmp_path))


class TestWorkflowOutputSubfolder:
    """Output files land under output_dir/<subfolder>, where <subfolder>
    mirrors the workflow file's position under a 'workflows' directory."""

    def test_top_level_workflow_is_flat(self):
        assert workflow_output_subfolder("/repo/workflows/ZImage.json") == ""

    def test_nested_workflow_gets_a_subfolder(self):
        assert (
            workflow_output_subfolder("/repo/workflows/ltx/SomeWorkflow.json") == "ltx"
        )

    def test_deeply_nested_workflow_keeps_the_full_relative_path(self):
        assert workflow_output_subfolder(
            "/repo/workflows/ltx/variants/A.json"
        ) == os.path.join("ltx", "variants")

    def test_workflow_outside_any_workflows_tree_is_flat(self):
        assert workflow_output_subfolder("/repo/somewhere/else/Foo.json") == ""

    def test_builtin_workflow_directly_in_dw_workflows_is_flat(self):
        import dw.workflow as workflow_module

        builtin = os.path.join(
            os.path.dirname(os.path.abspath(workflow_module.__file__)),
            "workflows",
            "builtin_example.json",
        )
        assert workflow_output_subfolder(builtin) == ""

    def test_no_file_spec_is_flat(self):
        assert workflow_output_subfolder("") == ""
        assert workflow_output_subfolder(None) == ""

    def test_effective_output_dir_appends_the_subfolder(self, tmp_path):
        workflows_dir = tmp_path / "workflows" / "ltx"
        workflows_dir.mkdir(parents=True)
        file_spec = str(workflows_dir / "Foo.json")
        workflow = Workflow({"id": "x"}, str(tmp_path / "outputs"), file_spec)
        assert workflow.effective_output_dir == str(tmp_path / "outputs" / "ltx")

    def test_effective_output_dir_is_flat_for_a_top_level_workflow(self, tmp_path):
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        file_spec = str(workflows_dir / "Foo.json")
        workflow = Workflow({"id": "x"}, str(tmp_path / "outputs"), file_spec)
        assert workflow.effective_output_dir == str(tmp_path / "outputs")

    def test_sub_workflow_computes_its_own_subfolder_not_the_parents(self, tmp_path):
        """A parent workflow nested under 'ltx' running a sub-workflow that
        lives directly under 'workflows' must not inherit the parent's
        subfolder - each workflow's own file position decides its output
        location, since workflow_from_file always receives the plain root
        output_dir, not the parent's effective_output_dir."""
        (tmp_path / "workflows" / "ltx").mkdir(parents=True)
        (tmp_path / "workflows" / "Child.json").write_text(
            json.dumps({"id": "child", "steps": []})
        )
        parent_spec = str(tmp_path / "workflows" / "ltx" / "Parent.json")
        parent = Workflow({"id": "parent"}, str(tmp_path / "outputs"), parent_spec)
        assert parent.effective_output_dir == str(tmp_path / "outputs" / "ltx")

        child = workflow_from_file(
            str(tmp_path / "workflows" / "Child.json"), parent.output_dir
        )
        assert child.effective_output_dir == str(tmp_path / "outputs")


class TestResultEviction:
    """Results no later step references are released after saving"""

    def test_references_are_found_wherever_they_nest(self):
        steps = [
            {
                "name": "later",
                "pipeline": {
                    "arguments": {
                        "image": "previous_result:gen",
                        "masks": [{"mask": "previous_result:segment.mask"}],
                    }
                },
            }
        ]
        assert referenced_result_names(steps) == {"gen", "segment.mask"}

    def test_a_constructed_objects_source_step_is_a_reference(self):
        # 'from_previous_result' names a step without the prefix - releasing
        # its result would break the reference built from it
        steps = [
            {
                "name": "later",
                "pipeline": {
                    "arguments": {
                        "references": [
                            {
                                "reference_type": "pkg.ImageReference",
                                "from_previous_result": "draw_subject",
                            }
                        ]
                    }
                },
            }
        ]
        assert referenced_result_names(steps) == {"draw_subject"}

    def test_unreferenced_results_are_released(self):
        results = {"gen": object(), "old": object()}
        release_unreferenced_results(results, {"gen"})
        assert list(results) == ["gen"]

    def test_property_references_keep_their_result(self):
        # 'segment.mask' must keep the result named 'segment' - and a step
        # literally named 'segment.mask' as well
        results = {"segment": object(), "segment.mask": object(), "old": object()}
        release_unreferenced_results(results, {"segment.mask"})
        assert sorted(results) == ["segment", "segment.mask"]

    def test_no_references_releases_everything(self):
        results = {"a": object(), "b": object()}
        release_unreferenced_results(results, set())
        assert results == {}


class TestSeedResolution:
    """Seeds resolve most-specific-first: pipeline > step > workflow"""

    def step_definition(self, configuration=None, pipeline_seed=None):
        pipeline = {"configuration": configuration or {}, "arguments": {}}
        if pipeline_seed is not None:
            pipeline["seed"] = pipeline_seed
        return {"name": "gen", "pipeline": pipeline}

    def cached_step_action(self, step_definition, seed, device="cpu"):
        """Run create_step_action down the cached-pipeline path"""
        with tempfile.TemporaryDirectory() as temp_output:
            workflow = Workflow({"id": "seeds", "steps": []}, temp_output, "")
            cached = Pipeline(step_definition["pipeline"], seed, device, MagicMock())
            # The cache is keyed by pipeline identity, not step name
            cache_key = pipeline_cache_key(step_definition["pipeline"])
            return workflow.create_step_action(
                step_definition, {}, {cache_key: cached}, seed, device
            )

    def test_generator_is_seeded_with_the_step_seed(self):
        action = self.cached_step_action(self.step_definition(), seed=222)
        assert action.argument_template["generator"].initial_seed() == 222

    def test_pipeline_seed_overrides_the_step_seed(self):
        action = self.cached_step_action(
            self.step_definition(pipeline_seed=333), seed=222
        )
        assert action.argument_template["generator"].initial_seed() == 333

    def test_no_generator_false_still_creates_a_generator(self):
        # no_generator is a boolean - an explicit false requests a generator
        action = self.cached_step_action(
            self.step_definition(configuration={"no_generator": False}), seed=1
        )
        assert "generator" in action.argument_template

    def test_no_generator_true_disables_the_generator(self):
        action = self.cached_step_action(
            self.step_definition(configuration={"no_generator": True}), seed=1
        )
        assert "generator" not in action.argument_template

    def test_cached_generator_lives_on_the_pipeline_device(self):
        # The pipeline's own device override wins over the workflow default,
        # matching the fresh-load path
        action = self.cached_step_action(
            self.step_definition(configuration={"device": "cpu"}),
            seed=1,
            device="cuda",
        )
        assert action.argument_template["generator"].device.type == "cpu"


class TestSubWorkflowSeedInheritance:
    """A delegated workflow runs under the parent's seed unless it names one.

    Left to itself a child draws its own random seed, which would put the work a
    workflow delegates outside the reach of the seed it was given.
    """

    def step_definition(self, path):
        return {"name": "child", "workflow": {"path": path, "arguments": {}}}

    def child_steps(self):
        return [
            {
                "name": "noop",
                "task": {"command": "get_image_size", "arguments": {}},
            }
        ]

    def child_action(self, child_definition, seed, tmp_path):
        child = tmp_path / "child.json"
        child.write_text(json.dumps(child_definition))
        parent = Workflow({"id": "parent", "steps": []}, str(tmp_path), "")
        return parent.create_step_action(
            self.step_definition(str(child)), {}, {}, seed, "cpu"
        )

    def test_child_inherits_the_parent_seed(self, tmp_path):
        action = self.child_action(
            {"id": "c", "steps": self.child_steps()}, 4242, tmp_path
        )
        assert action.workflow_definition["seed"] == 4242

    def test_child_seed_wins(self, tmp_path):
        action = self.child_action(
            {"id": "c", "seed": 99, "steps": self.child_steps()}, 4242, tmp_path
        )
        assert action.workflow_definition["seed"] == 99


class TestGlobalRngIsolation:
    """Workflow.run must not reseed the RNG the process may rely on"""

    def run_empty_workflow(self, definition):
        with tempfile.TemporaryDirectory() as temp_output:
            Workflow(definition, temp_output, "").run({})

    def test_run_with_an_explicit_seed_leaves_global_rng_alone(self):
        torch.manual_seed(1234)
        state = torch.get_rng_state()
        self.run_empty_workflow({"id": "w", "seed": 42, "steps": []})
        assert torch.equal(state, torch.get_rng_state())

    def test_run_without_a_seed_leaves_global_rng_alone(self):
        torch.manual_seed(1234)
        state = torch.get_rng_state()
        self.run_empty_workflow({"id": "w", "steps": []})
        assert torch.equal(state, torch.get_rng_state())


def test_workflow_validation_error_names_the_json_path_once(tmp_path):
    """Every entry point - the CLI, the server's /api/validate, a job load -
    goes through Workflow.validate(); the message it raises must name where
    in the document the failure is, and carry the prefix exactly once."""
    from dw.workflow import Workflow

    definition = {
        "id": "bad",
        "steps": [
            {
                "name": "gen",
                "seed": "not-a-number",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}"},
                    "from_pretrained_arguments": {"model_name": "m"},
                    "arguments": {},
                },
            }
        ],
    }
    with pytest.raises(Exception) as exc_info:
        Workflow(definition, str(tmp_path), "").validate()
    message = str(exc_info.value)
    assert "steps[0].seed" in message
    assert message.count("Validation error") == 1


class TestSubWorkflowConfinement:
    """A server-submitted workflow is confined to workflow_dir, but the
    builtin: workflows ship inside the package, outside any workflow_dir -
    a builtin step must still load there."""

    def test_builtin_sub_workflow_loads_under_a_confined_workflow(self, tmp_path):
        workflow_dir = tmp_path / "workflows"
        workflow_dir.mkdir()
        parent = Workflow(
            {"id": "parent", "steps": []},
            str(tmp_path / "outputs"),
            str(workflow_dir / "__inline__.json"),
            str(workflow_dir),
        )
        step = {"name": "child", "workflow": {"path": "builtin:test.json"}}

        child = parent.create_step_action(step, {}, {}, 42, "cpu")

        assert isinstance(child, Workflow)
        assert child.name == "test_job"


class TestSubWorkflowPathsAcrossTheCatalog:
    """A template under templates/ names a model config under models/ as
    '../models/x.json'. The path is relative to the referencing file, so it has
    to climb one directory - and it stays inside the workflows root, which is
    the confinement that matters."""

    def _catalog(self, tmp_path):
        import json

        (tmp_path / "templates").mkdir()
        (tmp_path / "models").mkdir()
        child = {
            "id": "child",
            "steps": [
                {
                    "name": "noop",
                    "task": {
                        "command": "get_dict_value",
                        "arguments": {"dict": {"k": 1}, "key": "k"},
                    },
                }
            ],
        }
        (tmp_path / "models" / "child.json").write_text(json.dumps(child))
        parent = {
            "id": "parent",
            "steps": [
                {
                    "name": "sub",
                    "workflow": {"path": "../models/child.json", "arguments": {}},
                }
            ],
        }
        parent_path = tmp_path / "templates" / "parent.json"
        parent_path.write_text(json.dumps(parent))
        return parent_path

    def test_a_parent_directory_step_inside_the_root_is_allowed(self, tmp_path):
        from dw.workflow import workflow_from_file

        parent_path = self._catalog(tmp_path)
        workflow = workflow_from_file(str(parent_path), str(tmp_path), str(tmp_path))

        # create_step_action is what resolves the path; run() is what calls it.
        # default_seed flows into the child's own "seed" (setdefault), whose
        # schema wants an int or string - None is only safe where a
        # SecurityError fires first, so this branch needs a real seed
        action = workflow.create_step_action(
            workflow.workflow_definition["steps"][0],
            shared_components={},
            previous_pipelines={},
            default_seed=42,
            device="cpu",
        )

        assert action is not None

    def test_a_parent_directory_step_escaping_the_root_is_refused(self, tmp_path):
        import json

        from dw.security import SecurityError
        from dw.workflow import workflow_from_file

        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside.json"
        outside.write_text(json.dumps({"id": "x", "steps": []}))
        parent = {
            "id": "parent",
            "steps": [
                {
                    "name": "sub",
                    "workflow": {"path": "../outside.json", "arguments": {}},
                }
            ],
        }
        parent_path = root / "parent.json"
        parent_path.write_text(json.dumps(parent))
        workflow = workflow_from_file(str(parent_path), str(root), str(root))

        with pytest.raises(SecurityError):
            workflow.create_step_action(
                workflow.workflow_definition["steps"][0],
                shared_components={},
                previous_pipelines={},
                default_seed=None,
                device="cpu",
            )

    def test_an_unconfined_run_still_refuses_climbing_out_of_the_catalog(
        self, tmp_path
    ):
        """No workflow_dir (a bare CLI run) still confines a relative
        sub-workflow reference - to the catalog root now, rather than
        relying on the '..' regex normpath removes."""
        import json

        from dw.security import SecurityError
        from dw.workflow import workflow_from_file

        (tmp_path / "workflows" / "templates").mkdir(parents=True)
        outside = tmp_path / "outside.json"
        outside.write_text(json.dumps({"id": "x", "steps": []}))
        parent = {
            "id": "parent",
            "steps": [
                {
                    "name": "sub",
                    "workflow": {"path": "../../outside.json", "arguments": {}},
                }
            ],
        }
        parent_path = tmp_path / "workflows" / "templates" / "parent.json"
        parent_path.write_text(json.dumps(parent))
        workflow = workflow_from_file(str(parent_path), str(tmp_path))

        with pytest.raises(SecurityError):
            workflow.create_step_action(
                workflow.workflow_definition["steps"][0],
                shared_components={},
                previous_pipelines={},
                default_seed=42,
                device="cpu",
            )

    def test_an_unconfined_run_may_climb_to_a_sibling_catalog_folder(self, tmp_path):
        """No workflow_dir, but the reference still stays under the nearest
        ancestor literally named 'workflows' - the catalog root - so it is
        still allowed to climb from templates/ to models/."""
        import json

        from dw.workflow import workflow_from_file

        (tmp_path / "workflows" / "templates").mkdir(parents=True)
        (tmp_path / "workflows" / "models").mkdir(parents=True)
        child = {
            "id": "child",
            "steps": [
                {
                    "name": "noop",
                    "task": {
                        "command": "get_dict_value",
                        "arguments": {"dict": {"k": 1}, "key": "k"},
                    },
                }
            ],
        }
        (tmp_path / "workflows" / "models" / "child.json").write_text(json.dumps(child))
        parent = {
            "id": "parent",
            "steps": [
                {
                    "name": "sub",
                    "workflow": {"path": "../models/child.json", "arguments": {}},
                }
            ],
        }
        parent_path = tmp_path / "workflows" / "templates" / "parent.json"
        parent_path.write_text(json.dumps(parent))
        workflow = workflow_from_file(str(parent_path), str(tmp_path))

        action = workflow.create_step_action(
            workflow.workflow_definition["steps"][0],
            shared_components={},
            previous_pipelines={},
            default_seed=42,
            device="cpu",
        )

        assert action is not None

    def test_a_file_outside_any_catalog_is_confined_to_its_own_directory(
        self, tmp_path
    ):
        """No 'workflows' ancestor at all - the referencing file's own
        directory is the confinement, so a sibling of a sibling is still
        refused even though the target file genuinely exists."""
        import json

        from dw.security import SecurityError
        from dw.workflow import workflow_from_file

        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        child = {"id": "child", "steps": []}
        (tmp_path / "b" / "child.json").write_text(json.dumps(child))
        parent = {
            "id": "parent",
            "steps": [
                {
                    "name": "sub",
                    "workflow": {"path": "../b/child.json", "arguments": {}},
                }
            ],
        }
        parent_path = tmp_path / "a" / "parent.json"
        parent_path.write_text(json.dumps(parent))
        workflow = workflow_from_file(str(parent_path), str(tmp_path))

        with pytest.raises(SecurityError):
            workflow.create_step_action(
                workflow.workflow_definition["steps"][0],
                shared_components={},
                previous_pipelines={},
                default_seed=42,
                device="cpu",
            )


def test_validate_reports_every_schema_error_at_once(tmp_path):
    """An agent iterating on a draft fixes all of them in one round trip."""
    from dw.workflow import Workflow

    definition = {
        "id": "bad",
        "variables": "not-an-object",
        "steps": [
            {
                "name": "gen",
                "seed": "not-a-number",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}"},
                    "from_pretrained_arguments": {"model_name": "m"},
                    "arguments": {},
                },
            }
        ],
    }
    with pytest.raises(Exception) as exc_info:
        Workflow(definition, str(tmp_path), "").validate()
    message = str(exc_info.value)
    assert message.startswith("Validation errors (2):")
    assert "  at steps[0].seed:" in message
    assert "  at variables:" in message
    assert message.count("Validation error") == 1


def test_validate_catches_a_reference_to_a_step_that_does_not_exist(tmp_path):
    """T005: a dangling 'previous_result:' used to validate clean and fail
    at run time, after every step before it had run - 42 minutes of
    generation, in the case that prompted this."""
    from dw.workflow import Workflow

    definition = {
        "id": "renamed",
        "steps": [
            {
                "name": "shot_1",
                "task": {"command": "compose_text", "arguments": {"parts": ["a"]}},
            },
            {
                "name": "episode",
                "task": {
                    "command": "concat_videos",
                    "arguments": {
                        "videos": [
                            "previous_result:shot_1",
                            "previous_result:shot_2_pat_deflects",
                        ]
                    },
                },
            },
        ],
    }
    with pytest.raises(Exception) as exc_info:
        Workflow(definition, str(tmp_path), "").validate()
    message = str(exc_info.value)
    assert "steps[1].task.arguments.videos[1]" in message
    assert "shot_2_pat_deflects" in message
    assert message.count("Validation error") == 1


def _workflow_from(definition, tmp_path):
    return Workflow(definition, str(tmp_path), "")


def _for_each_workflow(**overrides):
    definition = {
        "id": "fe",
        "variables": {
            "shots": [{"name": "a", "text": "A"}, {"name": "b", "text": "B"}]
        },
        "steps": [
            {
                "name": "shot",
                "for_each": "variable:shots",
                "task": {
                    "command": "compose_text",
                    "arguments": {"parts": ["item:text"]},
                },
                "result": {"content_type": "text/plain"},
            },
            {
                "name": "edit",
                "task": {
                    "command": "compose_text",
                    "arguments": {"parts": "gather:shot"},
                },
                "result": {"content_type": "text/plain"},
            },
        ],
    }
    definition.update(overrides)
    return definition


def test_validation_expands_for_each_before_the_reference_check(tmp_path):
    workflow = _workflow_from(_for_each_workflow(), tmp_path)  # the file's helper
    assert workflow.validation_errors() == []


def test_validation_reports_a_for_each_error_at_its_path(tmp_path):
    definition = _for_each_workflow()
    definition["steps"][1]["task"]["arguments"]["parts"] = "previous_result:shot"
    workflow = _workflow_from(definition, tmp_path)
    errors = workflow.validation_errors()
    assert len(errors) == 1
    assert errors[0]["path"] == "steps[1].task.arguments.parts"
    assert "gather:shot" in errors[0]["message"]


def test_validation_expands_the_callers_list_not_the_default(tmp_path):
    workflow = _workflow_from(_for_each_workflow(), tmp_path)
    # Two entries share a name only in the caller's list
    errors = workflow.validation_errors(
        arguments={"shots": [{"name": "a"}, {"name": "a"}]}
    )
    assert errors and "Duplicate entry name 'a'" in errors[0]["message"]


def test_validation_falls_back_to_the_default_when_the_arguments_are_bad(tmp_path):
    workflow = _workflow_from(_for_each_workflow(), tmp_path)
    # An undeclared argument is argument_errors' finding, not validation's
    assert workflow.validation_errors(arguments={"nope": 1}) == []


def test_validation_of_an_undeclared_variable_still_does_not_raise(tmp_path):
    definition = _for_each_workflow()
    definition["steps"][0]["for_each"] = "variable:missing"
    workflow = _workflow_from(definition, tmp_path)
    errors = workflow.validation_errors()
    assert errors and "variable:missing" in errors[0]["message"]
    assert errors[0]["path"] == "steps[0].for_each"


def test_an_unrelated_undeclared_variable_is_not_blamed_on_for_each(tmp_path):
    """A typo in one step used to leave the whole definition unsubstituted,
    so a perfectly good for_each list arrived as the literal
    'variable:shots' and the error blamed the list the author got right."""
    definition = _for_each_workflow()
    definition["steps"][0]["task"]["arguments"]["parts"] = [
        "item:text",
        "variable:promt",
    ]
    errors = _workflow_from(definition, tmp_path).validation_errors()
    assert len(errors) == 1
    assert errors[0]["path"] == "steps[0].task.arguments.parts[1]"
    assert "promt" in errors[0]["message"] and "shots" in errors[0]["message"]
    assert "for_each" not in errors[0]["message"]


def test_a_reference_error_after_a_group_names_the_step_in_the_file(tmp_path):
    """The expansion moves the later steps along; the path the author reads
    has to be the one in the file they wrote, not the expanded index."""
    definition = _for_each_workflow()
    definition["variables"]["shots"] = [
        {"name": "a", "text": "A"},
        {"name": "b", "text": "B"},
        {"name": "c", "text": "C"},
    ]
    definition["steps"].insert(
        0,
        {
            "name": "draw",
            "task": {"command": "compose_text", "arguments": {"parts": ["x"]}},
            "result": {"content_type": "text/plain"},
        },
    )
    definition["steps"][2]["task"]["arguments"]["parts"] = ["previous_result:nope"]
    errors = _workflow_from(definition, tmp_path).validation_errors()
    assert len(errors) == 1
    assert errors[0]["path"] == "steps[2].task.arguments.parts[0]"


def test_a_reference_error_inside_a_member_names_the_member(tmp_path):
    definition = _for_each_workflow()
    definition["steps"][0]["task"]["arguments"]["parts"] = [
        "item:text",
        {"from_previous_result": "nope"},
    ]
    errors = _workflow_from(definition, tmp_path).validation_errors()
    # One per member, each at the source step's path
    assert [e["path"] for e in errors] == [
        "steps[0].task.arguments.parts[1].from_previous_result",
        "steps[0].task.arguments.parts[1].from_previous_result",
    ]
    assert "in member 'shot@a'" in errors[0]["message"]
    assert "in member 'shot@b'" in errors[1]["message"]


def test_validation_realizes_a_constant_default_list(tmp_path):
    """A list defaulted to a constant: name used to fail validation with the
    string unsubstituted and then run fine, since only the run realized
    constants."""
    definition = _for_each_workflow()
    definition["variables"]["shots"] = "constant:tests.test_workflow.CONSTANT_SHOTS"
    workflow = _workflow_from(definition, tmp_path)
    assert workflow.validation_errors() == []
    assert [s["name"] for s in workflow.expanded_definition()["steps"]][:2] == [
        "shot@a",
        "shot@b",
    ]


def test_run_expands_for_each_and_names_the_members(tmp_path):
    workflow = _workflow_from(_for_each_workflow(seed=1), tmp_path)
    workflow.run({})
    names = [entry["step"] for entry in workflow.manifest]
    assert names == ["shot@a", "shot@b", "edit"]


def test_run_substitutes_the_callers_list(tmp_path):
    workflow = _workflow_from(_for_each_workflow(seed=1), tmp_path)
    workflow.run({"shots": [{"name": "only", "text": "X"}]})
    names = [entry["step"] for entry in workflow.manifest]
    assert names == ["shot@only", "edit"]


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


def test_a_variable_cycle_is_a_validation_error_at_variables(tmp_path):
    definition = _for_each_workflow()
    definition["variables"] = {
        "a": [{"x": "variable:b"}],
        "b": [{"y": "variable:a"}],
    }
    workflow = _workflow_from(definition, tmp_path)

    errors = workflow.validation_errors()

    assert [e["path"] for e in errors] == ["variables"]
    assert "a -> b -> a" in errors[0]["message"]
