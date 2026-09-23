import json
import os

from dw.result_fps import fps_errors

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAINED_SEGMENTS = os.path.join(
    REPO_ROOT, "workflows", "templates", "ltx2", "chained-segments.json"
)


def _step(name, result=None, **extra):
    step = {"name": name, "task": {"command": "gather_images", "arguments": {}}}
    if result is not None:
        step["result"] = result
    step.update(extra)
    return step


class TestFpsErrors:
    def test_a_clean_definition_has_none(self):
        definition = {
            "steps": [
                _step("a", {"content_type": "video/mp4", "fps": 24}),
                _step("b", {"content_type": "video/mp4"}),
            ]
        }
        assert fps_errors(definition) == []

    def test_a_whole_number_float_is_fine(self):
        # a rate variable declared as JSON 24.0 still carries no fraction
        definition = {"steps": [_step("a", {"fps": 24.0})]}
        assert fps_errors(definition) == []

    def test_a_fractional_fps_is_reported_at_its_path(self):
        definition = {"steps": [_step("a", {"fps": 23.976})]}
        errors = fps_errors(definition)
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].result.fps"
        assert "whole number" in errors[0]["message"]

    def test_a_non_positive_fps_is_reported(self):
        definition = {"steps": [_step("a", {"fps": 0})]}
        errors = fps_errors(definition)
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].result.fps"
        assert "greater than zero" in errors[0]["message"]

    def test_a_non_numeric_fps_is_reported(self):
        definition = {"steps": [_step("a", {"fps": "not-a-reference"})]}
        errors = fps_errors(definition)
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].result.fps"
        assert "number of frames" in errors[0]["message"]

    def test_a_boolean_fps_is_reported(self):
        definition = {"steps": [_step("a", {"fps": True})]}
        errors = fps_errors(definition)
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].result.fps"

    def test_an_expanded_member_reports_the_source_step_and_names_the_member(self):
        definition = {
            "steps": [
                _step("intro", {"fps": 24}),
                _step("shot@a", {"fps": 24}),
                _step("shot@b", {"fps": 0}),
            ]
        }
        errors = fps_errors(definition, source_indices=[0, 1, 1])
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[1].result.fps"
        assert "shot@b" in errors[0]["message"]

    def test_an_unsubstituted_reference_is_left_alone(self):
        definition = {"steps": [_step("a", {"fps": "variable:frame_rate"})]}
        assert fps_errors(definition) == []
        definition = {"steps": [_step("a", {"fps": "item:frame_rate"})]}
        assert fps_errors(definition) == []

    def test_no_steps_is_fine(self):
        assert fps_errors({}) == []
        assert fps_errors({"steps": "nope"}) == []


class TestValidationErrorsIntegration:
    def _workflow(self, definition, tmp_path):
        from dw.workflow import Workflow

        return Workflow(definition, str(tmp_path), "/w/workflows/Fps.json")

    def test_a_variable_driven_fps_resolves(self, tmp_path):
        definition = {
            "id": "fps_test",
            "variables": {"frame_rate": 24},
            "steps": [
                {
                    "name": "a",
                    "task": {"command": "gather_images", "arguments": {}},
                    "result": {
                        "content_type": "video/mp4",
                        "fps": "variable:frame_rate",
                    },
                }
            ],
        }
        workflow = self._workflow(definition, tmp_path)
        assert workflow.validation_errors() == []

    def test_a_bad_resolved_fps_is_refused_at_validate_time(self, tmp_path):
        definition = {
            "id": "fps_test",
            # float-typed, like the ltx2 templates' own frame_rate default -
            # an integer-typed variable would itself truncate 23.976 to 23
            # before fps_errors ever saw a fractional value
            "variables": {"frame_rate": 24.0},
            "steps": [
                {
                    "name": "a",
                    "task": {"command": "gather_images", "arguments": {}},
                    "result": {
                        "content_type": "video/mp4",
                        "fps": "variable:frame_rate",
                    },
                }
            ],
        }
        workflow = self._workflow(definition, tmp_path)
        errors = workflow.validation_errors(arguments={"frame_rate": 23.976})
        assert [e["path"] for e in errors] == ["steps[0].result.fps"]

    def test_an_item_driven_fps_is_checked_per_member(self, tmp_path):
        definition = {
            "id": "fps_test",
            "steps": [
                {
                    "name": "shot",
                    "for_each": [
                        {"name": "a", "rate": 24},
                        {"name": "b", "rate": -1},
                    ],
                    "task": {"command": "gather_images", "arguments": {}},
                    "result": {"content_type": "video/mp4", "fps": "item:rate"},
                }
            ],
        }
        errors = self._workflow(definition, tmp_path).validation_errors()
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].result.fps"
        assert "shot@b" in errors[0]["message"]

    def test_an_integer_literal_fps_is_unchanged(self, tmp_path):
        definition = {
            "id": "fps_test",
            "steps": [
                {
                    "name": "a",
                    "task": {"command": "gather_images", "arguments": {}},
                    "result": {"content_type": "video/mp4", "fps": 8},
                }
            ],
        }
        workflow = self._workflow(definition, tmp_path)
        assert workflow.validation_errors() == []

    def test_chained_segments_honors_a_frame_rate_argument(self, tmp_path):
        # the real #363 repro: chained-segments.json's result.fps is
        # "variable:frame_rate" - a caller setting frame_rate must see that
        # value reach the resolved step, not the template's own default
        definition = json.load(open(CHAINED_SEGMENTS, encoding="utf-8"))
        workflow = self._workflow(definition, tmp_path)
        assert workflow.validation_errors(arguments={"frame_rate": 30}) == []

        expanded = workflow.expanded_definition(arguments={"frame_rate": 30})
        step = next(
            s for s in expanded["steps"] if s["name"] == "chained_image_to_video"
        )
        assert step["result"]["fps"] == 30

    def test_chained_segments_default_run_is_unchanged(self, tmp_path):
        definition = json.load(open(CHAINED_SEGMENTS, encoding="utf-8"))
        workflow = self._workflow(definition, tmp_path)
        assert workflow.validation_errors() == []

        expanded = workflow.expanded_definition()
        step = next(
            s for s in expanded["steps"] if s["name"] == "chained_image_to_video"
        )
        assert step["result"]["fps"] == 24.0
