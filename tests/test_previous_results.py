"""
Unit tests for previous_results module
Tests cartesian product generation and result reference handling
"""

import pytest
from dw.previous_results import (
    get_iterations,
    get_previous_results,
    find_previous_result_refs,
)
from dw.result import Result
from PIL import Image


class TestFindPreviousResultRefs:
    """Test finding result references in argument templates"""

    def test_find_single_reference(self):
        arguments = {"image": "previous_result:step1"}
        refs = find_previous_result_refs(arguments)
        assert refs == {("image",): "step1"}

    def test_find_multiple_references(self):
        arguments = {
            "image": "previous_result:step1",
            "prompt": "previous_result:step2",
        }
        refs = find_previous_result_refs(arguments)
        assert refs == {("image",): "step1", ("prompt",): "step2"}

    def test_find_property_reference(self):
        arguments = {"text": "previous_result:step1.output"}
        refs = find_previous_result_refs(arguments)
        assert refs == {("text",): "step1.output"}

    def test_no_references(self):
        arguments = {"image": "path/to/image.jpg", "prompt": "test"}
        refs = find_previous_result_refs(arguments)
        assert refs == {}

    def test_mixed_references(self):
        arguments = {
            "image": "previous_result:step1",
            "static_value": "not_a_reference",
            "number": 42,
        }
        refs = find_previous_result_refs(arguments)
        assert refs == {("image",): "step1"}


class TestGetPreviousResults:
    """Test retrieving results from previous steps"""

    def test_get_all_artifacts(self):
        result = Result({})
        result.add_result(["item1", "item2", "item3"])
        previous_results = {"step1": result}

        artifacts = get_previous_results(previous_results, "step1")
        assert artifacts == ["item1", "item2", "item3"]

    def test_get_specific_property(self):
        result = Result({})
        result.add_result(
            [{"text": "first", "value": 1}, {"text": "second", "value": 2}]
        )
        previous_results = {"step1": result}

        texts = get_previous_results(previous_results, "step1.text")
        assert texts == ["first", "second"]

    def test_missing_result_raises_error(self):
        previous_results = {"step1": Result({})}

        with pytest.raises(KeyError) as exc_info:
            get_previous_results(previous_results, "step_missing")

        assert "step_missing" in str(exc_info.value)
        assert "Available results" in str(exc_info.value)

    def test_missing_result_with_property_raises_error(self):
        previous_results = {"step1": Result({})}

        with pytest.raises(KeyError) as exc_info:
            get_previous_results(previous_results, "step_missing.text")

        assert "step_missing" in str(exc_info.value)
        assert "Available results" in str(exc_info.value)

    def test_dotted_step_name_with_property(self):
        """Step names may themselves contain dots (e.g. 'v1.0')."""
        result = Result({})
        result.add_result(
            [{"mask": "mask1.png", "value": 1}, {"mask": "mask2.png", "value": 2}]
        )
        previous_results = {"v1.0": result}

        masks = get_previous_results(previous_results, "v1.0.mask")
        assert masks == ["mask1.png", "mask2.png"]

    def test_dotted_step_name_without_property(self):
        """A dotted step name referenced with no property is not mis-split."""
        result = Result({})
        result.add_result(["item1", "item2"])
        previous_results = {"v1.0": result}

        artifacts = get_previous_results(previous_results, "v1.0")
        assert artifacts == ["item1", "item2"]

    def test_longest_matching_step_name_wins(self):
        """When multiple known step names could be a prefix, prefer the longest."""
        short_result = Result({})
        short_result.add_result([{"text": "wrong"}])

        long_result = Result({})
        long_result.add_result([{"text": "right"}])

        previous_results = {"step": short_result, "step.sub": long_result}

        texts = get_previous_results(previous_results, "step.sub.text")
        assert texts == ["right"]

    def test_multi_dot_property_does_not_raise_unpack_error(self):
        """A property name containing dots should resolve the step name
        correctly and not raise a ValueError from a naive two-part split.
        The downstream property lookup treats "a.b" as a literal (missing)
        key and returns no values rather than erroring."""
        result = Result({})
        result.add_result([{"a": {"b": "nested"}}, {"a.b": "literal"}])
        previous_results = {"step": result}

        values = get_previous_results(previous_results, "step.a.b")
        assert values == ["literal"]


class TestGetIterations:
    """Test cartesian product generation for argument combinations"""

    def test_no_references_returns_single_iteration(self):
        template = {"prompt": "test", "num_steps": 25}
        previous_results = {}

        iterations = get_iterations(template, previous_results)
        assert len(iterations) == 1
        assert iterations[0] == {"prompt": "test", "num_steps": 25}

    def test_a_realized_prompt_reference_does_not_expand_iterations(
        self, tmp_path, monkeypatch
    ):
        # A 'prompt:' reference is resolved to plain text by realize_args
        # before iteration expansion ever sees the template - one string,
        # one run, never a cartesian factor
        import json
        from dw.arguments import realize_args

        (tmp_path / "scenic.json").write_text(json.dumps({"text": "a landscape"}))
        monkeypatch.setenv("DW_PROMPT_DIR", str(tmp_path))

        template = {"prompt": "prompt:scenic", "num_steps": 25}
        realize_args(template)
        iterations = get_iterations(template, {})
        assert len(iterations) == 1
        assert iterations[0] == {"prompt": "a landscape", "num_steps": 25}

    def test_single_reference_expands(self):
        result = Result({})
        result.add_result(["img1.jpg", "img2.jpg"])
        previous_results = {"step1": result}

        template = {"image": "previous_result:step1", "prompt": "test"}
        iterations = get_iterations(template, previous_results)

        assert len(iterations) == 2
        assert iterations[0] == {"image": "img1.jpg", "prompt": "test"}
        assert iterations[1] == {"image": "img2.jpg", "prompt": "test"}

    def test_multiple_references_create_cartesian_product(self):
        result1 = Result({})
        result1.add_result(["img1.jpg", "img2.jpg"])

        result2 = Result({})
        result2.add_result(["prompt1", "prompt2"])

        previous_results = {"images": result1, "prompts": result2}

        template = {
            "image": "previous_result:images",
            "prompt": "previous_result:prompts",
        }
        iterations = get_iterations(template, previous_results)

        # Should create 2x2 = 4 combinations
        assert len(iterations) == 4
        assert {"image": "img1.jpg", "prompt": "prompt1"} in iterations
        assert {"image": "img1.jpg", "prompt": "prompt2"} in iterations
        assert {"image": "img2.jpg", "prompt": "prompt1"} in iterations
        assert {"image": "img2.jpg", "prompt": "prompt2"} in iterations

    def test_property_reference(self):
        result = Result({})
        result.add_result(
            [{"text": "first", "value": 1}, {"text": "second", "value": 2}]
        )
        previous_results = {"step1": result}

        template = {"prompt": "previous_result:step1.text"}
        iterations = get_iterations(template, previous_results)

        assert len(iterations) == 2
        assert iterations[0]["prompt"] == "first"
        assert iterations[1]["prompt"] == "second"

    def test_list_template_returns_as_is(self):
        template = [{"prompt": "test1"}, {"prompt": "test2"}]
        previous_results = {}

        iterations = get_iterations(template, previous_results)
        assert iterations == template

    def test_three_way_cartesian_product(self):
        """Test with 3 dimensions: 2x2x2 = 8 combinations"""
        result1 = Result({})
        result1.add_result(["a", "b"])

        result2 = Result({})
        result2.add_result(["x", "y"])

        result3 = Result({})
        result3.add_result([1, 2])

        previous_results = {"r1": result1, "r2": result2, "r3": result3}

        template = {
            "param1": "previous_result:r1",
            "param2": "previous_result:r2",
            "param3": "previous_result:r3",
        }
        iterations = get_iterations(template, previous_results)

        assert len(iterations) == 8
        # Verify all combinations exist
        combinations = [(it["param1"], it["param2"], it["param3"]) for it in iterations]
        assert ("a", "x", 1) in combinations
        assert ("b", "y", 2) in combinations

    def test_nested_values_shared_not_copied_no_references(self):
        """Iteration dicts are shallow copies: nested values (e.g. loaded
        PIL images or video frame lists already realized into the template)
        must be the *same* object, not deep-copied, to avoid multiplying
        media memory usage."""
        marker = ["frame1", "frame2"]
        template = {"prompt": "test", "marker": marker}
        previous_results = {}

        iterations = get_iterations(template, previous_results)
        assert len(iterations) == 1
        assert iterations[0]["marker"] is marker

    def test_nested_values_shared_not_copied_with_references(self):
        """Same sharing contract applies to the cartesian-product path."""
        result = Result({})
        result.add_result(["img1.jpg", "img2.jpg"])
        previous_results = {"step1": result}

        marker = ["frame1", "frame2"]
        template = {"image": "previous_result:step1", "marker": marker}
        iterations = get_iterations(template, previous_results)

        assert len(iterations) == 2
        assert iterations[0]["marker"] is marker
        assert iterations[1]["marker"] is marker
        assert iterations[0]["marker"] is iterations[1]["marker"]

    def test_top_level_mutation_is_independent_per_iteration(self):
        """Popping/assigning a top-level key on one iteration must not leak
        to other iterations or back into the original template (matches how
        task handlers pop 'image'/'device' and pipeline.py pops 'prompt')."""
        result = Result({})
        result.add_result(["img1.jpg", "img2.jpg"])
        previous_results = {"step1": result}

        template = {"image": "previous_result:step1", "device": "cuda"}
        iterations = get_iterations(template, previous_results)

        iterations[0].pop("device")
        iterations[0]["prompt_embeds"] = "embedded"

        assert "device" not in iterations[0]
        assert "prompt_embeds" in iterations[0]
        assert iterations[1]["device"] == "cuda"
        assert "prompt_embeds" not in iterations[1]
        assert "device" in template
        assert "prompt_embeds" not in template


class ImageReference:
    """Stands in for MiniMaxH3ImageReference - a reference built from an image"""

    kind = "image"

    def __init__(self, image):
        self.image = image


class TestNestedReferences:
    """A reference nested inside an argument, which is where object descriptions live"""

    def test_a_nested_value_is_found_by_its_path(self):
        arguments = {
            "references": [
                {"reference_type": ImageReference, "from_previous_result": "draw"}
            ]
        }

        refs = find_previous_result_refs(arguments)

        assert refs == {("references", 0, "from_previous_result"): "draw"}

    def test_a_prefixed_value_is_found_at_any_depth(self):
        arguments = {"nested": {"prompt": "previous_result:write"}}

        refs = find_previous_result_refs(arguments)

        assert refs == {("nested", "prompt"): "write"}

    def test_the_object_is_built_from_the_step_output(self):
        image = Image.new("RGB", (8, 8))
        result = Result({})
        result.add_result([image])

        template = {
            "prompt": "test",
            "references": [
                {"reference_type": ImageReference, "from_previous_result": "draw"}
            ],
        }
        iterations = get_iterations(template, {"draw": result})

        assert len(iterations) == 1
        reference = iterations[0]["references"][0]
        assert isinstance(reference, ImageReference)
        assert reference.image is image

    def test_each_artifact_becomes_its_own_iteration(self):
        result = Result({})
        result.add_result([Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))])

        template = {
            "references": [
                {"reference_type": ImageReference, "from_previous_result": "draw"}
            ]
        }
        iterations = get_iterations(template, {"draw": result})

        assert len(iterations) == 2
        first, second = (i["references"][0] for i in iterations)
        assert first.image is not second.image

    def test_the_template_is_left_intact_for_the_next_iteration(self):
        result = Result({})
        result.add_result([Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))])

        description = {"reference_type": ImageReference, "from_previous_result": "draw"}
        template = {"references": [description]}
        get_iterations(template, {"draw": result})

        # Substituting into a copy is what lets the second iteration read the
        # description again - substituting in place would leave it holding the
        # first iteration's image
        assert template["references"][0] is description
        assert description["from_previous_result"] == "draw"

    def test_siblings_are_shared_rather_than_copied(self):
        result = Result({})
        result.add_result(["first", "second"])

        frames = [object()]
        template = {"video": frames, "prompt": "previous_result:write"}
        iterations = get_iterations(template, {"write": result})

        # Only the containers on the path to the substitution are copied - a
        # deep copy would duplicate the media the iterations mean to share
        assert iterations[0]["video"] is frames
        assert iterations[1]["video"] is frames


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
