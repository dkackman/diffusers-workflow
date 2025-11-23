"""
Unit tests for previous_results module
Tests cartesian product generation and result reference handling
"""
import pytest
from dw.previous_results import (
    get_iterations, get_previous_results, find_previous_result_refs
)
from dw.result import Result


class TestFindPreviousResultRefs:
    """Test finding result references in argument templates"""
    
    def test_find_single_reference(self):
        arguments = {"image": "previous_result:step1"}
        refs = find_previous_result_refs(arguments)
        assert refs == {"image": "step1"}
    
    def test_find_multiple_references(self):
        arguments = {
            "image": "previous_result:step1",
            "prompt": "previous_result:step2"
        }
        refs = find_previous_result_refs(arguments)
        assert refs == {"image": "step1", "prompt": "step2"}
    
    def test_find_property_reference(self):
        arguments = {"text": "previous_result:step1.output"}
        refs = find_previous_result_refs(arguments)
        assert refs == {"text": "step1.output"}
    
    def test_no_references(self):
        arguments = {"image": "path/to/image.jpg", "prompt": "test"}
        refs = find_previous_result_refs(arguments)
        assert refs == {}
    
    def test_mixed_references(self):
        arguments = {
            "image": "previous_result:step1",
            "static_value": "not_a_reference",
            "number": 42
        }
        refs = find_previous_result_refs(arguments)
        assert refs == {"image": "step1"}


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
        result.add_result([
            {"text": "first", "value": 1},
            {"text": "second", "value": 2}
        ])
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


class TestGetIterations:
    """Test cartesian product generation for argument combinations"""
    
    def test_no_references_returns_single_iteration(self):
        template = {"prompt": "test", "num_steps": 25}
        previous_results = {}
        
        iterations = get_iterations(template, previous_results)
        assert len(iterations) == 1
        assert iterations[0] == {"prompt": "test", "num_steps": 25}
    
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
            "prompt": "previous_result:prompts"
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
        result.add_result([
            {"text": "first", "value": 1},
            {"text": "second", "value": 2}
        ])
        previous_results = {"step1": result}
        
        template = {"prompt": "previous_result:step1.text"}
        iterations = get_iterations(template, previous_results)
        
        assert len(iterations) == 2
        assert iterations[0]["prompt"] == "first"
        assert iterations[1]["prompt"] == "second"
    
    def test_list_template_returns_as_is(self):
        template = [
            {"prompt": "test1"},
            {"prompt": "test2"}
        ]
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
            "param3": "previous_result:r3"
        }
        iterations = get_iterations(template, previous_results)
        
        assert len(iterations) == 8
        # Verify all combinations exist
        combinations = [
            (it["param1"], it["param2"], it["param3"]) 
            for it in iterations
        ]
        assert ("a", "x", 1) in combinations
        assert ("b", "y", 2) in combinations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
