"""
Unit tests for result module
Tests result storage, artifact management, and file saving
"""
import pytest
import os
import tempfile
import json
from PIL import Image
from dw.result import Result, get_artifact_list, guess_extension


class TestResult:
    """Test Result class functionality"""
    
    def test_add_single_result(self):
        result = Result({})
        result.add_result("test_value")
        assert result.result_list == ["test_value"]
    
    def test_add_list_result(self):
        result = Result({})
        result.add_result(["item1", "item2", "item3"])
        assert result.result_list == ["item1", "item2", "item3"]
    
    def test_add_string_strips_quotes(self):
        result = Result({})
        result.add_result('"test_string"  ')
        assert result.result_list == ["test_string"]
    
    def test_get_artifacts_from_simple_list(self):
        result = Result({})
        result.add_result(["item1", "item2"])
        artifacts = result.get_artifacts()
        assert artifacts == ["item1", "item2"]
    
    def test_get_artifact_properties(self):
        result = Result({})
        result.add_result([
            {"prop1": "value1", "prop2": "data1"},
            {"prop1": "value2", "prop2": "data2"}
        ])
        
        prop_values = result.get_artifact_properties("prop1")
        assert prop_values == ["value1", "value2"]
    
    def test_get_artifact_properties_missing_key(self):
        result = Result({})
        result.add_result([
            {"prop1": "value1"},
            {"prop2": "value2"}  # Missing prop1
        ])
        
        prop_values = result.get_artifact_properties("prop1")
        assert len(prop_values) == 1
        assert prop_values == ["value1"]
    
    def test_save_json_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_def = {"content_type": "application/json", "save": True}
            result = Result(result_def)
            result.add_result({"key": "value", "number": 42})
            
            result.save(temp_dir, "test_output")
            
            output_file = os.path.join(temp_dir, "test_output-0.json")
            assert os.path.exists(output_file)
            
            with open(output_file, "r") as f:
                loaded = json.load(f)
                assert loaded == {"key": "value", "number": 42}
    
    def test_save_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_def = {"content_type": "application/json", "save": False}
            result = Result(result_def)
            result.add_result({"key": "value"})
            
            result.save(temp_dir, "test_output")
            
            # Should not create any files
            files = os.listdir(temp_dir)
            assert len(files) == 0
    
    def test_save_no_content_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_def = {"save": True}  # No content_type
            result = Result(result_def)
            result.add_result("test")
            
            result.save(temp_dir, "test_output")
            
            # Should not create any files
            files = os.listdir(temp_dir)
            assert len(files) == 0
    
    def test_save_with_custom_base_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_def = {
                "content_type": "application/json",
                "save": True,
                "file_base_name": "custom_"
            }
            result = Result(result_def)
            result.add_result({"data": "test"})
            
            result.save(temp_dir, "output")
            
            output_file = os.path.join(temp_dir, "custom_output-0.json")
            assert os.path.exists(output_file)
    
    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "nested", "path")
            result_def = {"content_type": "application/json", "save": True}
            result = Result(result_def)
            result.add_result({"data": "test"})
            
            result.save(nested_dir, "test_output")
            
            assert os.path.exists(nested_dir)
            output_file = os.path.join(nested_dir, "test_output-0.json")
            assert os.path.exists(output_file)


class TestGetArtifactList:
    """Test artifact extraction from various result types"""
    
    def test_result_with_images(self):
        class MockResult:
            images = ["img1", "img2", "img3"]
        
        artifacts = get_artifact_list(MockResult())
        assert artifacts == ["img1", "img2", "img3"]
    
    def test_result_with_frames(self):
        class MockResult:
            frames = ["frame1", "frame2"]
        
        artifacts = get_artifact_list(MockResult())
        assert artifacts == ["frame1", "frame2"]
    
    def test_result_with_list(self):
        result = ["item1", "item2"]
        artifacts = get_artifact_list(result)
        assert artifacts == ["item1", "item2"]
    
    def test_result_single_item(self):
        result = "single_item"
        artifacts = get_artifact_list(result)
        assert artifacts == ["single_item"]
    
    def test_result_with_image_embeds(self):
        class MockResult:
            image_embeds = ["embed1", "embed2"]
        
        artifacts = get_artifact_list(MockResult())
        assert artifacts == ["embed1", "embed2"]


class TestGuessExtension:
    """Test MIME type to file extension conversion"""
    
    def test_image_jpeg(self):
        assert guess_extension("image/jpeg") == ".jpg"
    
    def test_image_png(self):
        assert guess_extension("image/png") == ".png"
    
    def test_application_json(self):
        assert guess_extension("application/json") == ".json"
    
    def test_audio_wav(self):
        # Special case handling
        assert guess_extension("audio/wav") == ".wav"
    
    def test_video_mp4(self):
        ext = guess_extension("video/mp4")
        assert ext in [".mp4", ".mpg4"]  # Both are valid
    
    def test_unknown_type(self):
        result = guess_extension("unknown/type")
        # Should return empty string or None for unknown types
        assert result in ["", None]
    
    def test_none_content_type(self):
        assert guess_extension(None) == ""
    
    def test_empty_content_type(self):
        assert guess_extension("") == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
