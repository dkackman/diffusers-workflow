"""
Unit tests for type_helpers module
Tests dynamic type loading and method checking
"""

import pytest
from dw.type_helpers import (
    get_type,
    load_type_from_name,
    load_type_from_full_name,
    has_method,
)


class TestGetType:
    """Test getting type from module"""

    def test_get_type_from_diffusers(self):
        # This would work if diffusers is installed
        # For testing, we'll use a built-in type
        import sys

        result = get_type("sys", "version")
        assert result is not None

    def test_get_type_invalid_module(self):
        with pytest.raises(ModuleNotFoundError):
            get_type("nonexistent_module", "SomeType")

    def test_get_type_invalid_attribute(self):
        with pytest.raises(AttributeError):
            get_type("sys", "NonExistentType")


class TestLoadTypeFromName:
    """Test loading type by name from diffusers"""

    def test_load_type_with_full_path(self):
        # Test with fully qualified name
        result = load_type_from_full_name("os.path.join")
        assert callable(result)
        assert result.__name__ == "join"

    def test_load_type_invalid_full_path(self):
        with pytest.raises(ModuleNotFoundError):
            load_type_from_full_name("fake.module.Type")

    def test_load_type_invalid_attribute_in_path(self):
        with pytest.raises(AttributeError):
            load_type_from_full_name("os.path.NonExistent")


class TestLoadTypeFromFullName:
    """Test loading type from fully qualified name"""

    def test_load_builtin_type(self):
        result = load_type_from_full_name("os.path.exists")
        assert callable(result)
        assert result.__name__ == "exists"

    def test_load_class_from_module(self):
        result = load_type_from_full_name("json.JSONDecoder")
        assert result.__name__ == "JSONDecoder"

    def test_load_nested_module(self):
        result = load_type_from_full_name("logging.handlers.RotatingFileHandler")
        assert result.__name__ == "RotatingFileHandler"


class TestHasMethod:
    """Test checking if object has callable method"""

    def test_has_method_true(self):
        class TestClass:
            def my_method(self):
                pass

        obj = TestClass()
        assert has_method(obj, "my_method") is True

    def test_has_method_false(self):
        class TestClass:
            my_attribute = "value"

        obj = TestClass()
        assert has_method(obj, "my_attribute") is False

    def test_has_method_nonexistent(self):
        class TestClass:
            pass

        obj = TestClass()
        assert has_method(obj, "nonexistent") is False

    def test_has_method_with_builtin(self):
        string = "test"
        assert has_method(string, "upper") is True
        assert has_method(string, "nonexistent") is False

    def test_has_method_with_property(self):
        class TestClass:
            @property
            def my_property(self):
                return "value"

        obj = TestClass()
        # Properties are not callable
        assert has_method(obj, "my_property") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
