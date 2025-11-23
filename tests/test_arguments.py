"""
Unit tests for arguments module
Tests argument realization, image/video fetching, and type loading
"""

import pytest
import os
import tempfile
from PIL import Image
from unittest.mock import patch, MagicMock
from dw.arguments import realize_args, fetch_image, fetch_video
from dw.security import SecurityError


class TestFetchImage:
    """Test image fetching from files and URLs"""

    def test_fetch_none_returns_none(self):
        assert fetch_image(None) is None

    def test_fetch_image_invalid_type(self):
        with pytest.raises(ValueError) as exc_info:
            fetch_image(123)
        assert "must be a string" in str(exc_info.value)

    def test_fetch_image_dict_format(self):
        """Test that image can be specified as dict with 'location' key"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_image = Image.new("RGB", (100, 100), color="blue")
            image_path = os.path.join(temp_dir, "test.png")
            test_image.save(image_path)

            # Test dict format
            loaded_image = fetch_image({"location": image_path})
            assert isinstance(loaded_image, Image.Image)
            assert loaded_image.size == (100, 100)

    def test_fetch_image_dict_missing_location(self):
        """Test that dict without 'location' key raises error"""
        with pytest.raises(ValueError) as exc_info:
            fetch_image({"invalid": "key"})
        assert "location" in str(exc_info.value).lower()

    def test_fetch_image_from_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test image
            test_image = Image.new("RGB", (100, 100), color="red")
            image_path = os.path.join(temp_dir, "test.jpg")
            test_image.save(image_path)

            # Fetch it
            loaded_image = fetch_image(image_path)
            assert isinstance(loaded_image, Image.Image)
            assert loaded_image.size == (100, 100)

    @patch("dw.arguments.load_image")
    @patch("dw.arguments.validate_url")
    def test_fetch_image_from_url(self, mock_validate_url, mock_load_image):
        mock_validate_url.return_value = "https://example.com/image.jpg"
        mock_image = Image.new("RGB", (100, 100))
        mock_load_image.return_value = mock_image

        result = fetch_image("https://example.com/image.jpg")

        mock_validate_url.assert_called_once_with("https://example.com/image.jpg")
        mock_load_image.assert_called_once_with("https://example.com/image.jpg")
        assert result == mock_image

    def test_fetch_image_invalid_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with invalid extension
            invalid_file = os.path.join(temp_dir, "test.txt")
            with open(invalid_file, "w") as f:
                f.write("not an image")

            with pytest.raises(SecurityError) as exc_info:
                fetch_image(invalid_file)
            assert "extension not allowed" in str(exc_info.value)

    def test_fetch_image_path_traversal(self):
        with pytest.raises(SecurityError):
            fetch_image("../../../etc/passwd")


class TestFetchVideo:
    """Test video fetching from files and URLs"""

    def test_fetch_none_returns_none(self):
        assert fetch_video(None) is None

    def test_fetch_video_invalid_type(self):
        with pytest.raises(ValueError) as exc_info:
            fetch_video(456)
        assert "must be a string" in str(exc_info.value)

    def test_fetch_video_dict_format(self):
        """Test that video can be specified as dict with 'location' key"""
        with patch("dw.arguments.load_video") as mock_load:
            with patch("dw.arguments.validate_url") as mock_validate:
                mock_validate.return_value = "https://example.com/video.mp4"
                mock_load.return_value = ["frame1", "frame2"]

                result = fetch_video({"location": "https://example.com/video.mp4"})
                assert result == ["frame1", "frame2"]

    def test_fetch_video_dict_missing_location(self):
        """Test that dict without 'location' key raises error"""
        with pytest.raises(ValueError) as exc_info:
            fetch_video({"url": "test.mp4"})
        assert "location" in str(exc_info.value).lower()

    @patch("dw.arguments.load_video")
    @patch("dw.arguments.validate_url")
    def test_fetch_video_from_url(self, mock_validate_url, mock_load_video):
        mock_validate_url.return_value = "https://example.com/video.mp4"
        mock_frames = ["frame1", "frame2"]
        mock_load_video.return_value = mock_frames

        result = fetch_video("https://example.com/video.mp4")

        mock_validate_url.assert_called_once_with("https://example.com/video.mp4")
        mock_load_video.assert_called_once_with("https://example.com/video.mp4")
        assert result == mock_frames

    def test_fetch_video_invalid_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with invalid extension
            invalid_file = os.path.join(temp_dir, "test.txt")
            with open(invalid_file, "w") as f:
                f.write("not a video")

            with pytest.raises(SecurityError) as exc_info:
                fetch_video(invalid_file)
            assert "extension not allowed" in str(exc_info.value)


class TestRealizeArgs:
    """Test argument realization and type loading"""

    def test_realize_dict_with_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            test_image = Image.new("RGB", (50, 50), color="blue")
            image_path = os.path.join(temp_dir, "test.png")
            test_image.save(image_path)

            args = {"image": image_path}
            realize_args(args)

            assert isinstance(args["image"], Image.Image)

    def test_realize_dict_with_input_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_image = Image.new("RGB", (50, 50), color="green")
            image_path = os.path.join(temp_dir, "input.jpg")
            test_image.save(image_path)

            args = {"input_image": image_path}
            realize_args(args)

            assert isinstance(args["input_image"], Image.Image)

    @patch("dw.arguments.load_type_from_name")
    def test_realize_type_reference(self, mock_load_type):
        mock_type = type("MockType", (), {})
        mock_load_type.return_value = mock_type

        args = {"scheduler_type": "DDPMScheduler"}
        realize_args(args)

        mock_load_type.assert_called_once_with("DDPMScheduler")
        assert args["scheduler_type"] == mock_type

    def test_realize_escaped_type_reference(self):
        # Type references wrapped in {} should be unescaped
        args = {"category_type": "{clothing}"}
        realize_args(args)

        assert args["category_type"] == "clothing"

    def test_realize_content_type_not_converted(self):
        # content_type should not be treated as a type reference
        args = {"content_type": "image/jpeg"}
        realize_args(args)

        assert args["content_type"] == "image/jpeg"

    def test_realize_nested_dict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_image = Image.new("RGB", (50, 50))
            image_path = os.path.join(temp_dir, "nested.jpg")
            test_image.save(image_path)

            args = {"outer": {"inner": {"image": image_path}}}
            realize_args(args)

            assert isinstance(args["outer"]["inner"]["image"], Image.Image)

    def test_realize_list_of_dicts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            img1 = Image.new("RGB", (50, 50))
            img2 = Image.new("RGB", (50, 50))
            path1 = os.path.join(temp_dir, "img1.jpg")
            path2 = os.path.join(temp_dir, "img2.jpg")
            img1.save(path1)
            img2.save(path2)

            args = [{"image": path1}, {"image": path2}]
            realize_args(args)

            assert isinstance(args[0]["image"], Image.Image)
            assert isinstance(args[1]["image"], Image.Image)

    def test_realize_already_loaded_type(self):
        # If value is already a type, leave it as-is
        mock_type = type("AlreadyLoaded", (), {})
        args = {"scheduler_type": mock_type}
        realize_args(args)

        assert args["scheduler_type"] == mock_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
