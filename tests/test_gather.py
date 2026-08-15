"""
Unit tests for gather module
Tests image/video gathering from files and URLs
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image
from dw.tasks.gather import gather_images, gather_videos, gather_inputs
from dw.security import SecurityError


class TestGatherImages:
    """Test image gathering functionality"""

    def test_gather_images_from_files(self):
        """Test gathering images from file glob pattern"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test images
            img1 = Image.new("RGB", (50, 50), color="red")
            img2 = Image.new("RGB", (50, 50), color="blue")
            path1 = os.path.join(temp_dir, "img1.jpg")
            path2 = os.path.join(temp_dir, "img2.jpg")
            img1.save(path1)
            img2.save(path2)

            # Gather images
            glob_pattern = os.path.join(temp_dir, "*.jpg")
            images = gather_images(glob=glob_pattern)

            assert len(images) == 2
            assert all(isinstance(img, Image.Image) for img in images)

    @patch("dw.tasks.gather.load_image")
    @patch("dw.tasks.gather.validate_url")
    def test_gather_images_from_urls(self, mock_validate_url, mock_load_image):
        """Test gathering images from URLs"""
        mock_validate_url.side_effect = lambda x: x
        mock_image = Image.new("RGB", (100, 100))
        mock_load_image.return_value = mock_image

        urls = ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]

        images = gather_images(urls=urls)

        assert len(images) == 2
        assert mock_validate_url.call_count == 2
        assert mock_load_image.call_count == 2

    def test_gather_images_mixed_sources(self):
        """Test gathering images from both files and URLs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image file
            img1 = Image.new("RGB", (50, 50))
            path1 = os.path.join(temp_dir, "local.jpg")
            img1.save(path1)

            glob_pattern = os.path.join(temp_dir, "*.jpg")

            with patch("dw.tasks.gather.load_image") as mock_load:
                with patch("dw.tasks.gather.validate_url") as mock_validate:
                    mock_validate.return_value = "https://example.com/remote.jpg"
                    mock_load.side_effect = [
                        Image.new("RGB", (50, 50)),  # For file
                        Image.new("RGB", (100, 100)),  # For URL
                    ]

                    images = gather_images(
                        glob=glob_pattern, urls=["https://example.com/remote.jpg"]
                    )

                    # Should have images from both sources
                    assert len(images) >= 1

    def test_gather_images_no_results_raises_error(self):
        """Test that gathering no images raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            gather_images(glob="/nonexistent/*.jpg", urls=[])

        assert "No images found" in str(exc_info.value)

    def test_gather_images_invalid_url(self):
        """Test that invalid URLs are rejected"""
        with pytest.raises(SecurityError):
            gather_images(urls=["file:///etc/passwd"])

    def test_gather_images_disallowed_extension_raises(self, tmp_path):
        """A glob match whose extension isn't in the image allowlist must raise,
        not be silently loaded (this is what routes gather through fetch_image's
        validate_file_extension check)."""
        payload = tmp_path / "payload.svg"
        payload.write_text("<svg></svg>")

        glob_pattern = os.path.join(str(tmp_path), "*.svg")

        with pytest.raises(SecurityError):
            gather_images(glob=glob_pattern)

    def test_gather_images_traversal_glob_raises(self, tmp_path):
        """A glob pattern that walks outside its own directory via '..' must be
        rejected by validate_path, not silently followed to read the target file."""
        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()
        secret_dir = tmp_path / "secret"
        secret_dir.mkdir()
        Image.new("RGB", (10, 10), color="green").save(secret_dir / "target.png")

        # Literally contains '..' in the matched path, which is what
        # glob_lib.glob() hands back for a pattern like this
        traversal_pattern = os.path.join(str(workflow_dir), "..", "secret", "*.png")

        with pytest.raises(SecurityError):
            gather_images(glob=traversal_pattern)

    def test_gather_images_none_defaults(self):
        """Test that None URLs parameter works (fixed mutable default)"""
        # This tests the fix for mutable default arguments
        with tempfile.TemporaryDirectory() as temp_dir:
            img = Image.new("RGB", (50, 50))
            path = os.path.join(temp_dir, "test.jpg")
            img.save(path)

            glob_pattern = os.path.join(temp_dir, "*.jpg")
            images = gather_images(glob=glob_pattern)  # urls=None

            assert len(images) == 1

    def test_gather_images_happy_path_tmp_path(self, tmp_path):
        """Allowed-extension local files under a glob still load fine after
        routing through fetch_image's validation."""
        img1 = Image.new("RGB", (16, 16), color="red")
        img2 = Image.new("RGB", (16, 16), color="blue")
        img1.save(tmp_path / "a.png")
        img2.save(tmp_path / "b.png")

        glob_pattern = os.path.join(str(tmp_path), "*.png")
        images = gather_images(glob=glob_pattern)

        assert len(images) == 2
        assert all(isinstance(img, Image.Image) for img in images)


class TestGatherVideos:
    """Test video gathering functionality"""

    @patch("dw.tasks.gather.load_video")
    @patch("dw.tasks.gather.validate_url")
    def test_gather_videos_from_urls(self, mock_validate_url, mock_load_video):
        """Test gathering videos from URLs"""
        mock_validate_url.side_effect = lambda x: x
        mock_frames = ["frame1", "frame2"]
        mock_load_video.return_value = mock_frames

        urls = ["https://example.com/video.mp4"]
        videos = gather_videos(urls=urls)

        assert len(videos) == 1
        assert videos[0] == mock_frames
        mock_validate_url.assert_called_once()

    def test_gather_videos_no_results_raises_error(self):
        """Test that gathering no videos raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            gather_videos(glob="/nonexistent/*.mp4", urls=[])

        assert "No videos found" in str(exc_info.value)

    def test_gather_videos_none_defaults(self):
        """Test that None URLs parameter works"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Since we can't easily create real video files in tests,
            # we'll just test that the function handles None properly
            with pytest.raises(ValueError) as exc_info:
                gather_videos(glob=os.path.join(temp_dir, "*.mp4"))

            assert "No videos found" in str(exc_info.value)

    def test_gather_videos_invalid_url(self):
        """Test that invalid URLs are rejected"""
        with pytest.raises(SecurityError):
            gather_videos(urls=["ftp://example.com/video.mp4"])

    def test_gather_videos_disallowed_extension_raises(self, tmp_path):
        """A glob match whose extension isn't in the video allowlist must raise,
        not be silently loaded."""
        payload = tmp_path / "payload.txt"
        payload.write_text("not a video")

        glob_pattern = os.path.join(str(tmp_path), "*.txt")

        with pytest.raises(SecurityError):
            gather_videos(glob=glob_pattern)

    def test_gather_videos_traversal_glob_raises(self, tmp_path):
        """A glob pattern that walks outside its own directory via '..' must be
        rejected, not silently followed to read the target file."""
        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()
        secret_dir = tmp_path / "secret"
        secret_dir.mkdir()
        (secret_dir / "target.mp4").write_bytes(b"not a real video")

        traversal_pattern = os.path.join(str(workflow_dir), "..", "secret", "*.mp4")

        with pytest.raises(SecurityError):
            gather_videos(glob=traversal_pattern)


class TestGatherInputs:
    """Test input gathering functionality"""

    def test_gather_inputs_returns_unchanged(self):
        """Test that gather_inputs returns inputs unchanged"""
        inputs = {"key1": "value1", "key2": "value2"}
        result = gather_inputs(inputs)

        assert result == inputs

    def test_gather_inputs_with_list(self):
        """Test gather_inputs with list input"""
        inputs = ["item1", "item2", "item3"]
        result = gather_inputs(inputs)

        assert result == inputs

    def test_gather_inputs_with_nested_structure(self):
        """Test gather_inputs with nested structures"""
        inputs = {"outer": {"inner": ["value1", "value2"]}}
        result = gather_inputs(inputs)

        assert result == inputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
