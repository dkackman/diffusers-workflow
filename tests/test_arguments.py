"""
Unit tests for arguments module
Tests argument realization, image/video fetching, and type loading
"""

import pytest
import os
import tempfile
from dataclasses import dataclass
from PIL import Image
from unittest.mock import patch, MagicMock
from dw.arguments import (
    build_objects,
    realize_args,
    fetch_constant,
    fetch_image,
    fetch_video,
    realize_constants,
)
from dw.security import InvalidInputError, SecurityError
from dw.variables import set_variables


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

    def test_fetch_image_list(self):
        """Test that fetch_image can handle a list of image specifications"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test images
            img1 = Image.new("RGB", (100, 100), color="red")
            img2 = Image.new("RGB", (200, 200), color="blue")
            path1 = os.path.join(temp_dir, "img1.png")
            path2 = os.path.join(temp_dir, "img2.png")
            img1.save(path1)
            img2.save(path2)

            # Test list of paths
            result = fetch_image([path1, path2])
            assert isinstance(result, list)
            assert len(result) == 2
            assert isinstance(result[0], Image.Image)
            assert isinstance(result[1], Image.Image)
            assert result[0].size == (100, 100)
            assert result[1].size == (200, 200)

    def test_fetch_image_list_with_dicts(self):
        """Test that fetch_image can handle a list of dict specifications"""
        with tempfile.TemporaryDirectory() as temp_dir:
            img1 = Image.new("RGB", (50, 50), color="green")
            img2 = Image.new("RGB", (75, 75), color="yellow")
            path1 = os.path.join(temp_dir, "test1.jpg")
            path2 = os.path.join(temp_dir, "test2.jpg")
            img1.save(path1)
            img2.save(path2)

            # Test list of dicts
            result = fetch_image([{"location": path1}, {"location": path2}])
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0].size == (50, 50)
            assert result[1].size == (75, 75)

    def test_fetch_image_already_loaded(self):
        """Test that fetch_image returns already-loaded PIL Images as-is"""
        img = Image.new("RGB", (100, 100))
        result = fetch_image(img)
        assert result is img

    def test_fetch_image_list_already_loaded(self):
        """Test that fetch_image handles already-loaded images in a list"""
        img1 = Image.new("RGB", (50, 50), color="red")
        img2 = Image.new("RGB", (50, 50), color="blue")

        result = fetch_image([img1, img2])

        assert isinstance(result, list)
        assert len(result) == 2
        # Images pass through unchanged (even though list is new)
        assert result[0] is img1
        assert result[1] is img2


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

    @patch("dw.arguments.load_video")
    @patch("dw.arguments.validate_url")
    def test_fetch_video_list(self, mock_validate_url, mock_load_video):
        """Test that fetch_video can handle a list of video specifications"""
        mock_validate_url.side_effect = lambda x: x
        mock_load_video.side_effect = [["frames1"], ["frames2"]]

        result = fetch_video(
            ["https://example.com/video1.mp4", "https://example.com/video2.mp4"]
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == ["frames1"]
        assert result[1] == ["frames2"]

    @patch("dw.arguments.load_video")
    @patch("dw.arguments.validate_url")
    def test_fetch_video_list_with_dicts(self, mock_validate_url, mock_load_video):
        """Test that fetch_video can handle a list of dict specifications"""
        mock_validate_url.side_effect = lambda x: x
        mock_load_video.side_effect = [["frames1"], ["frames2"]]

        result = fetch_video(
            [
                {"location": "https://example.com/video1.mp4"},
                {"location": "https://example.com/video2.mp4"},
            ]
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == ["frames1"]
        assert result[1] == ["frames2"]

    def test_fetch_video_already_loaded_frames(self):
        """Test that fetch_video returns already-loaded frames as-is"""
        # Create frame-like objects (PIL Images representing video frames)
        frame1 = Image.new("RGB", (100, 100))
        frame2 = Image.new("RGB", (100, 100))
        frames = [frame1, frame2]
        result = fetch_video(frames)
        # Should detect it's already loaded frames (list of PIL Images)
        # The list itself may be new but the frames are the same objects
        assert isinstance(result, list)
        assert len(result) == 2


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

    def test_realize_escaped_type_reference_survives_second_pass(self):
        # Variables are realized before the steps they are substituted into, so an
        # escaped value under a type-like key gets realized twice - the second pass
        # must not load the unescaped name as a type
        args = {"weights_dtype": "{int4}"}
        realize_args(args)
        realize_args(args)

        assert args["weights_dtype"] == "int4"

    def test_realize_escaped_variable_substituted_into_type_key(self):
        # The MiniMaxH3 shape: a variable named like a type key, substituted into
        # an argument of the same name
        variables = {"weights_dtype": "{int4}"}
        realize_args(variables)

        steps = {"arguments": {"weights_dtype": variables["weights_dtype"]}}
        realize_args(steps)

        assert steps["arguments"]["weights_dtype"] == "int4"

    def test_realize_escaped_offload_type_survives_second_pass(self):
        args = {"group_offload": {"offload_type": "{leaf_level}"}}
        realize_args(args)
        realize_args(args)

        assert args["group_offload"]["offload_type"] == "leaf_level"

    def test_realize_content_type_not_converted(self):
        # content_type should not be treated as a type reference
        args = {"content_type": "image/jpeg"}
        realize_args(args)

        assert args["content_type"] == "image/jpeg"

    def test_realize_offload_type_not_converted(self):
        # offload_type names a group offloading strategy, not a python type
        args = {"group_offload": {"offload_type": "leaf_level"}}
        realize_args(args)

        assert args["group_offload"]["offload_type"] == "leaf_level"

    def test_realize_escaped_offload_type_is_unescaped(self):
        # The previously mandatory {} escape keeps working after the key
        # was excluded from type conversion
        args = {"group_offload": {"offload_type": "{leaf_level}"}}
        realize_args(args)

        assert args["group_offload"]["offload_type"] == "leaf_level"

    def test_explicit_media_reference_loads_under_any_key(self):
        # A "mask" argument names no media in its key - the explicit form
        # says what it is instead
        with tempfile.TemporaryDirectory() as temp_dir:
            Image.new("RGB", (50, 50)).save(os.path.join(temp_dir, "mask.png"))

            args = {"mask": {"media_type": "image", "location": "mask.png"}}
            realize_args(args, base_dir=temp_dir)

            assert isinstance(args["mask"], Image.Image)

    def test_explicit_media_reference_in_a_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Image.new("RGB", (50, 50)).save(os.path.join(temp_dir, "m.png"))

            args = {"masks": [{"media_type": "image", "location": "m.png"}]}
            realize_args(args, base_dir=temp_dir)

            assert isinstance(args["masks"][0], Image.Image)

    def test_unknown_media_type_raises(self):
        args = {"mask": {"media_type": "audio", "location": "m.wav"}}
        with pytest.raises(ValueError) as exc_info:
            realize_args(args)

        assert "media_type" in str(exc_info.value)

    def test_bare_location_dict_is_left_alone_under_other_keys(self):
        # Without media_type, a dict with a location key belongs to its consumer
        args = {"config": {"location": "somewhere", "other": 1}}
        realize_args(args)

        assert args["config"] == {"location": "somewhere", "other": 1}

    def test_realize_image_relative_to_base_dir(self):
        # Workflow files name their media relative to themselves
        with tempfile.TemporaryDirectory() as temp_dir:
            Image.new("RGB", (50, 50)).save(os.path.join(temp_dir, "subject.png"))

            args = {"image": "subject.png"}
            realize_args(args, base_dir=temp_dir)

            assert isinstance(args["image"], Image.Image)

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


class Reference:
    """Stands in for a pipeline argument built from a file, e.g. MiniMaxH3ImageReference"""

    def __init__(self, location, **arguments):
        self.location = location
        self.arguments = arguments

    @classmethod
    def from_file(cls, media, **arguments):
        return cls(media, **arguments)


class TestRealizeObject:
    """Test constructing arguments that name a type and the file to build it from"""

    def reference_argument(self, location, **arguments):
        return {"reference_type": Reference, "from_file": location, **arguments}

    def test_object_is_constructed_from_a_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "voice.wav")
            open(audio_path, "w").close()

            args = {"references": [self.reference_argument(audio_path)]}
            realize_args(args)

            reference = args["references"][0]
            assert isinstance(reference, Reference)
            assert reference.location == audio_path

    def test_remaining_keys_are_passed_to_from_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "motion.mp4")
            open(video_path, "w").close()

            args = {"reference": self.reference_argument(video_path, fps=30.0)}
            realize_args(args)

            assert args["reference"].arguments == {"fps": 30.0}

    @patch("dw.arguments.validate_url")
    def test_object_is_constructed_from_a_url(self, mock_validate_url):
        url = "https://example.com/subject.jpg"
        mock_validate_url.return_value = url

        args = {"reference": self.reference_argument(url)}
        realize_args(args)

        assert args["reference"].location == url

    def test_type_reference_is_resolved_before_construction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "subject.png")
            Image.new("RGB", (10, 10)).save(image_path)

            args = {
                "reference": {
                    "reference_type": "tests.test_arguments.Reference",
                    "from_file": image_path,
                }
            }
            realize_args(args)

            assert isinstance(args["reference"], Reference)

    def test_without_a_type_the_dict_is_left_untouched(self):
        # A dict that merely contains a 'from_file' key is not an object
        # description - it belongs to whatever consumes it
        args = {"reference": {"from_file": "subject.png"}}
        realize_args(args)

        assert args["reference"] == {"from_file": "subject.png"}

    def test_with_two_types_it_raises(self):
        args = {
            "reference": {
                "reference_type": Reference,
                "other_type": Reference,
                "from_file": "subject.png",
            }
        }
        with pytest.raises(ValueError) as exc_info:
            realize_args(args)

        assert "exactly one" in str(exc_info.value)

    def test_with_an_escaped_type_it_raises(self):
        # An escaped '_type' stays a string, which cannot be constructed
        args = {
            "reference": {"reference_type": "{Reference}", "from_file": "subject.png"}
        }
        with pytest.raises(ValueError) as exc_info:
            realize_args(args)

        assert "must name a type" in str(exc_info.value)

    def test_previous_result_reference_raises_a_clear_error(self):
        args = {"reference": self.reference_argument("previous_result:generate_voice")}
        with pytest.raises(ValueError) as exc_info:
            realize_args(args)

        assert "previous step's result" in str(exc_info.value)

    def test_unresolved_variable_raises_a_clear_error(self):
        args = {"reference": self.reference_argument("variable:voice")}
        with pytest.raises(ValueError) as exc_info:
            realize_args(args)

        assert "variable" in str(exc_info.value)

    def test_relative_path_resolves_against_base_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "voice.wav")
            open(audio_path, "w").close()

            args = {"reference": self.reference_argument("voice.wav")}
            realize_args(args, base_dir=temp_dir)

            assert args["reference"].location == os.path.realpath(audio_path)

    def test_a_type_without_from_file_raises(self):
        args = {
            "reference": {"reference_type": Image.Image, "from_file": "subject.png"}
        }
        with pytest.raises(ValueError) as exc_info:
            realize_args(args)

        assert "from_file" in str(exc_info.value)

    def test_disallowed_extension_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "subject.sh")
            open(script_path, "w").close()

            args = {"reference": self.reference_argument(script_path)}
            with pytest.raises(SecurityError):
                realize_args(args)

    def test_path_traversal_raises(self):
        args = {"reference": self.reference_argument("../../../etc/passwd.wav")}
        with pytest.raises(SecurityError):
            realize_args(args)

    def test_dicts_without_from_file_are_untouched(self):
        args = {"configuration": {"component_type": Reference, "offload": "model"}}
        realize_args(args)

        assert args["configuration"] == {
            "component_type": Reference,
            "offload": "model",
        }


class ImageReference:
    """Stands in for MiniMaxH3ImageReference - a reference built from an image"""

    kind = "image"

    def __init__(self, image):
        self.image = image


class VideoReference:
    """Stands in for MiniMaxH3VideoReference - frames, and the rates they carry"""

    kind = "video"

    def __init__(self, frames, fps=None, audio=None, sample_rate=None):
        self.frames = frames
        self.fps = fps
        self.audio = audio
        self.sample_rate = sample_rate


class AudioReference:
    """Stands in for MiniMaxH3AudioReference - a waveform and its sample rate"""

    kind = "audio"

    def __init__(self, audio, sample_rate=None):
        self.audio = audio
        self.sample_rate = sample_rate


def audio_video(frames=None, audio=None, sample_rate=None):
    """A generated video artifact, the shape a pipeline's audio+video output takes"""
    from dw.result import AudioVideo

    if frames is None:
        frames = [Image.new("RGB", (8, 8))]
    return AudioVideo(frames, audio, sample_rate)


class TestDeferredObjectDescriptions:
    """An object built from a step's output is described now and built later"""

    def test_the_description_survives_realization(self):
        args = {
            "references": [
                {"reference_type": ImageReference, "from_previous_result": "draw"}
            ]
        }
        realize_args(args)

        # Nothing to build yet - the step it names has not run
        assert args["references"][0] == {
            "reference_type": ImageReference,
            "from_previous_result": "draw",
        }

    def test_a_type_that_declares_no_kind_is_rejected(self):
        args = {
            "reference": {"reference_type": Reference, "from_previous_result": "draw"}
        }

        with pytest.raises(ValueError, match="kind"):
            realize_args(args)

    def test_a_description_without_a_type_is_rejected(self):
        args = {"reference": {"from_previous_result": "draw"}}

        with pytest.raises(ValueError, match="_type"):
            realize_args(args)

    def test_two_types_are_rejected(self):
        args = {
            "reference": {
                "reference_type": ImageReference,
                "other_type": VideoReference,
                "from_previous_result": "draw",
            }
        }

        with pytest.raises(ValueError, match="exactly one"):
            realize_args(args)


class TestBuildObjects:
    """Building the objects once the step they reference has produced its media"""

    def test_an_image_reference_is_built_from_an_image(self):
        image = Image.new("RGB", (8, 8))
        arguments = {
            "references": [
                {"reference_type": ImageReference, "from_previous_result": image}
            ]
        }

        built = build_objects(arguments)

        reference = built["references"][0]
        assert isinstance(reference, ImageReference)
        assert reference.image is image

    def test_a_video_reference_carries_the_generated_soundtrack(self):
        import numpy
        import torch

        frames = [Image.new("RGB", (8, 8)) for _ in range(3)]
        audio = numpy.zeros((2, 100), dtype="float32")
        arguments = {
            "references": [
                {
                    "reference_type": VideoReference,
                    "from_previous_result": audio_video(frames, audio, 16000),
                }
            ]
        }

        reference = build_objects(arguments)["references"][0]

        assert len(reference.frames) == 3
        assert torch.is_tensor(reference.audio)
        assert reference.audio.shape == (2, 100)
        assert reference.sample_rate == 16000

    def test_a_video_reference_without_audio_conditions_on_motion_alone(self):
        arguments = {
            "reference": {
                "reference_type": VideoReference,
                "from_previous_result": audio_video(),
            }
        }

        reference = build_objects(arguments)["reference"]

        assert reference.audio is None
        assert reference.sample_rate is None

    def test_an_audio_reference_takes_the_soundtrack(self):
        import numpy

        arguments = {
            "reference": {
                "reference_type": AudioReference,
                "from_previous_result": audio_video(
                    audio=numpy.zeros((2, 50), dtype="float32"), sample_rate=24000
                ),
            }
        }

        reference = build_objects(arguments)["reference"]

        assert reference.audio.shape == (2, 50)
        assert reference.sample_rate == 24000

    def test_an_audio_reference_takes_a_bare_waveform(self):
        # A step that generates audio alone - a music pipeline, a slice_audio
        # task - produces the waveform itself, which carries no rate, so the
        # reference declares one alongside it
        import numpy

        arguments = {
            "reference": {
                "reference_type": AudioReference,
                "from_previous_result": numpy.zeros((50, 2), dtype="float32"),
                "sample_rate": 44100,
            }
        }

        reference = build_objects(arguments)["reference"]

        assert reference.audio.shape == (2, 50)
        assert reference.sample_rate == 44100

    def test_a_named_field_wins_over_the_one_the_media_carried(self):
        # A step that generated at another rate than the consuming pipeline reads
        arguments = {
            "reference": {
                "reference_type": VideoReference,
                "from_previous_result": audio_video(),
                "fps": 30.0,
            }
        }

        assert build_objects(arguments)["reference"].fps == 30.0

    def test_the_wrong_media_for_the_kind_is_an_error(self):
        arguments = {
            "reference": {
                "reference_type": ImageReference,
                "from_previous_result": audio_video(),
            }
        }

        with pytest.raises(ValueError, match="holds an image"):
            build_objects(arguments)

    def test_audio_asked_of_a_step_that_generated_none(self):
        arguments = {
            "reference": {
                "reference_type": AudioReference,
                "from_previous_result": audio_video(),
            }
        }

        with pytest.raises(ValueError, match="produced none"):
            build_objects(arguments)

    def test_arguments_with_nothing_to_build_come_back_unchanged(self):
        # Identity, not equality - rebuilding shared containers for every
        # iteration would multiply what the iterations were sharing
        arguments = {"prompt": "test", "references": [{"location": "a.png"}]}

        assert build_objects(arguments) is arguments


class MediaReference:
    """Stands in for MiniMaxH3VideoReference - a from_file() that takes the media and
    nothing else, and fields the workflow may still want to correct"""

    kind = "video"

    __dataclass_fields__ = {"frames": None, "fps": None, "audio": None}

    def __init__(self, frames, fps=24.0, audio=None):
        self.frames = frames
        self.fps = fps
        self.audio = audio

    @classmethod
    def from_file(cls, media):
        # What decoding a container gives you: the frames, the rate it claims, and
        # whatever soundtrack it carried
        return cls(frames=media, fps=25.0, audio="decoded soundtrack")


class TestFromFileFieldOverrides:
    """Keys a from_file() cannot take are set on the object it returns"""

    def video_argument(self, location, **arguments):
        return {"reference_type": MediaReference, "from_file": location, **arguments}

    def realize(self, **arguments):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "motion.mp4")
            open(video_path, "w").close()

            args = {"reference": self.video_argument(video_path, **arguments)}
            realize_args(args)
            return args["reference"]

    def test_a_rate_the_container_got_wrong_is_corrected(self):
        assert self.realize(fps=30.0).fps == 30.0

    def test_a_soundtrack_can_be_dropped_for_a_motion_only_reference(self):
        assert self.realize(audio=None).audio is None

    def test_what_the_workflow_does_not_name_is_left_as_decoded(self):
        reference = self.realize(fps=30.0)

        assert reference.audio == "decoded soundtrack"

    def test_a_field_the_object_does_not_have_is_an_error(self):
        with pytest.raises(ValueError, match="has no field 'fpss'"):
            self.realize(fpss=30.0)

    def test_a_from_file_taking_keyword_arguments_still_gets_them_all(self):
        # The generic case - a type that decodes the media itself is told how,
        # rather than having its result edited afterwards
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "motion.mp4")
            open(video_path, "w").close()

            args = {
                "reference": {
                    "reference_type": Reference,
                    "from_file": video_path,
                    "fps": 30.0,
                }
            }
            realize_args(args)

            assert args["reference"].arguments == {"fps": 30.0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@dataclass
class Condition:
    """Stands in for LTX2VideoCondition - a plain dataclass of frames and where
    they land, with no from_file() and no media kind of its own"""

    frames: object
    index: int = 0
    strength: float = 1.0


class TestConstructedFromArguments:
    """A type built from the arguments its description names, not from media"""

    def test_constructed_at_load_time(self):
        args = {
            "conditions": [
                {
                    "condition_type": Condition,
                    "from_arguments": {"frames": "a frame", "index": -1},
                }
            ]
        }
        realize_args(args)

        condition = args["conditions"][0]
        assert isinstance(condition, Condition)
        assert (condition.frames, condition.index, condition.strength) == (
            "a frame",
            -1,
            1.0,
        )

    def test_media_in_the_arguments_is_loaded_first(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "keyframe.png")
            Image.new("RGB", (8, 8)).save(path)

            args = {
                "conditions": [
                    {
                        "condition_type": Condition,
                        "from_arguments": {
                            "frames": {"media_type": "image", "location": path},
                            "index": 0,
                        },
                    }
                ]
            }
            realize_args(args)

        assert isinstance(args["conditions"][0].frames, Image.Image)

    def test_an_argument_naming_a_step_defers_construction(self):
        args = {
            "conditions": [
                {
                    "condition_type": Condition,
                    "from_arguments": {
                        "frames": "previous_result:base",
                        "index": -1,
                    },
                }
            ]
        }
        realize_args(args)

        # The step it names has not run, so there is nothing to build from yet
        assert args["conditions"][0]["from_arguments"]["frames"] == (
            "previous_result:base"
        )

    def test_built_once_the_step_it_names_has_run(self):
        description = {
            "condition_type": Condition,
            "from_arguments": {"frames": "the generated frames", "index": -1},
        }

        built = build_objects({"conditions": [description]})

        assert isinstance(built["conditions"][0], Condition)
        assert built["conditions"][0].frames == "the generated frames"
        assert built["conditions"][0].index == -1

    def test_a_description_without_a_type_is_rejected(self):
        args = {"conditions": [{"from_arguments": {"frames": "a frame"}}]}

        with pytest.raises(ValueError, match="_type"):
            realize_args(args)

    def test_arguments_that_are_not_a_dict_are_rejected(self):
        args = {"conditions": [{"condition_type": Condition, "from_arguments": []}]}

        with pytest.raises(ValueError, match="from_arguments"):
            realize_args(args)

    def test_keys_beside_the_arguments_are_rejected(self):
        # Silently ignoring them would drop a strength the workflow meant to set
        args = {
            "conditions": [
                {
                    "condition_type": Condition,
                    "from_arguments": {"frames": "a frame"},
                    "strength": 0.5,
                }
            ]
        }

        with pytest.raises(ValueError, match="strength"):
            realize_args(args)

    def test_an_argument_the_type_does_not_take_names_what_it_does(self):
        args = {
            "conditions": [
                {
                    "condition_type": Condition,
                    "from_arguments": {"frames": "a frame", "weight": 0.5},
                }
            ]
        }

        with pytest.raises(ValueError, match="frames, index, strength"):
            realize_args(args)


# A module of this test module's own, to reference constants that do not depend
# on what a library happens to declare today
SAMPLE_SCHEDULE = [1.0, 0.5, 0.25]
SAMPLE_PROMPT = "the default the library ships"


@dataclass(frozen=True)
class SampleConfig:
    max_new_tokens: int = 600


SAMPLE_CONFIG = SampleConfig()

MODULE = "tests.test_arguments"


class TestConstantReferences:
    """Constants declared in python, referenced instead of copied into JSON"""

    def test_a_constant_resolves_to_its_value(self):
        args = {"negative_prompt": f"constant:{MODULE}.SAMPLE_PROMPT"}

        realize_args(args)

        assert args["negative_prompt"] == SAMPLE_PROMPT

    def test_a_list_constant_resolves_to_its_value(self):
        args = {"sigmas": f"constant:{MODULE}.SAMPLE_SCHEDULE"}

        realize_args(args)

        assert args["sigmas"] == SAMPLE_SCHEDULE

    def test_an_attribute_of_a_constant_resolves(self):
        # A constant held in a config object is as much a constant as one
        # declared at module scope
        args = {"tokens": f"constant:{MODULE}.SAMPLE_CONFIG.max_new_tokens"}

        realize_args(args)

        assert args["tokens"] == 600

    def test_a_constant_resolves_inside_a_list(self):
        args = {"schedules": [f"constant:{MODULE}.SAMPLE_SCHEDULE"]}

        realize_args(args)

        assert args["schedules"] == [SAMPLE_SCHEDULE]

    def test_a_constant_resolves_under_any_argument_name(self):
        # The key conventions that load media do not get first refusal - what a
        # constant holds is the value, not a file to open
        args = {"image": f"constant:{MODULE}.SAMPLE_PROMPT"}

        realize_args(args)

        assert args["image"] == SAMPLE_PROMPT

    def test_a_mutable_constant_is_copied(self):
        args = {"sigmas": f"constant:{MODULE}.SAMPLE_SCHEDULE"}
        realize_args(args)

        # A pipeline that consumes its schedule in place would otherwise edit
        # the library's own constant, for every later run in the process
        args["sigmas"].clear()

        assert SAMPLE_SCHEDULE == [1.0, 0.5, 0.25]

    def test_a_type_is_not_a_constant(self):
        with pytest.raises(ValueError, match="_type"):
            fetch_constant("constant:diffusers.FluxPipeline")

    def test_a_function_is_not_a_constant(self):
        # The whole point of the restriction - a reference reads a value, it
        # does not reach anything the workflow could get called on its behalf
        with pytest.raises(ValueError, match="not a constant"):
            fetch_constant("constant:os.system")

    def test_an_unknown_constant_names_itself(self):
        with pytest.raises(ValueError, match="no_such_module.NAME"):
            fetch_constant("constant:no_such_module.NAME")

    def test_an_unknown_attribute_of_a_real_module_is_rejected(self):
        with pytest.raises(ValueError, match="NOT_A_REAL_CONSTANT"):
            fetch_constant(f"constant:{MODULE}.NOT_A_REAL_CONSTANT")

    def test_an_empty_name_is_rejected(self):
        with pytest.raises(InvalidInputError, match="empty"):
            fetch_constant("constant:")

    def test_a_name_that_is_not_a_dotted_name_is_rejected(self):
        # Checked before anything is imported - resolving a name runs the
        # module it names
        for name in ("../../etc/passwd", "__import__('os')", "os.system()"):
            with pytest.raises(InvalidInputError, match="Invalid constant name"):
                fetch_constant(f"constant:{name}")

    def test_a_name_that_is_too_long_is_rejected(self):
        with pytest.raises(InvalidInputError, match="too long"):
            fetch_constant("constant:" + ".".join(["a"] * 120))

    def test_a_string_that_is_not_a_reference_is_left_alone(self):
        args = {"prompt": "a constant: a thing that does not change"}

        realize_args(args)

        assert args["prompt"] == "a constant: a thing that does not change"

    def test_a_constant_declares_a_variable_type(self):
        # A variable's declared value is its type, and an argument passed in is
        # converted to it - which only holds if the constant resolves first
        variables = {"sigmas": f"constant:{MODULE}.SAMPLE_SCHEDULE"}

        realize_constants(variables)
        set_variables({"sigmas": "0.9, 0.5"}, variables)

        assert variables["sigmas"] == ["0.9", "0.5"]

    def test_realize_constants_leaves_everything_else_alone(self):
        # It runs before the variables are set, ahead of the pass that loads
        # media - a file that does not exist yet must not be opened here
        args = {
            "image": {"location": "not-yet-generated.png"},
            "prompt": "variable:prompt",
            "sigmas": [f"constant:{MODULE}.SAMPLE_SCHEDULE"],
        }

        realize_constants(args)

        assert args["image"] == {"location": "not-yet-generated.png"}
        assert args["prompt"] == "variable:prompt"
        assert args["sigmas"] == [SAMPLE_SCHEDULE]
