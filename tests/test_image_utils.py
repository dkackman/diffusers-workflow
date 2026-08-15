"""Tests for the process_image dispatch table in dw.tasks.image_utils.

Covers the parts not already exercised by test_resize_bucket.py and
test_strip_exif_and_watermark.py: unknown-processor errors, a cheap
non-detector dispatch, the processor-name enumeration export, and that
controlnet_aux detector loads are routed through the shared model cache so
repeated process_image calls (e.g. once per cartesian-product iteration)
don't reload weights from disk.
"""

import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from dw.tasks import model_cache
from dw.tasks.image_utils import available_processors, process_image


class TestUnknownProcessor(unittest.TestCase):
    def test_unknown_processor_raises_with_expected_message(self):
        img = Image.new("RGB", (10, 10))
        with self.assertRaises(Exception) as ctx:
            process_image(img, "not_a_real_processor", "cpu", {})
        self.assertEqual(
            str(ctx.exception), "Unknown image processor type: not_a_real_processor"
        )

    def test_unknown_processor_message_uses_lowered_name(self):
        img = Image.new("RGB", (10, 10))
        with self.assertRaises(Exception) as ctx:
            process_image(img, "NOT_REAL", "cpu", {})
        self.assertEqual(str(ctx.exception), "Unknown image processor type: not_real")


class TestCheapProcessorDispatch(unittest.TestCase):
    def test_resize_rescale_dispatches_through_table(self):
        img = Image.new("RGB", (800, 600))
        result = process_image(
            img, "resize_rescale", "cpu", {"height": 64, "width": 32}
        )
        self.assertEqual(result.size, (32, 64))

    def test_processor_name_is_case_insensitive(self):
        img = Image.new("RGB", (800, 600))
        result = process_image(
            img, "RESIZE_RESCALE", "cpu", {"height": 16, "width": 16}
        )
        self.assertEqual(result.size, (16, 16))


class TestAvailableProcessors(unittest.TestCase):
    def test_includes_known_names(self):
        names = available_processors()
        for expected in (
            "resize_bucket",
            "strip_exif",
            "add_watermark",
            "canny",
            "canny_cv",
            "dw_pose",
            "sam",
            "shuffle",
            "lineart_standard",
        ):
            self.assertIn(expected, names)

    def test_is_sorted_and_deduplicated(self):
        names = available_processors()
        self.assertEqual(names, sorted(set(names)))


class TestDetectorCaching(unittest.TestCase):
    """The from_pretrained detector family must load once and be reused."""

    def setUp(self):
        model_cache.clear_model_cache()

    def tearDown(self):
        model_cache.clear_model_cache()

    def test_pretrained_detector_loaded_once_for_repeated_calls(self):
        detector_instance = MagicMock()
        detector_instance.to.return_value = detector_instance
        detector_cls = MagicMock()
        detector_cls.from_pretrained.return_value = detector_instance

        mock_controlnet_aux = MagicMock()
        mock_controlnet_aux.MLSDdetector = detector_cls

        img = Image.new("RGB", (10, 10))
        with patch(
            "dw.tasks.image_utils._import_controlnet_aux",
            return_value=mock_controlnet_aux,
        ):
            process_image(img, "mlsd", "cpu", {})
            process_image(img, "mlsd", "cpu", {})
            process_image(img, "mlsd", "cpu", {})

        detector_cls.from_pretrained.assert_called_once_with("lllyasviel/Annotators")
        detector_instance.to.assert_called_once_with("cpu")
        self.assertEqual(detector_instance.call_count, 3)

    def test_zero_arg_detector_loaded_once_for_repeated_calls(self):
        detector_instance = MagicMock()
        detector_cls = MagicMock(return_value=detector_instance)

        mock_controlnet_aux = MagicMock()
        mock_controlnet_aux.CannyDetector = detector_cls

        img = Image.new("RGB", (10, 10))
        with patch(
            "dw.tasks.image_utils._import_controlnet_aux",
            return_value=mock_controlnet_aux,
        ):
            process_image(img, "canny", "cpu", {})
            process_image(img, "canny", "cpu", {})

        detector_cls.assert_called_once()
        self.assertEqual(detector_instance.call_count, 2)

    def test_sam_detector_is_not_moved_to_device(self):
        detector_instance = MagicMock()
        detector_cls = MagicMock()
        detector_cls.from_pretrained.return_value = detector_instance

        mock_controlnet_aux = MagicMock()
        mock_controlnet_aux.SamDetector = detector_cls

        img = Image.new("RGB", (10, 10))
        with patch(
            "dw.tasks.image_utils._import_controlnet_aux",
            return_value=mock_controlnet_aux,
        ):
            process_image(img, "sam", "cpu", {})

        detector_cls.from_pretrained.assert_called_once_with(
            "ybelkada/segment-anything", subfolder="checkpoints"
        )
        detector_instance.to.assert_not_called()

    def test_different_devices_load_independently(self):
        detector_instance = MagicMock()
        detector_instance.to.return_value = detector_instance
        detector_cls = MagicMock()
        detector_cls.from_pretrained.return_value = detector_instance

        mock_controlnet_aux = MagicMock()
        mock_controlnet_aux.MLSDdetector = detector_cls

        img = Image.new("RGB", (10, 10))
        with patch(
            "dw.tasks.image_utils._import_controlnet_aux",
            return_value=mock_controlnet_aux,
        ):
            process_image(img, "mlsd", "cpu", {})
            process_image(img, "mlsd", "cuda", {})

        self.assertEqual(detector_cls.from_pretrained.call_count, 2)


if __name__ == "__main__":
    unittest.main()
