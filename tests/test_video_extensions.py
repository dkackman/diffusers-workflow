"""A video argument's extension, refused for free when it is already knowable.

#347: `loop_frames`'s `video` argument took a still image by its own
docstring, but `asset:sheet.png` validated clean and then died inside
`fetch_video`'s extension gate in the first seconds of the run.
"""

import tempfile

from dw.video_extensions import _extension_problem, video_extension_errors
from dw.workflow import workflow_from_definition


def workflow_holding(video):
    return {
        "id": "holding",
        "steps": [
            {
                "name": "hold",
                "task": {
                    "command": "loop_frames",
                    "arguments": {"video": video, "num_frames": 121},
                },
                "result": {"content_type": "video/mp4", "fps": 24},
            }
        ],
    }


class TestTheFault:
    def test_a_still_asset_names_the_media_type_form(self):
        problem = _extension_problem("asset:sheet.png")

        assert "media_type" in problem
        assert "asset:sheet.png" in problem

    def test_a_video_asset_is_fine(self):
        assert _extension_problem("asset:clip.mp4") is None

    def test_a_video_output_reference_is_fine(self):
        assert _extension_problem("output:t/20260914-171601-adeee23c/i/x.mp4") is None

    def test_an_extension_the_run_would_also_refuse_is_named(self):
        assert _extension_problem("asset:notes.txt") is not None

    def test_a_url_is_left_to_the_run(self):
        """`fetch_video` never gates a URL's extension, so this pass must not
        refuse one either - #347's binding scope."""
        assert _extension_problem("https://example.com/sheet.png") is None

    def test_a_deferred_reference_is_not_yet_knowable(self):
        assert _extension_problem("previous_result:make_image") is None
        assert _extension_problem("variable:video_path") is None

    def test_a_path_with_no_extension_is_not_this_passs_complaint(self):
        assert _extension_problem("asset:sheet") is None

    def test_a_non_string_is_not_this_passs_complaint(self):
        assert _extension_problem(None) is None
        assert (
            _extension_problem({"media_type": "video", "location": "asset:x.mp4"})
            is None
        )


class TestTheValidationPass:
    def test_a_still_by_key_convention_is_refused_before_the_queue(self):
        workflow = workflow_from_definition(
            workflow_holding("asset:sheet.png"), tempfile.mkdtemp()
        )

        problems = workflow.validation_errors()

        assert any(
            problem["path"] == "steps[0].task.arguments.video" for problem in problems
        )

    def test_the_media_type_form_validates(self):
        workflow = workflow_from_definition(
            workflow_holding({"media_type": "image", "location": "asset:sheet.png"}),
            tempfile.mkdtemp(),
        )

        assert workflow.validation_errors() == []

    def test_a_media_type_video_reference_checks_its_location(self):
        workflow = workflow_from_definition(
            workflow_holding({"media_type": "video", "location": "asset:sheet.png"}),
            tempfile.mkdtemp(),
        )

        problems = workflow.validation_errors()

        assert any(
            problem["path"] == "steps[0].task.arguments.video.location"
            for problem in problems
        )

    def test_a_real_video_asset_validates(self):
        workflow = workflow_from_definition(
            workflow_holding("asset:clip.mp4"), tempfile.mkdtemp()
        )

        assert workflow.validation_errors() == []

    def test_nothing_is_reported_for_a_definition_with_no_video_arguments(self):
        assert video_extension_errors({"steps": [{"name": "a", "task": {}}]}) == []
