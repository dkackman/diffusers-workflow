"""The pre-run check of an H3 `guides` argument (dw/guides.py)."""

import tempfile

import pytest

from dw import guides as guides_module
from dw.guides import guides_errors
from dw.workflow import workflow_from_definition


def step(workflow="t2va", **arguments):
    return {
        "name": "video",
        "pipeline": {
            "configuration": {"component_type": "ModularPipeline"},
            "from_pretrained_arguments": {"workflow": workflow},
            "arguments": arguments,
        },
    }


def check(step_, probe=None):
    return guides_errors({"steps": [step_]}, probe=probe)


def one(step_, probe=None):
    errors = check(step_, probe)
    assert len(errors) == 1, errors
    return errors[0]


def fake_probe(kind="video", frame_count=22):
    return lambda path: {"kind": kind, "frame_count": frame_count}


@pytest.fixture(autouse=True)
def resolve_to_path(monkeypatch):
    """Resolution is not under test: any asset:/output:/path names a file."""
    monkeypatch.setattr(
        guides_module,
        "resolve_probe_path",
        lambda value, base_dir, what="": (
            None
            if not isinstance(value, str) or value.startswith("previous_result:")
            else "/resolved/" + value
        ),
    )


GOOD = {"video": "asset:clip.mp4", "frame": 17}


class TestClean:
    def test_no_guides(self):
        assert check(step()) == []

    def test_empty_list(self):
        assert check(step(guides=[])) == []

    @pytest.mark.parametrize("workflow", ["t2va", "fl2va"])
    def test_valid_guides(self, workflow):
        s = step(workflow, guides=[GOOD, {"video": "output:a/b.mp4", "frame": 0}])
        assert check(s) == []
        assert check(s, fake_probe()) == []

    def test_previous_result_is_left_alone(self):
        s = step(guides=[{"video": "previous_result:base", "frame": 17}])
        assert check(s, fake_probe(kind="image")) == []

    def test_unresolved_frame_is_left_alone(self):
        s = step(guides=[{"video": "asset:c.mp4", "frame": "previous_result:n"}])
        assert check(s) == []

    def test_no_workflow_is_left_to_run_time(self):
        s = step(guides=[GOOD])
        del s["pipeline"]["from_pretrained_arguments"]
        assert check(s) == []


class TestRefusals:
    def test_non_h3_pipeline(self):
        s = step(guides=[GOOD])
        s["pipeline"]["configuration"]["component_type"] = "StableDiffusionPipeline"
        error = one(s)
        assert "StableDiffusionPipeline" in error["message"]
        assert error["path"] == "steps[0].pipeline.arguments.guides"

    def test_ref2va_workflow(self):
        assert "ref2va" in one(step("ref2va", guides=[GOOD]))["message"]

    def test_references_argument(self):
        error = one(step(guides=[GOOD], references=["asset:r.png"]))
        assert "ref2va / references" in error["message"]

    def test_not_a_list(self):
        assert "must be a list" in one(step(guides=GOOD))["message"]

    def test_not_a_dict_entry(self):
        error = one(step(guides=["asset:c.mp4"]))
        assert error["path"] == "steps[0].pipeline.arguments.guides[0]"
        assert "{video, frame}" in error["message"]

    @pytest.mark.parametrize("key", ["audio", "extra"])
    def test_unknown_key(self, key):
        error = one(step(guides=[{**GOOD, key: "x"}]))
        assert key in error["message"]
        assert "{video, frame}" in error["message"]

    @pytest.mark.parametrize("missing", ["video", "frame"])
    def test_missing_key(self, missing):
        entry = {k: v for k, v in GOOD.items() if k != missing}
        error = one(step(guides=[entry]))
        assert "needs both" in error["message"]

    def test_too_many(self):
        error = one(step(guides=[GOOD] * 5))
        assert "at most 4" in error["message"]
        assert error["path"] == "steps[0].pipeline.arguments.guides"

    @pytest.mark.parametrize("frame", ["x", 1.5, True, -17, 5])
    def test_bad_frame(self, frame):
        error = one(step(guides=[{"video": "asset:c.mp4", "frame": frame}]))
        assert "'frame'" in error["message"]
        assert error["path"].endswith("guides[0]")

    def test_not_a_video(self):
        error = one(step(guides=[GOOD]), fake_probe(kind="image"))
        assert "must be a video" in error["message"]

    def test_clip_runs_past_the_render(self):
        # 40 frames snap to 39; at frame 102 that ends at 141, past 124
        s = step(guides=[{"video": "asset:c.mp4", "frame": 102}])
        error = one(s, fake_probe(frame_count=40))
        assert "past the end of the 124-frame render" in error["message"]

    def test_num_frames_is_aligned_up_before_comparing(self):
        # 100 -> 107; a 22-frame guide at 85 ends at 107, which fits
        fits = step(num_frames=100, guides=[{"video": "asset:c.mp4", "frame": 85}])
        assert check(fits, fake_probe(frame_count=22)) == []
        over = step(num_frames=100, guides=[{"video": "asset:c.mp4", "frame": 102}])
        assert (
            "past the end of the 107-frame render" in one(over, fake_probe())["message"]
        )

    def test_non_static_num_frames_skips_the_end_check(self):
        s = step(num_frames="previous_result:n", guides=[{**GOOD, "frame": 102}])
        assert check(s, fake_probe(frame_count=40)) == []

    def test_member_suffix_and_source_index(self):
        s = step(guides=GOOD)
        s["name"] = "video@a"
        errors = guides_errors({"steps": [{"name": "x"}, s]}, source_indices=[0, 0])
        assert errors[0]["message"].endswith("in member 'video@a'")
        assert errors[0]["path"].startswith("steps[0].")


def test_through_the_validation_entry_point():
    definition = {
        "id": "g",
        "steps": [step(guides=[{"video": "asset:c.mp4", "frame": 5}])],
    }
    workflow = workflow_from_definition(definition, tempfile.mkdtemp())
    found = [p for p in workflow.validation_errors() if p["path"].endswith("guides[0]")]
    assert len(found) == 1
    assert "multiple of 17" in found[0]["message"]
