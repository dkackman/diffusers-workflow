"""#1429: a video reference's length is priced and VRAM-charged.

The three arms #1401 measured on the ref2va template (RTX 3090, 24 GB): a whole
124-frame reference at 1344x768 OOM'd (job e6ecb7df55ec), the same at 960x544
ran at about 4.4x the unreferenced cost, and a 22-frame trim_video window at
1344x768 ran at about 1.35x. Every test reads the real catalog template."""

import json
import re
from pathlib import Path

import pytest

import dw.media
import dw.probe_paths
from dw.cost_scaling import declared_reference_exponent, reference_scaling
from dw.plan import _own_price
from dw.reference_limits import video_reference_frames
from dw.vram_estimate import required_vram_gb, vram_estimate_errors

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "workflows/templates/minimax/reference-to-video.json"
)
VIDEO = "diffusers.modular_pipelines.minimax_h3.MiniMaxH3VideoReference"
IMAGE = "diffusers.modular_pipelines.minimax_h3.MiniMaxH3ImageReference"
CARD = 24


@pytest.fixture(autouse=True)
def readable_clips(monkeypatch):
    """Resolution is not under test: any string is a file, probed to the
    frame count its name carries (`clip124.mp4`)."""
    monkeypatch.setattr(
        dw.probe_paths,
        "resolve_probe_path",
        lambda value, base_dir, what="": value if isinstance(value, str) else None,
    )

    def probe(path):
        found = re.search(r"clip(\d+)", str(path))
        if not found:
            return None
        return {"kind": "video", "frame_count": int(found.group(1)), "fps": 24.0}

    monkeypatch.setattr(dw.media, "probe_metadata", probe)
    return probe


def template():
    return json.loads(TEMPLATE.read_text())


def with_video(*sources, width=960, height=544, frames=124, extra=()):
    definition = template()
    definition["variables"].update(width=width, height=height, num_frames=frames)
    arguments = definition["steps"][0]["pipeline"]["arguments"]
    # the VRAM projection reads the definition as it will run, substituted
    arguments.update(width=width, height=height, num_frames=frames)
    references = arguments["references"]
    references[:] = (
        [{"reference_type": VIDEO, "from_file": source} for source in sources]
        + list(extra)
        + references
    )
    return definition


def errors(definition, probe):
    return vram_estimate_errors(definition, probe=probe)


class TestVram:
    def test_a_whole_clip_at_the_big_canvas_is_flagged(self, readable_clips):
        found = errors(
            with_video("clip124.mp4", width=1344, height=768), readable_clips
        )
        assert found and "video reference 124 frames" in found[0]["message"]

    def test_a_whole_clip_at_the_small_canvas_passes(self, readable_clips):
        assert not errors(with_video("clip124.mp4"), readable_clips)

    def test_a_22_frame_window_at_the_big_canvas_passes(self, readable_clips):
        definition = with_video("clip22.mp4", width=1344, height=768)
        assert not errors(definition, readable_clips)

    def test_several_video_references_sum(self, readable_clips):
        one = required_vram_gb(with_video("clip22.mp4"), probe=readable_clips)[0]
        two = required_vram_gb(
            with_video("clip22.mp4", "clip22.mp4"), probe=readable_clips
        )[0]
        base = required_vram_gb(with_video(), probe=readable_clips)[0]
        assert two - base == pytest.approx(2 * (one - base))

    def test_an_unreadable_length_is_not_charged(self, readable_clips):
        definition = with_video(
            "previous_result:missing", width=1344, height=768, frames=124
        )
        for reference in definition["steps"][0]["pipeline"]["arguments"]["references"]:
            if reference.get("from_file") == "previous_result:missing":
                reference.pop("from_file")
                reference["from_previous_result"] = "missing"
        assert not errors(definition, readable_clips)

    def test_an_image_reference_is_not_charged_by_length(self, readable_clips):
        definition = with_video(width=1344, height=768)
        definition["steps"][0]["pipeline"]["arguments"]["references"].append(
            {"reference_type": IMAGE, "from_file": "still124.png"}
        )
        base = required_vram_gb(
            with_video(width=1344, height=768), probe=readable_clips
        )
        grown = required_vram_gb(definition, probe=readable_clips)
        assert grown[0] - base[0] == pytest.approx(1.0)

    def test_the_reader_lists_video_references_only(self, readable_clips):
        definition = with_video("clip22.mp4")
        step = definition["steps"][0]
        assert video_reference_frames(
            step["pipeline"], definition["steps"], None, readable_clips
        ) == [22]


def driven():
    """The template with its video reference a variable (null: no reference,
    as the figure was measured), the way a catalog template offers one."""
    definition = with_video("variable:clip")
    definition["variables"]["clip"] = None
    return definition


def quote(definition, clip=None):
    variables = {**definition["variables"], "clip": clip}
    return _own_price(definition, {"variables": variables}, {}, "cuda", {})


class TestCost:
    def test_the_template_declares_the_exponent(self):
        assert declared_reference_exponent(template()) == 2.0
        assert declared_reference_exponent({"cost": [{"minutes": 1}]}) is None

    def test_a_whole_clip_prices_near_the_measured_4_4x(self):
        plain = quote(driven())["minutes"]
        price = quote(driven(), "clip124.mp4")
        assert price["minutes"] / plain == pytest.approx(4.0, abs=0.1)
        assert "video reference 124 of 124 frames" in price["scaled"]

    def test_a_22_frame_window_prices_near_the_measured_1_35x(self):
        plain = quote(driven())["minutes"]
        window = quote(driven(), "clip22.mp4")["minutes"]
        assert window / plain == pytest.approx(1.39, abs=0.05)

    def test_the_fixed_share_is_not_multiplied(self):
        definition = driven()
        definition["cost"][0]["fixed_minutes"] = 2.0
        plain = quote(definition)["minutes"]
        whole = quote(definition, "clip124.mp4")["minutes"]
        assert whole == pytest.approx(2.0 + (plain - 2.0) * 4.0, rel=0.03)

    def test_without_a_declared_exponent_a_video_reference_is_not_priced(self):
        definition = driven()
        del definition["cost"][0]["video_reference_exponent"]
        assert quote(definition, "clip124.mp4")["minutes"] == pytest.approx(
            quote(definition)["minutes"]
        )

    def test_an_unreadable_clip_is_priced_as_the_template(self):
        definition = driven()
        assert quote(definition, "other.mp4")["minutes"] == pytest.approx(
            quote(definition)["minutes"]
        )

    def test_an_inline_copy_with_a_clip_is_priced_against_the_template(self):
        ratio, reason = reference_scaling(
            template(), None, with_video("clip124.mp4"), None, template()
        )
        assert ratio == pytest.approx(4.0, abs=0.1)
        assert "video reference" in reason
        none, _ = reference_scaling(template(), None, template(), None, template())
        assert none == 1.0
