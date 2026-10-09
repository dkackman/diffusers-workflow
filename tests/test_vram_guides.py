"""#779: an optional bytes_per_guide_voxel charges the guides a step lays in.

The guide term is bytes_per_guide_voxel * sum(guide frames) * canvas / 2**30,
canvas being every voxel variable but num_frames. With a 2**20 bytes-per-voxel
price a guide of F frames on a W x H canvas costs F * W * H / 1024 GB, which
keeps the arithmetic in each test legible.
"""

import tempfile
from unittest.mock import patch

import pytest

import dw.guides
import dw.validation
from dw.media import probe_metadata
from dw.vram_estimate import (
    apply_vram_estimate,
    required_gb,
    required_vram_gb,
    vram_estimate_errors,
)
from dw.workflow import workflow_from_definition

from tests.test_media_info import write_mp4

PRICE = 2**20
CEILING = 24


def estimate(price=PRICE, bytes_per_voxel=0, base_gb=1):
    result = {
        "voxel_variables": ["width", "height", "num_frames"],
        "base_gb": base_gb,
        "bytes_per_voxel": bytes_per_voxel,
    }
    if price is not None:
        result["bytes_per_guide_voxel"] = price
    return result


def definition(
    guides=None, chain=None, width=64, height=32, num_frames=121, price=PRICE
):
    arguments = {"width": width, "height": height, "num_frames": num_frames}
    if guides is not None:
        arguments["guides"] = guides
    pipeline = {"arguments": arguments}
    if chain is not None:
        pipeline["chain"] = chain
    return {
        "cost": [{"device": "cuda", "name": "Card", "vram_gb": CEILING, "minutes": 1}],
        "vram_estimate": estimate(price),
        "steps": [{"name": "s1", "pipeline": pipeline}],
    }


def clip(name="asset:a.mp4", **extra):
    return {"video": name, "frame": 17, **extra}


def gb(frames, width=64, height=32):
    """The guide term, for `frames` guide frames, computed independently."""
    return PRICE * frames * width * height / 2**30


def fake_probe(frame_count=22):
    return lambda path: {"kind": "video", "frame_count": frame_count}


@pytest.fixture(autouse=True)
def resolve_to_path(monkeypatch):
    """Resolution is not under test: any named string is a file. The real-file
    test patches this back."""
    monkeypatch.setattr(
        dw.guides,
        "resolve_probe_path",
        lambda value, base_dir, what="": (
            None
            if not isinstance(value, str) or value.startswith("previous_result:")
            else "/resolved/" + value
        ),
    )


def project(d, probe=None):
    """The projected GB, through required_vram_gb."""
    result = required_vram_gb(d, probe=probe)
    return None if result is None else result[0]


BASE = 1.0  # base_gb, no per-voxel term


def test_without_the_key_guides_and_a_guide_chain_change_nothing():
    plain = definition(price=None)
    guided = definition(
        guides=[clip(), clip("asset:b.mp4")],
        chain={"continuity": "guide", "segments": 3},
        price=None,
    )
    assert project(plain) == pytest.approx(BASE)
    assert project(guided, fake_probe(22)) == pytest.approx(project(plain))
    assert project(guided) == pytest.approx(project(plain))
    assert required_gb(
        estimate(price=None), {"width": 64, "height": 32, "num_frames": 9}, 0, [22, 22]
    ) == pytest.approx(BASE)


def test_the_guide_term_is_frames_times_canvas():
    d = definition(guides=[clip()])
    assert project(d, fake_probe(22)) == pytest.approx(BASE + gb(22))
    assert gb(22) == pytest.approx(44.0)


def test_doubling_the_width_doubles_the_guide_term():
    narrow = project(definition(guides=[clip()]), fake_probe(22)) - BASE
    wide = project(definition(guides=[clip()], width=128), fake_probe(22)) - BASE
    assert wide == pytest.approx(2 * narrow)
    assert wide == pytest.approx(gb(22, width=128))


def test_guide_lengths_add_and_snap():
    d = definition(guides=[clip(), clip("asset:b.mp4")])
    assert project(d, fake_probe(23)) == pytest.approx(BASE + gb(22 + 22))
    assert project(d, fake_probe(40)) == pytest.approx(BASE + gb(39 + 39))


def test_a_guide_with_no_video_costs_nothing():
    d = definition(guides=[clip(), {"video": None, "frame": 3}])
    assert project(d, fake_probe(22)) == pytest.approx(BASE + gb(22))
    only_null = definition(guides=[{"video": None, "frame": 3}])
    assert project(only_null, fake_probe(22)) == pytest.approx(BASE)


@pytest.mark.parametrize(
    "probe",
    [
        None,
        lambda path: None,
        lambda path: {"kind": "video"},
        lambda path: {"kind": "video", "frame_count": 0},
    ],
    ids=["no probe", "probe says None", "no frame count", "zero frames"],
)
def test_an_unprobeable_guide_is_charged_at_num_frames(probe):
    d = definition(guides=[clip()], num_frames=60)
    assert project(d, probe) == pytest.approx(BASE + gb(60))


def test_a_raising_probe_is_not_swallowed_by_this_layer():
    # Report-only: validation's memoizing probe wraps probe_metadata, which
    # itself returns None on an unreadable file; a raise is a caller bug.
    def boom(path):
        raise OSError("unreadable")

    with pytest.raises(OSError):
        project(definition(guides=[clip()]), boom)


def test_a_previous_result_guide_is_charged_at_num_frames():
    d = definition(guides=[clip("previous_result:s0")], num_frames=60)
    assert project(d, fake_probe(22)) == pytest.approx(BASE + gb(60))


def test_the_error_names_the_guides_and_the_worst_case():
    d = definition(guides=[clip(), clip("previous_result:s0")], num_frames=124)
    errors = vram_estimate_errors(d, probe=fake_probe(22))
    assert len(errors) == 1
    message = errors[0]["message"]
    assert "with 2 guides (22+124 frames" in message
    assert "worst case" in message
    assert "bytes per unit of each guide's frames" in message


def test_a_probeable_guide_error_does_not_claim_a_worst_case():
    errors = vram_estimate_errors(definition(guides=[clip()]), probe=fake_probe(22))
    assert len(errors) == 1
    assert "with 1 guide (22 frames)" in errors[0]["message"]
    assert "worst case" not in errors[0]["message"]


@pytest.mark.parametrize(
    "chain, frames",
    [
        ({"continuity": "guide", "segments": 3}, 22),
        ({"continuity": "guide", "segments": 3, "guide_frames": 22}, 22),
        ({"continuity": "guide", "segments": 3, "guide_frames": 39}, 39),
        ({"continuity": "guide"}, 22),
    ],
)
def test_a_guide_chain_counts_one_extra_guide(chain, frames):
    assert project(definition(chain=chain)) == pytest.approx(BASE + gb(frames))
    # beside the step's own guides
    d = definition(guides=[clip()], chain=chain)
    assert project(d, fake_probe(22)) == pytest.approx(BASE + gb(22 + frames))


def test_a_one_segment_guide_chain_adds_none():
    chain = {"continuity": "guide", "segments": 1, "guide_frames": 39}
    assert project(definition(chain=chain)) == pytest.approx(BASE)


def test_another_continuity_adds_no_guide():
    chain = {"continuity": "latent", "segments": 3}
    assert project(definition(chain=chain)) == pytest.approx(BASE)


def test_validate_apply_and_required_agree_on_a_guided_definition():
    d = definition(
        guides=[clip(), clip("previous_result:s0")],
        chain={"continuity": "guide", "segments": 2},
        num_frames=124,
    )
    probe = fake_probe(22)
    projected = project(d, probe)
    assert projected == pytest.approx(BASE + gb(22 + 124 + 22))
    assert projected > CEILING

    # the message carries the same number
    message = vram_estimate_errors(d, probe=probe)[0]["message"]
    assert f"projects to {projected:.2f} GB" in message
    assert "with 3 guides (22+124+22 frames" in message

    # run-time backstop refuses at the same ceiling, accepts just above it
    with pytest.raises(ValueError, match=f"projects to {projected:.2f} GB"):
        apply_vram_estimate(d, None, "cuda", CEILING, probe=probe)
    for margin, refused in ((0.5, False), (-0.5, True)):
        card = {"device": "cuda", "name": "Card", "vram_gb": projected + margin}
        roomy = {**d, "cost": [{**card, "minutes": 1}]}
        if refused:
            with pytest.raises(ValueError):
                apply_vram_estimate(roomy, None, "cuda", 1000, probe=probe)
        else:
            apply_vram_estimate(roomy, None, "cuda", 1000, probe=probe)

    # validation refuses the same shape with the same number; it wants a
    # schema-complete workflow
    d = {**d, "id": "guided"}
    pipeline = d["steps"][0]["pipeline"]
    pipeline["configuration"] = {"component_type": "ModularPipeline"}
    pipeline["from_pretrained_arguments"] = {
        "model_name": "MiniMaxAI/MiniMax-H3",
        "workflow": "t2va",
    }
    with (
        patch.object(dw.validation, "get_device_type", return_value="cuda"),
        patch.object(dw.validation, "device_capacity_gb", return_value=CEILING),
        patch.object(dw.validation, "probe_metadata", probe),
    ):
        workflow = workflow_from_definition(d, tempfile.mkdtemp())
        errors = [e for e in workflow.validation_errors() if "GB VRAM" in e["message"]]
    assert len(errors) == 1
    assert f"projects to {projected:.2f} GB" in errors[0]["message"]
    assert "worst case" in errors[0]["message"]


def test_a_real_video_is_probed_through_probe_metadata(tmp_path, monkeypatch):
    from dw import probe_paths

    monkeypatch.setattr(dw.guides, "resolve_probe_path", probe_paths.resolve_probe_path)
    write_mp4(tmp_path / "clip.mp4", frames=23, with_audio=False)
    assert probe_metadata(str(tmp_path / "clip.mp4"))["frame_count"] == 23

    d = definition(guides=[clip("clip.mp4")])
    # 23 frames snap to 22
    with pytest.raises(ValueError, match="with 1 guide \\(22 frames\\)"):
        apply_vram_estimate(
            d, None, "cuda", CEILING, base_dir=str(tmp_path), probe=probe_metadata
        )
    required, _hard = required_vram_gb(d, base_dir=str(tmp_path), probe=probe_metadata)
    assert required == pytest.approx(BASE + gb(22))
