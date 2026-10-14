"""POST /api/gallery/compare (#871): outputs side by side with PSNR/SSIM
against a baseline, through the real route."""
# ruff: noqa: F811

import base64
import io

import numpy
import pytest
from PIL import Image

from tests.test_media_frames import write_ramp_mp4
from tests.test_server import asset_server, server, success_script  # noqa: F401


def decode(image):
    return Image.open(io.BytesIO(base64.b64decode(image["data"])))


def write_noise_png(path, width=32, height=16, seed=0):
    pixels = numpy.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=numpy.uint8
    )
    Image.fromarray(pixels).save(path)


@pytest.fixture
def outputs(tmp_path):
    folder = tmp_path / "outputs"
    folder.mkdir(exist_ok=True)
    return folder


def post(client, **body):
    return client.post("/api/gallery/compare", json=body)


def test_two_identical_videos_compare_as_identical(server, outputs):
    """The happy path: a grid PNG, the region, four default columns and a
    single metrics entry that reads identical."""
    write_ramp_mp4(outputs / "a.mp4")
    write_ramp_mp4(outputs / "b.mp4")

    with server(success_script) as client:
        response = post(client, names=["a.mp4", "b.mp4"])

        assert response.status_code == 200
        body = response.json()
        assert body["image"]["mime_type"] == "image/png"
        assert decode(body["image"]).size == (
            body["image"]["width"],
            body["image"]["height"],
        )
        assert len(body["metrics"]) == 1
        entry = body["metrics"][0]
        assert entry["name"] == "b.mp4" and entry["row"] == 1
        assert entry["psnr_db"]["mean"] == "identical"
        assert entry["ssim"]["mean"] == 1.0
        assert body["region"] == "full frame"
        assert [c["frame"] for c in body["columns"]] == [0, 8, 15, 23]
        assert [row["name"] for row in body["rows"]] == ["a.mp4", "b.mp4"]
        assert body["baseline"] == "a.mp4"
        assert body["downscaled_to"] is None


def test_a_moment_past_the_shorter_clip_is_a_blank_cell(server, outputs):
    """Only the shorter clip's missing frames go unmeasured."""
    write_ramp_mp4(outputs / "long.mp4", frames=24)
    write_ramp_mp4(outputs / "short.mp4", frames=12)

    with server(success_script) as client:
        body = post(
            client, names=["long.mp4", "short.mp4"], at=["frame:0", "frame:20"]
        ).json()

        cells = body["metrics"][0]["cells"]
        assert cells[0]["ssim"] == 1.0
        assert cells[1]["reason"].startswith("past end (12 frames)")
        assert cells[1]["psnr_db"] is None and cells[1]["ssim"] is None


def test_every_moment_past_every_video_is_refused(server, outputs):
    """With nothing to show at any column the request is the caller's error."""
    write_ramp_mp4(outputs / "a.mp4", frames=12)
    write_ramp_mp4(outputs / "b.mp4", frames=12)

    with server(success_script) as client:
        response = post(client, names=["a.mp4", "b.mp4"], at=["frame:20"])

        assert response.status_code == 400
        assert "past the end" in response.json()["detail"]


def test_a_size_mismatch_gets_no_numbers_and_says_why(server, outputs):
    """Resizing to force a comparison would itself be a difference."""
    write_ramp_mp4(outputs / "big.mp4", width=64, height=32)
    write_ramp_mp4(outputs / "small.mp4", width=32, height=16)

    with server(success_script) as client:
        response = post(client, names=["big.mp4", "small.mp4"])

        assert response.status_code == 200
        entry = response.json()["metrics"][0]
        assert entry["reason"] == "size differs from baseline: 32x16 vs 64x32"
        assert entry["psnr_db"] is None and entry["ssim"] is None
        assert entry["mean_abs_diff"] is None


def test_a_shared_crop_is_the_region_measured(server, outputs):
    """One box is shown and measured, and the answer names it."""
    write_ramp_mp4(outputs / "a.mp4")
    write_ramp_mp4(outputs / "b.mp4")

    with server(success_script) as client:
        body = post(client, names=["a.mp4", "b.mp4"], crop=[0, 0, 32, 16]).json()

        assert body["region"] == "crop [0, 0, 32, 16]"
        assert [row["crop"] for row in body["rows"]] == [[0, 0, 32, 16]] * 2


def test_per_output_crops_are_display_only(server, outputs):
    """One box per name changes what each row shows; the numbers stay those
    of the full frame."""
    write_ramp_mp4(outputs / "a.mp4")
    write_noise_png(outputs / "b.png")

    with server(success_script) as client:
        plain = post(client, names=["a.mp4", "b.png"]).json()
        cropped = post(
            client,
            names=["a.mp4", "b.png"],
            crop=[[0, 0, 16, 8], [8, 4, 16, 8]],
        ).json()

        assert plain["metrics"][0]["ssim"]["mean"] < 1.0
        assert cropped["region"] == "full frame (per-output crops are display only)"
        assert cropped["metrics"] == plain["metrics"]
        assert cropped["rows"][1]["crop"] == [8, 4, 16, 8]
        assert cropped["image"]["width"] < plain["image"]["width"]


def test_a_crop_out_of_bounds_for_one_output_is_refused(server, outputs):
    """The refusal names the output the box does not fit."""
    write_ramp_mp4(outputs / "a.mp4", width=32, height=16)
    write_ramp_mp4(outputs / "b.mp4", width=16, height=16)

    with server(success_script) as client:
        response = post(
            client, names=["a.mp4", "b.mp4"], crop=[[0, 0, 32, 16], [0, 0, 32, 16]]
        )

        assert response.status_code == 400
        assert "out of bounds for b.mp4" in response.json()["detail"]


def test_a_crop_list_of_the_wrong_length_is_refused(server, outputs):
    """One box for every output, or one for all."""
    write_ramp_mp4(outputs / "a.mp4")
    write_ramp_mp4(outputs / "b.mp4")

    with server(success_script) as client:
        response = post(client, names=["a.mp4", "b.mp4"], crop=[[0, 0, 4, 4]] * 3)

        assert response.status_code == 400
        assert "3 boxes for 2 names" in response.json()["detail"]


def test_one_name_is_refused(server, outputs):
    """A comparison needs at least two outputs."""
    write_ramp_mp4(outputs / "a.mp4")

    with server(success_script) as client:
        assert post(client, names=["a.mp4"]).status_code == 400


def test_an_unknown_name_is_not_found(server, outputs):
    """An output that is not there is a 404."""
    write_ramp_mp4(outputs / "a.mp4")

    with server(success_script) as client:
        assert post(client, names=["a.mp4", "missing.mp4"]).status_code == 404


@pytest.mark.parametrize("filename", ["tone.wav", "notes.txt"])
def test_audio_and_text_are_not_comparable(server, outputs, filename):
    """Only stills and videos compare."""
    write_ramp_mp4(outputs / "a.mp4")
    (outputs / filename).write_bytes(b"RIFF....WAVE")

    with server(success_script) as client:
        response = post(client, names=["a.mp4", filename])

        assert response.status_code == 400
        assert "not an image or a video" in response.json()["detail"]


@pytest.mark.parametrize("filename", ["broken.png", "broken.mp4"])
def test_an_undecodable_file_is_unprocessable(server, outputs, filename):
    """A file that resolved but will not decode is the file's problem."""
    write_ramp_mp4(outputs / "a.mp4")
    (outputs / filename).write_bytes(b"garbage" * 50)

    with server(success_script) as client:
        assert post(client, names=["a.mp4", filename]).status_code == 422


def test_an_oversized_video_is_refused(server, outputs, monkeypatch):
    """A frame past the decode ceiling is a 413 before any decoding."""
    write_ramp_mp4(outputs / "a.mp4")
    write_ramp_mp4(outputs / "b.mp4")
    monkeypatch.setattr("dw.server.routes.media.MAX_DECODE_PIXELS", 100)

    with server(success_script) as client:
        assert post(client, names=["a.mp4", "b.mp4"]).status_code == 413


def test_an_oversized_image_is_refused(server, outputs, monkeypatch):
    """A still's header past the ceiling is a 413 as well."""
    write_ramp_mp4(outputs / "a.mp4")
    write_noise_png(outputs / "b.png")
    monkeypatch.setattr("dw.server.inline_media.MAX_DECODE_PIXELS", 100)

    with server(success_script) as client:
        assert post(client, names=["a.mp4", "b.png"]).status_code == 413


def test_a_byte_budget_shrinks_the_whole_grid(server, outputs):
    """`downscaled_to` says what the budget shrank the grid to; with no
    budget it is None."""
    write_ramp_mp4(outputs / "a.mp4", width=512, height=256)
    write_noise_png(outputs / "b.png", width=512, height=256)
    names = ["a.mp4", "b.png"]

    with server(success_script) as client:
        full = post(client, names=names, max_dimension=512).json()
        small = post(
            client,
            names=names,
            max_dimension=512,
            max_total_bytes=len(full["image"]["data"]) // 4,
        ).json()

        assert full["downscaled_to"] is None
        assert small["downscaled_to"] is not None
        assert small["downscaled_to"] < max(
            full["image"]["width"], full["image"]["height"]
        )
        assert small["image"]["width"] < full["image"]["width"]
        assert len(small["image"]["data"]) < len(full["image"]["data"])
        assert decode(small["image"]).size == (
            small["image"]["width"],
            small["image"]["height"],
        )


def test_a_still_joins_a_video_comparison_in_every_column(server, outputs):
    """The still is the same image at each moment, measured per column."""
    write_ramp_mp4(outputs / "a.mp4")
    write_noise_png(outputs / "b.png")

    with server(success_script) as client:
        body = post(client, names=["a.mp4", "b.png"]).json()

        assert len(body["columns"]) == 4
        assert body["rows"][1]["kind"] == "image"
        entry = body["metrics"][0]
        assert len(entry["cells"]) == 4
        assert all(cell["ssim"] is not None for cell in entry["cells"])
        assert len({cell["psnr_db"] for cell in entry["cells"]}) > 1


def test_all_stills_make_one_column_and_ignore_at(server, outputs):
    """With no video there is no moment to name."""
    write_noise_png(outputs / "a.png", seed=1)
    write_noise_png(outputs / "b.png", seed=2)

    with server(success_script) as client:
        body = post(client, names=["a.png", "b.png"], at=[0, 1]).json()

        assert len(body["columns"]) == 1
        assert body["columns"][0]["frame"] is None
        assert any("`at` ignored" in note for note in body["notes"])


def test_the_baseline_can_be_the_second_name(server, outputs):
    """The baseline row is excluded from the metrics."""
    write_ramp_mp4(outputs / "a.mp4")
    write_ramp_mp4(outputs / "b.mp4")

    with server(success_script) as client:
        body = post(client, names=["a.mp4", "b.mp4"], baseline="b.mp4").json()

        assert body["baseline"] == "b.mp4"
        assert [entry["name"] for entry in body["metrics"]] == ["a.mp4"]
        assert body["metrics"][0]["row"] == 0


def test_an_asset_reference_compares_with_an_output(asset_server, tmp_path, outputs):
    """`asset:` names resolve through the asset library."""
    uploads = tmp_path / "assets" / "uploads"
    uploads.mkdir()
    write_noise_png(uploads / "ref.png")
    write_noise_png(outputs / "b.png")

    with asset_server(success_script) as client:
        response = post(client, names=["asset:uploads/ref.png", "b.png"])

        assert response.status_code == 200
        body = response.json()
        assert body["baseline"] == "asset:uploads/ref.png"
        assert body["metrics"][0]["ssim"]["mean"] == 1.0
