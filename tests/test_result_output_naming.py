import os

from dw.result import output_file_path


def test_output_file_path_no_collision_is_unchanged(tmp_path):
    path = output_file_path(str(tmp_path), "run-step.0-0.0.png")

    assert path == str(tmp_path / "run-step.0-0.0.png")


def test_output_file_path_dedupes_existing_file(tmp_path):
    (tmp_path / "run-step.0-0.0.png").write_bytes(b"first")

    path = output_file_path(str(tmp_path), "run-step.0-0.0.png")

    assert path == str(tmp_path / "run-step.0-0.0-2.png")


def test_output_file_path_dedupes_multiple_collisions(tmp_path):
    (tmp_path / "run-step.0-0.0.png").write_bytes(b"first")
    (tmp_path / "run-step.0-0.0-2.png").write_bytes(b"second")

    path = output_file_path(str(tmp_path), "run-step.0-0.0.png")

    assert path == str(tmp_path / "run-step.0-0.0-3.png")


def test_output_file_path_preserves_extension(tmp_path):
    (tmp_path / "clip.0-0.0.mp4").write_bytes(b"first")

    path = output_file_path(str(tmp_path), "clip.0-0.0.mp4")

    assert path == str(tmp_path / "clip.0-0.0-2.mp4")
