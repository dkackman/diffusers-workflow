"""A failure message's library paths, named back as references."""

import os

from dw.path_redaction import redact_paths


def test_a_file_under_an_asset_root_becomes_its_reference():
    text = redact_paths("/home/u/ws/assets/a/b.mp4 is silent", ["/home/u/ws/assets"])
    assert text == "asset:a/b.mp4 is silent"


def test_a_file_under_the_output_dir_becomes_its_reference():
    text = redact_paths("read /srv/out/id/run/x.png", output_dir="/srv/out")
    assert text == "read output:id/run/x.png"


def test_a_bare_root_is_named_not_shown():
    text = redact_paths("searched /home/u/ws/assets.", ["/home/u/ws/assets"])
    assert text == "searched the asset library."


def test_a_sibling_sharing_the_prefix_is_left_alone():
    text = redact_paths("/home/u/ws/assets2/x", ["/home/u/ws/assets"])
    assert text == "/home/u/ws/assets2/x"


def test_the_nested_root_wins():
    text = redact_paths(
        "/w/assets/common/x.png", ["/w/assets", "/w/assets/common"], None
    )
    assert text == "asset:x.png"


def test_a_resolved_spelling_is_redacted_too(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    text = redact_paths(f"{os.path.realpath(real)}/x.wav", [str(link)])
    assert text == "asset:x.wav"


def test_nothing_to_redact_returns_the_text():
    assert redact_paths(None, ["/a"]) is None
    assert redact_paths("plain", ["/a"], "/b") == "plain"


def test_a_name_with_a_space_still_loses_its_root():
    text = redact_paths("/w/assets/my clip.mp4 is silent", ["/w/assets"])
    assert text == "asset:my clip.mp4 is silent"
    assert "/w/assets" not in text


def test_a_quoted_path_keeps_its_quotes():
    text = redact_paths("cannot read '/w/assets/a.wav'", ["/w/assets"])
    assert text == "cannot read 'asset:a.wav'"
