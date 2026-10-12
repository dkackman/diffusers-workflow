"""An MP4 output keeps its index (moov) ahead of its media data, and the
gallery grid gets a video's first frame as a thumbnail - a grid of clips
must not open one stream per tile (dw.media.faststart,
dw.media_frames.first_frame)."""

import logging
import struct

import av
import torch
from PIL import Image

from dw import media, media_frames
from dw.result import Result


def _top_level_atoms(path):
    atoms = []
    with open(path, "rb") as f:
        position = 0
        while header := f.read(8):
            if len(header) < 8:
                break
            size, kind = struct.unpack(">I4s", header)
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
            atoms.append(kind.decode())
            position += size
            f.seek(position)
    return atoms


def _write_clip(path, num_frames=12, fps=8, sample_rate=8000, size=(32, 16)):
    from diffusers.utils.export_utils import encode_video

    frames = [Image.new("RGB", size, (20 * i, 40, 0)) for i in range(num_frames)]
    audio = torch.full((2, int(num_frames / fps * sample_rate)), 0.25)
    encode_video(
        frames,
        fps=fps,
        output_path=str(path),
        audio=audio,
        audio_sample_rate=sample_rate,
    )
    return str(path)


def _shape(path):
    with av.open(path) as container:
        return {
            "video_frames": sum(1 for _ in container.decode(video=0)),
            "audio_streams": len(container.streams.audio),
        }


def test_faststart_moves_the_index_ahead_of_the_media(tmp_path):
    path = _write_clip(tmp_path / "clip.mp4")
    before = _shape(path)
    atoms = _top_level_atoms(path)
    # the writer's own order - the reason faststart exists
    assert atoms.index("moov") > atoms.index("mdat")

    media.faststart(path)

    atoms = _top_level_atoms(path)
    assert atoms.index("moov") < atoms.index("mdat")
    assert _shape(path) == before
    assert not (tmp_path / "clip.mp4.faststart").exists()


def test_faststart_keeps_a_file_it_cannot_read(tmp_path, caplog):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"\x00" * 64)

    with caplog.at_level(logging.WARNING, logger="dw"):
        media.faststart(str(path))

    assert path.read_bytes() == b"\x00" * 64
    assert not (tmp_path / "broken.mp4.faststart").exists()
    assert "Kept" in caplog.text


def test_a_saved_mp4_output_is_faststart(tmp_path):
    frames = [Image.new("RGB", (32, 32), (i * 10, 0, 0)) for i in range(8)]
    result = Result({"content_type": "video/mp4", "fps": 8})
    result.add_result({"videos": [frames]})

    result.save(str(tmp_path), "clip")

    [saved] = list(tmp_path.rglob("*.mp4"))
    atoms = _top_level_atoms(saved)
    assert atoms.index("moov") < atoms.index("mdat")


def test_first_frame_is_the_clips_first_frame(tmp_path):
    path = _write_clip(tmp_path / "clip.mp4", size=(32, 16))

    image = media_frames.first_frame(path)

    assert image.size == (32, 16)
    # frame 0 is (0, 40, 0) - green, no red; frame 11 would be red 220
    red, green, _ = image.getpixel((16, 8))
    assert red < 30 and green > 20
