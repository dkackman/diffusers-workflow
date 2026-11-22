import pytest


def test_encoding_a_video_with_audio_leaves_no_encoder_threads_behind(tmp_path):
    """diffusers' audio path leaves the encoder in a reference cycle, and
    libx264's native threads live until it is collected (#1568: a test worker
    ran out of threads and every later test failed `thread.__init__() not
    called`)."""
    import numpy
    import torch

    from dw.writers import encode_video

    psutil = pytest.importorskip("psutil")
    process = psutil.Process()
    video = torch.zeros(8, 64, 64, 3, dtype=torch.uint8)
    audio = torch.from_numpy(numpy.zeros((1, 16000), dtype=numpy.float32))
    encode_video(video, 24, str(tmp_path / "warm.mp4"), audio=audio, audio_sample_rate=16000)
    before = process.num_threads()
    for index in range(4):
        encode_video(
            video, 24, str(tmp_path / f"{index}.mp4"), audio=audio, audio_sample_rate=16000
        )
    assert process.num_threads() - before < 20
