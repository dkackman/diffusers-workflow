"""`encode_h3_latents` (#1199 Task 17 Step 4): a finished take's pixels into
one MiniMax-H3 video latent, a VAE clip at a time.

The H3 video VAE encodes each 17-frame clip on its own and concatenates the
clips, so the chunked encode must equal diffusers' whole-take encode
(`encode_vae_condition`, what a `source_video` refine runs) bit for bit, on a
tiny VAE on CPU. No network and no real weights.
"""

import numpy
import pytest
import torch

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.models import AutoencoderKLMiniMaxH3  # noqa: E402
from diffusers.modular_pipelines.minimax_h3 import encoders  # noqa: E402

from dw.pipeline_processors.h3_mask_steps import fit_source_frames  # noqa: E402
from dw.result import AudioVideo  # noqa: E402
from dw.tasks import h3_latent_upscale as hlu  # noqa: E402
from dw.tasks.h3_latent_upscale import (  # noqa: E402
    decode_h3_latents,
    encode_h3_latents,
    encode_take,
)
from dw.tasks.model_cache import clear_model_cache  # noqa: E402

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
SEED = 42


@pytest.fixture(autouse=True)
def _clean_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()


def tiny_vae(tile=None):
    """tests/test_chain_h3_native.py's tiny H3 video VAE, with non-trivial
    latent statistics so the normalization is checked too."""
    torch.manual_seed(0)
    vae = AutoencoderKLMiniMaxH3(
        block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        latents_mean=tuple(0.01 * i for i in range(24)),
        latents_std=tuple(1.0 + 0.02 * i for i in range(24)),
    ).eval()
    if tile is not None:
        vae.enable_tiling(
            tile_sample_min_height=tile,
            tile_sample_min_width=tile,
            tile_sample_min_overlap_height=16,
            tile_sample_min_overlap_width=16,
        )
    return vae


def frames(count, height=32, width=64, seed=0):
    return numpy.random.default_rng(seed).integers(
        0, 255, (count, height, width, 3), "uint8"
    )


def whole_take(vae, pixels, num_frames):
    """What a `source_video` refine encodes: the take padded to the render's
    17m + 5 count by repeating its last frame, then encoded whole."""
    padded = fit_source_frames(pixels, num_frames)
    with torch.no_grad():
        return encoders.encode_vae_condition(
            vae,
            torch.from_numpy(numpy.ascontiguousarray(padded)).permute(3, 0, 1, 2)[None],
            PIXEL_MEAN,
            PIXEL_STD,
            SEED,
        )


def chunked(vae, pixels, reads=None):
    def read(start, stop):
        if reads is not None:
            reads.append((start, stop))
        return pixels[start:stop]

    return encode_take(
        vae, read, len(pixels), PIXEL_MEAN, PIXEL_STD, SEED, torch.device("cpu")
    )


# 1. Exactness


@pytest.mark.parametrize(
    "count, grid",
    [
        (22, 22),  # one clip and a 5-frame tail: on the grid
        (39, 39),  # two whole clips and a tail: on the grid
        (30, 39),  # off the grid: padded up to 39
        (23, 39),  # one past a grid count: the last clip is all padding
        (3, 5),  # shorter than the smallest grid count
    ],
)
def test_the_chunked_encode_equals_the_whole_take_encode(count, grid):
    vae = tiny_vae()
    pixels = frames(count)
    latents, num_frames = chunked(vae, pixels)
    assert num_frames == grid
    reference = whole_take(vae, pixels, grid)
    assert latents.shape == reference.shape
    assert latents.dtype == reference.dtype == torch.float32
    assert torch.equal(latents, reference)


def test_the_chunked_encode_is_exact_with_spatial_tiles_too():
    """A frame wider and taller than a tile: the VAE's own spatial tiling runs
    inside each clip, the same way in both encodes."""
    vae = tiny_vae(tile=32)
    pixels = frames(39, height=48, width=80, seed=1)
    latents, _ = chunked(vae, pixels)
    assert torch.equal(latents, whole_take(vae, pixels, 39))


# 2. Streaming: no read and no encode holds more than one clip


def test_frames_are_read_and_encoded_one_clip_at_a_time(monkeypatch):
    vae = tiny_vae()
    seen = []
    original = vae._encode_clip

    def spy(x):
        seen.append(x.shape[2])
        return original(x)

    monkeypatch.setattr(vae, "_encode_clip", spy)
    reads = []
    pixels = frames(56)  # 17 * 3 + 5: three clips and a tail
    chunked(vae, pixels, reads)
    assert seen == [17, 17, 17, 17]
    assert all(stop - start <= 17 for start, stop in reads)
    # Every real frame is read once, in order
    covered = [i for start, stop in reads for i in range(start, stop)]
    assert covered[: len(pixels)] == list(range(len(pixels)))


# 3. The task: sources, the record, refusals


def fake_decoder(vae):
    class Holder:
        _execution_device = torch.device("cpu")
        pixel_mean = PIXEL_MEAN
        pixel_std = PIXEL_STD
        keyframe_encode_seed = SEED

    holder = Holder()
    holder.vae = vae
    return holder


def test_the_task_returns_the_latent_and_records_the_take_length(monkeypatch):
    vae = tiny_vae()
    monkeypatch.setattr(hlu, "_load_decoder", lambda device, name: fake_decoder(vae))
    pixels = frames(30)
    take = AudioVideo(list(pixels), None, None, fps=24)

    result = encode_h3_latents(take)

    assert set(result) == {"latents", "num_frames", "source_frames"}
    assert result["num_frames"] == 39
    assert result["source_frames"] == 30
    assert torch.equal(result["latents"], whole_take(vae, pixels, 39))
    # (1, 24, 5m + 2, h / 16, w / 16): what a refine's `latents` takes
    assert tuple(result["latents"].shape) == (1, 24, 12, 2, 4)


def test_a_file_take_is_read_in_clip_ranges_and_matches_its_decode(
    monkeypatch, tmp_path
):
    from diffusers.utils.export_utils import encode_video
    from PIL import Image

    from dw import media_readers
    from dw.tasks.video_utils import load_audio_video

    pixels = frames(40, seed=3)
    path = str(tmp_path / "take.mp4")
    encode_video([Image.fromarray(p) for p in pixels], fps=24, output_path=path)
    decoded = numpy.stack([numpy.asarray(f) for f in load_audio_video(path).frames])

    vae = tiny_vae()
    monkeypatch.setattr(hlu, "_load_decoder", lambda device, name: fake_decoder(vae))
    ranges = []
    original = media_readers.read_frame_range

    def spy(file, start, stop):
        ranges.append((start, stop))
        return original(file, start, stop)

    monkeypatch.setattr(hlu, "read_frame_range", spy)

    result = encode_h3_latents(path)

    assert result["source_frames"] == len(decoded) == 40
    assert result["num_frames"] == 56
    assert ranges and all(stop - start <= 17 for start, stop in ranges)
    assert torch.equal(result["latents"], whole_take(vae, decoded, 56))


@pytest.mark.parametrize(
    "take, match",
    [
        (AudioVideo(list(frames(30)), None, None, fps=30), "24 fps"),
        (AudioVideo(list(frames(30, width=48)), None, None, fps=24), "multiple of 32"),
        (numpy.zeros((0, 32, 32, 3), numpy.uint8), "no frames"),
    ],
)
def test_a_take_the_refine_cannot_use_is_refused(monkeypatch, take, match):
    def no_load(device, name):
        raise AssertionError("refused before the VAE loads")

    monkeypatch.setattr(hlu, "_load_decoder", no_load)
    with pytest.raises(ValueError, match=match):
        encode_h3_latents(take)


# 3b. An untrusted header's frame count is bounded before anything is allocated


def _fake_header(monkeypatch, tmp_path, frames, fps=24.0, seconds=None):
    path = tmp_path / "take.mp4"
    path.write_bytes(b"x")
    monkeypatch.setattr(
        hlu,
        "video_shape",
        lambda p: {"frame_count": frames, "fps": fps, "width": 64, "height": 64},
    )
    monkeypatch.setattr(hlu, "media_duration", lambda p: seconds)

    def no_alloc(*a, **k):
        raise AssertionError("allocated before the frame count was bounded")

    monkeypatch.setattr(hlu, "_load_decoder", no_alloc)
    monkeypatch.setattr(torch, "empty", no_alloc)
    monkeypatch.setattr(hlu, "encode_take", no_alloc)
    return str(path)


def test_a_header_claiming_a_billion_frames_is_refused_before_any_allocation(
    monkeypatch, tmp_path
):
    path = _fake_header(monkeypatch, tmp_path, 10**9, seconds=10**9 / 24)
    with pytest.raises(ValueError, match="longest take"):
        encode_h3_latents(path)


def test_a_count_far_over_duration_times_fps_is_refused(monkeypatch, tmp_path):
    path = _fake_header(monkeypatch, tmp_path, 20000, seconds=5.0)
    with pytest.raises(ValueError, match="duration"):
        encode_h3_latents(path)


def test_a_count_slightly_over_duration_times_fps_is_clamped(monkeypatch, tmp_path):
    path = _fake_header(monkeypatch, tmp_path, 130, seconds=5.0)  # 120 expected
    seen = {}

    def load(device, name):
        raise RuntimeError("stop after the bound")

    monkeypatch.setattr(hlu, "_load_decoder", load)
    monkeypatch.setattr(
        hlu, "_take_problem", lambda n, *a: seen.setdefault("n", n) and None
    )
    with pytest.raises(RuntimeError, match="stop after"):
        encode_h3_latents(path)
    assert seen["n"] <= 121


# 4. The decode returns the take's own length


def test_the_decode_trims_back_to_the_takes_length(monkeypatch):
    from PIL import Image

    class FakePipeline:
        _execution_device = torch.device("cpu")

        def __call__(self, **kwargs):
            return [[Image.new("RGB", (16, 16))] * 39]

    monkeypatch.setattr(hlu, "_load_decoder", lambda device, name: FakePipeline())
    latents = torch.randn(1, 24, 12, 2, 4)
    assert len(decode_h3_latents(latents).frames) == 39
    assert len(decode_h3_latents(latents, num_frames=30).frames) == 30
    with pytest.raises(ValueError, match="39"):
        decode_h3_latents(latents, num_frames=40)


def test_the_decode_checks_its_count_before_decoding(monkeypatch):
    class Boom:
        _execution_device = torch.device("cpu")

        def __call__(self, **kwargs):
            raise AssertionError("decoded before the count was checked")

    monkeypatch.setattr(hlu, "_load_decoder", lambda device, name: Boom())
    latents = torch.randn(1, 24, 12, 2, 4)  # decodes to 17 * 2 + 5 = 39 frames
    with pytest.raises(ValueError, match="39"):
        decode_h3_latents(latents, num_frames=40)


@pytest.mark.parametrize("count", range(3, 80))
def test_decoded_frame_count_equals_the_real_decode(monkeypatch, count):
    from dw.tasks.h3_latent_upscale import decoded_frame_count

    vae = tiny_vae()
    ratio = vae.temporal_compression_ratio

    def stub(z):
        return torch.zeros(1, 3, z.shape[2] * ratio, 1, 1)

    monkeypatch.setattr(vae, "_decode_clip", stub)
    z = torch.zeros(1, 24, count, 1, 1)
    assert decoded_frame_count(count) == vae._decode(z).shape[2]


def test_a_valid_count_on_a_non_grid_latent_is_accepted(monkeypatch):
    from PIL import Image

    class FakePipeline:
        _execution_device = torch.device("cpu")

        def __call__(self, **kwargs):
            return [[Image.new("RGB", (16, 16))] * 26]

    monkeypatch.setattr(hlu, "_load_decoder", lambda device, name: FakePipeline())
    latents = torch.randn(1, 24, 8, 2, 4)  # 8 latents decode to 26 frames
    assert len(decode_h3_latents(latents, num_frames=24).frames) == 24


# 3c. The bound paths


def test_no_usable_duration_still_applies_the_frame_cap(monkeypatch, tmp_path):
    path = _fake_header(monkeypatch, tmp_path, hlu.MAX_TAKE_FRAMES + 1, seconds=None)
    with pytest.raises(ValueError, match="longest take"):
        encode_h3_latents(path)


@pytest.mark.parametrize("fps", [0, None])
def test_a_zero_or_missing_fps_skips_the_duration_check_but_keeps_the_cap(fps):
    from dw.tasks.h3_latent_upscale import bounded_take_frames

    assert bounded_take_frames(500, "x.mp4", fps) == 500
    with pytest.raises(ValueError, match="longest take"):
        bounded_take_frames(hlu.MAX_TAKE_FRAMES + 1, "x.mp4", fps)


def test_the_retry_recount_is_bounded_too(monkeypatch, tmp_path):
    from dw.tasks.h3_latent_upscale import ShortFrameRange

    path = _fake_header(monkeypatch, tmp_path, 100, seconds=None)
    monkeypatch.setattr(
        hlu,
        "_take_reader",
        lambda s: (None, 100, 24.0, path, (64, 64)),
    )
    monkeypatch.setattr(hlu, "_load_decoder", lambda d, n: SimpleNamespaceVae())
    calls = []

    def enc(*a, **k):
        calls.append(1)
        raise ShortFrameRange("x", 0, 1, 0)

    monkeypatch.setattr(hlu, "encode_take", enc)
    monkeypatch.setattr(hlu, "count_video_frames", lambda p: hlu.MAX_TAKE_FRAMES + 1)
    with pytest.raises(ValueError, match="longest take"):
        encode_h3_latents(path)
    assert len(calls) == 1


class SimpleNamespaceVae:
    vae = pixel_mean = pixel_std = keyframe_encode_seed = None
    _execution_device = torch.device("cpu")


def test_an_in_memory_take_over_the_cap_is_refused_before_allocation(monkeypatch):
    def no_alloc(*a, **k):
        raise AssertionError("allocated")

    monkeypatch.setattr(hlu, "_load_decoder", no_alloc)
    monkeypatch.setattr(torch, "empty", no_alloc)
    monkeypatch.setattr(hlu, "encode_take", no_alloc)
    monkeypatch.setattr(
        hlu,
        "_take_reader",
        lambda s: (None, hlu.MAX_TAKE_FRAMES + 1, 24.0, None, (64, 64)),
    )
    with pytest.raises(ValueError, match="longest take"):
        encode_h3_latents(object())


# 5. Registration


def test_the_command_is_registered_with_its_media_argument():
    import dw.tasks.model_handlers  # noqa: F401 - registers the task
    from dw.locations import TASK_MEDIA_ARGUMENTS
    from dw.tasks.task import _COMMAND_INFO

    assert _COMMAND_INFO["encode_h3_latents"]["summary"]
    assert TASK_MEDIA_ARGUMENTS.get("encode_h3_latents") == ("source",)


def test_a_source_outside_the_roots_is_refused_at_validate(tmp_path):
    import dw.tasks.model_handlers  # noqa: F401 - registers the task
    from dw.locations import location_errors

    workflow = {
        "id": "x",
        "steps": [
            {
                "name": "encode",
                "task": {
                    "command": "encode_h3_latents",
                    "arguments": {"source": "../../etc/take.mp4"},
                },
            }
        ],
    }
    errors = location_errors(workflow, base_dir=str(tmp_path))
    assert any("source" in e["path"] for e in errors), errors


# 6. The streamed decode: a piece of the take at a time, the same frames


def _streamed(vae, z, piece_clips):
    from dw.pipeline_processors.h3_windows import streamed_decode

    runs = []
    with torch.no_grad():
        streamed_decode(vae, z, runs.append, piece_clips)
    return runs


@pytest.mark.parametrize("piece_clips", [1, 2, 3])
@pytest.mark.parametrize("count", [3, 7, 8, 12, 13, 17, 22, 26, 27, 37, 42])
def test_the_streamed_decode_equals_the_whole_decode(piece_clips, count):
    vae = tiny_vae()
    z = torch.randn(1, 24, count, 2, 4, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        whole = vae.decode(z, return_dict=False)[0]
    runs = _streamed(vae, z, piece_clips)
    assert torch.equal(torch.cat(runs, dim=2), whole)


def test_a_long_take_decodes_in_several_pieces():
    vae = tiny_vae()
    z = torch.randn(1, 24, 37, 2, 4, generator=torch.Generator().manual_seed(8))
    # 37 latents are 7 clips: pieces of 2 clips run 2 + 2 + 2 + the last
    assert len(_streamed(vae, z, 2)) == 4


def test_each_piece_goes_through_the_vaes_own_decode(monkeypatch):
    """`decode` is where on-demand placement and offload hooks sit, so the
    pieces must not reach past it to the clip decoder."""
    vae = tiny_vae()
    calls = []
    real = vae.decode

    def spy(z, return_dict=True):
        calls.append(z.shape[2])
        return real(z, return_dict=return_dict)

    monkeypatch.setattr(vae, "decode", spy)
    z = torch.randn(1, 24, 22, 2, 4)
    _streamed(vae, z, 1)
    # Each piece is one clip of 5 latents plus the 2 its last clip also reads
    assert calls[:-1] == [7] * (len(calls) - 1)
    assert sum(1 for _ in calls) > 1


@pytest.mark.parametrize("piece_clips", [0, -1, 1.5, True, None])
def test_a_piece_size_that_is_not_a_whole_count_is_refused(piece_clips):
    from dw.pipeline_processors.h3_windows import streamed_decode

    with pytest.raises(ValueError, match="piece_clips"):
        streamed_decode(tiny_vae(), torch.zeros(1, 24, 7, 2, 4), print, piece_clips)


def _decode_pipeline(block, vae):
    pipeline = block.init_pipeline()
    pipeline.update_components(vae=vae)
    return pipeline


@pytest.mark.parametrize("output_type", ["pil", "np"])
def test_the_streamed_block_gives_the_stock_blocks_frames(output_type):
    from diffusers.modular_pipelines.minimax_h3.decoders import (
        MiniMaxH3VideoDecodeStep,
    )

    from dw.pipeline_processors.h3_windows import DwH3StreamedVideoDecodeStep

    vae = tiny_vae()
    latents = torch.randn(1, 24, 27, 2, 4, generator=torch.Generator().manual_seed(9))
    streamed = DwH3StreamedVideoDecodeStep()
    streamed.piece_clips = 1
    stock = _decode_pipeline(MiniMaxH3VideoDecodeStep(), vae)(
        latents=latents, output_type=output_type, output="videos"
    )
    ours = _decode_pipeline(streamed, vae)(
        latents=latents, output_type=output_type, output="videos"
    )
    if output_type == "np":
        assert ours.shape == stock.shape
        assert numpy.array_equal(ours, stock)
    else:
        assert len(ours) == len(stock) == 1
        assert len(ours[0]) == len(stock[0])
        for mine, theirs in zip(ours[0], stock[0]):
            assert numpy.array_equal(numpy.asarray(mine), numpy.asarray(theirs))


def test_decode_h3_latents_decodes_through_the_streamed_block(monkeypatch):
    from dw.pipeline_processors.h3_windows import DwH3StreamedVideoDecodeStep

    seen = {}

    def init_pipeline(self, *args, **kwargs):
        seen["block"] = type(self)
        raise RuntimeError("stop at the load")

    monkeypatch.setattr(DwH3StreamedVideoDecodeStep, "init_pipeline", init_pipeline)
    with pytest.raises(RuntimeError, match="stop at the load"):
        decode_h3_latents(torch.randn(1, 24, 12, 2, 4))
    assert seen["block"] is DwH3StreamedVideoDecodeStep


def test_a_vae_without_token_drop_decodes_whole():
    """No `token_drop` leaves no overlap tail to hold back; the stream must
    fall back to the stock decode rather than decode wrongly."""
    vae = tiny_vae()
    vae.token_overlap = 0
    z = torch.randn(1, 24, 10, 2, 4, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        whole = vae.decode(z, return_dict=False)[0]
    runs = _streamed(vae, z, 1)
    assert len(runs) == 1
    assert torch.equal(runs[0], whole)


def test_a_vfr_take_clamped_to_its_duration_warns(monkeypatch):
    # header claims 120 frames, duration x fps holds ~49 (#1232)
    from dw.tasks.h3_latent_upscale import bounded_take_frames

    warned = []
    monkeypatch.setattr(
        hlu, "emit_warning", lambda message, **d: warned.append(message)
    )
    monkeypatch.setattr(hlu, "media_duration", lambda path: 2.0)
    assert bounded_take_frames(120, "x.mp4", 24.0) == 49
    assert warned and "120" in warned[0] and "49" in warned[0]
    warned.clear()
    assert bounded_take_frames(49, "x.mp4", 24.0) == 49
    assert not warned
