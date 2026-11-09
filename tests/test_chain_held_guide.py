"""A guide chain held to a narration (match_audio + a step hold_audio).

Each later segment's held slice already starts at the guide span (the plan's
head trim is the guide length), so the guide must carry video only: a guide
that also carried the previous segment's audio would hold the overlap twice,
once as guide audio rows and once as the held rows written after them.
Plus the two validate warnings a held chain earns: a hold with no
match_audio (every segment holds the whole track from its start) and a
segment length off H3's 3-frame audio grid.
"""

import pytest
import torch
from PIL import Image

from dw.media_types import AudioVideo
from dw.pipeline_processors.chain import ChainConfig, run_chain
from dw.task_domains import frames_to_samples

MINIMAX_SNAP = {"modulus": 17, "remainder": 5, "min_frames": 124, "max_frames": 345}
FPS = 24
# 100 samples a frame keeps every boundary a whole sample
RATE = 2400


class FakePipeline:
    def __init__(self, output_factory):
        self.output_factory = output_factory
        self.output_dir = None
        self.file_prefix = None
        self.calls = []

    def _run_once(self, arguments):
        self.calls.append(arguments)
        return self.output_factory(arguments, len(self.calls) - 1)


def output(arguments, index):
    """Each frame's colour names its segment and frame; the generated audio
    is a constant naming the segment, never the held track's ramp."""
    num_frames = arguments.get("num_frames", 192)
    frames = [
        Image.new("RGB", (8, 8), (index * 50, frame % 256, 7))
        for frame in range(num_frames)
    ]
    samples = int(num_frames / FPS * RATE)
    audio = torch.full((1, 2, samples), -float(index + 1))
    return {"videos": [frames], "audio": audio, "sampling_rate": RATE}


@pytest.fixture
def reference_type():
    references = pytest.importorskip(
        "diffusers.modular_pipelines.minimax_h3.references"
    )
    return references.MiniMaxH3AudioReference


def ramp(samples):
    """A track whose every sample says where it came from."""
    return torch.arange(samples, dtype=torch.float32).repeat(2, 1)


def chain(**overrides):
    return {
        "match_audio": True,
        "continuity": "guide",
        "fps": FPS,
        "frame_snap": MINIMAX_SNAP,
    } | overrides


def arguments(held, num_frames=192):
    return {"prompt": "test", "num_frames": num_frames, "hold_audio": held}


# 192 + 2 * (192 - 22) frames: three whole segments
THREE_SEGMENTS = (192 + 2 * 170) * 100


def test_held_chain_guide_carries_no_audio(reference_type):
    held = reference_type(audio=ramp(THREE_SEGMENTS), sample_rate=RATE)
    pipeline = FakePipeline(output)

    run_chain(pipeline, chain(), arguments(held))

    assert len(pipeline.calls) == 3
    for call in pipeline.calls[1:]:
        (guide,) = call["guides"]
        assert guide.get("audio") is not True
        assert "audio_latents" not in guide
        # the guide's video carries frames alone, no soundtrack
        assert not isinstance(guide["video"], AudioVideo)
        assert len(guide["video"]) == 22


def test_held_chain_forces_carry_audio_off_and_says_so_once(
    reference_type, monkeypatch
):
    from dw.pipeline_processors import chain_config

    logged = []
    monkeypatch.setattr(chain_config, "emit_log", logged.append)
    held = reference_type(audio=ramp(THREE_SEGMENTS), sample_rate=RATE)

    config = ChainConfig(chain(carry_audio=True), arguments(held))

    assert config.carry_audio is False
    assert config.guide_holds_audio is False
    assert len([line for line in logged if "carry_audio" in line]) == 1


def test_an_unheld_guide_chain_still_carries_its_audio():
    config = ChainConfig(
        {"segments": 3, "continuity": "guide", "fps": FPS}, {"num_frames": 192}
    )

    assert config.carry_audio is True
    assert config.guide_holds_audio is True


def test_held_guide_slices_start_at_the_guide_span(reference_type):
    held = reference_type(audio=ramp(THREE_SEGMENTS), sample_rate=RATE)
    pipeline = FakePipeline(output)

    run_chain(pipeline, chain(), arguments(held))

    starts = [int(call["hold_audio"].audio[0, 0]) for call in pipeline.calls]
    step = frames_to_samples(192 - 22, FPS, RATE)
    assert starts == [0, step, 2 * step]
    samples = frames_to_samples(192, FPS, RATE)
    assert all(
        call["hold_audio"].audio.shape == (2, samples) for call in pipeline.calls
    )


def test_take_ends_at_the_narration_end(reference_type):
    # 20 s: 192 + 170 frames, then 118 more, snapped up to a 141-frame segment
    seconds = 20
    track = ramp(seconds * RATE)
    held = reference_type(audio=track, sample_rate=RATE)
    pipeline = FakePipeline(output)

    result = run_chain(pipeline, chain(), arguments(held))

    assert [call["num_frames"] for call in pipeline.calls] == [192, 192, 141]
    assert len(result.frames) == round(seconds * FPS)
    assert result.sample_rate == RATE
    assert torch.equal(torch.as_tensor(result.audio), track)


# --- validate warnings ------------------------------------------------------


def _step(chain_block, num_frames=192, held="narration.wav"):
    return {
        "name": "take",
        "pipeline": {
            "chain": chain_block,
            "arguments": {"prompt": "p", "num_frames": num_frames, "hold_audio": held},
        },
    }


def test_hold_without_match_audio_warns():
    from dw.guides import held_audio_warnings

    (warning,) = held_audio_warnings(
        {"steps": [_step({"segments": 3, "continuity": "guide"})]}
    )

    assert warning.startswith("steps[0].pipeline.chain.match_audio:")
    assert "every segment holds the whole track from its start" in warning
    assert "set match_audio: true to give each segment its slice" in warning


def test_a_matched_hold_on_the_grid_does_not_warn():
    from dw.guides import held_audio_warnings

    assert held_audio_warnings({"steps": [_step(chain())]}) == []


def test_off_grid_held_chain_warns():
    from dw.guides import held_audio_warnings

    (warning,) = held_audio_warnings({"steps": [_step(chain(), num_frames=209)]})

    assert warning.startswith("steps[0].pipeline.arguments.num_frames:")
    assert "H3's audio latents run at 40 a second" in warning
    assert "209 frames is off the 3-frame audio grid" in warning
    assert "141, 192, 243, 294, 345" in warning
    assert held_audio_warnings({"steps": [_step(chain(), num_frames=192)]}) == []


def test_an_unheld_or_unresolved_chain_does_not_warn():
    from dw.guides import held_audio_warnings

    unheld = _step({"segments": 3}, num_frames=209, held=None)
    unresolved = _step(chain(), num_frames="variable:frames")
    no_chain = {"name": "s", "pipeline": {"arguments": {"hold_audio": "a.wav"}}}

    assert held_audio_warnings({"steps": [unheld, unresolved, no_chain]}) == []


def test_the_held_audio_warning_check_is_registered():
    from dw import validation

    assert "held_audio_warnings" in [c.name for c in validation.WARNING_CHECKS]
