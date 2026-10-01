"""A modular step asked for several outputs (`output: [videos, audio,
sampling_rate, latents]`) keeps the dict the pipeline returned as its result
(step.py adds each iteration's raw output), and `previous_result:<step>.<key>`
names a key of that dict, spelled as the pipeline spells it (#499).

The pairing into an AudioVideo (`modular_artifacts`, dw/output_extraction.py) happens only
when the result is saved or read whole, so `sample_rate` - the AudioVideo's name
for it - is not a key the dict has. Asking for it used to come back as an empty
list, which gave the step reading it zero iterations, succeeding having written
nothing. A property no result carries is now an error.
"""

import numpy
import pytest
import torch
from PIL import Image

from dw.previous_results import get_iterations, get_previous_results
from dw.media_types import AudioVideo
from dw.result import Result
from dw.tasks.pair_audio import pair_audio

SAMPLE_RATE = 48000
FPS = 24
FRAME_COUNT = 5


def frames(width=64, height=32):
    return [
        Image.new("RGB", (width, height), (i * 40, 0, 0)) for i in range(FRAME_COUNT)
    ]


def modular_output():
    """What a modular H3 pipeline hands back for `output: [videos, audio,
    sampling_rate, latents]`: one dict, batched the way diffusers batches it."""
    samples = int(FRAME_COUNT / FPS * SAMPLE_RATE)
    return {
        "videos": [frames()],
        "audio": torch.zeros((1, 2, samples)),
        "sampling_rate": SAMPLE_RATE,
        "latents": torch.randn(1, 24, 2, 4, 8),
    }


def modular_result():
    """The base step's Result exactly as Step.run leaves it."""
    result = Result({})
    output = modular_output()
    result.add_result(output)
    return result, output


class TestAModularStepsPropertiesAreItsOutputKeys:
    def test_the_result_list_holds_the_raw_dict(self):
        result, output = modular_result()
        assert result.result_list == [output]

    @pytest.mark.parametrize("key", ["latents", "audio", "sampling_rate", "videos"])
    def test_every_requested_output_is_reachable_by_its_own_name(self, key):
        result, output = modular_result()
        values = get_previous_results({"base": result}, f"base.{key}")
        assert len(values) == 1
        assert values[0] is output[key]

    def test_sample_rate_is_not_a_key_and_says_what_is(self):
        result, _ = modular_result()
        with pytest.raises(ValueError) as raised:
            get_previous_results({"base": result}, "base.sample_rate")
        message = str(raised.value)
        assert "'sample_rate'" in message
        assert "sampling_rate" in message

    def test_a_property_nobody_has_raises(self):
        result, _ = modular_result()
        with pytest.raises(ValueError, match="nonexistent"):
            result.get_artifact_properties("nonexistent")

    def test_a_property_one_result_has_is_found_past_one_that_lacks_it(self):
        result = Result({})
        latents = torch.randn(1, 2)
        result.add_result([AudioVideo(frames(), None, None), {"latents": latents}])
        values = result.get_artifact_properties("latents")
        assert len(values) == 1
        assert values[0] is latents


def test_the_batched_audio_reaches_pair_audio_as_one_track():
    """diffusers batches the soundtrack (1, channels, samples); pair_audio
    takes the one-item batch as it is."""
    output = modular_output()
    paired = pair_audio(
        AudioVideo(frames(), None, None, fps=FPS),
        output["audio"],
        sample_rate=output["sampling_rate"],
        fit="video",
    )
    assert numpy.asarray(paired.audio).ndim == 2


class TestTheBasesOwnTakeSavesBesideIt:
    """`previous_result:base.videos` is the pipeline's batch - a list holding
    one frame list. pair_audio unwraps it; passed through, the encoder took
    the batch for a frame list and failed at save (#499, third bounce)."""

    def reference_step(self):
        return {
            "task": {
                "command": "pair_audio",
                "arguments": {
                    "video": "previous_result:base.videos",
                    "audio": "previous_result:base.audio",
                    "sample_rate": "previous_result:base.sampling_rate",
                    "fit": "video",
                },
            },
            "result": {
                "content_type": "video/mp4",
                "fps": FPS,
                "subfolder": "intermediate",
            },
        }

    def test_the_base_videos_save_as_one_mp4_with_its_frames_and_audio(self, tmp_path):
        import av

        base, _ = modular_result()
        step = self.reference_step()
        (arguments,) = get_iterations(step["task"]["arguments"], {"base": base})

        paired = pair_audio(**arguments)
        assert isinstance(paired.frames, list)
        assert len(paired.frames) == FRAME_COUNT

        result = Result(step["result"])
        result.add_result(paired)
        result.save(str(tmp_path), "reference")

        (written,) = tmp_path.rglob("*.mp4")
        with av.open(str(written)) as container:
            assert container.streams.audio
            decoded = sum(1 for _ in container.decode(video=0))
        assert decoded == FRAME_COUNT

    def test_a_five_dimensional_batch_of_one_is_unwrapped(self):
        video = numpy.zeros((1, FRAME_COUNT, 32, 64, 3), dtype=numpy.float32)
        paired = pair_audio(video, torch.zeros((1, 2, 100)), sample_rate=SAMPLE_RATE)
        assert paired.frames.shape == (FRAME_COUNT, 32, 64, 3)

    def test_a_frame_list_is_left_alone(self):
        video = frames()
        paired = pair_audio(video, torch.zeros((1, 2, 100)), sample_rate=SAMPLE_RATE)
        assert paired.frames is video

    def test_a_list_of_array_frames_is_not_mistaken_for_a_batch(self):
        video = [numpy.zeros((32, 64, 3), dtype=numpy.uint8)] * FRAME_COUNT
        paired = pair_audio(video, torch.zeros((1, 2, 100)), sample_rate=SAMPLE_RATE)
        assert paired.frames is video

    def test_fit_measures_the_unwrapped_video(self):
        paired = pair_audio(
            [frames()],
            torch.zeros((1, 2, 10)),
            sample_rate=SAMPLE_RATE,
            fps=FPS,
            fit="video",
        )
        assert paired.audio.shape[1] == int(round(FRAME_COUNT / FPS * SAMPLE_RATE))

    @pytest.mark.parametrize(
        "video",
        [
            [frames(), frames()],
            numpy.zeros((2, FRAME_COUNT, 32, 64, 3), dtype=numpy.float32),
        ],
    )
    def test_a_batch_of_several_is_refused(self, video):
        with pytest.raises(ValueError, match="batch of 2 videos"):
            pair_audio(video, torch.zeros((1, 2, 100)), sample_rate=SAMPLE_RATE)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
