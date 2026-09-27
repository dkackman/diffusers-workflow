"""A modular step asked for several outputs (`output: [videos, audio,
sampling_rate, latents]`) keeps the dict the pipeline returned as its result
(step.py adds each iteration's raw output), and `previous_result:<step>.<key>`
names a key of that dict, spelled as the pipeline spells it (#499).

The pairing into an AudioVideo (`modular_artifacts`, dw/result.py) happens only
when the result is saved or read whole, so `sample_rate` - the AudioVideo's name
for it - is not a key the dict has. Asking for it used to come back as an empty
list, which gave the step reading it zero iterations: the documented promotion
workflow's `mux` step succeeded having written nothing (M-F041). A property no
result carries is now an error.
"""

import json
import re
from pathlib import Path

import numpy
import pytest
import torch
from PIL import Image

from dw.previous_results import get_iterations, get_previous_results
from dw.result import AudioVideo, Result
from dw.tasks.pair_audio import pair_audio

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "WORKFLOW_GUIDE.md"
HEADING = "### Promoting an H3 take to 768p in latent space"

SAMPLE_RATE = 48000
FPS = 24
FRAME_COUNT = 5


def frames(width=64, height=32):
    return [Image.new("RGB", (width, height), (i * 40, 0, 0)) for i in range(FRAME_COUNT)]


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


def documented_workflow():
    """The promotion workflow docs/WORKFLOW_GUIDE.md shows, parsed from the
    guide itself, so the test runs what an agent reading it would copy."""
    text = GUIDE.read_text(encoding="utf-8")
    section = text[text.index(HEADING) :]
    block = re.search(r"```json\n(.*?)\n```", section, re.DOTALL).group(1)
    return json.loads(block)


def step_arguments(workflow, name):
    step = next(step for step in workflow["steps"] if step["name"] == name)
    return step


class TestTheDocumentedMuxRuns:
    """The real resolution path, from the guide's own JSON: the mux step gets
    one iteration and pair_audio writes a video with a soundtrack."""

    def previous_results(self):
        base, _ = modular_result()
        decode = Result({})
        decode.add_result(AudioVideo(frames(128, 64), None, None, fps=FPS))
        return {"base": base, "decode": decode}

    def test_every_property_the_guide_reads_off_base_is_an_output_it_asks_for(self):
        workflow = documented_workflow()
        base = step_arguments(workflow, "base")
        requested = set(base["pipeline"]["arguments"]["output"])
        text = json.dumps(workflow)
        read = set(re.findall(r"previous_result:base\.(\w+)", text))
        assert read
        assert read <= requested

    def test_the_mux_step_gets_exactly_one_iteration(self):
        mux = step_arguments(documented_workflow(), "mux")
        iterations = get_iterations(mux["task"]["arguments"], self.previous_results())
        assert len(iterations) == 1
        assert iterations[0]["sample_rate"] == SAMPLE_RATE

    def test_the_mux_step_saves_a_video_with_audio(self, tmp_path):
        mux = step_arguments(documented_workflow(), "mux")
        (arguments,) = get_iterations(mux["task"]["arguments"], self.previous_results())

        paired = pair_audio(**arguments)
        assert isinstance(paired, AudioVideo)
        assert paired.audio is not None
        assert paired.sample_rate == SAMPLE_RATE

        result = Result(mux["result"])
        result.add_result(paired)
        result.save(str(tmp_path), "mux")

        written = list(tmp_path.rglob("*.mp4"))
        assert len(written) == 1
        assert written[0].stat().st_size > 0

    def test_a_misspelt_property_fails_the_step_instead_of_skipping_it(self):
        mux = step_arguments(documented_workflow(), "mux")
        arguments = dict(mux["task"]["arguments"])
        arguments["sample_rate"] = "previous_result:base.sample_rate"
        with pytest.raises(ValueError, match="sample_rate"):
            get_iterations(arguments, self.previous_results())


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
