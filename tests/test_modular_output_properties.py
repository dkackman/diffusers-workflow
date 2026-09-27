"""A modular step asked for several outputs (`output: [videos, audio,
sampling_rate, latents]`) saves the muxed video/audio as one AudioVideo
artifact and carries anything else - `latents`, say - as a leftover dict
artifact beside it (`modular_artifacts`, dw/result.py). `get_artifact_properties`
has to look past the AudioVideo to find a property that only the dict carries
(#499), rather than raising as soon as it sees a result with no such
attribute.
"""

import pytest
import torch

from dw.previous_results import get_previous_results
from dw.result import AudioVideo, Result


def modular_result():
    """A Result shaped like modular_artifacts leaves one behind: the paired
    video/audio artifact, plus a leftover dict for any other requested output."""
    result = Result({"content_type": "video/mp4"})
    frames = ["frame1", "frame2"]
    audio = torch.zeros((2, 100))
    latents = torch.randn(1, 24, 5, 34, 60)
    result.add_result([AudioVideo(frames, audio, 48000), {"latents": latents}])
    return result, frames, audio, latents


class TestGetArtifactPropertiesAcrossModularArtifacts:
    def test_a_property_only_the_dict_carries_is_found(self):
        result, _, _, latents = modular_result()
        values = result.get_artifact_properties("latents")
        assert len(values) == 1
        assert torch.equal(values[0], latents)

    def test_a_property_only_the_audio_video_carries_is_found(self):
        result, frames, audio, _ = modular_result()

        assert result.get_artifact_properties("frames") == [frames]
        audio_values = result.get_artifact_properties("audio")
        assert len(audio_values) == 1
        assert torch.equal(audio_values[0], audio)

    def test_sample_rate_comes_off_the_audio_video(self):
        result, _, _, _ = modular_result()
        assert result.get_artifact_properties("sample_rate") == [48000]

    def test_a_property_nobody_has_still_raises(self):
        result, _, _, _ = modular_result()
        with pytest.raises(ValueError, match="nonexistent"):
            result.get_artifact_properties("nonexistent")


class TestThroughThePreviousResultResolutionPath:
    def test_previous_result_step_dot_latents_reaches_the_dict(self):
        result, _, _, latents = modular_result()
        previous_results = {"base": result}

        values = get_previous_results(previous_results, "base.latents")
        assert len(values) == 1
        assert torch.equal(values[0], latents)

    def test_previous_result_step_dot_sample_rate_reaches_the_audio_video(self):
        result, _, _, _ = modular_result()
        previous_results = {"base": result}

        assert get_previous_results(previous_results, "base.sample_rate") == [48000]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
