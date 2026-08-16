"""
Unit tests for result module
Tests result storage, artifact management, and file saving
"""

import pytest
import os
import tempfile
import json
import logging
import numpy
import soundfile
import torch
from PIL import Image
from unittest.mock import patch
from dw.result import (
    AudioVideo,
    Result,
    as_audio_track,
    get_artifact_list,
    guess_extension,
    normalize_audio,
)


class TestResult:
    """Test Result class functionality"""

    def test_add_single_result(self):
        result = Result({})
        result.add_result("test_value")
        assert result.result_list == ["test_value"]

    def test_add_list_result(self):
        result = Result({})
        result.add_result(["item1", "item2", "item3"])
        assert result.result_list == ["item1", "item2", "item3"]

    def test_add_string_strips_quotes(self):
        result = Result({})
        result.add_result('"test_string"  ')
        assert result.result_list == ["test_string"]

    def test_get_artifacts_from_simple_list(self):
        result = Result({})
        result.add_result(["item1", "item2"])
        artifacts = result.get_artifacts()
        assert artifacts == ["item1", "item2"]

    def test_get_artifact_properties(self):
        result = Result({})
        result.add_result(
            [
                {"prop1": "value1", "prop2": "data1"},
                {"prop1": "value2", "prop2": "data2"},
            ]
        )

        prop_values = result.get_artifact_properties("prop1")
        assert prop_values == ["value1", "value2"]

    def test_get_artifact_properties_missing_key(self):
        result = Result({})
        result.add_result([{"prop1": "value1"}, {"prop2": "value2"}])  # Missing prop1

        prop_values = result.get_artifact_properties("prop1")
        assert len(prop_values) == 1
        assert prop_values == ["value1"]

    def test_get_artifact_properties_raises_on_string_result_substring_present(self):
        # "prop1" is a substring of the caption - a membership test would wrongly
        # treat that as the property being present and then fail trying to index a
        # string with a non-integer key
        result = Result({})
        result.add_result("a caption mentioning prop1 by name")

        with pytest.raises(ValueError, match="prop1"):
            result.get_artifact_properties("prop1")

    def test_get_artifact_properties_raises_on_string_result_substring_absent(self):
        # Not a substring either - previously this silently produced an empty list
        # instead of surfacing that the referenced step has no such property
        result = Result({})
        result.add_result("a plain caption")

        with pytest.raises(ValueError, match="text"):
            result.get_artifact_properties("text")

    def test_get_artifact_properties_error_names_the_actual_type(self):
        result = Result({})
        result.add_result("a caption")

        with pytest.raises(ValueError, match="str"):
            result.get_artifact_properties("text")

    def test_save_json_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_def = {"content_type": "application/json", "save": True}
            result = Result(result_def)
            result.add_result({"key": "value", "number": 42})

            result.save(temp_dir, "test_output")

            output_file = os.path.join(temp_dir, "test_output-0.json")
            assert os.path.exists(output_file)

            with open(output_file, "r") as f:
                loaded = json.load(f)
                assert loaded == {"key": "value", "number": 42}

    def test_save_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_def = {"content_type": "application/json", "save": False}
            result = Result(result_def)
            result.add_result({"key": "value"})

            result.save(temp_dir, "test_output")

            # Should not create any files
            files = os.listdir(temp_dir)
            assert len(files) == 0

    def test_save_no_content_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_def = {"save": True}  # No content_type
            result = Result(result_def)
            result.add_result("test")

            result.save(temp_dir, "test_output")

            # Should not create any files
            files = os.listdir(temp_dir)
            assert len(files) == 0

    def test_save_with_custom_base_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_def = {
                "content_type": "application/json",
                "save": True,
                "file_base_name": "custom_",
            }
            result = Result(result_def)
            result.add_result({"data": "test"})

            result.save(temp_dir, "output")

            output_file = os.path.join(temp_dir, "custom_output-0.json")
            assert os.path.exists(output_file)

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "nested", "path")
            result_def = {"content_type": "application/json", "save": True}
            result = Result(result_def)
            result.add_result({"data": "test"})

            result.save(nested_dir, "test_output")

            assert os.path.exists(nested_dir)
            output_file = os.path.join(nested_dir, "test_output-0.json")
            assert os.path.exists(output_file)


class TestGetArtifactList:
    """Test artifact extraction from various result types"""

    def test_result_with_images(self):
        class MockResult:
            images = ["img1", "img2", "img3"]

        artifacts = get_artifact_list(MockResult())
        assert artifacts == ["img1", "img2", "img3"]

    def test_result_with_frames(self):
        class MockResult:
            frames = ["frame1", "frame2"]

        artifacts = get_artifact_list(MockResult())
        assert artifacts == ["frame1", "frame2"]

    def test_result_with_list(self):
        result = ["item1", "item2"]
        artifacts = get_artifact_list(result)
        assert artifacts == ["item1", "item2"]

    def test_result_single_item(self):
        result = "single_item"
        artifacts = get_artifact_list(result)
        assert artifacts == ["single_item"]

    def test_result_with_image_embeds(self):
        class MockResult:
            image_embeds = ["embed1", "embed2"]

        artifacts = get_artifact_list(MockResult())
        assert artifacts == ["embed1", "embed2"]

    def test_result_with_numpy_audios(self):
        # Under the diffusers default output_type='np', AudioPipelineOutput.audios is
        # already a numpy ndarray (AudioLDM2/StableAudio call .numpy() before returning),
        # so it has no .float()/.cpu() - only .T/.astype()
        class MockResult:
            audios = numpy.zeros((1, 2, 100), dtype=numpy.float32)

        artifacts = get_artifact_list(MockResult())
        assert len(artifacts) == 1
        assert isinstance(artifacts[0], numpy.ndarray)
        assert artifacts[0].shape == (100, 2)
        assert artifacts[0].dtype == numpy.float32

    def test_result_with_torch_audios(self):
        class MockResult:
            audios = torch.zeros((1, 2, 100), dtype=torch.bfloat16)

        artifacts = get_artifact_list(MockResult())
        assert len(artifacts) == 1
        assert isinstance(artifacts[0], numpy.ndarray)
        assert artifacts[0].shape == (100, 2)
        assert artifacts[0].dtype == numpy.float32

    def test_images_take_precedence_over_frames(self):
        # The output-field registry is consulted in order - a result exposing both
        # (which nothing real does, but the registry order must still be deterministic)
        # is dispatched on "images", matching the old hasattr chain's ordering.
        class MockResult:
            images = ["img1", "img2"]
            frames = ["frame1", "frame2"]

        artifacts = get_artifact_list(MockResult())
        assert artifacts == ["img1", "img2"]

    def test_unknown_output_logs_a_warning_and_falls_back_to_one_artifact(self, caplog):
        # A diffusers output shaped like BaseOutput (to_tuple + dict-like keys()) whose
        # field ("depth") no registered extractor knows about. The failure this produces
        # downstream is a confusing content-type mismatch on save - the warning here is
        # what makes that diagnosable.
        class UnknownOutput:
            def __init__(self, depth):
                self.depth = depth

            def to_tuple(self):
                return (self.depth,)

            def keys(self):
                return ["depth"]

        result = UnknownOutput(depth=["depth_map"])

        with caplog.at_level(logging.WARNING, logger="dw"):
            artifacts = get_artifact_list(result)

        assert artifacts == [result]
        assert "UnknownOutput" in caplog.text
        assert "depth" in caplog.text


class TestAudioVideoArtifacts:
    """Test pairing of generated video with the audio generated alongside it"""

    class MockResult:
        """Stands in for an LTX2PipelineOutput"""

        def __init__(self, frames, audio, sample_rate=None):
            self.frames = frames
            self.audio = audio
            if sample_rate is not None:
                self.audio_sample_rate = sample_rate

    def test_frames_without_audio_are_unchanged(self):
        artifacts = get_artifact_list(self.MockResult(["frame1", "frame2"], None))
        assert artifacts == ["frame1", "frame2"]

    def test_each_video_is_paired_with_its_own_audio(self):
        result = self.MockResult(["video1", "video2"], ["audio1", "audio2"], 48000)

        artifacts = get_artifact_list(result)

        assert [(a.frames, a.audio, a.sample_rate) for a in artifacts] == [
            ("video1", "audio1", 48000),
            ("video2", "audio2", 48000),
        ]

    def test_missing_sample_rate_is_none(self):
        artifacts = get_artifact_list(self.MockResult(["video1"], ["audio1"]))
        assert artifacts[0].sample_rate is None

    def test_video_without_a_matching_waveform_gets_no_audio(self):
        artifacts = get_artifact_list(self.MockResult(["video1", "video2"], ["audio1"]))
        assert artifacts[1].audio is None

    def test_numpy_waveform_becomes_a_float_tensor(self):
        track = as_audio_track(numpy.zeros((2, 100), dtype=numpy.int16))
        assert isinstance(track, torch.Tensor)
        assert track.dtype == torch.float32
        assert track.shape == (2, 100)

    def test_tensor_waveform_is_converted_to_float(self):
        track = as_audio_track(torch.zeros((2, 100), dtype=torch.bfloat16))
        assert track.dtype == torch.float32
        assert track.device.type == "cpu"


class TestModularOutputs:
    """Test artifacts from the outputs a modular pipeline returns together"""

    def test_videos_are_paired_with_their_own_audio(self):
        artifacts = get_artifact_list(
            {
                "videos": ["video1", "video2"],
                "audio": ["audio1", "audio2"],
                "sampling_rate": 16000,
            }
        )

        assert [(a.frames, a.audio, a.sample_rate) for a in artifacts] == [
            ("video1", "audio1", 16000),
            ("video2", "audio2", 16000),
        ]

    def test_videos_without_audio_are_the_artifacts(self):
        artifacts = get_artifact_list({"videos": ["video1", "video2"]})
        assert artifacts == ["video1", "video2"]

    def test_unrequested_sample_rate_is_none(self):
        artifacts = get_artifact_list({"videos": ["video1"], "audio": ["audio1"]})
        assert artifacts[0].sample_rate is None

    def test_audio_sample_rate_names_the_rate_too(self):
        artifacts = get_artifact_list(
            {"videos": ["video1"], "audio": ["audio1"], "audio_sample_rate": 48000}
        )
        assert artifacts[0].sample_rate == 48000

    def test_outputs_without_video_are_left_alone(self):
        # Saved key by key, as any other dictionary result is
        outputs = {"images": ["image1"], "latents": "latents"}
        assert get_artifact_list(outputs) == [outputs]

    def test_generated_audio_is_muxed_into_the_video(self):
        outputs = {
            "videos": ["frames"],
            "audio": torch.zeros((1, 2, 100)),
            "sampling_rate": 16000,
        }
        result = Result({"content_type": "video/mp4", "fps": 24})
        result.add_result(outputs)

        with (
            patch("dw.result.encode_video") as encode,
            patch("dw.result.export_to_video") as export,
            patch("dw.result.is_av_available", return_value=True),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            result.save(temp_dir, "test")

        export.assert_not_called()
        assert encode.call_args.kwargs["audio_sample_rate"] == 16000
        assert encode.call_args.kwargs["audio"].shape == (2, 100)

    def test_outputs_are_still_available_by_name(self):
        # The outputs stay a dictionary, so a later step can reference one of them
        result = Result({"content_type": "video/mp4"})
        result.add_result({"videos": ["video1"], "sampling_rate": 16000})

        assert result.get_artifact_properties("sampling_rate") == [16000]

    def test_extra_outputs_alongside_video_are_not_dropped(self):
        # images/latents are not part of the video/audio pairing, so they must still be
        # saved - not silently lost, as they were before the leftover dict was added
        artifacts = get_artifact_list(
            {
                "videos": ["video1"],
                "audio": ["audio1"],
                "sampling_rate": 16000,
                "images": ["image1"],
            }
        )

        assert len(artifacts) == 2
        assert isinstance(artifacts[0], AudioVideo)
        assert (artifacts[0].frames, artifacts[0].audio, artifacts[0].sample_rate) == (
            "video1",
            "audio1",
            16000,
        )
        assert artifacts[1] == {"images": ["image1"]}

    def test_consumed_keys_do_not_reappear_in_the_leftovers(self):
        # videos/audio/sampling_rate were already saved as the paired AudioVideo -
        # repeating them in the leftover dict would save them a second time
        artifacts = get_artifact_list(
            {
                "videos": ["video1"],
                "audio": ["audio1"],
                "sampling_rate": 16000,
                "latents": "latents",
            }
        )

        leftovers = artifacts[-1]
        assert set(leftovers.keys()) == {"latents"}

    def test_attribute_and_dict_paths_pair_audio_identically(self):
        # Attribute-based pipeline outputs (LTX-2) and modular pipelines' dict outputs
        # both hand frames-plus-audio off to the same frames_with_audio/
        # pair_audio_with_frames logic - equivalent inputs down each route must produce
        # equivalent results.
        class MockResult:
            frames = ["video1", "video2"]
            audio = ["audio1", "audio2"]
            audio_sample_rate = 48000

        attribute_path = get_artifact_list(MockResult())
        dict_path = get_artifact_list(
            {
                "videos": ["video1", "video2"],
                "audio": ["audio1", "audio2"],
                "sampling_rate": 48000,
            }
        )

        def as_tuples(artifacts):
            return [(a.frames, a.audio, a.sample_rate) for a in artifacts]

        assert as_tuples(attribute_path) == as_tuples(dict_path)


class TestSaveAudioVideo:
    """Test muxing generated audio into the saved video file"""

    def save(self, result_definition, artifact, content_type="video/mp4"):
        result = Result(result_definition)
        result.add_result(artifact)
        with (
            patch("dw.result.encode_video") as encode,
            patch("dw.result.export_to_video") as export,
            patch("dw.result.is_av_available", return_value=True),
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                result.save(temp_dir, "test")
        return encode, export

    def test_audio_is_muxed_into_the_video(self):
        audio = torch.zeros((2, 100))
        artifact = AudioVideo("frames", audio, 48000)

        encode, export = self.save({"content_type": "video/mp4", "fps": 24}, artifact)

        export.assert_not_called()
        encode.assert_called_once()
        arguments = encode.call_args.kwargs
        assert encode.call_args.args[0] == "frames"
        assert arguments["fps"] == 24
        assert arguments["audio_sample_rate"] == 48000
        assert arguments["output_path"].endswith(".mp4")

    def test_result_definition_overrides_the_sample_rate(self):
        artifact = AudioVideo("frames", torch.zeros((2, 100)), 48000)

        encode, _ = self.save(
            {"content_type": "video/mp4", "audio_sample_rate": 24000}, artifact
        )

        assert encode.call_args.kwargs["audio_sample_rate"] == 24000

    def test_video_without_audio_falls_back_to_export(self):
        artifact = AudioVideo("frames", None, 48000)

        encode, export = self.save({"content_type": "video/mp4"}, artifact)

        encode.assert_not_called()
        export.assert_called_once()

    def test_video_without_a_sample_rate_falls_back_to_export(self):
        artifact = AudioVideo("frames", torch.zeros((2, 100)), None)

        encode, export = self.save({"content_type": "video/mp4"}, artifact)

        encode.assert_not_called()
        export.assert_called_once()

    def test_other_containers_fall_back_to_export(self):
        artifact = AudioVideo("frames", torch.zeros((2, 100)), 48000)

        encode, export = self.save({"content_type": "video/webm"}, artifact)

        encode.assert_not_called()
        export.assert_called_once()

    def test_missing_pyav_falls_back_to_export(self):
        result = Result({"content_type": "video/mp4"})
        result.add_result(AudioVideo("frames", torch.zeros((2, 100)), 48000))

        with (
            patch("dw.result.encode_video") as encode,
            patch("dw.result.export_to_video") as export,
            patch("dw.result.is_av_available", return_value=False),
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                result.save(temp_dir, "test")

        encode.assert_not_called()
        export.assert_called_once()


class TestNormalizeAudio:
    """Test conversion of pipeline audio output into writable waveforms"""

    def test_mono_waveform_unchanged(self):
        waveform = numpy.zeros(1000, dtype=numpy.float32)
        waveforms = normalize_audio(waveform)
        assert len(waveforms) == 1
        assert waveforms[0].shape == (1000,)

    def test_channels_first_is_transposed(self):
        waveforms = normalize_audio(numpy.zeros((2, 1000), dtype=numpy.float32))
        assert len(waveforms) == 1
        assert waveforms[0].shape == (1000, 2)

    def test_samples_first_is_left_alone(self):
        waveforms = normalize_audio(numpy.zeros((1000, 2), dtype=numpy.float32))
        assert waveforms[0].shape == (1000, 2)

    def test_batch_is_split(self):
        waveforms = normalize_audio(numpy.zeros((3, 2, 1000), dtype=numpy.float32))
        assert len(waveforms) == 3
        assert all(waveform.shape == (1000, 2) for waveform in waveforms)

    def test_torch_tensor_converted(self):
        waveforms = normalize_audio(torch.zeros((2, 1000), dtype=torch.bfloat16))
        assert isinstance(waveforms[0], numpy.ndarray)
        assert waveforms[0].shape == (1000, 2)

    def test_unsupported_shape_raises(self):
        with pytest.raises(ValueError):
            normalize_audio(numpy.zeros((2, 2, 2, 1000), dtype=numpy.float32))


class TestSaveAudio:
    """Test saving audio results"""

    def test_save_channels_first_waveform(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = Result({"content_type": "audio/wav", "sample_rate": 44100})
            result.add_result(numpy.zeros((2, 4410), dtype=numpy.float32))

            result.save(temp_dir, "song")

            output_file = os.path.join(temp_dir, "song-0.0.wav")
            assert os.path.exists(output_file)
            data, sample_rate = soundfile.read(output_file)
            assert sample_rate == 44100
            assert data.shape == (4410, 2)

    def test_save_batched_waveforms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = Result({"content_type": "audio/wav"})
            result.add_result(numpy.zeros((2, 2, 4410), dtype=numpy.float32))

            result.save(temp_dir, "song")

            assert os.path.exists(os.path.join(temp_dir, "song-0.0-0.wav"))
            assert os.path.exists(os.path.join(temp_dir, "song-0.0-1.wav"))

    def test_default_sample_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = Result({"content_type": "audio/wav"})
            result.add_result(numpy.zeros((2, 4410), dtype=numpy.float32))

            result.save(temp_dir, "song")

            _, sample_rate = soundfile.read(os.path.join(temp_dir, "song-0.0.wav"))
            assert sample_rate == 44100

    def test_numpy_audios_output_saves_without_attribute_error(self):
        # Regression test: AudioPipelineOutput.audios is a numpy ndarray under the
        # default output_type='np', which has no .float()/.cpu() - saving used to raise
        # AttributeError and the audio was never written
        class MockResult:
            audios = numpy.zeros((1, 2, 4410), dtype=numpy.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = Result({"content_type": "audio/wav", "sample_rate": 44100})
            result.add_result(MockResult())

            result.save(temp_dir, "song")

            output_file = os.path.join(temp_dir, "song-0.0.wav")
            assert os.path.exists(output_file)
            data, sample_rate = soundfile.read(output_file)
            assert sample_rate == 44100
            assert data.shape == (4410, 2)

    def test_torch_audios_output_still_saves(self):
        class MockResult:
            audios = torch.zeros((1, 2, 4410), dtype=torch.bfloat16)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = Result({"content_type": "audio/wav", "sample_rate": 44100})
            result.add_result(MockResult())

            result.save(temp_dir, "song")

            output_file = os.path.join(temp_dir, "song-0.0.wav")
            assert os.path.exists(output_file)
            data, sample_rate = soundfile.read(output_file)
            assert sample_rate == 44100
            assert data.shape == (4410, 2)


class TestSaveCompressedAudio:
    """Test saving audio in the compressed formats soundfile supports"""

    def save(self, result_definition):
        """Save a second of noise and return the path of the file it produced"""
        temp_dir = tempfile.mkdtemp()
        result = Result({"sample_rate": 44100, **result_definition})
        # Noise rather than silence so that compression settings change the file size
        waveform = numpy.random.default_rng(7).uniform(-1, 1, (2, 44100))
        result.add_result(waveform.astype(numpy.float32))

        result.save(temp_dir, "song")

        extension = guess_extension(result_definition["content_type"])
        output_file = os.path.join(temp_dir, f"song-0.0{extension}")
        assert os.path.exists(output_file)
        return output_file

    def test_flac(self):
        info = soundfile.info(self.save({"content_type": "audio/flac"}))
        assert info.format == "FLAC"
        assert info.samplerate == 44100
        assert info.channels == 2

    def test_mp3(self):
        assert soundfile.info(self.save({"content_type": "audio/mpeg"})).format == "MP3"

    def test_ogg_vorbis(self):
        info = soundfile.info(self.save({"content_type": "audio/ogg"}))
        assert info.format == "OGG"
        assert info.subtype == "VORBIS"

    def test_opus_uses_the_ogg_container(self):
        # Opus has no extension of its own in libsndfile, and only encodes 48kHz audio
        info = soundfile.info(
            self.save({"content_type": "audio/opus", "sample_rate": 48000})
        )
        assert info.format == "OGG"
        assert info.subtype == "OPUS"

    def test_compressed_is_smaller_than_wav(self):
        wav = self.save({"content_type": "audio/wav"})
        mp3 = self.save({"content_type": "audio/mpeg"})
        assert os.path.getsize(mp3) < os.path.getsize(wav)

    def test_subtype_overrides_the_default(self):
        info = soundfile.info(
            self.save({"content_type": "audio/flac", "subtype": "PCM_24"})
        )
        assert info.subtype == "PCM_24"

    def test_long_audio_does_not_crash_the_vorbis_encoder(self):
        # libsndfile segfaults if handed more than 2**21 frames of vorbis in one call
        temp_dir = tempfile.mkdtemp()
        frames = (1 << 21) + 1000
        result = Result({"content_type": "audio/ogg", "sample_rate": 44100})
        result.add_result(numpy.zeros((2, frames), dtype=numpy.float32))

        result.save(temp_dir, "song")

        info = soundfile.info(os.path.join(temp_dir, "song-0.0.ogg"))
        assert info.frames == frames

    def test_compression_level_reaches_soundfile(self):
        loose = self.save({"content_type": "audio/ogg", "compression_level": 0.1})
        tight = self.save({"content_type": "audio/ogg", "compression_level": 1.0})
        assert os.path.getsize(tight) < os.path.getsize(loose)


class TestGuessExtension:
    """Test MIME type to file extension conversion"""

    def test_image_jpeg(self):
        assert guess_extension("image/jpeg") == ".jpg"

    def test_image_png(self):
        assert guess_extension("image/png") == ".png"

    def test_application_json(self):
        assert guess_extension("application/json") == ".json"

    def test_audio_wav(self):
        assert guess_extension("audio/wav") == ".wav"

    def test_audio_mpeg(self):
        assert guess_extension("audio/mpeg") == ".mp3"

    def test_audio_flac(self):
        assert guess_extension("audio/flac") == ".flac"

    def test_audio_ogg(self):
        # mimetypes suggests '.oga', which soundfile does not recognize
        assert guess_extension("audio/ogg") == ".ogg"

    def test_audio_opus(self):
        assert guess_extension("audio/opus") == ".ogg"

    def test_video_mp4(self):
        ext = guess_extension("video/mp4")
        assert ext in [".mp4", ".mpg4"]  # Both are valid

    def test_unknown_type(self):
        result = guess_extension("unknown/type")
        # Should return empty string or None for unknown types
        assert result in ["", None]

    def test_none_content_type(self):
        assert guess_extension(None) == ""

    def test_empty_content_type(self):
        assert guess_extension("") == ""


class TestMetadataEmbedding:
    """Test opt-in metadata embedding in saved images."""

    def test_png_metadata_embedded(self):
        """When embed_metadata is true, PNG should contain parameters text chunk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result_def = {
                "content_type": "image/png",
                "save": True,
                "embed_metadata": True,
            }
            result = Result(result_def)
            result.set_metadata(
                {
                    "workflow_id": "test_workflow",
                    "step_name": "generate",
                    "model_name": "test/model",
                    "arguments": {"prompt": "a cat", "num_inference_steps": 25},
                }
            )

            img = Image.new("RGB", (64, 64), color=(128, 64, 32))
            result.add_result(img)
            result.save(temp_dir, "test_output")

            output_file = os.path.join(temp_dir, "test_output-0.0.png")
            assert os.path.exists(output_file)

            saved_img = Image.open(output_file)
            assert "parameters" in saved_img.info
            metadata = json.loads(saved_img.info["parameters"])
            assert metadata["workflow_id"] == "test_workflow"
            assert metadata["arguments"]["prompt"] == "a cat"

    def test_no_metadata_when_not_enabled(self):
        """When embed_metadata is absent/false, no metadata should be embedded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result_def = {"content_type": "image/png", "save": True}
            result = Result(result_def)

            img = Image.new("RGB", (64, 64), color=(128, 64, 32))
            result.add_result(img)
            result.save(temp_dir, "test_output")

            output_file = os.path.join(temp_dir, "test_output-0.0.png")
            saved_img = Image.open(output_file)
            assert "parameters" not in saved_img.info

    def test_set_metadata_method(self):
        """set_metadata should store metadata on the Result instance."""
        result = Result({})
        assert result.metadata is None

        metadata = {"workflow_id": "test", "step_name": "step1"}
        result.set_metadata(metadata)
        assert result.metadata == metadata

    def test_metadata_with_embed_false(self):
        """When embed_metadata is explicitly false, no metadata embedded even if set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result_def = {
                "content_type": "image/png",
                "save": True,
                "embed_metadata": False,
            }
            result = Result(result_def)
            result.set_metadata({"workflow_id": "test"})

            img = Image.new("RGB", (64, 64), color=(128, 64, 32))
            result.add_result(img)
            result.save(temp_dir, "test_output")

            output_file = os.path.join(temp_dir, "test_output-0.0.png")
            saved_img = Image.open(output_file)
            assert "parameters" not in saved_img.info


class TestSegmentBackedSave:
    """Saving an AudioVideo whose frames replay from chain segment files."""

    def make_segments(self, tmp_path, counts=(4, 3)):
        from diffusers.utils import encode_video

        from dw.pipeline_processors.chain import SegmentedFrames

        paths = []
        for index, count in enumerate(counts):
            frames = [Image.new("RGB", (16, 16), (index * 80, 0, 0))] * count
            path = str(tmp_path / f"segment-{index}.mp4")
            encode_video(frames, fps=4, output_path=path)
            paths.append(path)
        return SegmentedFrames(paths)

    def decode_frame_count(self, path):
        import av

        with av.open(str(path)) as container:
            return sum(1 for _ in container.decode(video=0))

    def test_streams_segments_into_one_video(self, tmp_path):
        frames = self.make_segments(tmp_path)
        result = Result({"content_type": "video/mp4", "fps": 4})
        result.add_result(AudioVideo(frames, None, None))

        result.save(str(tmp_path), "final")

        output = tmp_path / "final-0.0.mp4"
        assert output.exists()
        assert self.decode_frame_count(output) == 7

    def test_the_segment_files_are_removed_after_a_successful_save(self, tmp_path):
        frames = self.make_segments(tmp_path)
        result = Result({"content_type": "video/mp4", "fps": 4})
        result.add_result(AudioVideo(frames, None, None))

        result.save(str(tmp_path), "final")

        assert list(tmp_path.glob("segment-*.mp4")) == []

    def test_audio_is_muxed_into_the_streamed_video(self, tmp_path):
        import av

        frames = self.make_segments(tmp_path)
        audio = numpy.zeros((2, int(7 / 4 * 8000)), dtype=numpy.float32)
        result = Result({"content_type": "video/mp4", "fps": 4})
        result.add_result(AudioVideo(frames, audio, 8000))

        result.save(str(tmp_path), "final")

        with av.open(str(tmp_path / "final-0.0.mp4")) as container:
            assert len(container.streams.audio) == 1

    def test_a_non_mp4_content_type_raises(self, tmp_path):
        frames = self.make_segments(tmp_path)
        result = Result({"content_type": "video/gif", "fps": 4})
        result.add_result(AudioVideo(frames, None, None))

        with pytest.raises(ValueError, match="video/mp4"):
            result.save(str(tmp_path), "final")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
