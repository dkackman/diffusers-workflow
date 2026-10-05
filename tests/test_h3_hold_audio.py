"""MiniMax-H3 audio hold (#598): the blocks dw inserts into diffusers' graph, the
refusals made before a load, and the held track a step's result plays."""

import copy
import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from dw.hold_audio import hold_audio_errors
from dw.media_types import AudioTrack, AudioVideo
from dw.output_extraction import modular_artifacts
from dw.pipeline_processors import h3_blocks
from dw.pipeline_processors.h3_blocks import (
    HELD_AUDIO_OUTPUT,
    HELD_AUDIO_RATE_OUTPUT,
    HELD_ROWS,
    HOLD_BLOCK,
    RELEASE_BLOCK,
    as_channels_samples,
    blocks,
    core_denoise_sequences,
    fit_latents,
    fit_samples,
    hold_audio_reference,
    holds_audio,
    insert_audio_hold,
)
from dw.pipeline_processors.pipeline import Pipeline
from dw.workflow import workflow_from_definition

minimax = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.modular_pipelines.modular_pipeline import PipelineState  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (  # noqa: E402
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
)
from diffusers.modular_pipelines.minimax_h3.references import (  # noqa: E402
    MiniMaxH3AudioReference,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = ("t2va", "fl2va", "ref2va")
RATE = 32000
CHANNELS = 4
HOP = RATE // MINIMAX_H3_AUDIO_LATENTS_PER_SECOND


def pipelines():
    """[(label, ModularPipeline)]: the whole graph and each pruned workflow."""
    blocks_ = minimax.MiniMaxH3Blocks()
    result = [("whole", blocks_.init_pipeline())]
    for name in WORKFLOWS:
        result.append((name, blocks_.get_workflow(name).init_pipeline()))
    return result


@pytest.fixture(scope="module")
def built():
    return pipelines()


# 1. Anchors in the installed diffusers


class TestAnchors:
    def test_inserted_next_to_the_anchors_in_every_shape(self, built):
        for label, pipeline in built:
            assert insert_audio_hold(pipeline) is True, label
            assert holds_audio(pipeline), label
            sequences = core_denoise_sequences(pipeline)
            assert sequences, label
            if label == "whole":
                assert [prefix for prefix, _ in sequences] == [""] * 3
            else:
                assert [prefix for prefix, _ in sequences] == ["denoise."]
            for prefix, sequence in sequences:
                names = list(sequence.sub_blocks)
                hold = names.index(prefix + HOLD_BLOCK)
                release = names.index(prefix + RELEASE_BLOCK)
                assert names[hold + 1] == prefix + "set_timesteps", label
                assert names[release + 1] == prefix + "after_denoise", label

    def test_second_call_does_not_duplicate(self, built):
        for label, pipeline in built:
            insert_audio_hold(pipeline)
            before = [list(s.sub_blocks) for _, s in core_denoise_sequences(pipeline)]
            assert insert_audio_hold(pipeline) is True
            after = [list(s.sub_blocks) for _, s in core_denoise_sequences(pipeline)]
            assert before == after, label
            for names in after:
                assert (
                    names.count(HOLD_BLOCK) + names.count("denoise." + HOLD_BLOCK) == 1
                )
                assert (
                    names.count(RELEASE_BLOCK) + names.count("denoise." + RELEASE_BLOCK)
                    == 1
                )

    def test_hold_audio_is_a_pipeline_input(self, built):
        for label, pipeline in built:
            insert_audio_hold(pipeline)
            names = [param.name for param in pipeline._blocks.inputs]
            assert "hold_audio" in names, label

    def test_a_pipeline_without_the_blocks_does_not_hold(self):
        assert not holds_audio(minimax.MiniMaxH3Blocks().init_pipeline())
        other = SimpleNamespace(_blocks=SimpleNamespace(sub_blocks={}))
        assert core_denoise_sequences(other) == []
        assert holds_audio(other) is False
        assert core_denoise_sequences(object()) == []
        assert insert_audio_hold(object()) is False

    def test_an_unanchored_sequence_is_skipped(self):
        pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
        pipeline._blocks.sub_blocks.pop("denoise.after_denoise")
        assert insert_audio_hold(pipeline) is False
        assert not holds_audio(pipeline)


# 2. Crop and pad


class TestFitting:
    def test_samples_crop(self):
        wave = torch.arange(10.0).repeat(2, 1)
        assert torch.equal(fit_samples(wave, 4), wave[:, :4])

    def test_samples_pad_with_silence(self):
        wave = torch.ones(2, 3)
        fitted = fit_samples(wave, 5)
        assert fitted.shape == (2, 5)
        assert torch.equal(fitted[:, :3], wave)
        assert torch.count_nonzero(fitted[:, 3:]) == 0

    def test_samples_exact_is_unchanged(self):
        wave = torch.ones(2, 3)
        assert torch.equal(fit_samples(wave, 3), wave)

    def test_latents_crop_per_channel(self):
        latents = torch.arange(2 * 6 * 3.0).reshape(2, 6, 3)
        assert torch.equal(fit_latents(latents, 4, None), latents[:, :4])

    def test_latents_pad_with_the_repeated_silence(self):
        latents = torch.ones(2, 2, 3)
        silence = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
        fitted = fit_latents(latents, 5, silence)
        assert fitted.shape == (2, 5, 3)
        assert torch.equal(fitted[:, :2], latents)
        for i in range(2, 5):
            assert torch.equal(fitted[:, i : i + 1], silence)

    def test_mono_gains_a_channel_axis(self):
        assert as_channels_samples(torch.zeros(7)).shape == (1, 7)
        assert as_channels_samples(torch.zeros(2, 7)).shape == (2, 7)
        assert as_channels_samples(torch.zeros(2, 7, dtype=torch.float64)).dtype == (
            torch.float32
        )


# 3 and 4. The real block path on CPU


class FakePosterior:
    def __init__(self, values):
        self.values = values

    def mode(self):
        return self.values


class FakeAudioVae:
    config = SimpleNamespace(
        latents_mean=[0.0] * CHANNELS, latents_std=[1.0] * CHANNELS
    )

    def encode(self, x, return_dict=False):
        """(batch, 1, samples) -> (batch, C, samples // HOP): each hop's mean, plus
        a per-channel offset."""
        n = x.shape[-1] // HOP
        hops = x[:, 0, : n * HOP].reshape(x.shape[0], n, HOP).mean(-1)
        offsets = torch.arange(CHANNELS, dtype=torch.float32)[None, :, None] * 0.01
        return (FakePosterior(hops[:, None, :] + offsets),)


def components():
    return SimpleNamespace(
        audio_sampling_rate=RATE,
        fps=24,
        audio_channels=2,
        audio_latent_channels=CHANNELS,
        _execution_device=torch.device("cpu"),
        audio_vae=FakeAudioVae(),
    )


NUM_FRAMES = 48  # two seconds at 24 fps
NUM_LATENTS = 2 * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND


def make_state(held=None, reference_rows=0):
    rows = reference_rows + 2 * NUM_LATENTS
    audio_latents = torch.randn(rows, CHANNELS, generator=torch.manual_seed(1))
    if reference_rows:
        audio_latents[:reference_rows] = 7.0
    state = PipelineState()
    state.set("audio_latents", audio_latents)
    state.set("num_audio_latents", NUM_LATENTS)
    state.set("num_condition_audio_rows", reference_rows)
    state.set("num_frames", NUM_FRAMES)
    if held is not None:
        state.set("hold_audio", held)
    return state


def expected_rows(waveform_32k):
    """The stub encode of a stereo waveform, channel-major."""
    x = waveform_32k[:, None]
    latents = FakeAudioVae().encode(x)[0].mode().transpose(1, 2)
    return latents.reshape(-1, CHANNELS)


class TestHoldBlock:
    def test_a_longer_track_is_cropped_and_written_over_the_target_rows(self):
        wave = torch.randn(2, 3 * RATE, generator=torch.manual_seed(2))
        state = make_state(MiniMaxH3AudioReference(audio=wave, sample_rate=RATE))
        comps = components()
        hold, _ = blocks()
        hold()(comps, state)

        latents = state.get("audio_latents")
        count = state.get("num_condition_audio_rows")
        assert count == 2 * NUM_LATENTS
        assert state.get(HELD_ROWS) == 2 * NUM_LATENTS
        expected = expected_rows(wave[:, : NUM_LATENTS * HOP])
        assert torch.allclose(latents[:count], expected, atol=1e-6)

        held = state.get(HELD_AUDIO_OUTPUT)
        assert held.shape == (1, 2, round(NUM_FRAMES / 24 * RATE))
        assert torch.equal(held[0], wave[:, : held.shape[-1]])
        assert state.get(HELD_AUDIO_RATE_OUTPUT) == RATE

    def test_a_shorter_track_is_padded_with_silence(self):
        wave = torch.randn(2, RATE, generator=torch.manual_seed(3))
        state = make_state(MiniMaxH3AudioReference(audio=wave, sample_rate=RATE))
        comps = components()
        blocks()[0]()(comps, state)

        latents = state.get("audio_latents")
        rows = state.get(HELD_ROWS)
        assert rows == 2 * NUM_LATENTS
        per_channel = latents[:rows].reshape(2, NUM_LATENTS, CHANNELS)
        half = MINIMAX_H3_AUDIO_LATENTS_PER_SECOND
        real = expected_rows(wave).reshape(2, half, CHANNELS)
        assert torch.allclose(per_channel[:, :half], real, atol=1e-6)
        # silence is zeros through the stub, so only the channel offset remains
        offsets = torch.arange(CHANNELS, dtype=torch.float32) * 0.01
        assert torch.allclose(
            per_channel[:, half:], offsets.expand(2, NUM_LATENTS - half, -1), atol=1e-6
        )

        held = state.get(HELD_AUDIO_OUTPUT)
        assert held.shape == (1, 2, 2 * RATE)
        assert torch.equal(held[0, :, :RATE], wave)
        assert torch.count_nonzero(held[0, :, RATE:]) == 0

    def test_the_held_track_keeps_its_own_rate_and_channels(self):
        wave = torch.randn(1, 3 * 16000, generator=torch.manual_seed(4))
        state = make_state(MiniMaxH3AudioReference(audio=wave, sample_rate=16000))
        pytest.importorskip("torchaudio")
        blocks()[0]()(components(), state)

        held = state.get(HELD_AUDIO_OUTPUT)
        assert held.shape == (1, 1, 2 * 16000)
        assert state.get(HELD_AUDIO_RATE_OUTPUT) == 16000
        assert state.get(HELD_ROWS) == 2 * NUM_LATENTS

    def test_reference_rows_in_front_are_untouched(self):
        k = 6
        wave = torch.randn(2, 2 * RATE, generator=torch.manual_seed(5))
        state = make_state(
            MiniMaxH3AudioReference(audio=wave, sample_rate=RATE), reference_rows=k
        )
        comps = components()
        hold, release = blocks()
        hold()(comps, state)

        latents = state.get("audio_latents")
        assert torch.equal(latents[:k], torch.full((k, CHANNELS), 7.0))
        assert state.get("num_condition_audio_rows") == k + 2 * NUM_LATENTS
        assert torch.allclose(
            latents[k:], expected_rows(wave[:, : NUM_LATENTS * HOP]), atol=1e-6
        )

        release()(comps, state)
        assert state.get("num_condition_audio_rows") == k
        assert state.get(HELD_ROWS) == 0

    def test_release_restores_the_count(self):
        wave = torch.randn(2, 2 * RATE, generator=torch.manual_seed(6))
        state = make_state(MiniMaxH3AudioReference(audio=wave, sample_rate=RATE))
        comps = components()
        hold, release = blocks()
        hold()(comps, state)
        assert state.get("num_condition_audio_rows") > 0
        release()(comps, state)
        assert state.get("num_condition_audio_rows") == 0


class TestNoHold:
    def test_the_hold_block_is_a_no_op(self):
        state = make_state(reference_rows=3)
        before = state.get("audio_latents").clone()
        blocks()[0]()(components(), state)

        assert torch.equal(state.get("audio_latents"), before)
        assert state.get("num_condition_audio_rows") == 3
        assert state.get(HELD_ROWS) == 0
        assert state.get(HELD_AUDIO_OUTPUT) is None

    def test_hold_then_release_leaves_the_state_as_it_was(self):
        """The real set-timesteps step needs schedulers and the packed layout, which
        is impractical to stub, so this compares the state across hold and release
        with no `hold_audio` instead."""
        state = make_state(reference_rows=3)
        before = state.get("audio_latents").clone()
        comps = components()
        hold, release = blocks()
        hold()(comps, state)
        release()(comps, state)

        assert torch.equal(state.get("audio_latents"), before)
        assert state.get("num_condition_audio_rows") == 3
        assert state.get(HELD_ROWS) == 0


# 5. Refusals


def h3_step(workflow="t2va", **arguments):
    return {
        "name": "video",
        "pipeline": {
            "configuration": {"component_type": "ModularPipeline"},
            "from_pretrained_arguments": {"workflow": workflow},
            "arguments": arguments,
        },
    }


def errors_for(step):
    return hold_audio_errors({"steps": [step]})


class TestRefusals:
    def test_a_non_h3_step_is_refused(self):
        step = h3_step(hold_audio="asset:track.wav")
        step["pipeline"]["configuration"]["component_type"] = "StableDiffusionPipeline"
        errors = errors_for(step)
        assert len(errors) == 1
        assert "StableDiffusionPipeline" in errors[0]["message"]
        assert errors[0]["path"].endswith("hold_audio")

    @pytest.mark.parametrize("workflow", ["i2v", "t2v", "text2image"])
    def test_a_workflow_that_is_not_an_h3_one_is_refused(self, workflow):
        errors = errors_for(h3_step(workflow, hold_audio="asset:track.wav"))
        assert len(errors) == 1
        assert workflow in errors[0]["message"]

    @pytest.mark.parametrize("workflow", WORKFLOWS)
    def test_the_h3_workflows_are_accepted(self, workflow):
        assert errors_for(h3_step(workflow, hold_audio="asset:track.wav")) == []

    @pytest.mark.parametrize("value", ["asset:frame.png", "clip.mp4", "x.png"])
    def test_a_non_audio_path_is_refused(self, value):
        errors = errors_for(h3_step(hold_audio=value))
        assert len(errors) == 1
        assert "not an audio file" in errors[0]["message"]

    def test_a_media_reference_is_refused(self):
        value = {"media_type": "image", "location": "asset:frame.png"}
        errors = errors_for(h3_step(hold_audio=value))
        assert len(errors) == 1
        assert "'image' reference" in errors[0]["message"]

    @pytest.mark.parametrize(
        "value", ["asset:track.wav", "previous_result:base.audio", "output:a/b.mp3"]
    )
    def test_audio_and_deferred_values_are_accepted(self, value):
        assert errors_for(h3_step(hold_audio=value)) == []

    def test_no_workflow_and_no_hold_audio_say_nothing(self):
        step = h3_step(hold_audio="asset:t.wav")
        del step["pipeline"]["from_pretrained_arguments"]
        assert errors_for(step) == []
        assert errors_for(h3_step()) == []


def hold_errors(definition):
    workflow = workflow_from_definition(definition, tempfile.mkdtemp())
    return [
        problem
        for problem in workflow.validation_errors()
        if "hold_audio" in problem["path"]
    ]


def template(*parts):
    with open(os.path.join(ROOT, "workflows", "templates", *parts)) as f:
        return json.load(f)


class TestValidationEntry:
    def test_sd15_template_refuses_hold_audio(self):
        definition = template("text-to-image.json")
        step = definition["steps"][0]
        assert (
            step["pipeline"]["configuration"]["component_type"]
            == "StableDiffusionPipeline"
        )
        step["pipeline"]["arguments"]["hold_audio"] = "asset:track.wav"

        problems = hold_errors(definition)

        assert len(problems) == 1
        assert "StableDiffusionPipeline" in problems[0]["message"]

    def test_h3_template_accepts_hold_audio(self):
        definition = copy.deepcopy(template("minimax", "video-with-audio.json"))
        definition["steps"][0]["pipeline"]["arguments"]["hold_audio"] = (
            "asset:track.wav"
        )

        assert hold_errors(definition) == []

    def test_h3_template_refuses_a_still(self):
        definition = copy.deepcopy(template("minimax", "video-with-audio.json"))
        definition["steps"][0]["pipeline"]["arguments"]["hold_audio"] = (
            "asset:frame.png"
        )

        assert len(hold_errors(definition)) == 1


class TestHoldAudioReference:
    @pytest.mark.parametrize("value", ["x.png", "notes.txt", "noextension"])
    def test_a_non_audio_path_raises(self, value):
        with pytest.raises(ValueError):
            hold_audio_reference(value)

    def test_a_non_audio_value_raises(self):
        with pytest.raises(ValueError):
            hold_audio_reference(12)
        with pytest.raises(ValueError):
            hold_audio_reference(object())

    def test_a_reference_passes_through(self):
        reference = MiniMaxH3AudioReference(audio=torch.zeros(2, 10), sample_rate=8000)
        assert hold_audio_reference(reference) is reference

    def test_an_audio_track_becomes_a_reference_with_its_rate(self):
        wave = torch.zeros(2, 100)
        reference = hold_audio_reference(AudioTrack(wave, 22050))
        assert isinstance(reference, MiniMaxH3AudioReference)
        assert reference.sample_rate == 22050
        assert reference.audio.shape == (2, 100)


class TestHoldAudioLocation:
    """A hold_audio path goes through the location owner (dw/locations.py)
    at the call, not just at validation."""

    @pytest.fixture
    def opened(self, monkeypatch):
        opened = []

        def from_file(cls, location):
            opened.append(location)
            return cls(audio=torch.zeros(2, 10), sample_rate=8000)

        monkeypatch.setattr(
            MiniMaxH3AudioReference, "from_file", classmethod(from_file)
        )
        return opened

    def test_a_relative_path_resolves_against_the_workflow_directory(
        self, tmp_path, opened
    ):
        (tmp_path / "track.wav").write_bytes(b"")

        hold_audio_reference("track.wav", str(tmp_path))

        assert opened == [os.path.realpath(tmp_path / "track.wav")]

    def test_a_path_outside_the_roots_is_refused_at_the_call(
        self, tmp_path, opened, monkeypatch
    ):
        from dw.security import SecurityError

        monkeypatch.delenv("DW_TRUST_WORKFLOWS")  # conftest trusts by default

        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "track.wav").write_bytes(b"")

        with pytest.raises(SecurityError, match="outside every directory"):
            hold_audio_reference(str(elsewhere / "track.wav"), str(workflow_dir))
        assert opened == []

    def test_the_step_hands_its_base_dir_to_the_check(self, tmp_path, opened):
        (tmp_path / "track.wav").write_bytes(b"")
        pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
        insert_audio_hold(pipeline)

        Pipeline._with_held_audio(
            ns(pipeline, str(tmp_path)), {"hold_audio": "track.wav"}
        )

        assert opened == [os.path.realpath(tmp_path / "track.wav")]

    def test_a_workflow_gives_its_pipelines_its_own_directory(self, tmp_path):
        workflow = workflow_from_definition(
            {"id": "w", "steps": []}, str(tmp_path / "out"), base_dir=str(tmp_path)
        )
        assert workflow.base_dir == os.path.realpath(tmp_path)


# 6. _with_held_audio


def ns(pipeline, base_dir=None):
    return SimpleNamespace(name="video", pipeline=pipeline, base_dir=base_dir)


class TestWithHeldAudio:
    def test_no_hold_audio_leaves_the_arguments_alone(self):
        arguments = {"prompt": "x", "output": ["videos", "audio"]}
        assert Pipeline._with_held_audio(ns(object()), arguments) is arguments

    def test_a_non_h3_pipeline_refuses(self):
        with pytest.raises(ValueError, match="MiniMax-H3"):
            Pipeline._with_held_audio(ns(object()), {"hold_audio": "t.wav"})

    def test_an_h3_pipeline_gets_the_reference_and_the_held_keys(self):
        pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
        insert_audio_hold(pipeline)
        reference = MiniMaxH3AudioReference(audio=torch.zeros(2, 10), sample_rate=8000)
        arguments = {"hold_audio": reference, "output": ["videos", "audio"]}

        result = Pipeline._with_held_audio(ns(pipeline), arguments)

        assert result["hold_audio"] is reference
        assert result["output"] == [
            "videos",
            "audio",
            HELD_AUDIO_OUTPUT,
            HELD_AUDIO_RATE_OUTPUT,
        ]
        assert arguments["output"] == ["videos", "audio"]  # the caller's is untouched

    def test_output_without_audio_gains_nothing(self):
        pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
        insert_audio_hold(pipeline)
        reference = MiniMaxH3AudioReference(audio=torch.zeros(2, 10), sample_rate=8000)
        result = Pipeline._with_held_audio(
            ns(pipeline), {"hold_audio": reference, "output": ["videos"]}
        )
        assert result["output"] == ["videos"]

    def test_a_non_audio_value_is_refused_on_an_h3_pipeline(self):
        pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
        insert_audio_hold(pipeline)
        with pytest.raises(ValueError):
            Pipeline._with_held_audio(ns(pipeline), {"hold_audio": "x.png"})


# 7. modular_artifacts


class TestModularArtifacts:
    def test_the_held_track_replaces_the_decoded_audio(self):
        frames = [["f0", "f1"]]
        decoded = torch.zeros(2, 10)
        held = torch.ones(1, 2, 20)
        result = {
            "videos": frames,
            "audio": [decoded],
            "sampling_rate": 48000,
            HELD_AUDIO_OUTPUT: held,
            HELD_AUDIO_RATE_OUTPUT: 32000,
        }

        artifacts = modular_artifacts(result)

        assert len(artifacts) == 1
        video = artifacts[0]
        assert isinstance(video, AudioVideo)
        assert torch.equal(video.audio, held[0])
        assert video.sample_rate == 32000

    def test_the_held_track_stands_without_decoded_audio(self):
        held = torch.ones(1, 2, 20)
        artifacts = modular_artifacts(
            {
                "videos": [["f"]],
                HELD_AUDIO_OUTPUT: held,
                HELD_AUDIO_RATE_OUTPUT: 24000,
            }
        )
        assert len(artifacts) == 1
        assert artifacts[0].sample_rate == 24000

    def test_the_held_keys_are_not_leftovers(self):
        result = {
            "videos": [["f"]],
            "audio": [torch.zeros(2, 4)],
            "sampling_rate": 48000,
            "latents": torch.zeros(1),
            HELD_AUDIO_OUTPUT: torch.ones(1, 2, 4),
            HELD_AUDIO_RATE_OUTPUT: 32000,
        }
        artifacts = modular_artifacts(result)

        assert len(artifacts) == 2
        leftovers = artifacts[-1]
        assert isinstance(leftovers, dict)
        assert set(leftovers) == {"latents"}

    def test_no_held_keys_no_change(self):
        audio = torch.zeros(2, 4)
        artifacts = modular_artifacts(
            {"videos": [["f"]], "audio": [audio], "sampling_rate": 48000}
        )
        assert artifacts[0].sample_rate == 48000
        assert torch.equal(artifacts[0].audio, audio)


def test_module_exports_stay_in_sync():
    assert h3_blocks.HOLD_BEFORE == "set_timesteps"
    assert h3_blocks.RELEASE_BEFORE == "after_denoise"
