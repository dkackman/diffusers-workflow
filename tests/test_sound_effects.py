"""generate_sound_effects (#1124 stage D): every refusal before any load, the
duration rules, the noise note, the run with stubbed models, the catalog
templates and the registration."""

import json
import os

import numpy
import pytest
import torch
from PIL import Image

from dw.media_types import AudioTrack
from dw.security import UntrustedWorkflowError
from dw.server.catalog_shape import derive_catalog_metadata
from dw.tasks import sound_effects
from dw.tasks.sound_effects import generate_sound_effects
from dw.trust import TRUST_WORKFLOWS_ENV_VAR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LICENCE = (
    "Apple Machine Learning Research Model is licensed under the Apple Machine "
    "Learning Research Model License Agreement."
)


@pytest.fixture
def no_load(monkeypatch):
    """Any model load or download is a failure."""
    import huggingface_hub

    def refuse(*args, **kwargs):
        raise AssertionError("a load ran before the refusal")

    monkeypatch.setattr(sound_effects, "_load_models", refuse)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)


@pytest.fixture
def two_second_clip(tmp_path):
    from diffusers.utils import export_to_video

    frames = [Image.new("RGB", (64, 64), (i * 5 % 256, 0, 0)) for i in range(48)]
    path = tmp_path / "clip.mp4"
    export_to_video(frames, str(path), fps=24)
    return str(path)


class TestRefusals:
    def test_neither_video_nor_prompt(self, no_load):
        with pytest.raises(ValueError, match="'video' or 'prompt'"):
            generate_sound_effects()

    def test_a_text_only_duration_over_20(self, no_load):
        with pytest.raises(ValueError, match="'duration'"):
            generate_sound_effects(prompt="rain", duration=21)

    def test_a_duration_longer_than_the_clip(self, no_load, two_second_clip):
        with pytest.raises(ValueError, match="'duration'"):
            generate_sound_effects(video=two_second_clip, duration=3)

    def test_an_unknown_argument(self, no_load):
        with pytest.raises(ValueError, match="'foo'"):
            generate_sound_effects(prompt="rain", foo=1)

    @pytest.mark.parametrize("steps", [0, "ten", True])
    def test_bad_steps(self, no_load, steps):
        with pytest.raises(ValueError, match="'steps'"):
            generate_sound_effects(prompt="rain", steps=steps)


class TestTrustGate:
    def test_untrusted_workflows_get_the_trust_refusal(self, monkeypatch, no_load):
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
        with pytest.raises(UntrustedWorkflowError):
            generate_sound_effects(prompt="rain", trust_remote_code=True)

    def test_trusted_workflows_get_a_plain_error(self, monkeypatch, no_load):
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "1")
        with pytest.raises(ValueError, match="trust_remote_code") as caught:
            generate_sound_effects(prompt="rain", trust_remote_code=True)
        assert not isinstance(caught.value, UntrustedWorkflowError)


class TestDuration:
    def test_text_only_defaults_to_8(self):
        assert sound_effects._check_duration(None, None) == 8.0

    def test_a_clip_sets_it_capped_at_20(self):
        assert sound_effects._check_duration(None, 30.0) == 20.0

    def test_20_is_allowed(self):
        assert sound_effects._check_duration(20, None) == 20.0


class TestWarnings:
    @pytest.fixture
    def warnings(self, monkeypatch):
        import dw.events

        seen = []
        monkeypatch.setattr(
            dw.events, "emit_warning", lambda message, **kw: seen.append(message)
        )
        return seen

    def sine(self, seconds=2):
        t = numpy.arange(44100 * seconds) / 44100
        return numpy.sin(2 * numpy.pi * 440 * t).astype(numpy.float32)[None]

    def test_white_noise_is_noted(self, warnings):
        noise = numpy.random.default_rng(0).standard_normal((1, 44100 * 2))
        sound_effects._warn(noise.astype(numpy.float32), 44100, 8.0)
        assert len(warnings) == 1
        assert "noise-like" in warnings[0]

    def test_a_sine_in_range_is_silent(self, warnings):
        sound_effects._warn(self.sine(), 44100, 8.0)
        assert warnings == []

    def test_a_short_duration_gets_only_the_duration_note(self, warnings):
        sound_effects._warn(self.sine(), 44100, 2.0)
        assert len(warnings) == 1
        assert "trained around" in warnings[0]

    def test_silence_is_not_noise(self, warnings):
        sound_effects._warn(numpy.zeros((1, 44100), dtype=numpy.float32), 44100, 8.0)
        assert not any("noise-like" in message for message in warnings)


class TestRunWithStubs:
    def test_text_only_run(self, monkeypatch):
        from dw.tasks import mmaudio_clip, mmaudio_model

        class Net:
            clip_seq_len = 64
            sync_seq_len = 192
            device = torch.device("cpu")

            def set_duration(self, duration):
                class Seq:
                    num_audio_frames = 10**7

                return Seq()

        class Clip:
            def encode_text(self, tokens):
                return torch.zeros(1, 77, 4)

        class Decoder:
            sample_rate = 44100

            def decode(self, latent):
                return torch.zeros(1, 1, 44100 * 6)

        monkeypatch.setattr(
            sound_effects,
            "_load_models",
            lambda device, dtype: (Net(), (Clip(), "tok"), None, Decoder()),
        )
        monkeypatch.setattr(mmaudio_clip, "tokenize", lambda tok, texts: texts)
        seen = {}

        def fake_sample(net, **kwargs):
            seen.update(kwargs)
            return torch.zeros(1, 10, 40)

        monkeypatch.setattr(mmaudio_model, "sample_latent", fake_sample)

        result = generate_sound_effects(prompt="door slam", duration=5, seed=3)

        assert isinstance(result, AudioTrack)
        assert result.sample_rate == 44100
        assert result.audio.shape == (1, round(5 * 44100))
        assert isinstance(seen["generator"], torch.Generator)
        assert seen["steps"] == 25
        assert seen["cfg_strength"] == 4.5
        assert seen["clip_features"] is None
        assert seen["text_features"] is not None


@pytest.fixture
def fractional_clip(tmp_path):
    """50 frames at 24 fps: 2.0833 s, the tester's qa-fit fixture's shape."""
    from diffusers.utils import export_to_video

    frames = [Image.new("RGB", (64, 64), (i * 5 % 256, 0, 0)) for i in range(50)]
    path = tmp_path / "fractional.mp4"
    export_to_video(frames, str(path), fps=24)
    return str(path)


class TestClipLengthDefault:
    """A clip's length is the default duration to the sample, fraction and all
    (#1133 bounce, C-F620: a 2.083 s clip came back as 2.0 s of sound)."""

    def test_frames_cover_the_whole_fractional_clip(self, fractional_clip):
        from dw.tasks.mmaudio_model import SequenceConfig

        seconds = sound_effects._clip_seconds(fractional_clip)
        assert seconds == pytest.approx(50 / 24)
        clip, sync, covered = sound_effects._frame_tensors(fractional_clip, seconds)
        assert covered == seconds
        seq = SequenceConfig(duration=covered)
        assert clip.shape[1] == seq.clip_seq_len
        # Synchformer's 16-frame segments 8 apart, 8 tokens each.
        assert ((sync.shape[1] - 16) // 8 + 1) * 8 == seq.sync_seq_len

    def test_a_clip_short_of_frames_falls_back_to_what_they_cover(
        self, monkeypatch, fractional_clip
    ):
        from dw import media

        real = media.sample_frames

        def short(path, fps, duration, size, crop=False):
            return real(path, fps, duration, size, crop=crop)[:-2]

        monkeypatch.setattr(media, "sample_frames", short)
        _, _, covered = sound_effects._frame_tensors(fractional_clip, 50 / 24)
        assert covered == pytest.approx(min(14 / 8, 50 / 25))

    def test_the_run_returns_the_clip_length(self, monkeypatch, fractional_clip):
        from dw.tasks import mmaudio_clip, mmaudio_model, mmaudio_sync_model
        from dw.tasks.mmaudio_model import SequenceConfig

        class Net:
            device = torch.device("cpu")

            def set_duration(self, duration):
                seq = SequenceConfig(duration=duration)
                self.clip_seq_len = seq.clip_seq_len
                self.sync_seq_len = seq.sync_seq_len
                return seq

        class Clip:
            def encode_frames(self, frames):
                return torch.zeros(1, frames.shape[1], 4)

            def encode_text(self, tokens):
                return torch.zeros(1, 77, 4)

        class Sync(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

        class Decoder:
            sample_rate = 44100

            def decode(self, latent):
                return torch.full((1, 1, 44100 * 3), 0.1)

        monkeypatch.setattr(
            sound_effects,
            "_load_models",
            lambda device, dtype: (Net(), (Clip(), "tok"), Sync(), Decoder()),
        )
        monkeypatch.setattr(mmaudio_clip, "tokenize", lambda tok, texts: texts)
        monkeypatch.setattr(
            mmaudio_sync_model,
            "encode_sync_frames",
            lambda model, frames: torch.zeros(
                1, ((frames.shape[1] - 16) // 8 + 1) * 8, 4
            ),
        )
        monkeypatch.setattr(
            mmaudio_model, "sample_latent", lambda net, **kw: torch.zeros(1, 10, 40)
        )

        result = generate_sound_effects(
            video=fractional_clip, prompt="a box sliding", seed=1
        )

        assert result.audio.shape == (1, round(50 / 24 * 44100))


def load_template(name):
    with open(os.path.join(ROOT, "workflows", "templates", name)) as handle:
        return json.load(handle)


class TestCatalog:
    def test_video_template(self):
        definition = load_template("video-to-sound-effects.json")
        meta = derive_catalog_metadata(definition)
        assert meta["shape"] == "audio"
        assert {"has-audio", "needs-input-media"} <= set(meta["traits"])
        commands = [step["task"]["command"] for step in definition["steps"]]
        assert "pair_audio" in commands

    def test_text_template(self):
        meta = derive_catalog_metadata(load_template("text-to-sound-effect.json"))
        assert meta["shape"] == "audio"
        assert "has-audio" in meta["traits"]
        assert "needs-input-media" not in meta["traits"]

    @pytest.mark.parametrize(
        "name", ["video-to-sound-effects.json", "text-to-sound-effect.json"]
    )
    def test_templates_carry_the_licences(self, name):
        description = load_template(name)["description"]
        assert "CC-BY-NC 4.0" in description
        assert LICENCE in description

    def test_the_command_summary_carries_the_licences(self):
        from dw.tasks.task import task_command_info

        summary = task_command_info("generate_sound_effects")["summary"]
        assert "CC-BY-NC 4.0" in summary
        assert LICENCE in summary

    def test_the_command_summary_documents_the_noise_warning(self):
        from dw.tasks.task import task_command_info

        summary = task_command_info("generate_sound_effects")["summary"]
        assert "noise-like" in summary
        assert "flatness >= 0.3" in summary
        assert "8 kHz" in summary


class TestLoadEvents:
    """A cold load names each model's repo and pinned revision on the run's
    events; a warm one says the models are cached (#1133 bounce)."""

    @pytest.fixture
    def events(self, monkeypatch):
        import huggingface_hub

        from dw import events as dw_events
        from dw.tasks import (
            mmaudio_clip,
            mmaudio_decoder,
            mmaudio_model,
            mmaudio_sync_model,
            model_cache,
        )

        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda repo_id=None, filename=None, revision=None, **_: (
                f"/fake/{repo_id}/{filename}"
            ),
        )
        monkeypatch.setattr(mmaudio_model, "load_network", lambda *a, **k: "net")
        monkeypatch.setattr(mmaudio_clip, "load_clip", lambda *a, **k: "clip")
        monkeypatch.setattr(mmaudio_clip, "load_clip_tokenizer", lambda *a: "tok")
        monkeypatch.setattr(
            mmaudio_sync_model, "load_synchformer", lambda *a, **k: "sync"
        )
        monkeypatch.setattr(mmaudio_decoder, "load_decoder", lambda *a, **k: "dec")
        monkeypatch.setattr(model_cache, "_cache", {})

        seen = []

        class Recorder(dw_events.RunContext):
            def emit(self, event_type, **data):
                seen.append((event_type, data))

        token = dw_events.activate_context(Recorder())
        yield seen
        dw_events.deactivate_context(token)

    def revisions(self):
        from dw.tasks import mmaudio_clip, mmaudio_decoder, mmaudio_model

        return {
            mmaudio_model.REPO_ID: mmaudio_model.REVISION,
            mmaudio_clip.CLIP_REPO_ID: mmaudio_clip.CLIP_REVISION,
            mmaudio_decoder.K6_REPO_ID: mmaudio_decoder.K6_REVISION,
        }

    def test_a_cold_load_names_every_repo_and_revision(self, events):
        sound_effects._load_models("cpu", torch.float32)

        logs = [data for kind, data in events if kind == "log"]
        assert len(logs) == 4
        assert {log["repo_id"]: log["revision"] for log in logs} == self.revisions()
        assert all(log["cached"] is False for log in logs)
        for log in logs:
            assert log["repo_id"] in log["message"]
            assert log["revision"] in log["message"]

        phases = [data for kind, data in events if kind == "phase"]
        assert [p["phase"] for p in phases] == ["loading"] * 4 + ["task"]
        for phase in phases[:4]:
            assert any(repo in phase["detail"] for repo in self.revisions())

    def test_a_warm_load_logs_cache_hits_without_a_phase(self, events):
        sound_effects._load_models("cpu", torch.float32)
        events.clear()

        sound_effects._load_models("cpu", torch.float32)

        assert [kind for kind, _ in events] == ["log"] * 4
        assert all(data["cached"] is True for _, data in events)
        assert {d["repo_id"]: d["revision"] for _, d in events} == self.revisions()


class TestRegistration:
    def test_it_consumes_the_device_and_declares_its_weights(self):
        from dw.tasks.registry import _COMMAND_REGISTRY, _COMMAND_RULES
        from dw.tasks.task import task_command_info

        task_command_info("generate_sound_effects")
        assert (
            _COMMAND_RULES["generate_sound_effects"]["required_weights"]
            == "dw.tasks.sound_effects.required_weights"
        )
        # a command that does not consume the device is wrapped to drop it
        assert (
            _COMMAND_REGISTRY["generate_sound_effects"].__name__
            == "_handle_generate_sound_effects"
        )
