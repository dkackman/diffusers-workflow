"""What a task step fetches from the Hub (#975): each model-backed command's
`required_weights` hook, that the loader resolves its repo through the same
helper, and the pin that a task module touching the Hub declares one."""

import ast
import glob
import importlib
import os
import re
from unittest.mock import MagicMock

import numpy
import pytest
from PIL import Image

from dw.tasks.registry import required_weights_hook


def declared(command, arguments=None):
    """The repos `command`'s hook names for `arguments`, in order."""
    hook = required_weights_hook(command)
    assert hook is not None, command
    return [entry["repo"] for entry in hook(arguments or {})]


class TestSegment:
    def test_the_defaults_are_the_detector_and_the_segmenter(self):
        assert declared("segment") == [
            "IDEA-Research/grounding-dino-base",
            "facebook/sam2-hiera-large",
        ]

    def test_model_name_overrides_the_detector(self):
        assert declared("segment", {"model_name": "org/dino"}) == [
            "org/dino",
            "facebook/sam2-hiera-large",
        ]

    def test_sam_model_name_overrides_the_segmenter(self):
        assert declared("segment", {"sam_model_name": "org/sam"}) == [
            "IDEA-Research/grounding-dino-base",
            "org/sam",
        ]

    def test_a_null_override_is_the_default(self):
        assert declared("segment", {"model_name": None}) == [
            "IDEA-Research/grounding-dino-base",
            "facebook/sam2-hiera-large",
        ]

    def test_a_reference_or_a_local_path_gives_no_entry_for_its_slot(self):
        assert declared("segment", {"model_name": "previous_result:dino"}) == [
            "facebook/sam2-hiera-large"
        ]
        assert declared("segment", {"sam_model_name": "/models/sam"}) == [
            "IDEA-Research/grounding-dino-base"
        ]
        assert (
            declared(
                "segment",
                {"model_name": "./models/dino", "sam_model_name": "/models/sam"},
            )
            == []
        )


class TestRemoveBackground:
    def test_is_the_one_fixed_repo(self):
        assert declared("remove_background") == ["briaai/RMBG-2.0"]

    def test_the_command_name_is_not_case_sensitive(self):
        assert declared("Remove_Background") == ["briaai/RMBG-2.0"]


class TestDiffusionUpscale:
    def test_the_default_mode_is_x4(self):
        assert declared("diffusion_upscale") == [
            "stabilityai/stable-diffusion-x4-upscaler"
        ]

    def test_x2_mode_names_the_latent_upscaler(self):
        assert declared("diffusion_upscale", {"mode": "x2"}) == [
            "stabilityai/sd-x2-latent-upscaler"
        ]

    def test_model_name_overrides_the_mode_default(self):
        assert declared(
            "diffusion_upscale", {"mode": "x2", "model_name": "org/up"}
        ) == ["org/up"]

    def test_an_unknown_mode_names_nothing(self):
        assert declared("diffusion_upscale", {"mode": "x8"}) == []

    def test_a_reference_mode_names_nothing(self):
        assert declared("diffusion_upscale", {"mode": "previous_result:m"}) == []

    def test_a_path_model_name_names_nothing(self):
        assert declared("diffusion_upscale", {"model_name": "/models/up"}) == []


class TestAudioTranscription:
    @pytest.mark.parametrize("command", ["transcribe_audio", "check_script"])
    def test_the_default_is_whisper_base(self, command):
        assert declared(command) == ["openai/whisper-base"]

    @pytest.mark.parametrize("command", ["transcribe_audio", "check_script"])
    def test_model_name_overrides_it(self, command):
        assert declared(command, {"model_name": "openai/whisper-small"}) == [
            "openai/whisper-small"
        ]

    def test_a_reference_names_nothing(self):
        assert declared("transcribe_audio", {"model_name": "previous_result:m"}) == []


class TestSpeechGeneration:
    def test_the_default_is_bark_small(self):
        assert declared("generate_speech") == ["suno/bark-small"]

    def test_model_name_overrides_it(self):
        assert declared("generate_speech", {"model_name": "org/tts"}) == ["org/tts"]

    def test_a_speaker_embedding_adds_the_speaker_encoder(self):
        assert declared(
            "generate_speech",
            {"model_name": "microsoft/speecht5_tts", "speaker_embedding": "me.wav"},
        ) == ["microsoft/speecht5_tts", "speechbrain/spkrec-xvect-voxceleb"]

    def test_a_local_model_still_lists_the_speaker_encoder(self):
        assert declared(
            "generate_speech",
            {"model_name": "/models/tts", "speaker_embedding": "me.wav"},
        ) == ["speechbrain/spkrec-xvect-voxceleb"]


class TestVoiceAttribution:
    def test_is_only_the_speaker_embedder(self):
        assert declared("attribute_voices") == ["speechbrain/spkrec-ecapa-voxceleb"]

    def test_the_stem_separator_is_not_listed(self):
        assert required_weights_hook("separate_stems") is None


class TestTextGeneration:
    def test_the_default_is_a_text_model(self):
        assert declared("text_generation") == ["Qwen/Qwen2.5-1.5B-Instruct"]

    def test_an_image_selects_the_vision_default(self):
        assert declared("text_generation", {"image": "cup.png"}) == [
            "HuggingFaceTB/SmolVLM-256M-Instruct"
        ]

    def test_an_empty_image_is_the_text_default(self):
        assert declared("text_generation", {"image": None}) == [
            "Qwen/Qwen2.5-1.5B-Instruct"
        ]

    def test_model_name_overrides_either_default(self):
        assert declared("text_generation", {"model_name": "org/lm"}) == ["org/lm"]
        assert declared("text_generation", {"model_name": "org/vlm", "image": "x"}) == [
            "org/vlm"
        ]

    def test_a_local_path_names_nothing(self):
        assert declared("text_generation", {"model_name": "/models/lm"}) == []


class TestVisionTasks:
    @pytest.mark.parametrize("command", ["image_to_text", "judge"])
    def test_the_default_is_the_vision_model(self, command):
        assert declared(command) == ["HuggingFaceTB/SmolVLM-256M-Instruct"]

    @pytest.mark.parametrize("command", ["image_to_text", "judge"])
    def test_model_name_overrides_it(self, command):
        assert declared(command, {"model_name": "org/vlm"}) == ["org/vlm"]

    @pytest.mark.parametrize("command", ["image_to_text", "judge"])
    def test_a_reference_names_nothing(self, command):
        assert declared(command, {"model_name": "previous_result:m"}) == []


class TestImageProcessors:
    @pytest.mark.parametrize(
        "processor, repo",
        [
            ("depth", "Intel/dpt-hybrid-midas"),
            ("depth_estimator", "Intel/dpt-large"),
            ("depth_estimator_tensor", "Intel/dpt-large"),
            ("openpose", "lllyasviel/Annotators"),
            ("mlsd", "lllyasviel/Annotators"),
            ("hed", "lllyasviel/Annotators"),
            ("lineart", "lllyasviel/Annotators"),
            ("teed", "fal-ai/teed"),
            ("anyline", "TheMistoAI/MistoLine"),
            ("sam", "ybelkada/segment-anything"),
            ("segmentation", "openmmlab/upernet-convnext-small"),
        ],
    )
    def test_a_model_backed_processor_names_its_repo(self, processor, repo):
        assert declared(processor) == [repo]

    @pytest.mark.parametrize(
        "processor",
        ["canny_cv", "resize_rescale", "crop_square", "dw_pose", "zoe_depth"],
    )
    def test_a_processor_that_fetches_nothing_from_the_hub_names_nothing(
        self, processor
    ):
        assert declared(processor) == []


class TestNoEntry:
    @pytest.mark.parametrize("command", ["separate_stems", "no_such_command"])
    def test_a_command_without_a_hook_has_none(self, command):
        assert required_weights_hook(command) is None

    def test_a_non_string_command_has_none(self):
        assert required_weights_hook(None) is None

    @pytest.mark.parametrize(
        "name", ["previous_result:x", "variable:m", "/models/dino", "./models/x", 3]
    )
    @pytest.mark.parametrize(
        "command, remaining",
        [
            ("segment", ["facebook/sam2-hiera-large"]),
            ("text_generation", []),
            ("transcribe_audio", []),
            ("diffusion_upscale", []),
            ("generate_speech", []),
        ],
    )
    def test_a_name_that_is_not_a_hub_id_never_raises_and_gives_no_entry(
        self, command, remaining, name
    ):
        assert declared(command, {"model_name": name}) == remaining

    def test_missing_arguments_are_the_defaults(self):
        assert required_weights_hook("transcribe_audio")(None) == [
            {"repo": "openai/whisper-base"}
        ]


class Reached(Exception):
    """Raised by a recording stand-in once it has what the test wants."""


def recording_cache(monkeypatch, module, stop_at=None):
    """Replace `module.cached_model` with a recorder of the cache keys it was
    given - the key carries the repo the loader resolved. It raises Reached on
    the first key whose first element is `stop_at` (or on the first, when None)
    without running the factory, so no model loads."""
    keys = []

    def fake(key, factory):
        keys.append(key)
        if stop_at is None or key[0] == stop_at:
            raise Reached
        return (None, None)

    monkeypatch.setattr(module, "cached_model", fake)
    return keys


class TestLoaderAgreesWithDeclaration:
    """The hook and the loader read one resolution helper, so what the plan
    lists is what the run loads, defaults and overrides alike."""

    @pytest.mark.parametrize(
        "arguments",
        [
            {},
            {"model_name": "org/dino"},
            {"sam_model_name": "org/sam"},
            {"model_name": "org/dino", "sam_model_name": "org/sam"},
        ],
    )
    def test_segment_loads_the_repos_it_declares(self, monkeypatch, arguments):
        from dw.tasks import segment

        keys = []
        boxes = MagicMock()
        boxes.__len__.return_value = 1
        processor = MagicMock()
        processor.post_process_grounded_object_detection.return_value = [
            {"boxes": boxes, "scores": MagicMock(), "labels": []}
        ]

        def fake(key, factory):
            keys.append(key)
            if key[0] == "segment_sam":
                raise Reached
            return processor, MagicMock()

        monkeypatch.setattr(segment, "cached_model", fake)
        with pytest.raises(Reached):
            segment.segment_image(Image.new("RGB", (8, 8)), "cup", **arguments)

        assert [key[1] for key in keys] == declared("segment", arguments)

    @pytest.mark.parametrize(
        "arguments",
        [
            {},
            {"mode": "x2"},
            {"model_name": "org/up"},
            {"mode": "x2", "model_name": "org/up"},
        ],
    )
    def test_diffusion_upscale_loads_the_repo_it_declares(self, monkeypatch, arguments):
        from dw.tasks import diffusion_upscale

        keys = recording_cache(monkeypatch, diffusion_upscale)
        with pytest.raises(Reached):
            diffusion_upscale.diffusion_upscale(Image.new("RGB", (8, 8)), **arguments)

        assert [key[2] for key in keys] == declared("diffusion_upscale", arguments)

    @pytest.mark.parametrize("arguments", [{}, {"model_name": "org/asr"}])
    def test_transcribe_audio_loads_the_repo_it_declares(self, monkeypatch, arguments):
        from dw.tasks import audio_transcription

        keys = recording_cache(monkeypatch, audio_transcription)
        with pytest.raises(Reached):
            audio_transcription.transcribe_audio(
                numpy.zeros(1600, dtype=numpy.float32), sample_rate=16000, **arguments
            )

        assert [key[1] for key in keys] == declared("transcribe_audio", arguments)

    @pytest.mark.parametrize("arguments", [{}, {"model_name": "org/tts"}])
    def test_generate_speech_loads_the_repo_it_declares(self, monkeypatch, arguments):
        from dw.tasks import speech_generation

        keys = recording_cache(monkeypatch, speech_generation)
        with pytest.raises(Reached):
            speech_generation.generate_speech("hello", **arguments)

        assert [key[1] for key in keys] == declared("generate_speech", arguments)

    @pytest.mark.parametrize(
        "arguments",
        [
            {},
            {"model_name": "org/lm"},
            {"image": Image.new("RGB", (8, 8))},
            {"image": Image.new("RGB", (8, 8)), "model_name": "org/vlm"},
        ],
    )
    def test_generate_text_loads_the_repo_it_declares(self, monkeypatch, arguments):
        from dw.tasks import text_generation

        keys = recording_cache(monkeypatch, text_generation)
        with pytest.raises(Reached):
            text_generation.generate_text("hi", **arguments)

        assert [key[2] for key in keys] == declared("text_generation", arguments)


TASKS = os.path.join(os.path.dirname(__file__), "..", "dw", "tasks")

# A module reaching the Hub by any of these loads weights the plan has to name
HUB_LOADS = re.compile(
    r"hf_hub_download|from_pretrained|from_hparams|hf_pipeline\(|pipeline\(\s*[\"']"
)

# Modules that fetch only outside the Hugging Face cache would be listed here
# with the reason; none do - voice_attribution's demucs download shares a module
# with a speechbrain load, which it declares.
OUTSIDE_THE_HF_CACHE = {}


def task_modules():
    for path in sorted(glob.glob(os.path.join(TASKS, "*.py"))):
        yield os.path.basename(path)[:-3], path


def declares_required_weights(path):
    """Whether the module defines `required_weights` or imports one under that
    name, at module level."""
    with open(path, encoding="utf-8") as file:
        tree = ast.parse(file.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "required_weights":
            return True
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == "required_weights":
                    return True
    return False


def hub_loading_modules():
    found = set()
    for name, path in task_modules():
        with open(path, encoding="utf-8") as file:
            if HUB_LOADS.search(file.read()):
                found.add(name)
    return found


# Commands that share a module with a declared hook but name none, and why
LOADS_NOTHING_THE_PLAN_LISTS = {
    # demucs downloads outside the Hugging Face cache
    "separate_stems": "voice_attribution's demucs",
    # Decodes with MiniMaxAI/MiniMax-H3's VAE, which the H3 pipeline step
    # that made the latents lists whole (#975 Q4)
    "decode_h3_latents": "the H3 repo, listed by its pipeline step",
    # Pastes crops back; loads no weights
    "paste_face_track": "nothing",
}


class TestEveryHubLoadIsDeclared:
    def test_the_scan_finds_the_modules_that_load_from_the_hub(self):
        found = hub_loading_modules()
        assert {
            "segment",
            "diffusion_upscale",
            "audio_transcription",
            "speech_generation",
            "text_generation",
            "depth_estimator",
            "image_utils",
        } <= found

    def test_a_module_that_loads_from_the_hub_declares_required_weights(self):
        undeclared = sorted(
            name
            for name, path in task_modules()
            if name in hub_loading_modules()
            and name not in OUTSIDE_THE_HF_CACHE
            and not declares_required_weights(path)
        )
        assert not undeclared, (
            f"{undeclared} load from the Hub with no required_weights, so the "
            "plan cannot list what they download (#975)"
        )

    def test_every_registered_command_in_a_declaring_module_names_its_hook(self):
        from dw.tasks import registry

        importlib.import_module("dw.tasks.model_handlers")
        declaring = {
            name
            for name, path in task_modules()
            if name in hub_loading_modules() and declares_required_weights(path)
        }
        unnamed = []
        for command, rules in registry._COMMAND_RULES.items():
            module = (rules.get("implementation") or "").split(".")
            if len(module) < 4 or module[:2] != ["dw", "tasks"]:
                continue
            if command in LOADS_NOTHING_THE_PLAN_LISTS:
                continue
            if module[2] in declaring and not rules.get("required_weights"):
                unnamed.append(command)
        assert not unnamed, f"{unnamed} register without required_weights"

    def test_every_named_hook_resolves_to_a_callable(self):
        from dw.tasks import registry

        importlib.import_module("dw.tasks.model_handlers")
        for command, rules in registry._COMMAND_RULES.items():
            if rules.get("required_weights"):
                assert callable(required_weights_hook(command)), command


def _dw_imports(path):
    """Every `dw` module `path` imports, top-level or inside a function (grimp
    counts both as edges), as dotted names relative to dw/tasks."""
    found = set()
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.ImportFrom) and (
            node.level or (node.module or "").startswith("dw.")
        ):
            found.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.startswith("dw.")}
    return found


class TestTheHookPathStaysAcyclic:
    """The plan asks the registry for a hook; the registry is read by
    dw/locations.py; the task modules import the location policy and
    dw/tasks/weights.py. So neither the registry nor weights.py may import a
    task module or the location policy, even lazily, or the architecture
    ratchet's `import_cycles` counts the loop (the one it did count: weights
    -> locations -> registry -> image_utils -> background_remover -> weights).
    The registry reaches a processor's hook through `register_weights_fallback`
    instead, and `validate_weight_name` lives in dw/security.py."""

    def test_weights_imports_only_the_security_leaf_from_dw(self):
        assert _dw_imports(os.path.join(TASKS, "weights.py")) == {"..security"}

    def test_the_registry_imports_nothing_from_dw(self):
        assert _dw_imports(os.path.join(TASKS, "registry.py")) == set()

    def test_a_processor_hook_is_registered_not_imported(self):
        from dw.tasks import image_utils, registry

        assert image_utils.processor_weights_hook in registry._WEIGHTS_FALLBACKS
        assert required_weights_hook("Remove_Background")({}) == [
            {"repo": "briaai/RMBG-2.0"}
        ]
        assert image_utils.processor_weights_hook("not_a_processor") is None
