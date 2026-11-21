"""The Pika H3 Flex templates (#1316, stage #1343): MiniMax-H3's weights under
Pika's adapter, on the fixed 7-point list Pika's blocks run."""

import json
import os

import pytest

from dw.adapter_compatibility import PARTITIONS, _problems, _trained_for
from dw.step_passes import STEP_LIST_KEY, step_list_passes

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(REPO_ROOT, "workflows", "templates", "pika-h3")
PIKA = "Pika-Labs/Pika-H3-Flex"
MINIMAX = "MiniMaxAI/MiniMax-H3"
WEIGHTS = {"transformer", "transformer_ref", "text_encoder", "vae", "audio_vae"}
PIKA_OWN = {"scheduler", "audio_scheduler", "tokenizer", "processor"}
STEP_LIST = [1000, 750, 500, 250, 100, 40, 0]

TEMPLATES = {
    "text-to-video.json": ("t2va", "fl2va/adapter.safetensors", "transformer"),
    "first-and-last-frame.json": ("fl2va", "fl2va/adapter.safetensors", "transformer"),
    "reference-to-video.json": (
        "ref2va",
        "ref2va/adapter.safetensors",
        "transformer_ref",
    ),
}


def _load(name):
    with open(os.path.join(FOLDER, name), encoding="utf-8") as handle:
        return json.load(handle)


def _pipeline(name):
    (step,) = _load(name)["steps"]
    return step["pipeline"]


def _keys(value):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from _keys(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _keys(inner)


def test_the_folder_holds_the_three_templates():
    found = {n for n in os.listdir(FOLDER) if n.endswith(".json")}
    assert found == set(TEMPLATES)


@pytest.mark.parametrize("name", TEMPLATES)
def test_no_shift_and_no_num_inference_steps(name):
    # As keys anywhere: the description names both, to say why they are absent
    definition = _load(name)
    found = set(_keys(definition["variables"])) | set(_keys(definition["steps"]))
    assert "num_inference_steps" not in found
    assert not [key for key in found if "shift" in key]


@pytest.mark.parametrize("name", TEMPLATES)
def test_the_stated_step_list_is_the_seven_point_list(name):
    pipeline = _pipeline(name)
    assert pipeline["configuration"]["configs"][STEP_LIST_KEY] == STEP_LIST
    assert step_list_passes(pipeline) == 6
    # Stated where update_components sets it, never as a call argument
    assert STEP_LIST_KEY not in pipeline["arguments"]
    assert STEP_LIST_KEY not in pipeline["from_pretrained_arguments"]


@pytest.mark.parametrize("name", TEMPLATES)
def test_the_workflow_runs_from_pikas_repo(name):
    workflow, _weight, _partition = TEMPLATES[name]
    pipeline = _pipeline(name)
    assert pipeline["configuration"]["component_type"] == "ModularPipeline"
    assert pipeline["from_pretrained_arguments"] == {
        "model_name": PIKA,
        "workflow": workflow,
    }


@pytest.mark.parametrize("name", TEMPLATES)
def test_the_five_weight_components_load_from_minimax_h3(name):
    load = _pipeline(name)["configuration"]["load_components"]
    sources = load["pretrained_model_name_or_path"]
    assert sources == {component: MINIMAX for component in WEIGHTS}
    # No default: the scheduler, tokenizer and processor stay Pika's
    assert "default" not in sources and not PIKA_OWN & set(sources)
    revisions = load["revision"]
    assert set(revisions) == WEIGHTS | {"default"}
    pinned = {revisions[c] for c in WEIGHTS}
    assert len(pinned) == 1 and len(pinned.pop()) == 40
    assert len(revisions["default"]) == 40


@pytest.mark.parametrize("name", TEMPLATES)
def test_each_adapter_routes_to_its_partition(name):
    workflow, weight, partition = TEMPLATES[name]
    pipeline = _pipeline(name)
    (lora,) = pipeline["loras"]
    assert lora["model_name"] == PIKA
    assert lora["weight_name"] == weight
    assert lora["scale"] == 1.0
    assert "alpha" not in lora
    assert PARTITIONS[_trained_for(weight)] == partition
    # The real pairing check validate runs: no warning, no refusal
    assert _problems(workflow, weight, PIKA) == []
    # The partition the adapter lands on is the one the template offloads
    components = pipeline["configuration"]["components"]
    assert partition in components
    assert pipeline["configuration"]["load_components"]["quantization_config"][
        partition
    ]


@pytest.mark.parametrize("name", TEMPLATES)
def test_cache_is_off_by_default(name):
    definition = _load(name)
    assert _pipeline(name)["configuration"]["cache"] == "variable:cache"
    assert definition["variables"]["cache"] is None


@pytest.mark.parametrize("name", TEMPLATES)
def test_the_presets_are_the_16_9_buckets(name):
    definition = _load(name)
    assert definition["presets"] == {
        "resolution": {
            "480p": {"width": 832, "height": 480},
            "768p": {"width": 1344, "height": 768},
        }
    }
    constraints = definition["variable_constraints"]
    assert constraints["width"]["modulus"] == 32
    assert constraints["height"]["modulus"] == 32
    frames = constraints["num_frames"]
    assert (frames["modulus"], frames["remainder"]) == (17, 5)
    assert (frames["min_frames"], frames["max_frames"], frames["snap"]) == (
        124,
        345,
        "up",
    )


@pytest.mark.parametrize("name", TEMPLATES)
def test_the_description_carries_the_caveats(name):
    description = _load(name)["description"]
    assert "Seeds are not comparable with the H3" in description
    assert "base match itself is sampled" in description
    for ratio in ("9:16", "1:1", "4:3", "3:4", "21:9"):
        assert ratio in description


def test_ref2va_uses_pikas_reference_classes():
    references = _pipeline("reference-to-video.json")["arguments"]["references"]
    types = [r["reference_type"] for r in references]
    image = "diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexImageReference"
    assert types == [
        image,
        image,
        image,
        "diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexAudioReference",
    ]


def test_ref2va_takes_up_to_three_images_the_extra_two_off_by_default():
    """#1343 bounce: a second and third image reference, @Image2 and @Image3,
    null by default so the default run sends one image (#478 drops a null)."""
    template = _load("reference-to-video.json")
    sources = [
        r["from_file"]
        for r in _pipeline("reference-to-video.json")["arguments"]["references"]
    ]
    assert sources == [
        "variable:subject",
        "variable:subject_2",
        "variable:subject_3",
        "variable:voice",
    ]
    assert template["variables"]["subject_2"] is None
    assert template["variables"]["subject_3"] is None
    assert "@Image2" in template["description"]
    assert "@Image3" in template["description"]
