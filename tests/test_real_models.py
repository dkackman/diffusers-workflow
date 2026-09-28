"""The catalog's reference models, run for real on this machine's accelerator.

Everything else in the suite mocks the pipeline, so none of it can see a
wrong image, a seed that stops reproducing on a new torch, or an audio chain
that drops a parameter between steps. These run the stock templates' own
models and check the output itself. They are the release gate
(`pytest -m integration`), promoted from the regression suite's smoke cases
named on each test.
"""

import copy
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from dw.events import RunContext
from dw.workflow import Workflow
from tests.test_examples import REPO_ROOT

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (torch.cuda.is_available() or torch.backends.mps.is_available()),
        reason="runs real models; needs an accelerator",
    ),
]

TEMPLATES = os.path.join(REPO_ROOT, "workflows", "templates")


def _template(name):
    with open(os.path.join(TEMPLATES, name), encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="module")
def pipelines():
    """SD 1.5 loads once for the module rather than once per run."""
    return {}


def _run(definition, output_dir, arguments, pipelines=None):
    warnings = []

    def sink(event):
        if event["event"] == "warning":
            warnings.append(event)

    workflow = Workflow(
        definition, output_dir, os.path.join(TEMPLATES, "text-to-image.json")
    )
    workflow.validate(arguments)
    workflow.run(
        arguments, previous_pipelines=pipelines, context=RunContext(on_event=sink)
    )
    return workflow.manifest, warnings


def _is_solid(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.float32)
    return float(pixels.std()) < 2.0


def test_a_fixed_seed_reproduces_the_image_at_the_requested_size(tmp_path, pipelines):
    """S-F004. file_base_name is in the step's cache key but not in the image,
    so the second run really denoises again rather than being served from
    the step cache - identical bytes are the seed's doing."""
    definition = _template("text-to-image.json")
    arguments = definition["steps"][0]["pipeline"]["arguments"]
    arguments.update({"width": 384, "height": 384, "num_inference_steps": 20})

    files = []
    for base_name in ("a", "b"):
        run = copy.deepcopy(definition)
        run["steps"][0]["result"]["file_base_name"] = base_name
        manifest, _ = _run(run, str(tmp_path), {"seed": 424242}, pipelines)
        [entry] = manifest
        assert not entry.get("reused")
        [path] = entry["files"]
        files.append(path)

    rendered = []
    for path in files:
        with Image.open(path) as image:
            assert image.size == (384, 384)
            rendered.append(np.asarray(image.convert("RGB")))
    # Pixels rather than bytes: each file's embedded metadata names its own run
    assert np.array_equal(*rendered), "the same seed rendered two different images"


def test_the_reference_template_renders_the_seed_the_safety_checker_blanked(
    tmp_path, pipelines
):
    """S-F020 (#133). This seed made SD 1.5's checker return a black frame
    under a succeeded job. The stock template loads without the checker, so
    it must render a picture here, and say nothing about blanking."""
    manifest, warnings = _run(
        _template("text-to-image.json"),
        str(tmp_path),
        {"prompt": "an apple", "seed": 3220371727974403},
        pipelines,
    )

    [entry] = manifest
    [path] = entry["files"]
    assert not _is_solid(path), f"{path} is a solid frame"
    assert not [w for w in warnings if w.get("kind") == "safety_checker_blanked"]


def test_a_speech_chain_keeps_every_parameter_to_the_saved_file(tmp_path):
    """S-F007. A chain that silently loses a parameter partway through is
    worse than one that fails, so the saved file is checked against every
    input: the slice's length, the voice's own rate carried through both
    tasks, and the fades at either end."""
    import soundfile

    definition = {
        "id": "speech_chain",
        "seed": 7,
        "steps": [
            {
                "name": "speak",
                "task": {
                    "command": "generate_speech",
                    "arguments": {
                        "text": "Somebody finished the pistachio and nobody "
                        "is saying anything.",
                        "model_name": _template("generate-speech.json")["variables"][
                            "model_name"
                        ],
                    },
                },
            },
            {
                "name": "clip",
                "task": {
                    "command": "slice_audio",
                    "arguments": {
                        "audio": "previous_result:speak",
                        "start_seconds": 0.5,
                        "duration_seconds": 1.5,
                    },
                },
            },
            {
                "name": "faded",
                "task": {
                    "command": "fade_audio",
                    "arguments": {
                        "audio": "previous_result:clip",
                        "fade_in_ms": 100,
                        "fade_out_ms": 100,
                    },
                },
                "result": {"content_type": "audio/wav"},
            },
        ],
    }

    manifest, _ = _run(definition, str(tmp_path), {})

    [entry] = [e for e in manifest if e["files"]]
    assert entry["step"] == "faded"
    [path] = entry["files"]
    samples, rate = soundfile.read(path, always_2d=True)
    # facebook/mms-tts-eng is a VITS checkpoint that generates at 16 kHz
    assert rate == 16000
    assert samples.shape[0] == int(1.5 * rate)
    level = np.abs(samples).max(axis=1)
    edge = int(0.01 * rate)
    assert level.max() > 0.01, "the clip is silent"
    assert level[:edge].max() < level.max() / 4, "no fade in"
    assert level[-edge:].max() < level.max() / 4, "no fade out"
