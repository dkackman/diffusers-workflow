"""A lora whose model_name is null is not loaded - the off switch.

A template's `loras` list is fixed JSON, so the only way a caller could run
a step without its adapter was to rewrite the workflow. Passing the lora's
variables as null looked like the answer and was worse than refusing it:
argument_errors and validation_errors both answered clean, the job queued,
the pipeline loaded for minutes, and load_loras then died on float(None)
with a TypeError that named no lora. A null model_name now means the entry
is skipped, validation says so at the path the caller wrote, and the run
says so again in its warnings.
"""

import json
import os

import pytest

from dw.adapter_compatibility import adapter_warnings, warn_adapters
from dw.pipeline_processors.pipeline import active_loras, load_loras
from dw.workflow import Workflow

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H3_TEMPLATE = os.path.join(
    REPO_ROOT, "workflows", "templates", "minimax", "video-with-audio.json"
)
LORA_VARIABLES = (
    "lora_model_name",
    "lora_weight_name",
    "lora_adapter_name",
    "lora_scale",
    "lora_alpha",
)


class RecordingPipeline:
    def __init__(self):
        self.loaded = []
        self.adapters = None

    def load_lora_weights(self, model_name, **kwargs):
        self.loaded.append((model_name, kwargs))

    def set_adapters(self, names, weights):
        self.adapters = (names, weights)


def h3_workflow(tmp_path):
    with open(H3_TEMPLATE, encoding="utf-8") as f:
        definition = json.load(f)
    return Workflow(
        definition,
        str(tmp_path),
        H3_TEMPLATE,
        workflow_dir=os.path.join(REPO_ROOT, "workflows"),
    )


class TestLoadLoras:
    def test_an_all_null_entry_loads_nothing(self):
        pipeline = RecordingPipeline()
        load_loras(
            [
                {
                    key: None
                    for key in (
                        "model_name",
                        "weight_name",
                        "adapter_name",
                        "scale",
                        "alpha",
                    )
                }
            ],
            pipeline,
        )
        assert pipeline.loaded == []
        assert pipeline.adapters is None

    def test_a_null_model_name_alone_is_enough(self):
        pipeline = RecordingPipeline()
        load_loras(
            [{"model_name": None, "weight_name": "turbo.safetensors", "scale": 1.0}],
            pipeline,
        )
        assert pipeline.loaded == []
        assert pipeline.adapters is None

    def test_the_other_entries_still_load(self):
        pipeline = RecordingPipeline()
        load_loras(
            [
                {"model_name": None, "adapter_name": "turbo"},
                {"model_name": "user/style", "adapter_name": "style", "scale": 0.5},
            ],
            pipeline,
        )
        assert [name for name, _ in pipeline.loaded] == ["user/style"]
        assert pipeline.adapters == (["style"], [0.5])

    def test_null_scale_and_adapter_name_take_their_defaults(self):
        """Nulled one at a time, these were float(None) and a None adapter."""
        pipeline = RecordingPipeline()
        load_loras(
            [{"model_name": "user/style", "adapter_name": None, "scale": None}],
            pipeline,
        )
        assert pipeline.loaded == [("user/style", {"adapter_name": "0"})]
        assert pipeline.adapters == (["0"], [1.0])

    def test_active_loras_is_what_placement_defers_for(self):
        assert active_loras(None) == []
        assert active_loras([{"model_name": None}]) == []
        assert active_loras([{"model_name": None}, {"model_name": "m"}]) == [
            {"model_name": "m"}
        ]


class TestValidation:
    def test_all_null_validates(self, tmp_path):
        arguments = {name: None for name in LORA_VARIABLES}
        assert h3_workflow(tmp_path).validation_errors(arguments=arguments) == []

    def test_the_caller_is_told_where_they_switched_it_off(self, tmp_path):
        warnings = h3_workflow(tmp_path).adapter_warnings({"lora_model_name": None})
        disabled = [w for w in warnings if "not loaded" in w]
        assert len(disabled) == 1
        assert disabled[0].startswith("arguments.lora_model_name: ")
        assert "num_inference_steps" in disabled[0]

    def test_nothing_is_said_when_the_lora_is_on(self, tmp_path):
        warnings = h3_workflow(tmp_path).adapter_warnings()
        assert not [w for w in warnings if "not loaded" in w]

    def test_a_literal_null_is_reported_at_its_step(self):
        definition = {
            "steps": [
                {
                    "name": "image",
                    "pipeline": {"loras": [{"model_name": None}]},
                }
            ]
        }
        (warning,) = adapter_warnings(definition)
        assert warning.startswith("steps[0].pipeline.loras[0].model_name: ")

    def test_the_run_says_it_too(self, monkeypatch):
        emitted = []
        monkeypatch.setattr(
            "dw.events.emit_warning",
            lambda message, **data: emitted.append((message, data.get("kind"))),
        )
        warn_adapters(
            {
                "steps": [
                    {"name": "image", "pipeline": {"loras": [{"model_name": None}]}}
                ]
            }
        )
        assert [kind for _, kind in emitted] == ["lora_disabled"]


def test_the_schema_takes_a_literal_null():
    with open(
        os.path.join(REPO_ROOT, "dw", "workflow_schema.json"), encoding="utf-8"
    ) as f:
        schema = json.load(f)
    assert "null" in schema["$defs"]["lora"]["properties"]["model_name"]["type"]


@pytest.mark.parametrize("name", LORA_VARIABLES)
def test_every_lora_variable_is_declared_on_the_template(name):
    """The skill names these as the off switch; a rename breaks it."""
    with open(H3_TEMPLATE, encoding="utf-8") as f:
        assert name in json.load(f)["variables"]
