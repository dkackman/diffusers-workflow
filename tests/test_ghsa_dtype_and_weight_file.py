"""Two security-suite findings, pinned where the gate holds.

GHSA-623p-847m-275p (SE-F063): a per-component torch_dtype dict's values are
put to the untrusted type gate at validation and at run time, naming the
component - not only a plain string torch_dtype.

GHSA-ch56-2p9f-95cg (SE-F052): a task's `filename` names a file inside the
model repo (upscale, restore_faces, interpolate_frames), so an absolute or
'..' name is refused at validation and again before hf_hub_download.
"""

import pytest

from dw.arguments import realize_per_component_dtype
from dw.locations import location_errors
from dw.security import SecurityError, UntrustedWorkflowError
from dw.type_references import component_type_errors


@pytest.fixture
def untrusted(monkeypatch):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")


# --- GHSA-623p: per-component torch_dtype ---------------------------------

_GATED = [
    ("vae", {"default": "torch.bfloat16", "vae": "builtins.eval"}),
    ("default", {"default": "torch.hub.load"}),
    ("vae", {"default": "torch.bfloat16", "vae": "os.system"}),
    ("vae", {"default": "torch.bfloat16", "vae": "torch.hub"}),
    ("vae", {"default": "torch.bfloat16", "vae": "{builtins.eval}"}),
    ("vae", {"default": "torch.bfloat16", "vae": {"default": "builtins.eval"}}),
]


def _pipeline_definition(torch_dtype):
    return {
        "id": "probe",
        "steps": [
            {
                "name": "s",
                "pipeline": {
                    "configuration": {"component_type": "StableDiffusionPipeline"},
                    "from_pretrained_arguments": {
                        "model_name": "org/model",
                        "torch_dtype": torch_dtype,
                    },
                    "arguments": {},
                },
            }
        ],
    }


@pytest.mark.parametrize("component, torch_dtype", _GATED)
def test_validation_refuses_each_gated_component_value(
    untrusted, component, torch_dtype
):
    errors = component_type_errors(_pipeline_definition(torch_dtype))
    assert [e["path"] for e in errors] == [
        f"steps[0].pipeline.from_pretrained_arguments.torch_dtype.{component}"
    ]


@pytest.mark.parametrize(
    "value", ["builtins.eval", "os.system", "torch.hub.load", "torch.hub"]
)
def test_run_time_realize_refuses_a_gated_component_value(untrusted, value):
    with pytest.raises(UntrustedWorkflowError):
        realize_per_component_dtype({"default": "torch.bfloat16", "vae": value})


def test_a_good_dict_still_validates_untrusted(untrusted):
    definition = _pipeline_definition(
        {"default": "torch.bfloat16", "vae": "torch.float32"}
    )
    assert component_type_errors(definition) == []


# --- GHSA-ch56: a task's weight `filename` --------------------------------

_BAD_FILENAMES = [
    "../../../../../../etc/hostname",
    "/etc/hostname",
    "weights/../../x.pth",
    "~/x.pth",
    "C:\\x.pth",
]

_COMMANDS = {
    "upscale": {"image": "asset:a.png", "model_name": "Kim2091/UltraSharp"},
    "restore_faces": {"image": "asset:a.png", "model_name": "org/gfpgan"},
    "interpolate_frames": {"video": "asset:a.mp4"},
}


def _task_definition(command, filename):
    return {
        "steps": [
            {
                "name": "s",
                "task": {
                    "command": command,
                    "arguments": {**_COMMANDS[command], "filename": filename},
                },
            }
        ]
    }


@pytest.mark.parametrize("command", sorted(_COMMANDS))
@pytest.mark.parametrize("filename", _BAD_FILENAMES)
def test_validation_refuses_a_filename_outside_the_repo(command, filename):
    errors = location_errors(_task_definition(command, filename))
    assert "steps[0].task.arguments.filename" in [e["path"] for e in errors]


@pytest.mark.parametrize("command", sorted(_COMMANDS))
@pytest.mark.parametrize("filename", ["4x-UltraSharp.pth", "weights/model.pth"])
def test_validation_passes_a_plain_relative_filename(command, filename):
    errors = location_errors(_task_definition(command, filename))
    assert "steps[0].task.arguments.filename" not in [e["path"] for e in errors]


def test_a_deferred_filename_is_left_to_the_run():
    errors = location_errors(
        _task_definition("upscale", "previous_result:weights.filename")
    )
    assert "steps[0].task.arguments.filename" not in [e["path"] for e in errors]


@pytest.mark.parametrize("filename", _BAD_FILENAMES)
def test_upscale_refuses_the_filename_before_downloading(monkeypatch, filename):
    import huggingface_hub

    from dw.tasks.upscale import resolve_model_path

    def no_download(**kwargs):
        raise AssertionError("hf_hub_download reached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_download)
    with pytest.raises(SecurityError):
        resolve_model_path("Kim2091/UltraSharp", filename)


@pytest.mark.parametrize("filename", _BAD_FILENAMES)
def test_rife_refuses_the_filename_before_downloading(monkeypatch, filename):
    import huggingface_hub

    from dw.tasks.interpolate_frames import _load_rife_model

    def no_download(**kwargs):
        raise AssertionError("hf_hub_download reached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_download)
    with pytest.raises(SecurityError):
        _load_rife_model("cpu", "org/rife", filename)


@pytest.mark.parametrize("filename", _BAD_FILENAMES)
def test_the_plan_lists_no_download_for_a_bad_filename(filename):
    from dw.tasks.interpolate_frames import required_weights

    assert required_weights({"model_name": "org/rife", "filename": filename}) == []
