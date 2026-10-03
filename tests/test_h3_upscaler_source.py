"""upscale_h3_latents names its weights by a Hub repo and a file inside it,
and nothing else (#499, SE-F039): the file name is joined onto the local cache
by hf_hub_download, so a name shaped like a path is a path on the server, and
the weights are read with safetensors, so a pickle format is refused by name.
Both are checked at validation (dw/locations.py) and again at run time, before
anything downloads, for a value that arrives through a variable."""

from unittest import mock

import pytest
import torch

from dw.locations import location_errors, validate_weight_name
from dw.security import InvalidInputError, PathTraversalError
from dw.trust import TRUST_WORKFLOWS_ENV_VAR
from dw.tasks import h3_latent_upscale
from dw.tasks.h3_latent_upscale import (
    DEFAULT_UPSCALER_REPO,
    DEFAULT_UPSCALER_WEIGHTS,
    upscale_h3_latents,
)

TRAVERSALS = [
    "../../../../etc/passwd.safetensors",
    "/etc/passwd.safetensors",
    "sub/../../x.safetensors",
    "sub/./x.safetensors",
    "sub//x.safetensors",
    "..\\x.safetensors",
    "C:/x.safetensors",
    "~/x.safetensors",
]
PICKLES = ["model.pth", "model.bin", "model.pt", "model.ckpt"]
# SE-F039 (b): this task's weight_name is a file name, never a path - the
# default repo's subfolder is the task's to know, not the caller's to name
SUBFOLDERED = [
    "sub/x.safetensors",
    "minimax_h3_latent_upscaler_3d_conv_v1/"
    "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
]
# SE-F039 (c): model_name is a Hub repo id, never a path or URL
NOT_REPO_IDS = ["/etc", "../x", "a/b/c", "https://x.test/r", "./models/up"]


def step(weight_name, command="upscale_h3_latents"):
    return {
        "name": "up",
        "task": {
            "command": command,
            "arguments": {
                "latents": "previous_result:base.latents",
                "weight_name": weight_name,
            },
        },
    }


class TestTheShape:
    def test_the_default_passes(self):
        assert validate_weight_name(
            DEFAULT_UPSCALER_WEIGHTS, (".safetensors",), subfolders=False
        )

    def test_a_subfolder_is_allowed_where_subfolders_are(self):
        assert validate_weight_name("sub/x.bin") == "sub/x.bin"

    @pytest.mark.parametrize("name", TRAVERSALS)
    def test_a_name_leaving_the_repo_is_refused(self, name):
        with pytest.raises((PathTraversalError, InvalidInputError)):
            validate_weight_name(name, (".safetensors",))

    @pytest.mark.parametrize("name", PICKLES)
    def test_a_pickle_format_is_refused_where_only_safetensors_is_read(self, name):
        with pytest.raises(InvalidInputError, match="safetensors"):
            validate_weight_name(name, (".safetensors",))

    @pytest.mark.parametrize("name", PICKLES)
    def test_another_suffix_is_allowed_where_none_is_named(self, name):
        assert validate_weight_name(name) == name


class TestAtValidation:
    def test_the_default_is_clean(self):
        assert location_errors({"steps": [step(DEFAULT_UPSCALER_WEIGHTS)]}) == []

    @pytest.mark.parametrize("name", TRAVERSALS + PICKLES + SUBFOLDERED)
    def test_the_upscaler_step_is_refused_at_its_path(self, name):
        errors = location_errors({"steps": [step(name)]})
        assert [error["path"] for error in errors] == [
            "steps[0].task.arguments.weight_name"
        ]

    def test_the_safetensors_rule_is_the_upscalers_alone(self):
        # An IP adapter's weights are a .bin in the catalog
        assert (
            location_errors({"steps": [step("ip-adapter.bin", command="other")]}) == []
        )

    def test_the_no_slash_rule_is_the_upscalers_alone(self):
        assert location_errors({"steps": [step("sub/x.bin", command="other")]}) == []

    @pytest.mark.parametrize("repo", NOT_REPO_IDS)
    def test_a_model_name_that_is_not_a_repo_id_is_refused(self, repo):
        definition = {"steps": [step(DEFAULT_UPSCALER_WEIGHTS)]}
        definition["steps"][0]["task"]["arguments"]["model_name"] = repo
        errors = location_errors(definition)
        assert [error["path"] for error in errors] == [
            "steps[0].task.arguments.model_name"
        ]

    def test_a_repo_id_model_name_is_clean(self):
        definition = {"steps": [step(DEFAULT_UPSCALER_WEIGHTS)]}
        definition["steps"][0]["task"]["arguments"]["model_name"] = "me/my-upscaler"
        assert location_errors(definition) == []

    def test_a_traversal_is_refused_on_any_step(self):
        assert location_errors({"steps": [step("../x.bin", command="other")]})

    def test_a_deferred_value_is_left_to_run_time(self):
        assert location_errors({"steps": [step("variable:weights")]}) == []


class TestAtRunTime:
    """The value a variable or an earlier step supplies was never in the
    document validation read, so the task checks again before downloading."""

    @pytest.mark.parametrize("name", TRAVERSALS + PICKLES + SUBFOLDERED)
    def test_refused_before_any_download(self, name):
        with (
            mock.patch("huggingface_hub.hf_hub_download") as download,
            mock.patch.object(h3_latent_upscale, "_load_upscaler") as load,
        ):
            with pytest.raises((PathTraversalError, InvalidInputError)):
                upscale_h3_latents(
                    torch.zeros(1, 24, 2, 4, 8), 256, 128, weight_name=name
                )
        download.assert_not_called()
        load.assert_not_called()

    @pytest.mark.parametrize("repo", NOT_REPO_IDS + [""])
    def test_a_model_name_that_is_not_a_repo_id_is_refused(self, repo):
        with mock.patch.object(h3_latent_upscale, "_load_upscaler") as load:
            with pytest.raises(InvalidInputError):
                h3_latent_upscale._check_upscaler_source(
                    repo or "/", DEFAULT_UPSCALER_WEIGHTS
                )
        load.assert_not_called()

    def test_the_defaults_pass(self):
        h3_latent_upscale._check_upscaler_source(
            DEFAULT_UPSCALER_REPO, DEFAULT_UPSCALER_WEIGHTS
        )


def test_a_refused_model_name_names_no_server_directory(tmp_path, monkeypatch):
    """The refusal names what the caller wrote, not the directories the
    server reads from."""
    monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
    workflow_dir = tmp_path / "wf"
    workflow_dir.mkdir()
    definition = {
        "steps": [
            {
                "name": "p",
                "pipeline": {
                    "from_pretrained_arguments": {"model_name": "/etc/secret"}
                },
            }
        ]
    }
    errors = location_errors(definition, base_dir=str(workflow_dir))
    assert errors
    for error in errors:
        assert str(tmp_path) not in error["message"]
