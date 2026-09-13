"""
Unit tests for dw/locations.py - the one policy for a location a workflow's
arguments supply.

The regression suite's SE-F014/F015/F018/F009/F019 all reduce to the same
hole: a workflow file is untrusted input, and every loader used to trust the
location it named. These tests are written from the untrusted side, so each
one overrides tests/conftest.py's autouse _trust_workflows_by_default fixture
back to the posture a deployed server actually runs on.
"""

import os
import tempfile
from unittest.mock import patch

import pytest
from PIL import Image

from dw.arguments import fetch_image
from dw.locations import (
    contained_matches,
    location_errors,
    token_host_allowed,
    validate_media_glob,
    validate_media_path,
    validate_media_url,
    validate_model_name,
    validate_remote_encoder_url,
)
from dw.security import (
    InvalidInputError,
    PathTraversalError,
    TRUST_WORKFLOWS_ENV_VAR,
)
from dw.tasks.gather import gather_images


@pytest.fixture
def untrusted(monkeypatch):
    """The posture a server runs on: workflow files are untrusted input."""
    monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")


@pytest.fixture
def trusted(monkeypatch):
    monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "1")


@pytest.fixture
def workflow_dir():
    """A directory standing in for the one a workflow file sits in - the
    first of the roots a location may point inside."""
    with tempfile.TemporaryDirectory() as directory:
        yield directory


def _image(path):
    Image.new("RGB", (4, 4), "red").save(path)
    return path


class TestMediaPathContainment:
    """SE-F014: a media argument may not name a file outside every root."""

    def test_absolute_path_outside_every_root_is_refused(self, untrusted, workflow_dir):
        with tempfile.TemporaryDirectory() as elsewhere:
            outside = _image(os.path.join(elsewhere, "secret.png"))
            with pytest.raises(PathTraversalError) as refusal:
                validate_media_path(outside, workflow_dir, "an image argument")
        assert "outside every directory this workflow may read" in str(refusal.value)

    def test_refusal_does_not_depend_on_the_file_existing(
        self, untrusted, workflow_dir
    ):
        """The oracle SE-F014 names: an out-of-root path that exists and one
        that does not must be refused the same way, or the refusal itself
        discriminates what is on the filesystem."""
        with tempfile.TemporaryDirectory() as elsewhere:
            present = _image(os.path.join(elsewhere, "present.png"))
            absent = os.path.join(elsewhere, "absent.png")

            with pytest.raises(PathTraversalError) as for_present:
                validate_media_path(present, workflow_dir, "an image argument")
            with pytest.raises(PathTraversalError) as for_absent:
                validate_media_path(absent, workflow_dir, "an image argument")

        assert str(for_present.value).replace(present, "X") == str(
            for_absent.value
        ).replace(absent, "X")

    def test_a_path_inside_the_workflow_directory_is_allowed(
        self, untrusted, workflow_dir
    ):
        inside = _image(os.path.join(workflow_dir, "subject.png"))
        assert validate_media_path(inside, workflow_dir) == os.path.realpath(inside)

    def test_a_relative_path_still_resolves_against_the_workflow(
        self, untrusted, workflow_dir
    ):
        _image(os.path.join(workflow_dir, "subject.png"))
        assert validate_media_path("subject.png", workflow_dir) == os.path.realpath(
            os.path.join(workflow_dir, "subject.png")
        )

    def test_trust_lifts_containment(self, trusted, workflow_dir):
        with tempfile.TemporaryDirectory() as elsewhere:
            outside = _image(os.path.join(elsewhere, "scratch.png"))
            assert validate_media_path(outside, workflow_dir) == os.path.realpath(
                outside
            )

    def test_fetch_image_refuses_an_out_of_root_absolute_path(
        self, untrusted, workflow_dir
    ):
        """The loader, not just the helper: SE-F014's probe (a) went through
        fetch_image and came back with a decoded 48x48 image."""
        with tempfile.TemporaryDirectory() as elsewhere:
            outside = _image(os.path.join(elsewhere, "debian-logo.png"))
            with pytest.raises(PathTraversalError):
                fetch_image(outside, workflow_dir)


class TestHostPolicy:
    """SE-F018 / SE-F009: a workflow may not send the server at its own
    network."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8765/api/server",
            "http://localhost:8765/api/server",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/image.png",
            "http://192.168.1.4/image.png",
        ],
    )
    def test_internal_addresses_are_refused(self, untrusted, url):
        with pytest.raises(InvalidInputError) as refusal:
            validate_media_url(url, "an image argument")
        assert "inside this deployment" in str(refusal.value)

    def test_a_hostname_resolving_to_loopback_is_refused(self, untrusted):
        """Resolved, not string-matched: a name that answers 127.0.0.1 is
        the same request as naming 127.0.0.1."""
        with patch(
            "dw.locations.socket.getaddrinfo",
            return_value=[(None, None, None, "", ("127.0.0.1", 80))],
        ):
            with pytest.raises(InvalidInputError):
                validate_media_url("http://sneaky.example.com/x.png", "an image")

    def test_a_public_host_is_allowed(self, untrusted):
        with patch(
            "dw.locations.socket.getaddrinfo",
            return_value=[(None, None, None, "", ("93.184.216.34", 80))],
        ):
            assert (
                validate_media_url("https://example.com/x.png", "an image")
                == "https://example.com/x.png"
            )

    def test_a_host_that_does_not_resolve_is_left_to_the_fetch(self, untrusted):
        """A typo is the loader's error to report, not a security refusal."""
        import socket

        with patch("dw.locations.socket.getaddrinfo", side_effect=socket.gaierror):
            assert validate_media_url("https://nope.invalid/x.png", "an image")

    def test_trust_lifts_the_host_policy(self, trusted):
        assert validate_media_url("http://127.0.0.1:8765/x.png", "an image")


class TestGlobContainment:
    """SE-F015: gather_images' glob enumerated and republished any readable
    directory."""

    def test_an_absolute_glob_outside_every_root_is_refused(
        self, untrusted, workflow_dir
    ):
        with tempfile.TemporaryDirectory() as elsewhere:
            _image(os.path.join(elsewhere, "debian-logo.png"))
            with pytest.raises(PathTraversalError) as refusal:
                validate_media_glob(
                    os.path.join(elsewhere, "*.png"), workflow_dir, "the images glob"
                )
        assert "outside every directory this workflow may read" in str(refusal.value)

    def test_gather_images_refuses_the_uncontained_glob(self, untrusted):
        with tempfile.TemporaryDirectory() as elsewhere:
            _image(os.path.join(elsewhere, "debian-logo.png"))
            with pytest.raises(PathTraversalError):
                gather_images(glob=os.path.join(elsewhere, "*.png"))

    def test_a_glob_inside_a_root_is_allowed(self, untrusted, workflow_dir):
        _image(os.path.join(workflow_dir, "a.png"))
        pattern = validate_media_glob(
            os.path.join(workflow_dir, "*.png"), workflow_dir, "the images glob"
        )
        assert pattern.endswith("*.png")

    def test_a_match_escaping_through_a_symlink_is_dropped(
        self, untrusted, workflow_dir
    ):
        """The pattern can be contained and still reach out: containment is
        re-checked on each match's real path."""
        with tempfile.TemporaryDirectory() as elsewhere:
            outside = _image(os.path.join(elsewhere, "outside.png"))
            inside = _image(os.path.join(workflow_dir, "inside.png"))
            link = os.path.join(workflow_dir, "linked.png")
            os.symlink(outside, link)
            kept = contained_matches(
                sorted([inside, link]), workflow_dir, "the images glob"
            )
        assert kept == [inside]

    def test_trust_lifts_glob_containment(self, trusted, workflow_dir):
        with tempfile.TemporaryDirectory() as elsewhere:
            pattern = os.path.join(elsewhere, "*.png")
            assert validate_media_glob(pattern, workflow_dir) == pattern


class TestRemoteEncoderUrl:
    """SE-F009: the field that POSTs this machine's HuggingFace token."""

    def test_a_non_https_scheme_is_refused(self, untrusted):
        with pytest.raises(InvalidInputError) as refusal:
            validate_remote_encoder_url("file:///etc/hostname")
        assert "may only reach an https endpoint" in str(refusal.value)

    def test_loopback_is_refused(self, untrusted):
        with pytest.raises(InvalidInputError):
            validate_remote_encoder_url("http://127.0.0.1:8765/api/server")

    def test_https_loopback_is_refused_too(self, untrusted):
        with pytest.raises(InvalidInputError) as refusal:
            validate_remote_encoder_url("https://127.0.0.1:8765/encode")
        assert "inside this deployment" in str(refusal.value)

    def test_the_token_only_goes_to_huggingface_hosts(self):
        assert token_host_allowed("api-inference.huggingface.co")
        assert token_host_allowed("abc.endpoints.huggingface.cloud")
        assert not token_host_allowed("evil.example.com")
        assert not token_host_allowed("huggingface.co.evil.example.com")

    def test_an_untrusted_run_withholds_the_token_from_a_third_party(self, untrusted):
        """Reachable, but without the credential - the exfiltration half of
        SE-F009 is closed even for a host the host policy allows."""
        from dw.pipeline_processors import remote

        sent = {}

        class _Response:
            ok = True
            headers = {"Content-Type": "application/octet-stream"}
            content = b""
            status_code = 200

        def _post(url, json=None, headers=None):
            sent["headers"] = headers
            return _Response()

        with (
            patch.object(remote.requests, "post", _post),
            patch.object(remote.torch, "load", return_value=_Embeds()),
            patch(
                "dw.locations.socket.getaddrinfo",
                return_value=[(None, None, None, "", ("93.184.216.34", 443))],
            ),
        ):
            remote.remote_text_encoder(["a"], "https://evil.example.com/encode", "cpu")

        assert "Authorization" not in sent["headers"]


class _Embeds:
    def to(self, device):
        return self


class TestModelName:
    """SE-F019: download_model checked its repo_id; model_name did not."""

    def test_a_repo_id_is_allowed(self, untrusted):
        assert validate_model_name("stabilityai/sd-turbo") == "stabilityai/sd-turbo"

    def test_an_absolute_path_outside_every_root_is_refused(
        self, untrusted, workflow_dir
    ):
        with pytest.raises(PathTraversalError):
            validate_model_name("/etc/passwd", workflow_dir)

    def test_a_traversal_shaped_name_is_refused(self, untrusted, workflow_dir):
        with pytest.raises(PathTraversalError):
            validate_model_name("org/name/../../x", workflow_dir)

    def test_a_local_model_directory_inside_a_root_is_allowed(
        self, untrusted, workflow_dir
    ):
        local = os.path.join(workflow_dir, "my-model")
        os.makedirs(local)
        assert validate_model_name(local, workflow_dir)

    @pytest.mark.parametrize(
        "name",
        ["http://127.0.0.1:8765/", "https://evil.example.com/model", "file:///etc"],
    )
    def test_a_url_shaped_name_is_refused(self, untrusted, workflow_dir, name):
        """#117: joined onto the workflow directory a URL resolved inside a
        root, so the one shape download_model refuses that reached
        model_name went on validating clean."""
        with pytest.raises(InvalidInputError):
            validate_model_name(name, workflow_dir)


class TestValidationTimeErrors:
    """Refused by validate_workflow rather than after a pipeline load."""

    def test_an_out_of_root_image_is_an_error_with_its_path(
        self, untrusted, workflow_dir
    ):
        definition = {
            "steps": [
                {
                    "name": "edit",
                    "pipeline": {"arguments": {"image": "/usr/share/pixmaps/x.png"}},
                }
            ]
        }
        errors = location_errors(definition, base_dir=workflow_dir)
        assert [error["path"] for error in errors] == [
            "steps[0].pipeline.arguments.image"
        ]

    def test_a_loopback_url_is_an_error(self, untrusted, workflow_dir):
        definition = {
            "steps": [
                {
                    "name": "edit",
                    "pipeline": {
                        "arguments": {"image": "http://127.0.0.1:8765/api/server"}
                    },
                }
            ]
        }
        assert location_errors(definition, base_dir=workflow_dir)

    def test_an_uncontained_glob_is_an_error(self, untrusted, workflow_dir):
        definition = {
            "steps": [
                {
                    "name": "gather",
                    "task": {
                        "command": "gather_images",
                        "arguments": {"glob": "/usr/share/pixmaps/*.png"},
                    },
                }
            ]
        }
        assert location_errors(definition, base_dir=workflow_dir)

    def test_a_path_shaped_model_name_is_an_error(self, untrusted, workflow_dir):
        definition = {
            "steps": [
                {
                    "name": "draw",
                    "pipeline": {
                        "from_pretrained_arguments": {"model_name": "/etc/passwd"}
                    },
                }
            ]
        }
        assert [
            error["path"]
            for error in location_errors(definition, base_dir=workflow_dir)
        ] == ["steps[0].pipeline.from_pretrained_arguments.model_name"]

    def test_a_remote_encoder_url_is_an_error(self, untrusted, workflow_dir):
        definition = {
            "steps": [
                {
                    "name": "draw",
                    "pipeline": {
                        "remote_text_encoder": {"url": "file:///etc/hostname"}
                    },
                }
            ]
        }
        assert [
            error["path"]
            for error in location_errors(definition, base_dir=workflow_dir)
        ] == ["steps[0].pipeline.remote_text_encoder.url"]

    def test_references_and_relative_paths_are_left_alone(
        self, untrusted, workflow_dir
    ):
        definition = {
            "steps": [
                {
                    "name": "edit",
                    "pipeline": {
                        "arguments": {
                            "image": "asset:iris.png",
                            "mask_image": "variable:mask",
                            "control_image": "previous_result:draw",
                        }
                    },
                }
            ]
        }
        assert location_errors(definition, base_dir=workflow_dir) == []

    @pytest.mark.parametrize(
        "location",
        [
            "../../../../../usr/share/pixmaps/debian-logo.png",
            "../../../../etc/hostname",
        ],
    )
    def test_a_relative_traversal_is_an_error_too(
        self, untrusted, workflow_dir, location
    ):
        """#124: the absolute spelling was refused here and the relative one
        only when the loader reached it - three seconds into a queued job,
        after validate_workflow had said to go ahead."""
        definition = {
            "steps": [
                {
                    "name": "probe",
                    "task": {
                        "command": "get_image_size",
                        "arguments": {"image": location},
                    },
                }
            ]
        }

        errors = location_errors(definition, base_dir=workflow_dir)

        assert [error["path"] for error in errors] == ["steps[0].task.arguments.image"]
        assert "'..'" in errors[0]["message"]

    def test_a_url_shaped_model_name_is_an_error(self, untrusted, workflow_dir):
        """#117 at validation time, where the tester found it."""
        definition = {
            "steps": [
                {
                    "name": "load",
                    "pipeline": {
                        "from_pretrained_arguments": {
                            "model_name": "http://127.0.0.1:8765/"
                        }
                    },
                }
            ]
        }

        assert [
            error["path"]
            for error in location_errors(definition, base_dir=workflow_dir)
        ] == ["steps[0].pipeline.from_pretrained_arguments.model_name"]

    def test_the_error_carries_the_authored_step_index(self, untrusted, workflow_dir):
        """A for_each member reports against the step the author wrote."""
        definition = {
            "steps": [
                {"name": "first", "task": {"command": "gather_inputs"}},
                {
                    "name": "shot@a",
                    "pipeline": {"arguments": {"image": "/etc/hosts"}},
                },
            ]
        }
        errors = location_errors(
            definition, source_indices=[0, 0], base_dir=workflow_dir
        )
        assert errors[0]["path"].startswith("steps[0]")
