"""
Unit tests for dw/locations.py - the one policy for a location a workflow's
arguments supply.

The regression suite's SE-F014/F015/F018/F009/F019 all reduce to the same
hole: a workflow file is untrusted input, and every loader used to trust the
location it named. These tests are written from the untrusted side, so each
one overrides tests/conftest.py's autouse _trust_workflows_by_default fixture
back to the posture a deployed server actually runs on.
"""

import http.server
import os
import socketserver
import tempfile
import threading
from unittest.mock import Mock, patch

import pytest
import requests
from PIL import Image

from dw.argument_media import fetch_image
from dw.locations import (
    _refuse_other_url,
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
    validate_url,
)
from dw.trust import TRUST_WORKFLOWS_ENV_VAR
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

    @pytest.mark.parametrize(
        "location",
        ["file:///usr/share/sounds/x.wav", "file:///nonexistent/x.png", "s3://b/x.png"],
    )
    def test_a_url_that_is_not_http_is_refused(self, location, workflow_dir):
        """#618: joined onto the workflow directory as a relative path, a
        file:// URL landed inside a root and passed containment. Refused
        whatever the trust posture - no loader opens one as a path."""
        with pytest.raises(InvalidInputError, match="only http\\(s\\) URLs"):
            validate_media_path(location, workflow_dir, "an audio argument")

    def test_an_upper_case_https_url_is_not_refused_as_another_scheme(self):
        """The scheme decision is validate_url's: urlparse lowercases it, so
        'HTTPS://' is an http(s) URL there and must not be refused here as an
        'HTTPS' URL (#618 arch review)."""
        assert validate_url("HTTPS://example.com/x.wav")
        _refuse_other_url("HTTPS://example.com/x.wav", "an audio argument")

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
            is_redirect = False
            headers = {"Content-Type": "application/octet-stream"}
            status_code = 200
            _content = b""

            @property
            def content(self):
                return self._content

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                return iter([b""])

            def close(self):
                pass

        class _Session:
            def request(self, method, url, **kwargs):
                sent["method"] = method
                sent["headers"] = kwargs.get("headers")
                sent["timeout"] = kwargs.get("timeout")
                return _Response()

            def close(self):
                pass

        with (
            patch("dw.locations._pinned_session", lambda url, address: _Session()),
            patch.object(remote.torch, "load", return_value=_Embeds()),
            patch(
                "dw.locations.socket.getaddrinfo",
                return_value=[(None, None, None, "", ("93.184.216.34", 443))],
            ),
        ):
            remote.remote_text_encoder(["a"], "https://evil.example.com/encode", "cpu")

        assert "Authorization" not in sent["headers"]
        assert sent["method"] == "POST"
        assert sent["timeout"] == remote.REMOTE_ENCODER_TIMEOUT

    def test_a_backslash_is_refused(self, untrusted):
        """#409: urllib.parse and urllib3 disagree on the host of a URL
        holding '\\', so the checked host need not be the dialed one."""
        with pytest.raises(InvalidInputError, match="backslash"):
            validate_remote_encoder_url("https://evil.example\\@huggingface.co/encode")


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
        with pytest.raises(PathTraversalError, match="Repo id must be in the form"):
            validate_model_name("/etc/passwd", workflow_dir)

    def test_a_traversal_shaped_name_is_refused(self, untrusted, workflow_dir):
        with pytest.raises(PathTraversalError, match="Repo id must be in the form"):
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

    @pytest.mark.parametrize("key", ["image", "hold_audio"])
    def test_a_file_url_is_an_error_with_its_path(self, untrusted, workflow_dir, key):
        definition = {
            "steps": [
                {
                    "name": "edit",
                    "pipeline": {"arguments": {key: "file:///usr/share/sounds/x.wav"}},
                }
            ]
        }
        errors = location_errors(definition, base_dir=workflow_dir)
        assert [error["path"] for error in errors] == [
            f"steps[0].pipeline.arguments.{key}"
        ]
        assert "'file' URL" in errors[0]["message"]

    def test_an_upper_case_https_url_is_checked_as_a_url(self, untrusted, workflow_dir):
        """Not refused for its scheme, and still sent through the host policy."""
        definition = {
            "steps": [
                {
                    "name": "edit",
                    "pipeline": {
                        "arguments": {"hold_audio": "HTTPS://example.com/x.wav"}
                    },
                }
            ]
        }
        with patch(
            "dw.locations.socket.getaddrinfo",
            return_value=[(None, None, None, "", ("93.184.216.34", 443))],
        ):
            assert location_errors(definition, base_dir=workflow_dir) == []
        with patch(
            "dw.locations.socket.getaddrinfo",
            return_value=[(None, None, None, "", ("127.0.0.1", 443))],
        ):
            errors = location_errors(definition, base_dir=workflow_dir)
        assert "inside this deployment" in errors[0]["message"]

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

    def test_a_backslash_url_is_an_error(self, untrusted, workflow_dir):
        definition = {
            "steps": [
                {
                    "name": "edit",
                    "pipeline": {
                        "arguments": {
                            "image": "http://169.254.169.254\\@example.com/a.png"
                        }
                    },
                }
            ]
        }
        errors = location_errors(definition, base_dir=workflow_dir)
        assert [error["path"] for error in errors] == [
            "steps[0].pipeline.arguments.image"
        ]
        assert "backslash" in errors[0]["message"]

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


class TestTaskMediaArguments:
    """A task argument that reads a file under a generic name - join_windows'
    'source' - is refused at validation like a media key (#630, SE-F042),
    rather than only when the run reaches the join."""

    @staticmethod
    def _join(source):
        return {
            "steps": [
                {"name": "first", "task": {"command": "gather_inputs"}},
                {
                    "name": "join",
                    "task": {
                        "command": "join_windows",
                        "arguments": {
                            "videos": ["previous_result:first"],
                            "source": source,
                            "num_frames": 17,
                            "overlap": 4,
                        },
                    },
                },
            ]
        }

    @pytest.mark.parametrize(
        "source",
        [
            "/etc/passwd",
            "/nonexistent-dw-probe/x.mp4",
            "file:///etc/passwd",
            "../../../../etc/passwd",
        ],
    )
    def test_an_unreadable_source_is_an_error_at_its_path(
        self, untrusted, workflow_dir, source
    ):
        errors = location_errors(self._join(source), base_dir=workflow_dir)

        assert [error["path"] for error in errors] == ["steps[1].task.arguments.source"]
        assert "'source'" in errors[0]["message"]

    @pytest.mark.parametrize(
        "source", ["asset:long.mp4", "variable:source_video", "previous_result:x"]
    )
    def test_a_reference_is_left_alone(self, untrusted, workflow_dir, source):
        assert location_errors(self._join(source), base_dir=workflow_dir) == []

    def test_a_path_inside_the_workflow_directory_is_allowed(
        self, untrusted, workflow_dir
    ):
        inside = os.path.join(workflow_dir, "long.mp4")
        assert location_errors(self._join(inside), base_dir=workflow_dir) == []

    def test_another_command_s_source_is_not_a_location(self, untrusted, workflow_dir):
        definition = self._join("/etc/passwd")
        definition["steps"][1]["task"]["command"] = "gather_inputs"
        assert location_errors(definition, base_dir=workflow_dir) == []


class TestApplyLutArguments:
    """apply_lut's `lut` and `media` are refused at validation, at their own
    argument paths, rather than only when the run reaches the step (#635,
    SE-F044) - and `lut`, which is never fetched, refuses a URL as a URL."""

    @staticmethod
    def _apply(lut, media="asset:portrait.jpg"):
        return {
            "steps": [
                {
                    "name": "lut",
                    "task": {
                        "command": "apply_lut",
                        "arguments": {"media": media, "lut": lut},
                    },
                }
            ]
        }

    @pytest.mark.parametrize(
        "lut",
        [
            "/etc/passwd",
            "/etc/passwd.cube",
            "/nonexistent-dw-probe/x.cube",
            "../../../../etc/passwd",
            "../../../../tmp/x.cube",
            "file:///etc/passwd.cube",
            "https://example.com/x.cube",
        ],
    )
    def test_an_unreadable_lut_is_an_error_at_its_path(
        self, untrusted, workflow_dir, lut
    ):
        errors = location_errors(self._apply(lut), base_dir=workflow_dir)

        assert [error["path"] for error in errors] == ["steps[0].task.arguments.lut"]
        assert "'lut'" in errors[0]["message"]

    def test_a_url_lut_is_refused_as_a_url(self, untrusted, workflow_dir):
        errors = location_errors(
            self._apply("https://example.com/x.cube"), base_dir=workflow_dir
        )

        assert "never fetched from a URL" in errors[0]["message"]
        assert "outside" not in errors[0]["message"]

    @pytest.mark.parametrize(
        "lut", ["asset:look.cube", "output:look.cube", "variable:look"]
    )
    def test_a_reference_is_left_alone(self, untrusted, workflow_dir, lut):
        assert location_errors(self._apply(lut), base_dir=workflow_dir) == []

    def test_a_lut_inside_the_workflow_directory_is_allowed(
        self, untrusted, workflow_dir
    ):
        inside = os.path.join(workflow_dir, "look.cube")
        assert location_errors(self._apply(inside), base_dir=workflow_dir) == []

    @pytest.mark.parametrize("command", ["apply_lut", "grade", "sharpen", "film_grain"])
    def test_a_literal_media_outside_the_roots_is_an_error(
        self, untrusted, workflow_dir, command
    ):
        definition = self._apply("asset:look.cube", media="/etc/passwd.jpg")
        definition["steps"][0]["task"]["command"] = command

        errors = location_errors(definition, base_dir=workflow_dir)

        assert [error["path"] for error in errors] == ["steps[0].task.arguments.media"]


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    """/ok answers 2 bytes; /host echoes the Host header; /big answers 1000
    bytes; /nolength streams 1000 bytes with no Content-Length; /hop
    redirects to /ok; /loop redirects to itself; /internal redirects to
    127.0.0.1; /auth echoes the Authorization header; /elsewhere 307s to
    /auth under another name for the same server."""

    def do_GET(self):
        self._answer()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._answer()

    def _answer(self):
        path = self.path
        if path.startswith("/hop"):
            return self._redirect(f"http://{self.headers['Host']}/ok")
        if path.startswith("/loop"):
            return self._redirect(f"http://{self.headers['Host']}/loop")
        if path.startswith("/internal"):
            return self._redirect("http://127.0.0.1:1/x")
        if path.startswith("/elsewhere"):
            port = self.headers["Host"].rsplit(":", 1)[1]
            return self._redirect(f"http://localhost:{port}/auth", status=307)
        if path.startswith("/host"):
            body = self.headers["Host"].encode()
        elif path.startswith("/auth"):
            body = (self.headers.get("Authorization") or "none").encode()
        elif path.startswith("/big") or path.startswith("/nolength"):
            body = b"x" * 1000
        else:
            body = b"ok"
        self.send_response(200)
        if not path.startswith("/nolength"):
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, target, status=302):
        self.send_response(status)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = socketserver.TCPServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _public(monkeypatch):
    """Every name resolves to a public address for the policy; the pin
    then decides what is actually dialed. An IP literal goes to the real
    resolver, since urllib3 calls the same getaddrinfo to dial the pin."""
    import ipaddress
    import socket

    real = socket.getaddrinfo

    def _resolve(host, *a, **k):
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return [(None, None, None, "", ("93.184.216.34", 80))]
        return real(host, *a, **k)

    monkeypatch.setattr("dw.locations.socket.getaddrinfo", _resolve)


class TestSafeRequest:
    """Review 2026-10-07 #3/#4: one path for every outbound request - the
    address dialed is the one the policy checked, the body is capped, and
    POST goes the same way as GET."""

    def test_the_pin_dials_the_validated_address_not_a_second_lookup(
        self, trusted, local_server
    ):
        from dw.locations import _pinned_session

        host, port = local_server.split(":")
        url = f"http://pinned.example:{port}/host"
        # pinned.example resolves nowhere; only the pin can reach the server
        response = _pinned_session(url, host).get(url, timeout=5, allow_redirects=False)
        assert response.content == f"pinned.example:{port}".encode()

    def test_trust_lifts_pinning_but_keeps_the_cap(self, trusted, local_server):
        from dw.locations import safe_get

        assert safe_get(f"http://{local_server}/ok", timeout=5).content == b"ok"
        with pytest.raises(InvalidInputError, match="larger than"):
            safe_get(f"http://{local_server}/big", timeout=5, max_bytes=100)
        with pytest.raises(InvalidInputError, match="larger than"):
            safe_get(f"http://{local_server}/nolength", timeout=5, max_bytes=100)

    def test_a_body_exactly_at_the_cap_is_kept(self, trusted, local_server):
        from dw.locations import safe_get

        response = safe_get(
            f"http://{local_server}/nolength", timeout=5, max_bytes=1000
        )
        assert len(response.content) == 1000
        with pytest.raises(InvalidInputError, match="larger than"):
            safe_get(f"http://{local_server}/nolength", timeout=5, max_bytes=999)

    def test_a_declared_length_over_the_cap_is_refused_before_the_body(
        self, trusted, local_server, monkeypatch
    ):
        from dw import locations

        read = []
        monkeypatch.setattr(
            locations, "_read_capped", lambda *a, **k: read.append(a) or b""
        )
        with pytest.raises(InvalidInputError, match="larger than"):
            locations.safe_get(f"http://{local_server}/big", timeout=5, max_bytes=100)
        assert read == []

    def test_a_redirect_to_an_internal_address_is_refused(
        self, untrusted, local_server, monkeypatch
    ):
        """The first hop is a public name (pinned here to the local server
        so the test can answer it); its 302 names 127.0.0.1, which the
        policy refuses before anything is dialed."""
        from dw import locations

        _public(monkeypatch)
        host, port = local_server.split(":")
        real_pinned = locations._pinned_session
        monkeypatch.setattr(
            locations, "_pinned_session", lambda url, address: real_pinned(url, host)
        )
        with pytest.raises(InvalidInputError, match="inside this deployment"):
            locations.safe_get(f"http://public.example:{port}/internal", timeout=5)

    def test_a_redirect_loop_is_refused(self, trusted, local_server):
        from dw.locations import MAX_MEDIA_REDIRECTS, safe_get

        with pytest.raises(InvalidInputError, match=f"{MAX_MEDIA_REDIRECTS} times"):
            safe_get(f"http://{local_server}/loop", timeout=5)

    def test_a_redirect_is_followed_to_a_good_target(self, trusted, local_server):
        from dw.locations import safe_get

        assert safe_get(f"http://{local_server}/hop", timeout=5).content == b"ok"

    def test_post_goes_through_the_same_path(self, trusted, local_server):
        from dw.locations import safe_post

        response = safe_post(
            f"http://{local_server}/ok", "a test endpoint", timeout=5, json={"a": 1}
        )
        assert response.content == b"ok"

    def test_a_redirect_to_another_host_drops_the_credential(
        self, trusted, local_server
    ):
        """requests.post dropped Authorization when a redirect changed host;
        hops followed by hand must too, or the token sent to one endpoint
        goes wherever that endpoint redirects."""
        from dw.locations import safe_post

        response = safe_post(
            f"http://{local_server}/elsewhere",
            "a test endpoint",
            timeout=5,
            json={"a": 1},
            headers={"Authorization": "Bearer secret"},
        )
        assert response.content == b"none"

    def test_the_credential_reaches_the_host_it_was_meant_for(
        self, trusted, local_server
    ):
        from dw.locations import safe_post

        response = safe_post(
            f"http://{local_server}/auth",
            "a test endpoint",
            timeout=5,
            headers={"Authorization": "Bearer secret"},
        )
        assert response.content == b"Bearer secret"

    def test_an_unresolvable_host_is_fetched_unpinned(self, untrusted, monkeypatch):
        """A typo stays the fetch's error to report (TestHostPolicy), and
        the pin must not crash on an empty resolution."""
        import socket

        from dw import locations

        monkeypatch.setattr(
            "dw.locations.socket.getaddrinfo", Mock(side_effect=socket.gaierror)
        )
        dialed = {}

        class _Session:
            def request(self, method, url, **kwargs):
                dialed["url"] = url
                raise requests.ConnectionError("no such host")

            def close(self):
                pass

        def _never_pinned(url, address):
            raise AssertionError("an unresolved name must not be pinned")

        monkeypatch.setattr(locations, "_pinned_session", _never_pinned)
        monkeypatch.setattr(locations, "_plain_session", lambda: _Session())
        with pytest.raises(requests.ConnectionError):
            locations.safe_get("https://nope.invalid/x.png", timeout=5)
        assert dialed["url"] == "https://nope.invalid/x.png"
