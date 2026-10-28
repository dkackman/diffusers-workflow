"""#978: tasks that fetch one file of a repo declare the file, the plan checks
the cache for that file, and facexlib's own downloads are listed as URLs."""

import copy
import subprocess
import sys
import textwrap

import pytest

import dw.plan_downloads
from dw.plan import build_plan, downloads_required, gate_warnings
from dw.tasks import face_track, h3_latent_upscale, interpolate_frames
from dw.tasks import remove_object, restore_faces, upscale
from dw.workflow import workflow_from_definition

PIN = "a" * 40


def workflow(*task_steps):
    return {
        "id": "file_weights",
        "description": "docs only",
        "summary": "docs only",
        "steps": [
            {
                "name": f"t{i}",
                "task": {"command": command, "arguments": arguments},
                "result": {"content_type": "image/png"},
            }
            for i, (command, arguments) in enumerate(task_steps)
        ],
    }


@pytest.fixture
def hub(tmp_path):
    return tmp_path / "hub"


@pytest.fixture
def plan(tmp_path, hub, monkeypatch):
    """build_plan against an empty fake hub cache, offline, with facexlib's
    weights all present so only the hub entries are in play."""
    monkeypatch.setattr(
        dw.plan_downloads, "scan_models", lambda cache_dir=None: {"repos": []}
    )
    weights = tmp_path / "facexlib" / "weights"
    weights.mkdir(parents=True)
    for url in (
        restore_faces.FACEXLIB_DETECTION_URL,
        restore_faces.FACEXLIB_PARSING_URL,
    ):
        (weights / url.rsplit("/", 1)[1]).write_bytes(b"w")
    monkeypatch.setattr(restore_faces, "package_dir", lambda name: weights.parent)

    def make(spec):
        candidate = workflow_from_definition(
            copy.deepcopy(spec), str(tmp_path), str(tmp_path), str(tmp_path)
        )
        return build_plan(
            candidate, {}, device="cuda", cache_dir=str(hub), lookup_sizes=False
        )["downloads_required"]

    return make


def cache(hub, repo, filename, commit=PIN, ref=None):
    """A file in the fake hub cache at `commit`, with `refs/<ref>` naming it
    only when `ref` is given."""
    root = hub / f"models--{repo.replace('/', '--')}"
    path = root / "snapshots" / commit / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"w")
    if ref:
        (root / "refs").mkdir(exist_ok=True)
        (root / "refs" / ref).write_text(commit)


class TestDeclarations:
    def test_upscale_with_a_filename_declares_the_file(self):
        assert upscale.required_weights(
            {"model_name": "org/up", "filename": "x.pth"}
        ) == [{"repo": "org/up", "filename": "x.pth", "revision": None}]

    def test_upscale_without_a_filename_declares_the_repo(self):
        assert upscale.required_weights({"model_name": "org/up"}) == [
            {"repo": "org/up"}
        ]

    def test_restore_faces_declares_its_spandrel_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(restore_faces, "package_dir", lambda name: tmp_path)
        (tmp_path / "weights").mkdir()
        for url in (
            restore_faces.FACEXLIB_DETECTION_URL,
            restore_faces.FACEXLIB_PARSING_URL,
        ):
            (tmp_path / "weights" / url.rsplit("/", 1)[1]).write_bytes(b"w")
        assert restore_faces.required_weights(
            {"model_name": "org/gfp", "filename": "g.pth"}
        ) == [{"repo": "org/gfp", "filename": "g.pth", "revision": None}]

    def test_remove_object_default_is_pinned(self):
        assert remove_object.required_weights({}) == [
            {
                "repo": remove_object.DEFAULT_REMOVER_REPO,
                "filename": remove_object.DEFAULT_REMOVER_WEIGHTS,
                "revision": remove_object.REMOVER_REVISION,
            }
        ]

    def test_remove_object_override_reads_main(self):
        assert remove_object.required_weights(
            {"model_name": "org/lama", "weight_name": "l.safetensors"}
        ) == [{"repo": "org/lama", "filename": "l.safetensors", "revision": None}]

    def test_h3_upscaler_default_is_its_subfolder_file_at_the_pin(self):
        assert h3_latent_upscale.required_weights({}) == [
            {
                "repo": h3_latent_upscale.DEFAULT_UPSCALER_REPO,
                "filename": f"{h3_latent_upscale.DEFAULT_UPSCALER_SUBFOLDER}/"
                f"{h3_latent_upscale.DEFAULT_UPSCALER_WEIGHTS}",
                "revision": h3_latent_upscale.UPSCALER_REVISION,
            }
        ]

    def test_h3_upscaler_known_source_has_its_folder_and_pin(self):
        repo = h3_latent_upscale.LMS_UPSCALER_REPO
        weight = "h3_upscaler_lms_v0.1.safetensors"
        folder, commit = h3_latent_upscale.KNOWN_UPSCALER_SOURCES[(repo, weight)]
        assert h3_latent_upscale.required_weights(
            {"model_name": repo, "weight_name": weight}
        ) == [{"repo": repo, "filename": f"{folder}/{weight}", "revision": commit}]

    def test_h3_upscaler_unknown_source_reads_its_root_at_main(self):
        assert h3_latent_upscale.required_weights(
            {"model_name": "org/h3up", "weight_name": "u.safetensors"}
        ) == [{"repo": "org/h3up", "filename": "u.safetensors", "revision": None}]

    def test_face_track_default_is_pinned(self):
        assert face_track.required_weights({}) == [
            {
                "repo": face_track.DEFAULT_DETECTOR_REPO,
                "filename": face_track.DEFAULT_DETECTOR_FILE,
                "revision": face_track.DETECTOR_REVISION,
            }
        ]

    def test_face_track_pin_is_for_the_default_repo_alone(self):
        assert face_track.required_weights(
            {"detector_repo": "org/yunet", "detector_file": "y.onnx"}
        ) == [{"repo": "org/yunet", "filename": "y.onnx", "revision": None}]

    def test_interpolate_frames_default_and_override(self):
        assert interpolate_frames.required_weights({}) == [
            {
                "repo": "imaginairy/rife-interpolation",
                "filename": "rife-flownet-4.13.2.safetensors",
                "revision": None,
            }
        ]
        assert interpolate_frames.required_weights(
            {"model_name": "org/rife", "filename": "r.safetensors"}
        ) == [{"repo": "org/rife", "filename": "r.safetensors", "revision": None}]

    @pytest.mark.parametrize(
        "arguments",
        [
            {"model_name": "previous_result:m", "filename": "r.safetensors"},
            {"model_name": "/models/rife", "filename": "r.safetensors"},
            {"model_name": "org/rife", "filename": "../../etc/passwd"},
            {"model_name": "org/rife", "filename": "/abs/r.safetensors"},
        ],
    )
    def test_a_reference_path_or_climbing_name_gives_no_entry(self, arguments):
        assert interpolate_frames.required_weights(arguments) == []


class TestFacexlib:
    def test_not_installed_lists_both_urls(self, monkeypatch):
        monkeypatch.setattr(restore_faces, "package_dir", lambda name: None)
        assert restore_faces.required_weights({}) == [
            {"url": restore_faces.FACEXLIB_DETECTION_URL},
            {"url": restore_faces.FACEXLIB_PARSING_URL},
        ]

    def test_without_parse_only_detection_is_listed(self, monkeypatch):
        monkeypatch.setattr(restore_faces, "package_dir", lambda name: None)
        assert restore_faces.required_weights({"use_parse": False}) == [
            {"url": restore_faces.FACEXLIB_DETECTION_URL}
        ]

    def test_a_weight_already_fetched_is_omitted(self, monkeypatch, tmp_path):
        (tmp_path / "weights").mkdir()
        (tmp_path / "weights" / "detection_Resnet50_Final.pth").write_bytes(b"w")
        monkeypatch.setattr(restore_faces, "package_dir", lambda name: tmp_path)
        assert restore_faces.required_weights({}) == [
            {"url": restore_faces.FACEXLIB_PARSING_URL}
        ]

    def test_the_plan_lists_them_as_url_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(restore_faces, "package_dir", lambda name: None)
        monkeypatch.setattr(
            dw.plan_downloads, "scan_models", lambda cache_dir=None: {"repos": []}
        )
        spec = workflow(("restore_faces", {"use_parse": False}))
        required = downloads_required(
            spec, str(tmp_path), str(tmp_path), str(tmp_path / "hub"), False
        )
        assert [(d["repo"], d.get("url")) for d in required] == [
            (None, restore_faces.FACEXLIB_DETECTION_URL)
        ]

    def test_the_hook_does_not_import_facexlib_or_cv2(self):
        """torch is not asserted: importing dw loads it already."""
        script = textwrap.dedent(
            """
            import sys
            from dw.tasks.registry import required_weights_hook
            import dw.tasks.model_handlers
            required_weights_hook("restore_faces")({"model_name": "org/gfp"})
            print(sorted({"facexlib", "cv2"} & set(sys.modules)))
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip().splitlines()[-1] == "[]"


class TestThePlanChecksTheFile:
    def test_an_uncached_file_is_listed_with_its_key(self, plan):
        assert plan(workflow(("interpolate_frames", {}))) == [
            {
                "repo": "imaginairy/rife-interpolation",
                "filename": "rife-flownet-4.13.2.safetensors",
                "revision": None,
                "gb": None,
                "gated": None,
                "access_blocked": None,
            }
        ]

    def test_a_pinned_file_is_cached_without_refs_main(self, plan, hub):
        cache(
            hub,
            face_track.DEFAULT_DETECTOR_REPO,
            face_track.DEFAULT_DETECTOR_FILE,
            commit=face_track.DETECTOR_REVISION,
        )
        assert plan(workflow(("crop_face_track", {}))) == []

    def test_a_file_at_main_is_cached_through_refs_main(self, plan, hub):
        cache(hub, "org/rife", "r.safetensors", ref="main")
        spec = workflow(
            (
                "interpolate_frames",
                {"model_name": "org/rife", "filename": "r.safetensors"},
            )
        )
        assert plan(spec) == []

    def test_another_file_of_the_repo_does_not_stand_in(self, plan, hub):
        cache(hub, "org/rife", "other.safetensors", ref="main")
        spec = workflow(
            (
                "interpolate_frames",
                {"model_name": "org/rife", "filename": "r.safetensors"},
            )
        )
        assert [(d["repo"], d["filename"]) for d in plan(spec)] == [
            ("org/rife", "r.safetensors")
        ]

    def test_a_whole_repo_entry_subsumes_the_file(self, plan):
        spec = workflow(
            ("upscale", {"model_name": "org/up"}),
            ("upscale", {"model_name": "org/up", "filename": "x.pth"}),
        )
        assert [(d["repo"], d.get("filename")) for d in plan(spec)] == [
            ("org/up", None)
        ]

    def test_two_steps_naming_one_file_list_it_once(self, plan):
        spec = workflow(("interpolate_frames", {}), ("interpolate_frames", {}))
        assert len(plan(spec)) == 1


class TestFileInfo:
    def test_sizes_the_file_from_its_head(self, monkeypatch):
        seen = {}

        class Meta:
            size = 3 * 1024**3

        def head(url, timeout=None):
            seen["url"] = url
            return Meta()

        monkeypatch.setattr(dw.plan_downloads, "get_hf_file_metadata", head)
        assert dw.plan_downloads._file_info("org/r", "sub/w.safetensors", PIN) == (
            3.0,
            False,
        )
        assert PIN in seen["url"] and seen["url"].endswith("sub/w.safetensors")

    def test_a_tiny_file_is_not_zero(self, monkeypatch):
        class Meta:
            size = 1000

        monkeypatch.setattr(
            dw.plan_downloads, "get_hf_file_metadata", lambda *a, **k: Meta()
        )
        assert dw.plan_downloads._file_info("org/r", "w.onnx", None) == (0.01, False)

    def test_a_gated_file_is_blocked(self, monkeypatch):
        import httpx
        from huggingface_hub.utils import GatedRepoError

        response = httpx.Response(403, request=httpx.Request("GET", "https://hf.co/x"))

        def gated(*a, **k):
            raise GatedRepoError("no access", response=response)

        monkeypatch.setattr(dw.plan_downloads, "get_hf_file_metadata", gated)
        assert dw.plan_downloads._file_info("org/r", "w.onnx", None) == (None, True)

    def test_offline_is_unknown(self, monkeypatch):
        def offline(*a, **k):
            raise OSError("offline")

        monkeypatch.setattr(dw.plan_downloads, "get_hf_file_metadata", offline)
        assert dw.plan_downloads._file_info("org/r", "w.onnx", None) == (None, None)


def test_gate_warnings_skip_an_entry_with_no_repo():
    assert (
        gate_warnings([{"repo": None, "url": "https://x/y", "access_blocked": True}])
        == []
    )
