"""
Unit tests for the on-disk cache of SDNQ-quantized components (#815, #826).

Covers the key, the hub-snapshot rule, the disk floor, save/load failure
handling, the sweep, the switch and directory, the load_components()
orchestration against a fake modular pipeline, one real sdnq round trip on
CPU, and the two call sites (load_component and the server start).
"""

import errno
import json
import logging
import os
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

import dw.quantized_cache as qc
from dw.events import RunContext, activate_context, deactivate_context

sdnq = pytest.importorskip("sdnq")
from sdnq import SDNQConfig  # noqa: E402

GIB = 2**30
COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
KEY = "c" * 64
VERSIONS = {"sdnq": "1", "diffusers": "2", "transformers": "3", "torch": "4"}
Usage = namedtuple("Usage", "total used free")


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    # set then delete, so monkeypatch restores whatever the run began with
    for name in (qc.ENABLE_ENV_VAR, qc.DIR_ENV_VAR):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    monkeypatch.setattr(qc, "installed_versions", lambda: dict(VERSIONS))


@pytest.fixture
def events():
    collected = []
    token = activate_context(RunContext(on_event=collected.append))
    yield collected
    deactivate_context(token)


@pytest.fixture
def cache_on(monkeypatch, tmp_path):
    root = tmp_path / "cache"
    monkeypatch.setenv(qc.ENABLE_ENV_VAR, "1")
    monkeypatch.setenv(qc.DIR_ENV_VAR, str(root))
    return root


def make_snapshot(
    hub,
    model="org/name",
    commit=COMMIT,
    subfolder=None,
    files=("model.safetensors",),
    config=True,
    ref="main",
):
    repo = hub / f"models--{model.replace('/', '--')}"
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    if ref:
        (repo / "refs" / ref).write_text(commit)
    folder = repo / "snapshots" / commit
    if subfolder:
        folder = folder / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    if config:
        (folder / "config.json").write_text("{}")
    for name in files:
        (folder / name).write_bytes(b"weights")
    return folder


def disk(total_gib, free_gib):
    return lambda _path: Usage(total_gib * GIB, 0, int(free_gib * GIB))


def fake_save(component, path):
    os.makedirs(path)
    with open(os.path.join(path, "model.safetensors"), "wb") as f:
        f.write(b"x" * 100)


def entry_json(directory, **fields):
    entry = {"versions": dict(VERSIONS), "bytes": 10, **fields}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / qc.ENTRY_FILE).write_text(json.dumps(entry))


def base_inputs(**overrides):
    values = {
        "model_name": "org/name",
        "subfolder": "vae",
        "commit": COMMIT,
        "quantizer": qc.quantizer_arguments(SDNQConfig(weights_dtype="int8")),
        "torch_dtype": torch.bfloat16,
    }
    values.update(overrides)
    return qc.key_inputs(**values)


# 1. The key


class TestKey:
    def test_identical_inputs_give_identical_key(self):
        assert qc.cache_key(base_inputs()) == qc.cache_key(base_inputs())
        assert qc._KEY_PATTERN.match(qc.cache_key(base_inputs()))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"model_name": "org/other"},
            {"subfolder": "unet"},
            {"commit": OTHER_COMMIT},
            {"quantizer": qc.quantizer_arguments(SDNQConfig(weights_dtype="int4"))},
            {"torch_dtype": torch.float16},
        ],
    )
    def test_each_input_changes_the_key(self, overrides):
        assert qc.cache_key(base_inputs(**overrides)) != qc.cache_key(base_inputs())

    def test_package_version_changes_the_key(self, monkeypatch):
        before = qc.cache_key(base_inputs())
        monkeypatch.setattr(
            qc, "installed_versions", lambda: {**VERSIONS, "sdnq": "other"}
        )
        assert qc.cache_key(base_inputs()) != before

    def test_installed_versions_reads_the_four_packages(self, monkeypatch):
        monkeypatch.undo()
        qc.installed_versions.cache_clear()
        versions = qc.installed_versions()
        assert set(versions) == set(qc.VERSIONED_PACKAGES)
        assert all(isinstance(v, str) and v for v in versions.values())

    def test_placement_arguments_do_not_change_the_key(self):
        plain = SDNQConfig(weights_dtype="int8")
        placed = SDNQConfig(
            weights_dtype="int8",
            quantization_device="cpu",
            return_device="cpu",
            use_quantized_matmul=True,
        )
        a = base_inputs(quantizer=qc.quantizer_arguments(plain))
        b = base_inputs(quantizer=qc.quantizer_arguments(placed))
        assert qc.cache_key(a) == qc.cache_key(b)

    def test_quantizer_arguments_drop_only_the_unkeyed_ones(self):
        described = qc.quantizer_arguments(SDNQConfig(weights_dtype="int8"))
        assert described["class"].endswith("SDNQConfig")
        assert described["arguments"]["weights_dtype"] == "int8"
        for name in qc.UNKEYED_ARGUMENTS:
            assert name not in described["arguments"]

    def test_empty_subfolder_and_none_are_the_same(self):
        assert qc.cache_key(base_inputs(subfolder=None)) == qc.cache_key(
            base_inputs(subfolder="")
        )

    def test_entry_path_refuses_a_non_key(self, tmp_path):
        with pytest.raises(ValueError):
            qc.entry_path(str(tmp_path), "../escape")


# 2. The snapshot rule


class TestSnapshot:
    def test_complete_single_safetensors(self, tmp_path):
        make_snapshot(tmp_path, subfolder="vae")
        assert qc.lookup_commit("org/name", "vae", None, tmp_path) == COMMIT

    def test_complete_without_subfolder(self, tmp_path):
        make_snapshot(tmp_path)
        assert qc.lookup_commit("org/name", None, "main", tmp_path) == COMMIT

    def test_sharded_index_with_every_shard(self, tmp_path):
        folder = make_snapshot(tmp_path, files=("a-1.safetensors", "a-2.safetensors"))
        index = {"weight_map": {"x": "a-1.safetensors", "y": "a-2.safetensors"}}
        (folder / "model.safetensors.index.json").write_text(json.dumps(index))
        assert qc.lookup_commit("org/name", None, None, tmp_path) == COMMIT

    def test_sharded_index_with_a_missing_shard(self, tmp_path):
        folder = make_snapshot(tmp_path, files=("a-1.safetensors",))
        index = {"weight_map": {"x": "a-1.safetensors", "y": "a-2.safetensors"}}
        (folder / "model.safetensors.index.json").write_text(json.dumps(index))
        assert qc.lookup_commit("org/name", None, None, tmp_path) is None

    def test_sharded_index_with_a_dangling_symlink(self, tmp_path):
        folder = make_snapshot(tmp_path, files=("a-1.safetensors",))
        os.symlink(tmp_path / "no-such-blob", folder / "a-2.safetensors")
        index = {"weight_map": {"x": "a-1.safetensors", "y": "a-2.safetensors"}}
        (folder / "model.safetensors.index.json").write_text(json.dumps(index))
        assert qc.lookup_commit("org/name", None, None, tmp_path) is None

    def test_symlink_to_a_real_blob_is_complete(self, tmp_path):
        folder = make_snapshot(tmp_path, files=())
        blob = tmp_path / "blob"
        blob.write_bytes(b"w")
        os.symlink(blob, folder / "model.safetensors")
        assert qc.lookup_commit("org/name", None, None, tmp_path) == COMMIT

    def test_dangling_single_file(self, tmp_path):
        folder = make_snapshot(tmp_path, files=())
        os.symlink(tmp_path / "gone", folder / "model.safetensors")
        assert qc.lookup_commit("org/name", None, None, tmp_path) is None

    def test_missing_config(self, tmp_path):
        make_snapshot(tmp_path, config=False)
        assert qc.lookup_commit("org/name", None, None, tmp_path) is None

    def test_variant_file_only(self, tmp_path):
        make_snapshot(tmp_path, files=("model.fp16.safetensors",))
        assert qc.lookup_commit("org/name", None, None, tmp_path) is None

    def test_no_weights_at_all(self, tmp_path):
        make_snapshot(tmp_path, files=())
        assert qc.lookup_commit("org/name", None, None, tmp_path) is None

    def test_forty_hex_revision_passes_through(self, tmp_path):
        assert qc.resolve_commit("org/name", OTHER_COMMIT, tmp_path) == OTHER_COMMIT

    def test_unknown_ref_or_repo(self, tmp_path):
        assert qc.resolve_commit("org/name", "main", tmp_path) is None
        make_snapshot(tmp_path)
        assert qc.resolve_commit("org/name", "dev", tmp_path) is None

    def test_ref_that_is_not_a_commit(self, tmp_path):
        make_snapshot(tmp_path)
        (tmp_path / "models--org--name" / "refs" / "main").write_text("garbage")
        assert qc.resolve_commit("org/name", None, tmp_path) is None

    def test_hub_cache_defaults_to_the_hub_constant(self, tmp_path, monkeypatch):
        from huggingface_hub import constants

        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
        make_snapshot(tmp_path)
        assert qc.lookup_commit("org/name") == COMMIT


class TestSaveCommit:
    @staticmethod
    def loaded(commit):
        return SimpleNamespace(config=SimpleNamespace(_commit_hash=commit))

    def test_recorded_that_disagrees_with_refs_is_unpinned(self, tmp_path):
        make_snapshot(tmp_path, commit=OTHER_COMMIT)
        assert (
            qc.save_commit(self.loaded(COMMIT), "org/name", cache_dir=tmp_path) is None
        )

    def test_recorded_that_agrees_is_used(self, tmp_path):
        make_snapshot(tmp_path)
        assert (
            qc.save_commit(self.loaded(COMMIT), "org/name", cache_dir=tmp_path)
            == COMMIT
        )

    def test_recorded_with_no_ref_is_used(self, tmp_path):
        assert (
            qc.save_commit(self.loaded(COMMIT), "org/name", cache_dir=tmp_path)
            == COMMIT
        )

    def test_recorded_in_a_dict_config(self, tmp_path):
        component = SimpleNamespace(config={"_commit_hash": COMMIT})
        assert qc.save_commit(component, "org/name", cache_dir=tmp_path) == COMMIT

    def test_nothing_recorded_and_incomplete_is_unpinned(self, tmp_path):
        make_snapshot(tmp_path, files=())
        assert qc.save_commit(nn.Linear(1, 1), "org/name", cache_dir=tmp_path) is None

    def test_nothing_recorded_and_complete_uses_the_ref(self, tmp_path):
        make_snapshot(tmp_path, subfolder="vae")
        assert (
            qc.save_commit(nn.Linear(1, 1), "org/name", "vae", cache_dir=tmp_path)
            == COMMIT
        )

    def test_malformed_recorded_commit_is_ignored(self, tmp_path):
        assert (
            qc.save_commit(self.loaded("short"), "org/name", cache_dir=tmp_path) is None
        )


# 3. The floor


class TestFloor:
    def test_floor_is_ten_percent_of_a_small_volume(self):
        assert qc.floor_bytes(100 * GIB) == 10 * GIB

    def test_floor_caps_at_fifty_gib_on_a_large_volume(self):
        assert qc.floor_bytes(4000 * GIB) == 50 * GIB
        assert qc.floor_bytes(500 * GIB) == 50 * GIB

    def test_low_disk_skips_and_writes_nothing(self, tmp_path, monkeypatch):
        component = nn.Linear(100, 100)
        estimate = qc.component_nbytes(component)
        monkeypatch.setattr(qc.shutil, "disk_usage", disk(100, 10 + 0))
        # free - estimate < floor (10 GiB) for any estimate > 0
        outcome = qc.save_entry(str(tmp_path), KEY, component, {})
        assert estimate > 0
        assert outcome == ("skipped", qc.LOW_DISK, None)
        assert os.listdir(tmp_path) == []

    def test_one_byte_short_of_the_floor_skips(self, tmp_path, monkeypatch):
        component = nn.Linear(100, 100)
        estimate = qc.component_nbytes(component)
        floor = 10 * GIB
        monkeypatch.setattr(
            qc.shutil,
            "disk_usage",
            lambda _p: Usage(100 * GIB, 0, floor + estimate - 1),
        )
        assert qc.save_entry(str(tmp_path), KEY, component, {})[0:2] == (
            "skipped",
            qc.LOW_DISK,
        )

    def test_ample_disk_saves(self, tmp_path, monkeypatch):
        component = nn.Linear(100, 100)
        estimate = qc.component_nbytes(component)
        monkeypatch.setattr(
            qc.shutil,
            "disk_usage",
            lambda _p: Usage(100 * GIB, 0, 10 * GIB + estimate),
        )
        monkeypatch.setattr(sdnq, "save_sdnq_model", fake_save)
        result, reason, nbytes = qc.save_entry(
            str(tmp_path), KEY, component, {"model_name": "org/name"}
        )
        assert (result, reason) == ("saved", None)
        assert nbytes == 100
        entry = qc.read_entry(str(tmp_path / KEY))
        assert entry["key"] == KEY and entry["bytes"] == 100
        assert entry["model_name"] == "org/name" and entry["last_hit"] is None
        assert sorted(os.listdir(tmp_path)) == [KEY]

    def test_unreachable_root_is_skipped(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        outcome = qc.save_entry(str(blocker / "sub"), KEY, nn.Linear(1, 1), {})
        assert outcome[0] == "skipped" and outcome[1].startswith("save_failed")


# 4. The start-up warning


class TestLogDisk:
    @pytest.fixture(autouse=True)
    def _logging(self, caplog):
        caplog.set_level(logging.INFO, logger="dw")
        self.caplog = caplog

    def warned(self):
        return [r for r in self.caplog.records if r.levelno == logging.WARNING]

    def test_empty_cache_below_the_floor_warns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qc.shutil, "disk_usage", disk(100, 9))
        assert qc.log_disk(str(tmp_path)) is True
        assert len(self.warned()) == 1

    def test_empty_cache_just_above_the_floor_is_quiet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qc.shutil, "disk_usage", disk(100, 10.5))
        assert qc.log_disk(str(tmp_path)) is False
        assert self.warned() == []

    def test_margin_is_the_largest_entry(self, tmp_path, monkeypatch):
        entry_json(tmp_path / ("1" * 64), bytes=2 * GIB)
        entry_json(tmp_path / ("2" * 64), bytes=5 * GIB)
        # floor is 10 GiB, the largest entry 5 GiB
        monkeypatch.setattr(qc.shutil, "disk_usage", disk(100, 14))
        assert qc.log_disk(str(tmp_path)) is True
        assert len(self.warned()) == 1
        monkeypatch.setattr(qc.shutil, "disk_usage", disk(100, 16))
        assert qc.log_disk(str(tmp_path)) is False
        assert len(self.warned()) == 1

    def test_entry_without_readable_json_is_sized_from_disk(
        self, tmp_path, monkeypatch
    ):
        broken = tmp_path / ("3" * 64)
        broken.mkdir()
        (broken / "w").write_bytes(b"x" * 10)
        assert qc._entry_sizes(str(tmp_path)) == [10]


# 5. Failures


class TestSaveFailures:
    @pytest.fixture(autouse=True)
    def _roomy(self, monkeypatch):
        monkeypatch.setattr(qc.shutil, "disk_usage", disk(1000, 900))

    def test_enospc(self, tmp_path, monkeypatch):
        def full(component, path):
            os.makedirs(path)
            (tmp_path / "scratch").write_text("")
            with open(os.path.join(path, "partial"), "w") as f:
                f.write("x")
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(sdnq, "save_sdnq_model", full)
        root = tmp_path / "root"
        outcome = qc.save_entry(str(root), KEY, nn.Linear(2, 2), {})
        assert outcome == ("skipped", "save_failed: ENOSPC", None)
        assert os.listdir(root) == []

    def test_generic_exception_names_its_type(self, tmp_path, monkeypatch):
        def boom(component, path):
            os.makedirs(path)
            raise RuntimeError("nope")

        monkeypatch.setattr(sdnq, "save_sdnq_model", boom)
        outcome = qc.save_entry(str(tmp_path), KEY, nn.Linear(2, 2), {})
        assert outcome == ("skipped", "save_failed: RuntimeError", None)
        assert os.listdir(tmp_path) == []


# 6. The race


class TestRace:
    def test_another_worker_wins(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qc.shutil, "disk_usage", disk(1000, 900))
        root = str(tmp_path)

        def racing(component, path):
            fake_save(component, path)
            # another worker's finished entry lands first
            winner = os.path.join(root, KEY)
            os.makedirs(winner)
            with open(os.path.join(winner, "model.safetensors"), "wb") as f:
                f.write(b"y" * 7)

        monkeypatch.setattr(sdnq, "save_sdnq_model", racing)
        outcome = qc.save_entry(root, KEY, nn.Linear(2, 2), {})
        assert outcome == ("saved", qc.ALREADY_SAVED, 7)
        assert os.listdir(root) == [KEY]


# 7. A corrupt entry


class TestLoadEntry:
    def test_missing_entry_is_a_miss(self, tmp_path):
        assert qc.load_entry(str(tmp_path), KEY, nn.Linear, None, "cpu", {}) is None

    def test_failing_load_deletes_the_entry(self, tmp_path, monkeypatch):
        entry_json(tmp_path / KEY)
        monkeypatch.setattr(
            sdnq, "load_sdnq_model", MagicMock(side_effect=ValueError("corrupt"))
        )
        assert qc.load_entry(str(tmp_path), KEY, nn.Linear, None, "cpu", {}) is None
        assert not (tmp_path / KEY).exists()

    def test_hit_passes_arguments_and_records_last_hit(self, tmp_path, monkeypatch):
        entry_json(tmp_path / KEY)
        module = nn.Linear(2, 2)
        loader = MagicMock(return_value=module)
        monkeypatch.setattr(sdnq, "load_sdnq_model", loader)
        loaded = qc.load_entry(
            str(tmp_path),
            KEY,
            nn.Linear,
            torch.float32,
            "cpu",
            {"dequantize_fp32": True, "use_quantized_matmul": False},
        )
        assert loaded is module
        kwargs = loader.call_args.kwargs
        assert kwargs["model_cls"] is nn.Linear and kwargs["dtype"] is torch.float32
        assert kwargs["dequantize_fp32"] is True
        assert kwargs["use_quantized_matmul"] is False
        assert qc.read_entry(str(tmp_path / KEY))["last_hit"] is not None


# 8. The sweep


class TestSweep:
    def test_nothing_without_a_root(self, tmp_path):
        assert qc.sweep(str(tmp_path / "absent")) == (0, 0)

    def test_deletes_only_what_is_stale(self, tmp_path):
        tmp_dir = tmp_path / f"{'1' * 64}.tmp-9-abcd"
        tmp_dir.mkdir()
        (tmp_dir / "w").write_bytes(b"x" * 10)
        (tmp_path / f"{'2' * 64}.tmp-9-file").write_bytes(b"x" * 5)
        stale = tmp_path / ("3" * 64)
        entry_json(stale, versions={**VERSIONS, "torch": "old"})
        (stale / "w").write_bytes(b"x" * 20)
        garbage = tmp_path / ("4" * 64)
        garbage.mkdir()
        (garbage / qc.ENTRY_FILE).write_text("{not json")
        (garbage / "w").write_bytes(b"x" * 1)
        no_json = tmp_path / ("5" * 64)
        no_json.mkdir()
        (no_json / "w").write_bytes(b"x" * 2)
        not_object = tmp_path / ("7" * 64)
        not_object.mkdir()
        (not_object / qc.ENTRY_FILE).write_text("[1]")
        current = tmp_path / ("6" * 64)
        entry_json(current)
        (current / "w").write_bytes(b"x" * 99)
        (tmp_path / "notes.txt").write_text("keep")
        (tmp_path / "other").mkdir()

        count, freed = qc.sweep(str(tmp_path))

        assert count == 6
        assert freed >= 10 + 5 + 20 + 1 + 2
        assert sorted(os.listdir(tmp_path)) == sorted(["6" * 64, "notes.txt", "other"])
        assert (current / "w").exists()


# 9. The switch and the directory


class TestSwitchAndDirectory:
    @pytest.mark.parametrize(
        "value,expected",
        [("1", True), ("0", False), ("true", False), ("", False), (None, False)],
    )
    def test_enabled_only_for_one(self, monkeypatch, value, expected):
        if value is not None:
            monkeypatch.setenv(qc.ENABLE_ENV_VAR, value)
        assert qc.enabled() is expected

    def test_default_root_is_under_hf_home(self, tmp_path, monkeypatch):
        from huggingface_hub import constants

        monkeypatch.setattr(constants, "HF_HOME", str(tmp_path / "hf"))
        assert qc.cache_root() == str(tmp_path / "hf" / "dw-quantized")

    def test_configured_root_is_made_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(qc.DIR_ENV_VAR, "rel/cache")
        assert qc.cache_root() == str(tmp_path / "rel" / "cache")

    def test_startup_with_the_cache_off_does_nothing(self, tmp_path, monkeypatch):
        from huggingface_hub import constants

        monkeypatch.setattr(constants, "HF_HOME", str(tmp_path / "hf"))
        qc.configure_at_startup()
        assert not (tmp_path / "hf").exists()
        assert qc.DIR_ENV_VAR not in os.environ

    def test_startup_with_the_cache_off_leaves_a_configured_dir(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(qc.DIR_ENV_VAR, "rel")
        monkeypatch.setenv(qc.ENABLE_ENV_VAR, "0")
        qc.configure_at_startup()
        assert os.environ[qc.DIR_ENV_VAR] == "rel"
        assert not (tmp_path / "rel").exists()

    def test_startup_with_the_cache_on_pins_and_creates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(qc.ENABLE_ENV_VAR, "1")
        monkeypatch.setenv(qc.DIR_ENV_VAR, "rel")
        monkeypatch.setattr(qc.shutil, "disk_usage", disk(1000, 900))
        stale = tmp_path / "rel" / ("8" * 64)
        entry_json(stale, versions={"torch": "old"})
        qc.configure_at_startup()
        assert os.environ[qc.DIR_ENV_VAR] == str(tmp_path / "rel")
        assert (tmp_path / "rel").is_dir()
        assert not stale.exists()

    def test_startup_with_the_default_root_pins_it(self, tmp_path, monkeypatch):
        from huggingface_hub import constants

        monkeypatch.setattr(constants, "HF_HOME", str(tmp_path / "hf"))
        monkeypatch.setenv(qc.ENABLE_ENV_VAR, "1")
        monkeypatch.setattr(qc.shutil, "disk_usage", disk(1000, 900))
        qc.configure_at_startup()
        assert os.environ[qc.DIR_ENV_VAR] == str(tmp_path / "hf" / "dw-quantized")
        assert (tmp_path / "hf" / "dw-quantized").is_dir()

    def test_unusable_root_does_not_raise(self, tmp_path, monkeypatch):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        monkeypatch.setenv(qc.ENABLE_ENV_VAR, "1")
        monkeypatch.setenv(qc.DIR_ENV_VAR, str(blocker / "sub"))
        qc.configure_at_startup()


# 10. Orchestration


class FakePipeline:
    """A modular pipeline as the cache sees one."""

    def __init__(self, specs):
        self._component_specs = {spec.name: spec for spec in specs}
        for name in self._component_specs:
            setattr(self, name, None)
        self.load_calls = []
        self.loaded = []
        self.updates = []

    def load_components(self, **kwargs):
        self.load_calls.append(kwargs)
        names = kwargs.get("names")
        for name in self._component_specs:
            if names is not None and name not in names:
                continue
            if getattr(self, name) is None:
                setattr(self, name, nn.Linear(4, 4))
                self.loaded.append(name)

    def update_components(self, **kwargs):
        self.updates.append(kwargs)
        for name, component in kwargs.items():
            setattr(self, name, component)


def make_spec(name, subfolder=None, model="org/name", **fields):
    values = {
        "name": name,
        "default_creation_method": "from_pretrained",
        "pretrained_model_name_or_path": model,
        "subfolder": subfolder or name,
        "revision": None,
        "variant": None,
        "type_hint": nn.Linear,
        "load_id": f"{model}|{subfolder or name}",
    }
    values.update(fields)
    return SimpleNamespace(**values)


def results(events):
    return [(e["component"], e["result"]) for e in events if e["event"] == qc.EVENT]


class TestLoadComponents:
    @pytest.fixture
    def hub(self, tmp_path):
        hub = tmp_path / "hub"
        for sub in ("vae", "text_encoder"):
            make_snapshot(hub, subfolder=sub)
        return hub

    @pytest.fixture(autouse=True)
    def _fakes(self, monkeypatch):
        monkeypatch.setattr(qc.shutil, "disk_usage", disk(1000, 900))
        monkeypatch.setattr(sdnq, "save_sdnq_model", fake_save)

    def arguments(self, hub, **extra):
        return {
            "quantization_config": SDNQConfig(weights_dtype="int8"),
            "torch_dtype": torch.bfloat16,
            "cache_dir": str(hub),
            **extra,
        }

    def pipeline(self):
        return FakePipeline([make_spec("vae"), make_spec("text_encoder")])

    def test_cache_off_is_a_plain_passthrough(self, hub, events, tmp_path):
        pipeline = self.pipeline()
        arguments = self.arguments(hub)
        qc.load_components(pipeline, arguments)
        assert pipeline.load_calls == [arguments]
        assert pipeline.loaded == ["vae", "text_encoder"]
        assert events == []
        assert not (tmp_path / "cache").exists()

    def test_first_load_misses_then_saves(self, hub, events, cache_on):
        pipeline = self.pipeline()
        qc.load_components(pipeline, self.arguments(hub))
        assert results(events) == [
            ("vae", "miss"),
            ("vae", "saved"),
            ("text_encoder", "miss"),
            ("text_encoder", "saved"),
        ]
        for event in events:
            assert {"component", "result", "seconds", "bytes"} <= set(event)
        saved = [e for e in events if e["result"] == "saved"]
        assert all(e["bytes"] == 100 for e in saved)
        entries = [n for n in os.listdir(cache_on) if qc._KEY_PATTERN.match(n)]
        assert len(entries) == 2
        assert pipeline.updates == []

    def test_second_load_hits(self, hub, events, cache_on, monkeypatch):
        qc.load_components(self.pipeline(), self.arguments(hub))
        events.clear()

        cached = nn.Linear(4, 4)
        loader = MagicMock(side_effect=lambda *a, **k: nn.Linear(4, 4))
        monkeypatch.setattr(sdnq, "load_sdnq_model", loader)
        pipeline = self.pipeline()
        qc.load_components(pipeline, self.arguments(hub))

        assert results(events) == [("vae", "hit"), ("text_encoder", "hit")]
        assert all(e["bytes"] == qc.component_nbytes(cached) for e in events)
        assert len(pipeline.updates) == 1
        assert set(pipeline.updates[0]) == {"vae", "text_encoder"}
        for name, component in pipeline.updates[0].items():
            spec = pipeline._component_specs[name]
            assert component._diffusers_load_id == spec.load_id
        assert pipeline.loaded == []
        assert loader.call_count == 2

    def test_named_components_have_the_hit_removed(
        self, hub, events, cache_on, monkeypatch
    ):
        qc.load_components(self.pipeline(), self.arguments(hub))
        events.clear()
        monkeypatch.setattr(sdnq, "load_sdnq_model", lambda *a, **k: nn.Linear(4, 4))

        pipeline = self.pipeline()
        qc.load_components(pipeline, self.arguments(hub, names=["vae"]))

        assert results(events) == [("vae", "hit")]
        assert pipeline.load_calls[0]["names"] == []
        assert pipeline.loaded == []

    def test_a_string_name_is_accepted(self, hub, events, cache_on, monkeypatch):
        qc.load_components(self.pipeline(), self.arguments(hub))
        events.clear()
        monkeypatch.setattr(sdnq, "load_sdnq_model", lambda *a, **k: nn.Linear(4, 4))
        pipeline = self.pipeline()
        qc.load_components(pipeline, self.arguments(hub, names="vae"))
        assert pipeline.load_calls[0]["names"] == []

    def test_a_failing_entry_falls_back_to_a_fresh_load(
        self, hub, events, cache_on, monkeypatch
    ):
        qc.load_components(self.pipeline(), self.arguments(hub))
        events.clear()
        monkeypatch.setattr(
            sdnq, "load_sdnq_model", MagicMock(side_effect=OSError("bad"))
        )
        pipeline = self.pipeline()
        qc.load_components(pipeline, self.arguments(hub))
        assert pipeline.loaded == ["vae", "text_encoder"]
        assert {r for r in results(events) if r[1] == "miss"} == {
            ("vae", "miss"),
            ("text_encoder", "miss"),
        }
        assert ("vae", "saved") in results(events)

    def test_a_different_config_is_a_different_entry(
        self, hub, events, cache_on, monkeypatch
    ):
        qc.load_components(self.pipeline(), self.arguments(hub))
        events.clear()
        arguments = self.arguments(hub, quantization_config=SDNQConfig("int4"))
        qc.load_components(self.pipeline(), arguments)
        assert ("vae", "hit") not in results(events)
        assert ("vae", "saved") in results(events)

    def test_per_component_configs_are_read(self, hub, events, cache_on):
        arguments = self.arguments(
            hub,
            quantization_config={
                "default": None,
                "vae": SDNQConfig(weights_dtype="int8"),
            },
        )
        qc.load_components(self.pipeline(), arguments)
        assert {name for name, _ in results(events)} == {"vae"}

    def test_a_non_sdnq_config_is_left_alone(self, hub, events, cache_on):
        pipeline = self.pipeline()
        arguments = self.arguments(hub, quantization_config=object())
        qc.load_components(pipeline, arguments)
        assert events == []
        assert pipeline.loaded == ["vae", "text_encoder"]
        assert not cache_on.exists()

    def test_no_quantization_config_is_left_alone(self, hub, events, cache_on):
        pipeline = self.pipeline()
        qc.load_components(pipeline, {"torch_dtype": torch.bfloat16})
        assert events == [] and pipeline.loaded == ["vae", "text_encoder"]

    def test_variant_is_unsupported(self, hub, events, cache_on):
        pipeline = FakePipeline([make_spec("vae", variant="fp16")])
        qc.load_components(pipeline, self.arguments(hub))
        assert results(events) == [("vae", "skipped")]
        assert events[0]["reason"] == qc.UNSUPPORTED
        assert pipeline.loaded == ["vae"]

    def test_a_variant_keyword_is_unsupported(self, hub, events, cache_on):
        pipeline = FakePipeline([make_spec("vae")])
        qc.load_components(pipeline, self.arguments(hub, variant="fp16"))
        assert events[0]["reason"] == qc.UNSUPPORTED

    def test_local_path_is_skipped(self, hub, events, cache_on, tmp_path):
        local = tmp_path / "local-model"
        local.mkdir()
        pipeline = FakePipeline([make_spec("vae", model=str(local), subfolder="vae")])
        qc.load_components(pipeline, self.arguments(hub))
        assert results(events) == [("vae", "skipped")]
        assert events[0]["reason"] == qc.LOCAL_PATH
        assert pipeline.loaded == ["vae"]

    def test_incomplete_snapshot_loads_but_is_not_saved(
        self, events, cache_on, tmp_path
    ):
        pipeline = FakePipeline([make_spec("vae")])
        qc.load_components(pipeline, self.arguments(tmp_path / "empty-hub"))
        assert results(events) == [("vae", "miss"), ("vae", "skipped")]
        assert events[1]["reason"] == qc.UNPINNED_SNAPSHOT
        assert pipeline.loaded == ["vae"]

    def test_components_already_loaded_are_not_candidates(self, hub, events, cache_on):
        pipeline = self.pipeline()
        pipeline.vae = nn.Linear(1, 1)
        qc.load_components(pipeline, self.arguments(hub))
        assert {name for name, _ in results(events)} == {"text_encoder"}

    def test_unreadable_specs_fall_through_to_the_plain_load(
        self, hub, events, cache_on
    ):
        pipeline = self.pipeline()
        pipeline._component_specs = None
        pipeline.load_components = MagicMock()
        qc.load_components(pipeline, self.arguments(hub))
        pipeline.load_components.assert_called_once()
        assert events == []

    def test_a_failing_save_never_fails_the_load(
        self, hub, events, cache_on, monkeypatch
    ):
        def boom(component, path):
            raise OSError(errno.ENOSPC, "full")

        monkeypatch.setattr(sdnq, "save_sdnq_model", boom)
        pipeline = FakePipeline([make_spec("vae")])
        qc.load_components(pipeline, self.arguments(hub))
        assert pipeline.loaded == ["vae"]
        assert events[-1]["reason"] == "save_failed: ENOSPC"


# 11. A real round trip


class TestRealRoundTrip:
    def test_save_entry_then_load_entry_reproduces_the_forward_pass(
        self, tmp_path, monkeypatch
    ):
        from diffusers.models import UNet2DModel

        monkeypatch.setattr(qc.shutil, "disk_usage", disk(1000, 900))
        source = tmp_path / "source"
        UNet2DModel(
            sample_size=8,
            in_channels=3,
            out_channels=3,
            block_out_channels=(32, 64),
            layers_per_block=1,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
            norm_num_groups=8,
            attention_head_dim=8,
        ).save_pretrained(source)
        config = SDNQConfig(
            weights_dtype="int8",
            quant_conv=True,
            minimum_allowed_numel=256,
            minimum_allowed_channel_size=8,
        )
        inputs = qc.key_inputs(
            "org/tiny",
            None,
            COMMIT,
            qc.quantizer_arguments(config),
            torch.float32,
        )
        arguments = config.to_dict()
        quantized = UNet2DModel.from_pretrained(
            source, quantization_config=config, torch_dtype=torch.float32
        )
        # the quantizer really ran: something is stored as int8
        assert any(p.dtype == torch.int8 for p in quantized.state_dict().values())

        root = str(tmp_path / "cache")
        key = qc.cache_key(inputs)
        result, reason, nbytes = qc.save_entry(root, key, quantized, inputs)
        assert (result, reason) == ("saved", None) and nbytes > 0
        assert qc.read_entry(os.path.join(root, key))["model_name"] == "org/tiny"

        loaded = qc.load_entry(root, key, UNet2DModel, torch.float32, "cpu", arguments)
        assert loaded is not None

        sample = torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
        step = torch.tensor([1])
        with torch.no_grad():
            expected = quantized(sample, step).sample
            actual = loaded(sample, step).sample
        assert torch.allclose(expected, actual, atol=1e-4)
        assert qc.read_entry(os.path.join(root, key))["last_hit"] is not None


# 12. The save point and the server start


class TestCallSites:
    def configuration(self, component):
        component_type = MagicMock()
        component_type.__name__ = "MockModularPipeline"
        component_type.from_pretrained.return_value = component
        return {
            "component_type": component_type,
            "load_components": {"torch_dtype": torch.bfloat16},
            "components": {"vae": {"truncate_layers": {"layers": 1}}},
        }

    def test_load_component_goes_through_the_cache_and_before_pruning(self):
        from dw.pipeline_processors import components

        component = MagicMock()
        component.vae = MagicMock()
        order = []
        configuration = self.configuration(component)

        with (
            patch.object(
                components.quantized_cache,
                "load_components",
                side_effect=lambda *a: order.append("cache"),
            ) as cache,
            patch.object(
                components,
                "truncate_module_lists",
                side_effect=lambda *a: order.append("truncate"),
            ),
            patch.object(
                components,
                "replace_modules_with_identity",
                side_effect=lambda *a: order.append("identity"),
            ),
        ):
            loaded = components.load_component(
                "pipeline", configuration, {"model_name": "m"}, "cpu"
            )
            components.configure_components(loaded, configuration, "cpu")

        cache.assert_called_once_with(component, {"torch_dtype": torch.bfloat16})
        component.load_components.assert_not_called()
        assert order == ["cache", "truncate", "identity"]

    def test_the_cache_off_still_reaches_load_components(self):
        from dw.pipeline_processors import components

        component = MagicMock()
        components.load_component(
            "pipeline", self.configuration(component), {"model_name": "m"}, "cpu"
        )
        component.load_components.assert_called_once_with(torch_dtype=torch.bfloat16)

    def test_serve_run_configures_the_cache_before_the_app_exists(self):
        import dw.server.app
        import dw.serve as serve

        order = []
        config = SimpleNamespace(
            host="127.0.0.1",
            port=1,
            token=None,
            log_level="INFO",
            workflow_dir=".",
            output_dir=None,
            prompt_dir=None,
            asset_dir=None,
            examples_dirs=[],
            workspace_root=None,
            mcp=False,
            devices=None,
        )
        with (
            patch("dw.startup"),
            patch.object(
                qc, "configure_at_startup", side_effect=lambda: order.append("cache")
            ),
            patch.object(
                dw.server.app,
                "create_app",
                side_effect=lambda **kw: order.append("app"),
            ),
            patch.object(dw.server.app, "default_ui_dir", return_value=None),
            patch("uvicorn.run", side_effect=lambda *a, **k: order.append("run")),
        ):
            serve.run(config)

        assert order == ["cache", "app", "run"]
