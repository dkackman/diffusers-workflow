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
import subprocess
import sys
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from huggingface_hub.file_download import repo_folder_name
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
    repo = hub / repo_folder_name(repo_id=model, repo_type="model")
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

    def test_the_key_does_not_follow_the_process_hash_seed(self):
        # SDNQConfig dedupes modules_to_not_convert through a set, so each
        # worker process holds it in its own order - H3's transformer keyed
        # differently on each card until the key sorted it
        script = (
            "import json\n"
            "from sdnq import SDNQConfig\n"
            "import dw.quantized_cache as qc\n"
            "config = SDNQConfig(weights_dtype='int4', use_quantized_matmul=True,"
            " modules_to_not_convert=[f'block_{i}' for i in range(9)],"
            " modules_dtype_dict={'int8': ['a', 'b', 'c', 'd']})\n"
            "print(json.dumps([config.modules_to_not_convert,"
            " qc.quantizer_arguments(config)]))\n"
        )
        orders, quantizers = set(), set()
        for seed in ("1", "2", "3", "4"):
            out = subprocess.run(
                [sys.executable, "-c", script],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()[-1]
            order, quantizer = json.loads(out)
            orders.add(tuple(order))
            quantizers.add(json.dumps(quantizer, sort_keys=True))
        assert len(orders) > 1, "the seeds did not reorder the list"
        assert len(quantizers) == 1

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

    def test_commit_revision_names_its_snapshot(self, tmp_path):
        make_snapshot(tmp_path, commit=OTHER_COMMIT, ref=None)
        assert qc.resolve_commit("org/name", OTHER_COMMIT, tmp_path) == OTHER_COMMIT
        assert qc.resolve_commit("org/name", COMMIT, tmp_path) is None

    def test_resolution_goes_through_huggingface_hub(self, tmp_path, monkeypatch):
        import huggingface_hub

        calls = []
        real = huggingface_hub.try_to_load_from_cache

        def spy(repo_id, filename, **kwargs):
            calls.append((repo_id, filename, kwargs["revision"]))
            return real(repo_id, filename, **kwargs)

        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", spy)
        make_snapshot(tmp_path, subfolder="vae")
        assert qc.lookup_commit("org/name", "vae", None, tmp_path) == COMMIT
        assert ("org/name", "vae/config.json", "main") in calls
        assert ("org/name", "vae/model.safetensors", COMMIT) in calls

    def test_unknown_ref_or_repo(self, tmp_path):
        assert qc.resolve_commit("org/name", "main", tmp_path) is None
        make_snapshot(tmp_path)
        assert qc.resolve_commit("org/name", "dev", tmp_path) is None

    def test_ref_that_is_not_a_commit(self, tmp_path):
        make_snapshot(tmp_path)
        (tmp_path / "models--org--name" / "refs" / "main").write_text("garbage")
        assert qc.resolve_commit("org/name", None, tmp_path) is None

    def test_repo_folder_linked_to_another_disk(self, tmp_path):
        # a big repo moved off the hub's volume and symlinked back in
        other = tmp_path / "other"
        folder = make_snapshot(other, subfolder="transformer", files=())
        index = {"weight_map": {"x": "a-1.safetensors", "y": "a-2.safetensors"}}
        (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps(index)
        )
        for shard in ("a-1.safetensors", "a-2.safetensors"):
            (folder / shard).write_bytes(b"w")
        hub = tmp_path / "hub"
        hub.mkdir()
        name = repo_folder_name(repo_id="org/name", repo_type="model")
        os.symlink(other / name, hub / name)
        assert qc.lookup_commit("org/name", "transformer", None, hub) == COMMIT
        assert (
            qc.save_commit(nn.Linear(1, 1), "org/name", "transformer", cache_dir=hub)
            == COMMIT
        )

    def test_hub_root_linked_to_another_disk(self, tmp_path):
        make_snapshot(tmp_path / "real", subfolder="vae")
        os.symlink(tmp_path / "real", tmp_path / "hub")
        assert qc.lookup_commit("org/name", "vae", None, tmp_path / "hub") == COMMIT

    def test_blob_outside_the_repo_folder_is_not_complete(self, tmp_path):
        folder = make_snapshot(tmp_path / "hub", files=())
        blob = tmp_path / "elsewhere.json"
        blob.write_text(json.dumps({"weight_map": {}}))
        os.symlink(blob, folder / "model.safetensors.index.json")
        assert qc.lookup_commit("org/name", None, None, tmp_path / "hub") is None

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

    def test_recorded_with_no_ref_is_unpinned(self, tmp_path):
        # no lookup could name it, so a save under it is never read
        assert (
            qc.save_commit(self.loaded(COMMIT), "org/name", cache_dir=tmp_path) is None
        )

    def test_recorded_over_an_incomplete_snapshot_is_unpinned(self, tmp_path):
        make_snapshot(tmp_path, files=())
        assert (
            qc.save_commit(self.loaded(COMMIT), "org/name", cache_dir=tmp_path) is None
        )

    @pytest.mark.parametrize("recorded", [True, False])
    @pytest.mark.parametrize("complete", [True, False])
    def test_a_save_is_always_one_a_lookup_reads(self, tmp_path, recorded, complete):
        make_snapshot(
            tmp_path, subfolder="vae", files=("model.safetensors",) if complete else ()
        )
        component = self.loaded(COMMIT) if recorded else nn.Linear(1, 1)
        saved = qc.save_commit(component, "org/name", "vae", cache_dir=tmp_path)
        looked_up = qc.lookup_commit("org/name", "vae", None, tmp_path)
        assert saved is None or saved == looked_up

    def test_recorded_in_a_dict_config(self, tmp_path):
        make_snapshot(tmp_path)
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
        with pytest.raises(ValueError):
            qc.load_entry(str(tmp_path), KEY, nn.Linear, None, "cpu", {})
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
        misses = [e for e in events if e["result"] == "miss"]
        assert {e["reason"] for e in misses} == {"load_failed: OSError: bad"}
        # the bad entry went, so the save is a fresh one, not a lost race
        saved = [e for e in events if e["result"] == "saved"]
        assert saved and all("reason" not in e for e in saved)

    def test_a_miss_names_why_nothing_was_read(self, tmp_path, events, cache_on):
        hub = tmp_path / "hub"
        make_snapshot(hub, subfolder="vae")
        make_snapshot(hub, subfolder="text_encoder", files=())
        qc.load_components(self.pipeline(), self.arguments(hub))
        reasons = {
            e["component"]: e.get("reason") for e in events if e["result"] == "miss"
        }
        assert reasons == {"vae": "no_entry", "text_encoder": "incomplete_snapshot"}
        assert ("text_encoder", "skipped") in results(events)

    def test_a_repo_linked_off_the_hub_hits_on_the_second_load(
        self, tmp_path, events, cache_on, monkeypatch
    ):
        # the bounce: run 1 saved under the recorded commit, and run 2's
        # lookup, refused by the containment check, never read it
        other = tmp_path / "other"
        for sub in ("vae", "text_encoder"):
            make_snapshot(other, subfolder=sub)
        hub = tmp_path / "hub"
        hub.mkdir()
        name = repo_folder_name(repo_id="org/name", repo_type="model")
        os.symlink(other / name, hub / name)

        class Recording(FakePipeline):
            def load_components(self, names=None, **kwargs):
                super().load_components(names=names, **kwargs)
                for name in self.loaded:
                    getattr(self, name).config = SimpleNamespace(_commit_hash=COMMIT)

        specs = [make_spec("vae"), make_spec("text_encoder")]
        qc.load_components(Recording(specs), self.arguments(hub))
        assert [r for r in results(events) if r[1] == "saved"] == [
            ("vae", "saved"),
            ("text_encoder", "saved"),
        ]
        events.clear()
        monkeypatch.setattr(sdnq, "load_sdnq_model", lambda *a, **k: nn.Linear(4, 4))
        qc.load_components(Recording(specs), self.arguments(hub))
        assert results(events) == [("vae", "hit"), ("text_encoder", "hit")]

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

    def test_a_load_that_writes_into_its_config_still_hits_next_time(
        self, hub, events, cache_on, monkeypatch
    ):
        # sdnq appends the modules it left unquantized to the config's own
        # modules_to_not_convert as it loads, and which ones can vary by
        # card: the key must be the config as the workflow gave it
        class WritingPipeline(FakePipeline):
            def __init__(self, specs, card):
                super().__init__(specs)
                self.card = card

            def load_components(self, **kwargs):
                config = kwargs["quantization_config"]
                config.modules_to_not_convert.append(f"skipped_on_{self.card}")
                super().load_components(**kwargs)

        def load(card):
            pipeline = WritingPipeline([make_spec("vae")], card)
            qc.load_components(pipeline, self.arguments(hub))
            return pipeline

        monkeypatch.setattr(sdnq, "load_sdnq_model", lambda *a, **k: nn.Linear(4, 4))
        load("cuda:0")
        assert results(events) == [("vae", "miss"), ("vae", "saved")]
        events.clear()

        pipeline = load("cuda:1")
        assert results(events) == [("vae", "hit")]
        assert pipeline.loaded == []
        entries = [n for n in os.listdir(cache_on) if qc._KEY_PATTERN.match(n)]
        assert len(entries) == 1

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
        arguments = qc.config_snapshot(config)
        quantized = UNet2DModel.from_pretrained(
            source, quantization_config=config, torch_dtype=torch.float32
        )
        # the quantizer really ran: something is stored as int8
        assert any(p.dtype == torch.int8 for p in quantized.state_dict().values())
        # and it wrote into the config it was given, which the key read
        # before the load does not see
        assert config.modules_to_not_convert
        assert inputs["quantizer"]["arguments"]["modules_to_not_convert"] == []
        assert qc.quantizer_arguments(config) != inputs["quantizer"]

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


# 13. The per-component path (#827)


class PlacedModule(nn.Linear):
    """A loaded component that records where it was put."""

    def __init__(self):
        super().__init__(4, 4)
        self.placed_on = []

    def to(self, *args, **kwargs):
        self.placed_on.append(args[0] if args else kwargs.get("device"))
        return self


class TestFromPretrained:
    @pytest.fixture
    def hub(self, tmp_path):
        hub = tmp_path / "hub"
        make_snapshot(hub, subfolder="transformer")
        return hub

    @pytest.fixture(autouse=True)
    def _fakes(self, monkeypatch):
        monkeypatch.setattr(qc.shutil, "disk_usage", disk(1000, 900))
        monkeypatch.setattr(sdnq, "save_sdnq_model", fake_save)

    def arguments(self, hub, **extra):
        return {
            "subfolder": "transformer",
            "torch_dtype": torch.bfloat16,
            "cache_dir": str(hub),
            "quantization_config": SDNQConfig(
                weights_dtype="uint4",
                quantization_device="cuda",
                return_device="cuda",
                use_quantized_matmul=True,
            ),
            **extra,
        }

    def load(self, hub, loader, **extra):
        return qc.from_pretrained(
            "transformer", nn.Linear, "org/name", self.arguments(hub, **extra), loader
        )

    def test_cache_off_is_a_plain_passthrough(self, hub, events, tmp_path):
        fresh = nn.Linear(4, 4)
        loader = MagicMock(return_value=fresh)
        assert self.load(hub, loader) is fresh
        loader.assert_called_once_with()
        assert events == []
        assert not (tmp_path / "cache").exists()

    def test_a_load_no_sdnq_config_quantizes_is_untouched(self, hub, events, cache_on):
        fresh = nn.Linear(4, 4)
        loader = MagicMock(return_value=fresh)
        arguments = self.arguments(hub)
        del arguments["quantization_config"]
        assert (
            qc.from_pretrained("vae", nn.Linear, "org/name", arguments, loader) is fresh
        )
        assert events == []

    def test_a_miss_loads_and_saves(self, hub, events, cache_on):
        fresh = nn.Linear(4, 4)
        loader = MagicMock(return_value=fresh)
        assert self.load(hub, loader) is fresh
        loader.assert_called_once_with()
        assert results(events) == [("transformer", "miss"), ("transformer", "saved")]
        assert events[0]["reason"] == qc.NO_ENTRY
        assert events[0]["bytes"] == qc.component_nbytes(fresh)
        assert events[1]["bytes"] == 100
        (entry,) = [n for n in os.listdir(cache_on) if qc._KEY_PATTERN.match(n)]
        stored = qc.read_entry(str(cache_on / entry))
        assert stored["model_name"] == "org/name"
        assert stored["subfolder"] == "transformer"
        assert stored["commit"] == COMMIT
        assert "return_device" not in stored["quantizer"]["arguments"]

    def test_a_hit_skips_the_load_and_is_placed_on_return_device(
        self, hub, events, cache_on, monkeypatch
    ):
        self.load(hub, MagicMock(return_value=nn.Linear(4, 4)))
        events.clear()

        cached = PlacedModule()
        sdnq_load = MagicMock(return_value=cached)
        monkeypatch.setattr(sdnq, "load_sdnq_model", sdnq_load)
        loader = MagicMock()
        assert self.load(hub, loader) is cached

        loader.assert_not_called()
        assert results(events) == [("transformer", "hit")]
        assert events[0]["bytes"] == qc.component_nbytes(cached)
        kwargs = sdnq_load.call_args.kwargs
        assert kwargs["model_cls"] is nn.Linear
        assert kwargs["dtype"] is torch.bfloat16
        assert kwargs["device"] == "cuda"
        assert kwargs["use_quantized_matmul"] is True
        assert cached.placed_on == ["cuda"]

    def test_a_different_weights_dtype_misses(self, hub, events, cache_on):
        self.load(hub, MagicMock(return_value=nn.Linear(4, 4)))
        events.clear()
        arguments = self.arguments(hub)
        arguments["quantization_config"] = SDNQConfig(weights_dtype="int8")
        qc.from_pretrained(
            "transformer", nn.Linear, "org/name", arguments, lambda: nn.Linear(4, 4)
        )
        assert results(events) == [("transformer", "miss"), ("transformer", "saved")]

    def test_a_corrupt_entry_falls_back_to_a_fresh_load(
        self, hub, events, cache_on, monkeypatch
    ):
        self.load(hub, MagicMock(return_value=nn.Linear(4, 4)))
        events.clear()
        monkeypatch.setattr(
            sdnq, "load_sdnq_model", MagicMock(side_effect=ValueError("corrupt"))
        )
        fresh = nn.Linear(4, 4)
        assert self.load(hub, MagicMock(return_value=fresh)) is fresh
        assert results(events) == [("transformer", "miss"), ("transformer", "saved")]
        assert events[0]["reason"] == f"{qc.LOAD_FAILED}: ValueError: corrupt"

    def test_a_resharded_refs_main_with_missing_shards_gives_no_lookup(
        self, tmp_path, events, cache_on, monkeypatch
    ):
        # The LTX-2.5 re-shard: refs/main names a commit whose index lists
        # shards that are not on disk
        hub = tmp_path / "hub"
        folder = make_snapshot(
            hub, subfolder="transformer", files=("m-1-of-2.safetensors",)
        )
        index = {
            "weight_map": {"a": "m-1-of-2.safetensors", "b": "m-2-of-2.safetensors"}
        }
        (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps(index)
        )
        sdnq_load = MagicMock()
        monkeypatch.setattr(sdnq, "load_sdnq_model", sdnq_load)
        lookup = MagicMock(wraps=qc.load_entry)
        monkeypatch.setattr(qc, "load_entry", lookup)
        fresh = nn.Linear(4, 4)

        assert self.load(hub, MagicMock(return_value=fresh)) is fresh

        lookup.assert_not_called()
        sdnq_load.assert_not_called()
        assert results(events) == [
            ("transformer", "miss"),
            ("transformer", "skipped"),
        ]
        assert events[0]["reason"] == qc.INCOMPLETE_SNAPSHOT
        assert events[1]["reason"] == qc.UNPINNED_SNAPSHOT
        assert not cache_on.exists() or os.listdir(cache_on) == []

    def test_a_local_path_is_skipped(self, tmp_path, events, cache_on):
        fresh = nn.Linear(4, 4)
        result = qc.from_pretrained(
            "transformer",
            nn.Linear,
            str(tmp_path),
            self.arguments(tmp_path),
            lambda: fresh,
        )
        assert result is fresh
        assert results(events) == [("transformer", "skipped")]
        assert events[0]["reason"] == qc.LOCAL_PATH

    def test_a_keyword_the_key_does_not_cover_is_skipped(self, hub, events, cache_on):
        fresh = nn.Linear(4, 4)
        loader = MagicMock(return_value=fresh)
        assert self.load(hub, loader, ignore_mismatched_sizes=True) is fresh
        loader.assert_called_once_with()
        assert results(events) == [("transformer", "skipped")]
        assert events[0]["reason"] == qc.UNSUPPORTED

    def test_a_variant_is_skipped(self, hub, events, cache_on):
        self.load(hub, lambda: nn.Linear(4, 4), variant="fp16")
        assert results(events) == [("transformer", "skipped")]
        assert events[0]["reason"] == qc.UNSUPPORTED


class TestPerComponentCallSite:
    """load_component, the path load_optional_component takes, end to end."""

    def configuration(self, component_type):
        return {"component_type": component_type, "preserve_device_placement": True}

    def test_download_watch_is_used_on_a_miss_and_skipped_on_a_hit(self, cache_on):
        from dw.pipeline_processors import components

        component_type = MagicMock()
        component_type.__name__ = "MockTransformer"
        component_type.from_pretrained.return_value = nn.Linear(4, 4)
        cached = nn.Linear(4, 4)
        lookups = iter([(None, qc.NO_ENTRY), (cached, None)])

        with (
            patch.object(components.download_watch, "watch") as watch,
            patch.object(qc, "_lookup", side_effect=lambda *a: next(lookups)),
            patch.object(qc, "_save"),
            patch.object(qc, "is_local_path", return_value=False),
        ):
            for _ in range(2):
                components.load_component(
                    "transformer",
                    self.configuration(component_type),
                    {
                        "model_name": "org/name",
                        "subfolder": "transformer",
                        "quantization_config": SDNQConfig(weights_dtype="uint4"),
                    },
                    "cpu",
                )

        # the miss loaded under the watch, the hit reached neither
        watch.assert_called_once()
        assert watch.call_args.args == ("org/name",)
        component_type.from_pretrained.assert_called_once()

    def test_the_cache_off_loads_under_the_watch(self):
        from dw.pipeline_processors import components

        component_type = MagicMock()
        component_type.__name__ = "MockTransformer"
        with patch.object(components.download_watch, "watch") as watch:
            components.load_component(
                "transformer",
                self.configuration(component_type),
                {
                    "model_name": "org/name",
                    "quantization_config": SDNQConfig(weights_dtype="uint4"),
                },
                "cpu",
            )
        watch.assert_called_once()
        component_type.from_pretrained.assert_called_once_with(
            "org/name", quantization_config=ANY_CONFIG
        )

    def test_a_real_component_misses_saves_then_hits_with_the_same_output(
        self, tmp_path, cache_on, events
    ):
        # A real diffusers model in a real hub-cache layout, quantized by sdnq
        # on the first load and read back from the cache on the second
        from diffusers.models import UNet2DModel

        from dw.pipeline_processors import components

        hub = tmp_path / "hub"
        folder = make_snapshot(hub, subfolder="unet", files=(), config=False)
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
        ).save_pretrained(folder)

        def load():
            return components.load_component(
                "unet",
                {"component_type": UNet2DModel},
                {
                    "model_name": "org/name",
                    "subfolder": "unet",
                    "cache_dir": str(hub),
                    "local_files_only": True,
                    "torch_dtype": torch.float32,
                    "quantization_config": SDNQConfig(
                        weights_dtype="int8",
                        quant_conv=True,
                        minimum_allowed_numel=256,
                        minimum_allowed_channel_size=8,
                        return_device="cpu",
                    ),
                },
                "cpu",
            )

        first = load()
        assert results(events) == [("unet", "miss"), ("unet", "saved")]
        events.clear()
        with patch.object(UNet2DModel, "from_pretrained") as fresh:
            second = load()
        fresh.assert_not_called()
        assert results(events) == [("unet", "hit")]
        assert any(p.dtype == torch.int8 for p in second.state_dict().values())

        sample = torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
        step = torch.tensor([1])
        with torch.no_grad():
            expected = first(sample, step).sample
            actual = second(sample, step).sample
        assert torch.allclose(expected, actual, atol=1e-4)


class _AnyConfig:
    def __eq__(self, other):
        return qc.is_sdnq_config(other)


ANY_CONFIG = _AnyConfig()


class TestTransformersRoundTrip:
    """An entry is saved under the model's own tensor names, so a transformers
    model with a weight conversion mapping loads back (#827: LTX-2.5's Gemma 4
    text encoder saved in its checkpoint layout and failed every lookup)."""

    @staticmethod
    def gemma4():
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "Gemma4UnifiedForConditionalGeneration"):
            pytest.skip("this transformers has no Gemma 4")
        config = transformers.Gemma4UnifiedConfig(
            text_config=dict(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                global_head_dim=32,
                vocab_size=512,
                hidden_size_per_layer_input=16,
                vocab_size_per_layer_input=512,
                num_kv_shared_layers=0,
            ),
            vision_config=dict(
                mm_embed_dim=64, mm_posemb_size=32, output_proj_dims=64, patch_size=4
            ),
            audio_config=dict(audio_embed_dim=32),
        )
        return transformers.Gemma4UnifiedForConditionalGeneration, config

    def test_a_model_with_a_conversion_mapping_saves_and_loads_back(self, tmp_path):
        model_cls, config = self.gemma4()
        torch.manual_seed(0)
        source = tmp_path / "source"
        model_cls(config).to(torch.bfloat16).save_pretrained(source)
        arguments = {
            "weights_dtype": "int8",
            "dequantize_fp32": False,
            "modules_to_not_convert": ["embed_vision", "embed_audio", "lm_head"],
            "minimum_allowed_numel": 256,
            "minimum_allowed_channel_size": 8,
        }
        quantized = model_cls.from_pretrained(
            source,
            quantization_config=SDNQConfig(**arguments),
            torch_dtype=torch.bfloat16,
        )
        root = tmp_path / "cache"
        result, reason, _ = qc.save_entry(str(root), KEY, quantized, {})
        assert (result, reason) == ("saved", None)
        # the save left the model's own save_pretrained in place
        assert "save_pretrained" not in vars(quantized)

        loaded = qc.load_entry(
            str(root), KEY, model_cls, torch.bfloat16, "cpu", arguments
        )
        expected = quantized.state_dict()
        actual = loaded.state_dict()
        assert set(actual) == set(expected)
        assert any(t.dtype == torch.int8 for t in actual.values())
        for name, tensor in expected.items():
            assert torch.equal(actual[name], tensor), name

    def test_a_load_failure_reason_carries_the_first_line_of_the_message(self):
        error = RuntimeError("Error(s) in loading state_dict:\n\tMissing key(s)")
        assert qc._error_detail(error) == (
            "RuntimeError: Error(s) in loading state_dict:"
        )
        long = qc._error_detail(ValueError("x" * 500))
        assert long.startswith("ValueError: x") and long.endswith("...")
        assert len(long) == len("ValueError: ") + qc._REASON_DETAIL_CHARS
        assert qc._error_detail(OSError()) == "OSError"
