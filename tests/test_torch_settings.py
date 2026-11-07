"""The PyTorch settings reach every process that runs a pipeline (#1229).

`configure_torch` applies `enable_tf32`, `cudnn_benchmark` and
`cudnn_deterministic`; the served worker calls it itself (startup() runs only
in the parent), and narrates what it ran with once per process. The defaults
make same-seed output reproducible: deterministic on, benchmark and TF32 off.
"""

import json
import queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import dw
from dw.settings import Settings, get_settings_full_path, load_settings


class _FakeBackends:
    """Stands in for torch.backends, so no test here needs CUDA."""

    def __init__(self):
        self.cudnn = SimpleNamespace(benchmark=None, deterministic=None, enabled=None)
        self.cuda = SimpleNamespace(matmul=SimpleNamespace(allow_tf32=None))


@pytest.fixture
def backends(monkeypatch):
    fake = _FakeBackends()
    monkeypatch.setattr(dw.torch, "backends", fake)
    precisions = []
    monkeypatch.setattr(dw.torch, "set_float32_matmul_precision", precisions.append)
    fake.precisions = precisions
    return fake


def _settings(**overrides):
    values = dict(enable_tf32=False, cudnn_benchmark=False, cudnn_deterministic=True)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("value", [True, False])
def test_configure_torch_sets_cudnn_benchmark_from_settings(backends, value):
    dw.configure_torch(_settings(cudnn_benchmark=value), "cuda")
    assert backends.cudnn.benchmark is value


@pytest.mark.parametrize("value", [True, False])
def test_configure_torch_sets_cudnn_deterministic_from_settings(backends, value):
    dw.configure_torch(_settings(cudnn_deterministic=value), "cuda")
    assert backends.cudnn.deterministic is value
    assert backends.cudnn.enabled is True


@pytest.mark.parametrize("value", [True, False])
def test_configure_torch_sets_tf32_from_settings(backends, value):
    dw.configure_torch(_settings(enable_tf32=value), "cuda")
    assert backends.cuda.matmul.allow_tf32 is value
    assert backends.precisions == ["high" if value else "highest"]


def test_configure_torch_leaves_cudnn_alone_off_cuda(backends):
    dw.configure_torch(_settings(cudnn_benchmark=True), "mps")
    assert backends.cudnn.benchmark is None
    assert backends.cudnn.deterministic is None


def test_report_reads_the_flags_back_from_torch(backends):
    backends.cudnn.benchmark = True
    backends.cudnn.deterministic = False
    backends.cuda.matmul.allow_tf32 = False
    assert dw.torch_flags_report() == (
        "torch: cudnn benchmark=true deterministic=false, tf32=false"
    )


def test_defaults_make_output_reproducible():
    assert Settings.cudnn_deterministic is True
    assert Settings.cudnn_benchmark is False
    assert Settings.enable_tf32 is False


def test_get_fallbacks_agree_with_the_class_defaults():
    # No settings.json in the isolated settings dir (tests/conftest.py)
    assert not get_settings_full_path().exists()
    loaded = load_settings()
    for key in ("enable_tf32", "cudnn_benchmark", "cudnn_deterministic"):
        assert getattr(loaded, key) == getattr(Settings, key), key


def test_an_explicit_settings_key_overrides_the_default():
    get_settings_full_path().write_text(
        json.dumps(
            {"enable_tf32": True, "cudnn_benchmark": True, "cudnn_deterministic": False}
        )
    )
    loaded = load_settings()
    assert loaded.enable_tf32 is True
    assert loaded.cudnn_benchmark is True
    assert loaded.cudnn_deterministic is False


def _make_worker():
    with patch("dw.worker.setup_logging"), patch(
        "dw.worker.get_device_type", return_value="cuda"
    ), patch("dw.worker.configure_torch") as configure:
        from dw.worker import WorkflowWorker

        worker = WorkflowWorker(queue.Queue(), queue.Queue())
    return worker, configure


def test_worker_initialisation_applies_the_torch_settings():
    worker, configure = _make_worker()
    configure.assert_called_once()
    applied_settings, device_type = configure.call_args.args
    assert device_type == "cuda"
    assert applied_settings.cudnn_deterministic is Settings.cudnn_deterministic


def test_worker_initialisation_applies_a_settings_json_key():
    get_settings_full_path().write_text(json.dumps({"cudnn_deterministic": False}))
    _, configure = _make_worker()
    assert configure.call_args.args[0].cudnn_deterministic is False


class _StubWorkflow:
    name = "stub"
    manifest = []

    def run(self, arguments, previous_pipelines=None, context=None, **_):
        return []


def _drain(result_queue):
    messages = []
    while True:
        try:
            messages.append(result_queue.get_nowait())
        except queue.Empty:
            return messages


def _execute(worker):
    command = {
        "definition": {"id": "stub", "steps": []},
        "file_spec": "/w/x.json",
        "source": "path",
        "arguments": {},
        "output_dir": "/tmp",
    }
    with patch("dw.worker.workflow_from_snapshot", return_value=_StubWorkflow()):
        worker._handle_execute(command)
    return _drain(worker.result_queue)


def _flag_lines(messages):
    return [
        m
        for m in messages
        if m.get("type") == "progress"
        and m.get("event") == "log"
        and str(m.get("message", "")).startswith("torch: cudnn")
    ]


def test_first_job_narrates_the_flags_once_per_process(backends):
    worker, _ = _make_worker()
    backends.cudnn.benchmark = False
    backends.cudnn.deterministic = True
    backends.cuda.matmul.allow_tf32 = False

    with patch("dw.worker.configure_torch") as reapply:
        first = _execute(worker)
        second = _execute(worker)

    lines = _flag_lines(first)
    assert [m["message"] for m in lines] == [
        "torch: cudnn benchmark=false deterministic=true, tf32=false"
    ]
    assert _flag_lines(second) == []
    # Applied again at each job start, in case an import moved them
    assert reapply.call_count == 2
