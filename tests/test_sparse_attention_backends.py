"""The backends the server reports it can run (#931): the UI's dropdown."""

import pytest
import torch

from dw import sparse_attention
from dw.acceleration_warnings import CUDA_ONLY_ATTENTION_BACKENDS
from dw.sparse_attention import available_attention_backends

from tests.test_server import server, success_script  # noqa: F401


@pytest.fixture(autouse=True)
def fresh_kernel_probe():
    """The probe is cached for the process: each test starts and ends cold,
    so one that patches the import sees its own answer, and leaves none.
    Held here, since a test may have swapped the module's name by teardown."""
    probe = sparse_attention._kernels_importable
    probe.cache_clear()
    yield
    probe.cache_clear()


def test_mps_lists_no_cuda_backends():
    names = available_attention_backends("mps")
    assert "native" in names and "sparse" not in names and "sage_hub" not in names
    assert "_native_cudnn" not in names
    assert not set(names) & CUDA_ONLY_ATTENTION_BACKENDS


def test_cuda_lists_sparse_when_kernels_present(monkeypatch):
    monkeypatch.setattr(sparse_attention, "_kernels_importable", lambda: True)
    names = available_attention_backends("cuda")
    assert names[0] == "native"
    assert "sparse" in names and "_native_cudnn" in names
    # sparse comes before the sorted CUDA-only names
    assert names[names.index("sparse") + 1 :] == sorted(CUDA_ONLY_ATTENTION_BACKENDS)


def test_cuda_omits_sparse_without_kernels(monkeypatch):
    monkeypatch.setattr(sparse_attention, "_kernels_importable", lambda: False)
    names = available_attention_backends("cuda")
    assert "sparse" not in names and "sage_hub" in names


def test_names_are_all_diffusers_backends(monkeypatch):
    from diffusers.models.attention_dispatch import AttentionBackendName

    monkeypatch.setattr(sparse_attention, "_kernels_importable", lambda: True)
    known = {b.value for b in AttentionBackendName}
    assert set(available_attention_backends("cuda")) - {"sparse"} <= known


def test_listing_never_initializes_cuda():
    available_attention_backends("cuda")
    assert not torch.cuda.is_initialized()


def test_listing_never_touches_the_card(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the listing touched CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)
    monkeypatch.setattr(torch.cuda, "init", boom)
    monkeypatch.setattr(sparse_attention, "_kernels_importable", lambda: True)
    assert "sparse" in available_attention_backends("cuda")


def test_kernel_probe_swallows_every_error(monkeypatch):
    import builtins

    real = builtins.__import__

    def failing(name, *args, **kwargs):
        if name.startswith("comfy_kitchen"):
            raise RuntimeError("broken install")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing)
    assert sparse_attention._kernels_importable() is False


def test_kernel_probe_imports_once(monkeypatch):
    """The first health poll pays for the import; later ones do not."""
    import builtins

    real = builtins.__import__
    imports = []

    def counting(name, *args, **kwargs):
        if name.startswith("comfy_kitchen"):
            imports.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", counting)
    first = sparse_attention._kernels_importable()
    assert sparse_attention._kernels_importable() is first
    assert len(imports) == 1


def test_health_carries_attention_backends(server):  # noqa: F811
    with server(success_script) as client:
        body = client.get("/api/health").json()
    assert "native" in body["attention_backends"]
    assert isinstance(body["attention_backends"], list)
