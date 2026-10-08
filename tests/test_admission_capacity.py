"""Review 2026-10-07 #2: a declared vram_estimate is held to the pool's
largest card, not the process's default device (card 0 under --devices)."""

from types import SimpleNamespace

from dw.server import admission
from dw.validation import workflow_context
from dw.workflow import workflow_from_definition


def empty_workflow(tmp_path):
    return workflow_from_definition({"id": "w", "steps": []}, str(tmp_path / "out"))


def test_the_context_defaults_to_the_devices_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr("dw.validation.device_capacity_gb", lambda: 12.0)
    assert workflow_context(empty_workflow(tmp_path)).capacity_gb == 12.0


def test_the_context_takes_a_pool_ceiling_instead(tmp_path, monkeypatch):
    monkeypatch.setattr("dw.validation.device_capacity_gb", lambda: 12.0)
    assert workflow_context(empty_workflow(tmp_path), capacity_gb=24.0).capacity_gb == 24.0


def _patched_admit_for(monkeypatch, state):
    seen = {}
    monkeypatch.setattr(admission, "admit", lambda **kw: seen.update(kw))
    monkeypatch.setattr(admission, "ceiling_index", lambda state, workspace: None)
    monkeypatch.setattr(admission, "resolution_library", lambda state, workspace: None)
    monkeypatch.setattr(admission, "server_prompt_library", lambda state: None)
    admission.admit_for(state, "workspace")
    return seen


def test_admit_for_hands_the_pools_largest_ceiling_to_admit(monkeypatch):
    manager = SimpleNamespace(largest_ceiling_gb=lambda: 24.0)
    seen = _patched_admit_for(monkeypatch, SimpleNamespace(job_manager=manager))
    assert seen["capacity_gb"] == 24.0


def test_an_unread_pool_falls_back_to_the_device(monkeypatch):
    """A lazily started worker has not read its card yet: None here means
    workflow_context's own default, not a refusal of everything."""
    manager = SimpleNamespace(largest_ceiling_gb=lambda: None)
    seen = _patched_admit_for(monkeypatch, SimpleNamespace(job_manager=manager))
    assert seen["capacity_gb"] is None


def test_a_state_without_a_manager_falls_back_to_the_device(monkeypatch):
    seen = _patched_admit_for(monkeypatch, SimpleNamespace())
    assert seen["capacity_gb"] is None
