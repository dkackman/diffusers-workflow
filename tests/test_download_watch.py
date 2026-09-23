"""A download from_pretrained triggers is watched, not intercepted (#343):
byte growth in the Hugging Face cache counts as progress, so a real download
does not trip the phase-stall watchdog, and a download that genuinely stops
still does."""

import time
from unittest.mock import patch

from dw import download_watch
from dw import events as events_module
from dw.events import RunContext


def _fast_watchdog():
    return patch.multiple(
        events_module,
        PHASE_STALL_THRESHOLD_SECONDS=0.1,
        PHASE_STALL_CHECK_INTERVAL_SECONDS=0.02,
    )


def _fast_watch(monkeypatch):
    monkeypatch.setattr(download_watch, "CHECK_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(download_watch, "EMIT_INTERVAL_SECONDS", 0.05)


def test_growing_download_emits_progress_and_suppresses_stall(tmp_path, monkeypatch):
    _fast_watch(monkeypatch)
    repo_id = "some-org/some-model"
    blob_dir = tmp_path / "models--some-org--some-model" / "blobs"
    blob_dir.mkdir(parents=True)
    blob_file = blob_dir / "abc123.incomplete"

    events = []
    context = RunContext(on_event=events.append)
    with _fast_watchdog():
        context.enter_run()
        try:
            context.note_phase("loading")
            with download_watch.DownloadWatch(
                repo_id, context, cache_dir=str(tmp_path)
            ):
                for _ in range(6):
                    with open(blob_file, "ab") as f:
                        f.write(b"x" * 4096)
                    time.sleep(0.06)
        finally:
            context.exit_run()

    progress = [e for e in events if e["event"] == "download_progress"]
    assert progress, "growing bytes must be reported as download_progress"
    assert progress[0]["repo_id"] == repo_id
    assert all(e["downloaded_bytes"] > 0 for e in progress)

    stalls = [e for e in events if e.get("kind") == "phase_stall"]
    assert not stalls, "a real, ongoing download must not read as a stall"


def test_stalled_download_still_stalls(tmp_path, monkeypatch):
    _fast_watch(monkeypatch)
    repo_id = "some-org/some-model"
    blob_dir = tmp_path / "models--some-org--some-model" / "blobs"
    blob_dir.mkdir(parents=True)
    blob_file = blob_dir / "abc123.incomplete"
    blob_file.write_bytes(b"x" * 4096)  # written once, never grows again

    events = []
    context = RunContext(on_event=events.append)
    with _fast_watchdog():
        context.enter_run()
        try:
            context.note_phase("loading")
            with download_watch.DownloadWatch(
                repo_id, context, cache_dir=str(tmp_path)
            ):
                time.sleep(0.3)
        finally:
            context.exit_run()

    assert not [e for e in events if e["event"] == "download_progress"]
    stalls = [e for e in events if e.get("kind") == "phase_stall"]
    assert stalls, "bytes that stopped growing must still read as a stall"
