"""A download from_pretrained triggers is watched, not intercepted (#343).

What the tester held the earlier fixes to, and what these pin:

- bytes arriving count as progress, so a real download does not trip the
  phase-stall watchdog - measured through the hub's own xet progress
  reporter, which counts network bytes while hf_xet holds the file on disk
  unchanged (on lem, 67 MB written while 700 MB had arrived);
- bytes that stop arriving *do* stall: a flat download still reports, so
  phase_detail reads `no bytes for Ns`, but the watchdog sees the silence;
- `downloaded_bytes` never goes backwards when a finished blob is published
  into the shared store and replaced by a symlink;
- the size labels are the decimal units they claim to be;
- a cancel during a download aborts it rather than waiting it out.

The reporter driven here is huggingface_hub's real
`XetDownloadProgressReporter`, fed the group reports hf_xet would send - not
a stub of DownloadWatch's own counting.
"""

import logging
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from huggingface_hub.utils import _xet_progress_reporting as xet_reporting
from huggingface_hub.utils.tqdm import tqdm as hub_tqdm

from dw import download_watch
from dw import events as events_module
from dw.download_watch import format_progress
from dw.events import RunContext, WorkflowCancelled

REPO_ID = "some-org/some-model"


# Scaled down from production's 1s check / 5s emit / 30s stall, keeping the
# margin between emit and stall wide enough that scheduler jitter cannot
# open a stall-sized gap between two growing progress events.
def _fast_watchdog():
    return patch.multiple(
        events_module,
        PHASE_STALL_THRESHOLD_SECONDS=0.4,
        PHASE_STALL_CHECK_INTERVAL_SECONDS=0.02,
    )


@pytest.fixture(autouse=True)
def _fast_watch(monkeypatch):
    monkeypatch.setattr(download_watch, "CHECK_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(download_watch, "EMIT_INTERVAL_SECONDS", 0.05)


def _group_report(transferred, written, total):
    return SimpleNamespace(
        total_bytes_completed=written,
        total_transfer_bytes_completed=transferred,
        total_bytes=total,
        total_bytes_completion_rate=None,
        total_transfer_bytes_completion_rate=None,
    )


def _reporter(total):
    return xet_reporting.XetDownloadProgressReporter(
        reconstruction_desc="model.safetensors: reconstructing file",
        transfer_desc="model.safetensors: downloading bytes",
        total=total,
        log_level=logging.INFO,
        name="huggingface_hub.xet_get",
    )


def _run_loading(events, body, cache_dir):
    context = RunContext(on_event=events.append)
    with _fast_watchdog():
        context.enter_run()
        try:
            context.note_phase("loading")
            with download_watch.DownloadWatch(REPO_ID, context, cache_dir=cache_dir):
                body(context)
        finally:
            context.exit_run()
    return context


def _progress(events):
    return [e for e in events if e["event"] == "download_progress"]


def _stalls(events):
    return [e for e in events if e.get("kind") == "phase_stall"]


def test_xet_transfer_counts_while_the_file_on_disk_is_flat(tmp_path):
    # hf_xet's real shape: network bytes climb steadily while the bytes
    # written to disk sit still until the end - the disk alone reads as a hang
    total = 3_000_000_000
    events = []

    def body(_context):
        reporter = _reporter(total)
        for i in range(1, 25):
            reporter.update_progress(_group_report(i * 20_000_000, 67_000_000, total))
            time.sleep(0.04)
        reporter.update_progress(_group_report(480_000_000, total, total))

    _run_loading(events, body, str(tmp_path))

    progress = _progress(events)
    assert progress, "transfer bytes must be reported as download_progress"
    assert progress[0]["repo_id"] == REPO_ID
    sizes = [e["downloaded_bytes"] for e in progress]
    assert sizes == sorted(sizes)
    # Climbs well past the 67 MB on disk, and in steps, not one jump at the end
    assert max(sizes) >= 300_000_000
    assert len(set(sizes)) >= 3
    assert all(e["bytes_per_second"] >= 0 for e in progress)
    assert not _stalls(events), "a download receiving bytes must not stall"


def test_bytes_that_stop_arriving_stall_and_say_so(tmp_path):
    events = []

    def body(_context):
        reporter = _reporter(3_000_000_000)
        for i in range(1, 6):
            reporter.update_progress(_group_report(i * 10_000_000, 0, 3_000_000_000))
            time.sleep(0.04)
        time.sleep(1.2)  # three stall thresholds with nothing arriving

    _run_loading(events, body, str(tmp_path))

    flat = [e for e in _progress(events) if e["bytes_per_second"] == 0]
    assert flat, "a flat download still reports, so phase_detail can say so"
    assert flat[-1]["seconds_since_bytes_changed"] >= 0.8
    assert flat[-1]["downloaded_bytes"] == 50_000_000
    assert "no bytes for" in format_progress(
        REPO_ID,
        flat[-1]["downloaded_bytes"],
        flat[-1]["bytes_per_second"],
        flat[-1]["seconds_since_bytes_changed"],
    )

    stalls = _stalls(events)
    assert stalls, "zero-growth reports must not keep the watchdog quiet"
    assert stalls[0]["phase"] == "loading"
    assert "download_progress" in stalls[0]["message"]


def test_publishing_a_blob_to_the_shared_store_does_not_shrink_the_total(tmp_path):
    # The HTTP path, on disk as huggingface_hub 1.32 lays it out: the file
    # grows as blobs/<etag>.incomplete, is renamed to blobs/<etag>, then moved
    # into <cache>/blobs/xx/<sha> and replaced by a symlink to it
    blob_dir = tmp_path / "models--some-org--some-model" / "blobs"
    blob_dir.mkdir(parents=True)
    shared = tmp_path / "blobs" / "13"
    shared.mkdir(parents=True)
    events = []

    def body(_context):
        incomplete = blob_dir / "etag1.a1b2c3d4.incomplete"
        for _ in range(8):
            with open(incomplete, "ab") as f:
                f.write(b"x" * 1_000_000)
            time.sleep(0.04)
        done = blob_dir / "etag1"
        os.rename(incomplete, done)
        target = shared / "1392abc"
        os.rename(done, target)
        os.symlink(os.path.relpath(target, blob_dir), done)
        time.sleep(0.2)

    _run_loading(events, body, str(tmp_path))

    progress = _progress(events)
    sizes = [e["downloaded_bytes"] for e in progress]
    assert sizes and sizes == sorted(sizes), f"downloaded_bytes went backwards: {sizes}"
    assert sizes[-1] == 8_000_000
    assert all(e["bytes_per_second"] >= 0 for e in progress)


def test_a_cached_load_reports_nothing(tmp_path):
    blob_dir = tmp_path / "models--some-org--some-model" / "blobs"
    blob_dir.mkdir(parents=True)
    (blob_dir / "etag1").write_bytes(b"x" * 4096)
    events = []
    _run_loading(events, lambda _context: time.sleep(0.2), str(tmp_path))
    assert not _progress(events)


def test_cancel_aborts_an_xet_download_and_reads_as_cancelled(tmp_path, monkeypatch):
    aborted = []
    monkeypatch.setattr(
        download_watch, "_abort_xet_downloads", lambda: aborted.append(True)
    )
    events = []

    def body(context):
        context.cancel()
        deadline = time.monotonic() + 2
        while not aborted and time.monotonic() < deadline:
            time.sleep(0.01)
        # What hf_xet raises in the loading thread once its session aborts
        raise RuntimeError("Operation cancelled: Task cancelled")

    with pytest.raises(WorkflowCancelled):
        _run_loading(events, body, str(tmp_path))
    assert aborted, "a cancel during loading must abort the in-flight download"


def test_cancel_stops_a_plain_http_download_at_its_next_chunk(tmp_path):
    events = []

    def body(context):
        bar = hub_tqdm(total=10, unit="B", disable=True)
        bar.update(1)
        context.cancel()
        bar.update(1)

    with pytest.raises(WorkflowCancelled):
        _run_loading(events, body, str(tmp_path))


def test_hooks_are_removed_when_the_last_watch_closes(tmp_path):
    reporter_method = vars(xet_reporting.XetDownloadProgressReporter)["update_progress"]
    had_update = "update" in vars(hub_tqdm)
    _run_loading([], lambda _context: None, str(tmp_path))
    assert (
        vars(xet_reporting.XetDownloadProgressReporter)["update_progress"]
        is reporter_method
    )
    assert ("update" in vars(hub_tqdm)) == had_update


def test_sizes_are_labelled_in_the_units_they_are():
    # The tester's own reading: 1,879,623,333 bytes rendered as "1.8 GB"
    assert download_watch._format_bytes(1_879_623_333) == "1.9 GB"
    assert download_watch._format_bytes(38_000_000) == "38.0 MB"
    assert (
        format_progress(REPO_ID, 12_300_000_000, 38_000_000)
        == f"downloading {REPO_ID}: 12.3 GB, 38.0 MB/s"
    )
    assert (
        format_progress(REPO_ID, 12_300_000_000, 0, 45.2)
        == f"downloading {REPO_ID}: 12.3 GB, no bytes for 45s"
    )


def test_a_non_progress_event_does_not_reset_the_watchdog():
    events = []
    context = RunContext(on_event=events.append)
    with _fast_watchdog():
        context.enter_run()
        try:
            context.note_phase("loading")
            for _ in range(40):
                context.emit("download_progress", counts_as_progress=False, x=1)
                time.sleep(0.02)
        finally:
            context.exit_run()
    assert _stalls(events)
    assert all("counts_as_progress" not in e for e in events)
