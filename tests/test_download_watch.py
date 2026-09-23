"""A download from_pretrained triggers is watched, not intercepted (#343):
byte growth in the Hugging Face cache counts as progress, so a real download
does not trip the phase-stall watchdog.

Progress ticks on a timer, not only on growth (#343 follow-up): an
xet-backed transfer reconstructs against its local CAS cache in bursts and
can leave the watched size unchanged on disk for well past the phase-stall
threshold while genuinely still running underneath - confirmed by polling a
real large download (stabilityai/sd-turbo's non-fp16 unet, ~3.5GB) directly,
which sat with an unchanged blob size for 20+ seconds more than once before
jumping hundreds of MB at a time. There is no on-disk signal, in the blob
directory or in HF_XET_CACHE, that fills that gap, so the trade this file
makes deliberately: a watched download cannot read as a stall for as long as
DownloadWatch's thread is alive, in exchange for never mistaking a
reconstruction pause for one. What that costs is real: an `from_pretrained`
call that hangs with no eventual timeout or exception, for the length of the
hang, does not raise a phase-stall warning either - orthogonal to what #343
reported, and a smaller loss than 17.8 minutes of false ones."""

import time
from unittest.mock import patch

from dw import download_watch
from dw import events as events_module
from dw.events import RunContext


# Scaled down from production's 5s emit / 30s stall, keeping the margin
# between them wide enough that scheduler jitter cannot open a stall-sized
# gap between two progress events. Each test still runs longer than the
# threshold, so silence would trip the watchdog.
def _fast_watchdog():
    return patch.multiple(
        events_module,
        PHASE_STALL_THRESHOLD_SECONDS=0.4,
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
                for _ in range(20):
                    with open(blob_file, "ab") as f:
                        f.write(b"x" * 4096)
                    time.sleep(0.04)
        finally:
            context.exit_run()

    progress = [e for e in events if e["event"] == "download_progress"]
    assert progress, "growing bytes must be reported as download_progress"
    assert progress[0]["repo_id"] == repo_id
    assert all(e["downloaded_bytes"] > 0 for e in progress)

    stalls = [e for e in events if e.get("kind") == "phase_stall"]
    assert not stalls, "a real, ongoing download must not read as a stall"


def test_unchanging_size_still_heartbeats_and_does_not_stall(tmp_path, monkeypatch):
    # A size that never changes for the life of the watch is exactly what a
    # healthy xet reconstruction pause looks like on disk (#343 follow-up) -
    # there is no growth-based signal available to tell it apart from a
    # genuine hang, so the watch must still speak up on its own cadence.
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
                time.sleep(0.8)
        finally:
            context.exit_run()

    progress = [e for e in events if e["event"] == "download_progress"]
    assert progress, "an active watch must heartbeat even with no growth"
    assert all(e["bytes_per_second"] == 0 for e in progress)

    stalls = [e for e in events if e.get("kind") == "phase_stall"]
    assert not stalls, "a watched download must not read as a stall"


def test_large_file_freeze_then_burst_does_not_stall(tmp_path, monkeypatch):
    # Reproduces the shape measured against a real ~3.5GB xet-backed download
    # (stabilityai/sd-turbo unet): the tracked size holds flat for stretches
    # well past one emit interval, then jumps hundreds of MB at once when
    # reconstruction catches up - not a mock of DownloadWatch's own size
    # function, a real directory written on that timing.
    _fast_watch(monkeypatch)
    repo_id = "some-org/some-model"
    blob_dir = tmp_path / "models--some-org--some-model" / "blobs"
    blob_dir.mkdir(parents=True)
    blob_file = blob_dir / "abc123.incomplete"
    blob_file.write_bytes(b"x" * 1024)

    events = []
    context = RunContext(on_event=events.append)
    with _fast_watchdog():
        context.enter_run()
        try:
            context.note_phase("loading")
            with download_watch.DownloadWatch(
                repo_id, context, cache_dir=str(tmp_path)
            ):
                # Freeze well past one emit interval (0.05s) ...
                time.sleep(0.3)
                # ... then a burst, then freeze again.
                with open(blob_file, "ab") as f:
                    f.write(b"x" * (16 * 1024 * 1024))
                time.sleep(0.3)
        finally:
            context.exit_run()

    progress = [e for e in events if e["event"] == "download_progress"]
    assert progress, "growth and quiet stretches must both be reported"
    assert progress[-1]["downloaded_bytes"] >= 16 * 1024 * 1024

    stalls = [e for e in events if e.get("kind") == "phase_stall"]
    assert not stalls, "a freeze-then-burst download must not read as a stall"
