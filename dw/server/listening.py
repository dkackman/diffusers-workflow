"""Whether a notes channel is listening: the last time one polled each
workspace's request list, or the whole server's (`seen_all`, the read a
channel that watches every workspace makes). In memory only; a restart
forgets it."""

import threading
import time
from datetime import datetime, timezone

LISTENING_WINDOW = 30  # seconds a poll keeps a workspace "listening"


class Listening:
    def __init__(self, clock=time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._last = {}
        self._all = None  # the last poll of every workspace at once

    def seen(self, workspace: str) -> None:
        with self._lock:
            self._last[workspace] = self._clock()

    def seen_all(self) -> None:
        """A poll that covers every workspace, those made after it too."""
        with self._lock:
            self._all = self._clock()

    def state(self, workspace: str) -> dict:
        with self._lock:
            polls = [t for t in (self._last.get(workspace), self._all) if t]
        if not polls:
            return {"listening": False, "last_seen": None}
        last = max(polls)
        stamp = datetime.fromtimestamp(last, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return {
            "listening": self._clock() - last <= LISTENING_WINDOW,
            "last_seen": stamp,
        }
