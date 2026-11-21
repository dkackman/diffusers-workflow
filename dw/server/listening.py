"""Whether a notes channel is listening: the last time one polled each
workspace's request list. In memory only; a restart forgets it."""

import threading
import time
from datetime import datetime, timezone

LISTENING_WINDOW = 30  # seconds a poll keeps a workspace "listening"


class Listening:
    def __init__(self, clock=time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._last = {}

    def seen(self, workspace: str) -> None:
        with self._lock:
            self._last[workspace] = self._clock()

    def state(self, workspace: str) -> dict:
        with self._lock:
            last = self._last.get(workspace)
        if last is None:
            return {"listening": False, "last_seen": None}
        stamp = datetime.fromtimestamp(last, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return {
            "listening": self._clock() - last <= LISTENING_WINDOW,
            "last_seen": stamp,
        }
