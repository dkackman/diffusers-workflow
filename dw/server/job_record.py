"""A job and the vocabulary a job is described in: its states, the forms of
cost acknowledgement, the spec fields a rerun needs and the event bound.

`Job` is the live record the runner and the SSE clients share; `JobHistory`
(`job_history.py`) persists the finished ones and `JobManager` (`jobs.py`)
queues them.
"""

import threading
import time
import uuid

from ..devices import label_of
from ..download_watch import format_progress
from ..workspace import DEFAULT_WORKSPACE_NAME

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATES = (SUCCEEDED, FAILED, CANCELLED)

# Which form of cost acknowledgement a job was queued with (#85): none (the
# web UI and every HTTP caller that sends nothing), a bare boolean, or one
# bound to the plan that was validated
ACK_NONE = "none"
ACK_BOOLEAN = "boolean"
ACK_BOUND = "bound"

# The spec fields a rerun needs - shared by persistence and live rerun
RERUN_SPEC_KEYS = (
    "workflow_path",
    "workflow",
    "base_dir",
    # A rerun belongs in the workspace the original ran in, so the roots
    # that decided that are part of what history keeps
    "workspace",
    "output_dir",
    "asset_dir",
    # so a rerun is attributed to the same catalog entry
    "catalog_name",
    "workflow_dir",
    # what the original run was consented to, kept for the record - a
    # rerun's own request decides its form
    "acknowledged_cost",
)

# A long run emits thousands of progress events; the tail is what explains
# the outcome. Bounded so history stays a summary store, not an event log
MAX_PERSISTED_EVENTS = 200


class Job:
    """One workflow execution request and everything observed about it."""

    def __init__(self, spec):
        self.id = uuid.uuid4().hex[:12]
        self.spec = spec
        self.workflow_name = spec["workflow_name"]
        self.catalog_name = spec.get("catalog_name")
        self.status = QUEUED
        # Set by JobManager.cancel on a running job: the job's thread
        # checks it before sending Execute, since a cancel that reaches an
        # idle worker is ignored
        self.cancel_requested = False
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None
        self.manifest = []
        # A copy: run-time warnings are appended to this list (see
        # _note_progress) and the spec is what a rerun is built from
        self.warnings = list(spec.get("warnings", []))
        self.error = None
        self.traceback = None
        # Which run this job turned out to be - reported by the worker's
        # run_start event, unknown until then and forever for a job that
        # never got that far
        self.run_id = None
        self.run_dir = None
        self.run_version = None
        # The card the job ran on - its ordinal ("cuda:1") and its name
        # ("NVIDIA GeForce RTX 3090", None for a card with none) - set when
        # it starts running, so None while queued and forever for a job
        # cancelled before it ran (#462, #693). `device` is the two as one
        # label
        self.device_ordinal = None
        self.device_card = None
        # Which form of cost acknowledgement queued this job (#85)
        self.acknowledged = spec.get("acknowledged") or ACK_NONE
        # The worker's own high-water mark for this run, from its final
        # memory_info message - None for a run that never got that far (#243)
        self.host_memory_peak_rss_mb = None
        # This job's own contribution to that process-lifetime figure -
        # growth since the job's first phase boundary, or the job's current
        # rss when it caused no growth (#272). None for a run that never got
        # a memory_info message at all
        self.host_memory_job_peak_rss_mb = None
        self.events = []
        # The running summary a poll reads - see _note_progress. Kept as the
        # events arrive rather than derived from the log on request, because
        # the log is trimmed to its last MAX_PERSISTED_EVENTS and a caller
        # polling a long render should not have to page through it to learn
        # that something moved
        self.last_event_at = None
        self.phase = None
        self.phase_detail = None
        self.phase_started_at = None
        self.step_name = None
        self.parent_step = None
        self.step_index = None
        self.total_steps = None
        self.denoise_step = None
        self.denoise_total_steps = None
        self.condition = threading.Condition()

    @property
    def device(self):
        """`"cuda:1 NVIDIA GeForce RTX 3090"`, or None before it ran."""
        return label_of(self.device_ordinal, self.device_card)

    def add_event(self, event):
        with self.condition:
            # `at` is seconds since the job started (since it was created,
            # for the events before that). Phases say what a step is waiting
            # on; only a clock on each event says what it cost - the
            # lead-in from `step_start` to the first `pipeline_step` on a
            # reused pipeline is the number a "slow start" report needs
            since = self.started_at if self.started_at is not None else self.created_at
            self.events.append(
                {"seq": len(self.events), "at": round(time.time() - since, 1), **event}
            )
            self._note_progress(event)
            self.condition.notify_all()

    def _note_progress(self, event):
        """Fold one event into the running summary.

        A single-step generation emits `generating` and then nothing until it
        is done, so 'no new events' is the normal state of a healthy run and
        says nothing about whether it is progressing. What answers that is
        how long it has been that way, and how far into the denoise loop it
        got - both of which are here rather than in the event log.
        """
        now = time.time()
        self.last_event_at = now
        kind = event.get("event")
        if kind == "phase":
            self.phase = event.get("phase")
            self.phase_detail = event.get("detail")
            self.phase_started_at = now
        elif kind == "pipeline_step":
            self.denoise_step = event.get("step")
            self.denoise_total_steps = event.get("total_steps")
        elif kind == "download_progress":
            # Folded into phase_detail rather than a field of its own - a
            # poller already reads phase_detail for what the loading phase
            # is waiting on, and the next "phase" event (loading ending)
            # overwrites it same as any other detail (#343)
            self.phase_detail = format_progress(
                event.get("repo_id"),
                event.get("downloaded_bytes"),
                event.get("bytes_per_second"),
                event.get("seconds_since_bytes_changed"),
            )
        elif kind == "warning":
            # Both channels, on purpose: the event log keeps the moment it
            # happened, `warnings` keeps it where a caller who polled the
            # finished job will actually look, since a warning about the
            # artifact outlives the run that noticed it (#82). The step it
            # fired in is the run's, not the warning's - the engine warns
            # from inside a step without knowing which one it is.
            #
            # A phase-stall report (#176) is the exception: it is a moment,
            # not a fact about the result - a 90 s cold load says "still in
            # phase 'loading'" three times and then succeeds - so it stays
            # in the event log only. `warnings` is the channel a consumer
            # reads after the run, and the regression suites assert it is
            # empty on a clean one.
            message = event.get("message")
            if message and event.get("kind") != "phase_stall":
                named = f"{self.step_name}: {message}" if self.step_name else message
                if named not in self.warnings:
                    self.warnings.append(named)
        elif kind == "step_start":
            self.step_name = event.get("step")
            # A sub-workflow counts its own steps from zero; what a caller
            # watching a composed run needs is where the run it queued has
            # got to, so the parent's counter wins when the event carries
            # one and the step name stays the child's (#90)
            self.parent_step = event.get("parent_step")
            self.step_index = event.get("parent_index", event.get("index"))
            self.total_steps = event.get("parent_total_steps", event.get("total_steps"))
            # A new step's denoise loop has not started; the previous step's
            # count would read as this one's progress
            self.denoise_step = None
            self.denoise_total_steps = None

    def progress(self):
        """Where a running job has got to, or None for one that has not
        started - a terminal job has a manifest, which is a better answer
        than a stale phase, except for FAILED: the manifest is only the
        steps that finished, not the one that was running when the job died,
        and that phase (`loading` / `generating` / `decoding` / `saving`) is
        the fastest way to tell what killed it without reading a traceback
        (#269). Frozen at `finished_at` rather than read against the current
        clock, so `seconds_in_phase` reports how long the dead step had been
        running rather than growing forever after the job is long over."""
        if self.last_event_at is None or self.status not in (RUNNING, FAILED):
            return None
        now = (
            self.finished_at
            if self.status == FAILED and self.finished_at
            else time.time()
        )
        summary = {
            "step": self.step_name,
            # The step of the queued workflow the one above is running
            # inside, for a composed run; null when they are the same thing
            "parent_step": self.parent_step,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "phase": self.phase,
            "phase_detail": self.phase_detail,
            "seconds_in_phase": (
                round(now - self.phase_started_at, 1) if self.phase_started_at else None
            ),
            # The one number that separates a slow run from a hung one -
            # but only once the denoise loop is running, see below
            "seconds_since_event": round(now - self.last_event_at, 1),
            # Always present, null until the loop starts. A key that only
            # appears once there is a count to report cannot be told apart
            # from a key that is missing because nothing is happening: the
            # lead-in to `generating` - encoding the prompt and any
            # reference image or audio - is over a minute of silence on a
            # large video model, and read as an absent counter it looks
            # exactly like a wedged denoise loop. Null here means the loop
            # has not started; a number that stops moving is the stuck one
            "denoise_step": self.denoise_step,
            "denoise_total_steps": self.denoise_total_steps,
        }
        return summary

    def finish(self, status, error=None, traceback_text=None):
        self.status = status
        self.finished_at = time.time()
        self.error = error
        self.traceback = traceback_text
        self.add_event({"event": "job_status", "status": status})

    def events_after(self, after_seq):
        # Clamped: an 'after' below -1 would slice from the END of the log
        # (events[-4:] for after=-5) and silently drop the earlier events a
        # client asking for everything expects
        after_seq = max(after_seq, -1)
        with self.condition:
            return self.events[after_seq + 1 :]

    def wait_for_event(self, after_seq, timeout):
        """Block until an event past after_seq exists or the job ends."""
        with self.condition:
            if len(self.events) > after_seq + 1 or self.status in TERMINAL_STATES:
                return
            self.condition.wait(timeout)

    def summary(self):
        return {
            "id": self.id,
            "workflow": self.workflow_name,
            "workflow_name": self.catalog_name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            # Which workspace this job runs in - a live job's spec may not
            # carry one yet (e.g. a caller that never named a workspace),
            # so it defaults the same way history's column does
            "workspace": self.spec.get("workspace") or DEFAULT_WORKSPACE_NAME,
            "run_id": self.run_id,
            # The run's ordinal - 'v4' - so the job that just ran can be
            # named the way the gallery will name it
            "run_version": self.run_version,
            "device": self.device,
            "acknowledged": self.acknowledged,
        }

    def detail(self):
        return {
            **self.summary(),
            "arguments": self.spec.get("arguments", {}),
            "warnings": self.warnings,
            "manifest": self.manifest,
            "error": self.error,
            "traceback": self.traceback,
            "event_count": len(self.events),
            "run_dir": self.run_dir,
            "acknowledged_cost": self.spec.get("acknowledged_cost"),
            "progress": self.progress(),
        }
