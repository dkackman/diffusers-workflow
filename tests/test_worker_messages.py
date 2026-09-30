"""The worker protocol's typed messages: one frozen dataclass per command and
per reply, on a wire that stays the dicts it always was.

Two properties per type: `from_wire(to_wire(x)) == x`, so nothing a message
carries is lost crossing the queue, and `to_wire()` is exactly the dict the
worker and JobManager exchanged before the types existed - the queue, the
worker's command loop and every raw-dict assertion in tests/test_worker*.py
depend on that shape.
"""

import pytest

from dw.worker import (
    Cancel,
    Cancelled,
    ClearMemory,
    Execute,
    Failed,
    MemoryCleared,
    MemoryInfo,
    MemoryStatus,
    MemoryStatusReply,
    Output,
    ProbeCache,
    ProbeCacheReply,
    Progress,
    Shutdown,
    Succeeded,
    UnknownReply,
    WorkerCrashed,
    WorkflowLoaded,
    parse_reply,
)

DEFINITION = {"id": "x", "steps": []}

# (message, the dict it put on the queue before it had a type)
EXAMPLES = [
    (
        Execute(
            definition=DEFINITION,
            file_spec="/w/x.json",
            source="path",
            workflow_dir="/w",
            output_dir="/out",
            arguments={"prompt": "p"},
            log_level="INFO",
        ),
        {
            "type": "execute",
            "arguments": {"prompt": "p"},
            "output_dir": "/out",
            "log_level": "INFO",
            "definition": DEFINITION,
            "file_spec": "/w/x.json",
            "source": "path",
            "workflow_dir": "/w",
        },
    ),
    (
        Execute(
            definition=DEFINITION,
            file_spec="/w/__inline__.json",
            source="inline",
            workflow_dir=None,
            output_dir="/out",
            arguments={},
            log_level="DEBUG",
            asset_dir="/assets",
        ),
        {
            "type": "execute",
            "arguments": {},
            "output_dir": "/out",
            "log_level": "DEBUG",
            "asset_dir": "/assets",
            "definition": DEFINITION,
            "file_spec": "/w/__inline__.json",
            "source": "inline",
            "workflow_dir": None,
        },
    ),
    (
        ProbeCache(
            request_id="r-1",
            definition=DEFINITION,
            file_spec="/w/x.json",
            source="path",
            workflow_dir="/w",
            output_dir="/out",
            arguments={},
            asset_dir="/assets",
        ),
        {
            "type": "probe_cache",
            "request_id": "r-1",
            "definition": DEFINITION,
            "file_spec": "/w/x.json",
            "source": "path",
            "arguments": {},
            "output_dir": "/out",
            "workflow_dir": "/w",
            "asset_dir": "/assets",
        },
    ),
    (Cancel(), {"type": "cancel"}),
    (Shutdown(), {"type": "shutdown"}),
    (ClearMemory(request_id="r-2"), {"type": "clear_memory", "request_id": "r-2"}),
    (MemoryStatus(request_id="r-3"), {"type": "memory_status", "request_id": "r-3"}),
    (
        WorkflowLoaded(workflow_name="flux"),
        {"type": "workflow_loaded", "workflow_name": "flux"},
    ),
    (Output(message="Cancelling..."), {"type": "output", "message": "Cancelling..."}),
    (
        Progress(event={"event": "step_start", "step": "gen", "index": 0}),
        {"type": "progress", "event": "step_start", "step": "gen", "index": 0},
    ),
    (
        MemoryInfo(info={"run_count": 1}),
        {"type": "memory_info", "info": {"run_count": 1}},
    ),
    (
        Succeeded(
            message="Workflow completed successfully",
            run_count=2,
            manifest=[{"step": "gen", "files": ["/out/a.png"]}],
        ),
        {
            "type": "success",
            "message": "Workflow completed successfully",
            "run_count": 2,
            "manifest": [{"step": "gen", "files": ["/out/a.png"]}],
        },
    ),
    (
        Cancelled(message="Workflow run cancelled", manifest=[]),
        {"type": "cancelled", "message": "Workflow run cancelled", "manifest": []},
    ),
    (
        Failed(message="Workflow execution error: boom", traceback="tb", manifest=[]),
        {
            "type": "error",
            "message": "Workflow execution error: boom",
            "traceback": "tb",
            "manifest": [],
        },
    ),
    (
        Failed(message="Unknown command type: nope"),
        {"type": "error", "message": "Unknown command type: nope"},
    ),
    (
        WorkerCrashed(message="boom", traceback="tb"),
        {"type": "worker_crashed", "message": "boom", "traceback": "tb"},
    ),
    (
        MemoryStatusReply(request_id="r-3", info={"gpu_available": True}),
        {
            "type": "memory_status",
            "request_id": "r-3",
            "info": {"gpu_available": True},
        },
    ),
    (
        MemoryCleared(request_id="r-2", info={"gpu_available": False}),
        {
            "type": "memory_cleared",
            "request_id": "r-2",
            "info": {"gpu_available": False},
        },
    ),
    (
        ProbeCacheReply(request_id="r-1", cached=["gen"]),
        {"type": "probe_cache", "request_id": "r-1", "cached": ["gen"]},
    ),
    (
        ProbeCacheReply(request_id=None, cached=None, error="bad file"),
        {
            "type": "probe_cache",
            "request_id": None,
            "cached": None,
            "error": "bad file",
        },
    ),
]


def _name(example):
    message, wire = example
    return f"{type(message).__name__}-{len(wire)}"


@pytest.mark.parametrize("message, wire", EXAMPLES, ids=[_name(e) for e in EXAMPLES])
def test_to_wire_is_todays_dict(message, wire):
    assert message.to_wire() == wire


@pytest.mark.parametrize("message, wire", EXAMPLES, ids=[_name(e) for e in EXAMPLES])
def test_from_wire_inverts_to_wire(message, wire):
    assert type(message).from_wire(message.to_wire()) == message


REPLY_TYPES = (
    WorkflowLoaded,
    Output,
    Progress,
    MemoryInfo,
    Succeeded,
    Cancelled,
    Failed,
    WorkerCrashed,
    MemoryStatusReply,
    MemoryCleared,
    ProbeCacheReply,
)


@pytest.mark.parametrize(
    "message, wire",
    [e for e in EXAMPLES if isinstance(e[0], REPLY_TYPES)],
    ids=[_name(e) for e in EXAMPLES if isinstance(e[0], REPLY_TYPES)],
)
def test_parse_reply_names_the_type_the_wire_carries(message, wire):
    assert parse_reply(wire) == message


def test_an_unknown_reply_type_is_kept_whole_rather_than_raised():
    reply = parse_reply({"type": "pong", "run_count": 0})
    assert isinstance(reply, UnknownReply)
    assert reply.type == "pong"
    assert reply.message == {"type": "pong", "run_count": 0}
    assert UnknownReply.from_wire(reply.to_wire()) == reply


def test_a_failure_echoes_the_request_it_answers_only_when_there_was_one():
    """A request handler that raises answers with an error carrying the
    command's request_id, so the reader waiting on that id gets it; a run's
    own failure has no id and its wire shape is unchanged."""
    answered = Failed(message="Command processing error: x", request_id="r-9")
    assert answered.to_wire() == {
        "type": "error",
        "message": "Command processing error: x",
        "request_id": "r-9",
    }
    assert parse_reply(answered.to_wire()) == answered
