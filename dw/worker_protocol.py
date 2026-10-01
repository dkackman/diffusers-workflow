"""
The worker's wire protocol: one frozen dataclass per command and reply.

The queue carries their wire dicts (to_wire/from_wire), and parse_reply is
where a reply dict becomes a type. A request's reply echoes its request_id
(see WorkerManager.request). This module imports neither dw.worker nor
dw.worker_manager, so the parent reads the types without loading the worker.
"""

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, List, Optional, Tuple

# ------------------------------------------------------------------ protocol
# One frozen dataclass per command and reply; the queue carries their wire
# dicts (to_wire/from_wire), and parse_reply is where a reply dict becomes a
# type. A request's reply echoes its request_id (see WorkerManager.request).


@dataclass(frozen=True)
class _Message:
    """Wire mapping shared by every message: `TYPE` is the dict's "type",
    each field is a key, and a field named in `OPTIONAL` is left off the
    wire while it is None - the keys that were only ever sent when set."""

    TYPE: ClassVar[str] = ""
    OPTIONAL: ClassVar[Tuple[str, ...]] = ()

    def to_wire(self) -> Dict[str, Any]:
        wire = {"type": self.TYPE}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None and f.name in self.OPTIONAL:
                continue
            wire[f.name] = value
        return wire

    @classmethod
    def from_wire(cls, wire: Dict[str, Any]):
        return cls(**{f.name: wire.get(f.name) for f in fields(cls)})


# Commands, parent to worker


@dataclass(frozen=True)
class Execute(_Message):
    """Run the admitted snapshot (see _handle_execute)."""

    TYPE: ClassVar[str] = "execute"
    OPTIONAL: ClassVar[Tuple[str, ...]] = ("asset_dir",)
    definition: Dict[str, Any]
    file_spec: str
    source: str
    workflow_dir: Optional[str]
    output_dir: str
    arguments: Dict[str, Any]
    log_level: str
    asset_dir: Optional[str] = None


@dataclass(frozen=True)
class ProbeCache(_Message):
    """Execute's snapshot, asking which steps the step cache would serve."""

    TYPE: ClassVar[str] = "probe_cache"
    OPTIONAL: ClassVar[Tuple[str, ...]] = ("asset_dir", "log_level")
    request_id: str
    definition: Dict[str, Any]
    file_spec: str
    source: str
    output_dir: str
    arguments: Dict[str, Any]
    workflow_dir: Optional[str] = None
    asset_dir: Optional[str] = None
    log_level: Optional[str] = None


@dataclass(frozen=True)
class Cancel(_Message):
    TYPE: ClassVar[str] = "cancel"


@dataclass(frozen=True)
class Shutdown(_Message):
    TYPE: ClassVar[str] = "shutdown"


@dataclass(frozen=True)
class ClearMemory(_Message):
    TYPE: ClassVar[str] = "clear_memory"
    request_id: str


@dataclass(frozen=True)
class MemoryStatus(_Message):
    TYPE: ClassVar[str] = "memory_status"
    request_id: str


# Replies, worker to parent


@dataclass(frozen=True)
class WorkflowLoaded(_Message):
    TYPE: ClassVar[str] = "workflow_loaded"
    workflow_name: str


@dataclass(frozen=True)
class Output(_Message):
    TYPE: ClassVar[str] = "output"
    message: str


@dataclass(frozen=True)
class Progress(_Message):
    """A run event, whose keys sit beside "type" on the wire."""

    TYPE: ClassVar[str] = "progress"
    event: Dict[str, Any]

    def to_wire(self) -> Dict[str, Any]:
        return {"type": self.TYPE, **self.event}

    @classmethod
    def from_wire(cls, wire: Dict[str, Any]):
        return cls(event={k: v for k, v in wire.items() if k != "type"})


@dataclass(frozen=True)
class MemoryInfo(_Message):
    TYPE: ClassVar[str] = "memory_info"
    info: Dict[str, Any]


@dataclass(frozen=True)
class Succeeded(_Message):
    TYPE: ClassVar[str] = "success"
    message: str
    run_count: int
    manifest: List[Dict[str, Any]]


@dataclass(frozen=True)
class Cancelled(_Message):
    TYPE: ClassVar[str] = "cancelled"
    message: str
    manifest: List[Dict[str, Any]]


@dataclass(frozen=True)
class Failed(_Message):
    """A run's failure, or a command the worker could not handle. It carries
    a request_id only when it answers a request, so the reader waiting on
    that id gets it rather than timing out."""

    TYPE: ClassVar[str] = "error"
    OPTIONAL: ClassVar[Tuple[str, ...]] = ("traceback", "manifest", "request_id")
    message: str
    traceback: Optional[str] = None
    manifest: Optional[List[Dict[str, Any]]] = None
    request_id: Optional[str] = None


@dataclass(frozen=True)
class WorkerCrashed(_Message):
    TYPE: ClassVar[str] = "worker_crashed"
    message: str
    traceback: str


@dataclass(frozen=True)
class MemoryStatusReply(_Message):
    TYPE: ClassVar[str] = "memory_status"
    request_id: Optional[str]
    info: Dict[str, Any]


@dataclass(frozen=True)
class MemoryCleared(_Message):
    TYPE: ClassVar[str] = "memory_cleared"
    request_id: Optional[str]
    info: Dict[str, Any]


@dataclass(frozen=True)
class ProbeCacheReply(_Message):
    TYPE: ClassVar[str] = "probe_cache"
    OPTIONAL: ClassVar[Tuple[str, ...]] = ("error",)
    request_id: Optional[str]
    cached: Optional[List[str]]
    error: Optional[str] = None


@dataclass(frozen=True)
class UnknownReply(_Message):
    """A reply dict whose type no class claims, kept whole."""

    message: Dict[str, Any]

    @property
    def type(self):
        return self.message.get("type")

    def to_wire(self) -> Dict[str, Any]:
        return dict(self.message)

    @classmethod
    def from_wire(cls, wire: Dict[str, Any]):
        return cls(message=dict(wire))


_REPLY_TYPES = {
    reply.TYPE: reply
    for reply in (
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
}


def parse_reply(wire: Dict[str, Any]):
    """The typed reply a worker message dict is - an UnknownReply for a type
    no class claims, never an exception."""
    reply_type = _REPLY_TYPES.get(wire.get("type"), UnknownReply)
    return reply_type.from_wire(wire)
