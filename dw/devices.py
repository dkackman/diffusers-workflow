"""Which card a server's worker runs on, and how a job record names it (#462).

`dw.serve --devices cuda:1` (or the `devices` setting) picks the card the
server's worker runs on. One entry only, for now: the worker pool that runs
one job per card is a later stage of #462, and until it lands a second entry
is refused at startup rather than silently ignored.

**The worker is pinned, not pointed.** A worker for `cuda:1` is spawned with
`CUDA_VISIBLE_DEVICES=1` and `DW_DEVICE=cuda` in its environment, so inside it
that card is the only one and is simply `cuda`. That is the fix, not just the
mechanism: the engine reads index 0 in places that would otherwise measure
the wrong card from a `cuda:1` worker - `device_memory_stats`, the per-run
`reset_peak_memory_stats`, and `ComponentsManager`'s auto-offload
`mem_get_info` (dw/pipeline_processors/placement.py). The variables are set in
the *parent* around `Process.start()` (`pinned_environment`): `dw/__init__.py`
imports torch, so nothing in the child runs early enough to set them there.

**The server process sees every card**, so it names one by its ordinal -
`device_label()` gives `"cuda:1 NVIDIA GeForce RTX 3090"`, the string a job
record's `device` carries and the card observed cost buckets by
(dw/server/observed_cost.py).

A device with no explicit index (`cuda`, the default on every box that
names none) is not pinned at all, so an unconfigured server spawns its
worker exactly as it did before.
"""

import contextlib
import os
import threading

from . import _TORCH_AVAILABLE, backend_available, detect_device, get_device

if _TORCH_AVAILABLE:
    import torch

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
DEVICE_ENV_VAR = "DW_DEVICE"

# Mutating os.environ around Process.start() is process-wide; two managers
# spawning at once must not interleave their pins
_ENVIRONMENT_LOCK = threading.Lock()


class DeviceConfigError(ValueError):
    """A `--devices`/`devices` value this server cannot run on."""


def parse_devices(value):
    """`"cuda:0,cuda:1"`, or a list of names, as a list of device names.

    Every entry must be a device torch understands; the order is kept and
    blanks are dropped."""
    if value is None:
        return []
    entries = value.split(",") if isinstance(value, str) else list(value)
    devices = []
    for entry in entries:
        if not isinstance(entry, str):
            raise DeviceConfigError(f"A device is a name like 'cuda:1', not {entry!r}")
        entry = entry.strip()
        if not entry:
            continue
        if _TORCH_AVAILABLE:
            try:
                torch.device(entry)
            except (RuntimeError, TypeError, ValueError):
                raise DeviceConfigError(
                    f"'{entry}' is not a device name - use one like 'cuda:1'"
                ) from None
        devices.append(entry)
    return devices


def cards_present():
    """Every CUDA card this process can see, as `cuda:N (name)` - what a
    refusal names so the operator can pick one that exists."""
    if not (_TORCH_AVAILABLE and torch.cuda.is_available()):
        return []
    return [
        f"cuda:{index} ({card_name(f'cuda:{index}') or 'unknown card'})"
        for index in range(torch.cuda.device_count())
    ]


def check_device_present(device):
    """Refuse a device this machine does not have, naming what it does have.

    At startup rather than at the first job: a server that comes up on
    `cuda:7` would otherwise fail every job it ever ran."""
    kind = torch.device(device).type if _TORCH_AVAILABLE else device
    if kind == "cuda":
        index = torch.device(device).index or 0
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if index >= count:
            present = cards_present()
            have = (
                "the cards present are " + ", ".join(present)
                if present
                else "this machine has no CUDA card"
            )
            raise DeviceConfigError(f"There is no {device}: {have}")
        return
    if not backend_available(kind):
        raise DeviceConfigError(f"{device} is not available on this machine")


def resolve_serve_devices(cli_value, settings_value):
    """The devices a server runs on - `--devices`, else the `devices`
    setting - or None when neither names any, which leaves today's single
    device (`DW_DEVICE`, then the `device` setting, then detection) alone.

    Raises DeviceConfigError for more than one entry (until the worker pool
    lands) and for a device this machine does not have."""
    raw = cli_value if cli_value else settings_value
    devices = parse_devices(raw)
    if not devices:
        return None
    if len(devices) > 1:
        raise DeviceConfigError(
            f"--devices names {len(devices)} devices ({', '.join(devices)}); "
            "this server runs one worker, so name one card"
        )
    for device in devices:
        check_device_present(device)
    return devices


def _explicit_cuda_index(device):
    """The index of a CUDA device named with one ('cuda:1' -> 1), else None
    - a bare 'cuda' is never pinned, so an unconfigured server is unchanged."""
    if not (_TORCH_AVAILABLE and device):
        return None
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        return None
    if parsed.type != "cuda" or parsed.index is None:
        return None
    return parsed.index


def worker_environment(device):
    """The environment variables that pin a worker to `device`, or {} for a
    device that needs no pin.

    The index is the server's view of the card. When the server itself runs
    under a `CUDA_VISIBLE_DEVICES` list, index N is that list's Nth entry,
    and that entry is what the worker is given."""
    index = _explicit_cuda_index(device)
    if index is None:
        return {}
    visible = os.environ.get(CUDA_VISIBLE_DEVICES)
    physical = str(index)
    if visible:
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if index < len(entries):
            physical = entries[index]
    return {CUDA_VISIBLE_DEVICES: physical, DEVICE_ENV_VAR: "cuda"}


@contextlib.contextmanager
def pinned_environment(variables):
    """Set `variables` in os.environ for the duration, then put back what
    was there - wrapped around `Process.start()`, which is when a spawned
    child copies the environment."""
    if not variables:
        yield
        return
    with _ENVIRONMENT_LOCK:
        saved = {name: os.environ.get(name) for name in variables}
        os.environ.update(variables)
        try:
            yield
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def device_ordinal(device=None):
    """`cuda:1`, `cuda:0` for a bare `cuda`, `mps`, `cpu` - the device as
    the server process addresses it. None is the device dw runs on."""
    device = device or get_device()
    if not _TORCH_AVAILABLE:
        return str(device)
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        return str(device)
    if parsed.type == "cuda":
        return f"cuda:{parsed.index or 0}"
    return parsed.type


def card_name(device=None):
    """The card's own name - 'NVIDIA GeForce RTX 3090', 'Apple M5 Pro (MPS)'
    - or None where there is none to read (CPU, or a probe that fails)."""
    if not _TORCH_AVAILABLE:
        return None
    ordinal = device_ordinal(device)
    try:
        parsed = torch.device(ordinal)
        if parsed.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_name(parsed.index or 0)
        if parsed.type == "mps" and backend_available("mps"):
            from . import _apple_chip_name

            return f"{_apple_chip_name()} (MPS)"
    except (RuntimeError, AssertionError, AttributeError):
        return None
    return None


def device_label(device=None):
    """`"cuda:1 NVIDIA GeForce RTX 3090"` - the ordinal, then the card's
    name when there is one. What a job record's `device` holds."""
    ordinal = device_ordinal(device)
    name = card_name(ordinal)
    return f"{ordinal} {name}" if name else ordinal


def card_of(label):
    """The card name in a `device_label` string, or None for a label that
    carries none (`cpu`) or no label at all."""
    if not label or " " not in label:
        return None
    return label.split(" ", 1)[1]


def is_default_device(device=None):
    """Whether `device` is the card this box runs on when nothing names one
    - the card every job recorded before jobs carried a `device` ran on."""
    return device_ordinal(device) == device_ordinal(detect_device())
