"""What the worker process and its machine are using in host RAM.

`device_memory_stats` answers for the accelerator, which on this project is
often the least informative half of the question: the templates here keep
weights in host memory by design (`offload: "sequential"`, `group_offload`),
so a card can sit near-empty through a generation whose weights are very
much resident somewhere. This module is the other half.

Three methods, in order, none of them required: psutil when it happens to be
installed (a transitive dependency here, not a declared one), Linux's
`/proc` otherwise, and `resource.getrusage` for the process's peak, which is
available on every POSIX platform. Nothing raises - a reading that cannot be
taken is reported as None rather than failing the call that asked for it.
"""

import logging
import os

logger = logging.getLogger("dw")

__all__ = [
    "host_memory_stats",
    "host_memory_fields",
    "trim_host_memory",
    "release_host_caches",
    "pinned_host_memory_fields",
]

_MB = 1024.0 * 1024.0


def host_memory_stats():
    """Snapshot of host memory for this process and the machine it is on.

    Returns:
        dict with keys, any of which may be None when the platform cannot
        answer:
            rss_mb (float or None): resident set size of *this* process -
                in the worker, the weights it is holding
            peak_rss_mb (float or None): the high-water mark of the above,
                which is what says whether a run that has finished released
                what it took
            total_mb (float or None): the machine's physical memory
            available_mb (float or None): what can be handed out without
                swapping - MemAvailable on Linux, not free memory, since
                page cache is reclaimable
    """
    stats = {
        "rss_mb": None,
        "peak_rss_mb": _peak_rss_mb(),
        "total_mb": None,
        "available_mb": None,
    }

    for method in (_psutil_stats, _proc_stats):
        try:
            reading = method()
        except Exception as e:  # a memory reading is never worth an exception
            logger.debug(f"Host memory via {method.__name__} failed: {e}")
            continue
        for key, value in reading.items():
            if stats.get(key) is None and value is not None:
                stats[key] = value
        if all(stats[key] is not None for key in ("rss_mb", "total_mb")):
            break

    return _hold_the_high_water_mark(stats)


def _hold_the_high_water_mark(stats):
    """Keep peak_rss_mb >= rss_mb, which is what a high-water mark means.

    The two readings come from different places - getrusage's ru_maxrss,
    quantized to whole pages and taken first, against psutil's rss taken a
    moment later - so a process that has never peaked meaningfully above its
    current size reports them within a megabyte of each other in either
    order. `peak - rss` is the whole point of the pair (what a run took and
    did not give back), and a small negative there reads as "these fields
    are not comparable" rather than "nothing leaked" (#83).
    """
    peak, rss = stats["peak_rss_mb"], stats["rss_mb"]
    if peak is not None and rss is not None and peak < rss:
        stats["peak_rss_mb"] = rss
    return stats


def _psutil_stats():
    import psutil

    virtual = psutil.virtual_memory()
    return {
        "rss_mb": psutil.Process().memory_info().rss / _MB,
        "total_mb": virtual.total / _MB,
        "available_mb": virtual.available / _MB,
    }


def _proc_stats():
    """Linux without psutil: /proc/self/statm for this process, /proc/meminfo
    for the machine. Both absent elsewhere, which is what the None default
    covers."""
    reading = {"rss_mb": None, "total_mb": None, "available_mb": None}

    try:
        with open("/proc/self/statm", "r") as f:
            pages = int(f.read().split()[1])
        reading["rss_mb"] = pages * os.sysconf("SC_PAGE_SIZE") / _MB
    except (OSError, ValueError, IndexError, AttributeError):
        pass

    wanted = {"MemTotal:": "total_mb", "MemAvailable:": "available_mb"}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] in wanted:
                    # meminfo is in kB
                    reading[wanted[parts[0]]] = int(parts[1]) / 1024.0
    except (OSError, ValueError):
        pass

    return reading


def _peak_rss_mb():
    """getrusage's high-water mark, in kB on Linux and bytes on macOS - the
    one place in this module where the unit depends on the platform."""
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if not peak:
        return None
    return peak / _MB if sys.platform == "darwin" else peak / 1024.0


# The memory payload's own names for the above, beside its gpu_* keys
FIELD_NAMES = {
    "rss_mb": "host_memory_rss_mb",
    "peak_rss_mb": "host_memory_peak_rss_mb",
    "total_mb": "host_memory_total_mb",
    "available_mb": "host_memory_available_mb",
}


def host_memory_fields():
    """host_memory_stats under the names the memory payload reports, with
    the readings this platform cannot take left out rather than sent as
    null - a key that is absent says "not measurable here", where a null
    would read as "measured, and nothing"."""
    stats = host_memory_stats()
    return {
        FIELD_NAMES[key]: value for key, value in stats.items() if value is not None
    }


def trim_host_memory():
    """Hand memory the process has already freed back to the operating
    system, and report how much that was in MB (0.0 when the platform has no
    way to ask).

    Dropping the last reference to a model frees it inside the process, not
    back to the kernel: glibc keeps the arenas the weights were read into and
    hands them out again to *this* process. That is normally invisible and
    correct - until a template needs 96% of host RAM to run at all, at which
    point the several GB the previous job's arenas are sitting on is the
    difference between a run and a SIGKILL five minutes in (#98). The
    templates here load tens of GB of weights through host memory, so the
    arenas in question are large ones and the fragmentation that keeps
    `malloc_trim` from returning them is the exception rather than the rule.

    Linux/glibc only: `malloc_trim` is a GNU extension. Everywhere else -
    macOS included - this is a no-op that reports 0.0, because there is
    nothing to ask and a fabricated number would be worse than none.
    """
    import ctypes
    import ctypes.util
    import sys

    if not sys.platform.startswith("linux"):
        return 0.0
    before = host_memory_stats()["rss_mb"]
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        libc.malloc_trim(ctypes.c_size_t(0))
    except (OSError, AttributeError) as e:
        # musl and friends have no malloc_trim; not having one is not an error
        logger.debug(f"malloc_trim unavailable: {e}")
        return 0.0
    after = host_memory_stats()["rss_mb"]
    if before is None or after is None:
        return 0.0
    return max(0.0, before - after)


def pinned_host_memory_fields():
    """What CUDA's pinned-host allocator is holding, in MB, or {} where
    there is none to report.

    Group offloading with `use_stream` stages a component's weights through
    *pinned* host memory, which torch caches per process exactly as it
    caches device memory: freeing the tensors returns the blocks to that
    cache, not to the OS, so they stay in this process's RSS and count
    against the next job's host budget. It is invisible in every figure the
    memory payload carried before - `rss_mb` includes it without saying so
    and `gpu_memory_*` does not see it at all (#98).
    """
    try:
        import torch

        stats = torch.cuda.host_memory_stats()
    except Exception:
        return {}
    fields = {}
    for key, name in (
        ("allocated_bytes.all.current", "host_pinned_allocated_mb"),
        ("reserved_bytes.all.current", "host_pinned_reserved_mb"),
    ):
        value = stats.get(key)
        if value is not None:
            fields[name] = value / _MB
    return fields


def release_host_caches():
    """Give back host memory this process is holding but no longer using,
    and report what came back in MB.

    Two caches, neither of which `gc.collect()` touches:

    - torch's pinned-host allocator, where group offloading's staging
      buffers live. `_host_emptyCache` frees the blocks nothing is using;
      blocks a still-loaded pipeline is staging through are in use and are
      not touched, so this is safe to call with models resident.
    - glibc's heap arenas, via `malloc_trim`. Freeing a large block inside
      the process does not hand its pages back to the kernel.

    Together they are why a worker that has released every model still sat
    on 14.5 GB, which is the difference between the next job running and
    being OOM-killed five minutes in on a template that needs 96% of host
    RAM (#98).
    """
    before = host_memory_stats()["rss_mb"]
    try:
        import torch

        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception as e:  # a cleanup is never worth failing the run for
        logger.debug(f"Could not empty the pinned host cache: {e}")
    trim_host_memory()
    after = host_memory_stats()["rss_mb"]
    if before is None or after is None:
        return 0.0
    return max(0.0, before - after)
