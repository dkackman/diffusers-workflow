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

__all__ = ["host_memory_stats", "host_memory_fields"]

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
