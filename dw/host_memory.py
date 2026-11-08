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

import contextlib
import json
import logging
import os
import stat
import tempfile

from filelock import FileLock, Timeout

logger = logging.getLogger("dw")

__all__ = [
    "host_memory_stats",
    "host_memory_fields",
    "trim_host_memory",
    "release_host_caches",
    "pinned_host_memory_fields",
    "process_rss_mb",
    "committed_rss_mb",
    "committed_mb",
    "device_mapped_mb",
    "rss_split_mb",
    "HostClaims",
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


def read_oom_score_adj(pid):
    """The kernel's `oom_score_adj` for process `pid`, read now from /proc, or
    None where it cannot be read (not Linux, or the process is gone). The
    value in force, not the one the server meant to write (#881)."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        with open(f"/proc/{pid}/oom_score_adj", "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def process_rss_mb(pid):
    """Resident set size of another process - a server's worker, read from
    the server - in MB, or None where it cannot be read (the process is
    gone, or neither psutil nor /proc is there). A pool of workers shares
    one machine's RAM, so each worker's share is what says which one the
    kernel's OOM killer would pick (#462)."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        import psutil

        return psutil.Process(pid).memory_info().rss / _MB
    except Exception as e:
        logger.debug(f"psutil could not read process {pid}: {e}")
    try:
        with open(f"/proc/{pid}/statm", "r") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / _MB
    except (OSError, ValueError, IndexError, AttributeError):
        return None


# The memory payload's own names for the above, beside its gpu_* keys
FIELD_NAMES = {
    "rss_mb": "host_memory_rss_mb",
    "peak_rss_mb": "host_memory_peak_rss_mb",
    "total_mb": "host_memory_total_mb",
    "available_mb": "host_memory_available_mb",
}


SPLIT_FIELD_NAMES = {
    "anon_mb": "host_memory_rss_anon_mb",
    "file_mb": "host_memory_rss_file_mb",
    "shmem_mb": "host_memory_rss_shmem_mb",
    "committed_mb": "host_memory_committed_mb",
}


def rss_split_mb(status_path="/proc/self/status"):
    """This process's resident set split into `anon_mb` (heap and tensors the
    process owns - committed memory), `file_mb` (pages of mapped files, such
    as memory-mapped safetensors, which the kernel can drop and re-read) and
    `shmem_mb` (shared memory and MAP_SHARED anonymous pages, which the
    kernel can only swap). Only Linux says; elsewhere, or on a kernel without
    the split, {}.

    `rss_mb` is the sum of the three, so a loader that starts mapping the
    checkpoint instead of copying it raises `rss_mb` and the peak by the size
    of the weights while committed memory does not move (#709)."""
    wanted = {"RssAnon:": "anon_mb", "RssFile:": "file_mb", "RssShmem:": "shmem_mb"}
    split = {}
    try:
        with open(status_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] in wanted:
                    split[wanted[parts[0]]] = int(parts[1]) / 1024.0
    except (OSError, ValueError):
        return {}
    return split


# The mappings of the NVIDIA driver's device files. Pinned host memory the
# driver hands out (cudaHostAlloc - torch's pinned cache, group offload's
# stream staging) is mapped into the process through /dev/nvidiactl, and the
# kernel files those pages under RssFile, not RssAnon or RssShmem, though no
# file backs them and the kernel cannot drop them
_DEVICE_MAPPING_PREFIX = "/dev/nvidia"


def device_mapped_mb(smaps_path="/proc/self/smaps"):
    """The resident pages of this process's /dev/nvidia* mappings, in MB:
    pinned host memory the NVIDIA driver mapped in, which RssFile counts
    beside the checkpoint pages but which, unlike them, is committed (#869).
    0.0 with no such mapping; None where smaps cannot be read (not Linux, or
    the process is gone).

    Measured on lem (2026-11-05) on the bench server's H3 worker, pid
    224454: RssAnon 3,119,024 kB + RssFile 211,752 kB + RssShmem 16,120 kB
    = VmRSS exactly, with VmPin and VmLck 0. Grouping /proc/<pid>/smaps by
    mapping, the /dev/nvidiactl and /dev/nvidia-uvm mappings held 64,784 kB
    of Rss with Anonymous 0, and RssFile was those plus the shared
    libraries' 146,952 kB of file pages - the driver's pages land in
    RssFile. The /dev/zero shared mappings (16,128 kB) are RssShmem. The
    worker's payload carried no host_pinned_* figure at the time (torch's
    pinned cache held nothing), so the driver mapping is read here, not
    torch's own count."""
    total_kb = 0
    device = False
    try:
        with open(smaps_path, "r") as f:
            for line in f:
                head, _, rest = line.partition(" ")
                if "-" in head and not head.endswith(":"):
                    # A mapping's header: "start-end perms offset dev inode path"
                    fields = rest.split(None, 4)
                    path = fields[4].strip() if len(fields) > 4 else ""
                    device = path.startswith(_DEVICE_MAPPING_PREFIX)
                elif device and head == "Rss:":
                    total_kb += int(rest.split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return total_kb / 1024.0


def _committed_from(split, device_mb):
    """RssAnon + RssShmem + the driver's pinned pages, or None without an
    RssAnon reading or a device reading - a part missing would under-charge."""
    anon = split.get("anon_mb")
    if anon is None or device_mb is None:
        return None
    return anon + split.get("shmem_mb", 0.0) + device_mb


def committed_mb(status_path="/proc/self/status", smaps_path="/proc/self/smaps"):
    """What a process has committed in host RAM, in MB: memory the kernel
    cannot reclaim by dropping pages, so MemAvailable does not count it as
    free (#869). RssAnon (heap and tensors) + RssShmem (shared memory, which
    MemAvailable leaves out) + the pages of its /dev/nvidia* mappings (the
    driver's pinned host memory, filed under RssFile; `device_mapped_mb`).
    The rest of RssFile - the checkpoint pages a worker mmaps - is page
    cache, left out.

    None where it cannot be read: not Linux, a status without RssAnon, or a
    failed read.

    `smaps_path` None reads no device mappings and the term is DROPPED: the
    result is RssAnon + RssShmem alone, short by the driver's pinned pages."""
    split = rss_split_mb(status_path)
    device = device_mapped_mb(smaps_path) if smaps_path else 0.0
    return _committed_from(split, device)


def _has_proc():
    """Whether this platform has a /proc to read another process from."""
    return os.path.isdir("/proc/self")


def _proc_path(pid, name):
    return f"/proc/{pid}/{name}"


def committed_rss_mb(pid, status_path=None, smaps_path=None, device=True):
    """What another process - a server's worker - has committed in host RAM,
    in MB, as `committed_mb` reads it: its mapped checkpoint pages left out,
    page cache that MemAvailable already counts as free and that a second
    worker mapping the same weights shares (#869).

    Where there is no /proc (macOS), its whole RSS, as `process_rss_mb`
    reads it - the peaks it is compared with are RSS there too, since no run
    recorded a committed peak. On Linux, a failed read or a status without
    RssAnon is None, not the whole RSS: the gate subtracts this reading from
    a running job's peak and credits it for the idle worker, so a larger
    figure would under-charge, and None charges the whole peak.

    `device=False` reads status alone, no smaps (RssAnon + RssShmem, without
    the driver's pinned pages): for a caller holding a lock, which adds the
    term from the worker's own report (`device_term_mb`)."""
    if status_path is None:
        if not isinstance(pid, int) or pid <= 0:
            return None
        if not _has_proc():
            return process_rss_mb(pid)
        status_path = _proc_path(pid, "status")
        smaps_path = _proc_path(pid, "smaps") if device else None
    return committed_mb(status_path, smaps_path)


def device_term_mb(report):
    """The driver-pinned part of a worker's last memory report (committed -
    anon - shmem, from `host_memory_fields`), or None with no usable report."""
    try:
        committed = report["host_memory_committed_mb"]
        anon = report["host_memory_rss_anon_mb"]
    except (KeyError, TypeError):
        return None
    if committed is None or anon is None:
        return None
    return max(0.0, committed - anon - (report.get("host_memory_rss_shmem_mb") or 0.0))


def host_memory_fields():
    """host_memory_stats under the names the memory payload reports, with
    the readings this platform cannot take left out rather than sent as
    null - a key that is absent says "not measurable here", where a null
    would read as "measured, and nothing"."""
    stats = host_memory_stats()
    fields = {
        FIELD_NAMES[key]: value for key, value in stats.items() if value is not None
    }
    split = rss_split_mb()
    for key, value in split.items():
        fields[SPLIT_FIELD_NAMES[key]] = value
    # What the host-RAM gate charges a run (#869); read only where the split
    # is, so a platform without /proc sends neither
    committed = _committed_from(split, device_mapped_mb()) if split else None
    if committed is not None:
        fields[SPLIT_FIELD_NAMES["committed_mb"]] = committed
    return fields


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


# --- Claims across dw.serve processes on one host (#1227) ---
#
# What the dw.serve processes on one host have claimed of its RAM (#1227).
#
# `pool.host_ram_wait` gates a job on the host's MemAvailable less the growth
# the jobs already running have still to make. A second dw.serve on the same
# machine is invisible to that sum: its workers hold what they have loaded
# (MemAvailable shows it) but not what they are about to load, so two servers
# each starting a 65 GB job on a 125 GB host both pass their own gate and the
# kernel OOM-kills one.
#
# Each server publishes, to a directory every server on the host shares, one
# file named for its pid listing its running jobs as `[worker_pid, peak_mb]`;
# `foreign` reads the other live servers' files and answers in the form
# `host_ram_wait` takes for its own running jobs, `(peak_mb, committed_mb now)`.
# `lock` is a `filelock.FileLock` around a pass's check-and-publish, so two servers starting
# at once see each other's claim rather than both passing.
#
# The directory is `DW_HOST_CLAIMS_DIR`, else a per-user folder in the system
# temp dir. A file whose
# server pid is gone is deleted when read. Nothing here raises: a claim that
# cannot be written or read leaves the gate where it was before this module.

ENV_DIR = "DW_HOST_CLAIMS_DIR"
LOCK_TIMEOUT_S = 10


def claims_dir():
    explicit = os.environ.get(ENV_DIR)
    if explicit:
        return explicit
    user = os.getuid() if hasattr(os, "getuid") else "user"
    return os.path.join(tempfile.gettempdir(), f"dw-host-claims-{user}")


def _ensure_private_dir(directory):
    """Create `directory` 0700 and refuse it unless it is a real directory
    this user owns that no one else can write. The default folder sits at a
    predictable path in the shared temp dir, so another local user could
    create it first and plant claims or symlinks in it. Raises OSError."""
    os.makedirs(directory, mode=0o700, exist_ok=True)
    info = os.lstat(directory)
    if not stat.S_ISDIR(info.st_mode):
        raise OSError(f"host claims folder {directory} is not a directory")
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise OSError(f"host claims folder {directory} is owned by another user")
    if info.st_mode & 0o022:
        raise OSError(f"host claims folder {directory} is writable by others")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # PermissionError: it exists, as another user's
        return True
    return True


class HostClaims:
    def __init__(self, directory=None, pid=None):
        self.directory = directory or claims_dir()
        self.pid = pid if pid is not None else os.getpid()
        # A folder named on purpose (DW_HOST_CLAIMS_DIR, or a caller's) may be
        # shared between users by design; the default one must be private
        self._private = directory is None and not os.environ.get(ENV_DIR)

    def _prepare(self):
        """Create the folder; raises OSError for a default one that is not
        private (_ensure_private_dir)."""
        if self._private:
            _ensure_private_dir(self.directory)
        else:
            os.makedirs(self.directory, mode=0o700, exist_ok=True)

    def _path(self, pid):
        return os.path.join(self.directory, f"{pid}.json")

    @contextlib.contextmanager
    def lock(self):
        """Hold the host-wide lock for one check-and-publish; best effort -
        a lock that cannot be taken is logged and the pass carries on."""
        held = None
        try:
            self._prepare()
            held = FileLock(
                os.path.join(self.directory, ".lock"), timeout=LOCK_TIMEOUT_S
            )
            held.acquire()
        except (OSError, Timeout) as e:
            logger.debug(f"Host claims lock unavailable: {e}")
            held = None
        try:
            yield
        finally:
            if held is not None:
                held.release()

    def publish(self, claims):
        """Record this server's running jobs, `[(worker_pid, peak_mb)]` or
        `[(worker_pid, peak_mb, pinned_mb)]`, replacing its last record; none
        removes the file. `pinned_mb` is the worker's driver-pinned host
        memory (`device_term_mb`), which a reader cannot take from /proc
        status, so the reader charges the job on the basis the local gate
        does (`dispatch._worker_rss_mb`) rather than as growth to come."""
        path = self._path(self.pid)
        try:
            if not claims:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)
                return
            self._prepare()
            # A fresh name opened O_EXCL, never a fixed one: a planted symlink
            # at a predictable name would have the write land on its target
            fd, temp = tempfile.mkstemp(
                prefix=f"{self.pid}.", suffix=".tmp", dir=self.directory
            )
            try:
                if not self._private and hasattr(os, "fchmod"):
                    # mkstemp makes it 0600; a shared folder's other servers
                    # must read it, as they could the open()-made file before
                    os.fchmod(fd, 0o644)
                with os.fdopen(fd, "w") as f:
                    json.dump(
                        [
                            [int(claim[0]), float(claim[1])]
                            + ([float(claim[2])] if len(claim) > 2 else [])
                            for claim in claims
                        ],
                        f,
                    )
                os.replace(temp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.remove(temp)
                raise
        except OSError as e:
            logger.debug(f"Could not publish host claims: {e}")

    def withdraw(self):
        self.publish([])

    def foreign(self):
        """`[(peak_mb, committed_mb or None)]` for every job the other live
        servers on this host are running."""
        out = []
        try:
            if self._private:
                _ensure_private_dir(self.directory)
            names = os.listdir(self.directory)
        except OSError:
            return out
        for name in names:
            stem, dot, ext = name.partition(".")
            if ext != "json" or not stem.isdigit() or int(stem) == self.pid:
                continue
            path = os.path.join(self.directory, name)
            if not _alive(int(stem)):
                with contextlib.suppress(OSError):
                    os.remove(path)
                continue
            try:
                with open(path) as f:
                    entries = json.load(f)
                for entry in entries:
                    worker_pid, peak_mb = entry[0], entry[1]
                    pinned = float(entry[2]) if len(entry) > 2 else 0.0
                    committed = committed_rss_mb(int(worker_pid), device=False)
                    if committed is not None:
                        committed += pinned
                    out.append((float(peak_mb), committed))
            except (OSError, ValueError, TypeError, IndexError):
                continue
        return out
