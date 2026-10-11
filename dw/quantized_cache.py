"""On-disk cache of SDNQ on-the-fly-quantized components (#815).

A component an SDNQ config quantizes as it loads goes through the same steps on
every cold load: read the bf16 weights, copy them to the accelerator, quantize,
copy them back. The same config yields the same int4/int8 weights each time, so
this keeps them: the first load saves the quantized component with sdnq's own
`save_sdnq_model` (safetensors plus a JSON config - no pickle), and a later load
of the same config reads it back with `load_sdnq_model` instead.

Off unless `DW_QUANTIZED_CACHE=1`. `DW_QUANTIZED_CACHE_DIR` moves the root from
`<HF_HOME>/dw-quantized`; `dw.serve` resolves it to an absolute path once and
pins it in the environment its workers inherit (`configure_at_startup`).

**The key** is a sha256 over the component's repo, subfolder, the commit of the
snapshot the load reads, the quantizer config minus the arguments that do not
change the stored weights (the devices and the quantized-matmul switches),
`torch_dtype`, and the installed sdnq/diffusers/transformers/torch versions. A
stale hit silently gives the wrong weights, so the key errs complete: anything
it cannot pin (a local path, a snapshot whose files are not all on disk, a
commit the loaded object and the hub cache disagree on) is not cached.

**The entry** is `<root>/<64-hex key>/`: sdnq's files plus `entry.json`, the
key's inputs in readable form with the size and the created and last-hit
times. It is written to a `<key>.tmp-*` sibling and renamed into place, so a
reader never sees half an entry and two workers saving the same key at once
leave one. Pruning, `sdnq_optimize`, LoRAs and offload hooks all run after the
save point, so an entry holds the untruncated component as quantized.

**Limits:** a save that would leave less free than `min(50 GiB, 10% of the
volume)` is skipped, and the server-start sweep deletes every entry built
against other package versions. Nothing else evicts; clearing is
`rm -rf "$DW_QUANTIZED_CACHE_DIR"`.

Every lookup and save emits a `quantized_cache` job event,
`{component, result: hit|miss|saved|skipped, reason?, seconds, bytes}`, and
none of it can fail a job: a cache that cannot help steps aside.
"""

import errno
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import secrets
import shutil
import time
from functools import lru_cache

from .events import get_context
from .security import validate_path

logger = logging.getLogger("dw")

ENABLE_ENV_VAR = "DW_QUANTIZED_CACHE"
DIR_ENV_VAR = "DW_QUANTIZED_CACHE_DIR"
DEFAULT_DIR_NAME = "dw-quantized"
ENTRY_FILE = "entry.json"
EVENT = "quantized_cache"

# A save never takes the volume below the smaller of these: on a large disk
# 50 GiB is room for a model download, on a small one 10% is
LOW_DISK_FLOOR_BYTES = 50 * 2**30
LOW_DISK_FLOOR_FRACTION = 0.10

VERSIONED_PACKAGES = ("sdnq", "diffusers", "transformers", "torch")

# Quantizer arguments that place or run the weights rather than shape them -
# the same int4 tensors come out whichever card quantized them, and a hit
# re-applies them at load
UNKEYED_ARGUMENTS = (
    "quantization_device",
    "return_device",
    "use_quantized_matmul",
    "use_quantized_matmul_conv",
    "non_blocking",
)

# Quantizer arguments that are sets of module names held as lists (or a dict
# of them). SDNQConfig dedupes each through `list(set(...))`, so their order
# follows the process's string-hash seed - one order per worker - and the key
# sorts them, or each card would key the same config differently
SET_ARGUMENTS = (
    "modules_to_not_convert",
    "modules_to_not_use_matmul",
    "modules_dtype_dict",
)

# Reasons a `skipped` event carries
LOCAL_PATH = "local_path"
UNPINNED_SNAPSHOT = "unpinned_snapshot"
LOW_DISK = "low_disk"
UNSUPPORTED = "unsupported"
ALREADY_SAVED = "already saved by another worker"

# Reasons a `miss` event carries: why the load found nothing to read
INCOMPLETE_SNAPSHOT = "incomplete_snapshot"
NO_ENTRY = "no_entry"
LOAD_FAILED = "load_failed"

_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_TMP_MARKER = ".tmp-"


def enabled():
    """Whether the cache reads, writes and sweeps - only `DW_QUANTIZED_CACHE=1`."""
    return os.environ.get(ENABLE_ENV_VAR) == "1"


def cache_root():
    """The cache's root directory, absolute."""
    configured = os.environ.get(DIR_ENV_VAR)
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    from huggingface_hub import constants

    return os.path.abspath(os.path.join(constants.HF_HOME, DEFAULT_DIR_NAME))


@lru_cache(maxsize=1)
def installed_versions():
    return {
        package: importlib.metadata.version(package) for package in VERSIONED_PACKAGES
    }


def is_sdnq_config(config):
    return type(config).__name__ == "SDNQConfig" and type(config).__module__.startswith(
        "sdnq"
    )


def emit(component, result, reason=None, seconds=None, nbytes=None):
    data = {
        "component": component,
        "result": result,
        "seconds": None if seconds is None else round(seconds, 2),
        "bytes": nbytes,
    }
    if reason is not None:
        data["reason"] = reason
    logger.info(
        f"Quantized cache: {component} {result}" + (f" ({reason})" if reason else "")
    )
    get_context().emit(EVENT, **data)


# --- the key ---------------------------------------------------------------


def quantizer_arguments(config):
    """The quantizer config as keyed: its class and every argument that shapes
    the stored weights. Read before the load - sdnq clears the device fields
    on the config it quantized with."""
    arguments = config_snapshot(config)
    for name in UNKEYED_ARGUMENTS:
        arguments.pop(name, None)
    for name in SET_ARGUMENTS:
        value = arguments.get(name)
        if isinstance(value, list):
            arguments[name] = sorted(value, key=str)
        elif isinstance(value, dict):
            arguments[name] = {
                k: sorted(v, key=str) if isinstance(v, list) else v
                for k, v in value.items()
            }
    return {
        "class": f"{type(config).__module__}.{type(config).__qualname__}",
        "arguments": arguments,
    }


def config_snapshot(config):
    """A deep copy of `config.to_dict()` in plain JSON types. `to_dict()`
    hands back the config's own lists and dicts, and sdnq's load writes into
    them (it appends the modules it left unquantized to
    `modules_to_not_convert`), so a shallow read taken before the load turns
    into the config after it - the save's key then differs from the next
    lookup's, and from one card's load to another's."""
    return json.loads(json.dumps(config.to_dict(), sort_keys=True, default=str))


def key_inputs(model_name, subfolder, commit, quantizer, torch_dtype):
    return {
        "model_name": model_name,
        "subfolder": subfolder or "",
        "commit": commit,
        "quantizer": quantizer,
        "torch_dtype": None if torch_dtype is None else str(torch_dtype),
        "versions": installed_versions(),
    }


def cache_key(inputs):
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- the snapshot a load reads ----------------------------------------------


def _hub_cache_dir(cache_dir=None):
    if cache_dir:
        return os.path.abspath(str(cache_dir))
    from huggingface_hub import constants

    return os.path.abspath(constants.HF_HUB_CACHE)


def is_local_path(model_name):
    return os.path.isdir(str(model_name))


def _cached_file(model_name, filename, revision, cache_dir):
    """The path of `filename` in the hub cache's snapshot for `revision`,
    resolved offline by huggingface_hub (a branch or tag through `refs/`,
    the blob behind the symlink checked present), or None."""
    from huggingface_hub import try_to_load_from_cache

    try:
        path = try_to_load_from_cache(
            str(model_name),
            filename,
            cache_dir=_hub_cache_dir(cache_dir),
            revision=revision,
        )
    except Exception:
        return None
    return path if isinstance(path, str) else None


def _in_folder(subfolder, name):
    return f"{subfolder}/{name}" if subfolder else name


def resolve_commit(model_name, revision=None, cache_dir=None, subfolder=None):
    """The commit `revision` names in the hub cache, read offline: the
    snapshot in which huggingface_hub finds `<subfolder>/config.json` for
    it. None when the cache holds no such snapshot."""
    filename = _in_folder(subfolder, "config.json")
    path = _cached_file(model_name, filename, revision or "main", cache_dir)
    if path is None:
        return None
    commit = os.path.basename(os.path.normpath(path[: -len(filename)]))
    return commit if _COMMIT_PATTERN.match(commit) else None


def _is_unvarianted_weights(name, suffix):
    # 'model.safetensors' but not 'model.fp16.safetensors' - a variant is
    # a file of its own, and a load without one never reads it
    return name.endswith(suffix) and "." not in name[: -len(suffix)]


def snapshot_complete(model_name, commit, subfolder=None, cache_dir=None):
    """Whether the snapshot at `commit` holds the component whole: its
    `config.json` and every weight file it names, each present on disk.
    huggingface_hub finds each file; which files a component needs (each
    shard in the index's `weight_map`, or the unvarianted single file) is
    dw's own rule."""
    from huggingface_hub.file_download import repo_folder_name

    hub = _hub_cache_dir(cache_dir)

    def present(name):
        return _cached_file(model_name, _in_folder(subfolder, name), commit, hub)

    try:
        # contained in the repo's own folder, symlinks resolved: a repo
        # folder linked out to another disk keeps its blobs beside it
        repo = os.path.join(
            hub, repo_folder_name(repo_id=str(model_name), repo_type="model")
        )
        config = present("config.json")
        if config is None:
            return False
        names = os.listdir(validate_path(os.path.dirname(config), repo))
        indexes = [
            n for n in names if _is_unvarianted_weights(n, ".safetensors.index.json")
        ]
        if indexes:
            for index in indexes:
                index_path = present(index)
                if index_path is None:
                    return False
                with open(validate_path(index_path, repo)) as f:
                    weight_map = json.load(f).get("weight_map", {})
                if any(present(shard) is None for shard in set(weight_map.values())):
                    return False
            return True
        singles = [n for n in names if _is_unvarianted_weights(n, ".safetensors")]
        return bool(singles) and all(present(n) is not None for n in singles)
    except Exception:
        return False


def lookup_commit(model_name, subfolder=None, revision=None, cache_dir=None):
    """The commit to look a load up under, or None when its snapshot is not
    complete on disk - the load is about to download or fail, and a key on
    that commit would name files nobody read."""
    commit = resolve_commit(model_name, revision, cache_dir, subfolder)
    if commit is None or not snapshot_complete(
        model_name, commit, subfolder, cache_dir
    ):
        return None
    return commit


def recorded_commit(component):
    """The commit the library recorded on a loaded object, where it records
    one (transformers' `config._commit_hash`)."""
    config = getattr(component, "config", None)
    commit = getattr(config, "_commit_hash", None)
    if commit is None and isinstance(config, dict):
        commit = config.get("_commit_hash")
    return commit if isinstance(commit, str) and _COMMIT_PATTERN.match(commit) else None


def save_commit(component, model_name, subfolder=None, revision=None, cache_dir=None):
    """The commit to save a just-loaded component under, or None when no one
    snapshot can be named. It is the commit the same load's lookup would
    name, so nothing is written that no later lookup reads; a recorded
    commit that disagrees with it means the hub cache moved under the load."""
    commit = lookup_commit(model_name, subfolder, revision, cache_dir)
    if commit is None:
        return None
    recorded = recorded_commit(component)
    if recorded is not None and recorded != commit:
        return None
    return commit


# --- entries ----------------------------------------------------------------


def entry_path(root, key):
    if not _KEY_PATTERN.match(key):
        raise ValueError(f"not a cache key: {key!r}")
    return validate_path(os.path.join(root, key), root)


def _tree_bytes(path):
    total = 0
    for directory, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(directory, name))
            except OSError:
                pass
    return total


def read_entry(path):
    with open(os.path.join(path, ENTRY_FILE)) as f:
        entry = json.load(f)
    if not isinstance(entry, dict):
        raise ValueError("entry.json is not an object")
    return entry


def _write_json(path, data):
    tmp = f"{path}{_TMP_MARKER}{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def component_nbytes(component):
    """The in-memory size of a component's parameters and buffers - what its
    entry will take on disk, give or take headers and config."""
    total = 0
    for tensors in (component.parameters(), component.buffers()):
        for tensor in tensors:
            total += tensor.numel() * tensor.element_size()
    return total


def floor_bytes(total):
    return min(LOW_DISK_FLOOR_BYTES, int(LOW_DISK_FLOOR_FRACTION * total))


def load_entry(root, key, model_cls, torch_dtype, device, config_arguments):
    """Load a hit, placed on `device`. None on a miss. An entry that fails to
    load is deleted, so the fresh quantize that follows can replace it, and
    the error is raised on for the miss to name."""
    path = entry_path(root, key)
    if not os.path.isdir(path):
        return None
    try:
        from sdnq import load_sdnq_model

        component = load_sdnq_model(
            path,
            model_cls=model_cls,
            dtype=torch_dtype,
            device=device,
            dequantize_fp32=config_arguments.get("dequantize_fp32"),
            use_quantized_matmul=config_arguments.get("use_quantized_matmul"),
        )
        component = component.to(device)
    except Exception as e:
        logger.warning(
            f"Quantized cache entry {key} failed to load ({type(e).__name__}: {e}) "
            "- deleting it and quantizing fresh"
        )
        shutil.rmtree(path, ignore_errors=True)
        raise
    try:
        entry = read_entry(path)
        entry["last_hit"] = time.time()
        _write_json(os.path.join(path, ENTRY_FILE), entry)
    except Exception as e:
        logger.debug(f"Could not record the hit on {key}: {e}")
    return component


def save_entry(root, key, component, inputs):
    """Save a just-quantized component as `key`. Returns (result, reason,
    bytes) for the event: `saved`, or `skipped` with why. Never raises."""
    estimate = component_nbytes(component)
    try:
        os.makedirs(root, exist_ok=True)
        usage = shutil.disk_usage(root)
    except OSError as e:
        logger.warning(f"Quantized cache: cannot reach {root}: {e}")
        return "skipped", f"save_failed: {_error_name(e)}", None
    floor = floor_bytes(usage.total)
    if usage.free - estimate < floor:
        logger.warning(
            f"Quantized cache: not saving a {estimate / 2**30:.1f} GiB entry - "
            f"{usage.free / 2**30:.1f} GiB free, and the floor is "
            f"{floor / 2**30:.1f} GiB"
        )
        return "skipped", LOW_DISK, None

    final = entry_path(root, key)
    tmp = validate_path(
        os.path.join(root, f"{key}{_TMP_MARKER}{os.getpid()}-{secrets.token_hex(4)}"),
        root,
    )
    try:
        from sdnq import save_sdnq_model

        save_sdnq_model(component, tmp)
        nbytes = _tree_bytes(tmp)
        now = time.time()
        _write_json(
            os.path.join(tmp, ENTRY_FILE),
            {**inputs, "key": key, "bytes": nbytes, "created": now, "last_hit": None},
        )
        try:
            os.rename(tmp, final)
        except OSError as e:
            if e.errno in (errno.ENOTEMPTY, errno.EEXIST) and os.path.isdir(final):
                shutil.rmtree(tmp, ignore_errors=True)
                return "saved", ALREADY_SAVED, _tree_bytes(final)
            raise
        return "saved", None, nbytes
    except Exception as e:
        logger.warning(
            f"Quantized cache: saving {key} failed ({type(e).__name__}: {e}) "
            "- the job goes on without it"
        )
        shutil.rmtree(tmp, ignore_errors=True)
        return "skipped", f"save_failed: {_error_name(e)}", None


def _error_name(error):
    code = getattr(error, "errno", None)
    if code is not None and code in errno.errorcode:
        return errno.errorcode[code]
    return type(error).__name__


# --- server start -----------------------------------------------------------


def sweep(root):
    """Delete every entry built against other package versions or with no
    readable `entry.json`, and every leftover `*.tmp-*`. Returns (count,
    bytes freed)."""
    if not os.path.isdir(root):
        return 0, 0
    versions = installed_versions()
    count = freed = 0
    for name in os.listdir(root):
        path = validate_path(os.path.join(root, name), root)
        if _TMP_MARKER in name:
            stale = True
        elif _KEY_PATTERN.match(name) and os.path.isdir(path):
            try:
                stale = read_entry(path).get("versions") != versions
            except Exception:
                stale = True
        else:
            continue
        if not stale:
            continue
        size = _tree_bytes(path) if os.path.isdir(path) else os.path.getsize(path)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
        count += 1
        freed += size
    return count, freed


def _entry_sizes(root):
    sizes = []
    if not os.path.isdir(root):
        return sizes
    for name in os.listdir(root):
        if not _KEY_PATTERN.match(name):
            continue
        path = validate_path(os.path.join(root, name), root)
        try:
            size = int(read_entry(path)["bytes"])
        except Exception:
            size = _tree_bytes(path)
        sizes.append(size)
    return sizes


def log_disk(root):
    """One line for the cache's footprint and the volume's free space - a
    warning when free space is already under the floor plus the largest entry
    cached, so a full disk shows before the first save is skipped. Returns
    whether it warned."""
    sizes = _entry_sizes(root)
    usage = shutil.disk_usage(root)
    floor = floor_bytes(usage.total)
    margin = max(sizes, default=0)
    message = (
        f"Quantized cache at {root}: {len(sizes)} entries, "
        f"{sum(sizes) / 2**30:.1f} GiB in use, {usage.free / 2**30:.1f} GiB free"
    )
    if usage.free < floor + margin:
        logger.warning(
            f"{message} - under the {floor / 2**30:.1f} GiB floor plus the largest "
            f"entry ({margin / 2**30:.1f} GiB), so new entries may be skipped"
        )
        return True
    logger.info(message)
    return False


def configure_at_startup():
    """Pin the root for the workers, sweep it and report the disk - at server
    start, before any worker spawns. Nothing at all while the cache is off."""
    if not enabled():
        return
    root = cache_root()
    os.environ[DIR_ENV_VAR] = root
    try:
        os.makedirs(root, exist_ok=True)
        count, freed = sweep(root)
        logger.info(
            f"Quantized cache sweep: deleted {count} stale entries, "
            f"freed {freed / 2**30:.1f} GiB"
        )
        log_disk(root)
    except OSError as e:
        logger.warning(f"Quantized cache at {root} is unusable: {e}")


# --- the load ---------------------------------------------------------------

# ComponentSpec fields a load_components() keyword overrides, as diffusers
# reads them (ComponentSpec.load)
_SPEC_FIELDS = ("pretrained_model_name_or_path", "subfolder", "revision", "variant")


def _per_component(arguments, name):
    """The keywords diffusers' load_components() hands one component: a dict
    value is per name with a `default`, any other value is for all of them."""
    resolved = {}
    for key, value in arguments.items():
        if not isinstance(value, dict):
            resolved[key] = value
        elif name in value:
            resolved[key] = value[name]
        elif "default" in value:
            resolved[key] = value["default"]
    return resolved


def _names_to_load(pipeline, arguments):
    """The components load_components() is about to load, as it chooses them."""
    specs = pipeline._component_specs
    names = arguments.get("names")
    if names is None:
        names = [
            name
            for name, spec in specs.items()
            if spec.default_creation_method == "from_pretrained"
            and spec.pretrained_model_name_or_path is not None
            and getattr(pipeline, name, None) is None
        ]
    elif isinstance(names, str):
        names = [names]
    workflow = arguments.get("workflow")
    if workflow is not None:
        expected = {
            spec.name
            for spec in pipeline.blocks.get_workflow(workflow).expected_components
        }
        names = [name for name in names if name in expected]
    return [name for name in names if name in specs]


class _Candidate:
    """One component an SDNQ config quantizes on this load, keyed."""

    def __init__(self, name, spec, kwargs):
        self.name = name
        fields = {field: getattr(spec, field, None) for field in _SPEC_FIELDS}
        fields.update({k: kwargs[k] for k in _SPEC_FIELDS if k in kwargs})
        self.model_name = fields["pretrained_model_name_or_path"]
        self.subfolder = fields["subfolder"]
        self.revision = fields["revision"]
        self.variant = fields["variant"]
        self.cache_dir = kwargs.get("cache_dir")
        self.model_cls = spec.type_hint
        self.load_id = getattr(spec, "load_id", None)
        config = kwargs["quantization_config"]
        self.torch_dtype = kwargs.get("torch_dtype", kwargs.get("dtype"))
        # Read now: sdnq clears the devices on the config it quantized with
        self.return_device = getattr(config, "return_device", None) or "cpu"
        self.config_arguments = config_snapshot(config)
        self.quantizer = quantizer_arguments(config)

    def unsupported(self):
        if self.model_name is None or self.model_cls is None or self.variant:
            return UNSUPPORTED
        if is_local_path(self.model_name):
            return LOCAL_PATH
        return None

    def inputs(self, commit):
        return key_inputs(
            self.model_name, self.subfolder, commit, self.quantizer, self.torch_dtype
        )


def load_components(pipeline, arguments):
    """`pipeline.load_components(**arguments)`, with every SDNQ-quantized
    component read from the cache where an entry exists and saved to it where
    one does not. With the cache off this is that call and nothing more."""
    if not enabled():
        pipeline.load_components(**arguments)
        return

    try:
        candidates = _candidates(pipeline, arguments)
    except Exception as e:
        logger.warning(f"Quantized cache: could not read the component specs: {e}")
        candidates = []

    root = cache_root()
    hits = {}
    misses = []
    for candidate in candidates:
        reason = candidate.unsupported()
        if reason is not None:
            emit(candidate.name, "skipped", reason=reason)
            continue
        commit = lookup_commit(
            candidate.model_name,
            candidate.subfolder,
            candidate.revision,
            candidate.cache_dir,
        )
        if commit is None:
            # Not all on disk - the load downloads it, and the save after
            # resolves the snapshot it ended up reading
            misses.append((candidate, INCOMPLETE_SNAPSHOT))
            continue
        key = cache_key(candidate.inputs(commit))
        started = time.monotonic()
        try:
            component = load_entry(
                root,
                key,
                candidate.model_cls,
                candidate.torch_dtype,
                candidate.return_device,
                candidate.config_arguments,
            )
        except Exception as e:
            logger.warning(f"Quantized cache: lookup of {candidate.name} failed: {e}")
            misses.append((candidate, f"{LOAD_FAILED}: {_error_name(e)}"))
            continue
        if component is None:
            misses.append((candidate, NO_ENTRY))
            continue
        if candidate.load_id is not None:
            # Keeps the spec update_components() would otherwise reset
            component._diffusers_load_id = candidate.load_id
        hits[candidate.name] = component
        emit(
            candidate.name,
            "hit",
            seconds=time.monotonic() - started,
            nbytes=_safe_nbytes(component),
        )

    if hits:
        pipeline.update_components(**hits)
        if arguments.get("names") is not None:
            names = arguments["names"]
            names = [names] if isinstance(names, str) else names
            arguments = {
                **arguments,
                "names": [name for name in names if name not in hits],
            }

    started = time.monotonic()
    pipeline.load_components(**arguments)
    seconds = time.monotonic() - started

    for candidate, reason in misses:
        component = getattr(pipeline, candidate.name, None)
        nbytes = _safe_nbytes(component) if component is not None else None
        emit(candidate.name, "miss", reason=reason, seconds=seconds, nbytes=nbytes)
        if component is None:
            continue
        _save(root, candidate, component)


def _candidates(pipeline, arguments):
    quantization = arguments.get("quantization_config")
    if quantization is None:
        return []
    forwarded = {k: v for k, v in arguments.items() if k not in ("names", "workflow")}
    candidates = []
    for name in _names_to_load(pipeline, arguments):
        kwargs = _per_component(forwarded, name)
        if not is_sdnq_config(kwargs.get("quantization_config")):
            continue
        candidates.append(_Candidate(name, pipeline._component_specs[name], kwargs))
    return candidates


def _safe_nbytes(component):
    try:
        return component_nbytes(component)
    except Exception:
        return None


def _save(root, candidate, component):
    commit = save_commit(
        component,
        candidate.model_name,
        candidate.subfolder,
        candidate.revision,
        candidate.cache_dir,
    )
    if commit is None:
        emit(candidate.name, "skipped", reason=UNPINNED_SNAPSHOT)
        return
    inputs = candidate.inputs(commit)
    started = time.monotonic()
    result, reason, nbytes = save_entry(root, cache_key(inputs), component, inputs)
    emit(
        candidate.name,
        result,
        reason=reason,
        seconds=time.monotonic() - started,
        nbytes=nbytes,
    )
