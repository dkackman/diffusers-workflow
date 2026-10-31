"""What a run would have to download before its first step: the hub repos,
single-file checkpoints and task weights the expanded definition (and each
composed child) names that the cache does not already hold, with each one's
size and gate status when the hub is asked (#382, #975, #981).

Everything here is a walk over the expanded definition: a `model_name` or
`loras` repo under a `from_pretrained_arguments`, a `from_single_file` URL, a
`load_components` repo override, and the repos a task declares through its
command's `required_weights` hook. The cache is read through `hub_cache` and
`quantized_cache`, never a directory probe of the plan's own - a free
pre-flight must not be an existence oracle for paths on the box. Nothing here
knows a model; `dw/plan.py` assembles the answer into the plan beside the
estimate and the fingerprint, and this module never imports it back.
"""

import copy
import json
import logging

from huggingface_hub import model_info
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import GatedRepoError, HFValidationError, validate_repo_id

from .hub_cache import repo_download_incomplete, scan_models
from .quantized_cache import _cached_file
from . import references
from .realize import read_sub_workflow
from .security import validate_url

logger = logging.getLogger("dw")

SIZE_LOOKUP_TIMEOUT = 5.0
GIB = 1024**3
FROM_PRETRAINED_KEY = "from_pretrained_arguments"
MODEL_NAME_KEY = "model_name"
# An adapter names its repo directly, not through from_pretrained_arguments
LORAS_KEY = "loras"
WEIGHT_NAME_KEY = "weight_name"
SUBFOLDER_KEY = "subfolder"
SINGLE_FILE_KEY = "from_single_file"
VARIANT_KEY = "variant"
LOAD_COMPONENTS_KEY = "load_components"
OVERRIDE_KEY = "pretrained_model_name_or_path"


def downloads_required(expanded, base_dir, workflow_dir, cache_dir, lookup_sizes):
    """The hub repos and checkpoint URLs the run would fetch before its
    first step: every `model_name` in the expanded definition (and in each
    composed child) that `scan_models` does not find, plus every one it does
    find but whose cached copy is left over from an interrupted pull
    (#382) - a component the load needs is missing, or a blob is still
    `.incomplete` - plus every `from_single_file` that is a URL, and every
    repo a task step declares through its command's `required_weights` hook
    (#975, dw/tasks/weights.py), listed once when a pipeline step loads it
    too. A task that fetches one file of a repo declares the file, keyed by
    `(repo, filename, revision)`: it is checked in the cache file by file
    (`_cached_file` - a pinned commit needs no `refs/main`, and another
    cached file of the repo does not stand in for it), sized by a HEAD of
    that file, and dropped when the same walk needs the repo whole. Sizes come
    from the hub when asked and are None whenever it does not answer - an
    offline box is a state, not an error, so nothing here raises or logs
    above debug.
    """
    names = []
    variants = {}
    urls = []
    files = []
    _collect_sources(expanded, names, variants, urls, files)
    _collect_task_weights(expanded, names, files, urls)
    for path, _arguments in _sub_workflow_paths(expanded):
        raw = read_sub_workflow(path, base_dir, workflow_dir)
        if raw is None:
            continue
        try:
            child = json.loads(raw)
        except ValueError:
            continue
        _collect_sources(child, names, variants, urls, files)
        _collect_task_weights(child, names, files, urls)
    present = {repo.get("repo_id") for repo in scan_models(cache_dir).get("repos", [])}
    required = []
    for name in names:
        # A name not shaped like a hub id is a local checkout - decided by
        # shape, never by touching the disk: the name came from the request
        # body, and a free pre-flight must not be a directory-existence oracle
        if not _is_repo_id(name):
            continue
        if name in present and not repo_download_incomplete(
            name, cache_dir, variant=variants.get(name)
        ):
            continue
        entry = {"repo": name, "gb": None, "gated": None, "access_blocked": None}
        if lookup_sizes:
            entry["gb"], entry["gated"], entry["access_blocked"] = _model_info(name)
        required.append(entry)
    for repo, filename, revision in files:
        # A whole-repo entry for the same repo already covers the file
        if repo in names or not _is_repo_id(repo):
            continue
        if _cached_file(repo, filename, revision or "main", cache_dir) is not None:
            continue
        entry = {
            "repo": repo,
            "filename": filename,
            "revision": revision,
            "gb": None,
            "gated": None,
            "access_blocked": None,
        }
        if lookup_sizes:
            entry["gb"], entry["access_blocked"] = _file_info(repo, filename, revision)
        required.append(entry)
    for url in urls:
        required.append(
            {
                "repo": None,
                "url": url,
                "gb": None,
                "gated": None,
                "access_blocked": None,
            }
        )
    return required


def _collect_sources(tree, names, variants, urls, files=None):
    """Every from_pretrained source in a tree, first-seen order, deduplicated.
    `variants` collects each name's requested `variant` (first-seen), used to
    tell a repo that is merely missing its fp16 files from one that is fully
    cached without them.

    A `loras` entry counts too. It carries its repo under `model_name`
    directly rather than inside a `from_pretrained_arguments` block, so the
    walk missed it: `templates/ltx2/reference-sheet` on a box that had
    every base weight but not the IC-LoRA answered `downloads_required:
    []` and then pulled it mid-run, which is the one question the field
    exists to answer (found verifying #151).

    A `loras` entry that names its `weight_name` loads that one file, so with
    `files` given it is keyed there as `(repo, [subfolder/]weight_name, None)`
    and sized as the file, not as the repo it sits in (#1088). A weight name
    still holding a reference is no file yet: the repo stands for it.
    """
    if isinstance(tree, dict):
        for lora in tree.get(LORAS_KEY) or []:
            name = lora.get(MODEL_NAME_KEY) if isinstance(lora, dict) else None
            if not isinstance(name, str):
                continue
            weight = lora.get(WEIGHT_NAME_KEY)
            if (
                files is not None
                and isinstance(weight, str)
                and not references.is_ref(references.DEFERRED, weight)
            ):
                subfolder = lora.get(SUBFOLDER_KEY)
                if isinstance(subfolder, str) and subfolder:
                    weight = f"{subfolder.strip('/')}/{weight}"
                key = (name, weight, None)
                if key not in files:
                    files.append(key)
            elif name not in names:
                names.append(name)
        source = tree.get(FROM_PRETRAINED_KEY)
        if isinstance(source, dict):
            name = source.get(MODEL_NAME_KEY)
            if isinstance(name, str) and name not in names:
                names.append(name)
                variant = source.get(VARIANT_KEY)
                if isinstance(variant, str):
                    variants[name] = variant
            single = source.get(SINGLE_FILE_KEY)
            if isinstance(single, str) and _is_url(single) and single not in urls:
                urls.append(single)
        _collect_component_overrides(tree.get(LOAD_COMPONENTS_KEY), names)
        for value in tree.values():
            _collect_sources(value, names, variants, urls, files)
    elif isinstance(tree, list):
        for value in tree:
            _collect_sources(value, names, variants, urls, files)


def _collect_component_overrides(arguments, names):
    """The repos a modular step's `load_components.pretrained_model_name_or_path`
    names: a string for every component, or a dict keyed by component name
    with an optional `default`. Diffusers merges it over each component's
    spec, so the repo is pulled mid-run (#981). The base repo is still listed
    whole - the subfolder an override replaces is not subtracted."""
    if not isinstance(arguments, dict):
        return
    value = arguments.get(OVERRIDE_KEY)
    for name in value.values() if isinstance(value, dict) else [value]:
        if isinstance(name, str) and name not in names:
            names.append(name)


def _collect_task_weights(tree, names, files, urls):
    """Every repo, file or URL a task step in a tree declares, appended after
    the pipeline sources and deduplicated against them - a file as a
    `(repo, filename, revision)` key in `files`. Each `task` block's
    command is looked up by name and its hook called on the block's
    arguments - expanded at the top level; a composed child's are its raw
    definition, so a `variable:` value there is no repo and gives no entry.

    A hook that raises costs its entries, never the validate: the plan
    answers what it can name, and the run still loads what it loads.
    """
    if isinstance(tree, dict):
        task = tree.get("task")
        if isinstance(task, dict) and isinstance(task.get("command"), str):
            arguments = task.get("arguments")
            for entry in _task_weights(
                task["command"], arguments if isinstance(arguments, dict) else {}
            ):
                repo, url = entry.get("repo"), entry.get("url")
                filename = entry.get("filename")
                if isinstance(repo, str) and isinstance(filename, str):
                    key = (repo, filename, entry.get("revision"))
                    if key not in files:
                        files.append(key)
                elif isinstance(repo, str) and repo not in names:
                    names.append(repo)
                elif repo is None and isinstance(url, str) and _is_url(url):
                    if url not in urls:
                        urls.append(url)
        for value in tree.values():
            _collect_task_weights(value, names, files, urls)
    elif isinstance(tree, list):
        for value in tree:
            _collect_task_weights(value, names, files, urls)


def _task_weights(command, arguments):
    # Imported here: the registry loads task modules only when a workflow has
    # a task step, so a pipeline-only validate pays nothing for it
    from .tasks.registry import required_weights_hook

    try:
        hook = required_weights_hook(command)
        entries = hook(copy.deepcopy(arguments)) if hook is not None else []
    except Exception as error:
        logger.debug(f"required_weights for task {command!r} failed: {error}")
        return []
    return [entry for entry in entries or [] if isinstance(entry, dict)]


def gate_warnings(downloads_required):
    """One line per required download this box's token is blocked from -
    the pre-flight signal #186 asked for, so a gated repo is a
    `validate_workflow`-visible condition rather than a 403 the run only
    discovers after it has loaded everything ahead of that step."""
    return [
        f"{entry['repo']} is gated and not accessible with this box's "
        "Hugging Face token - accept its license at "
        f"https://huggingface.co/{entry['repo']} before running this workflow"
        for entry in downloads_required
        # A URL entry has no repo to send anyone to
        if entry.get("access_blocked") and entry.get("repo")
    ]


def _is_repo_id(name):
    try:
        validate_repo_id(name)
        return True
    except HFValidationError:
        return False


def _is_url(value):
    if not value.startswith(("http://", "https://")):
        return False
    try:
        validate_url(value)
        return True
    except Exception:
        return False


def _model_info(name):
    """A repo's size in GiB, gate status, and whether this box's token is
    blocked from it - a `(gb, gated, access_blocked)` triple.

    `gated` is `model_info`'s own field (`False` / `"auto"` / `"manual"`),
    readable only when the call succeeds - and the hub answers `model_info`
    for a gated repo regardless of whether this token has been granted
    access, since that call serves metadata rather than file bytes (found
    verifying #186: `access_blocked` came back `false` for repos this box's
    token was actually refused on). A `GatedRepoError` straight out of
    `model_info` is still a real signal - some other endpoint behind it
    checked and refused - but its absence proves nothing, so a `gated` repo
    that got this far is checked for real with a HEAD request against one of
    its own files (`_probe_gate_blocked`), the same request class a run's
    actual load would make and the one place the hub's 403 for "gated, token
    not accepted" actually appears.
    """
    try:
        info = model_info(name, files_metadata=True, timeout=SIZE_LOOKUP_TIMEOUT)
    except GatedRepoError as e:
        logger.debug(f"Gate not accepted for {name}: {e}")
        return None, True, True
    except Exception as e:
        logger.debug(f"No size for {name}: {e}")
        return None, None, None
    total = sum(s.size for s in (info.siblings or []) if getattr(s, "size", None))
    gb = round(total / GIB, 1) if total else None
    gated = getattr(info, "gated", False) or False
    access_blocked = False
    if gated:
        access_blocked = _probe_gate_blocked(name, info.siblings or [])
    return gb, gated, access_blocked


def _file_info(repo, filename, revision):
    """One file's size in GiB and whether this box's token is blocked from it
    - a `(gb, access_blocked)` pair from a HEAD of the file itself, the call
    `_probe_gate_blocked` makes. `_model_info` would size the whole repo at
    main, overstating one file of a multi-file repo. A tiny file reads 0.01
    rather than 0, which would look like nothing to fetch."""
    try:
        metadata = get_hf_file_metadata(
            hf_hub_url(repo, filename, revision=revision),
            timeout=SIZE_LOOKUP_TIMEOUT,
        )
    except GatedRepoError as e:
        logger.debug(f"Gate not accepted for {repo}/{filename}: {e}")
        return None, True
    except Exception as e:
        logger.debug(f"No size for {repo}/{filename}: {e}")
        return None, None
    size = getattr(metadata, "size", None)
    gb = max(round(size / GIB, 2), 0.01) if size else None
    return gb, False


def _probe_gate_blocked(name, siblings):
    """HEAD one real file of a gated repo to see whether this box's token is
    actually accepted - `model_info` succeeding says nothing either way
    (#186). `None` when there is no file to probe or the probe fails for a
    reason other than the gate, since that is "unknown", not "not blocked".
    """
    filename = next(
        (s.rfilename for s in siblings if getattr(s, "rfilename", None)), None
    )
    if filename is None:
        return None
    try:
        get_hf_file_metadata(hf_hub_url(name, filename), timeout=SIZE_LOOKUP_TIMEOUT)
        return False
    except GatedRepoError as e:
        logger.debug(f"Gate not accepted for {name}: {e}")
        return True
    except Exception as e:
        logger.debug(f"Gate probe inconclusive for {name}: {e}")
        return None


def _sub_workflow_paths(expanded):
    """The local (non-builtin) sub-workflow path and composing arguments of
    every composing step, as (path, arguments) - `expanded` has already
    substituted and expanded `for_each`, so each occurrence carries the
    concrete arguments that step actually passes the child (#341)."""
    for step in expanded.get("steps") or []:
        reference = step.get("workflow") if isinstance(step, dict) else None
        path = reference.get("path") if isinstance(reference, dict) else None
        if isinstance(path, str) and not references.is_ref(references.BUILTIN, path):
            arguments = reference.get("arguments")
            yield path, arguments if isinstance(arguments, dict) else {}
