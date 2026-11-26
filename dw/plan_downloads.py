"""What a run would have to download before its first step: the hub repos,
single-file checkpoints and task weights the expanded definition (and each
composed child) names that the cache does not already hold, with each one's
size and gate status when the hub is asked (#382, #975, #981).

Everything here is a walk over the expanded definition: a `model_name` or
`loras` repo under a `from_pretrained_arguments`, a `from_single_file` URL, a
`load_components` repo override, the repos a task declares through its
command's `required_weights` hook, and the face models a chain whose `check`
is on reads for its face_drop. A modular `model_name` whose `load_components`
moves components to another repo is planned by component from its
`modular_model_index.json` (#1343), not whole. The cache is read through `hub_cache` and
`quantized_cache`, never a directory probe of the plan's own - a free
pre-flight must not be an existence oracle for paths on the box. Nothing here
knows a model; `dw/plan.py` assembles the answer into the plan beside the
estimate and the fingerprint, and this module never imports it back.
"""

import copy
import json
import logging
from pathlib import Path

from huggingface_hub import model_info
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import (
    GatedRepoError,
    HFValidationError,
    RepositoryNotFoundError,
    build_hf_headers,
    get_session,
    validate_repo_id,
)

from .hub_cache import index_components, repo_download_incomplete, scan_models
from .quantized_cache import _SPEC_FIELDS, _cached_file, _per_component
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
CONFIGURATION_KEY = "configuration"
REVISION_KEY = "revision"
# The file a modular repo names its components' loading specs in
MODULAR_INDEX = "modular_model_index.json"
# The file a pipeline repo names its components in
MODEL_INDEX = "model_index.json"


class _RepoNotFound(Exception):
    """The hub answered that a repo does not exist - raised by `_model_info`
    so its `(gb, gated, access_blocked)` shape stays as it is."""


def downloads_required(expanded, base_dir, workflow_dir, cache_dir, lookup_sizes):
    """The hub repos and checkpoint URLs the run would fetch before its
    first step: every `model_name` in the expanded definition (and in each
    composed child) that `scan_models` does not find, plus every one it does
    find but whose cached copy is left over from an interrupted pull
    (#382) - a component the load needs is missing, or a blob is still
    `.incomplete` - plus every `from_single_file` that is a URL, and every
    repo a task step declares through its command's `required_weights` hook
    (#975, dw/tasks/weights.py) - and the face models a checked chain's
    face_drop reads - listed once when a pipeline step loads it
    too. A task that fetches one file of a repo declares the file, keyed by
    `(repo, filename, revision)`: it is checked in the cache file by file
    (`_cached_file` - a pinned commit needs no `refs/main`, and another
    cached file of the repo does not stand in for it), sized by a HEAD of
    that file, and dropped when the same walk needs the repo whole. Sizes come
    from the hub when asked and are None whenever it does not answer - an
    offline box is a state, not an error, so nothing here raises or logs
    above debug.
    """
    names, variants, urls, files, partials, whole, skips = _walk(
        expanded, base_dir, workflow_dir
    )
    present = {repo.get("repo_id") for repo in scan_models(cache_dir).get("repos", [])}
    required = []
    listed_whole = set()
    for name in names:  # grows: a component the index sends elsewhere
        # A name not shaped like a hub id is a local checkout - decided by
        # shape, never by touching the disk: the name came from the request
        # body, and a free pre-flight must not be a directory-existence oracle
        if not _is_repo_id(name):
            continue
        if name in partials and name not in whole:
            planned = _component_plan(
                name,
                *partials[name],
                cache_dir,
                lookup_sizes,
                variant=variants.get(name),
            )
            if planned is not None:
                entries, elsewhere = planned
                required.extend(entries)
                # A component the index points at another repo, not overridden
                names.extend(repo for repo in elsewhere if repo not in names)
                continue
        listed_whole.add(name)
        extra = {"skip": skips[name]} if skips.get(name) else {}
        if name in present and not repo_download_incomplete(
            name, cache_dir, variant=variants.get(name), **extra
        ):
            continue
        entry = {"repo": name, "gb": None, "gated": None, "access_blocked": None}
        if lookup_sizes:
            try:
                entry["gb"], entry["gated"], entry["access_blocked"] = _model_info(
                    name, variants.get(name), skips.get(name)
                )
            except _RepoNotFound:
                entry["not_found"] = True
        required.append(entry)
    for repo, filename, revision in files:
        # A whole-repo entry for the same repo already covers the file; one
        # planned by component does not - an adapter is no component
        if repo in listed_whole or not _is_repo_id(repo):
            continue
        if _cached_file(repo, filename, revision or "main", cache_dir) is not None:
            continue
        if revision is None and _blob_cached(repo, filename, cache_dir, lookup_sizes):
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


def repos_named(expanded, base_dir, workflow_dir):
    """Every Hub repo the run names, deduplicated in first-seen order: the
    walk `downloads_required` makes - step and component `model_name`s,
    `loras`, `load_components` overrides, task weights (a file-level weight
    counts as its repo) and composed children - with no cache or hub I/O.

    `expanded` is the definition after its arguments are substituted, so a
    `variable:`-supplied `model_name` is its resolved repo. A name not shaped
    like a hub id - a local path, a URL, a reference left unresolved - is
    dropped by shape, as the plan drops it. What dw/model_use.py stamps when
    the run starts (#1445)."""
    names, _variants, _urls, files, _partials, _whole, _skips = _walk(
        expanded, base_dir, workflow_dir
    )
    named = []
    for name in [*names, *(repo for repo, _filename, _revision in files)]:
        if isinstance(name, str) and name not in named and _is_repo_id(name):
            named.append(name)
    return named


def _walk(expanded, base_dir, workflow_dir):
    """The sources the expanded definition and each composed child name, as
    `(names, variants, urls, files, partials, whole, skips)` - see `_collect_sources`
    and `_collect_task_weights`."""
    names = []
    variants = {}
    urls = []
    files = []
    partials = {}
    whole = set()
    skips = {}
    _collect_sources(expanded, names, variants, urls, files, partials, whole, skips)
    _collect_task_weights(expanded, names, files, urls, whole)
    for path, _arguments in _sub_workflow_paths(expanded):
        raw = read_sub_workflow(path, base_dir, workflow_dir)
        if raw is None:
            continue
        try:
            child = json.loads(raw)
        except ValueError:
            continue
        _collect_sources(child, names, variants, urls, files, partials, whole, skips)
        _collect_task_weights(child, names, files, urls, whole)
    return names, variants, urls, files, partials, whole, skips


def _collect_sources(
    tree, names, variants, urls, files=None, partials=None, whole=None, skips=None
):
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
    still holding a reference is no file yet, and a name `file_entries`
    refuses is none either: the repo stands for it.

    A modular step whose `load_components` overrides where some components
    load from (`pretrained_model_name_or_path`) does not need its base repo
    whole - Pika H3 Flex's five weight components come from MiniMax-H3, so
    only its schedulers, tokenizer and processor come from Pika's own 220 GB
    repo (#1343). With `partials` given, such a base is keyed there as
    `name -> (load_components, revision)` and planned by component
    (`_component_plan`); a base any other source names with no override goes
    in `whole`, and is listed whole as before.
    """
    if isinstance(tree, dict):
        for lora in tree.get(LORAS_KEY) or []:
            _collect_lora_source(lora, names, files, whole)
        source = tree.get(FROM_PRETRAINED_KEY)
        load_components = _step_load_components(tree)
        if isinstance(source, dict):
            name = source.get(MODEL_NAME_KEY)
            if isinstance(name, str):
                if name not in names:
                    names.append(name)
                    variant = source.get(VARIANT_KEY)
                    if isinstance(variant, str):
                        variants[name] = variant
                if skips is not None:
                    # A component passed as null (`"safety_checker": null`) is
                    # not loaded, so its folder is not fetched (#1580)
                    skips.setdefault(name, set()).update(
                        key
                        for key, value in source.items()
                        if value is None and isinstance(key, str)
                    )
                overrides = isinstance(load_components, dict) and _moves_to_a_repo(
                    load_components.get(OVERRIDE_KEY)
                )
                if partials is not None and overrides:
                    revision = source.get(REVISION_KEY)
                    partials.setdefault(
                        name,
                        (
                            load_components,
                            revision if isinstance(revision, str) else None,
                        ),
                    )
                elif whole is not None:
                    whole.add(name)
            single = source.get(SINGLE_FILE_KEY)
            if isinstance(single, str) and _is_url(single) and single not in urls:
                urls.append(single)
        _collect_component_overrides(tree.get(LOAD_COMPONENTS_KEY), names, whole)
        for value in tree.values():
            _collect_sources(
                value, names, variants, urls, files, partials, whole, skips
            )
    elif isinstance(tree, list):
        for value in tree:
            _collect_sources(
                value, names, variants, urls, files, partials, whole, skips
            )


def _moves_to_a_repo(value):
    """Whether a `pretrained_model_name_or_path` override sends a component to
    a hub repo. A local path (not shaped like a hub id) moves it off the base
    without fetching anything, so the base is planned whole, exactly as with
    no override: its other components still load from it (#1490)."""
    values = value.values() if isinstance(value, dict) else [value]
    return any(isinstance(v, str) and _is_repo_id(v) for v in values)


def _step_load_components(tree):
    """The `load_components` block of the step `tree` is, wherever it sits:
    beside `from_pretrained_arguments` or under the step's `configuration`."""
    configuration = tree.get(CONFIGURATION_KEY)
    if isinstance(configuration, dict) and isinstance(
        configuration.get(LOAD_COMPONENTS_KEY), dict
    ):
        return configuration[LOAD_COMPONENTS_KEY]
    value = tree.get(LOAD_COMPONENTS_KEY)
    return value if isinstance(value, dict) else None


def _collect_lora_source(lora, names, files, whole=None):
    """One `loras` entry into `files` (its single weight file, when it names
    one and `files` is collected) or `names` (its whole repo otherwise)."""
    name = lora.get(MODEL_NAME_KEY) if isinstance(lora, dict) else None
    if not isinstance(name, str):
        return
    weight = lora.get(WEIGHT_NAME_KEY)
    entries = []
    if (
        files is not None
        and isinstance(weight, str)
        and not references.is_ref(references.DEFERRED, weight)
    ):
        subfolder = lora.get(SUBFOLDER_KEY)
        if isinstance(subfolder, str) and subfolder:
            weight = f"{subfolder.strip('/')}/{weight}"
        from .tasks.weights import file_entries

        entries = file_entries(name, weight, None)
    for entry in entries:
        key = (entry["repo"], entry["filename"], entry["revision"])
        if key not in files:
            files.append(key)
    if not entries:
        if name not in names:
            names.append(name)
        if whole is not None:
            whole.add(name)


def _collect_component_overrides(arguments, names, whole=None):
    """The repos a modular step's `load_components.pretrained_model_name_or_path`
    names: a string for every component, or a dict keyed by component name
    with an optional `default`. Diffusers merges it over each component's
    spec, so the repo is pulled mid-run (#981). An override repo is listed
    whole; the base repo it stands in for is planned by component when the
    walk knows it (`_collect_sources`, `_component_plan`)."""
    if not isinstance(arguments, dict):
        return
    value = arguments.get(OVERRIDE_KEY)
    for name in value.values() if isinstance(value, dict) else [value]:
        if isinstance(name, str):
            if name not in names:
                names.append(name)
            if whole is not None:
                whole.add(name)


def _collect_task_weights(tree, names, files, urls, whole=None):
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
            _add_weights(
                _task_weights(
                    task["command"], arguments if isinstance(arguments, dict) else {}
                ),
                names,
                files,
                urls,
                whole,
            )
        chain = tree.get("chain")
        if isinstance(chain, dict) and _chain_checked(chain.get("check")):
            _add_weights(_chain_check_weights(), names, files, urls, whole)
        for value in tree.values():
            _collect_task_weights(value, names, files, urls, whole)
    elif isinstance(tree, list):
        for value in tree:
            _collect_task_weights(value, names, files, urls, whole)


def _add_weights(entries, names, files, urls, whole=None):
    """Weight entries into the walk's lists, deduplicated: a file as a
    `(repo, filename, revision)` key, a whole repo by name, a bare URL."""
    for entry in entries:
        repo, url = entry.get("repo"), entry.get("url")
        filename = entry.get("filename")
        if isinstance(repo, str) and isinstance(filename, str):
            key = (repo, filename, entry.get("revision"))
            if key not in files:
                files.append(key)
        elif isinstance(repo, str):
            if repo not in names:
                names.append(repo)
            if whole is not None:
                whole.add(repo)
        elif repo is None and isinstance(url, str) and _is_url(url):
            if url not in urls:
                urls.append(url)


def _chain_checked(check):
    """Whether a chain's `check` is on: true or an object. A reference left
    unresolved (a composed child's raw definition) is not known to be."""
    return check is True or isinstance(check, dict)


def _chain_check_weights():
    """A checked chain's face_drop reads YuNet and SFace, fetched at run time
    like a task's (`chain_check.face_base`), so the plan lists them too."""
    from .tasks.face_drift import required_weights

    try:
        return [entry for entry in required_weights({}) if isinstance(entry, dict)]
    except Exception as error:
        logger.debug(f"required_weights for the chain check failed: {error}")
        return []


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


def missing_repo_errors(downloads_required, value_paths=()):
    """One validation error per required repo the hub says does not exist -
    a misspelled or renamed `model_name` the run would only find when it
    reached that step. A private repo this box's token cannot see gets the
    same answer from the hub, and fails the run the same way.

    `value_paths` is the `(path, string)` pairs of what the caller wrote:
    the error points at the first that names the repo. One written nowhere
    there (a composed child's, a task's own weight) has no path to give and
    keeps the bare `model_name`."""
    written = {}
    for path, value in value_paths:
        written.setdefault(value, path)
    return [
        {
            "path": written.get(entry["repo"], "model_name"),
            "message": (
                f"'{entry['repo']}' is not a model repo on Hugging Face (or is "
                "private and this box's token cannot see it) - check the "
                "model_name"
            ),
        }
        for entry in downloads_required
        if entry.get("not_found") and entry.get("repo")
    ]


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


def _blob_cached(repo, filename, cache_dir, check_hub):
    """Whether `filename` sits in some snapshot of the repo other than the
    one `refs/main` names (#1392). The hub moves `main` when any file of the
    repo changes; the loader finds the unchanged file's blob already cached
    and links it into the new snapshot without a download, while
    `_cached_file` reads only the `refs/main` snapshot and calls the file
    missing. With `check_hub` the blob must be the one the hub serves now
    (an LFS etag is its blob's sha256); offline, or when the hub does not
    answer, the cached copy stands, as it will for the run."""
    from huggingface_hub.file_download import repo_folder_name

    from .quantized_cache import _hub_cache_dir

    try:
        snapshots = (
            Path(_hub_cache_dir(cache_dir))
            / repo_folder_name(repo_id=repo, repo_type="model")
            / "snapshots"
        )
        blobs = set()
        for snapshot in snapshots.iterdir() if snapshots.is_dir() else []:
            path = snapshot / filename
            if path.is_file():
                blobs.add(path.resolve().name)
    except (OSError, ValueError):
        return False
    if not blobs:
        return False
    if check_hub:
        try:
            etag = get_hf_file_metadata(
                hf_hub_url(repo, filename), timeout=SIZE_LOOKUP_TIMEOUT
            ).etag
        except Exception:
            return True
        etag = etag.strip('"') if isinstance(etag, str) else ""
        if len(etag) == 64:
            return etag in blobs
    return True


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


def _component_plan(
    base, load_components, base_revision, cache_dir, lookup_sizes, variant=None
):
    """What a modular step fetches from `base` when its `load_components`
    sends some components elsewhere (#1343): an `(entries, elsewhere)` pair,
    or `None` when the repo's `modular_model_index.json` can't be read, which
    lists `base` whole as before - an overstatement, never a miss.

    The index names each component's loading spec; `load_components` is laid
    over it per component (`_per_component`, a dict keyed by name with a
    `default`), the way diffusers merges them at load. A component left on
    `base` is kept, by subfolder; one moved to another repo is that repo's
    entry, listed whole by the walk. `elsewhere` is a repo the index itself
    points a component at, which no override named. `entries` is one
    `{"repo", "components", "revision", "gb", "gated", "access_blocked"}`
    per revision with something left to fetch - none when it is all cached.
    """
    index = _read_modular_index(base, base_revision, cache_dir, lookup_sizes)
    if not isinstance(index, dict):
        return None
    by_revision = {}
    classes = {}
    elsewhere = []
    for component, value in index.items():
        if not (isinstance(value, (list, tuple)) and len(value) == 3):
            continue
        spec = value[2] if isinstance(value[2], dict) else {}
        resolved = _per_component(load_components, component)
        overlay = {
            field: resolved[field]
            for field in _SPEC_FIELDS
            if isinstance(resolved.get(field), str)
        }
        repo = overlay.get(OVERRIDE_KEY) or spec.get(OVERRIDE_KEY) or base
        if repo != base:
            if OVERRIDE_KEY not in overlay and isinstance(repo, str):
                elsewhere.append(repo)
            continue
        subfolder = overlay.get(SUBFOLDER_KEY) or spec.get(SUBFOLDER_KEY) or component
        revision = overlay.get(REVISION_KEY) or spec.get(REVISION_KEY) or base_revision
        by_revision.setdefault(revision, set()).add(subfolder)
        if isinstance(value[1], str):
            classes[(revision, subfolder)] = value[1]
    entries = []
    for revision, subfolders in by_revision.items():
        entry = _components_entry(
            base,
            revision,
            sorted(subfolders),
            cache_dir,
            lookup_sizes,
            classes={s: classes.get((revision, s)) for s in subfolders},
            variant=variant,
        )
        if entry is not None:
            entries.append(entry)
    return entries, elsewhere


def _components_entry(
    base, revision, subfolders, cache_dir, lookup_sizes, classes=None, variant=None
):
    """The entry for `subfolders` of `base` at `revision`, or `None` when
    every file of them is cached. Sized from the hub's file list at that
    revision, counting only the files not cached; offline, a subfolder whose
    snapshot directory holds files counts as cached."""
    entry = {
        "repo": base,
        "components": subfolders,
        "revision": revision,
        "gb": None,
        "gated": None,
        "access_blocked": None,
    }
    # A component folder the snapshot already holds whole is cached, the way
    # the whole-repo check reads a modular repo: the hub's file list also
    # names files a load never fetches (#1488)
    classes = classes or {}
    subfolders = [
        s
        for s in subfolders
        if not _component_held(base, revision, s, cache_dir, classes.get(s), variant)
    ]
    if not subfolders:
        return None
    entry["components"] = subfolders
    prefixes = tuple(f"{subfolder}/" for subfolder in subfolders)
    if lookup_sizes:
        try:
            info = model_info(
                base,
                revision=revision,
                files_metadata=True,
                timeout=SIZE_LOOKUP_TIMEOUT,
            )
        except GatedRepoError as e:
            logger.debug(f"Gate not accepted for {base}: {e}")
            entry["gated"] = entry["access_blocked"] = True
            return entry
        except RepositoryNotFoundError as e:
            logger.debug(f"No such repo {base}: {e}")
            entry["not_found"] = True
            return entry
        except Exception as e:
            logger.debug(f"No file list for {base}: {e}")
            info = None
        if info is not None:
            missing = [
                s
                for s in (info.siblings or [])
                if getattr(s, "rfilename", "").startswith(prefixes)
                and _cached_file(base, s.rfilename, revision or "main", cache_dir)
                is None
            ]
            if not missing:
                return None
            total = sum(getattr(s, "size", None) or 0 for s in missing)
            entry["gb"] = max(round(total / GIB, 2), 0.01) if total else None
            entry["gated"] = getattr(info, "gated", False) or False
            entry["access_blocked"] = (
                _probe_gate_blocked(base, missing) if entry["gated"] else False
            )
            return entry
    index_path = _cached_file(base, MODULAR_INDEX, revision or "main", cache_dir)
    if index_path is not None:
        # The snapshot is listed, never probed by name: a subfolder can come
        # from the request body, and must not make this an existence oracle
        held = {
            child.name
            for child in Path(index_path).parent.iterdir()
            if child.is_dir() and any(child.iterdir())
        }
        if set(subfolders) <= held:
            return None
    return entry


def _component_held(
    base, revision, subfolder, cache_dir, class_name=None, variant=None
):
    """Whether `subfolder` of `base`'s snapshot at `revision` is on disk and
    complete by the #894 component rule. The snapshot is found through the
    cached modular index, never by probing a name from the request. The
    component's `class_name` (from the index) and the step's `variant` go to
    the rule, so a weight-bearing folder holding only configs is not held.
    A component with no known class is not held: the list overstates."""
    if not isinstance(class_name, str):
        return False
    from .hub_cache import _component_incomplete

    index_path = _cached_file(base, MODULAR_INDEX, revision or "main", cache_dir)
    if index_path is None:
        return False
    snapshot = Path(index_path).parent
    try:
        held = {child.name for child in snapshot.iterdir() if child.is_dir()}
        return subfolder in held and not _component_incomplete(
            str(snapshot / subfolder), variant, class_name
        )
    except OSError:
        return False


def _read_modular_index(base, revision, cache_dir, lookup_sizes):
    """`base`'s `modular_model_index.json` at `revision` as a dict: the cached
    copy, else (with `lookup_sizes`) the hub's, read without caching it.
    `None` when neither answers."""
    path = _cached_file(base, MODULAR_INDEX, revision or "main", cache_dir)
    try:
        if path is not None:
            return json.loads(Path(path).read_text())
        if lookup_sizes:
            return _fetch_modular_index(base, revision)
    except Exception as e:
        logger.debug(f"No modular index for {base}: {e}")
    return None


def _fetch_modular_index(base, revision):
    """The hub's `modular_model_index.json` for `base` at `revision`."""
    return _fetch_index(base, MODULAR_INDEX, revision)


def _fetch_index(repo, filename, revision=None):
    """One of the hub's index files for `repo`, read without caching it."""
    response = get_session().get(
        hf_hub_url(repo, filename, revision=revision),
        headers=build_hf_headers(),
        timeout=SIZE_LOOKUP_TIMEOUT,
        follow_redirects=True,
    )
    response.raise_for_status()
    return json.loads(response.text)


def _pipeline_components(repo):
    """The component names `repo`'s `model_index.json` declares
    (`hub_cache.index_components`), or None when it can't be read."""
    try:
        return set(index_components(_fetch_index(repo, MODEL_INDEX)))
    except Exception as e:
        logger.debug(f"No model index for {repo}: {e}")
        return None


def _model_info(name, variant=None, skip=None):
    """A repo's size in GiB (what `from_pretrained` fetches, `_load_bytes`), gate status, and whether this box's token is
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
    except RepositoryNotFoundError as e:
        # After GatedRepoError, which subclasses it. The hub's answer, not a
        # failure to ask - an offline box raises something else and stays
        # unknown - so the run would fail on this repo too
        logger.debug(f"No such repo {name}: {e}")
        raise _RepoNotFound(name) from e
    except Exception as e:
        logger.debug(f"No size for {name}: {e}")
        return None, None, None
    gb = _repo_gb(name, info.siblings or [], variant, skip)
    gated = getattr(info, "gated", False) or False
    access_blocked = False
    if gated:
        access_blocked = _probe_gate_blocked(name, info.siblings or [])
    return gb, gated, access_blocked


def _repo_gb(name, siblings, variant=None, skip=None):
    """`_load_bytes` in GiB, or None when the size can't be worked out: no
    file sizes, a `model_index.json` that can't be read, or no file set
    diffusers would accept (a variant the repo does not have). Never 0.0,
    which would read as nothing to fetch (#1745)."""
    components = None
    if any(getattr(s, "rfilename", None) == MODEL_INDEX for s in siblings):
        components = _pipeline_components(name)
        if components is None:
            return None
    try:
        total = _load_bytes(siblings, variant, skip, components)
    except (EnvironmentError, ValueError) as e:
        logger.debug(f"No file set for {name}: {e}")
        return None
    if not total:
        return None
    return round(total / GIB, 1) or max(round(total / GIB, 2), 0.01)


def _load_bytes(siblings, variant=None, skip=None, components=None):
    """Bytes a diffusers `from_pretrained` of a repo fetches, from its file
    list. A repo with a `model_index.json` is sized by the file set diffusers
    itself selects (`variant_compatible_siblings` under `_get_ignore_patterns`,
    the calls `DiffusionPipeline.download` makes): one weight format per
    component, the requested `variant` with a plain-file fallback, root
    checkpoints left out, a component named in `skip` (passed as null) left
    out. `components` is what the index declares: only those folders are held
    to the one-format rule, as in `download`, so a folder the index does not
    name (an ONNX or OpenVINO export) cannot push the selection to pickled
    weights (#1745); None takes every folder. A repo without a
    `model_index.json` (a modular or single-checkpoint
    repo) is summed whole, an upper bound (#1580). Diffusers is imported
    here, not at module load, to keep it off the validate import path."""
    sized = [s for s in siblings if getattr(s, "size", None)]
    sizes = {getattr(s, "rfilename", None) or "": s.size for s in sized}
    if "model_index.json" not in sizes:
        return sum(s.size for s in sized)
    from diffusers.pipelines.pipeline_loading_utils import (
        _get_ignore_patterns,
        filter_model_files,
        variant_compatible_siblings,
    )

    skip = set(skip or ())
    filenames = set(sizes)
    folders = {f.split("/")[0] for f in filenames if "/" in f}
    if components is not None:
        folders &= set(components)
    model_folders = {
        f.rpartition("/")[0] for f in filter_model_files(filenames)
    } & folders
    try:
        ignore = _get_ignore_patterns(
            list(skip),
            model_folders,
            filenames,
            True,
            False,
            False,
            False,
            False,
            variant,
        )
    except EnvironmentError:
        # No complete safetensors set: diffusers falls back to pickled weights
        ignore = _get_ignore_patterns(
            list(skip),
            model_folders,
            filenames,
            False,
            True,
            False,
            False,
            False,
            variant,
        )
    chosen, _ = variant_compatible_siblings(
        filenames, variant=variant, ignore_patterns=ignore
    )
    selected = set(chosen) | {"model_index.json"}
    selected |= {f for f in filenames if f.split("/")[0] in folders - model_folders}
    selected |= {f"{k}/config.json" for k in model_folders}
    return sum(
        sizes[f]
        for f in selected & filenames
        if f.split("/")[0] not in skip or "/" not in f
    )


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
