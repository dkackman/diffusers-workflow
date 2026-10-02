"""The workflow and prompt catalog as the server lists it: per-entry card
metadata, name resolution across the search path, and the two detail caches.

Module-level on purpose: the caches are keyed by file path and mtime and are
shared by every app and workspace in the process.
"""

import json
import os

from fastapi import HTTPException

from .. import references
from ..security import (
    InvalidInputError,
    SecurityError,
    validate_path,
    validate_prompt_reference,
)
from ..library import suggest_workflow_names
from .catalog_shape import derive_catalog_metadata
from .observed_cost import declared_drivers

# What each workflow produces and takes, for listing cards - cached by mtime
_workflow_detail_cache = {}


def _prune_missing(cache):
    """Forget cached files that are gone from disk.

    Pruning by what one listing named would be wrong here: the workflow
    cache is shared by every workspace, and a listing only ever sees one
    workspace's search path, so anything cached for another workspace would
    be thrown away and re-parsed on the next switch. Existence is the test
    that holds for all of them at once.

    `list(cache)` copies the keys in one step, so a request thread inserting
    meanwhile cannot break the scan, and `pop` tolerates an entry another
    thread already pruned.
    """
    for stale in [path for path in list(cache) if not os.path.exists(path)]:
        cache.pop(stale, None)


def collect_prompt_references(value):
    """Every stored-prompt name a definition references, at any depth - so
    deleting a prompt can warn which workflows would break."""
    found = set()
    if isinstance(value, str):
        name = references.ref_name(references.PROMPT, value)
        if name is not None:
            found.add(name.strip())
    elif isinstance(value, dict):
        for item in value.values():
            found |= collect_prompt_references(item)
    elif isinstance(value, list):
        for item in value:
            found |= collect_prompt_references(item)
    return found


def catalog_name_from_root(path, root):
    """The listing name a resolved workflow path has under a root.

    None when the path is not under the root after all - a name that does
    not name an entry is worse than no name for anything that later joins
    on it.
    """
    if root is None:
        return None
    relative = os.path.relpath(path, root)
    if relative.startswith(".."):
        return None
    return os.path.splitext(relative)[0].replace(os.sep, "/")


def catalog_name_for(path, source):
    """The listing name a resolved workflow path has within its source.

    None when the run came from an inline definition, or when the path is
    not under the source root after all - a name that does not name an
    entry is worse than no name for anything that later joins on it.
    """
    if source is None:
        return None
    return catalog_name_from_root(path, source.root)


def attach_observed(details, observed_costs, workspace_name=None):
    """Fold this box's own history into each detail, as `observed`.

    Separate from `workflow_details` because that cache is keyed on a file's
    mtime and this figure changes when no file has: a job finishing moves
    every number here. A detail carries `cost_drivers` and the defaults they
    take, which is everything the aggregate needs - the file is not read a
    second time.

    A detail's own `writable` says whether its entry is this workspace's own
    copy or a shared catalog one (#274): only the former is scoped to
    `workspace_name`, so two workspaces' saves of the same name do not leak
    into each other's figure, while a template or example still pools every
    workspace's runs of it, matching #154.
    """
    if observed_costs is None or not observed_costs.refresh():
        return details
    for name, detail in details.items():
        drivers = detail.get("cost_drivers") or {}
        # The shape `observed_for` reads: the drivers with their defaults,
        # and the variable names, which is what the no-drivers fallback
        # (default-arguments-only runs) compares a job's arguments against
        surrogate = {
            "cost_drivers": sorted(drivers),
            "variables": {
                **{variable: None for variable in detail.get("variable_names") or []},
                **drivers,
            },
        }
        workspace = workspace_name if detail.get("writable") else None
        observed = observed_costs.observed(
            name, surrogate, fresh=False, workspace=workspace
        )
        if observed:
            detail["observed"] = observed
    return details


def workflow_details(sources_by_name):
    """Per-workflow card metadata: output kinds, step and variable counts,
    and the variable names themselves - enough for an agent to pick a
    workflow and know what to pass it without fetching each candidate, and,
    for a list-driven workflow, what an entry of each list carries. The
    names but not their defaults: across the workflows on disk the defaults
    are an order of magnitude more payload, on a listing the UI reloads.

    Takes the name -> source mapping the search path produced, so each
    entry also says where it came from and whether it can be written to -
    what a client needs to decide between offering save and offering
    save-a-copy.
    """
    details = {}
    for name, source in sources_by_name.items():
        path = os.path.join(source.root, f"{name}.json")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        cached = _workflow_detail_cache.get(path)
        if cached and cached[0] == mtime:
            # The cached detail is placement-free; the origin and writability
            # are the source's, and a warm cache must still carry them or a
            # second listing loses the fields a client decides save-vs-copy on
            details[name] = {
                **cached[1],
                "origin": source.origin,
                "writable": source.writable,
            }
            continue
        try:
            with open(path, "r") as file:
                definition = json.load(file)
            kinds = sorted(
                {
                    step["result"]["content_type"].split("/")[0]
                    for step in definition.get("steps", [])
                    if isinstance(step.get("result"), dict)
                    and "content_type" in step["result"]
                }
            )
            variables = definition.get("variables", {}) or {}
            metadata = derive_catalog_metadata(definition)
            cost = definition.get("cost")
            detail = {
                "kinds": kinds,
                "steps": len(definition.get("steps", [])),
                "variables": len(variables),
                "variable_names": sorted(variables),
                "description": str(definition.get("description", "") or ""),
                # Empty for a template; a catalog name for a model config, which
                # is what lets a client show the two as different kinds of thing
                "configures": str(definition.get("configures", "") or ""),
                "prompt_refs": sorted(collect_prompt_references(definition)),
                "shape": metadata["shape"],
                "traits": metadata["traits"],
                "summary": metadata["summary"],
                "lists": metadata["lists"],
                # What a variable's value is allowed to be, so the rule is
                # read rather than guessed at (#96)
                "constraints": definition.get("variable_constraints") or {},
                # The variables the author says move this workflow's cost,
                # with what they default to - what buckets this box's own
                # runs into comparable ones (#93). Carried here so an
                # observed figure needs no second read of the file
                "cost_drivers": {
                    name: (definition.get("variables") or {}).get(name)
                    for name in declared_drivers(definition)
                },
                "cost": cost if isinstance(cost, list) and cost else None,
            }
        except Exception:
            detail = {
                "kinds": [],
                "steps": 0,
                "variables": 0,
                "variable_names": [],
                "description": "",
                "prompt_refs": [],
                "shape": "utility",
                "traits": [],
                "summary": "",
                "lists": {},
                "constraints": {},
                "cost_drivers": {},
                "cost": None,
            }
        _workflow_detail_cache[path] = (mtime, detail)
        # Cached by content, not by placement: the same file listed from a
        # different source keeps its parsed detail and gets fresh origins
        details[name] = {
            **detail,
            "origin": source.origin,
            "writable": source.writable,
        }
    _prune_missing(_workflow_detail_cache)
    # A model config names its template as a catalog name. Resolve it here,
    # where the whole listing is in hand, so a badge is a link to a real card
    # rather than a string - and say which name did not resolve. A config
    # also takes its shape and traits from the template: what it makes is
    # the template's business, what it costs is its own. Entries can be the
    # very dict cached above (a cache hit skips the copy at the origin
    # merge), so copy before mutating - otherwise a stale "not found yet"
    # verdict would stick in the cache and outlive the typo once the
    # template it names is added.
    for name, detail in details.items():
        named = detail.get("configures", "")
        if not named:
            continue
        detail = dict(detail)
        template = details.get(named)
        if template is None:
            detail["configures_missing"] = named
            detail["configures"] = ""
        else:
            detail["shape"] = template["shape"]
            detail["traits"] = list(template["traits"])
        details[name] = detail
    return details


def _unknown_workflow_detail(library, name):
    """'Unknown workflow: x', with a '- did you mean ...?' pointer when the
    catalog holds something `name` could be short for or a typo of (#397) -
    otherwise a caller has to spend a list_workflows call and guess the
    right shape/traits to find the entry it already knows by its short
    name."""
    detail = f"Unknown workflow: {name}"
    suggestions = suggest_workflow_names(library, name)
    if len(suggestions) == 1:
        detail += f" - did you mean {suggestions[0]}?"
    elif suggestions:
        detail += f" - did you mean one of: {', '.join(suggestions)}?"
    return detail


def resolve_readable_workflow(library, name):
    """The path a name has anywhere on the search path, and its root.

    Reads span every root - the workspace's own workflows, any examples
    directory, and the packaged builtins - front to back, so a workspace
    copy shadows the example it came from.
    """
    found = library.find(name)
    if found is None:
        raise HTTPException(
            status_code=404, detail=_unknown_workflow_detail(library, name)
        )
    return found


def resolve_writable_workflow(library, name):
    """Where a save goes: always the writable source, whatever the name
    currently resolves to.

    Saving a workflow opened from an example is not an overwrite of that
    example - it is a copy into the user's own library, which is what makes
    the read-only roots safe to browse and edit from.
    """
    source = library.writable_root()
    if source is None:
        raise HTTPException(
            status_code=409, detail="This server has no writable workflow directory"
        )
    path = library.path_in(source, name, allow_create=True)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown workflow: {name}")
    return path, source


def resolve_workflow_reference(workflow_path, library):
    """A submitted workflow_path, resolved to a file on disk, and the source
    it lives in - the same search path the /api/workflows CRUD routes read
    from, spanning every root rather than confining to one, since a run of
    an example is a read and reads are not confined to the writable root.

    Tried as a stored workflow name first - exactly what /api/workflows
    hands out, with or without .json and nested names included - so an
    agent can run what a listing gave it. A relative or absolute path that
    already names a file under one of the sources resolves the same way:
    os.path.abspath handles a path relative to the server's cwd, and
    `root_for_path` holds it to that root's containment check.

    Anything that resolves under no source - an unknown name, a traversal
    attempt, or a real file elsewhere on disk - is rejected with 400,
    rather than silently opened: a workflow_path is not a general
    filesystem path.

    Returns (None, None) when workflow_path itself is None - an inline
    workflow submission names no path to resolve.
    """
    if workflow_path is None:
        return None, None
    found = library.find(workflow_path)
    if found is not None:
        return found
    candidate = os.path.abspath(workflow_path)
    source = library.root_for_path(candidate)
    if source is not None:
        # The containment check re-applied to the path this returns, rather
        # than trusted from root_for_path's answer about it - and applied
        # before anything asks the filesystem about the path, so a
        # workflow_path outside every source cannot be used to find out
        # whether a file exists there
        try:
            confined = validate_path(candidate, source.root, allow_create=False)
        except SecurityError:
            confined = None
        if confined is not None and os.path.isfile(confined):
            return confined, source
    detail = f"workflow_path must name a workflow the server can reach: {workflow_path}"
    suggestions = suggest_workflow_names(library, workflow_path)
    if len(suggestions) == 1:
        detail += f" - did you mean {suggestions[0]}?"
    elif suggestions:
        detail += f" - did you mean one of: {', '.join(suggestions)}?"
    raise HTTPException(status_code=400, detail=detail)


# What each prompt says about itself, for listing cards - cached by mtime
_prompt_detail_cache = {}


def prompt_details(paths):
    """Per-prompt card metadata: description, intended model, tags - and
    the text itself, which the editors show as the tooltip wherever a
    prompt: reference stands in for it.

    Keyed by path rather than by name under one directory: the prompt
    library is a search path now, and two roots can hold the same name.
    """
    details = {}
    for name, path in paths.items():
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        cached = _prompt_detail_cache.get(path)
        if cached and cached[0] == mtime:
            details[name] = cached[1]
            continue
        try:
            with open(path, "r") as file:
                definition = json.load(file)
            detail = {
                "description": str(definition.get("description", "") or ""),
                "intended_model": str(definition.get("intended_model", "") or ""),
                "tags": [str(tag) for tag in definition.get("tags", []) or []],
                "text": str(definition.get("text", "") or ""),
            }
        except Exception:
            detail = {"description": "", "intended_model": "", "tags": [], "text": ""}
        _prompt_detail_cache[path] = (mtime, detail)
        details[name] = detail
    # By existence, not by what this listing named: the cache spans every
    # root on the search path, and one listing shows only the names that
    # were not shadowed
    _prune_missing(_prompt_detail_cache)
    return details


def matching_prompts(details, tag, intended_model):
    """The prompt names matching the filters, or None when no filter was given.

    Case-insensitive and exact per value: a `tags` entry or the whole
    `intended_model`, never a substring - `minimax-music` must not match
    `minimax-music3`, which is the confusion the one-spelling-per-family
    rule exists to prevent.
    """
    if tag is None and intended_model is None:
        return None
    wanted_tag = tag.lower() if tag is not None else None
    wanted_model = intended_model.lower() if intended_model is not None else None
    matches = set()
    for name, detail in details.items():
        if wanted_tag is not None and wanted_tag not in {
            str(each).lower() for each in detail.get("tags") or []
        }:
            continue
        if (
            wanted_model is not None
            and str(detail.get("intended_model") or "").lower() != wanted_model
        ):
            continue
        matches.add(name)
    return matches


def resolve_prompt_name(prompt_dir, name, allow_create=False):
    """The on-disk path for a prompt name, confined to prompt_dir.

    The name is held to the same rule 'prompt:' references enforce - a save
    the API accepted but no workflow could ever reference would be a trap. A
    save is told what is wrong with the name; a read just misses."""
    bare = name.removesuffix(".json")
    try:
        validate_prompt_reference(bare)
    except InvalidInputError as e:
        status = 400 if allow_create else 404
        raise HTTPException(status_code=status, detail=str(e))
    try:
        return validate_path(
            os.path.join(prompt_dir, f"{bare}.json"),
            prompt_dir,
            allow_create=allow_create,
        )
    except SecurityError as e:
        raise HTTPException(status_code=404, detail=f"Unknown prompt: {e}")
