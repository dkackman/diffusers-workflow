"""What a task step fetches from the Hub: the `required_weights` hook's helpers.

Every model-backed task module exports `required_weights(arguments) ->
list[dict]` - each dict `{"repo": <hub id>}` for a whole repo, `{"repo",
"filename", "revision"}` for one file of it (revision None is main), or
`{"url"}` for a fetch from outside the Hub - and its loader
reads the repo it loads from the same resolution, so the plan's
`downloads_required` (dw/plan.py) and the run cannot disagree about which
argument wins over which default (#975). A command names its hook on its
registration (`register_command(required_weights=...)`, a dotted path, so the
plan path imports a task module only when a workflow has that task).

A hook reads the step's *expanded* arguments, which at validate time can still
hold a reference (`previous_result:mask`) or a local checkout path. Neither is
a download it can name, so it gives no entry and never raises: whether a name
is a hub id is decided by its shape, never by touching the disk, as the
pipeline walk decides it.

Torch-free on purpose: `dw/plan.py` imports it on every validate.
"""

import importlib.util
from pathlib import Path

from huggingface_hub.utils import HFValidationError, validate_repo_id


def argument_or_default(arguments, name, default):
    """The value the task will load for `name`: the step's own argument when
    it gives one, else the task's default - the one rule both the hook and the
    loader read, so an override and a default cannot be resolved two ways."""
    value = (arguments or {}).get(name)
    return default if value is None else value


def is_hub_repo(value):
    """Whether `value` is shaped like a hub repo id - a reference
    (`previous_result:x`), a path (`/models/x`, `./x`) or a non-string is not."""
    if not isinstance(value, str):
        return False
    try:
        validate_repo_id(value)
        return True
    except HFValidationError:
        return False


def repo_entries(*names):
    """`[{"repo": name}, ...]` for each name shaped like a hub repo id, in
    order and once each; every other value is dropped silently."""
    entries = []
    for name in names:
        if is_hub_repo(name) and {"repo": name} not in entries:
            entries.append({"repo": name})
    return entries


def file_entries(repo, filename, revision=None):
    """`[{"repo", "filename", "revision"}]` for one file of a hub repo, or
    `[]` when the repo is not shaped like a hub id or the file name is not a
    plain relative name inside it - the plan checks the cache by joining the
    name onto it, so a name climbing out would make that check a probe of
    the disk."""
    from ..locations import validate_weight_name
    from ..security import SecurityError

    if not is_hub_repo(repo):
        return []
    try:
        validate_weight_name(filename)
    except SecurityError:
        return []
    return [{"repo": repo, "filename": filename, "revision": revision}]


def package_dir(name):
    """The directory a package is installed in, found without importing it,
    or None when it is not installed - facexlib's import loads cv2, which a
    validate should not pay for."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    locations = list(spec.submodule_search_locations or []) if spec else []
    return Path(locations[0]) if locations else None
