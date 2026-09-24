"""The local file a validate-time media probe may read, or None.

`dissolve_frame_errors` (#400) and `slice_preflight` (#402) each probe an
input's real length before the run, and each echoes what the probe found.
A free `validate_workflow` must not answer questions about files the run
itself would refuse to read, so a literal path goes through the same
containment policy the run applies (`validate_media_path`, lifted under
--trust-workflows) and is resolved against the workflow's directory; one it
refuses defers to the run, and `location_errors` reports the refusal.
"""

import os

from .arguments import resolve_path_references
from .assets import is_asset_reference
from .locations import is_http_url, validate_media_path
from .runs import is_output_reference

# Left to the run-time check: not yet resolved to a real file at the point
# validation walks the expanded definition.
UNRESOLVED_PREFIXES = ("previous_result:", "variable:", "item:", "gather:")


def resolve_probe_path(value, base_dir, what="a media argument"):
    """The local file `value` names, or None when it is not yet resolvable,
    is not a local file, does not exist, or lies outside what this workflow
    may read - any of which defers the check to the run.

    An `asset:`/`output:` reference resolves through its own confined
    resolver; a literal path is resolved against `base_dir` and must pass
    `validate_media_path`.
    """
    if not isinstance(value, str) or not value:
        return None
    if value.startswith(UNRESOLVED_PREFIXES) or is_http_url(value):
        return None
    if is_asset_reference(value) or is_output_reference(value):
        try:
            value = resolve_path_references(value, base_dir)
        except Exception:
            # Existence/traversal problems belong to reference_name_errors
            # and reference resolution at run time, not to this check
            return None
        if not isinstance(value, str):
            return None
    else:
        try:
            value = validate_media_path(value, base_dir, what, require_exists=False)
        except Exception:
            # A refusal is location_errors' to report; echoing anything
            # about the file here would answer what the run will not read
            return None
    return value if os.path.isfile(value) else None


__all__ = ["UNRESOLVED_PREFIXES", "resolve_probe_path"]
