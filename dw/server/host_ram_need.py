"""How much host RAM a run is projected to hold at its peak (#869).

Two jobs on two cards share one host, and a pair of heavy ones can run it
out of memory together. This module is the figure a dispatcher compares
with the host's free RAM; it projects and never refuses.

A run's *memory key* is what decides its host-RAM footprint, read from the
definition with the caller's arguments resolved:
  - the sorted pipeline identity (component_type, model_name, workflow) of
    every pipeline step (`dw/vram_estimate.py` pipeline_identity);
  - every `quantization_config`, `offload` and `group_offload` value, found
    the way `vram_inheritance._found` finds them.
The workflow's name is not in it, so a renamed copy and an inline copy of a
template share the template's history, and a different `weights_dtype` (fed
into a quantization config) or offload is a different key. Frame count,
resolution and reference count are not in it either (#869's D3).

`projected_peak` answers, in order:
  - `observed`: the largest `host_memory_job_peak_rss_mb` of this machine's
    successful runs with the same key (the max, not a median: under-
    projecting is the failure this exists for);
  - `fit`: `host_memory_projection`'s projection for a list run, from this
    workflow's own history;
  - otherwise None - no figure.
"""

import hashlib
import json
import logging

from ..host_memory_projection import projected_mb
from ..variables import VariableNotFoundError, replace_variables
from ..vram_estimate import pipeline_identity
from ..vram_inheritance import _found

logger = logging.getLogger("dw")

# Where a run's host-RAM footprint changes without its pipeline changing.
# group_offload is offload by another name: it streams weights from host RAM
_SETTINGS = ("quantization_config", "offload", "group_offload")


def _pipeline_steps(node):
    """Every dict under `node` that loads a pipeline, a for_each step's too."""
    if isinstance(node, dict):
        if isinstance(node.get("pipeline"), dict):
            yield node
        for value in node.values():
            yield from _pipeline_steps(value)
    elif isinstance(node, list):
        for item in node:
            yield from _pipeline_steps(item)


def memory_key(definition, arguments=None):
    """The run's memory key as a short hex digest, or None for a definition
    that loads no pipeline it can name (a task-only workflow) or that
    references a variable it doesn't declare."""
    if not isinstance(definition, dict):
        return None
    variables = definition.get("variables")
    variables = dict(variables) if isinstance(variables, dict) else {}
    variables.update(arguments or {})
    # Deep substitution, not `_found`'s: `weights_dtype` sits inside a
    # quantization config rather than being one
    try:
        steps = replace_variables(definition.get("steps"), variables)
    except VariableNotFoundError:
        return None
    identities = {pipeline_identity(step) for step in _pipeline_steps(steps)}
    identities.discard(None)
    if not identities:
        return None
    parts = {"pipelines": sorted(json.dumps(identity) for identity in identities)}
    for key in _SETTINGS:
        parts[key] = sorted(_found(steps, key, {}))
    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def projected_peak(definition, arguments, history, *, rows=None, list_entries=None):
    """(mb, basis, runs) for this run's projected host-RAM peak, or None.

    `history` is the server's `JobHistory` (this machine's runs); `rows` are
    this workflow's own finished runs as `ObservedCosts.rows_for` answers
    them, and `list_entries` the request's list lengths - the fit's inputs,
    absent for an inline definition, which has no history by name.
    """
    key = memory_key(definition, arguments)
    if key is not None and history is not None:
        peaks = history.peaks_for_key(key)
        if peaks:
            return max(peaks), "observed", len(peaks)
    fit = projected_mb(definition, list_entries, rows or [])
    if fit is not None:
        return fit["projected_mb"], "fit", fit["runs"]
    return None


def host_memory_block(need):
    """`need` as validate's `plan.host_memory`, or None for no figure."""
    if not need:
        return None
    mb, basis, runs = need
    return {"projected_mb": round(mb), "basis": basis, "runs": runs}
