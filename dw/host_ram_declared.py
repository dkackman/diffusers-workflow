"""The host RAM a workflow declares it needs, and the refusal when a host can
never hold it (#1227).

A workflow's top-level `host_ram_gb` (a number, or a `variable:` a
`presets.tier` bundle sets) is the figure; `with_margin` is the margin the
host-RAM gate keeps over any projected peak. Admission refuses a run whose
declaration plus margin exceeds the machine's total RAM (`host_too_small`),
and a `preference:` resolution skips a target for the same verdict
(dw/preferences.py), so the two never disagree.

It lives outside `dw/server/` because the engine's preference resolver reads
it; `dw/server/host_ram_need.py` (the projected peak) and `dw/server/pool.py`
(the dispatch gate) import it from here.
"""

from .presets import effective_variables
from .variables import VariableNotFoundError, replace_variables

# The share of a projected peak the gate keeps free above it, and its floor.
# A recorded committed peak is the largest of readings taken at phase
# boundaries (dw/worker.py's _on_event), not a high-water mark - committed
# memory has none to read - so a transient between two boundaries (a
# component's weights staged before its old copy is freed) is not in it and
# the peak is a lower bound. The RSS peak it replaced hid this: ru_maxrss
# counted ~80 GB of checkpoint page cache on an H3 run (#869)
HOST_RAM_MARGIN_FRACTION = 0.10
HOST_RAM_MARGIN_MIN_MB = 2048


def with_margin(peak_mb):
    """`peak_mb` with the gate's margin over it."""
    return peak_mb + max(peak_mb * HOST_RAM_MARGIN_FRACTION, HOST_RAM_MARGIN_MIN_MB)


def declared_mb(definition, arguments=None):
    """The host RAM (MB) the definition declares for the run (`host_ram_gb`,
    a number or a `variable:` reference a presets bundle sets), or None when
    it declares none or the figure does not resolve to a positive number
    (#1227)."""
    if not isinstance(definition, dict) or "host_ram_gb" not in definition:
        return None
    variables = effective_variables(definition, arguments)
    try:
        value = replace_variables({"v": definition["host_ram_gb"]}, variables)["v"]
    except VariableNotFoundError:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value) * 1024.0


def host_too_small(definition, arguments, total_mb):
    """The refusal for a run whose declared host RAM (with the gate's margin)
    exceeds the machine's total RAM - it could never start, however long it
    waited - or None. No declaration or no reading of the host: no refusal.
    Only the declaration refuses; an observed peak only ever delays (#1227)."""
    declared = declared_mb(definition, arguments)
    if declared is None or not total_mb:
        return None
    needed = with_margin(declared)
    if needed <= total_mb:
        return None
    return (
        f"this run declares about {declared / 1024:.0f} GB of host RAM "
        f"({needed / 1024:.0f} GB with the gate's margin) and this host has "
        f"{total_mb / 1024:.0f} GB in total; choose a lighter tier or weights "
        "dtype, or run it on a larger host"
    )
