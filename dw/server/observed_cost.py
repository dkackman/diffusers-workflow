"""What this server's own history says a workflow costs.

`cost` is a maintainer's claim: runs they measured, on cards they name,
written into the workflow. `dw/workflow_schema.json` says of it *"Never
derived; absent means unknown"*, and that sentence is load-bearing - so this
module does not touch it. It reports a sibling, `observed`, which is only ever
this box's own finished jobs (#93, split out of #91).

The report that asked for it: `list_workflows(shape="shot",
traits="identity-referenced")` answered `cost: null` for seven of eight
entries, on a box that had run `templates/minimax/reference-to-video` five
times at one size - 511, 464.9, 478, ~460, ~467 seconds. The consumer is told
to quote a price before spending GPU minutes and the catalog answered
"unknown" for the template their series actually used.

Four rules, each of which is a way the naive median would lie:

- **Comparability.** A 141-frame run does not inform a 124-frame estimate.
  A workflow declares `cost_drivers` - the variables that move its cost - and
  runs are bucketed by those values; the bucket reported is the one a caller
  gets by *default*, so it stays comparable to the curated figure. A workflow
  declaring no drivers falls back to default-arguments-only runs, which is
  thin but never wrong.
- **Cold is not warm.** The same `text-to-image` run is 13.6 s with the model
  on disk and 6.3 s with it resident. Averaging those describes neither, so
  `cold_minutes` and `warm_minutes` are separate, each with its own run
  count, and the one comparable to curated `cost` (wall clock *including*
  model load) is the one named `cold`.
- **A cached run is not a run.** A seeded workflow whose every step hit the
  step cache finishes in seconds and wrote nothing. Counting those would
  collapse the figure toward zero for exactly the templates that get re-run
  most.
- **Legibility over stability.** History is prunable, so the number moves.
  `runs` and `since` are what make that readable rather than surprising.

Everything comes off the job row - one query, no per-run JSON reads from the
run directories, so the figures survive a pruned run directory.
"""

import json
import logging
import time

logger = logging.getLogger("dw")

COST_DRIVERS_KEY = "cost_drivers"
# A phase event as `json.dumps` wrote it. Matched in SQL so a run's 200
# persisted events are never parsed just to ask whether the weights loaded.
LOADING_MARKER = '"phase": "loading"'
# Persisted events are trimmed to the newest 200 (`MAX_PERSISTED_EVENTS`), so
# a long run's `loading` phase can be gone from the record. Such a run counts
# toward `runs` and toward neither side of the split - claiming it was warm
# because the evidence was trimmed is the one answer that would be wrong.
EVENT_CAP = 200


def declared_drivers(definition):
    """The variables the author says move this workflow's cost.

    A name the workflow declares no variable for is dropped rather than
    bucketed on: its effective value is None for every run, so it would put
    every run in one bucket while looking like it had partitioned them.
    `tests/test_observed_cost.py` sweeps the catalog for one, since a driver
    that quietly checks nothing is the failure this whole module exists to
    avoid.
    """
    drivers = definition.get(COST_DRIVERS_KEY)
    if not isinstance(drivers, list):
        return []
    variables = definition.get("variables") or {}
    return [name for name in drivers if isinstance(name, str) and name in variables]


def _bucket_key(definition, arguments):
    """What makes two runs of this workflow comparable, or None when they
    are not comparable at all.

    With declared drivers: the effective value of each, which is what the run
    passed or else the stored default. Without: only a run that overrode
    nothing that matters is comparable, and since nothing is declared to
    matter, that means a run that overrode nothing at all.
    """
    variables = definition.get("variables") or {}
    drivers = declared_drivers(definition)
    if drivers:
        return tuple(
            (name, _comparable(arguments.get(name, variables.get(name))))
            for name in sorted(drivers)
        )
    if any(key in variables for key in arguments):
        return None
    return ()


def _comparable(value):
    """A driver value as something hashable and stable across JSON round
    trips - 124 and "124" are one bucket, since the engine coerces.

    A *list* driver buckets on its length, not its contents: a `shots` list
    costs what its number of entries costs, and two four-shot runs of one
    template are comparable however different their prompts. Bucketing on
    contents would give every run its own bucket and so a permanent
    `runs: 1`, which is the same as reporting nothing.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return ("entries", len(value))
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return json.dumps(value, sort_keys=True, default=str)


def _every_step_was_reused(manifest_text):
    """Whether this run generated nothing - every manifest entry a step-cache
    hit republishing an earlier run's files."""
    try:
        manifest = json.loads(manifest_text or "[]")
    except (TypeError, ValueError):
        return False
    if not isinstance(manifest, list) or not manifest:
        return False
    return all(
        isinstance(entry, dict) and entry.get("reused") is True for entry in manifest
    )


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _minutes_block(prefix, durations):
    """One side of the cold/warm split, or {} when it has no runs."""
    if not durations:
        return {}
    return {
        f"{prefix}_minutes": round(_median(durations) / 60, 2),
        f"{prefix}_runs": len(durations),
        f"{prefix}_range_minutes": [
            round(min(durations) / 60, 2),
            round(max(durations) / 60, 2),
        ],
    }


def observed_for(definition, rows, device=None, device_name=None, arguments=None):
    """This server's history for one workflow, as the `observed` block, or
    None when it has nothing comparable to report.

    `rows` are that workflow's finished runs as `history_rows` yields them.
    The device is the server's current one: the history holds no device
    column, so a figure is about whatever accelerator this box has now - true
    for every box that has not had its card swapped, and `runs`/`since` are
    what let a reader notice if it has.

    `arguments` are the run being asked about, and they choose the bucket:
    the listing asks with none, so its figure is the one the *defaults* give
    and stays comparable to a curated `cost`, while a plan asks with the
    caller's own values and gets the figure for the run it is quoting, or
    None when this box has never run that shape (#154).
    """
    wanted = _bucket_key(definition, arguments or {})
    if wanted is None:
        # No declared drivers and the caller overrode something: nothing here
        # is comparable to the run being asked about
        return None
    cold, warm, unclassified, earliest = [], [], 0, None
    for row in rows:
        if _every_step_was_reused(row["manifest"]):
            continue
        try:
            row_arguments = json.loads(row["arguments"] or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(row_arguments, dict):
            continue
        if _bucket_key(definition, row_arguments) != wanted:
            continue
        duration = row["duration"]
        if duration is None or duration <= 0:
            continue
        if row["had_load"]:
            cold.append(duration)
        elif row["events_at_cap"]:
            unclassified += 1
        else:
            warm.append(duration)
        finished = row["finished_at"]
        if finished and (earliest is None or finished < earliest):
            earliest = finished

    runs = len(cold) + len(warm) + unclassified
    if not runs:
        return None
    variables = definition.get("variables") or {}
    drivers = declared_drivers(definition)
    observed = {
        "device": device,
        "name": device_name,
        "runs": runs,
        "comparable": "drivers" if drivers else "default-arguments",
        **_minutes_block("cold", cold),
        **_minutes_block("warm", warm),
    }
    if drivers:
        # A list driver is reported as the length it buckets on, so the
        # figure says what it is a figure *for* without carrying a whole
        # default shot list into the listing
        # The *effective* driver values - the caller's where they supplied
        # one, else the stored default - so the block says which run it is a
        # figure for rather than always describing the defaults
        effective = {
            name: (arguments or {}).get(name, variables.get(name))
            for name in sorted(drivers)
        }
        observed["drivers"] = {
            name: len(value) if isinstance(value, list) else value
            for name, value in effective.items()
        }
    if unclassified:
        # Says the numbers do not add up, and why, rather than letting a
        # reader assume a trimmed run was a warm one
        observed["unclassified_runs"] = unclassified
    if earliest:
        observed["since"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(earliest))
    return observed


class ObservedCosts:
    """The `observed` block per workflow name, recomputed when the job table
    has moved.

    One query for the whole listing rather than one per workflow, cached
    against the jobs table's own high-water mark - the same reason
    `workflow_details` caches against a file's mtime, except that a job
    landing changes every figure and no file changes at all.
    """

    def __init__(self, history):
        self.history = history
        self._mark = None
        self._rows = {}
        self._device = None

    def device(self):
        """The accelerator every figure here is about, as (type, card name).

        Computed once: it cannot change inside a process, and the listing is
        the agent's hot path - asking torch for the card's name on every call
        would be a torch round trip per catalog read. The history holds no
        device column, so this is the server's current accelerator rather
        than the one each run used; a box whose card was swapped mid-history
        is the one case that misreports, and `runs`/`since` are what let a
        reader notice.
        """
        if self._device is None:
            from .. import device_memory_stats, get_device, get_device_type

            kind, card = None, None
            try:
                kind = get_device_type(get_device())
            except Exception:
                logger.debug("observed cost: could not read the device type")
            try:
                # The card's marketing name, which only CUDA reports. Its
                # absence is not the device's absence, so the two are asked
                # for separately - one try around both let an ImportError on
                # the second answer `device: null` for a box plainly running
                # on CUDA, which is the silent-null shape this whole field
                # exists to replace
                card = device_memory_stats().get("device_name")
            except Exception:
                logger.debug("observed cost: could not read the device name")
            self._device = (kind, card)
        return self._device

    def rows_for(self, name):
        """That workflow's comparable-candidate runs, refreshing the cache
        when the table has moved."""
        if self.history is None:
            return []
        try:
            mark = self.history.watermark()
        except Exception:
            logger.debug("observed cost: could not read the job watermark")
            return []
        if mark != self._mark:
            try:
                self._rows = self.history.finished_runs()
            except Exception:
                logger.debug("observed cost: could not read job history")
                self._rows = {}
                return []
            self._mark = mark
        return self._rows.get(name, [])

    def observed(self, name, definition, arguments=None):
        rows = self.rows_for(name)
        if not rows:
            return None
        device, device_name = self.device()
        try:
            return observed_for(
                definition,
                rows,
                device=device,
                device_name=device_name,
                arguments=arguments,
            )
        except Exception:
            # A figure is a nicety; a listing that 500s because one row had a
            # shape nobody expected is not
            logger.debug("observed cost: could not aggregate %s", name, exc_info=True)
            return None
