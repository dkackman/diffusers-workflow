"""Model preferences: what a `preference:<key>` sub-workflow path runs (#1626).

A preference is a named ladder - key -> tier -> an ordered list of targets -
that a composing step names instead of a catalog path:

    {"workflow": {"path": "preference:text_to_image", "arguments": {...}}}

The engine names no key. This module owns the three parts:

  - `load_preferences()`: the shipped `dw/preference_defaults.json` (#1625's
    ladders, in the package so a wheel carries it), overlaid
    key by key by `preferences.json` in the settings directory
    (`~/.diffusers_helper/`, beside `settings.json`). An overlay key replaces
    the whole shipped key - tiers are never merged - and the overlay's
    `default_tier`, when it names one, replaces the shipped one. Read only.
  - `resolve(key, tier, host, open_target)`: the first target of the rung
    that fits this host, as a `Resolution` (target, arguments, skips, notes).
  - A ContextVar session (`activate` / `deactivate`) carrying the host facts
    and the admission pin: the map of every resolution a job's admission
    made, which the worker and a rerun replay rather than resolve again, so
    an edit to the store mid-queue never changes a queued job.

A target is a catalog name or `{"workflow": name, "arguments": {...},
"devices": [...]}`. It is skipped, with its reason, when its `devices` list
leaves out `get_device_type()`, its declared `host_ram_gb` (with the gate's
margin) is more than this host's RAM (`host_ram_declared.host_too_small`, the
verdict admission refuses on), its declared `vram_estimate` on this device
type (`vram_estimate.required_vram_gb`, the hard figure dispatch holds a card
to) is more than the smallest card, or it is gated and this host has no
access. Not downloaded is a note, not a skip, and so is a `cost` row measured
on a bigger card than this one: that figure says where the catalog was
measured, not what the target needs. No fit refuses - a resolution never
drops to another tier.

The tier comes from the composing step's `tier` argument, else the enclosing
workflow's folded `tier` variable (walking up the composition chain), else
the store's `default_tier`. `Workflow._expand` (dw/workflow.py) is the one
caller: it rewrites each `preference:` step to its target before any reader
of `workflow.path` sees the definition, so pricing, downloads, ownership,
warnings, realization, validation and the run all follow the resolved child.
"""

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import references
from .security import InvalidInputError, validate_variable_name

logger = logging.getLogger("dw")

TIERS = ("draft", "standard", "best")
DEFAULT_TIER = "standard"
# The argument a composing step picks its rung with
TIER_ARGUMENT = "tier"
PREFERENCES_FILE = "preferences.json"
SOURCE_SHIPPED = "shipped"
SOURCE_SERVER = "server"

SHIPPED_PATH = Path(__file__).resolve().parent / "preference_defaults.json"


class PreferenceError(InvalidInputError):
    """A `preference:` path that cannot resolve: an unknown key, a bad key or
    tier name, a stale target, no target that fits, or a target that fails
    path or argument validation. The message is what validate and the run
    both say. `skipped` is the no-fit refusal's list of every target passed
    over ([{"workflow", "reason"}]), so a reader can strike each one; it is
    empty for every other refusal."""

    def __init__(self, message, skipped=None):
        super().__init__(message)
        self.skipped = list(skipped or [])


def pin_key(path, tier):
    """The admission pin's key for one resolution: the path as written and
    the tier it resolved at, so a rerun at another tier resolves fresh."""
    return f"{path}@{tier}"


def overlay_path():
    """Where the server's own preferences overlay lives."""
    from .settings import get_settings_dir

    return get_settings_dir() / PREFERENCES_FILE


def _read(path):
    """A preferences document, or None when the file is absent. A file that
    is present but unreadable is logged and treated as absent - a broken
    overlay must not take the shipped ladder down with it."""
    try:
        with open(path) as file:
            document = json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        logger.warning(f"Preferences file {path} could not be read: {e}")
        return None
    if not isinstance(document, dict):
        logger.warning(f"Preferences file {path} is not a JSON object")
        return None
    return document


def load_preferences(shipped=None, overlay=None):
    """The merged store as {"default_tier", "preferences": {key: ladder},
    "sources": {key: "shipped" | "server"}, "contracts": {key: {"required",
    "optional"}}}. `shipped` and `overlay` are paths, defaulting to the
    shipped file and the settings-directory overlay. The contracts - the
    arguments a key's targets take - are the shipped file's alone: the
    overlay never adds or replaces one."""
    shipped_doc = _read(shipped or SHIPPED_PATH) or {}
    overlay_doc = _read(overlay or overlay_path()) or {}
    ladders = {}
    sources = {}
    for document, source in (
        (shipped_doc, SOURCE_SHIPPED),
        (overlay_doc, SOURCE_SERVER),
    ):
        entries = document.get("preferences")
        if not isinstance(entries, dict):
            continue
        for key, ladder in entries.items():
            # The whole key, never a per-tier merge
            ladders[key] = ladder
            sources[key] = source
    # A hand-edited overlay naming no real tier falls back to the shipped one
    default_tier = overlay_doc.get("default_tier")
    if default_tier not in TIERS:
        default_tier = shipped_doc.get("default_tier")
    contracts = shipped_doc.get("contracts")
    return {
        "default_tier": default_tier or DEFAULT_TIER,
        "preferences": ladders,
        "sources": sources,
        "contracts": dict(contracts) if isinstance(contracts, dict) else {},
    }


@dataclass
class Host:
    """The facts a fit is judged on. `gate_check`, when set, takes
    (target name, its definition, its arguments, the Workflow it opens as)
    and answers (skip reason or None, note or None) - the gated-access and
    not-downloaded probe, which costs Hub calls, so it is only set where a
    plan already pays for them (admission)."""

    device_type: Optional[str] = None
    smallest_card_gb: Optional[float] = None
    host_ram_gb: Optional[float] = None
    gate_check: Optional[Callable] = None


def default_host():
    """This process's own facts: its device type, its card's capacity and
    the machine's RAM, each None when it cannot be read."""
    from . import device_capacity_gb, get_device_type
    from .host_memory import host_memory_stats

    try:
        device_type = get_device_type()
    except Exception:  # a host with no backend still validates
        device_type = None
    try:
        card = device_capacity_gb()
    except Exception:
        card = None
    total_mb = (host_memory_stats() or {}).get("total_mb")
    return Host(
        device_type=device_type,
        smallest_card_gb=card,
        host_ram_gb=total_mb / 1024.0 if total_mb else None,
    )


@dataclass
class Session:
    """One job's preference context: the host facts, the pins it replays
    (`pinned`, from admission or a rerun) and every resolution it made
    (`resolved`, what admission hands the job and the manifest records)."""

    host: Optional[Host] = None
    pinned: dict = field(default_factory=dict)
    resolved: dict = field(default_factory=dict)
    store: Optional[dict] = None

    def facts(self):
        if self.host is None:
            self.host = default_host()
        return self.host

    def preferences(self):
        if self.store is None:
            self.store = load_preferences()
        return self.store


_SESSION = ContextVar("dw_preference_session", default=None)


def activate(session):
    """Make `session` the active one; returns the token `deactivate` takes."""
    return _SESSION.set(session)


def deactivate(token):
    _SESSION.reset(token)


@contextmanager
def activated(session):
    """`session` as the active one for the block."""
    token = activate(session)
    try:
        yield session
    finally:
        deactivate(token)


def active_session():
    """The active session, or None."""
    return _SESSION.get()


def recorded(workflow):
    """What a manifest pins for a run: every resolution the active session
    made - the composed children's too - else the workflow's own."""
    session = active_session()
    if session is not None and session.resolved:
        return dict(session.resolved)
    # A stand-in for a Workflow (a test's) may have no resolutions to give
    pins = getattr(workflow, "preference_pins", None)
    return pins() if callable(pins) else {}


def current_session():
    """The active session, or a fresh one (this host's facts, no pins) when
    none is - a CLI validate or run resolves against the box it is on."""
    session = _SESSION.get()
    return session if session is not None else Session()


@dataclass
class Resolution:
    """One resolved `preference:` path."""

    path: str
    tier: str
    target: str
    arguments: dict
    skipped: list  # [{"workflow", "reason"}]
    source: str
    notes: list = field(default_factory=list)
    pinned: bool = False

    def pin(self):
        """The manifest's entry for it."""
        return {
            "target": self.target,
            "arguments": self.arguments,
            "skipped": self.skipped,
            "source": self.source,
        }

    def line(self):
        """The plan's one line for it."""
        text = f"{self.path} (tier {self.tier}) -> {self.target}"
        if self.skipped:
            text += "; skipped " + ", ".join(
                f"{skip['workflow']}: {skip['reason']}" for skip in self.skipped
            )
        text += f"; source: {self.source}"
        if self.pinned:
            text += " (pinned)"
        for note in self.notes:
            text += f"; {note}"
        return text


def check_key(key):
    """The key of a `preference:` path, validated as a variable name."""
    try:
        return validate_variable_name(key)
    except InvalidInputError as e:
        raise PreferenceError(
            f"Invalid preference key '{key}': {e}. A key is a name like "
            "'text_to_image' (letters, digits, '_' and '-')"
        ) from e


def check_tier(tier):
    if tier not in TIERS:
        raise PreferenceError(
            f"Unknown tier '{tier}': a preference tier is one of " + ", ".join(TIERS)
        )
    return tier


def _target(entry, key, tier):
    """(workflow, arguments, devices) of one ladder entry."""
    if isinstance(entry, str):
        return entry, {}, None
    if isinstance(entry, dict) and isinstance(entry.get("workflow"), str):
        arguments = entry.get("arguments") or {}
        devices = entry.get("devices")
        if not isinstance(arguments, dict) or (
            devices is not None
            and not (
                isinstance(devices, list)
                and all(isinstance(device, str) for device in devices)
            )
        ):
            raise PreferenceError(
                f"Preference '{key}' tier {tier}: a target's 'arguments' must "
                "be an object and its 'devices' a list of device types"
            )
        return entry["workflow"], dict(arguments), devices
    raise PreferenceError(
        f"Preference '{key}' tier {tier}: a target is a workflow name or "
        f"{{workflow, arguments, devices}}, not {json.dumps(entry)}"
    )


def card_need_gb(definition, arguments, device_type, child=None):
    """What a card must hold for a target on `device_type`, as `(gb, hard)`,
    or None when nothing says (no `vram_estimate` and no `cost` entry for
    this device). The figure is the dispatcher's own
    (`vram_estimate.required_vram_gb`), read off the target's expanded
    definition with its arguments, so a target the fit passes is one the
    pool admits: a declared `vram_estimate` is hard, a `cost` row's card is
    only where the catalog was measured."""
    from .vram_estimate import required_vram_gb

    expanded = definition
    base_dir = None
    if child is not None:
        try:
            expanded = child.expanded_definition(arguments or None)
            base_dir = os.path.dirname(child.file_spec)
        except Exception:  # an unexpandable target is refused elsewhere
            expanded = definition
    return required_vram_gb(expanded, arguments, device_type, base_dir)


def _over_card(definition, arguments, host, child):
    """The `(gb, hard)` figure of a target that asks for more than `host`'s
    smallest card, or None."""
    device = host.device_type
    need = card_need_gb(definition, arguments, device, child) if device else None
    if need and host.smallest_card_gb and need[0] > host.smallest_card_gb:
        return need
    return None


def unchecked_fit_note(definition, arguments, host, child=None):
    """What to say of a target chosen on a card smaller than the one its
    `cost` was measured on, or None: a soft figure never skips."""
    over = _over_card(definition, arguments, host, child)
    if over is None or over[1]:
        return None
    return (
        f"measured on a {over[0]:g} GB card; this one has "
        f"{host.smallest_card_gb:.0f} GB, so the fit is not checked"
    )


def skip_reason(definition, arguments, devices, host, child=None):
    """Why a target does not fit `host`, or None when it does."""
    from .host_ram_declared import host_too_small

    device = host.device_type
    if devices is not None and device is not None and device not in devices:
        return f"{'/'.join(devices) or 'no-device'}-only"
    # Admission's own verdict, margin included (#1227)
    total_mb = host.host_ram_gb * 1024.0 if host.host_ram_gb else None
    try:
        too_small = host_too_small(definition, arguments, total_mb)
    except (ValueError, TypeError):  # an unreadable declaration is no figure
        too_small = None
    if too_small:
        return f"host RAM: {too_small}"
    over = _over_card(definition, arguments, host, child)
    if over is not None and over[1]:
        return (
            f"needs {over[0]:g} GB VRAM on {device}, the smallest card has "
            f"{host.smallest_card_gb:.0f} GB"
        )
    return None


def _replay(path, tier, key, pinned, opened):
    """An admission pin's resolution, replayed: its target must still open."""
    from .library import SubWorkflowNotFound

    target = pinned.get("target")
    try:
        opened(target)
    except SubWorkflowNotFound as e:
        raise PreferenceError(
            f"Preference '{key}' tier {tier}: the pinned target '{target}' "
            "is no longer in the catalog"
        ) from e
    return Resolution(
        path=path,
        tier=tier,
        target=target,
        arguments=dict(pinned.get("arguments") or {}),
        skipped=list(pinned.get("skipped") or []),
        source=pinned.get("source") or SOURCE_SHIPPED,
        pinned=True,
    )


def _rung(store, key, tier):
    """The store's entries for `key` at `tier`; refused when there are none."""
    ladders = store.get("preferences") or {}
    if key not in ladders:
        known = ", ".join(sorted(ladders)) or "<none>"
        raise PreferenceError(
            f"Unknown preference '{key}': no such key in the preferences "
            f"store; known keys: {known}"
        )
    ladder = ladders[key]
    entries = ladder.get(tier) if isinstance(ladder, dict) else None
    if not isinstance(entries, list) or not entries:
        raise PreferenceError(f"Preference '{key}' has no targets for tier {tier}")
    return entries


def resolve(path, tier, host, open_target, store, *, pinned=None):
    """Resolve one `preference:` path at `tier` on `host`.

    `open_target(name)` opens a target the way the composing workflow would
    open it as a sub-workflow, answering (definition, Workflow) and raising
    what resolution raises - SubWorkflowNotFound for a stale target. `store`
    is `load_preferences()`'s answer. `pinned`, an admission pin's entry for
    this path and tier, is replayed rather than resolved (its target must
    still open).

    Returns a Resolution; raises PreferenceError for every refusal.
    """
    from .library import SubWorkflowNotFound
    from .security import SecurityError

    key = check_key(references.ref_name(references.PREFERENCE, path))
    check_tier(tier)

    def opened(name):
        try:
            return open_target(name)
        except SubWorkflowNotFound:
            raise
        except PreferenceError:
            raise
        except SecurityError as e:
            raise PreferenceError(
                f"Preference '{key}' tier {tier}: target '{name}' is refused: {e}"
            ) from e

    if pinned:
        return _replay(path, tier, key, pinned, opened)
    targets = [_target(entry, key, tier) for entry in _rung(store, key, tier)]

    # Every stale target in the rung is named, not only the first
    candidates = []
    stale = []
    for name, arguments, devices in targets:
        try:
            definition, child = opened(name)
        except SubWorkflowNotFound:
            stale.append(name)
            continue
        candidates.append((name, arguments, devices, definition, child))
    if stale:
        raise PreferenceError(
            f"Preference '{key}' tier {tier} names "
            + ("a target" if len(stale) == 1 else "targets")
            + " not in the catalog: "
            + ", ".join(f"'{name}'" for name in stale)
        )

    skipped = []
    for name, arguments, devices, definition, child in candidates:
        bad = target_argument_errors(definition, arguments)
        if bad:
            raise PreferenceError(
                f"Preference '{key}' tier {tier}: target '{name}' arguments "
                "are refused: "
                + "; ".join(f"{error['path']}: {error['message']}" for error in bad)
            )
        reason = skip_reason(definition, arguments, devices, host, child)
        notes = []
        if reason is None and host.gate_check is not None:
            try:
                reason, note = host.gate_check(name, definition, arguments, child)
            except Exception as e:  # a probe failing is no reason to skip
                logger.debug(f"Preference gate probe for {name} failed: {e}")
                reason, note = None, None
            if note:
                notes.append(note)
        if reason is not None:
            skipped.append({"workflow": name, "reason": reason})
            continue
        unchecked = unchecked_fit_note(definition, arguments, host, child)
        if unchecked:
            notes.append(unchecked)
        return Resolution(
            path=path,
            tier=tier,
            target=name,
            arguments=arguments,
            skipped=skipped,
            source=(store.get("sources") or {}).get(key, SOURCE_SHIPPED),
            notes=notes,
        )
    raise PreferenceError(
        f"Preference '{key}' tier {tier}: no target fits this host - "
        + "; ".join(f"{skip['workflow']}: {skip['reason']}" for skip in skipped),
        skipped=skipped,
    )


def downloads_gate_check(cache_dir=None, lookup_sizes=True):
    """A `Host.gate_check` reading `plan_downloads.downloads_required` for a
    target: skipped when a repo it needs is gated and blocked for this box's
    token, noted when it needs a download."""
    from .plan_downloads import downloads_required

    def check(name, definition, arguments, child):
        expanded = child.expanded_definition(arguments or None)
        required = downloads_required(
            expanded,
            os.path.dirname(child.file_spec),
            child.workflow_dir,
            cache_dir,
            lookup_sizes,
        )
        blocked = [
            entry["repo"]
            for entry in required
            if entry.get("access_blocked") and entry.get("repo")
        ]
        if blocked:
            return f"gated, no access to {', '.join(blocked)}", None
        if not required:
            return None, None
        sizes = [entry.get("gb") for entry in required]
        if all(isinstance(size, (int, float)) for size in sizes):
            return None, f"will download {sum(sizes):.1f} GB"
        return None, "will download its weights"

    return check


# ------------------------------------------------------------------ the store
# The server writes the overlay (GET/PUT/DELETE /api/preferences,
# dw/server/routes/system.py); nothing else does. One lock per process
# around read-change-write, and the write is atomic, so a reader - admission,
# a worker - sees the old document or the new one, never half of either.

import threading  # noqa: E402

_overlay_lock = threading.Lock()


def _overlay_document(path):
    """The overlay as {"default_tier"?, "preferences": {...}}, {} when absent."""
    document = _read(path) or {}
    entries = document.get("preferences")
    return {
        **(
            {"default_tier": document["default_tier"]}
            if document.get("default_tier") in TIERS
            else {}
        ),
        "preferences": dict(entries) if isinstance(entries, dict) else {},
    }


def _store_overlay(path, document):
    """Write the overlay, or remove it when it holds nothing."""
    from .runs import write_json_atomic

    if not document.get("preferences") and "default_tier" not in document:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(path), document)


def update_overlay(change, overlay=None, shipped=None):
    """Read the merged store and the overlay, let `change(merged, document)`
    edit the overlay document in place (raising refuses the write - the
    caller's preconditions run here, under the lock), write it, and return
    the merged store after the write."""
    path = overlay or overlay_path()
    with _overlay_lock:
        merged = load_preferences(shipped=shipped, overlay=path)
        document = _overlay_document(path)
        change(merged, document)
        _store_overlay(path, document)
        return load_preferences(shipped=shipped, overlay=path)


def target_argument_errors(definition, arguments):
    """What is wrong with a ladder target's `arguments`: what a caller's
    would get (`variables.argument_errors`: undeclared, won't coerce), and
    the reference-name rule a step argument meets
    (`reference_names.reference_fault`: an `asset:`/`output:`/`prompt:`
    name that can never resolve, `..` among them). A target's arguments are
    merged into the step after that pass has read it, so the store holds
    them to it itself, at PUT and again at resolve (#1723, SE-F085)."""
    from .reference_names import reference_fault
    from .variables import argument_errors

    errors = argument_errors(definition, arguments)
    if isinstance(arguments, dict):
        for name, value in arguments.items():
            for text in _strings(value):
                fault = reference_fault(text)
                if fault is not None:
                    errors.append({"path": f"arguments.{name}", "message": fault})
    return errors


def _strings(value):
    """Every string inside an argument value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def _ladder_targets(key, ladder):
    """[(tier, name, arguments, devices)] of a ladder, in order, refusing with
    PreferenceError a bad key name, a value that is not {tier: [target,
    ...]}, an unknown tier, an empty rung or a target of the wrong shape."""
    check_key(key)
    if not isinstance(ladder, dict) or not ladder:
        raise PreferenceError(
            f"Preference '{key}': a ladder is an object of tiers "
            f"({', '.join(TIERS)}), each an ordered list of targets"
        )
    targets = []
    for tier, entries in ladder.items():
        check_tier(tier)
        if not isinstance(entries, list) or not entries:
            raise PreferenceError(
                f"Preference '{key}' tier {tier}: a rung is a non-empty list of targets"
            )
        for entry in entries:
            if isinstance(entry, dict):
                extra = set(entry) - {"workflow", "arguments", "devices"}
                if extra:
                    raise PreferenceError(
                        f"Preference '{key}' tier {tier}: a target takes only "
                        f"workflow, arguments and devices, not {', '.join(sorted(extra))}"
                    )
            targets.append((tier, *_target(entry, key, tier)))
    return targets


def _target_problem(key, tier, name, arguments, open_target):
    """Why a PUT would refuse one target: (stale, message), or None when it
    would store it."""
    from .library import SubWorkflowNotFound
    from .security import SecurityError

    try:
        definition, _child = open_target(name)
    except SubWorkflowNotFound:
        return True, (
            f"Preference '{key}' tier {tier}: target '{name}' is not in the catalog"
        )
    except (SecurityError, InvalidInputError) as e:
        return False, (
            f"Preference '{key}' tier {tier}: target '{name}' is refused: {e}"
        )
    except ValueError as e:  # a catalog file that is not JSON
        return False, (
            f"Preference '{key}' tier {tier}: target '{name}' does not parse: {e}"
        )
    if not isinstance(definition, dict):
        return False, (
            f"Preference '{key}' tier {tier}: target '{name}' is not a workflow"
        )
    bad = target_argument_errors(definition, arguments)
    if bad:
        return False, (
            f"Preference '{key}' tier {tier}: target '{name}' arguments "
            "are refused: "
            + "; ".join(f"{error['path']}: {error['message']}" for error in bad)
        )
    return None


def check_ladder(key, ladder, open_target):
    """Refuse a ladder a PUT may not store, with PreferenceError: a bad key
    name, a value that is not {tier: [target, ...]}, an unknown tier, an
    empty rung, a target of the wrong shape, a stale target (every one in
    the ladder named) or target arguments its workflow refuses. Whether a
    target fits this host is not a refusal here: that is resolution's, at
    validate, which names every skip."""
    stale = []
    for tier, name, arguments, _devices in _ladder_targets(key, ladder):
        problem = _target_problem(key, tier, name, arguments, open_target)
        if problem is None:
            continue
        is_stale, message = problem
        if not is_stale:
            raise PreferenceError(message)
        stale.append(name)
    if stale:
        names = list(dict.fromkeys(stale))
        raise PreferenceError(
            f"Preference '{key}' names "
            + ("a target" if len(names) == 1 else "targets")
            + " not in the catalog: "
            + ", ".join(f"'{name}'" for name in names)
        )
    return ladder


def target_problems(key, ladder, open_target):
    """{target name: why a PUT would refuse it} over a whole ladder - not in
    the catalog, arguments refused, refused by security - where
    `check_ladder` stops at the first. A shape error still raises
    PreferenceError, worded as `check_ladder` words it."""
    problems = {}
    for tier, name, arguments, _devices in _ladder_targets(key, ladder):
        problem = _target_problem(key, tier, name, arguments, open_target)
        if problem is not None:
            problems.setdefault(name, problem[1])
    return problems


KNOWN_BLOCKED = "gated, no access (last full check)"


def skip_known_blocked(check, names):
    """A `Host.gate_check` that skips each target in `names` - ones the last
    full check found gated for this box's token - without asking `check`,
    and asks `check` for the rest."""
    blocked = set(names or ())

    def gate(name, definition, arguments, child):
        if name in blocked:
            return KNOWN_BLOCKED, None
        return check(name, definition, arguments, child)

    return gate


def check_draft(key, ladder, store, host, open_target, contract=None):
    """What saving `ladder` under `key` would give, without saving it: per
    tier in the ladder, `resolutions`'s pick over the store with this ladder
    in place, the targets after the pick (`after`), each target a PUT would
    refuse (`target_errors`) and, per target, each `contract["required"]`
    argument its workflow does not declare (`warnings`). Keys with nothing
    to say are left out. A shape error raises PreferenceError."""
    from .library import SubWorkflowNotFound
    from .security import SecurityError

    problems = target_problems(key, ladder, open_target)
    draft = {**store, "preferences": {**(store.get("preferences") or {}), key: ladder}}
    picks = resolutions(key, draft, host, open_target)
    required = list((contract or {}).get("required") or [])
    answer = {}
    for tier, found in picks.items():
        names = [_target(entry, key, tier)[0] for entry in ladder[tier]]
        entry = {
            field: found[field]
            for field in ("target", "skipped", "notes", "error")
            if found.get(field)
        }
        if "target" in entry:
            after = names[len(found.get("skipped") or []) + 1 :]
            if after:
                entry["after"] = after
        errors = {name: problems[name] for name in names if name in problems}
        if errors:
            entry["target_errors"] = errors
        warnings = {}
        for name in dict.fromkeys(names):
            if not required or name in problems:
                continue
            try:
                definition, _child = open_target(name)
            except (SubWorkflowNotFound, SecurityError, InvalidInputError, ValueError):
                continue
            if not isinstance(definition, dict):
                continue
            declared = definition.get("variables") or {}
            missing = [
                f"does not take {var}" for var in required if var not in declared
            ]
            if missing:
                warnings[name] = missing
        if warnings:
            entry["warnings"] = warnings
        answer[tier] = entry
    return answer


def resolutions(key, store, host, open_target):
    """This host's pick at each tier of one key, as {tier: {"target",
    "skipped", "notes", "line"} | {"error", "skipped"?}} - what `resolve`
    answers for `preference:<key>`, a refusal as its message. A no-fit
    refusal also carries `skipped`: every target passed over, with why."""
    answer = {}
    ladder = (store.get("preferences") or {}).get(key)
    for tier in TIERS:
        if not isinstance(ladder, dict) or tier not in ladder:
            continue
        path = references.make_ref(references.PREFERENCE, key)
        try:
            found = resolve(path, tier, host, open_target, store)
        except Exception as e:  # noqa: BLE001 - a refusal is the answer
            answer[tier] = {"error": str(e)}
            skipped = getattr(e, "skipped", None)
            if isinstance(e, PreferenceError) and skipped:
                answer[tier]["skipped"] = list(skipped)
            continue
        answer[tier] = {
            "target": found.target,
            "skipped": found.skipped,
            "notes": found.notes,
            "line": found.line(),
        }
    return answer
