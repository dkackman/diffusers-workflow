"""Model preferences: what a `preference:<key>` sub-workflow path runs (#1626).

A preference is a named ladder - key -> tier -> an ordered list of targets -
that a composing step names instead of a catalog path:

    {"workflow": {"path": "preference:text_to_image", "arguments": {...}}}

The engine names no key. This module owns the three parts:

  - `load_preferences()`: the shipped `workflows/preferences.json`, overlaid
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
leaves out `get_device_type()`, its declared `host_ram_gb` is above this
host's RAM, its `cost` entry for this device type needs more VRAM than the
smallest card, or it is gated and this host has no access. Not downloaded is
a note, not a skip; a target with no `cost` entry for this device is not
skipped. No fit refuses - a resolution never drops to another tier.

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

SHIPPED_PATH = Path(__file__).resolve().parent.parent / "workflows" / PREFERENCES_FILE


class PreferenceError(InvalidInputError):
    """A `preference:` path that cannot resolve: an unknown key, a bad key or
    tier name, a stale target, no target that fits, or a target that fails
    path or argument validation. The message is what validate and the run
    both say."""


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
    "sources": {key: "shipped" | "server"}}. `shipped` and `overlay` are
    paths, defaulting to the shipped file and the settings-directory
    overlay."""
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
    default_tier = overlay_doc.get("default_tier") or shipped_doc.get("default_tier")
    return {
        "default_tier": default_tier or DEFAULT_TIER,
        "preferences": ladders,
        "sources": sources,
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


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def declared_host_ram_gb(definition, arguments):
    """The host RAM (GB) a target declares (`host_ram_gb`, a number or a
    `variable:` a presets bundle sets), or None."""
    if not isinstance(definition, dict) or "host_ram_gb" not in definition:
        return None
    from .presets import effective_variables
    from .variables import VariableNotFoundError, replace_variables

    try:
        variables = effective_variables(definition, arguments)
        value = replace_variables({"v": definition["host_ram_gb"]}, variables)["v"]
    except (VariableNotFoundError, ValueError, TypeError):
        return None
    value = _number(value)
    return value if value and value > 0 else None


def card_need_gb(definition, device_type):
    """The least VRAM a target's `cost` entries for `device_type` declare, or
    None when it has no entry for this device (not skipped: no figure)."""
    needs = [
        _number(entry.get("vram_gb"))
        for entry in (definition.get("cost") or [])
        if isinstance(entry, dict) and entry.get("device") == device_type
    ]
    needs = [need for need in needs if need is not None]
    return min(needs) if needs else None


def skip_reason(definition, arguments, devices, host):
    """Why a target does not fit `host`, or None when it does."""
    device = host.device_type
    if devices is not None and device is not None and device not in devices:
        return f"{'/'.join(devices) or 'no-device'}-only"
    ram = declared_host_ram_gb(definition, arguments)
    if ram is not None and host.host_ram_gb and ram > host.host_ram_gb:
        return f"needs {ram:g} GB host RAM, this host has {host.host_ram_gb:.0f} GB"
    need = card_need_gb(definition, device) if device else None
    if need is not None and host.smallest_card_gb and need > host.smallest_card_gb:
        return (
            f"needs {need:g} GB VRAM on {device}, the smallest card has "
            f"{host.smallest_card_gb:.0f} GB"
        )
    return None


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
    from .variables import argument_errors

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
        target = pinned.get("target")
        arguments = pinned.get("arguments") or {}
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
            arguments=dict(arguments),
            skipped=list(pinned.get("skipped") or []),
            source=pinned.get("source") or SOURCE_SHIPPED,
            pinned=True,
        )

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
    targets = [_target(entry, key, tier) for entry in entries]

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
        bad = argument_errors(definition, arguments)
        if bad:
            raise PreferenceError(
                f"Preference '{key}' tier {tier}: target '{name}' arguments "
                "are refused: "
                + "; ".join(f"{error['path']}: {error['message']}" for error in bad)
            )
        reason = skip_reason(definition, arguments, devices, host)
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
        + "; ".join(f"{skip['workflow']}: {skip['reason']}" for skip in skipped)
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
