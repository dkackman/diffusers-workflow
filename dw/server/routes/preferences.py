"""The preferences store over HTTP: GET, PUT and DELETE /api/preferences.

The store is `dw/preferences.py`'s: the shipped ladders, with this server's
overlay (the settings directory's preferences.json) replacing whole keys.
These routes are the overlay's only writer. A PUT is refused with 422 when
its ladder is the wrong shape, names a workflow the catalog does not hold,
or gives a target arguments its workflow refuses; whether a target fits
this host is not a refusal here - that is validate's, which names every skip.
`If-Match` is opt-in, as on the note routes, and a stale one is a 412.

Resolutions answered here use the host's device and smallest card, without
the gated-access and download probe admission adds, so a read makes no Hub
calls. A job already queued keeps the target admission pinned into it.
"""

import json
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response

from ... import preferences
from ...workflow import workflow_from_definition
from ...workspace import Workspace
from ..api_models_library import (
    DefaultTierSet,
    DefaultTierWrite,
    PreferenceReset,
    PreferenceSet,
    PreferencesRead,
)
from ..changes import WORKFLOWS, ChangeRoute, writes
from ..deps import internal_error, selected_workspace
from ..preconditions import check_if_match, content_etag_of

router = APIRouter(route_class=ChangeRoute)

# A stale If-Match: 412 Precondition Failed (the note routes predate this and
# answer 409)
STALE = 412

# The inline workflow targets are opened from: no steps of its own, only a
# place in the workspace for resolve_sub_workflow_reference to search from
_OPENER = {
    "id": "preferences",
    "steps": [
        {
            "name": "noop",
            "pipeline": {"configuration": {}},
        }
    ],
}


def _key_bytes(store, key):
    ladders = store.get("preferences") or {}
    if key not in ladders:
        return b"null"
    return json.dumps(
        {"source": store["sources"].get(key), "ladder": ladders[key]},
        sort_keys=True,
    ).encode("utf-8")


def _key_etag(store, key):
    return content_etag_of(_key_bytes(store, key))


def _tier_etag(store):
    return content_etag_of(store["default_tier"].encode("utf-8"))


def _host(request):
    host = preferences.default_host()
    manager = getattr(request.app.state, "job_manager", None)
    smallest = getattr(manager, "smallest_card_gb", None)
    if callable(smallest):
        try:
            card = smallest()
        except Exception:  # noqa: BLE001 - the host's own figure stands
            card = None
        if card:
            host.smallest_card_gb = card
    return host


def _opener(ws: Workspace):
    """`open_target(name)` the way a workflow in `ws` opens a preference
    target, opening each name once."""
    workflow = workflow_from_definition(
        json.loads(json.dumps(_OPENER)), ws.outputs, None, ws.workflows
    )
    return workflow._open_preference_target


def _entry(request, ws, store, key, resolve):
    entry = {
        "source": store["sources"][key],
        "etag": _key_etag(store, key),
        "ladder": store["preferences"][key],
    }
    if resolve:
        entry["resolutions"] = preferences.resolutions(
            key, store, _host(request), _opener(ws)
        )
    return entry


def _unknown(store, key):
    known = ", ".join(sorted(store.get("preferences") or {})) or "<none>"
    return HTTPException(
        status_code=404,
        detail=f"Unknown preference '{key}'; known keys: {known}",
    )


def _key_expectation(request, key):
    if request.headers.get("if-match") is None:
        return None
    label = f"Preference '{key}'"
    return lambda store: check_if_match(
        request, _key_bytes(store, key), label, status_code=STALE
    )


def _checked_key(key):
    try:
        return preferences.check_key(key)
    except preferences.PreferenceError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get(
    "/api/preferences",
    response_model=PreferencesRead,
    response_model_exclude_unset=True,
)
def get_preferences(
    request: Request,
    response: Response,
    key: Optional[str] = Query(None, description="One key only"),
    resolve: bool = Query(
        False, description="Add this host's pick per tier to each key"
    ),
    ws: Workspace = Depends(selected_workspace),
):
    """The merged ladders - shipped, with this server's overlay replacing
    whole keys - each with its `source` and etag, and `default_tier`."""
    store = preferences.load_preferences()
    if key is not None:
        _checked_key(key)
        if key not in store["preferences"]:
            raise _unknown(store, key)
        response.headers["ETag"] = _key_etag(store, key)
    keys = [key] if key is not None else sorted(store["preferences"])
    return {
        "default_tier": store["default_tier"],
        "etag": _tier_etag(store),
        "preferences": {k: _entry(request, ws, store, k, resolve) for k in keys},
    }


@router.put(
    "/api/preferences/{key}",
    response_model=PreferenceSet,
    response_model_exclude_unset=True,
)
@writes(WORKFLOWS, shared=True)
def put_preference(
    request: Request,
    key: str,
    ladder: dict[str, Any] = Body(
        description="Tier -> an ordered list of targets: a workflow name, or "
        "{workflow, arguments?, devices?}"
    ),
    ws: Workspace = Depends(selected_workspace),
):
    """Replace a key's ladder in this server's overlay (a new key adds one).

    Refused with 422 when the ladder is the wrong shape, names a workflow
    not in the catalog, or gives a target arguments it refuses; with 412
    when `If-Match` no longer names the key as it stands. The answer carries
    this host's pick per tier, so a ladder nothing on this host fits shows
    validate's refusal now."""
    _checked_key(key)
    try:
        preferences.check_ladder(key, ladder, _opener(ws))
    except preferences.PreferenceError as e:
        raise HTTPException(status_code=422, detail=str(e))
    expect = _key_expectation(request, key)

    def change(merged, document):
        if expect:
            expect(merged)
        document["preferences"][key] = ladder

    try:
        store = preferences.update_overlay(change)
    except OSError:
        raise internal_error(f"Could not write preference '{key}'")
    return {"key": key, **_entry(request, ws, store, key, True)}


@router.delete(
    "/api/preferences/{key}",
    response_model=PreferenceReset,
    response_model_exclude_unset=True,
)
@writes(WORKFLOWS, shared=True)
def delete_preference(
    request: Request,
    key: str,
    ws: Workspace = Depends(selected_workspace),
):
    """Reset a key to what ships: drop this server's ladder for it. A key
    only the server had is gone after; one it never overrode is unchanged."""
    _checked_key(key)
    expect = _key_expectation(request, key)

    def change(merged, document):
        if key not in merged["preferences"]:
            raise _unknown(merged, key)
        if expect:
            expect(merged)
        document["preferences"].pop(key, None)

    try:
        store = preferences.update_overlay(change)
    except OSError:
        raise internal_error(f"Could not reset preference '{key}'")
    if key not in store["preferences"]:
        return {"key": key, "removed": True}
    return {"key": key, "removed": False, **_entry(request, ws, store, key, True)}


@router.put("/api/preferences", response_model=DefaultTierSet)
@writes(WORKFLOWS, shared=True)
def put_default_tier(request: Request, body: DefaultTierWrite = Body()):
    """Set the tier a `preference:` reference without `tier` resolves at.
    `If-Match` names the etag GET /api/preferences answers beside it."""
    tier = body.default_tier
    expecting = request.headers.get("if-match") is not None

    def change(merged, document):
        if expecting:
            check_if_match(
                request,
                merged["default_tier"].encode("utf-8"),
                "The default tier",
                status_code=STALE,
            )
        document["default_tier"] = tier

    try:
        store = preferences.update_overlay(change)
    except OSError:
        raise internal_error("Could not write the default tier")
    return {"default_tier": store["default_tier"], "etag": _tier_etag(store)}
