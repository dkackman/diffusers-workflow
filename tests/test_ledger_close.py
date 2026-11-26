"""Only a user write closes a request; an agent proposes the close (#1716).

The ledger's rule (`dw/ledger.py`: `_check_close`, `_close_proposal`), its
route (`RequestWrite.close_proposal`) and the MCP `set_request` parameter
(`propose_close`) that carries an agent's proposal."""

import json

import httpx
import pytest

from dw import ledger

from .test_ledger import item, plan, refused, write
from .test_ledger_job_link import WS, client, root  # noqa: F401 (fixtures)
from .test_mcp_server import server_over, tools_of

PROPOSAL = {"state": "done", "reason": "every shot has a keeper"}


def opened(tmp_path, by="agent"):
    return write(tmp_path, plan(), by=by)


def closes(view):
    return [entry.get("close") for entry in view["history"]]


# --- who closes ------------------------------------------------------------


@pytest.mark.parametrize("state", ["done", "abandoned"])
def test_an_agent_cannot_close_a_request(tmp_path, state):
    opened(tmp_path)
    error = refused(tmp_path, plan(revision=1, state=state), code="close_needs_user")
    assert error.status == 422
    assert "close_proposal" in str(error)
    assert ledger.get_request(str(tmp_path), "cats")["state"] == "open"


def test_an_agent_cannot_move_one_closed_state_to_the_other(tmp_path):
    opened(tmp_path)
    write(tmp_path, plan(revision=1, state="done"), by="user")
    refused(tmp_path, plan(revision=2, state="abandoned"), code="close_needs_user")


def test_an_agent_cannot_create_a_request_closed(tmp_path):
    refused(tmp_path, plan(state="done"), code="close_needs_user")


def test_a_user_closes_and_an_agent_reopens(tmp_path):
    opened(tmp_path)
    assert write(tmp_path, plan(revision=1, state="done"), by="user")["state"] == "done"
    assert write(tmp_path, plan(revision=2, state="open"))["state"] == "open"


def test_an_agent_write_that_keeps_a_closed_state_passes(tmp_path):
    opened(tmp_path)
    write(tmp_path, plan(revision=1, state="abandoned"), by="user")
    left_out = write(tmp_path, plan(revision=2, why="tidy"))
    assert left_out["state"] == "abandoned"
    kept = write(tmp_path, plan(revision=3, state="abandoned", why="again"))
    assert kept["state"] == "abandoned"


def test_an_agents_full_ledger_says_to_propose_a_close(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "MAX_REQUESTS", 1)
    opened(tmp_path)
    agent = refused(tmp_path, plan(), code="ledger_full", request_id="dogs")
    assert "propose a close or delete one" in str(agent)
    user = refused(tmp_path, plan(), code="ledger_full", request_id="dogs", by="user")
    assert "close one" in str(user)


# --- the proposal ------------------------------------------------------------


def test_a_proposal_is_stored_pending_and_kept_by_writes_that_leave_it_out(tmp_path):
    opened(tmp_path)
    view = write(tmp_path, plan(revision=1, close_proposal=PROPOSAL))
    proposal = view["close_proposal"]
    assert {
        k: proposal[k] for k in ("state", "reason", "by", "revision", "outcome")
    } == {
        "state": "done",
        "reason": PROPOSAL["reason"],
        "by": "agent",
        "revision": 2,
        "outcome": "pending",
    }
    assert proposal["at"]
    assert view["state"] == "open"
    assert closes(view)[-1] == "close_proposed"

    agent = write(tmp_path, plan(revision=2, why="later"))
    user = write(tmp_path, plan(revision=3, why="user edit"), by="user")
    assert agent["close_proposal"] == proposal == user["close_proposal"]
    assert ledger.get_request(str(tmp_path), "cats")["close_proposal"] == proposal


def test_an_agent_withdraws_its_proposal_with_null(tmp_path):
    opened(tmp_path)
    write(tmp_path, plan(revision=1, close_proposal=PROPOSAL))
    view = write(tmp_path, plan(revision=2, close_proposal=None))
    assert "close_proposal" not in view
    assert closes(view)[-1] == "close_withdrawn"


def test_the_user_declines_with_a_note(tmp_path):
    opened(tmp_path)
    write(tmp_path, plan(revision=1, close_proposal=PROPOSAL))
    view = write(
        tmp_path,
        plan(
            revision=2, close_proposal={"outcome": "declined", "note": "shot b drifts"}
        ),
        by="user",
    )
    proposal = view["close_proposal"]
    assert proposal["outcome"] == "declined"
    assert proposal["note"] == "shot b drifts"
    assert proposal["decided_at"]
    assert view["state"] == "open"
    assert closes(view)[-1] == "close_declined"


def test_a_user_close_accepts_the_pending_proposal(tmp_path):
    opened(tmp_path)
    write(tmp_path, plan(revision=1, close_proposal=PROPOSAL))
    view = write(tmp_path, plan(revision=2, state="done"), by="user")
    assert view["state"] == "done"
    assert view["close_proposal"]["outcome"] == "accepted"
    assert view["close_proposal"]["decided_at"]
    assert closes(view)[-1] == "close_accepted"


def test_a_proposal_after_a_decline_replaces_it_as_pending(tmp_path):
    opened(tmp_path)
    write(tmp_path, plan(revision=1, close_proposal=PROPOSAL))
    write(tmp_path, plan(revision=2, close_proposal={"outcome": "declined"}), by="user")
    again = {"state": "abandoned", "reason": "the user moved on"}
    view = write(tmp_path, plan(revision=3, close_proposal=again))
    proposal = view["close_proposal"]
    assert proposal["outcome"] == "pending"
    assert proposal["state"] == "abandoned"
    assert "decided_at" not in proposal and "note" not in proposal


def test_a_decline_needs_a_pending_proposal(tmp_path):
    opened(tmp_path)
    refused(
        tmp_path,
        plan(revision=1, close_proposal={"outcome": "declined"}),
        code="no_close_proposal",
        by="user",
    )


def test_a_closed_request_takes_no_proposal(tmp_path):
    opened(tmp_path)
    write(tmp_path, plan(revision=1, state="done"), by="user")
    error = refused(
        tmp_path, plan(revision=2, close_proposal=PROPOSAL), code="request_closed"
    )
    assert error.status == 422


@pytest.mark.parametrize(
    "sent",
    [
        {"state": "open", "reason": "r"},
        {"state": "done"},
        {"state": "done", "reason": "two\nlines"},
        {"state": "done", "reason": "x" * 201},
        {"state": "done", "reason": "r", "by": "user"},
    ],
)
def test_a_malformed_proposal_is_refused(tmp_path, sent):
    opened(tmp_path)
    refused(tmp_path, plan(revision=1, close_proposal=sent))


def test_the_list_row_says_a_close_is_proposed_only_while_pending(tmp_path):
    from dw import ledger_views

    def row():
        return ledger_views.summary_view(ledger.get_request(str(tmp_path), "cats"))

    opened(tmp_path)
    assert "close_proposed" not in row()
    write(tmp_path, plan(revision=1, close_proposal=PROPOSAL))
    assert row()["close_proposed"] is True
    write(tmp_path, plan(revision=2, close_proposal={"outcome": "declined"}), by="user")
    assert "close_proposed" not in row()


def test_a_file_without_a_proposal_still_reads(tmp_path):
    opened(tmp_path)
    view = ledger.get_request(str(tmp_path), "cats")
    assert "close_proposal" not in view


# --- the route ---------------------------------------------------------------


def put(client, body, by):  # noqa: F811
    return client.put(f"/api/workspaces/{WS}/requests/cats", json={**body, "by": by})


def test_the_route_carries_the_handshake(client):  # noqa: F811
    assert put(client, plan(items=[item("a")]), "agent").status_code == 200
    closed = put(client, plan(revision=1, items=[item("a")], state="done"), "agent")
    assert closed.status_code == 422, closed.text
    assert closed.json()["detail"]["code"] == "close_needs_user"

    proposed = put(
        client, plan(revision=1, items=[item("a")], close_proposal=PROPOSAL), "agent"
    )
    assert proposed.status_code == 200, proposed.text
    assert proposed.json()["close_proposal"]["outcome"] == "pending"

    read = client.get(f"/api/workspaces/{WS}/requests/cats")
    assert read.json()["close_proposal"]["outcome"] == "pending"
    listed = client.get(f"/api/workspaces/{WS}/requests").json()["requests"]
    assert listed[0]["close_proposed"] is True

    accepted = put(client, plan(revision=2, items=[item("a")], state="done"), "user")
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["close_proposal"]["outcome"] == "accepted"
    assert accepted.json()["history"][-1]["close"] == "close_accepted"


def test_the_route_refuses_unknown_proposal_fields(client):  # noqa: F811
    put(client, plan(items=[item("a")]), "agent")
    r = put(
        client,
        plan(revision=1, items=[item("a")], close_proposal={**PROPOSAL, "x": 1}),
        "agent",
    )
    assert r.status_code == 422, r.text


@pytest.mark.parametrize(
    "sent, form, fault",
    [
        ({"state": "open", "reason": "x"}, "{state: done|abandoned, reason}", "state"),
        (
            {"state": "closed", "reason": "x"},
            "{state: done|abandoned, reason}",
            "state",
        ),
        ({"state": "done"}, "{state: done|abandoned, reason}", "reason"),
        ({"reason": "x"}, "{state: done|abandoned, reason}", "state"),
        ({**PROPOSAL, "colour": "red"}, "{state: done|abandoned, reason}", "colour"),
        ({**PROPOSAL, "by": "user"}, "{state: done|abandoned, reason}", "by"),
        ({"outcome": "accepted"}, '{outcome: "declined", note}', "outcome"),
        ({"outcome": "declined", "x": 1}, '{outcome: "declined", note}', "x"),
    ],
)
def test_a_malformed_proposal_is_one_error_naming_its_form(
    client,  # noqa: F811
    sent,
    form,
    fault,
):
    """Checked as the one form its keys say it is (#1716's bounce): an
    agent's bad proposal is not told the decline's `outcome` is required."""
    put(client, plan(items=[item("a")]), "agent")
    r = put(client, plan(revision=1, items=[item("a")], close_proposal=sent), "agent")
    assert r.status_code == 422, r.text
    [error] = r.json()["detail"]
    assert error["loc"][-1] == "close_proposal"
    assert form in error["msg"]
    assert f"- {fault}: " in error["msg"]
    other = "outcome" if "state" in form else "state"
    assert f"{other}: " not in error["msg"]
    if "state" in form:
        assert "propose_close" in error["msg"]


# --- MCP -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_request_sends_propose_close_only_when_given():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "cats", "revision": 1})

    server = server_over(handler)
    body = {
        "request": "cats",
        "title": "Cats",
        "goal": "three cats",
        "items": [],
        "revision": 1,
        "why": "w",
    }
    await server.call_tool("set_request", body)
    await server.call_tool("set_request", {**body, "propose_close": PROPOSAL})
    await server.call_tool("set_request", {**body, "propose_close": None})

    assert "close_proposal" not in sent[0]
    assert sent[1]["close_proposal"] == PROPOSAL
    assert sent[2]["close_proposal"] is None
    assert all(b["by"] == "agent" for b in sent)


@pytest.mark.asyncio
async def test_set_request_says_only_the_user_closes():
    tools = await tools_of(server_over(lambda r: httpx.Response(200, json={})))
    tool = tools["set_request"]
    assert "propose_close" in tool.input_schema["properties"]
    assert "propose_close" not in tool.input_schema.get("required", [])
    # typed `{state, reason} | null`, and no `{}` sentinel default shown
    schema = tool.input_schema["properties"]["propose_close"]
    proposal, null = schema["anyOf"]
    assert null == {"type": "null"}
    assert proposal["properties"]["state"]["enum"] == ["done", "abandoned"]
    assert sorted(proposal["required"]) == ["reason", "state"]
    assert "default" not in schema
    for words in ("`open` reopens", "only the user", "close_proposal.outcome"):
        assert words in tool.description, words
