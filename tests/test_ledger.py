"""The requests ledger: `dw/ledger.py`'s model and rules, its routes, and
the file at a workspace root."""

import json
import os
import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from dw import ledger
from dw.security import InvalidInputError, validate_ledger_key
from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.workspace import REQUESTS_FILE_NAME, Workspace
from dw_mcp.client import api_path

from .test_server import ScriptedWorkerManager, success_script

TOKEN = "ledger-token"


def plan(revision=0, items=None, **fields):
    body = {
        "title": "A cat short",
        "goal": "Three shots of a cat",
        "items": items if items is not None else [item("a"), item("b", ["a"])],
        "revision": revision,
        "why": "first plan",
    }
    body.update(fields)
    return body


def item(key, depends_on=(), **fields):
    return {
        "key": key,
        "title": f"shot {key}",
        "depends_on": list(depends_on),
        **fields,
    }


def write(root, body, request_id="cats", by="agent", **kw):
    return ledger.put_request(str(root), request_id, body, by=by, **kw)


def refused(root, body, code="invalid_request", **kw):
    with pytest.raises(ledger.LedgerError) as e:
        write(root, body, **kw)
    assert e.value.code == code, str(e.value)
    return e.value


# --- the key pattern ------------------------------------------------------


class TestKeys:
    @pytest.mark.parametrize("key", ["cats", "a", "0", "a-b-9", "a" * 64])
    def test_a_key_is_accepted(self, key):
        assert validate_ledger_key(key) == key

    @pytest.mark.parametrize(
        "key", ["../x", "a/b", "A", "a" * 65, "", "-a", "a_b", "a.b", "a\n", 5]
    )
    def test_a_bad_key_is_refused(self, key):
        with pytest.raises(InvalidInputError):
            validate_ledger_key(key)

    @pytest.mark.parametrize("key", ["../x", "a/b", "A", "a" * 65])
    def test_a_bad_request_id_or_item_key_is_refused(self, tmp_path, key):
        refused(tmp_path, plan(), request_id=key)
        refused(tmp_path, plan(items=[item(key)]))


# --- revisions and the prompt ----------------------------------------------


class TestRevisions:
    def test_create_then_update(self, tmp_path):
        first = write(tmp_path, plan(prompt="make a cat short"))
        assert first["revision"] == 1
        assert first["prompt"] == "make a cat short"
        assert first["state"] == "open"
        assert [i["status"] for i in first["items"]] == ["planned", "planned"]
        second = write(tmp_path, plan(revision=1, why="second"))
        assert second["revision"] == 2
        assert second["prompt"] == "make a cat short"
        assert [h["why"] for h in second["history"]] == ["first plan", "second"]
        assert os.path.isfile(tmp_path / REQUESTS_FILE_NAME)

    def test_a_stale_revision_is_refused_with_the_current_one(self, tmp_path):
        write(tmp_path, plan())
        error = refused(tmp_path, plan(revision=0), code="stale_revision")
        assert error.status == 409
        assert error.detail()["current_revision"] == 1
        refused(tmp_path, plan(revision=5), code="stale_revision")

    def test_prompt_is_write_once(self, tmp_path):
        write(tmp_path, plan(prompt="the ask"))
        error = refused(
            tmp_path, plan(revision=1, prompt="another"), code="prompt_is_write_once"
        )
        assert error.status == 422
        # the same prompt sent back, or none, is fine
        write(tmp_path, plan(revision=1, prompt="the ask"))
        assert write(tmp_path, plan(revision=2))["prompt"] == "the ask"

    def test_goal_changes_are_recorded_in_history(self, tmp_path):
        write(tmp_path, plan())
        view = write(tmp_path, plan(revision=1, goal="Four shots"))
        assert view["history"][0]["goal"] == "Three shots of a cat"
        assert view["history"][1]["goal"] == "Four shots"
        view = write(tmp_path, plan(revision=2, goal="Four shots"))
        assert "goal" not in view["history"][2]

    def test_history_keeps_the_first_and_the_latest(self, tmp_path):
        for revision in range(ledger.MAX_HISTORY + 5):
            view = write(tmp_path, plan(revision=revision, why=f"r{revision}"))
        assert len(view["history"]) == ledger.MAX_HISTORY
        assert view["history"][0]["why"] == "r0"
        assert view["history"][-1]["why"] == f"r{ledger.MAX_HISTORY + 4}"

    def test_concurrent_writes_one_wins(self, tmp_path):
        write(tmp_path, plan())
        barrier = threading.Barrier(2)
        results = []

        def attempt(why):
            barrier.wait()
            try:
                results.append(write(tmp_path, plan(revision=1, why=why))["revision"])
            except ledger.LedgerError as e:
                results.append(e.code)

        threads = [threading.Thread(target=attempt, args=(w,)) for w in ("x", "y")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(results, key=str) == [2, "stale_revision"]
        assert ledger.get_request(str(tmp_path), "cats")["revision"] == 2


# --- items -----------------------------------------------------------------


def links(*keys):
    def linked(_request_id, key):
        if key in keys:
            return [{"job": f"job-{key}", "status": "succeeded", "run_id": "r1"}]
        return []

    return linked


class TestItems:
    def test_an_item_without_jobs_is_deleted_when_left_out(self, tmp_path):
        write(tmp_path, plan(items=[item("a"), item("b"), item("c")]))
        view = write(tmp_path, plan(revision=1, items=[item("a"), item("c")]))
        assert [i["key"] for i in view["items"]] == ["a", "c"]

    def test_an_item_with_jobs_is_dropped_to_the_end(self, tmp_path):
        write(tmp_path, plan(items=[item("a"), item("b"), item("c")]))
        view = write(
            tmp_path,
            plan(revision=1, items=[item("a"), item("c")]),
            linked_jobs=links("b"),
        )
        assert [(i["key"], i["status"]) for i in view["items"]] == [
            ("a", "planned"),
            ("c", "planned"),
            ("b", "dropped"),
        ]

    def test_an_item_sent_back_dropped_is_removed(self, tmp_path):
        write(tmp_path, plan(items=[item("a"), item("b")]))
        view = write(
            tmp_path,
            plan(revision=1, items=[item("a"), item("b", dropped=True)]),
            linked_jobs=links("b"),
        )
        assert view["items"][-1]["status"] == "dropped"

    def test_re_adding_a_dropped_key_un_drops_it(self, tmp_path):
        write(tmp_path, plan(items=[item("a"), item("b")]))
        write(tmp_path, plan(revision=1, items=[item("a")]), linked_jobs=links("b"))
        view = write(
            tmp_path,
            plan(revision=2, items=[item("b"), item("a")]),
            linked_jobs=links("b"),
        )
        assert [(i["key"], i["status"]) for i in view["items"]] == [
            ("b", "planned"),
            ("a", "planned"),
        ]

    def test_a_plan_read_back_writes_as_it_came(self, tmp_path):
        view = write(tmp_path, plan())
        again = write(tmp_path, plan(revision=1, items=view["items"]))
        assert again["items"] == view["items"]

    def test_an_unknown_item_field_is_refused(self, tmp_path):
        refused(tmp_path, plan(items=[item("a", colour="red")]))

    def test_an_unknown_request_field_is_refused(self, tmp_path):
        refused(tmp_path, plan(owner="me"))

    def test_a_duplicate_key_is_refused(self, tmp_path):
        refused(tmp_path, plan(items=[item("a"), item("a")]))

    def test_accepted_must_be_a_linked_succeeded_job(self, tmp_path):
        refused(tmp_path, plan(items=[item("a", accepted={"job": "job-a"})]))
        view = write(
            tmp_path,
            plan(items=[item("a", accepted={"job": "job-a"})]),
            linked_jobs=links("a"),
        )
        assert view["items"][0]["accepted"] == {"job": "job-a", "run_id": "r1"}

        def failed(_r, _k):
            return [{"job": "job-x", "status": "failed", "run_id": None}]

        refused(
            tmp_path,
            plan(revision=1, items=[item("a", accepted={"job": "job-x"})]),
            linked_jobs=failed,
        )


class TestDependencies:
    def test_a_cycle_is_refused(self, tmp_path):
        error = refused(
            tmp_path,
            plan(items=[item("a", ["c"]), item("b", ["a"]), item("c", ["b"])]),
        )
        assert "cycle" in str(error)

    def test_self_reference_is_refused(self, tmp_path):
        assert "itself" in str(refused(tmp_path, plan(items=[item("a", ["a"])])))

    def test_another_request_or_unknown_key_is_refused(self, tmp_path):
        write(tmp_path, plan(items=[item("x")]), request_id="other")
        refused(tmp_path, plan(items=[item("a", ["x"])]))

    def test_at_most_sixteen(self, tmp_path):
        keys = [f"k{n}" for n in range(17)]
        items = [item(k) for k in keys]
        refused(tmp_path, plan(items=items + [item("a", keys)]))
        write(tmp_path, plan(items=items + [item("a", keys[:16])]))

    def test_a_diamond_is_not_a_cycle(self, tmp_path):
        write(
            tmp_path,
            plan(
                items=[
                    item("a"),
                    item("b", ["a"]),
                    item("c", ["a"]),
                    item("d", ["b", "c"]),
                ]
            ),
        )


# --- caps and values ---------------------------------------------------------


class TestCaps:
    @pytest.mark.parametrize(
        "field,limit",
        [("title", 200), ("why", 200), ("goal", 20000), ("prompt", 20000)],
    )
    def test_text_caps(self, tmp_path, field, limit):
        refused(tmp_path, plan(**{field: "x" * (limit + 1)}))
        write(tmp_path, plan(**{field: "x" * limit}))

    def test_item_title_cap(self, tmp_path):
        refused(tmp_path, plan(items=[item("a", title="x" * 201)]))

    @pytest.mark.parametrize("field", ["title", "why"])
    def test_one_line_fields(self, tmp_path, field):
        refused(tmp_path, plan(**{field: "two\nlines"}))

    @pytest.mark.parametrize("field", ["title", "goal", "why", "prompt"])
    def test_control_characters_are_refused(self, tmp_path, field):
        refused(tmp_path, plan(**{field: "bell\x07"}))

    def test_control_characters_in_item_text_are_refused(self, tmp_path):
        refused(tmp_path, plan(items=[item("a", title="esc\x1b[0m")]))
        refused(tmp_path, plan(items=[item("a", arguments={"p": "nul\x00"})]))

    def test_item_count_cap(self, tmp_path):
        many = [item(f"i{n}") for n in range(ledger.MAX_ITEMS + 1)]
        refused(tmp_path, plan(items=many))
        write(tmp_path, plan(items=many[:-1]))

    def test_request_count_cap(self, tmp_path):
        for n in range(ledger.MAX_REQUESTS):
            write(tmp_path, plan(items=[]), request_id=f"r{n}")
        error = refused(tmp_path, plan(items=[]), code="ledger_full")
        assert error.status == 409

    def test_file_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ledger, "MAX_FILE_BYTES", 4000)
        write(tmp_path, plan(goal="x" * 1000), request_id="one")
        error = refused(tmp_path, plan(goal="x" * 4000), code="ledger_too_large")
        assert error.status == 413
        # nothing was written
        assert ledger.get_request(str(tmp_path), "cats") is None

    def test_the_real_file_cap_is_about_two_megabytes(self):
        assert ledger.MAX_FILE_BYTES == 2 * 1024 * 1024

    @pytest.mark.parametrize(
        "value", [-1, 10_001, float("inf"), float("nan"), "5", True]
    )
    def test_estimate_bounds(self, tmp_path, value):
        refused(tmp_path, plan(items=[item("a", estimate_minutes=value)]))

    @pytest.mark.parametrize("value", [0, 10_000, 2.5, None])
    def test_estimate_in_bounds(self, tmp_path, value):
        view = write(tmp_path, plan(items=[item("a", estimate_minutes=value)]))
        assert view["items"][0]["estimate_minutes"] == value

    def test_arguments(self, tmp_path):
        args = {"prompt": "a cat", "steps": 20, "cfg": 3.5, "flag": True, "x": None}
        view = write(tmp_path, plan(items=[item("a", arguments=args)]))
        assert view["items"][0]["arguments"] == args

    @pytest.mark.parametrize(
        "args",
        [
            {"bad name": 1},
            {"../x": 1},
            {"a": [1, 2]},
            {"a": {"b": 1}},
            {"a": "x" * 20001},
            {"a": float("inf")},
            {f"a{n}": n for n in range(65)},
            ["a"],
        ],
    )
    def test_bad_arguments_are_refused(self, tmp_path, args):
        refused(tmp_path, plan(items=[item("a", arguments=args)]))

    def test_sixty_four_arguments_are_fine(self, tmp_path):
        write(
            tmp_path,
            plan(items=[item("a", arguments={f"a{n}": n for n in range(64)})]),
        )

    @pytest.mark.parametrize("name", ["../x", "/etc/passwd", "a b", "a//b"])
    def test_a_bad_workflow_name_is_refused(self, tmp_path, name):
        refused(tmp_path, plan(items=[item("a", workflow=name)]))

    @pytest.mark.parametrize("name", ["my-shot", "models/flux-dev", "builtin:upscale"])
    def test_a_workflow_name_is_stored(self, tmp_path, name):
        assert (
            write(tmp_path, plan(items=[item("a", workflow=name)]))["items"][0][
                "workflow"
            ]
            == name
        )

    def test_bad_state_and_revision(self, tmp_path):
        refused(tmp_path, plan(state="paused"))
        refused(tmp_path, plan(revision=-1))
        refused(tmp_path, plan(revision=True))


class TestReading:
    def test_no_root_reads_empty_and_refuses_writes(self):
        assert ledger.read_requests(None) == {}
        assert ledger.list_requests(None) == []
        with pytest.raises(ledger.LedgerError) as e:
            ledger.put_request(None, "cats", plan(), by="agent")
        assert (e.value.status, e.value.code) == (409, "no_workspace_root")

    def test_a_corrupt_file_reads_empty_and_is_never_overwritten(self, tmp_path):
        (tmp_path / REQUESTS_FILE_NAME).write_text("{not json")
        assert ledger.list_requests(str(tmp_path)) == []
        refused(tmp_path, plan(), code="ledger_unreadable")
        assert (tmp_path / REQUESTS_FILE_NAME).read_text() == "{not json"

    def test_the_list_is_newest_first_with_counts(self, tmp_path):
        write(tmp_path, plan(), request_id="old")
        write(tmp_path, plan(items=[item("a")]), request_id="new")
        rows = ledger.list_requests(str(tmp_path))
        assert {r["id"] for r in rows} == {"old", "new"}
        old = next(r for r in rows if r["id"] == "old")
        assert old["items"] == 2
        assert old["status_counts"] == {"planned": 2}


# --- over HTTP -----------------------------------------------------------------


@pytest.fixture
def workspace_root(tmp_path):
    return Workspace(tmp_path / "studio", "flag").ensure()


@pytest.fixture
def server(workspace_root, tmp_path):
    def make(token=None, rooted=True):
        manager = JobManager(
            workspace_root.outputs,
            worker_manager=ScriptedWorkerManager(success_script),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=workspace_root.workflows,
        )
        kwargs = {"workspace": workspace_root.root} if rooted else {}
        app = create_app(
            workflow_dir=workspace_root.workflows,
            output_dir=workspace_root.outputs,
            job_manager=manager,
            prompt_dir=workspace_root.prompts,
            asset_dir=workspace_root.assets,
            token=token,
            **kwargs,
        )
        return TestClient(app, base_url="http://localhost")

    return make


URL = "/api/workspaces/default/requests"


class TestRoutes:
    def test_put_get_list(self, server, workspace_root):
        with server() as client:
            created = client.put(f"{URL}/cats", json=plan(prompt="the ask"))
            assert created.status_code == 200, created.text
            body = created.json()
            assert body["revision"] == 1
            assert body["history"][0]["by"] == "user"
            assert client.get(f"{URL}/cats").json() == body
            listing = client.get(URL).json()
            assert listing["requests"][0]["id"] == "cats"
            assert client.get(f"{URL}/dogs").status_code == 404
        stored = json.loads(
            open(os.path.join(workspace_root.root, REQUESTS_FILE_NAME)).read()
        )
        assert list(stored["requests"]) == ["cats"]

    def test_refusals_carry_their_code(self, server):
        with server() as client:
            client.put(f"{URL}/cats", json=plan(prompt="the ask"))
            stale = client.put(f"{URL}/cats", json=plan())
            assert stale.status_code == 409
            assert stale.json()["detail"]["code"] == "stale_revision"
            assert stale.json()["detail"]["current_revision"] == 1
            once = client.put(f"{URL}/cats", json=plan(revision=1, prompt="other"))
            assert once.status_code == 422
            assert once.json()["detail"]["code"] == "prompt_is_write_once"
            cycle = client.put(
                f"{URL}/cats", json=plan(revision=1, items=[item("a", ["a"])])
            )
            assert cycle.status_code == 400
            assert client.get(f"{URL}/A").status_code == 400
            assert client.put(f"{URL}/A", json=plan()).status_code == 400

    def test_a_review_hold_round_trips_and_a_short_one_is_refused(self, server):
        def until(seconds):
            when = datetime.now(timezone.utc) + timedelta(seconds=seconds)
            return when.isoformat(timespec="seconds")

        with server() as client:
            hold = until(3600)
            body = plan(items=[item("a", review_until=hold)])
            assert client.put(f"{URL}/cats", json=body).status_code == 200
            got = client.get(f"{URL}/cats").json()["items"][0]
            assert got["review_until"] == hold
            assert got["review_set_at"] is not None
            assert got["review"]["state"] == "holding"
            short = plan(revision=1, items=[item("a", review_until=until(59))])
            refusal = client.put(f"{URL}/cats", json=short)
            assert refusal.status_code == 422
            assert refusal.json()["detail"]["code"] == "review_too_soon"

    @pytest.mark.parametrize(
        "request_id", ["", "../x", "a/b", "/etc/hosts", "../../etc/hosts", "..", "."]
    )
    def test_an_id_with_a_slash_or_none_meets_the_id_rule(self, server, request_id):
        # The MCP client percent-encodes the id (api_path), and the router
        # decodes it back: an id with a '/' (or an empty one) must still reach
        # the ledger's id rule, not fall through to a bare 404/405.
        path = api_path("api", "workspaces", "default", "requests", request_id)
        with server() as client:
            for reply in (client.get(path), client.put(path, json=plan())):
                assert reply.status_code == 400, (request_id, reply.text)
                assert "request id" in reply.text.lower(), reply.text
            assert client.get(URL).json()["requests"] == []

    def test_a_named_workspace_has_its_own_ledger(self, server, workspace_root):
        with server() as client:
            assert (
                client.post("/api/workspaces", json={"name": "shots"}).status_code
                == 201
            )
            url = "/api/workspaces/shots/requests"
            assert client.put(f"{url}/cats", json=plan()).status_code == 200
            assert client.get(URL).json()["requests"] == []
            assert os.path.isfile(
                os.path.join(workspace_root.root, "shots", REQUESTS_FILE_NAME)
            )
            assert client.get("/api/workspaces/nope/requests").status_code == 404

    def test_deleting_a_workspace_that_holds_requests_succeeds(self, server):
        with server() as client:
            client.post("/api/workspaces", json={"name": "shots"})
            client.put("/api/workspaces/shots/requests/cats", json=plan())
            deleted = client.delete("/api/workspaces/shots?acknowledged=true")
            assert deleted.status_code == 200, deleted.text
            names = [
                w["name"] for w in client.get("/api/workspaces").json()["workspaces"]
            ]
            assert "shots" not in names

    def test_a_workspace_with_no_root_reads_empty_and_refuses(self, server):
        with server(rooted=False) as client:
            assert client.get(URL).json()["requests"] == []
            refused = client.put(f"{URL}/cats", json=plan())
            assert refused.status_code == 409
            assert refused.json()["detail"]["code"] == "no_workspace_root"

    def test_the_token_is_required_and_no_query_token_is_honoured(self, server):
        auth = {"Authorization": f"Bearer {TOKEN}"}
        with server(token=TOKEN) as client:
            assert client.put(f"{URL}/cats", json=plan()).status_code == 401
            assert (
                client.put(f"{URL}/cats?token={TOKEN}", json=plan()).status_code == 401
            )
            assert (
                client.put(f"{URL}/cats", json=plan(), headers=auth).status_code == 200
            )
            for path in (URL, f"{URL}/cats"):
                assert client.get(path).status_code == 401
                assert client.get(f"{path}?token={TOKEN}").status_code == 401
                assert client.get(path, headers=auth).status_code == 200

    def test_no_ledger_route_is_marked_query_token_ok(self, server):
        with server() as client:
            routes = [
                r for r in client.app.routes if "/requests" in getattr(r, "path", "")
            ]
        assert len(routes) == 4
        for route in routes:
            assert not getattr(route.endpoint, "query_token_ok", False), route.path


class TestReviewFixes:
    """0.21.0 review."""

    def test_a_dropped_items_dependency_can_leave_the_plan(self, tmp_path):
        write(tmp_path, plan(items=[item("a"), item("b", ["a"])]))
        write(tmp_path, plan(revision=1, items=[item("a")]), linked_jobs=links("b"))
        view = write(
            tmp_path, plan(revision=2, items=[item("c")]), linked_jobs=links("b")
        )
        assert [(i["key"], i["status"]) for i in view["items"]] == [
            ("c", "planned"),
            ("b", "dropped"),
        ]

    def test_a_live_item_still_needs_its_dependency(self, tmp_path):
        refused(tmp_path, plan(items=[item("b", ["a"])]))

    def test_an_estimate_too_large_for_a_float_is_refused(self, tmp_path):
        refused(tmp_path, plan(items=[item("a", estimate_minutes=10**400)]))

    def test_an_unreadable_ledger_does_not_name_its_folder(self, tmp_path):
        (tmp_path / REQUESTS_FILE_NAME).mkdir()
        error = refused(tmp_path, plan(), code="ledger_unreadable")
        assert str(tmp_path) not in str(error)
