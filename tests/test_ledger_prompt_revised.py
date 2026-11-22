"""A take that ran with a saved prompt reads `revised` once that prompt's
text changes (#1558): the pure views, the re-accept basis, `PromptDigests`,
the digests `realize` and the job manager record and read back."""

import json
from datetime import timedelta
import threading
from collections import OrderedDict

import pytest

from dw import ledger, prompts as prompts_module
from dw.ledger_views import is_revised, request_view, revised_because, summary_view
from dw.prompts import PromptDigests, prompt_library, prompt_text_digest
from dw.realize import _inline_prompt
from dw.server.jobs import JobManager

from .test_ledger import item, plan, write
from .test_ledger_revised import ARGS, T0, WF, by_key, stamp, take

D1, D2, D3 = "d1", "d2", "d3"
PARGS = {"prompt": "prompt:p"}


def prompt_item(key="a", arguments=PARGS, changed=0, **kw):
    return {
        "key": key,
        "title": key,
        "depends_on": [],
        "arguments": arguments,
        "workflow": WF,
        "changed_at": stamp(T0 + timedelta(seconds=changed)),
        **kw,
    }


def digests(mapping):
    return lambda name: mapping.get(name)


def ptake(digest_map, **kw):
    return take(**{"arguments": PARGS, **kw}, prompt_digests=digest_map)


# --- 1-3: the pure rule ------------------------------------------------------


def test_prompt_text_change_reads_revised():
    a, jobs = prompt_item(), [ptake({"p": D1})]
    assert not is_revised(a, jobs, digests({"p": D1}))
    assert is_revised(a, jobs, digests({"p": D2}))
    assert revised_because(a, jobs, prompts=digests({"p": D2})) == ["prompt:p"]


def test_a_callable_recorded_digest_is_read():
    a, jobs = prompt_item(), [ptake(lambda: {"p": D1})]
    assert not is_revised(a, jobs, digests({"p": D1}))
    assert is_revised(a, jobs, digests({"p": D2}))


def test_arguments_and_prompt_both_named():
    a = prompt_item(arguments={"prompt": "prompt:p", "x": 1}, changed=100)
    jobs = [ptake({"p": D1}, created=10)]
    because = revised_because(a, jobs, prompts=digests({"p": D2}))
    assert "arguments" in because and "prompt:p" in because


@pytest.mark.parametrize("recorded", [None, "absent"])
def test_a_pre_stage_take_is_not_revised_by_a_prompt(recorded):
    job = take(arguments=PARGS)
    if recorded == "absent":
        assert "prompt_digests" not in job
    else:
        job["prompt_digests"] = None
    assert not is_revised(prompt_item(), [job], digests({"p": D2}))


def test_a_deleted_prompt_stays_done():
    a, jobs = prompt_item(), [ptake({"p": D1})]
    assert not is_revised(a, jobs, digests({}))


# --- 4: dependents -----------------------------------------------------------


def request_of(*items):
    return {"id": "r", "title": "t", "goal": "g", "state": "open", "items": items}


def test_dependents_of_a_revised_item_are_not_ready():
    a = prompt_item()
    b = prompt_item("b", arguments=ARGS, depends_on=["a"])
    request = request_of(a, b)
    jobs = [ptake({"p": D1})]
    moved = digests({"p": D2})
    view = request_view(request, jobs, prompts=moved)
    keys = by_key(view)
    assert keys["a"]["status"] == "revised"
    assert keys["b"]["ready"] is False
    assert summary_view(request, jobs, prompts=moved)["status_counts"]["revised"] == 1
    held = by_key(request_view(request, jobs, prompts=digests({"p": D1})))
    assert held["a"]["status"] == "done" and held["b"]["ready"] is True


# --- 5: finding the names ----------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [{"x": ["prompt:p"]}, {"prompt": "prompt: p "}, {"x": {"y": [{"z": "prompt:p"}]}}],
)
def test_nested_and_spaced_references_name_p(arguments):
    a, jobs = prompt_item(arguments=arguments), [ptake({"p": D1}, arguments=arguments)]
    assert revised_because(a, jobs, prompts=digests({"p": D2})) == ["prompt:p"]


# --- 6: re-accept through put_request ---------------------------------------


def test_reaccept_records_prompt_digests(tmp_path):
    jobs = []
    linked = lambda rid, key: [j for j in jobs if j["item"] == key]  # noqa: E731
    kw = {"linked_jobs": linked, "jobs_for": lambda rid: list(jobs)}

    def read(prompts):
        request = ledger._load(str(tmp_path))["cats"]
        return by_key(ledger.request_view(request, jobs, prompts=prompts))["a"]

    def body(revision, **fields):
        return plan(
            revision=revision,
            items=[item("a", workflow=WF, arguments=PARGS, **fields)],
            why="w",
        )

    view = write(tmp_path, body(0), **kw)
    jobs.append(ptake({"p": D1}, created=10))
    assert read(digests({"p": D1}))["status"] == "done"
    assert read(digests({"p": D2}))["status"] == "revised"

    view = write(
        tmp_path,
        body(view["revision"], accepted={"job": "j1"}),
        prompts=digests({"p": D2}),
        **kw,
    )
    basis = by_key(view)["a"]["accepted_basis"]
    assert basis["prompts"] == {"p": D2} and "at" in basis
    assert read(digests({"p": D2}))["status"] == "done"
    assert read(digests({"p": D3}))["status"] == "revised"

    # a new take on the current text makes it done although j1 stays
    # accepted: the basis judges the accepted take only (C-F1217 step 3)
    jobs.append(ptake({"p": D3}, job="j2", created=500))
    a = read(digests({"p": D3}))
    assert a["status"] == "done" and "revised_because" not in a
    assert a["result"]["job"] == "j1"


def test_basis_judges_only_the_accepted_take():
    a = prompt_item(
        accepted={"job": "j1"},
        accepted_basis={"at": stamp(T0), "prompts": {"p": D2}},
    )
    old = ptake({"p": D1})
    assert not is_revised(a, [old], digests({"p": D2}))
    # j2 ran on D1: the accepted basis (D2) must not make it look current
    other = ptake({"p": D1}, job="j2", created=500)
    a_other = {**a, "accepted": {"job": "j9"}}
    assert is_revised(a_other, [other], digests({"p": D2}))
    assert not is_revised(a_other, [ptake({"p": D2}, job="j2")], digests({"p": D2}))


# --- 7: PromptDigests --------------------------------------------------------


def save(directory, name, text, **fields):
    path = directory / f"{name}.json"
    path.write_text(json.dumps({"text": text, **fields}))
    return path


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts_module, "_digests", OrderedDict())
    directory = tmp_path / "prompts"
    directory.mkdir()
    return directory, prompt_library(prompt_dir=str(directory))


@pytest.fixture
def loads(monkeypatch):
    calls = []
    real = prompts_module.load_prompt

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(prompts_module, "load_prompt", counting)
    return calls


def test_digest_is_the_text_digest(library):
    directory, lib = library
    save(directory, "p", "a harbour")
    assert PromptDigests(lib)("p") == prompt_text_digest("a harbour")


def test_description_and_tags_do_not_move_the_digest(library):
    directory, lib = library
    save(directory, "p", "a harbour")
    first = PromptDigests(lib)("p")
    save(directory, "p", "a harbour", description="new words", tags=["x", "y"])
    assert PromptDigests(lib)("p") == first


def test_a_text_rewrite_moves_the_digest_in_a_new_instance(library):
    directory, lib = library
    save(directory, "p", "a harbour")
    first = PromptDigests(lib)("p")
    save(directory, "p", "a harbour at dusk, longer")
    assert PromptDigests(lib)("p") == prompt_text_digest("a harbour at dusk, longer")
    assert PromptDigests(lib)("p") != first


def test_a_name_resolves_once_per_instance(library, loads):
    directory, lib = library
    save(directory, "p", "a harbour")
    memo = PromptDigests(lib)
    assert memo("p") == memo("p") == memo("p")
    assert len(loads) == 1


def test_the_module_cache_spans_instances(library, loads):
    directory, lib = library
    save(directory, "p", "a harbour")
    PromptDigests(lib)("p")
    PromptDigests(lib)("p")
    assert len(loads) == 1


def test_a_missing_name_is_none(library):
    _, lib = library
    assert PromptDigests(lib)("nope") is None


# --- 8: realize --------------------------------------------------------------


def test_inline_prompt_records_the_digest(tmp_path):
    directory = tmp_path / "prompts"
    directory.mkdir()
    save(directory, "p", "a harbour at dusk")
    annotations = {"prompts": []}
    text = _inline_prompt("prompt: p ", annotations, str(directory), None)
    assert text == "a harbour at dusk"
    assert annotations["prompt_digests"] == {"p": prompt_text_digest(text)}


# --- 9: the job manager reads a take's digests ------------------------------


class FakeManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._digests = OrderedDict()
        self.jobs = {}


def read_digests(directory, job_id="j"):
    return JobManager._prompt_digests(FakeManager(), job_id, directory)


def test_prompt_digests_read_from_the_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"workflow": {"prompt_digests": {"p": "abc"}}})
    )
    assert read_digests(str(tmp_path)) == {"p": "abc"}


def test_prompt_digests_missing_file_or_key_is_none(tmp_path):
    assert read_digests(str(tmp_path)) is None
    (tmp_path / "manifest.json").write_text(json.dumps({"workflow": {}}))
    assert read_digests(str(tmp_path), "k") is None
    assert read_digests(None, "m") is None
