"""The gallery's opt-in `prompts=true` (run id -> the prompt its job was
submitted with) and the memo on its `media=true` duration probe."""

import os

from dw.server import outputs
from dw.server import outputs as job_prompts_mod
from dw.server.outputs import PROMPT_TEXT_MAX, job_prompts, prompt_text, run_prompts

RUN = "20261122-000000-aaaaaaaa"
OTHER_RUN = "20261122-000001-bbbbbbbb"


def _record(history, job_id, run_id, arguments, workspace="default"):
    class Row:
        id = job_id
        workflow_name = "w"
        catalog_name = None
        status = "succeeded"
        created_at = 1.0
        started_at = None
        finished_at = None
        manifest = []
        warnings = []
        error = None
        events = []
        run_dir = None
        acknowledged = "none"

    row = Row()
    row.run_id = run_id
    row.spec = {"workspace": workspace, "arguments": arguments}
    history.record(row)


def _history(client):
    return client.app.state.job_manager.history


def test_prompt_text_reads_a_string_a_list_and_nothing_else():
    assert prompt_text({"prompt": "a cat"}) == "a cat"
    assert prompt_text({"prompt": ["a cat", 3, "a dog"]}) == "a cat\na dog"
    assert prompt_text({"prompt": "x" * (PROMPT_TEXT_MAX + 50)}) == "x" * 2000
    for arguments in (None, {}, {"prompt": ""}, {"prompt": 7}, {"prompt": []}, [1]):
        assert prompt_text(arguments) is None
    assert PROMPT_TEXT_MAX == 2000


def test_the_gallery_lists_each_runs_prompt_when_asked(
    gallery_client, output_root, write_run
):
    write_run(
        output_root, "wf", RUN, [{"step": "a", "files": ["a.png"]}], files=["a.png"]
    )
    write_run(
        output_root,
        "wf",
        OTHER_RUN,
        [{"step": "a", "files": ["b.png"]}],
        files=["b.png"],
    )
    _record(_history(gallery_client), "job-a", RUN, {"prompt": "a red fox"})
    # A run of another workspace, and a job with no prompt, add nothing
    _record(_history(gallery_client), "job-x", "elsewhere", {"prompt": "no"}, "qa")
    _record(_history(gallery_client), "job-b", OTHER_RUN, {"seed": 4})

    body = gallery_client.get("/api/gallery", params={"prompts": "true"}).json()

    assert body["prompts"] == {RUN: "a red fox"}


def test_the_gallery_says_nothing_about_prompts_unless_asked(
    gallery_client, output_root, write_run
):
    write_run(
        output_root, "wf", RUN, [{"step": "a", "files": ["a.png"]}], files=["a.png"]
    )
    _record(_history(gallery_client), "job-a", RUN, {"prompt": "a red fox"})

    body = gallery_client.get("/api/gallery").json()

    # Key for key what a caller that never heard of the option got before
    assert set(body) == {
        "files",
        "total",
        "offset",
        "limit",
        "folders",
        "subfolders",
        "workspace",
    }
    assert "duration_seconds" not in body["files"][0]


def _counting_probe(monkeypatch):
    calls = []

    def probe(path):
        calls.append(path)
        return 2.5

    outputs._duration_at.cache_clear()
    monkeypatch.setattr(outputs, "media_duration", probe)
    return calls


def test_a_second_media_listing_probes_nothing_again(
    gallery_client, output_root, write_run, monkeypatch
):
    write_run(
        output_root,
        "wf",
        RUN,
        [{"step": "a", "files": ["a.mp4", "b.wav"]}],
        files=["a.mp4", "b.wav"],
    )
    calls = _counting_probe(monkeypatch)

    for _ in range(2):
        files = gallery_client.get("/api/gallery", params={"media": "true"}).json()[
            "files"
        ]
        assert [f["duration_seconds"] for f in files] == [2.5, 2.5]

    assert sorted(os.path.basename(path) for path in calls) == ["a.mp4", "b.wav"]


def test_a_rewritten_file_is_probed_again(
    gallery_client, output_root, write_run, monkeypatch
):
    run_dir = write_run(
        output_root, "wf", RUN, [{"step": "a", "files": ["a.mp4"]}], files=["a.mp4"]
    )
    calls = _counting_probe(monkeypatch)
    gallery_client.get("/api/gallery", params={"media": "true"})

    path = os.path.join(run_dir, "a.mp4")
    with open(path, "wb") as handle:
        handle.write(b"a longer file than before")
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    gallery_client.get("/api/gallery", params={"media": "true"})

    assert len(calls) == 2


def test_a_media_listing_reads_the_header_and_never_decodes(
    gallery_client, output_root, write_run, monkeypatch
):
    """`media=true` reads each clip's length from its container header
    (`media_duration`): the figure `probe_media` reports as
    `duration_seconds`, without decoding the soundtrack or the picture."""
    import av

    import dw.media
    from tests.test_media_info import write_mp4, write_wav

    run_dir = write_run(
        output_root,
        "wf",
        RUN,
        [{"step": "a", "files": ["a.mp4", "b.wav"]}],
        files=[],
    )
    write_mp4(os.path.join(run_dir, "a.mp4"))
    write_wav(os.path.join(run_dir, "b.wav"), seconds=1.5)
    expected = {
        name: dw.media.probe_media(os.path.join(run_dir, name))["duration_seconds"]
        for name in ("a.mp4", "b.wav")
    }
    outputs._duration_at.cache_clear()
    decoded = []
    real_decode = av.container.InputContainer.decode

    def counting_decode(self, *args, **kwargs):
        decoded.append(True)
        yield from real_decode(self, *args, **kwargs)

    monkeypatch.setattr(av.container.InputContainer, "decode", counting_decode)

    files = gallery_client.get("/api/gallery", params={"media": "true"}).json()["files"]

    assert {os.path.basename(f["name"]): f["duration_seconds"] for f in files} == (
        expected
    )
    assert decoded == []


def test_a_shared_run_id_takes_the_newest_job(gallery_client):
    history = _history(gallery_client)
    _record(history, "job-old", RUN, {"prompt": "first"})
    _record(history, "job-new", RUN, {"prompt": "second"})
    with history._connect() as connection:
        connection.execute("UPDATE jobs SET created_at = 5 WHERE id = 'job-new'")
        connection.execute("UPDATE jobs SET created_at = 1 WHERE id = 'job-old'")
    assert run_prompts(history, "default", [RUN]) == {RUN: "second"}


def test_an_invalid_row_is_read_once(gallery_client):
    history = _history(gallery_client)
    _record(history, "job-bad", RUN, {"prompt": "x"})
    with history._connect() as connection:
        connection.execute("UPDATE jobs SET arguments = 'not json'")
    calls = []
    real = history.submitted_prompts
    history.submitted_prompts = lambda *a, **k: calls.append(a) or real(*a, **k)
    manager = gallery_client.app.state.job_manager
    job_prompts_mod._PROMPT_MEMO.clear()
    assert job_prompts(manager, ["job-bad"]) == {}
    assert job_prompts(manager, ["job-bad"]) == {}
    assert len(calls) == 1


def test_gallery_prompts_cover_only_the_returned_page(
    gallery_client, output_root, write_run
):
    for run, name in ((RUN, "a.png"), (OTHER_RUN, "b.png")):
        write_run(
            output_root, "wf", run, [{"step": "a", "files": [name]}], files=[name]
        )
        _record(_history(gallery_client), f"job-{name}", run, {"prompt": name})
    body = gallery_client.get(
        "/api/gallery", params={"prompts": "true", "limit": 1}
    ).json()
    assert len(body["prompts"]) == len(body["files"]) == 1


def test_the_run_id_index_exists(gallery_client):
    history = _history(gallery_client)
    with history._connect() as connection:
        names = {r[1] for r in connection.execute("PRAGMA index_list(jobs)")}
    assert "jobs_run_id" in names
