# MCP Field Report Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining items of the 2026-09-10 Opus-over-MCP field report (`/Users/don/testing/10/dw-improvements.md`): MCP payload bloat (#4), workspace sharp edges (#5), audio/video inspection (#6), catalog costs (#9) and the download recipe (#10).

**Architecture:** Two independent PRs. PR-2 is the MCP surface only (`dw_mcp/`, docs) - it changes what tools return and accept, never what the server does. PR-3 adds one server-side probe (`dw/media_info.py`, PyAV) and surfaces it on the gallery-metadata route and its MCP tool. Items 1-3 already shipped on branch `field-report-1-3` (commit `5f8f910`).

**Tech Stack:** Python 3.12, FastAPI, httpx MockTransport for MCP tests, PyAV (already a dependency for muxing), pytest.

**Spec:** `/Users/don/testing/10/dw-improvements.md` (the report) and the assessment in this conversation's memory note `mcp-field-report-2026-09-10`.

## Global Constraints

- Never `eval`/`exec`/`shell=True`; every filesystem read goes through `dw/security.py` validators (`_output_file` in `dw/server/app.py` already confines gallery paths - reuse it, do not open a path the route did not validate).
- MCP tool wrappers in `dw_mcp/server.py` are thin: behaviour lives in `dw_mcp/<module>.py` with a matching `tests/test_mcp_<module>.py` using `httpx.MockTransport`.
- Docstrings on MCP tools are the tool descriptions an agent reads; a change in behaviour changes the docstring in the same commit.
- `docs/MCP.md` and `docs/WORKFLOW_GUIDE.md` ("Authoring a workflow from an agent") describe the same conventions; change both when one changes.
- Commit messages end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

---

# PR-2: MCP surface (branch `field-report-mcp-surface`, from `master` after PR-1 merges)

### Task 1: `export_job` stops inlining the three JSON files

**Files:**
- Modify: `dw_mcp/exports.py:12-48`
- Modify: `dw_mcp/server.py:675-700` (docstring only)
- Test: `tests/test_mcp_exports.py:60-68`

**Interfaces:**
- Produces: `export_job(client, job_id, overwrite=False) -> dict` with keys `job_id, where, directory, zip_url, files, total_bytes, missing, next` - no `workflow`, `manifest`, `job`.

- [ ] **Step 1: Rewrite the failing test**

Replace `test_the_three_json_files_come_back_inline` in `tests/test_mcp_exports.py` with:

```python
def test_the_three_json_files_stay_in_the_zip():
    """A music-video export inlined 55 KB of workflow, manifest and job row
    that the zip already carries and get_job_workflow / get_job already
    serve - it blew past the tool output limit. The listing says they are
    there; the bytes are not repeated."""
    client, _ = exporting()

    result = exports.export_job(client, "job-1")

    assert "workflow" not in result
    assert "manifest" not in result
    assert "job" not in result
    assert "get_job_workflow" in result["next"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_mcp_exports.py -q -k stay_in_the_zip`
Expected: FAIL on `assert "workflow" not in result`.

- [ ] **Step 3: Drop the three keys and say where they live**

In `dw_mcp/exports.py`, remove the `"workflow"`, `"manifest"`, `"job"` entries from the returned dict and change the `next` text to:

```python
        "next": "The directory is on the server. To give the user the files, "
        "fetch zip_url and unpack it into exports/ under the session's "
        "working directory - it is the user's deliverable, not a temporary "
        "file, so not a scratch or temp directory. The archive already "
        "unpacks into one folder named after the job id; do not create "
        "that folder first or the id is doubled in the path. workflow.json, "
        "manifest.json and job.json are inside it - they are not repeated "
        "here; get_job_workflow and get_job serve them individually.",
```

Update the docstring's "Returns the directory, the zip URL, the file list with sizes and the total, and the three JSON files inline" to "... the file list with sizes and the total. The three JSON files are in the zip, not repeated here." Make the same edit to the `export_job` docstring in `dw_mcp/server.py`.

- [ ] **Step 4: Run the file's tests**

Run: `python -m pytest tests/test_mcp_exports.py tests/test_mcp_server.py -q -k export`
Expected: all pass. If `test_mcp_server.py` asserts on the inline keys, update that assertion the same way.

- [ ] **Step 5: Commit**

```bash
git add dw_mcp/exports.py dw_mcp/server.py tests/test_mcp_exports.py tests/test_mcp_server.py
git commit -m "export_job: leave the three JSON files in the zip"
```

### Task 2: `wait_for_job` returns a slim job

**Files:**
- Modify: `dw_mcp/diagnose.py:103-142`
- Modify: `dw_mcp/server.py:633-641` (docstring)
- Test: `tests/test_mcp_diagnose.py:274-330`

**Interfaces:**
- Produces: `slim_job(job: dict) -> dict` in `dw_mcp/diagnose.py`, keeping exactly `id, workflow_name, status, created_at, started_at, finished_at, workspace, run_id, warnings, error, event_count` plus `manifest` only when status is terminal. `arguments` and `traceback` are dropped. `wait_for_job`'s `job` value is `slim_job(job)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcp_diagnose.py`:

```python
FAT_JOB = {
    "id": "job-1",
    "workflow_name": "minimax/dialogue-short",
    "status": "running",
    "created_at": 1.0,
    "started_at": 2.0,
    "finished_at": None,
    "workspace": "default",
    "run_id": None,
    "arguments": {"shot_1_cold_open": "x" * 6000},
    "warnings": [],
    "manifest": None,
    "error": None,
    "traceback": None,
    "event_count": 12,
    "run_dir": None,
}


def test_wait_for_job_does_not_echo_the_arguments_on_every_poll(monkeypatch):
    """An H3 workflow's arguments are 4-6k tokens of prompt text; a
    45-minute render is polled many times. They are get_job's to serve,
    once."""
    monkeypatch.setattr(diagnose, "MAX_WAIT_SECONDS", 0)
    client, _ = scripted({("GET", "/api/jobs/job-1"): (200, FAT_JOB)})

    result = diagnose.wait_for_job(client, "job-1", timeout_seconds=0)

    assert result["still_running"] is True
    assert "arguments" not in result["job"]
    assert "traceback" not in result["job"]
    assert result["job"]["status"] == "running"
    assert result["job"]["event_count"] == 12


def test_wait_for_job_keeps_the_manifest_and_error_once_terminal(monkeypatch):
    done = {**FAT_JOB, "status": "failed", "manifest": {"steps": []}, "error": "boom"}
    client, _ = scripted({("GET", "/api/jobs/job-1"): (200, done)})

    result = diagnose.wait_for_job(client, "job-1")

    assert result["job"]["manifest"] == {"steps": []}
    assert result["job"]["error"] == "boom"
    assert "arguments" not in result["job"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_mcp_diagnose.py -q -k "echo_the_arguments or once_terminal"`
Expected: FAIL on `assert "arguments" not in result["job"]`.

- [ ] **Step 3: Implement `slim_job` and use it**

In `dw_mcp/diagnose.py`, above `wait_for_job`:

```python
_SLIM_KEYS = (
    "id",
    "workflow_name",
    "status",
    "created_at",
    "started_at",
    "finished_at",
    "workspace",
    "run_id",
    "warnings",
    "error",
    "event_count",
)


def slim_job(job):
    """A job row without its arguments and traceback - what a poll needs.

    The arguments of an H3 workflow are thousands of tokens of prompt text,
    repeated on every poll of a long render; get_job serves them once. The
    manifest is kept only once the job is terminal, when it names files.
    """
    slim = {key: job.get(key) for key in _SLIM_KEYS if key in job}
    if job.get("status") in TERMINAL_STATUSES:
        slim["manifest"] = job.get("manifest")
    return slim
```

In both `return` dicts inside `wait_for_job`, change `"job": job` to `"job": slim_job(job)`. Add to the terminal return a `"next": "get_job(job_id) for the arguments and traceback, get_job_workflow(job_id) for the realized workflow."`.

Update the `wait_for_job` docstrings in `dw_mcp/diagnose.py` and `dw_mcp/server.py`: append "Returns a slim job - status, warnings, error, and the manifest once finished - without the arguments; get_job has those."

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_mcp_diagnose.py tests/test_mcp_server.py -q`
Expected: all pass. `test_wait_for_job_returns_promptly_once_terminal` may assert `result["job"] == <full row>`; if so change it to compare `result["job"]["status"]` and `result["job"]["id"]`.

- [ ] **Step 5: Commit**

```bash
git add dw_mcp/diagnose.py dw_mcp/server.py tests/test_mcp_diagnose.py tests/test_mcp_server.py
git commit -m "wait_for_job: poll with a slim job, not the arguments"
```

### Task 3: `run_workflow` and `validate_workflow` take an optional `workspace`

**Files:**
- Modify: `dw_mcp/diagnose.py:33-66`
- Modify: `dw_mcp/authoring.py:12-26`
- Modify: `dw_mcp/server.py:483-491, 582-605`
- Test: `tests/test_mcp_diagnose.py`, `tests/test_mcp_authoring.py`

**Interfaces:**
- Consumes: `DwClient._scoped` (`dw_mcp/client.py:230`) adds `params["workspace"]` only when `client.workspace != DEFAULT_WORKSPACE`, and uses `setdefault`, so a `workspace` already present in `params` wins.
- Produces: `run_workflow(client, ..., workspace=None)` and `validate_workflow(client, workflow=None, name=None, workspace=None)`; when `workspace` is given it is sent as the `workspace` query parameter of that one request and the session's workspace is unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcp_diagnose.py`:

```python
def test_run_pins_a_job_to_a_named_workspace_without_switching():
    """A session restart resets the session workspace to default, and a
    concat job then resolved output: references against the wrong root.
    A run can name its workspace itself, for that one request."""
    client, seen = submitting()
    client.workspace = "music-video"

    diagnose.run_workflow(
        client, workflow_path="w", acknowledged_cost=True, workspace="dialogue-short"
    )

    assert seen[0]["params"]["workspace"] == "dialogue-short"
    assert client.workspace == "music-video"


def test_run_sends_the_session_workspace_when_none_is_named():
    client, seen = submitting()
    client.workspace = "music-video"

    diagnose.run_workflow(client, workflow_path="w", acknowledged_cost=True)

    assert seen[0]["params"]["workspace"] == "music-video"
```

Append to `tests/test_mcp_authoring.py` (use its existing `scripted`/recording helper; the route is `("POST", "/api/validate")`):

```python
def test_validate_can_name_a_workspace_for_one_request():
    client, seen = scripted({("POST", "/api/validate"): (200, {"valid": True})})

    authoring.validate_workflow(client, name="w", workspace="dialogue-short")

    assert seen[0]["params"]["workspace"] == "dialogue-short"
    assert client.workspace == DEFAULT_WORKSPACE
```

`tests/test_mcp_authoring.py`'s `scripted` records only `(method, path)` keys in `seen`. Add a second helper beside it rather than changing the existing one's shape:

```python
def scripted_with_params(routes):
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append({"key": key, "params": dict(request.url.params)})
        status, body = routes.get(key, (404, {"detail": f"unrouted {key}"}))
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen
```

and use it in the new test (`client, seen = scripted_with_params(...)`). Import `DEFAULT_WORKSPACE` from `dw_mcp.client`.

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_mcp_diagnose.py tests/test_mcp_authoring.py -q -k workspace`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'workspace'`.

- [ ] **Step 3: Thread the parameter**

`dw_mcp/diagnose.py`:

```python
def run_workflow(
    client,
    workflow_path=None,
    inline_workflow=None,
    arguments=None,
    acknowledged_cost=False,
    workspace=None,
):
    ...
    # A named workspace pins this one job rather than the session: a
    # restarted session forgets use_workspace, and a job that resolves
    # output: references in the wrong root fails after it was queued
    params = {"workspace": workspace} if workspace else None
    job = client.post_json("/api/jobs", payload, params=params)
```

`dw_mcp/authoring.py`:

```python
def validate_workflow(client, workflow=None, name=None, workspace=None):
    ...
    params = {"workspace": workspace} if workspace else None
    if name is not None:
        return client.post_json("/api/validate", {"workflow_path": name}, params=params)
    return client.post_json("/api/validate", {"workflow": workflow}, params=params)
```

`dw_mcp/server.py`: add `workspace: str | None = None` to both tool signatures, pass it through, and append to each docstring: "`workspace` names the workspace for this one call without switching the session to it - use it to pin a job whose `output:` or `asset:` references live in a workspace other than the session's."

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_mcp_diagnose.py tests/test_mcp_authoring.py tests/test_mcp_server.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add dw_mcp/diagnose.py dw_mcp/authoring.py dw_mcp/server.py tests/test_mcp_diagnose.py tests/test_mcp_authoring.py
git commit -m "run_workflow/validate_workflow: pin one call to a named workspace"
```

### Task 4: `create_workspace(use=...)` and a louder result

**Files:**
- Modify: `dw_mcp/workspaces.py:49-53`
- Modify: `dw_mcp/server.py:460-466`
- Test: `tests/test_mcp_workspaces.py:84-89`

**Interfaces:**
- Consumes: `use_workspace(client, name)` (`dw_mcp/workspaces.py`) which sets `client.workspace` after checking the server's listing.
- Produces: `create_workspace(client, name, use=False) -> dict` = the server's body plus `current` (the session workspace after the call) and `next`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_mcp_workspaces.py` `TestLifecycle`, add beside `test_creating_one_does_not_switch_to_it`:

```python
    def test_creating_one_says_it_did_not_switch(self):
        """A create-then-run sequence landed a five-shot job in the wrong
        workspace; the result now says where the session still is."""
        client, _ = recording({"name": "shots"})
        result = create_workspace(client, "shots")
        assert result["current"] == DEFAULT_WORKSPACE
        assert "use_workspace" in result["next"]

    def test_creating_with_use_switches_to_it(self):
        client, seen = recording({"name": "shots"})
        result = create_workspace(client, "shots", use=True)
        assert client.workspace == "shots"
        assert result["current"] == "shots"
```

`recording` scripts one response for every request; `use_workspace` GETs the listing and checks the name is in it. If the recorded body `{"name": "shots"}` makes `use_workspace` raise "No workspace named", script the listing instead: look at how `test_deleting_the_current_one_falls_back_to_the_default` uses `listing("default", "shots")` and use that body for the `use=True` test.

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_mcp_workspaces.py -q -k "says_it_did_not_switch or with_use"`
Expected: FAIL (`KeyError: 'current'`, then `TypeError` on `use`).

- [ ] **Step 3: Implement**

```python
def create_workspace(client, name, use=False):
    """Make a new workspace on the server. It gets its own workflows, assets
    and outputs, and shares the server's one prompt library. Creating it
    does not switch to it unless `use` is true - the natural
    create-then-run sequence otherwise runs in the workspace the session
    was already in, and the result says which that is."""
    body = client.post_json("/api/workspaces", {"name": name})
    if use:
        use_workspace(client, name)
    return {
        **body,
        "current": client.workspace,
        "next": (
            f"This session now works in '{client.workspace}'."
            if use
            else f"This session still works in '{client.workspace}' - call "
            f"use_workspace('{name}') or pass workspace='{name}' to "
            f"run_workflow before running anything meant for it."
        ),
    }
```

`dw_mcp/server.py`: `def create_workspace(name: str, use: bool = False) -> dict:` passing `use=use`; docstring: replace "Creating does not switch to it: call use_workspace after." with "Pass use=true to switch this session to it as well; otherwise the session stays where it was and the result says so."

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_mcp_workspaces.py tests/test_mcp_server.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add dw_mcp/workspaces.py dw_mcp/server.py tests/test_mcp_workspaces.py
git commit -m "create_workspace: optional use=, and say where the session still is"
```

### Task 5: Catalog costs for the cut templates

**Files:**
- Modify: `workflows/templates/minimax/music-video.json:2-4`, `workflows/templates/minimax/dialogue-short.json:2-4`, `workflows/templates/assemble-and-score.json:2-4`, `workflows/templates/dissolve-between-shots.json:2-4`
- Test: `tests/test_catalog_structure.py`

**Interfaces:**
- Consumes: the `cost` array format at `workflows/templates/minimax/music.json:4`: `[{"device": "cuda", "name": "RTX 3090", "vram_gb": 24, "minutes": 3.6}]`, read at `dw/server/app.py:223`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_catalog_structure.py`:

```python
COSTED = {
    "workflows/templates/minimax/music-video.json": 35,
    "workflows/templates/minimax/dialogue-short.json": 42,
    "workflows/templates/assemble-and-score.json": 0.2,
    "workflows/templates/dissolve-between-shots.json": 0.2,
}


@pytest.mark.parametrize("path,minutes", sorted(COSTED.items()))
def test_the_cut_templates_quote_a_measured_cost(path, minutes):
    """Measured on an RTX 3090, 2026-09-10; without a figure an agent
    cannot quote a price before spending 40 minutes of GPU."""
    definition = json.load(open(path))
    entry = definition["cost"][0]
    assert entry["name"] == "RTX 3090"
    assert entry["minutes"] == minutes
```

(Use the file's existing imports; it already opens every workflow, so `json` and `pytest` are present.)

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_catalog_structure.py -q -k measured_cost`
Expected: 4 FAIL with `KeyError: 'cost'`.

- [ ] **Step 3: Add the cost arrays**

Insert after `"description"` (or after `"summary"` where one exists) in each file, matching the indentation used in `music.json`:

```json
    "cost": [
        {"device": "cuda", "name": "RTX 3090", "vram_gb": 24, "minutes": 35}
    ],
```

with `35` (music-video: 4 shots + song + portrait), `42` (dialogue-short: 5 shots + 2 portraits), `0.2` (assemble-and-score and dissolve-between-shots: no generation, 5-13 s measured).

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_catalog_structure.py tests/test_examples.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add workflows/templates
git commit -m "Catalog: measured 3090 costs for the cut templates"
```

### Task 6: The download recipe in the guide

**Files:**
- Modify: `docs/WORKFLOW_GUIDE.md` after step 6 of "The loop" (line ~346)
- Modify: `docs/MCP.md` (a pointer in the client-configuration section, after the `claude mcp add` block at line ~91)

- [ ] **Step 1: Add a step 7 to "The loop"**

After the `get_output_image` step, add:

```markdown
7. Getting the files to the user's machine. `download_output` and `export_job`
   write on the machine running `dw.serve`, which over a remote `--mcp`
   endpoint is the GPU box. The last mile of every deliverable is the `url`
   each `list_gallery` entry carries (or `export_job`'s `zip_url`), fetched
   with the same bearer token the MCP connection uses:

       curl -H "Authorization: Bearer $DW_TOKEN" \
            -o exports/still.png "$DW_SERVER/outputs/ltx2/Gyre/20260910-.../still.png"

   Put the result under `exports/` in the session's working directory - it is
   the user's deliverable, not a temporary file.
```

- [ ] **Step 2: Point to it from MCP.md**

After the `claude mcp add ... --header` block, add one sentence: "The same token fetches generated files: see step 7 of `The loop` in [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#the-loop)."

- [ ] **Step 3: Check the guide's tests still pass**

Run: `python -m pytest tests/test_mcp_guides.py tests/test_plugin_skills.py -q`
Expected: pass (the guides tool indexes sections by heading; no heading changed).

- [ ] **Step 4: Commit**

```bash
git add docs/WORKFLOW_GUIDE.md docs/MCP.md
git commit -m "Guide: the last mile - fetching outputs with the MCP token"
```

### Task 7: Update the report's authors

- [ ] **Step 1:** Run the full suite: `python -m pytest tests -q` - expected all pass.
- [ ] **Step 2:** Open a PR titled "MCP surface: slim payloads, pinned workspaces, costs" with the field-report item numbers (#4, #5, #9, #10) in the body.

---

# PR-3: Media info on gallery metadata (branch `field-report-media-info`, independent of PR-2)

### Task 8: `probe_media(path)` in `dw/media_info.py`

**Files:**
- Create: `dw/media_info.py`
- Test: `tests/test_media_info.py`

**Interfaces:**
- Produces: `probe_media(path: str) -> dict | None`. For audio: `{"kind": "audio", "duration_seconds": float, "sample_rate": int, "channels": int, "peak_dbfs": float, "mean_dbfs": float}`. For video: the same plus `"kind": "video", "fps": float, "frame_count": int, "width": int, "height": int`; the audio fields are present only when the file carries an audio stream. `None` for a file PyAV cannot open. Levels are computed from the decoded waveform: `peak_dbfs = 20*log10(max|x|)`, `mean_dbfs = 20*log10(rms(x))`, `-inf` clamped to `-120.0`.

- [ ] **Step 1: Write the failing tests**

`tests/test_media_info.py`:

```python
"""probe_media reports what the server knows about a generated file and
would otherwise not say - an agent cannot listen, so duration and level
are the only way it checks an audio deliverable."""

import math

import numpy
import pytest

from dw.media_info import probe_media


def write_wav(path, seconds=2.0, sample_rate=8000, amplitude=0.5):
    import wave

    t = numpy.arange(int(seconds * sample_rate)) / sample_rate
    samples = (numpy.sin(2 * numpy.pi * 220 * t) * amplitude * 32767).astype("<i2")
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(numpy.stack([samples, samples], 1).tobytes())


def write_mp4(path, frames=12, fps=6, width=32, height=16, with_audio=True):
    import av

    container = av.open(str(path), "w")
    video = container.add_stream("libx264", rate=fps)
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    audio = container.add_stream("aac", rate=8000) if with_audio else None
    if audio is not None:
        audio.layout = "stereo"
    for _ in range(frames):
        frame = av.VideoFrame.from_ndarray(
            numpy.zeros((height, width, 3), numpy.uint8), format="rgb24"
        )
        for packet in video.encode(frame):
            container.mux(packet)
    if audio is not None:
        total = 8000 * frames // fps
        tone = numpy.stack([numpy.full(total, 0.25, numpy.float32)] * 2)
        for start in range(0, total, 1024):
            chunk = av.AudioFrame.from_ndarray(
                numpy.ascontiguousarray(tone[:, start : start + 1024]),
                format="fltp",
                layout="stereo",
            )
            chunk.sample_rate = 8000
            chunk.pts = start
            for packet in audio.encode(chunk):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    container.close()


def test_a_wav_reports_duration_rate_channels_and_level(tmp_path):
    write_wav(tmp_path / "score.wav", seconds=2.0, sample_rate=8000, amplitude=0.5)

    info = probe_media(str(tmp_path / "score.wav"))

    assert info["kind"] == "audio"
    assert info["duration_seconds"] == pytest.approx(2.0, abs=0.01)
    assert info["sample_rate"] == 8000
    assert info["channels"] == 2
    # a 0.5-amplitude sine peaks at -6 dBFS and sits at -9 dBFS rms
    assert info["peak_dbfs"] == pytest.approx(-6.0, abs=0.2)
    assert info["mean_dbfs"] == pytest.approx(-9.0, abs=0.2)


def test_a_video_reports_its_picture_and_its_soundtrack(tmp_path):
    write_mp4(tmp_path / "shot.mp4", frames=12, fps=6, width=32, height=16)

    info = probe_media(str(tmp_path / "shot.mp4"))

    assert info["kind"] == "video"
    assert info["frame_count"] == 12
    assert info["fps"] == pytest.approx(6.0)
    assert (info["width"], info["height"]) == (32, 16)
    assert info["duration_seconds"] == pytest.approx(2.0, abs=0.1)
    assert info["sample_rate"] == 8000
    assert info["channels"] == 2
    assert info["peak_dbfs"] == pytest.approx(-12.0, abs=1.0)


def test_a_silent_video_has_no_audio_fields(tmp_path):
    write_mp4(tmp_path / "mute.mp4", with_audio=False)

    info = probe_media(str(tmp_path / "mute.mp4"))

    assert info["kind"] == "video"
    assert "sample_rate" not in info
    assert "peak_dbfs" not in info


def test_silence_is_clamped_not_minus_infinity(tmp_path):
    write_wav(tmp_path / "quiet.wav", amplitude=0.0)

    info = probe_media(str(tmp_path / "quiet.wav"))

    assert info["peak_dbfs"] == -120.0
    assert not math.isinf(info["mean_dbfs"])


def test_a_file_that_is_not_media_answers_none(tmp_path):
    (tmp_path / "notes.txt").write_text("not media")

    assert probe_media(str(tmp_path / "notes.txt")) is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_media_info.py -q`
Expected: FAIL with `ModuleNotFoundError: dw.media_info`.

- [ ] **Step 3: Implement `dw/media_info.py`**

```python
"""What the server knows about a generated media file and would otherwise
not say. An agent cannot listen: duration against a ceiling, peak against
a normalization target and the level at a seam are the only checks it can
make on an audio deliverable, and every one of them was being made by
fetching the file and running ffprobe by hand.
"""

import logging
import math

import av
import numpy

logger = logging.getLogger("dw")

# The floor a level is reported at rather than -inf, which JSON cannot carry
SILENCE_DBFS = -120.0


def _dbfs(value):
    if value <= 0:
        return SILENCE_DBFS
    return max(SILENCE_DBFS, 20.0 * math.log10(float(value)))


def _audio_levels(container, stream):
    """Peak and rms of the whole decoded track, in dBFS."""
    peak = 0.0
    total = 0.0
    count = 0
    for frame in container.decode(stream):
        samples = frame.to_ndarray()
        if samples.dtype.kind in "iu":
            samples = samples.astype(numpy.float32) / numpy.iinfo(samples.dtype).max
        samples = samples.astype(numpy.float32)
        peak = max(peak, float(numpy.abs(samples).max(initial=0.0)))
        total += float(numpy.square(samples).sum())
        count += samples.size
    rms = math.sqrt(total / count) if count else 0.0
    return _dbfs(peak), _dbfs(rms)


def probe_media(path):
    """Duration, format and level of an audio or video file, or None.

    Video answers fps, frame_count, width and height, plus the soundtrack's
    sample_rate, channels, peak_dbfs and mean_dbfs when it carries one;
    audio answers the soundtrack fields. Levels come from decoding the
    whole track, which is cheap next to generating it.
    """
    try:
        container = av.open(path)
    except Exception as e:
        logger.debug(f"Not probeable as media: {path}: {e}")
        return None
    with container:
        video = container.streams.video[0] if container.streams.video else None
        audio = container.streams.audio[0] if container.streams.audio else None
        if video is None and audio is None:
            return None
        info = {}
        if video is not None:
            info["kind"] = "video"
            info["fps"] = float(video.average_rate) if video.average_rate else None
            info["frame_count"] = int(video.frames) if video.frames else None
            info["width"] = int(video.width)
            info["height"] = int(video.height)
        else:
            info["kind"] = "audio"
        if container.duration is not None:
            info["duration_seconds"] = container.duration / av.time_base
        if audio is not None:
            info["sample_rate"] = int(audio.rate)
            info["channels"] = int(audio.channels)
            info["peak_dbfs"], info["mean_dbfs"] = _audio_levels(container, audio)
        return info
```

If `frame_count` comes back `None` for the test mp4 (some muxers do not write the count), count the decoded video frames instead: `sum(1 for _ in container.decode(video))` - do that only when `video.frames` is 0, and decode video before audio so the two passes do not interleave.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_media_info.py -q`
Expected: 5 pass. Adjust the level tolerances only if AAC's lossy encode shifts the mp4 peak by more than 1 dB - the wav assertions must hold exactly as written.

- [ ] **Step 5: Commit**

```bash
git add dw/media_info.py tests/test_media_info.py
git commit -m "media_info: probe duration, format and level of a generated file"
```

### Task 9: The gallery metadata route carries `media`

**Files:**
- Modify: `dw/server/app.py:1820-1834`
- Test: `tests/test_server.py:1062-1110`

**Interfaces:**
- Consumes: `probe_media` from Task 8; `_output_file(name, ws.outputs)` already validates and confines the path.
- Produces: `GET /api/gallery/{name}/metadata` answers `{"name", "metadata", "job", "media"}`, `media` being `probe_media(path)` for a file whose extension is in `MEDIA_KINDS` as `audio` or `video`, else `None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`, next to `test_gallery_lists_media_and_reads_metadata`:

```python
def test_gallery_metadata_describes_audio_and_video(server, tmp_path):
    """A generated mp3 answered metadata: null and nothing else, so every
    duration and level check was ffprobe by hand. The route now says what
    the server knows."""
    from tests.test_media_info import write_mp4, write_wav

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        write_wav(outputs / "score-gen.0-0.0.wav", seconds=2.0)
        write_mp4(outputs / "shot-gen.0-0.0.mp4", frames=12, fps=6)
        Image.new("RGB", (4, 4)).save(outputs / "still-gen.0-0.0.png")

        score = client.get("/api/gallery/score-gen.0-0.0.wav/metadata").json()
        assert score["metadata"] is None
        assert score["media"]["kind"] == "audio"
        assert score["media"]["duration_seconds"] == pytest.approx(2.0, abs=0.01)
        assert score["media"]["channels"] == 2

        shot = client.get("/api/gallery/shot-gen.0-0.0.mp4/metadata").json()
        assert shot["media"]["kind"] == "video"
        assert shot["media"]["frame_count"] == 12

        still = client.get("/api/gallery/still-gen.0-0.0.png/metadata").json()
        assert still["media"] is None
```

(`from PIL import Image` and `pytest` are already imported in that file; if `tests/` is not a package, move `write_wav`/`write_mp4` into `tests/conftest.py` as plain functions and import from there instead.)

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_server.py -q -k describes_audio_and_video`
Expected: FAIL with `KeyError: 'media'`.

- [ ] **Step 3: Add `media` to the route**

In `dw/server/app.py`, import `from dw.media_info import probe_media` beside the other `dw.` imports, and change `gallery_metadata`:

```python
    @app.get("/api/gallery/{name:path}/metadata")
    def gallery_metadata(name: str, ws: Workspace = Depends(selected_workspace)):
        """Generation metadata embedded in a saved image ('workflow' inside
        it is the full definition the editor can reopen), plus the job that
        produced the file when history remembers one, plus - for audio and
        video - what the file itself holds: duration, format and level,
        which is how an agent that cannot listen checks a track."""
        path = _output_file(name, ws.outputs)
        metadata = read_embedded_metadata(path)
        try:
            job = manager.history.job_for_file(name, workspace=ws.name)
        except Exception:
            job = None
        extension = os.path.splitext(path)[1].lower()
        media = (
            probe_media(path)
            if MEDIA_KINDS.get(extension) in ("audio", "video")
            else None
        )
        return {"name": name, "metadata": metadata, "job": job, "media": media}
```

Keep the existing "Scoped to this workspace" comment above the `job_for_file` call.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_server.py -q -k gallery`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add dw/server/app.py tests/test_server.py
git commit -m "Gallery metadata: describe audio and video files"
```

### Task 10: The MCP tool and the skills say what to check

**Files:**
- Modify: `dw_mcp/catalog.py:104-107`, `dw_mcp/server.py:287-295`
- Modify: `plugins/dw/skills/minimax-music3/SKILL.md` (the `get_gallery_metadata` duration check), `plugins/dw/skills/minimax-h3/SKILL.md` (if it mentions checking audio)
- Modify: `docs/MCP.md` tool table entry for `get_gallery_metadata`
- Test: `tests/test_mcp_catalog.py`, `tests/test_plugin_skills.py`

**Interfaces:**
- Consumes: the `media` block from Task 9.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcp_catalog.py` (using its `scripted` helper and the metadata route):

```python
def test_gallery_metadata_passes_the_media_block_through_and_says_how_to_read_it():
    body = {
        "name": "score.mp3",
        "metadata": None,
        "job": {"id": "job-1", "status": "succeeded"},
        "media": {"kind": "audio", "duration_seconds": 45.05, "peak_dbfs": -1.0},
    }
    client, _ = scripted({("GET", "/api/gallery/score.mp3/metadata"): (200, body)})

    result = catalog.get_gallery_metadata(client, "score.mp3")

    assert result["media"]["duration_seconds"] == 45.05
    assert "audio_duration" in result["next"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_mcp_catalog.py -q -k media_block`
Expected: FAIL with `KeyError: 'next'`.

- [ ] **Step 3: Implement**

`dw_mcp/catalog.py`:

```python
def get_gallery_metadata(client, name):
    """Metadata embedded in a saved file: the full workflow that made it,
    plus the job that produced it when history remembers one, plus for
    audio and video what the file holds - duration, sample rate, channels,
    fps, size, peak and mean level in dBFS."""
    body = client.get_json(api_path("api", "gallery", name, "metadata"))
    media = body.get("media")
    if media and media.get("kind") in ("audio", "video"):
        body["next"] = (
            "Check duration_seconds against what was asked for: a Music 3 "
            "track that lands within 0.2 s of its audio_duration ceiling was "
            "cut off, one well short of it finished naturally. peak_dbfs is "
            "the level normalize_audio would be given; mean_dbfs below -40 "
            "on a track that should be full is a near-silent render."
        )
    return body
```

`dw_mcp/server.py` `get_gallery_metadata` docstring: append "For audio and video the `media` block carries duration, sample rate, channels, fps, size and level - the checks an agent that cannot listen makes on a deliverable."

`plugins/dw/skills/minimax-music3/SKILL.md`: where it tells the agent to use `get_gallery_metadata` to check duration, state the rule: "`media.duration_seconds` within 0.2 s of `audio_duration` means the ceiling cut the track; raise the ceiling and rerun. Well short of it means the song finished." `tests/test_plugin_skills.py` pins numbers to diffusers symbols - 0.2 s is a measurement, not a model constant, so it needs no pin; run the test to confirm.

`docs/MCP.md`: extend the `get_gallery_metadata` row with "and, for audio/video, a `media` block (duration, rate, channels, fps, size, peak/mean dBFS)".

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_mcp_catalog.py tests/test_mcp_server.py tests/test_plugin_skills.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add dw_mcp/catalog.py dw_mcp/server.py plugins/dw/skills docs/MCP.md tests/test_mcp_catalog.py
git commit -m "get_gallery_metadata: surface the media block and how to read it"
```

### Task 11: PR

- [ ] **Step 1:** `python -m pytest tests -q` - all pass.
- [ ] **Step 2:** Open a PR titled "Gallery metadata describes audio and video" citing field-report #6 and #6a. Note in the body that the per-second level envelope (#6's last ask) is deliberately not in this PR - see Deferred.

---

# Deferred (not planned here; each is its own brainstorm)

- **`loop_audio` task (#2b).** A room-tone bed under a cut is the only complete fix for the seam hole; needs `loop_audio(source, target_frames, fps, crossfade_ms)` composing with `mix_audio` + `pair_audio`. Small task, but a template (`dialogue-short`) should demonstrate it, and that wants a measured bed on real H3 output first.
- **`compose_text` task (#8).** Character bibles written once and assembled per shot as a task step (`previous_result:compose_shot_n`), instead of `{{var}}` interpolation, which the no-interpolation rule in `docs/WORKFLOW_GUIDE.md:285` exists to forbid. Needs a decision on the template syntax (positional parts vs named) - brainstorm first.
- **List-driven step count for generation templates (#7).** `music-video`/`dialogue-short` generate per shot; a `shots: [...]` variable that fans out *generation* steps is an engine feature (a `for_each` over a variable), not a template edit. `assemble-and-score` already takes a list after PR-1.
- **Per-second level envelope (#6).** `probe_media(path, envelope=True)` returning `[rms_dbfs per second]`; cheap once Task 8 exists, but it belongs behind an opt-in query parameter so the default metadata call stays small - the same lesson as #4.
