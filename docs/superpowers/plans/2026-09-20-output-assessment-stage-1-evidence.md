# Output Assessment, Stage 1 (Evidence) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an MCP agent hear a generated video's soundtrack (or an excerpt of any track) and see frames from a generated video - the perceptual half of #193 - without any new engine task.

**Architecture:** Two small PyAV-only modules beside `dw/media_info.py` (`dw/media_audio.py` extracts a WAV excerpt, `dw/media_frames.py` seeks out frames and composes tiles), two new gallery routes in `dw/server/app.py` in the existing `{name:path}/metadata` family, and on the MCP side `get_output_audio` moves to the new audio route and gains `start`/`duration`, and a new `get_output_frames` tool returns `ImageContent` tiles. Nothing touches the worker or the queue; everything runs in the server process like `metadata?envelope=true`.

**Tech Stack:** Python 3.11, FastAPI, PyAV 18 (`av`), numpy, Pillow, httpx MockTransport for MCP tests, pytest via `venv/bin/python -m pytest`.

**Spec:** `docs/proposals/output-assessment.md` - sections 2 and 5 are what this stage implements; section 3's `shots` (boundaries) is stage 2, so in this stage seam tiles take the boundaries as an explicit argument and nothing reads a manifest.

## Global Constraints

- Inline payloads are capped at `MAX_RETURNED_BYTES` (4 MB base64) in `dw_mcp/media.py`; a whole file over it is refused, never truncated, and the refusal names the excerpt path (spec section 2).
- An excerpt names itself: the answer carries `excerpt: {start, duration, of}`.
- New routes accept an `asset:` name the way `metadata` does (#127), and `workspace`.
- Every path reaches the disk through `_output_file` / `_asset_file` (`validate_path` inside) - never a bare `os.path.join` (CLAUDE.md, Security Rules; CodeQL models these as sanitizers).
- `dw_mcp/` must not import `dw.*` (a test guards that boundary).
- Tool count is pinned: `docs/MCP.md` line ~190 says how many tools; `tests/test_mcp_server.py::test_the_stated_tool_count_is_the_registered_one` compares it to `EXPECTED_TOOLS`.
- Commit messages: `feat(mcp): #193 - ...` / `feat(server): #193 - ...`, ending with the attribution line the session reminder gives.
- Run tests with `venv/bin/python -m pytest` (torch is present in that venv).
- Branch: `feat/193-evidence` off `develop`.

---

### Task 1: `dw/media_audio.py` - a WAV excerpt of any file's soundtrack

**Files:**
- Create: `dw/media_audio.py`
- Test: `tests/test_media_audio.py`

**Interfaces:**
- Consumes: `tests/test_media_info.py::write_mp4(path, frames, fps, width, height, with_audio)` and `write_wav(path, seconds, sample_rate, amplitude)` fixtures (8 kHz stereo, 220 Hz tone).
- Produces: `extract_audio(path, start=None, duration=None) -> (bytes, dict)` - 16-bit PCM WAV bytes and `{"sample_rate": int, "channels": int, "duration_seconds": float, "of_seconds": float, "start": float, "excerpt": bool}`. Raises `NoSoundtrack` (subclass of `ValueError`) when the file has no audio stream, and `ValueError` when `start` is past the end or `duration <= 0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_media_audio.py
"""extract_audio hands back a soundtrack, or a named slice of one, as WAV -
what get_output_audio needs for a muxed video and for a track too long to
send whole."""

import io
import wave

import numpy
import pytest

from dw.media_audio import NoSoundtrack, extract_audio
from tests.test_media_info import write_mp4, write_wav


def read_wav(data):
    with wave.open(io.BytesIO(data)) as handle:
        frames = handle.readframes(handle.getnframes())
        samples = numpy.frombuffer(frames, dtype="<i2").reshape(-1, handle.getnchannels())
        return handle.getframerate(), samples


def test_a_video_soundtrack_comes_back_whole_as_wav(tmp_path):
    write_mp4(tmp_path / "shot.mp4", frames=12, fps=6)  # 2 s of tone at 8 kHz

    data, info = extract_audio(str(tmp_path / "shot.mp4"))

    rate, samples = read_wav(data)
    assert rate == 8000
    assert samples.shape[1] == 2
    assert info["channels"] == 2
    assert info["sample_rate"] == 8000
    assert info["excerpt"] is False
    assert info["of_seconds"] == pytest.approx(2.0, abs=0.1)
    assert info["duration_seconds"] == pytest.approx(info["of_seconds"], abs=0.05)
    # the tone is there, not silence
    assert numpy.abs(samples).max() > 1000


def test_an_excerpt_is_cut_where_asked_and_says_so(tmp_path):
    write_wav(tmp_path / "score.wav", seconds=4.0)

    data, info = extract_audio(str(tmp_path / "score.wav"), start=1.0, duration=0.5)

    rate, samples = read_wav(data)
    assert samples.shape[0] == pytest.approx(0.5 * rate, abs=rate * 0.02)
    assert info["excerpt"] is True
    assert info["start"] == 1.0
    assert info["duration_seconds"] == pytest.approx(0.5, abs=0.02)
    assert info["of_seconds"] == pytest.approx(4.0, abs=0.05)


def test_an_excerpt_past_the_end_is_clipped_to_it(tmp_path):
    write_wav(tmp_path / "score.wav", seconds=2.0)

    data, info = extract_audio(str(tmp_path / "score.wav"), start=1.5, duration=5.0)

    rate, samples = read_wav(data)
    assert samples.shape[0] == pytest.approx(0.5 * rate, abs=rate * 0.02)
    assert info["duration_seconds"] == pytest.approx(0.5, abs=0.02)


def test_a_start_past_the_end_is_refused(tmp_path):
    write_wav(tmp_path / "score.wav", seconds=2.0)

    with pytest.raises(ValueError, match="past the end"):
        extract_audio(str(tmp_path / "score.wav"), start=3.0, duration=1.0)


def test_a_non_positive_duration_is_refused(tmp_path):
    write_wav(tmp_path / "score.wav", seconds=2.0)

    with pytest.raises(ValueError, match="duration"):
        extract_audio(str(tmp_path / "score.wav"), start=0.0, duration=0.0)


def test_a_silent_video_has_no_soundtrack(tmp_path):
    write_mp4(tmp_path / "mute.mp4", frames=6, fps=6, with_audio=False)

    with pytest.raises(NoSoundtrack):
        extract_audio(str(tmp_path / "mute.mp4"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_media_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dw.media_audio'`

- [ ] **Step 3: Write the module**

```python
# dw/media_audio.py
"""A soundtrack, or a named slice of one, as WAV bytes - the form
get_output_audio hands an agent for a muxed video (whose container it
cannot play) or for a track too long to send whole (#193).

PyAV only, like media_info: this runs in the server process, where a
request must not pull in torch or materialise frames.
"""

import io
import logging
import wave

import av
import numpy
from av.audio.resampler import AudioResampler

logger = logging.getLogger("dw")


class NoSoundtrack(ValueError):
    """The file has no audio stream to extract."""


def extract_audio(path, start=None, duration=None):
    """The soundtrack of `path` as 16-bit PCM WAV bytes, plus what was cut.

    With `start` and `duration` (seconds) only that slice is decoded and
    returned; the info dict says so (`excerpt: True`) and carries the whole
    track's length as `of_seconds`, so a slice always names itself. A
    slice that runs past the end is clipped to it; a `start` past the end
    is refused, since an empty answer would read as a silent track.

    Returns:
        (wav_bytes, info) with info = {sample_rate, channels,
        duration_seconds, of_seconds, start, excerpt}
    """
    excerpt = start is not None or duration is not None
    start = float(start or 0.0)
    if excerpt and (duration is None or float(duration) <= 0):
        raise ValueError("An excerpt needs a duration above zero")
    if start < 0:
        raise ValueError("An excerpt cannot start before zero")

    with av.open(path) as container:
        if not container.streams.audio:
            raise NoSoundtrack(f"{path} has no soundtrack")
        stream = container.streams.audio[0]
        total = (
            float(container.duration / av.time_base)
            if container.duration is not None
            else None
        )
        if total is not None and start >= total:
            raise ValueError(
                f"start {start:.2f}s is past the end of a {total:.2f}s track"
            )
        stop = start + float(duration) if excerpt else None
        if stop is not None and total is not None:
            stop = min(stop, total)

        rate = int(stream.rate)
        channels = int(stream.channels)
        layout = "stereo" if channels == 2 else ("mono" if channels == 1 else stream.layout.name)
        resampler = AudioResampler(format="s16", layout=layout, rate=rate)

        if start > 0:
            # Seek to the keyframe at or before `start`; the frames decoded
            # before `start` are then dropped sample-accurately below
            container.seek(int(start / stream.time_base), stream=stream, backward=True)

        pieces = []
        seen = 0  # samples of the track before the current frame
        started = False
        for frame in container.decode(stream):
            frame_start = (
                float(frame.pts * stream.time_base) if frame.pts is not None else seen / rate
            )
            for chunk in resampler.resample(frame):
                samples = chunk.to_ndarray()  # (1, samples * channels) packed s16
                samples = samples.reshape(-1, channels)
                chunk_start = frame_start
                chunk_end = chunk_start + samples.shape[0] / rate
                if chunk_end <= start:
                    seen += samples.shape[0]
                    continue
                if not started and chunk_start < start:
                    samples = samples[int((start - chunk_start) * rate) :]
                    chunk_start = start
                started = True
                if stop is not None and chunk_end > stop:
                    samples = samples[: max(0, int((stop - chunk_start) * rate))]
                pieces.append(samples)
                if stop is not None and chunk_end >= stop:
                    break
        # flush the resampler
        if stop is None:
            for chunk in resampler.resample(None):
                pieces.append(chunk.to_ndarray().reshape(-1, channels))

    pcm = numpy.concatenate(pieces) if pieces else numpy.zeros((0, channels), "<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "w") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.astype("<i2").tobytes())

    returned = pcm.shape[0] / rate
    return buffer.getvalue(), {
        "sample_rate": rate,
        "channels": channels,
        "duration_seconds": returned,
        "of_seconds": total if total is not None else returned,
        "start": start,
        "excerpt": excerpt,
    }
```

- [ ] **Step 4: Run the tests until they pass**

Run: `venv/bin/python -m pytest tests/test_media_audio.py -v`
Expected: PASS (6 tests). If the AAC fixture's `of_seconds` is a hair over 2.0 from codec padding, the `abs=0.1` tolerance covers it; do not loosen further - a wide tolerance would hide a broken cut.

- [ ] **Step 5: Commit**

```bash
git add dw/media_audio.py tests/test_media_audio.py
git commit -m "feat(server): #193 - extract_audio, a WAV excerpt of any file's soundtrack"
```

---

### Task 2: `GET /api/gallery/{name}/audio` - the soundtrack route

**Files:**
- Modify: `dw/server/app.py` (after `gallery_metadata`, ~line 2810; import beside `from ..media_info import probe_media` at line 74)
- Test: `tests/test_server.py` (beside `test_gallery_metadata_describes_audio_and_video`, ~line 1250, and the asset tests ~line 4534)

**Interfaces:**
- Consumes: `extract_audio`, `NoSoundtrack` from Task 1; `_output_file(name, root)`, `_asset_file(reference, ws)`, `is_asset_reference`, `MEDIA_KINDS`, `selected_workspace` already in `app.py`.
- Produces: the route. Response is `audio/wav` bytes with headers `X-DW-Duration` (the whole track, seconds), and when an excerpt was asked for `X-DW-Excerpt-Start`, `X-DW-Excerpt-Duration`. An audio-only file asked for whole is served as its own bytes with its own content type (no transcode) and only `X-DW-Duration`. 404 when the file has no soundtrack; 400 when `start`/`duration` are bad.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_server.py - add near test_gallery_metadata_describes_audio_and_video

def test_gallery_audio_extracts_a_videos_soundtrack(server, tmp_path):
    """get_output_audio refused video/mp4 outright, so a generated clip's
    soundtrack could only be heard by fetching the file and demuxing it by
    hand (#193). The route hands the track back as WAV."""
    import io
    import wave
    from tests.test_media_info import write_mp4

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        write_mp4(outputs / "shot-gen.0-0.0.mp4", frames=12, fps=6)

        response = client.get("/api/gallery/shot-gen.0-0.0.mp4/audio")

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert float(response.headers["x-dw-duration"]) == pytest.approx(2.0, abs=0.1)
        assert "x-dw-excerpt-start" not in response.headers
        with wave.open(io.BytesIO(response.content)) as handle:
            assert handle.getframerate() == 8000
            assert handle.getnchannels() == 2


def test_gallery_audio_serves_an_audio_file_as_itself_when_asked_whole(server, tmp_path):
    from tests.test_media_info import write_wav

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        write_wav(outputs / "score-gen.0-0.0.wav", seconds=1.0)
        raw = (outputs / "score-gen.0-0.0.wav").read_bytes()

        response = client.get("/api/gallery/score-gen.0-0.0.wav/audio")

        assert response.status_code == 200
        assert response.content == raw
        assert float(response.headers["x-dw-duration"]) == pytest.approx(1.0, abs=0.05)


def test_gallery_audio_cuts_an_excerpt_and_names_it(server, tmp_path):
    import io
    import wave
    from tests.test_media_info import write_wav

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        write_wav(outputs / "score-gen.0-0.0.wav", seconds=4.0)

        response = client.get(
            "/api/gallery/score-gen.0-0.0.wav/audio", params={"start": 1.0, "duration": 0.5}
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-dw-excerpt-start"] == "1.0"
        assert float(response.headers["x-dw-excerpt-duration"]) == pytest.approx(0.5, abs=0.02)
        assert float(response.headers["x-dw-duration"]) == pytest.approx(4.0, abs=0.05)
        with wave.open(io.BytesIO(response.content)) as handle:
            assert handle.getnframes() == pytest.approx(4000, abs=100)


def test_gallery_audio_refuses_a_bad_excerpt_and_a_mute_file(server, tmp_path):
    from tests.test_media_info import write_mp4, write_wav

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        write_wav(outputs / "score-gen.0-0.0.wav", seconds=2.0)
        write_mp4(outputs / "mute-gen.0-0.0.mp4", frames=6, fps=6, with_audio=False)

        past = client.get(
            "/api/gallery/score-gen.0-0.0.wav/audio", params={"start": 5.0, "duration": 1.0}
        )
        assert past.status_code == 400
        assert "past the end" in past.json()["detail"]

        mute = client.get("/api/gallery/mute-gen.0-0.0.mp4/audio")
        assert mute.status_code == 404
        assert "soundtrack" in mute.json()["detail"]

        still = client.get("/api/gallery/missing.png/audio")
        assert still.status_code == 404


def test_gallery_audio_reads_an_asset_reference(asset_server, tmp_path):
    from tests.test_media_info import write_wav

    with asset_server(success_script) as client:
        write_wav(tmp_path / "assets" / "bed.wav", seconds=1.0)

        response = client.get(
            "/api/gallery/asset:bed.wav/audio", params={"start": 0.0, "duration": 0.25}
        )

        assert response.status_code == 200
        assert float(response.headers["x-dw-excerpt-duration"]) == pytest.approx(0.25, abs=0.02)
```

Check how `asset_server` lays out its assets directory before relying on `tmp_path / "assets"`: read `test_gallery_metadata_reads_an_asset_reference` (~line 4534) and copy its setup exactly.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_server.py -k gallery_audio -v`
Expected: FAIL with 404s (route does not exist; FastAPI answers "Not Found").

- [ ] **Step 3: Add the route**

Add the import at line 74:

```python
from ..media_info import probe_media
from ..media_audio import NoSoundtrack, extract_audio
```

Add after `gallery_metadata`:

```python
    @app.get("/api/gallery/{name:path}/audio")
    def gallery_audio(
        name: str,
        start: Optional[float] = None,
        duration: Optional[float] = None,
        ws: Workspace = Depends(selected_workspace),
    ):
        """The soundtrack of an output or asset, as WAV - a muxed video's
        track, which `get_output_audio` used to refuse outright, or an
        excerpt (`start` + `duration`, seconds) of a track too long to send
        whole (#193). An excerpt names itself in the response headers
        (`X-DW-Excerpt-Start`, `X-DW-Excerpt-Duration`) beside the whole
        track's `X-DW-Duration`, so a cut is never silent (#204).

        An audio-only file asked for whole is served as its own bytes in its
        own encoding - there is nothing to extract, and a transcode would
        change what the agent hears."""
        if is_asset_reference(name):
            path = _asset_file(name, ws)
        else:
            path = _output_file(name, ws.outputs)
        extension = os.path.splitext(path)[1].lower()
        kind = MEDIA_KINDS.get(extension)
        if kind not in ("audio", "video"):
            raise HTTPException(status_code=404, detail=f"{name} carries no soundtrack")

        excerpt = start is not None or duration is not None
        if kind == "audio" and not excerpt:
            media = probe_media(path) or {}
            headers = {"X-DW-Duration": str(media.get("duration_seconds", ""))}
            media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            return FileResponse(path, media_type=media_type, headers=headers)

        try:
            data, info = extract_audio(path, start=start, duration=duration)
        except NoSoundtrack:
            raise HTTPException(status_code=404, detail=f"{name} carries no soundtrack")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        headers = {"X-DW-Duration": str(info["of_seconds"])}
        if info["excerpt"]:
            headers["X-DW-Excerpt-Start"] = str(info["start"])
            headers["X-DW-Excerpt-Duration"] = str(info["duration_seconds"])
        return Response(content=data, media_type="audio/wav", headers=headers)
```

Add `import mimetypes` to the stdlib imports at the top of `app.py` (line 8 block) if it is not already there.

- [ ] **Step 4: Run the tests until they pass**

Run: `venv/bin/python -m pytest tests/test_server.py -k gallery_audio -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add dw/server/app.py tests/test_server.py
git commit -m "feat(server): #193 - GET /api/gallery/{name}/audio serves a soundtrack or an excerpt of one"
```

---

### Task 3: `get_output_audio` moves to the route and takes an excerpt

**Files:**
- Modify: `dw_mcp/client.py:177-197` (`get_bytes_if`)
- Modify: `dw_mcp/media.py:106-140` (`get_output_audio`)
- Modify: `dw_mcp/server.py:550-574` (the tool)
- Modify: `tests/test_mcp_server.py:359` (wiring row), `tests/test_mcp_media.py`
- Modify: `docs/MCP.md:237` (tool row) and `:456-458` (the "Images and audio only" bullet), `dw_mcp/CLAUDE.md` (the `get_output_audio` sentence)

**Interfaces:**
- Consumes: Task 2's route and headers.
- Produces: `DwClient.get_media_if(path, accept_content_type, workspace=None, params=None) -> (body | None, content_type, headers)`; `media.get_output_audio(client, name, start=None, duration=None, workspace=None) -> {name, data, mime_type, bytes, duration_seconds, excerpt: None | {start, duration, of}}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mcp_media.py - add

def serving_with_headers(content, content_type, headers):
    def handler(request):
        return httpx.Response(
            200, content=content, headers={"content-type": content_type, **headers}
        )

    return DwClient(transport=httpx.MockTransport(handler))


def test_audio_is_fetched_from_the_gallery_audio_route():
    seen = []

    def handler(request):
        seen.append((request.url.path, dict(request.url.params)))
        return httpx.Response(
            200,
            content=b"riff",
            headers={"content-type": "audio/wav", "x-dw-duration": "2.0"},
        )

    client = DwClient(transport=httpx.MockTransport(handler))
    result = get_output_audio(client, "run/shot.mp4")

    assert seen == [("/api/gallery/run%2Fshot.mp4/audio", {})]
    assert result["mime_type"] == "audio/wav"
    assert result["duration_seconds"] == 2.0
    assert result["excerpt"] is None


def test_an_excerpt_is_asked_for_and_reported():
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(
            200,
            content=b"riff",
            headers={
                "content-type": "audio/wav",
                "x-dw-duration": "240.0",
                "x-dw-excerpt-start": "10.0",
                "x-dw-excerpt-duration": "2.0",
            },
        )

    client = DwClient(transport=httpx.MockTransport(handler))
    result = get_output_audio(client, "cut.mp4", start=10.0, duration=2.0)

    assert seen == [{"start": "10.0", "duration": "2.0"}]
    assert result["excerpt"] == {"start": 10.0, "duration": 2.0, "of": 240.0}


def test_a_whole_track_over_budget_is_refused_and_told_to_excerpt():
    big = b"\0" * (MAX_RETURNED_BYTES * 3 // 4 + 1024)
    client = serving_with_headers(big, "audio/wav", {"x-dw-duration": "240.0"})

    with pytest.raises(DwApiError) as caught:
        get_output_audio(client, "cut.mp4")

    assert "start" in str(caught.value) and "duration" in str(caught.value)


def test_a_non_audio_answer_is_refused():
    client = serving(b"{}", "application/json")

    with pytest.raises(DwApiError, match="not audio"):
        get_output_audio(client, "thing.json")
```

Also update the existing `get_output_audio` tests in this file that assert the `/outputs/` path or the old refusal wording on `video/mp4` - read them (`grep -n get_output_audio tests/test_mcp_media.py`) and change the ones whose premise this task removes: a `video/mp4` body is no longer a refusal case at the client (the server now answers WAV), so delete that test or turn it into the "non audio answer" one above.

In `tests/test_mcp_server.py` change the wiring row:

```python
    ("get_output_audio", {"name": "out.wav"}, "GET", "/api/gallery/out.wav/audio"),
```

and the wiring handler in `test_each_tool_calls_its_endpoint` already answers `audio/wav` for paths ending `.wav` - it ends with `/audio` now, so change that branch to `if request.url.path.endswith("/audio"):`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_mcp_media.py tests/test_mcp_server.py -k "audio" -v`
Expected: FAIL - path is `/outputs/...`, `excerpt` key missing, `TypeError` on `start=`.

- [ ] **Step 3: Extend the client**

In `dw_mcp/client.py`, replace `get_bytes_if` with:

```python
    def get_media_if(self, path, accept_content_type, workspace=None, params=None):
        """Like `get_bytes`, but the body is only downloaded when
        `accept_content_type(content_type)` is true, and the response
        headers come back with it - a media route says what it cut in
        them.

        Headers arrive before the body over HTTP, so a rejection closes the
        connection having read nothing past them - useful for `/outputs`,
        where a rejected file (a video, say) can be arbitrarily large.
        Returns `(None, content_type, headers)` on rejection, `(body,
        content_type, headers)` on acceptance. An error status is still
        raised either way, since the body has to be read to report it.
        """
        kwargs = {"params": params} if params else {}
        response = self._stream_request("GET", path, workspace=workspace, **kwargs)
        try:
            content_type = response.headers.get("content-type", "")
            if response.status_code < 400 and not accept_content_type(content_type):
                return None, content_type, response.headers
            self._call_httpx(response.read, path)
            self._raise_for_status(response, path)
            return response.content, content_type, response.headers
        finally:
            response.close()

    def get_bytes_if(self, path, accept_content_type, workspace=None):
        """`get_media_if` without the headers, for the callers that only
        want the body."""
        body, content_type, _headers = self.get_media_if(
            path, accept_content_type, workspace=workspace
        )
        return body, content_type
```

Check `_scoped` merges `params` rather than replacing them (read `dw_mcp/client.py:250-270`); if it does `kwargs.setdefault("params", {})` then `.setdefault("workspace", ...)`, passing `params` works as-is. If it replaces, fix `_scoped` to merge.

- [ ] **Step 4: Rewrite the handler**

In `dw_mcp/media.py` replace `get_output_audio`:

```python
def get_output_audio(client, name, start=None, duration=None, workspace=None):
    """One soundtrack from the gallery as base64 WAV - an audio output, or
    the track muxed into a video (#193) - for a clip short enough to fit
    MAX_RETURNED_BYTES whole, or an excerpt of one that is not.

    Audio is not resized the way an image is - there is no downscale of a
    waveform that keeps it meaningful to listen to - so a whole clip over
    budget is refused rather than truncated (#204). The way to hear part of
    a long track is to *ask* for the part: `start` and `duration` in
    seconds, and the answer names what it cut in `excerpt`, so a slice is
    never mistaken for the whole."""

    def is_audio(content_type):
        return bool(content_type) and content_type.startswith("audio/")

    params = {}
    if start is not None:
        params["start"] = start
    if duration is not None:
        params["duration"] = duration
    body, content_type, headers = client.get_media_if(
        api_path("api", "gallery", name, "audio"),
        is_audio,
        workspace=workspace,
        params=params,
    )
    if body is None:
        raise DwApiError(
            f"{name} answered {content_type or 'no declared type'}, not audio - "
            "this tool returns a soundtrack only. Use get_output_image for "
            "an image, or get_gallery_metadata for other media."
        )

    base64_size = 4 * math.ceil(len(body) / 3)
    if base64_size > MAX_RETURNED_BYTES:
        raise DwApiError(
            f"{name} is {len(body)} bytes, which would be {base64_size} "
            f"bytes base64-encoded - over the {MAX_RETURNED_BYTES} byte "
            "limit for an inline clip. Ask for an excerpt with `start` and "
            "`duration` (seconds) - get_gallery_metadata's envelope says "
            "where to look - or use download_output for the whole file."
        )

    excerpt = None
    if "x-dw-excerpt-start" in headers:
        excerpt = {
            "start": float(headers["x-dw-excerpt-start"]),
            "duration": float(headers["x-dw-excerpt-duration"]),
            "of": _float_header(headers, "x-dw-duration"),
        }
    return {
        "name": name,
        "data": base64.b64encode(body).decode("ascii"),
        "mime_type": content_type,
        "bytes": len(body),
        "duration_seconds": _float_header(headers, "x-dw-duration"),
        "excerpt": excerpt,
    }


def _float_header(headers, key):
    value = headers.get(key)
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None
```

- [ ] **Step 5: Update the tool in `dw_mcp/server.py`**

```python
    def get_output_audio(
        name: str,
        start: float | None = None,
        duration: float | None = None,
        workspace: str | None = None,
    ) -> list[AudioContent | TextContent]:
        """Listen to a generated soundtrack, named as `list_gallery` or a
        job's manifest reports it - an audio output, or the track muxed
        into a video (the audio analogue of `get_output_image`). There is
        no downscale for audio, so a whole clip too large to fit inline is
        refused rather than cut; hear part of a long one by asking for the
        part - `start` and `duration` in seconds, around a seam or a
        moment `get_gallery_metadata`'s envelope located. The text part
        reports the whole track's length and, for an excerpt, exactly what
        was cut, so a slice is never mistaken for the whole. To *see* a
        video, `get_output_frames`.

        `workspace` names the workspace for this one call without
        switching the session to it - the same pin `run_workflow`
        takes (#99)."""
        result = media.get_output_audio(
            client, name, start=start, duration=duration, workspace=workspace
        )
        audio = AudioContent(
            type="audio", data=result["data"], mime_type=result["mime_type"]
        )
        lines = [f"name: {result['name']}", f"bytes: {result['bytes']}"]
        if result["duration_seconds"] is not None:
            lines.append(f"duration_seconds: {result['duration_seconds']}")
        if result["excerpt"]:
            e = result["excerpt"]
            lines.append(f"excerpt: {e['duration']}s from {e['start']}s of {e['of']}s")
        telemetry = TextContent(type="text", text="\n".join(lines))
        return [audio, telemetry]
```

- [ ] **Step 6: Run the tests until they pass**

Run: `venv/bin/python -m pytest tests/test_mcp_media.py tests/test_mcp_server.py tests/test_mcp_client.py -v`
Expected: PASS. `test_wrapper_handler_map_covers_every_defaulted_handler` still passes (`get_output_audio` is already in the map).

- [ ] **Step 7: Update the docs**

`docs/MCP.md` line 237 row:

```
| `get_output_audio(name, start=None, duration=None, workspace=None)` | `name`, `start`, `duration`, `workspace` | Listen to a generated soundtrack as base64 WAV - an audio output, or the track muxed into a video (#193). No downscale exists for audio, so a whole clip over the 4MB budget is refused rather than cut (#204); ask for the part instead with `start` and `duration` in seconds, and the text part names what was cut (`excerpt: 2.0s from 10.0s of 240.0s`) so a slice is never mistaken for the whole. `get_gallery_metadata`'s envelope says where in a track to look. `workspace` names the workspace for this one call without switching the session to it |
```

`docs/MCP.md` lines ~456-458: replace the "Images and audio only" bullet with:

```
- **Images, sound and frames.** `get_output_image` returns an image,
  `get_output_audio` a soundtrack (an audio file's, or the one muxed into a
  video) whole or as a named excerpt, and `get_output_frames` frames of a
  video as images - there is no video content type over MCP, so a video is
  seen as frames and heard as its track. `get_output_audio` refuses a whole
  clip whose base64 size would exceed the same 4MB budget; ask for an
  excerpt instead.
```

`dw_mcp/CLAUDE.md`: replace the sentence beginning "`get_output_audio` is `get_output_image`'s sibling for audio" through "no `VideoContent` type to return it as." with:

```
`get_output_audio` is `get_output_image`'s sibling for sound (`media.py`,
#204, #193): it reads `GET /api/gallery/{name}/audio`, which extracts a
video's muxed track as WAV and cuts an excerpt on `start`/`duration`; a
whole clip over the same 4MB budget is still refused rather than cut short,
and the refusal says to ask for an excerpt. Video has no MCP content type,
so `get_output_frames` returns frames of one as `ImageContent` - specific
moments, a contact sheet, or the frame pair either side of each seam.
```

- [ ] **Step 8: Run the doc and MCP suites, then commit**

Run: `venv/bin/python -m pytest tests/test_mcp_server.py tests/test_docs_links.py -q`
Expected: PASS.

```bash
git add dw_mcp/client.py dw_mcp/media.py dw_mcp/server.py dw_mcp/CLAUDE.md docs/MCP.md tests/test_mcp_media.py tests/test_mcp_server.py
git commit -m "feat(mcp): #193 - get_output_audio hears a video's soundtrack and takes an excerpt"
```

---

### Task 4: `dw/media_frames.py` - frames by moment, a contact sheet, seam tiles

**Files:**
- Create: `dw/media_frames.py`
- Test: `tests/test_media_frames.py`

**Interfaces:**
- Consumes: `dw.tasks.video_utils._grid_tile(frame, index, fps, tile_width, label)`, `_compose_grid(tiles, columns)`, `_default_columns(count)`, `_evenly_spaced_indices(total, count)` (all exist; `frame_grid` at `video_utils.py:177` is their caller). `video_utils` imports torch; that is fine in the server process, which already has it loaded.
- Produces:
  - `frames_at(path, moments) -> list[Tile]` where `moments` is a list of `float` seconds or `"frame:N"` strings and `Tile` is a `dict` `{"label": str, "frame": int, "seconds": float, "image": PIL.Image}`.
  - `contact_sheet(path, count, tile_width=320) -> Tile` (one tile whose `image` is the grid, `label` = `"contact sheet, N frames"`).
  - `seam_tiles(path, boundaries, names=None, tile_width=320) -> list[Tile]` where `boundaries` is a list of frame indexes at which each shot after the first *starts*, so seam *i* is between frame `boundaries[i]-1` and `boundaries[i]`; one side-by-side tile per seam, `label` = `"seam 1: <A> | <B>"` using `names` (defaults `"shot 1"`, `"shot 2"`, ...).
  - `video_shape(path) -> {"frame_count": int, "fps": float | None, "width": int, "height": int}`.
  - All raise `ValueError` for a moment past the end, a boundary outside `1..frame_count-1`, a file with no video stream.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_media_frames.py
"""Frames out of a file by seeking, not by decoding it whole - what
get_output_frames needs so an agent can look at a moment, a contact sheet
or the frame pair either side of a seam (#193)."""

import numpy
import pytest
from PIL import Image

from dw.media_frames import contact_sheet, frames_at, seam_tiles, video_shape


def write_ramp_mp4(path, frames=24, fps=6, width=32, height=16):
    """A clip whose frame N is a flat grey of value N*10, so a returned
    frame says which one it is."""
    import av

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "0", "preset": "ultrafast"}  # lossless, so grey survives
    for index in range(frames):
        pixels = numpy.full((height, width, 3), index * 10, numpy.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def grey_of(image):
    return int(numpy.asarray(image.convert("L")).mean().round())


def test_video_shape_reads_the_container(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)

    shape = video_shape(str(tmp_path / "ramp.mp4"))

    assert shape == {"frame_count": 24, "fps": 6.0, "width": 32, "height": 16}


def test_frames_at_seeks_to_seconds_and_frame_indexes(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)

    tiles = frames_at(str(tmp_path / "ramp.mp4"), [0.0, 2.0, "frame:23"])

    assert [t["frame"] for t in tiles] == [0, 12, 23]
    assert [t["seconds"] for t in tiles] == pytest.approx([0.0, 2.0, 23 / 6])
    assert [grey_of(t["image"]) for t in tiles] == pytest.approx([0, 120, 230], abs=6)
    assert tiles[1]["label"] == "00:02.0 (frame 12)"


def test_a_moment_past_the_end_is_refused(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=6, fps=6)

    with pytest.raises(ValueError, match="past the end"):
        frames_at(str(tmp_path / "ramp.mp4"), [5.0])
    with pytest.raises(ValueError, match="past the end"):
        frames_at(str(tmp_path / "ramp.mp4"), ["frame:6"])


def test_a_contact_sheet_tiles_evenly_spaced_frames(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)

    tile = contact_sheet(str(tmp_path / "ramp.mp4"), count=4, tile_width=32)

    assert tile["label"] == "contact sheet, 4 frames"
    assert tile["image"].width == 32 * 2  # 4 tiles, two columns
    assert tile["image"].height == 16 * 2
    # first tile is frame 0, last is frame 23
    first = tile["image"].crop((0, 0, 32, 16))
    last = tile["image"].crop((32, 16, 64, 32))
    assert grey_of(first) < 20
    assert grey_of(last) > 200


def test_seam_tiles_pair_the_frames_either_side_of_each_boundary(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)

    tiles = seam_tiles(
        str(tmp_path / "ramp.mp4"),
        boundaries=[8, 16],
        names=["shot@a", "shot@b", "shot@c"],
        tile_width=32,
    )

    assert [t["label"] for t in tiles] == ["seam 1: shot@a | shot@b", "seam 2: shot@b | shot@c"]
    assert [t["frame"] for t in tiles] == [8, 16]
    image = tiles[0]["image"]
    assert image.width == 64 and image.height == 16
    assert grey_of(image.crop((0, 0, 32, 16))) == pytest.approx(70, abs=6)  # frame 7
    assert grey_of(image.crop((32, 0, 64, 16))) == pytest.approx(80, abs=6)  # frame 8


def test_a_boundary_outside_the_clip_is_refused(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=6, fps=6)

    with pytest.raises(ValueError, match="boundary"):
        seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=[0])
    with pytest.raises(ValueError, match="boundary"):
        seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=[6])
```

Note: the label tests burn text into the tile via `_grid_tile(label=True)`; `grey_of` tolerances of ±6 absorb the small label. If a label pushes a mean past the tolerance at 32x16, pass `label=False` inside `seam_tiles`/`frames_at` for the *sub*-tiles and put the label only in the returned dict - the server route burns nothing, the MCP telemetry text carries the label. Prefer that: the spec's "names burned in" is satisfied by a caption strip, which is Task 5's concern, not this module's.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_media_frames.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dw.media_frames'`

- [ ] **Step 3: Write the module**

```python
# dw/media_frames.py
"""Frames out of a video file by seeking to them - a moment, an evenly
spaced contact sheet, or the frame pair either side of a seam - without
decoding the clip whole. The server process runs this per request; a
four-minute 1080p clip materialised as PIL frames is tens of gigabytes,
so nothing here ever holds more than the frames it returns (#193).
"""

import logging

import av
from PIL import Image

from .tasks.video_utils import (
    _compose_grid,
    _default_columns,
    _evenly_spaced_indices,
    _format_timestamp,
)

logger = logging.getLogger("dw")


def video_shape(path):
    """Frame count, fps and size, from the container's own headers where
    they are written and by counting otherwise."""
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"{path} has no video stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        count = int(stream.frames) if stream.frames else None
        if count is None:
            count = sum(1 for _ in container.decode(stream))
        return {
            "frame_count": count,
            "fps": fps,
            "width": int(stream.width),
            "height": int(stream.height),
        }


def frames_at(path, moments):
    """One tile per moment - a float in seconds or "frame:N" - in the order
    asked for. Each tile is {label, frame, seconds, image}."""
    shape = video_shape(path)
    indexes = [_moment_to_index(moment, shape) for moment in moments]
    images = _read_frames(path, indexes)
    return [_tile(index, images[index], shape) for index in indexes]


def contact_sheet(path, count, tile_width=320):
    """N evenly spaced frames (first and last included) tiled into one
    image - frame_grid without a workflow."""
    if int(count) < 1:
        raise ValueError("count must be at least 1")
    shape = video_shape(path)
    count = min(int(count), shape["frame_count"])
    indexes = _evenly_spaced_indices(shape["frame_count"], count)
    images = _read_frames(path, indexes)
    tiles = [_fit_width(images[index], tile_width) for index in indexes]
    grid = _compose_grid(tiles, _default_columns(len(tiles)))
    return {
        "label": f"contact sheet, {len(tiles)} frames",
        "frame": indexes[0],
        "seconds": _seconds(indexes[0], shape),
        "image": grid,
        "frames": list(indexes),
    }


def seam_tiles(path, boundaries, names=None, tile_width=320):
    """For each boundary (the frame index a shot *starts* at), the last
    frame before it and the first frame at it, side by side - the seam and
    continuity evidence in one image. Seam i sits between shot i and shot
    i+1; `names` names the shots, "shot 1".. by default."""
    shape = video_shape(path)
    total = shape["frame_count"]
    for boundary in boundaries:
        if not 1 <= int(boundary) <= total - 1:
            raise ValueError(
                f"boundary {boundary} is not inside the clip (1..{total - 1})"
            )
    boundaries = [int(b) for b in boundaries]
    names = list(names or [f"shot {n + 1}" for n in range(len(boundaries) + 1)])
    if len(names) != len(boundaries) + 1:
        raise ValueError(
            f"{len(boundaries)} boundaries make {len(boundaries) + 1} shots, "
            f"but {len(names)} names were given"
        )
    wanted = sorted({b - 1 for b in boundaries} | set(boundaries))
    images = _read_frames(path, wanted)
    tiles = []
    for seam, boundary in enumerate(boundaries):
        before = _fit_width(images[boundary - 1], tile_width)
        after = _fit_width(images[boundary], tile_width)
        pair = _compose_grid([before, after], 2)
        tiles.append(
            {
                "label": f"seam {seam + 1}: {names[seam]} | {names[seam + 1]}",
                "frame": boundary,
                "seconds": _seconds(boundary, shape),
                "image": pair,
            }
        )
    return tiles


def _moment_to_index(moment, shape):
    total = shape["frame_count"]
    if isinstance(moment, str) and moment.startswith("frame:"):
        index = int(moment[len("frame:") :])
    else:
        fps = shape["fps"]
        if fps is None:
            raise ValueError("this clip has no frame rate, so name a frame: 'frame:N'")
        index = int(round(float(moment) * fps))
    if index < 0:
        index += total
    if not 0 <= index < total:
        raise ValueError(f"{moment!r} is past the end of a {total}-frame clip")
    return index


def _seconds(index, shape):
    return index / shape["fps"] if shape["fps"] else float(index)


def _tile(index, image, shape):
    fps = shape["fps"]
    stamp = _format_timestamp(index, fps) if fps else f"#{index}"
    return {
        "label": f"{stamp} (frame {index})",
        "frame": index,
        "seconds": _seconds(index, shape),
        "image": image,
    }


def _fit_width(image, tile_width):
    height = max(1, round(image.height * tile_width / image.width))
    return image.resize((tile_width, height), Image.LANCZOS).convert("RGB")


def _read_frames(path, indexes):
    """The frames at these indexes, as {index: PIL image}, in one forward
    pass that seeks to the keyframe before each wanted frame rather than
    decoding from the top. Decodes are dropped as soon as they are past."""
    wanted = sorted(set(int(i) for i in indexes))
    found = {}
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        position = 0  # index of the next frame decode() will yield
        for target in wanted:
            if target < position or target - position > 2 * (int(fps) if fps else 24):
                # seek back or a long way forward: land on the keyframe at
                # or before the target, then read up to it
                seconds = target / fps if fps else 0.0
                container.seek(
                    int(seconds / stream.time_base), stream=stream, backward=True
                )
                position = None
            for frame in container.decode(stream):
                if position is None:
                    # first frame after a seek says where we landed
                    position = (
                        int(round(float(frame.pts * stream.time_base) * fps))
                        if fps and frame.pts is not None
                        else 0
                    )
                if position == target:
                    found[target] = frame.to_image()
                    position += 1
                    break
                position += 1
        missing = [i for i in wanted if i not in found]
        if missing:
            raise ValueError(f"could not decode frame(s) {missing} of {path}")
    return found
```

The seek-then-count logic assumes constant frame rate, which every file this engine writes has (`encode_video` / `export_to_video` write a fixed `fps`). If the ramp test's returned greys are off by one frame, the keyframe landing computed `position` wrong - fall back to `position = 0` and `container.seek(0)` for that file (correct, slower) rather than loosen the test.

- [ ] **Step 4: Run the tests until they pass**

Run: `venv/bin/python -m pytest tests/test_media_frames.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add dw/media_frames.py tests/test_media_frames.py
git commit -m "feat(server): #193 - media_frames seeks out moments, a contact sheet and seam pairs"
```

---

### Task 5: `GET /api/gallery/{name}/frames` - the frames route

**Files:**
- Modify: `dw/server/app.py` (after `gallery_audio` from Task 2; import beside `extract_audio`)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: Task 4's `frames_at`, `contact_sheet`, `seam_tiles`, `video_shape`.
- Produces: the route. Query: exactly one of `at` (repeatable: `?at=1.5&at=frame:12`), `count` (int), `seams` (`true` or a comma list of 1-based seam numbers); `boundaries` (comma list of frame indexes - required with `seams` in this stage; stage 2 fills it from the manifest when absent); `names` (comma list); `max_dimension` (int, default 512, the longest side of each returned tile). Answer:

```json
{"name": "...", "frame_count": 120, "fps": 24.0, "width": 960, "height": 544,
 "tiles": [{"label": "00:02.0 (frame 48)", "frame": 48, "seconds": 2.0,
            "data": "<base64 png>", "mime_type": "image/png", "width": 512, "height": 290}]}
```

400 when zero or more than one selector is given, when `seams` has no `boundaries`, or when a moment/boundary is out of range; 404 when the file is not a video.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_server.py - add after the gallery_audio tests

def _png_of(tile):
    import base64
    import io
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(tile["data"])))


def test_gallery_frames_returns_the_moments_asked_for(server, tmp_path):
    """No tool returned a frame of a video, so judging a clip meant handing
    the user the file (#193, #245). The route seeks out the moments named."""
    from tests.test_media_frames import write_ramp_mp4

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        write_ramp_mp4(outputs / "shot-gen.0-0.0.mp4", frames=24, fps=6, width=64, height=32)

        response = client.get(
            "/api/gallery/shot-gen.0-0.0.mp4/frames",
            params=[("at", "0.0"), ("at", "frame:12"), ("max_dimension", "32")],
        )

        assert response.status_code == 200
        body = response.json()
        assert body["frame_count"] == 24 and body["fps"] == 6.0
        assert [t["frame"] for t in body["tiles"]] == [0, 12]
        assert body["tiles"][0]["mime_type"] == "image/png"
        assert body["tiles"][0]["width"] == 32  # downscaled to max_dimension
        assert _png_of(body["tiles"][0]).size == (32, 16)


def test_gallery_frames_makes_a_contact_sheet(server, tmp_path):
    from tests.test_media_frames import write_ramp_mp4

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        write_ramp_mp4(outputs / "shot-gen.0-0.0.mp4", frames=24, fps=6)

        response = client.get(
            "/api/gallery/shot-gen.0-0.0.mp4/frames", params={"count": 4}
        )

        assert response.status_code == 200
        tiles = response.json()["tiles"]
        assert len(tiles) == 1
        assert tiles[0]["label"] == "contact sheet, 4 frames"
        assert tiles[0]["frames"] == [0, 7, 15, 23]


def test_gallery_frames_pairs_the_frames_at_each_seam(server, tmp_path):
    from tests.test_media_frames import write_ramp_mp4

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        write_ramp_mp4(outputs / "cut-gen.0-0.0.mp4", frames=24, fps=6)

        response = client.get(
            "/api/gallery/cut-gen.0-0.0.mp4/frames",
            params={"seams": "true", "boundaries": "8,16", "names": "a,b,c"},
        )

        assert response.status_code == 200
        tiles = response.json()["tiles"]
        assert [t["label"] for t in tiles] == ["seam 1: a | b", "seam 2: b | c"]

        second = client.get(
            "/api/gallery/cut-gen.0-0.0.mp4/frames",
            params={"seams": "2", "boundaries": "8,16"},
        )
        assert [t["label"] for t in second.json()["tiles"]] == ["seam 2: shot 2 | shot 3"]


def test_gallery_frames_refuses_bad_selectors(server, tmp_path):
    from tests.test_media_frames import write_ramp_mp4
    from tests.test_media_info import write_wav

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        write_ramp_mp4(outputs / "shot-gen.0-0.0.mp4", frames=6, fps=6)
        write_wav(outputs / "score-gen.0-0.0.wav")

        none = client.get("/api/gallery/shot-gen.0-0.0.mp4/frames")
        assert none.status_code == 400 and "one of" in none.json()["detail"]

        two = client.get(
            "/api/gallery/shot-gen.0-0.0.mp4/frames", params={"count": 2, "at": "0"}
        )
        assert two.status_code == 400

        no_boundaries = client.get(
            "/api/gallery/shot-gen.0-0.0.mp4/frames", params={"seams": "true"}
        )
        assert no_boundaries.status_code == 400
        assert "boundaries" in no_boundaries.json()["detail"]

        past = client.get(
            "/api/gallery/shot-gen.0-0.0.mp4/frames", params={"at": "9.0"}
        )
        assert past.status_code == 400 and "past the end" in past.json()["detail"]

        audio = client.get("/api/gallery/score-gen.0-0.0.wav/frames", params={"count": 1})
        assert audio.status_code == 404


def test_gallery_frames_reads_an_asset_reference(asset_server, tmp_path):
    from tests.test_media_frames import write_ramp_mp4

    with asset_server(success_script) as client:
        write_ramp_mp4(tmp_path / "assets" / "ref.mp4", frames=6, fps=6)

        response = client.get("/api/gallery/asset:ref.mp4/frames", params={"count": 2})

        assert response.status_code == 200
        assert response.json()["tiles"][0]["frames"] == [0, 5]
```

(Use the same `asset_server` setup as the Task 2 asset test.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_server.py -k gallery_frames -v`
Expected: FAIL with 404 Not Found from FastAPI.

- [ ] **Step 3: Add the route**

Import beside `extract_audio`:

```python
from ..media_frames import contact_sheet, frames_at, seam_tiles, video_shape
```

Add after `gallery_audio`:

```python
    FRAME_MIN_DIMENSION = 64

    @app.get("/api/gallery/{name:path}/frames")
    def gallery_frames(
        name: str,
        at: Optional[List[str]] = Query(None),
        count: Optional[int] = None,
        seams: Optional[str] = None,
        boundaries: Optional[str] = None,
        names: Optional[str] = None,
        max_dimension: int = 512,
        ws: Workspace = Depends(selected_workspace),
    ):
        """Frames of a video output or asset, as PNG tiles - the way an
        agent with no video content type sees what a run made (#193).
        Exactly one selector: `at` (repeatable; seconds, or "frame:N"),
        `count` (an evenly spaced contact sheet, `frame_grid` without a
        workflow), or `seams` ("true", or a comma list of 1-based seam
        numbers) for the last frame before and first frame after each
        boundary, side by side. `boundaries` is the comma list of frame
        indexes each shot after the first starts at, and `names` the
        shots' names; both are required with `seams` until a joined file
        carries its own (stage 2 of docs/proposals/output-assessment.md).
        Tiles are downscaled to `max_dimension` on their longest side."""
        if is_asset_reference(name):
            path = _asset_file(name, ws)
        else:
            path = _output_file(name, ws.outputs)
        if MEDIA_KINDS.get(os.path.splitext(path)[1].lower()) != "video":
            raise HTTPException(status_code=404, detail=f"{name} is not a video")

        chosen = [key for key, value in (("at", at), ("count", count), ("seams", seams)) if value]
        if len(chosen) != 1:
            raise HTTPException(
                status_code=400,
                detail="Pass exactly one of `at`, `count` or `seams`"
                + (f" - got {', '.join(chosen)}" if chosen else ""),
            )
        limit = max(FRAME_MIN_DIMENSION, int(max_dimension))

        try:
            if at:
                moments = [m if m.startswith("frame:") else float(m) for m in at]
                tiles = frames_at(path, moments)
            elif count:
                tiles = [contact_sheet(path, count, tile_width=limit)]
            else:
                if not boundaries:
                    raise HTTPException(
                        status_code=400,
                        detail="`seams` needs `boundaries`: the frame index each "
                        "shot after the first starts at, comma-separated - this "
                        "file carries none of its own",
                    )
                starts = [int(b) for b in boundaries.split(",") if b.strip()]
                shot_names = [n.strip() for n in names.split(",")] if names else None
                tiles = seam_tiles(path, starts, names=shot_names, tile_width=limit)
                if seams.lower() != "true":
                    wanted = {int(s) for s in seams.split(",") if s.strip()}
                    tiles = [t for i, t in enumerate(tiles, start=1) if i in wanted]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        shape = video_shape(path)
        return {
            "name": name,
            **shape,
            "tiles": [_encoded_tile(tile, limit) for tile in tiles],
        }

    def _encoded_tile(tile, limit):
        image = tile["image"]
        longest = max(image.width, image.height)
        if longest > limit:
            scale = limit / longest
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = {
            key: value for key, value in tile.items() if key != "image"
        }
        encoded.update(
            {
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "mime_type": "image/png",
                "width": image.width,
                "height": image.height,
            }
        )
        return encoded
```

Add `import base64` to the stdlib imports if absent. The `contact_sheet` and `seam_tiles` tile widths are per sub-tile; the composed grid can exceed `limit` on its longest side, so `_encoded_tile` fits the *composite* to `limit` - that is the behaviour the `max_dimension` test asserts.

- [ ] **Step 4: Run the tests until they pass**

Run: `venv/bin/python -m pytest tests/test_server.py -k "gallery_frames or gallery_audio" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/server/app.py tests/test_server.py
git commit -m "feat(server): #193 - GET /api/gallery/{name}/frames returns moments, a contact sheet or seam pairs"
```

---

### Task 6: `get_output_frames` MCP tool

**Files:**
- Modify: `dw_mcp/media.py` (new handler + a budget helper)
- Modify: `dw_mcp/server.py` (tool after `get_output_audio`; `get_output_image` docstring; the instructions block ~line 116 and ~143)
- Modify: `tests/test_mcp_server.py` (`EXPECTED_TOOLS`, `TOOL_WIRING`, `WRAPPER_HANDLER_MAP`), `tests/test_mcp_media.py`
- Modify: `docs/MCP.md` (tool count, table row, the numbered loop ~line 376)

**Interfaces:**
- Consumes: Task 5's route and JSON shape; `_encode_within_budget` / `MAX_RETURNED_BYTES` in `dw_mcp/media.py`.
- Produces: `media.get_output_frames(client, name, at=None, seams=None, count=None, boundaries=None, names=None, max_dimension=512, workspace=None) -> {name, frame_count, fps, tiles: [{label, frame, seconds, data, mime_type, width, height}], downscaled_to: int | None}`; the MCP tool returns `[ImageContent, ..., TextContent]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mcp_media.py - add

from dw_mcp.media import get_output_frames


def tile_json(width, height, label="00:00.0 (frame 0)", frame=0, seconds=0.0):
    return {
        "label": label,
        "frame": frame,
        "seconds": seconds,
        "data": base64.b64encode(png_bytes(width, height)).decode("ascii"),
        "mime_type": "image/png",
        "width": width,
        "height": height,
    }


def frames_server(tiles, seen=None):
    def handler(request):
        if seen is not None:
            seen.append((request.url.path, list(request.url.params.multi_items())))
        return httpx.Response(
            200, json={"name": "x.mp4", "frame_count": 24, "fps": 6.0, "width": 64, "height": 32, "tiles": tiles}
        )

    return DwClient(transport=httpx.MockTransport(handler))


def test_frames_are_asked_for_by_moment_and_come_back_labelled():
    seen = []
    client = frames_server([tile_json(64, 32), tile_json(64, 32, "00:02.0 (frame 12)", 12, 2.0)], seen)

    result = get_output_frames(client, "run/x.mp4", at=[0.0, "frame:12"])

    assert seen[0][0] == "/api/gallery/run%2Fx.mp4/frames"
    assert ("at", "0.0") in seen[0][1] and ("at", "frame:12") in seen[0][1]
    assert [t["label"] for t in result["tiles"]] == ["00:00.0 (frame 0)", "00:02.0 (frame 12)"]
    assert result["downscaled_to"] is None


def test_seams_send_boundaries_and_names():
    seen = []
    client = frames_server([tile_json(128, 32, "seam 1: a | b", 8, 1.33)], seen)

    get_output_frames(client, "cut.mp4", seams=[1], boundaries=[8, 16], names=["a", "b", "c"])

    params = dict(seen[0][1])
    assert params["seams"] == "1"
    assert params["boundaries"] == "8,16"
    assert params["names"] == "a,b,c"


def test_two_selectors_are_refused_before_any_request():
    client = frames_server([])

    with pytest.raises(DwApiError, match="one of"):
        get_output_frames(client, "x.mp4", at=[0.0], count=4)


def test_tiles_over_budget_are_shrunk_together_and_say_so():
    # three noisy 2048x1024 tiles: well over 4MB base64 between them
    tiles = []
    for n in range(3):
        tiles.append(
            {
                **tile_json(2048, 1024, frame=n, seconds=float(n)),
                "data": base64.b64encode(noise_png_bytes(2048, 1024, seed=n)).decode("ascii"),
            }
        )
    client = frames_server(tiles)

    result = get_output_frames(client, "x.mp4", at=[0, 1, 2], max_dimension=2048)

    total = sum(len(t["data"]) for t in result["tiles"])
    assert total <= MAX_RETURNED_BYTES
    assert len(result["tiles"]) == 3  # shrunk, not dropped
    assert result["downscaled_to"] is not None and result["downscaled_to"] < 2048
    assert all(decoded(t).width == result["tiles"][0]["width"] for t in result["tiles"])
```

In `tests/test_mcp_server.py`: add `"get_output_frames"` to `EXPECTED_TOOLS`; add the wiring row

```python
    ("get_output_frames", {"name": "out.mp4", "count": 2}, "GET", "/api/gallery/out.mp4/frames"),
```

and in `test_each_tool_calls_its_endpoint`'s handler add, before the `/outputs/` branch:

```python
        if request.url.path.endswith("/frames"):
            return httpx.Response(
                200,
                json={"name": "out.mp4", "frame_count": 2, "fps": 6.0, "width": 1, "height": 1,
                      "tiles": [{"label": "contact sheet, 2 frames", "frame": 0, "seconds": 0.0,
                                 "data": base64.b64encode(PNG_1X1).decode("ascii"),
                                 "mime_type": "image/png", "width": 1, "height": 1}]},
            )
```

and `"get_output_frames": (media, "get_output_frames")` to `WRAPPER_HANDLER_MAP`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_mcp_media.py tests/test_mcp_server.py -k "frames or wiring or tool_count" -v`
Expected: FAIL - `ImportError: cannot import name 'get_output_frames'`, wiring table mismatch, tool count mismatch.

- [ ] **Step 3: Write the handler**

In `dw_mcp/media.py`:

```python
def get_output_frames(
    client,
    name,
    at=None,
    seams=None,
    count=None,
    boundaries=None,
    names=None,
    max_dimension=512,
    workspace=None,
):
    """Frames of a generated video as images - the way to *see* a clip when
    there is no video content type to return it as (#193, #210). One
    selector per call: `at` (moments: seconds, or "frame:N"), `count` (an
    evenly spaced contact sheet) or `seams` (True, or seam numbers from 1:
    the last frame before and the first frame after each boundary, side by
    side). `boundaries` is the list of frame indexes each shot after the
    first starts at, `names` the shots' names - both needed with `seams`
    until a joined file carries its own.

    Every tile is fitted to `max_dimension`; when the whole answer would
    still exceed MAX_RETURNED_BYTES the tiles are shrunk *together* - the
    same dimension for all, halved until they fit - rather than any being
    dropped, and `downscaled_to` says what they were shrunk to. A seam
    pair at half size is still a seam pair; a seam pair missing is a
    different answer."""
    chosen = [key for key, value in (("at", at), ("count", count), ("seams", seams)) if value]
    if len(chosen) != 1:
        raise DwApiError(
            "Pass exactly one of `at`, `count` or `seams`"
            + (f" - got {', '.join(chosen)}" if chosen else "")
        )
    params = [("max_dimension", str(max(MIN_DIMENSION, int(max_dimension))))]
    if at:
        params += [("at", str(moment)) for moment in at]
    elif count:
        params.append(("count", str(int(count))))
    else:
        params.append(("seams", "true" if seams is True else ",".join(str(s) for s in seams)))
        if boundaries:
            params.append(("boundaries", ",".join(str(int(b)) for b in boundaries)))
        if names:
            params.append(("names", ",".join(names)))

    body = client.get_json(
        api_path("api", "gallery", name, "frames"), params=params, workspace=workspace
    )
    tiles, downscaled_to = _fit_tiles_within_budget(body.get("tiles", []))
    return {
        "name": name,
        "frame_count": body.get("frame_count"),
        "fps": body.get("fps"),
        "tiles": tiles,
        "downscaled_to": downscaled_to,
    }


def _fit_tiles_within_budget(tiles):
    """Shrink every tile by the same factor until their base64 sizes sum
    to MAX_RETURNED_BYTES or less. Returns (tiles, downscaled_to) with
    downscaled_to None when nothing had to shrink."""
    total = sum(len(tile["data"]) for tile in tiles)
    if total <= MAX_RETURNED_BYTES or not tiles:
        return tiles, None
    images = [Image.open(io.BytesIO(base64.b64decode(tile["data"]))) for tile in tiles]
    for image in images:
        image.load()
    limit = max(max(image.width, image.height) for image in images)
    while True:
        limit = max(MIN_DIMENSION, limit // 2)
        shrunk = []
        for tile, image in zip(tiles, images):
            sized = _fit(image, limit)
            buffer = io.BytesIO()
            sized.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            shrunk.append({**tile, "data": encoded, "width": sized.width, "height": sized.height})
        if sum(len(t["data"]) for t in shrunk) <= MAX_RETURNED_BYTES or limit <= MIN_DIMENSION:
            return shrunk, limit
        images = [Image.open(io.BytesIO(base64.b64decode(t["data"]))) for t in shrunk]
```

Check `client.get_json` accepts `params` (read `dw_mcp/client.py`'s `get_json`); it does for `get_gallery_metadata` (`params={"envelope": "true"}`), and a list of pairs is what httpx wants for a repeated `at` - confirm httpx accepts a list of tuples for `params` (it does).

- [ ] **Step 4: Register the tool**

In `dw_mcp/server.py` after `get_output_audio`:

```python
    def get_output_frames(
        name: str,
        at: list[str | float] | None = None,
        seams: bool | list[int] | None = None,
        count: int | None = None,
        boundaries: list[int] | None = None,
        names: list[str] | None = None,
        max_dimension: int = 512,
        workspace: str | None = None,
    ) -> list[ImageContent | TextContent]:
        """See a generated video, named as `list_gallery` or a job's manifest
        reports it: there is no video content type over MCP, so a clip is
        looked at as frames (and heard with `get_output_audio`). One
        selector per call - `count` for an evenly spaced contact sheet of
        the whole clip (start here: `count=12` shows a clip's shape in one
        image), `at` for specific moments (seconds, or "frame:N"), or
        `seams` (true, or seam numbers from 1) for the last frame before
        and the first frame after each join, side by side - the picture to
        check a cut or a continuity break against. `seams` needs
        `boundaries` (the frame index each shot after the first starts at;
        sum the shots' frame counts from `get_gallery_metadata`) and takes
        `names` for the shots. Each tile is fitted to `max_dimension`; the
        text part lists every tile's label, frame and time, and says if the
        set was shrunk to fit the 4MB budget. What you see here outranks any
        number `get_gallery_metadata` reports.

        `workspace` names the workspace for this one call without
        switching the session to it (#99)."""
        result = media.get_output_frames(
            client,
            name,
            at=at,
            seams=seams,
            count=count,
            boundaries=boundaries,
            names=names,
            max_dimension=max_dimension,
            workspace=workspace,
        )
        parts = [
            ImageContent(type="image", data=tile["data"], mime_type=tile["mime_type"])
            for tile in result["tiles"]
        ]
        lines = [
            f"name: {result['name']}",
            f"frame_count: {result['frame_count']}  fps: {result['fps']}",
        ]
        for tile in result["tiles"]:
            lines.append(
                f"- {tile['label']}  [{tile['width']}x{tile['height']}]"
            )
        if result["downscaled_to"]:
            lines.append(
                f"downscaled_to: {result['downscaled_to']} (every tile, to fit the inline budget)"
            )
        parts.append(TextContent(type="text", text="\n".join(lines)))
        return parts
```

Register it wherever the file lists tools for `mcp.tool()` (find how `get_output_audio` is registered - `grep -n "get_output_audio" dw_mcp/server.py` - and mirror it exactly).

Then edit `get_output_image`'s docstring: replace "Images only: audio is `get_output_audio`'s and video is refused, so inspect a video with `get_gallery_metadata` or hand the user the file." with "Images only: a soundtrack is `get_output_audio`'s and a video's frames are `get_output_frames`'s."

In the instructions block (~line 116) change "`get_output_image` to actually look at what was made and say whether it answers the request." to "`get_output_image` (a still), `get_output_frames` (a video, as a contact sheet or the frames either side of each seam) and `get_output_audio` (a soundtrack, or an excerpt of one) to actually look at and listen to what was made and say whether it answers the request."

- [ ] **Step 5: Update `docs/MCP.md`**

- Line ~190: bump the stated tool count by one (read the current number; `test_the_stated_tool_count_is_the_registered_one` checks it).
- Add a table row after `get_output_audio`'s:

```
| `get_output_frames(name, at=None, seams=None, count=None, boundaries=None, names=None, max_dimension=512, workspace=None)` | `name`, `at`, `seams`, `count`, `boundaries`, `names`, `max_dimension`, `workspace` | See a generated video as frames, since there is no video content type over MCP (#193). One selector per call: `count` for an evenly spaced contact sheet, `at` for moments (seconds or `"frame:N"`), `seams` (true, or seam numbers from 1) for the last frame before and first frame after each join side by side - with `boundaries`, the frame index each shot after the first starts at, and `names`. Tiles are fitted to `max_dimension` and, when the set would exceed the 4MB budget, shrunk together rather than dropped; the text part lists each tile and says so |
```

- Line ~376, the numbered loop: after "5. `get_output_image(name)` to look at a result image" add "6. `get_output_frames(name, count=12)` to look at a result video, and `get_output_audio(name, start, duration)` to hear it" and renumber what follows.

- [ ] **Step 6: Run the whole MCP and docs suites**

Run: `venv/bin/python -m pytest tests/test_mcp_media.py tests/test_mcp_server.py tests/test_mcp_client.py tests/test_docs_links.py tests/test_server_mcp.py -q`
Expected: PASS, including `test_the_wiring_table_covers_every_registered_tool`, `test_the_stated_tool_count_is_the_registered_one`, `test_wrapper_handler_map_covers_every_defaulted_handler`, and `TestNoEngineImport` (the `dw_mcp` no-`dw` import guard - `media.py` uses PIL only).

- [ ] **Step 7: Commit**

```bash
git add dw_mcp/media.py dw_mcp/server.py docs/MCP.md tests/test_mcp_media.py tests/test_mcp_server.py
git commit -m "feat(mcp): #193 - get_output_frames, a video seen as moments, a contact sheet or seam pairs"
```

---

### Task 7: Full suite, the mounted MCP surface, and the hand-off

**Files:**
- Modify: `CLAUDE.md` (the `MCP Server` paragraph is one line pointing at `dw_mcp/CLAUDE.md`; no change needed unless the count of "seven tools require acknowledged_cost" sentence there needs one - it does not, nothing here is gated)
- Modify: `docs/proposals/output-assessment.md` - flip the Status line's "No code changes yet" to "Stage 1 (evidence) landed on `develop` as <sha>".

- [ ] **Step 1: Run the full suite**

Run: `venv/bin/python -m pytest -q -x`
Expected: PASS. Two places may break that no task above named: (a) `tests/test_server_mcp.py` exercises the *mounted* MCP surface (`dw.serve --mcp`), which registers the same tools - a snapshot of tool names there gains `get_output_frames`; (b) any test asserting `get_output_audio`'s old `/outputs` refusal wording for `video/mp4`. Fix each by updating the expectation to the new behaviour, never by weakening the new code.

- [ ] **Step 2: Try it against a real file**

```bash
venv/bin/python -m dw.serve &   # from a checkout with an outputs/ holding one mp4
# in another shell, with the MCP server pointed at it:
#   get_output_frames(name="<some>.mp4", count=8)
#   get_output_audio(name="<some>.mp4", start=0, duration=2)
```

Confirm the frames arrive as images in the client and the audio part plays. Note the response time on a 300-frame clip; `frames_at` with three moments should be well under a second.

- [ ] **Step 3: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill: merge `feat/193-evidence` into `develop` with a merge commit (the repo's pattern is `Merge branch 'fix/...' into develop`), do not push or deploy without asking - lem deploys are a separate, explicit step in this repo (see memory: lem restart recipe).

- [ ] **Step 4: Note the release items**

Append to whatever release-note draft is open (or open `docs/RELEASING.md`'s "unreleased" section if it has one): `get_output_audio` now reads `GET /api/gallery/{name}/audio`, extracts a video's soundtrack, and takes `start`/`duration`; new tool `get_output_frames`; new routes `/api/gallery/{name}/audio` and `/frames`.

---

## Self-review

**Spec coverage (sections 2 and 5 only - this is stage 1):**
- `get_output_frames` with `at` / `seams` / `count`, same budget path, halve-before-drop, says which - Tasks 4, 5, 6. ✔
- `get_output_audio` mux-aware, moves to the gallery route, `start`/`duration`, `excerpt: {start, duration, of}`, whole-file refusal kept, audio-only file served as-is - Tasks 1, 2, 3. ✔
- Routes accept `asset:` and `workspace`; paths through `_output_file`/`_asset_file` - Tasks 2, 5. ✔
- Runs in the server process as a sync `def` - Tasks 2, 5. ✔
- `media.shots` in metadata, and `seams` reading boundaries from the manifest/file - **stage 2, deliberately not here**; the route and tool take `boundaries` explicitly and their docstrings say why.
- Tool descriptions and the instructions block name the new tools; `dw_mcp/CLAUDE.md` and `docs/MCP.md` updated - Tasks 3, 6. ✔
- `minimax-h3` step 4 rewrite - **stage 4**, per the spec's staging.

**Placeholder scan:** none - every step has its code or its exact edit.

**Type consistency:** `extract_audio -> (bytes, dict)` used by Task 2 as `data, info`; header names `X-DW-Duration` / `X-DW-Excerpt-Start` / `X-DW-Excerpt-Duration` match between Task 2 (set) and Task 3 (read, lower-cased by httpx); `get_media_if` returns a 3-tuple in Task 3 and `get_bytes_if` keeps its 2-tuple for `get_output_image`/`get_output_text`; `Tile` dict keys `label/frame/seconds/image` in Task 4 become `label/frame/seconds/data/mime_type/width/height` in Task 5 (`_encoded_tile` strips `image`, keeps `frames` on a contact sheet), and Task 6 reads exactly those; `boundaries` is a frame-index list everywhere.
