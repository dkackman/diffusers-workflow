# Output Assessment Stage 1 - Review Fixes and Improvements

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four correctness findings from the 2026-09-20 review of `feat/193-evidence`, then make the evidence tools sharper for an agent judging its own output: self-describing contact sheets, a per-seam difference number, sound around a moment, and skills that map each failure mode to the tool that shows it.

**Architecture:** All work stays on `feat/193-evidence`, on the three layers stage 1 built - `dw/media_frames.py` / `dw/media_audio.py` (decode by seeking), the `GET /api/gallery/{name}/frames|audio` routes in `dw/server/app.py`, and the `get_output_frames` / `get_output_audio` tools in `dw_mcp/media.py` + `dw_mcp/server.py`. Tasks 1-4 are pre-merge fixes (each one commit); tasks 5-9 are improvements that can land after the merge if the branch needs to go first. No new files.

**Tech Stack:** Python 3, PyAV (`av`), Pillow, FastAPI, the MCP Python SDK, pytest (`venv/bin/python -m pytest`).

**Spec:** `docs/proposals/output-assessment.md` (stage 1 = "Evidence"); the review findings are in this plan's task headers. The stage-1 plan this extends is `docs/superpowers/plans/2026-09-20-output-assessment-stage-1-evidence.md`.

## Global Constraints

- Work on branch `feat/193-evidence`; `git checkout feat/193-evidence` first. Never commit to `develop`.
- Tests run with `venv/bin/python -m pytest` (torch is in that venv; see the `local-test-venv` memory). Never ship tests to lem.
- The MCP tool surface must stay under `SURFACE_BUDGET = 13_800` tokens (`tests/test_mcp_server.py::test_the_tool_surface_fits_the_budget`). It was measured at 13_741 - about 59 tokens (~236 characters of description + JSON schema) of headroom. **Never raise the budget**; pay for new description text by cutting restated text, as the comment above the constant describes.
- Each plugin skill must stay under `SKILL_SIZE_LIMIT = 12 * 1024` bytes (`tests/test_plugin_skills.py`). `minimax-h3/SKILL.md` is at 12_276 bytes (12 free), `ltx-2.5/SKILL.md` at 12_209 (79 free). Every byte added is a byte cut. A phrase a test pins must sit on **one line** - a line wrap inside it fails the test.
- All filesystem access in routes goes through the existing `_output_file` / `_asset_file` helpers (CodeQL sanitizers). Nothing here opens a path any other way.
- Commit messages: `fix(server): #193 - ...` / `feat(mcp): #193 - ...` in the branch's existing style, ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Every route error is a `ValueError` in the module → 400 in the route (the existing `except ValueError` wrap). Do not add new exception types.

---

### Task 1: Cap the contact sheet and seam count; shrink frames as they are decoded

**Review finding:** `contact_sheet` clamps `count` only to `frame_count`, and `_read_frames` returns every wanted frame at full resolution before any is tiled. `count=2000` on a 1080p clip holds ~12 GB in the server process; `seams=true` with a long `boundaries` list does the same. `MAX_FRAME_MOMENTS` protects `at` only.

**Files:**
- Modify: `dw/media_frames.py` (`contact_sheet`, `seam_tiles`, `_read_frames`, module docstring)
- Test: `tests/test_media_frames.py`, `tests/test_server.py`

**Interfaces:**
- Produces: `MAX_CONTACT_SHEET_FRAMES = 64` and `MAX_SEAMS = 32` module constants in `dw/media_frames.py`; `_read_frames(path, indexes, fit=None)` where `fit` is a callable `(PIL.Image, index) -> PIL.Image` applied to each frame the moment it is decoded (the index is what Task 6 stamps on the tile).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_media_frames.py`:

```python
def test_a_contact_sheet_over_the_frame_cap_is_refused(tmp_path):
    from dw.media_frames import MAX_CONTACT_SHEET_FRAMES

    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)

    with pytest.raises(ValueError, match=f"{MAX_CONTACT_SHEET_FRAMES}"):
        contact_sheet(str(tmp_path / "ramp.mp4"), MAX_CONTACT_SHEET_FRAMES + 1)


def test_more_seams_than_the_cap_are_refused(tmp_path):
    from dw.media_frames import MAX_SEAMS

    write_ramp_mp4(tmp_path / "ramp.mp4", frames=MAX_SEAMS + 3, fps=6)
    boundaries = list(range(1, MAX_SEAMS + 2))  # MAX_SEAMS + 1 seams

    with pytest.raises(ValueError, match=f"{MAX_SEAMS}"):
        seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=boundaries)
    # a `wanted` subset under the cap is still served
    tiles = seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=boundaries, wanted={1, 2})
    assert len(tiles) == 2


def test_read_frames_fits_each_frame_as_it_is_decoded(tmp_path):
    from dw.media_frames import _read_frames

    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)
    seen = []

    def fit(image, index):
        seen.append((image.size, index))
        return image.resize((8, 4))

    found = _read_frames(str(tmp_path / "ramp.mp4"), [0, 5, 23], fit=fit)

    assert seen == [
        ((32, 16), 0),
        ((32, 16), 5),
        ((32, 16), 23),
    ]  # ran per frame, at source size
    assert all(image.size == (8, 4) for image in found.values())


def test_a_contact_sheet_never_holds_a_full_size_frame(tmp_path, monkeypatch):
    """The point of the cap and the fitter together: a 1080p clip's contact
    sheet is built from tiles, not from a list of 1080p images."""
    import dw.media_frames as module

    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6, width=64, height=32)
    sizes = []
    real_read = module._read_frames

    def spying_read(path, indexes, fit=None):
        found = real_read(path, indexes, fit=fit)
        sizes.extend(image.size for image in found.values())
        return found

    monkeypatch.setattr(module, "_read_frames", spying_read)

    contact_sheet(str(tmp_path / "ramp.mp4"), 4, tile_width=16)

    assert sizes and all(size == (16, 8) for size in sizes)
```

Append to `tests/test_server.py` after `test_gallery_frames_caps_the_number_of_moments`:

```python
def test_gallery_frames_caps_the_contact_sheet(server, tmp_path):
    from dw.media_frames import MAX_CONTACT_SHEET_FRAMES
    from tests.test_media_frames import write_ramp_mp4

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        write_ramp_mp4(outputs / "long.mp4", frames=MAX_CONTACT_SHEET_FRAMES + 2, fps=6)

        response = client.get(
            "/api/gallery/long.mp4/frames",
            params={"count": MAX_CONTACT_SHEET_FRAMES + 1},
        )

        assert response.status_code == 400
        assert str(MAX_CONTACT_SHEET_FRAMES) in response.json()["detail"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_media_frames.py tests/test_server.py -q -k "frame_cap or seams_than or fits_each or full_size or caps_the_contact" 2>&1 | tail -15`
Expected: 5 failures - `ImportError: cannot import name 'MAX_CONTACT_SHEET_FRAMES'`, `TypeError: _read_frames() got an unexpected keyword argument 'fit'`, and the seam cap not raised.

- [ ] **Step 3: Implement**

In `dw/media_frames.py`, add after `logger = ...`:

```python
# Each cell of a contact sheet and each side of a seam pair is a seek, a
# decode and a resize in the server process; a sheet past 64 cells is
# unreadable anyway, and `at` has its own cap in the route
# (MAX_FRAME_MOMENTS). Both are ValueErrors, so the route answers 400.
MAX_CONTACT_SHEET_FRAMES = 64
MAX_SEAMS = 32
```

In `contact_sheet`, replace everything from the `shape = ...` line to the `tiles = [...]` line:

```python
    if int(count) > MAX_CONTACT_SHEET_FRAMES:
        raise ValueError(
            f"count {count} is more than a contact sheet holds "
            f"({MAX_CONTACT_SHEET_FRAMES}); ask for a smaller one, or `at` for moments"
        )
    shape = shape if shape is not None else video_shape(path)
    count = min(int(count), shape["frame_count"])
    indexes = _evenly_spaced_indices(shape["frame_count"], count)
    images = _read_frames(
        path, indexes, fit=lambda image, _index: _fit_width(image, tile_width)
    )
    tiles = [images[index] for index in indexes]
```

In `seam_tiles`, after `chosen = [...]` is built and before `frame_indexes = ...`:

```python
    if len(chosen) > MAX_SEAMS:
        raise ValueError(
            f"{len(chosen)} seams is more than one call serves ({MAX_SEAMS}); "
            "name the seams wanted (`seams=1,2,...`)"
        )
```

and change the read + tile lines to fit as decoded:

```python
    images = _read_frames(
        path, frame_indexes, fit=lambda image, _index: _fit_width(image, tile_width)
    )
    tiles = []
    for seam, boundary in chosen:
        before = images[boundary - 1]
        after = images[boundary]
```

In `_read_frames`, change the signature to `def _read_frames(path, indexes, fit=None):`, add to its docstring `"`fit(image, index)`, when given, is applied to each frame as it is decoded, so a caller tiling many frames never holds one at source size."`, and change the found line to:

```python
                if position == target:
                    image = frame.to_image()
                    found[target] = fit(image, target) if fit is not None else image
```

Update the module docstring's last sentence to: `so nothing here ever holds more than the frames it returns, each already fitted to its tile where a caller asked for many (#193).`

- [ ] **Step 4: Run the tests to verify they pass, then the whole frames suite**

Run: `venv/bin/python -m pytest tests/test_media_frames.py tests/test_server.py -q -k "frames or seam" 2>&1 | tail -5`
Expected: all pass. The existing `test_a_sub_tile_is_never_upscaled_past_the_source` still passes because `_fit_width` is unchanged.

- [ ] **Step 5: Commit**

```bash
git add dw/media_frames.py tests/test_media_frames.py tests/test_server.py
git commit -m "fix(server): #193 - cap contact-sheet cells and seams, fit frames as they are decoded

A contact sheet or a seam set materialised every wanted frame at source
size before tiling; count=2000 on a 1080p clip was gigabytes in the
server process. Caps both, and _read_frames takes a fitter so a tile
is the only size a many-frame caller ever holds.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: An excerpt is cut correctly when frames carry no pts

**Review finding:** In `extract_audio`, `seen` advances only in the skip-before-`start` branch. For a stream whose frames have no `pts`, every kept frame gets the same `frame_start`, `chunk_end` never reaches `stop`, and the excerpt returns the whole remainder while its headers claim a cut.

**Files:**
- Modify: `dw/media_audio.py:130-160`
- Test: `tests/test_media_audio.py`

**Interfaces:**
- Consumes: `extract_audio(path, start=0.0, duration=None) -> (bytes, info)` as it exists.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_media_audio.py` (the file already imports `mock`, `numpy`, `pytest`, `write_wav`):

```python
def stripping_pts(real_open):
    """Wrap av.open so every decoded frame loses its pts - the shape of a
    raw stream, which the fallback clock has to handle."""

    class Container:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def decode(self, *args, **kwargs):
            for frame in self._inner.decode(*args, **kwargs):
                frame.pts = None
                yield frame

    return lambda *args, **kwargs: Container(real_open(*args, **kwargs))


def test_an_excerpt_from_the_top_is_cut_even_when_frames_carry_no_pts(tmp_path):
    import av

    write_wav(tmp_path / "score.wav", seconds=4.0)

    with mock.patch("dw.media_audio.av.open", side_effect=stripping_pts(av.open)):
        data, info = extract_audio(str(tmp_path / "score.wav"), start=0.0, duration=0.5)

    rate, samples = read_wav(data)
    assert samples.shape[0] == pytest.approx(0.5 * rate, abs=rate * 0.02)
    assert info["duration_seconds"] == pytest.approx(0.5, abs=0.02)


def test_an_excerpt_after_a_seek_is_cut_where_asked_when_frames_carry_no_pts(tmp_path):
    """Without pts a seek's landing is unknowable, so the reader starts
    over from the top and counts samples - slower, but the right audio."""
    import av

    write_wav(tmp_path / "score.wav", seconds=4.0)
    reference, _ = extract_audio(str(tmp_path / "score.wav"), start=1.0, duration=0.5)

    with mock.patch("dw.media_audio.av.open", side_effect=stripping_pts(av.open)):
        data, info = extract_audio(str(tmp_path / "score.wav"), start=1.0, duration=0.5)

    rate, samples = read_wav(data)
    _, expected = read_wav(reference)
    assert samples.shape[0] == pytest.approx(expected.shape[0], abs=rate * 0.02)
    assert info["start"] == 1.0
    # the same audio, not the first half second of the file
    n = min(len(samples), len(expected)) - 64
    assert numpy.abs(samples[:n].astype(int) - expected[:n].astype(int)).mean() < 200
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_media_audio.py -q -k "no_pts" 2>&1 | tail -8`
Expected: FAIL - the first returns ~4.0 s of samples instead of 0.5; the second returns audio from 0.0 s.

- [ ] **Step 3: Implement**

In `dw/media_audio.py`, replace the decode loop (from `pieces = []` through the inner `for chunk` loop's end) with:

```python
pieces = []
seen = 0  # samples decoded so far - the clock for a frame with no pts
started = False
done = False  # Break outer loop when stop time is reached
sought = start > 0
restarted = False
for frame in container.decode(stream):
    if done:
        break
    if frame.pts is None and sought and not restarted:
        # No pts means the seek's landing cannot be known, so the
        # sample count is the only clock there is - and it has to
        # count from the top. Start over once and read to `start`.
        container.seek(0, stream=stream, backward=True)
        resampler = AudioResampler(format="s16", layout=layout, rate=rate)
        restarted = True
        seen = 0
        continue
    frame_start = (
        float((frame.pts - anchor) * stream.time_base)
        if frame.pts is not None
        else seen / rate
    )
    for chunk in resampler.resample(frame):
        samples = chunk.to_ndarray()  # (1, samples * channels) packed s16
        samples = samples.reshape(-1, channels)
        chunk_start = frame_start
        chunk_end = chunk_start + samples.shape[0] / rate
        seen += samples.shape[0]
        frame_start = (
            chunk_end  # a frame yielding two chunks: the second follows the first
        )
        if chunk_end <= start:
            continue
        if not started and chunk_start < start:
            samples = samples[int((start - chunk_start) * rate) :]
            chunk_start = start
        started = True
        if stop is not None and chunk_end > stop:
            samples = samples[: max(0, int((stop - chunk_start) * rate))]
        pieces.append(samples)
        if stop is not None and chunk_end >= stop:
            done = True
            break
```

(`continue` inside the `if frame.pts is None ...` branch skips the frame decoded from the seek landing; the seek to 0 re-primes `container.decode`, which in PyAV continues from the new position on the next iteration of the same generator.)

- [ ] **Step 4: Run the audio suite**

Run: `venv/bin/python -m pytest tests/test_media_audio.py tests/test_server.py -q -k "audio" 2>&1 | tail -5`
Expected: all pass, including the existing `start_time` anchoring tests (`frame_start` for a pts-bearing frame is unchanged).

If the seek-to-0 inside an active `decode` generator does not resume in your PyAV (the two new tests fail with wrong audio while the rest pass), restructure to `break` out of the loop, set `restarted = True`, and wrap the whole loop in `while True:` that re-enters `container.decode(stream)` after the seek. Keep the `seen` accounting exactly as above.

- [ ] **Step 5: Commit**

```bash
git add dw/media_audio.py tests/test_media_audio.py
git commit -m "fix(server): #193 - extract_audio keeps its sample clock for frames with no pts

seen advanced only while skipping to start, so a pts-less stream's every
kept frame read as the same instant and the excerpt never stopped. Count
every chunk, and after a seek on such a stream start over from the top.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: A seek that overshoots its target restarts from the top instead of decoding to EOF

**Review finding:** In `_read_frames`, if the position estimated from the first post-seek frame is already past `target` (an off-rate or VFR file), `position == target` can never hit; the loop decodes the clip to its end and the miss surfaces as a 400 "could not decode frame(s)" that blames the caller.

**Files:**
- Modify: `dw/media_frames.py` (`_read_frames`)
- Test: `tests/test_media_frames.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_media_frames.py`:

```python
def test_a_seek_that_lands_past_its_target_recovers_from_the_top(tmp_path):
    """A VFR or off-rate file can make the seek land *after* the wanted
    frame. The reader must then read from the top rather than scan to EOF
    and report the frame as undecodable."""
    import av
    from unittest import mock

    write_shifted_ramp_mp4(tmp_path / "gop.mp4", frames=60, fps=6, offset=0)
    real_open = av.open

    class Overshooting:
        def __init__(self, inner):
            self._inner = inner
            self.seeks = []

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def seek(self, offset, **kwargs):
            self.seeks.append(offset)
            stream = kwargs["stream"]
            # first seek lands two keyframes late; a later seek is honest
            late = int(20 / 6 / stream.time_base) if len(self.seeks) == 1 else 0
            return self._inner.seek(offset + late, **kwargs)

    proxies = []

    def opening(*args, **kwargs):
        proxies.append(Overshooting(real_open(*args, **kwargs)))
        return proxies[-1]

    with mock.patch("dw.media_frames.av.open", side_effect=opening):
        tiles = frames_at(str(tmp_path / "gop.mp4"), ["frame:35"])

    assert grey_of(tiles[0]["image"]) == pytest.approx((35 % 25) * 10, abs=6)
    assert len(proxies[-1].seeks) == 2  # the overshoot, then the recovery
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_media_frames.py -q -k overshoot 2>&1 | tail -6`
Expected: FAIL with `ValueError: could not decode frame(s) [35]`.

- [ ] **Step 3: Implement**

In `_read_frames`, the per-target loop becomes:

```python
        for target in wanted:
            if target < position or target - position > 2 * (int(fps) if fps else 24):
                seconds = target / fps if fps else 0.0
                container.seek(
                    int(seconds / stream.time_base) + start_pts,
                    stream=stream,
                    backward=True,
                )
                position = None
            recovered = False
            for frame in container.decode(stream):
                if position is None:
                    # first frame after a seek says where we landed
                    position = (
                        int(round(float((frame.pts - start_pts) * stream.time_base) * fps))
                        if fps and frame.pts is not None
                        else 0
                    )
                    if position > target and not recovered:
                        # Landed past the target: the keyframe estimate was
                        # wrong for this file (off-rate or VFR). Reading on
                        # would scan to EOF and blame the caller; read from
                        # the top once instead, which is always correct.
                        container.seek(start_pts, stream=stream, backward=True)
                        position = None
                        recovered = True
                        continue
                if position == target:
                    image = frame.to_image()
                    found[target] = fit(image, target) if fit is not None else image
                    position += 1
                    break
                position += 1
```

- [ ] **Step 4: Run the frames suite**

Run: `venv/bin/python -m pytest tests/test_media_frames.py -q 2>&1 | tail -4`
Expected: all pass. `test_frames_at_is_correct_when_the_stream_has_a_non_zero_start` proves the recovery seek (to `start_pts`) is anchored the same way as the first.

- [ ] **Step 5: Commit**

```bash
git add dw/media_frames.py tests/test_media_frames.py
git commit -m "fix(server): #193 - a frame seek that overshoots reads from the top rather than to EOF

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `seams=,` is a 400, not an empty 200

**Review finding:** `seams=,` is truthy, parses to `wanted = set()`, and the route answers 200 with `tiles: []`.

**Files:**
- Modify: `dw/server/app.py:2976-2980`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Add beside `test_gallery_frames_refuses_bad_selectors`:

```python
def test_gallery_frames_refuses_an_empty_seam_list(server, tmp_path):
    from tests.test_media_frames import write_ramp_mp4

    with server(success_script) as client:
        write_ramp_mp4(tmp_path / "outputs" / "cut.mp4", frames=24, fps=6)

        response = client.get(
            "/api/gallery/cut.mp4/frames", params={"seams": ",", "boundaries": "8"}
        )

        assert response.status_code == 400
        assert "seams" in response.json()["detail"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_server.py -q -k empty_seam 2>&1 | tail -4`
Expected: FAIL - `assert 200 == 400`.

- [ ] **Step 3: Implement**

After `wanted = (...)` in `gallery_frames`:

```python
                if wanted is not None and not wanted:
                    raise HTTPException(
                        status_code=400,
                        detail="`seams` names no seam - pass `true` for every seam, "
                        "or seam numbers from 1",
                    )
```

- [ ] **Step 4: Run and commit**

Run: `venv/bin/python -m pytest tests/test_server.py -q -k gallery_frames 2>&1 | tail -3` - all pass.

```bash
git add dw/server/app.py tests/test_server.py
git commit -m "fix(server): #193 - an empty seams list is a 400

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**Checkpoint:** tasks 1-4 are the pre-merge set. Run `venv/bin/python -m pytest -q -x 2>&1 | tail -3` here and expect the full suite green before continuing or merging.

---

### Task 5: The frames tool says where `boundaries` come from

**Improvement:** `seams` needs `boundaries`, and nothing tells the agent where to get them until stage 2 puts `shots` on the artifact. Until then the boundaries of a hard-cut join are the running sum of each shot's frame count, which `get_gallery_metadata` reports for each shot's own file (the `intermediate/` members `shot@<name>`), or `num_frames` of each entry in the `shots` list of `get_job_workflow`'s realized workflow.

**Files:**
- Modify: `dw_mcp/server.py:597-604` (docstring), `dw_mcp/media.py:196-203` (docstring), `docs/MCP.md:238` (table row)
- Test: `tests/test_mcp_server.py` (budget test; a description pin)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcp_server.py` beside the existing `get_output_frames` description test at line ~399:

```python
@pytest.mark.asyncio
async def test_the_frames_tool_says_where_boundaries_come_from():
    """Until a joined file carries its own shots (stage 2), the agent has to
    derive seam boundaries; the tool has to say from what, or `seams` is a
    parameter nobody can fill in."""
    server = server_over(ok({}))
    tools = await tools_of(server)
    text = tools["get_output_frames"].description
    assert "get_gallery_metadata" in text
    assert "frame_count" in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_mcp_server.py -q -k "boundaries_come_from or fits_the_budget" 2>&1 | tail -4`
Expected: the new test FAILS; the budget test passes (baseline).

- [ ] **Step 3: Rewrite the docstring within budget**

Replace the `get_output_frames` docstring in `dw_mcp/server.py` with exactly:

```python
        """See a generated video as frames - there is no video content
        type over MCP. One selector per call: `count` for a contact sheet,
        `at` for moments (seconds, or "frame:N"), or `seams` (true, or seam
        numbers from 1) for the frame pair either side of each join. `seams`
        needs `boundaries`: each later shot's first frame, the running sum of
        the shots' `frame_count` from `get_gallery_metadata` on their own
        files; `names` names the shots. Over budget, tiles shrink together,
        never drop.

        `workspace` pins this call to another workspace (#99)."""
```

Add the same sentence about `frame_count` / `get_gallery_metadata` to the `dw_mcp/media.py` docstring (no budget cost there) and to the `docs/MCP.md` row at line 238, replacing "with `boundaries`, the frame index each shot after the first starts at, and `names`" with "`boundaries` is each later shot's first frame - the running sum of the shots' `frame_count` from `get_gallery_metadata` on their own `intermediate/` files - and `names` names them".

- [ ] **Step 4: Run both tests**

Run: `venv/bin/python -m pytest tests/test_mcp_server.py -q -k "boundaries_come_from or fits_the_budget or get_output_frames" 2>&1 | tail -4`
Expected: PASS. If the budget test fails, the message prints the total; cut the phrase "there is no video content\n        type over MCP. " (it is restated in `instructions` and MCP.md) before anything else.

- [ ] **Step 5: Commit**

```bash
git add dw_mcp/server.py dw_mcp/media.py docs/MCP.md tests/test_mcp_server.py
git commit -m "docs(mcp): #193 - get_output_frames says where seam boundaries come from

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Contact-sheet cells carry their timestamp on the image

**Improvement:** The text part lists `frame (s)` per cell, but correlating a 3×4 grid to a list is where a vision model slips. `frame_grid`'s `_grid_tile` in `dw/tasks/video_utils.py:255` already burns a timestamp into a tile; reuse it, keeping `_fit_width`'s never-upscale rule (which `_grid_tile` lacks).

**Files:**
- Modify: `dw/media_frames.py` (`contact_sheet`, new `_stamped_tile`)
- Test: `tests/test_media_frames.py`

**Interfaces:**
- Consumes: `_grid_tile(frame, index, fps, tile_width, label) -> PIL.Image` from `dw/tasks/video_utils.py`; Task 1's `fit(image, index)`.
- Produces: contact-sheet cells labelled; moments and seam tiles unchanged (a seam pair must stay pixel-comparable for Task 7).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_media_frames.py`:

```python
def test_contact_sheet_cells_are_stamped_with_their_timestamp(tmp_path):
    """A flat grey cell has zero variance; a stamped one does not. The stamp
    is what lets a reader of the image, not the text, say which cell is
    which."""
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6, width=64, height=32)

    sheet = contact_sheet(str(tmp_path / "ramp.mp4"), 4, tile_width=64)

    columns = 2  # _default_columns(4)
    for cell in range(4):
        row, col = divmod(cell, columns)
        corner = sheet["image"].crop((col * 64, row * 32, col * 64 + 32, row * 32 + 16))
        assert numpy.asarray(corner.convert("L")).std() > 5, (
            f"cell {cell} carries no stamp"
        )
    # the stamp sits in a corner: the opposite corner is still the flat grey
    cell = sheet["image"].crop((32, 16, 64, 32))
    assert numpy.asarray(cell.convert("L")).std() < 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_media_frames.py -q -k stamped 2>&1 | tail -4`
Expected: FAIL on `cell 0 carries no stamp`.

- [ ] **Step 3: Implement**

In `dw/media_frames.py`, import `_grid_tile` beside the other `video_utils` imports, and add:

```python
def _stamped_tile(image, index, fps, tile_width):
    """A contact-sheet cell: fitted like `_fit_width` (never upscaled) and
    stamped with its timestamp by `frame_grid`'s own tile maker, so the
    sheet says which cell is which without the text part."""
    return _grid_tile(image, index, fps, min(int(tile_width), image.width), label=True)
```

In `contact_sheet`, change the fitter to:

```python
    images = _read_frames(
        path,
        indexes,
        fit=lambda image, index: _stamped_tile(image, index, shape["fps"], tile_width),
    )
```

- [ ] **Step 4: Run the frames suites**

Run: `venv/bin/python -m pytest tests/test_media_frames.py tests/test_server.py -q -k "frames or seam or contact" 2>&1 | tail -4`
Expected: all pass. `test_gallery_frames_makes_a_contact_sheet` checks `frames`, not pixels, so it is unaffected.

- [ ] **Step 5: Commit**

```bash
git add dw/media_frames.py tests/test_media_frames.py
git commit -m "feat(server): #193 - contact-sheet cells carry their own timestamp

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Each seam tile carries a `difference` number

**Improvement:** Twelve seam pairs are twelve images to look at; a per-seam mean absolute pixel difference (0-255, on the fitted tiles already in hand) lets an agent rank them and look only at the worst. Stage 3's probes will supersede this with a proper rule; this is the cheap version available because both frames are already decoded. Reported as a number only - no threshold, no verdict.

**Files:**
- Modify: `dw/media_frames.py` (`seam_tiles`), `dw_mcp/server.py` (`get_output_frames` text part), `docs/MCP.md:238`
- Test: `tests/test_media_frames.py`, `tests/test_server.py`, `tests/test_mcp_server.py`

**Interfaces:**
- Produces: seam tile dicts gain `"difference": float` (mean |before − after| over RGB, 0-255). `_encoded_tile` copies every non-`image` key, so the route and `dw_mcp/media.py` pass it through with no change.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_media_frames.py`:

```python
def test_a_seam_tile_reports_the_mean_difference_across_the_join(tmp_path):
    write_ramp_mp4(tmp_path / "ramp.mp4", frames=24, fps=6)

    tiles = seam_tiles(str(tmp_path / "ramp.mp4"), boundaries=[8], tile_width=32)

    # frame 7 is grey 70, frame 8 grey 80: a flat difference of 10
    assert tiles[0]["difference"] == pytest.approx(10.0, abs=6)
```

Append to `tests/test_server.py`:

```python
def test_gallery_frames_seam_tiles_carry_their_difference(server, tmp_path):
    from tests.test_media_frames import write_ramp_mp4

    with server(success_script) as client:
        write_ramp_mp4(tmp_path / "outputs" / "cut.mp4", frames=24, fps=6)

        response = client.get(
            "/api/gallery/cut.mp4/frames", params={"seams": "true", "boundaries": "8"}
        )

        assert response.status_code == 200
        assert response.json()["tiles"][0]["difference"] == pytest.approx(10.0, abs=6)
```

In `tests/test_mcp_server.py`, find the existing seams call at line ~341 (`"get_output_frames", {"name": "cut.mp4", "seams": [1]}`) and add `"difference": 12.5` to the tile JSON it serves, then assert on the text part: `assert "difference: 12.5" in text_part.text`.

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_media_frames.py tests/test_server.py tests/test_mcp_server.py -q -k difference 2>&1 | tail -6`
Expected: `KeyError: 'difference'` ×2, and the text assertion fails.

- [ ] **Step 3: Implement**

In `seam_tiles`, inside the `for seam, boundary in chosen:` loop after `before`/`after`:

```python
difference = float(
    numpy.abs(
        numpy.asarray(before, dtype=numpy.int16)
        - numpy.asarray(after, dtype=numpy.int16)
    ).mean()
)
```

(add `import numpy` at the top) and `"difference": round(difference, 2)` to the tile dict. In `dw_mcp/server.py`'s `get_output_frames`, in the per-tile loop:

```python
            if tile.get("difference") is not None:
                where += f"  difference: {tile['difference']}"
```

placed before `lines.append(...)`. In `docs/MCP.md` row 238, after "side by side", add "(each pair carries `difference`, the mean pixel change across the join, 0-255 - rank seams by it and look at the worst)".

- [ ] **Step 4: Run and commit**

Run: `venv/bin/python -m pytest tests/test_media_frames.py tests/test_server.py tests/test_mcp_server.py tests/test_mcp_media.py -q 2>&1 | tail -3` - all pass (the budget test is untouched: the text part is not the description).

```bash
git add dw/media_frames.py dw_mcp/server.py docs/MCP.md tests/
git commit -m "feat(server): #193 - a seam tile carries the mean pixel difference across its join

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `hear` - sound around each `at` moment in the same call

**Improvement:** Checking lip-sync or a hit point today is a frames call and an audio call with no shared clock. `get_output_frames(at=[...], hear=2.0)` returns, after each moment's image, the 2 s of soundtrack centred on it, via the existing audio route - client-side only, no server change.

**Files:**
- Modify: `dw_mcp/media.py` (`get_output_frames`), `dw_mcp/server.py` (`get_output_frames`), `docs/MCP.md:238`, `docs/RELEASING.md` (Unreleased)
- Test: `tests/test_mcp_media.py`, `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `get_output_audio(client, name, start, duration, workspace) -> {data, mime_type, bytes, duration_seconds, excerpt}` and `DwApiError` from `dw_mcp/media.py`.
- Produces: `get_output_frames(..., hear=None)`; each `at` tile gains `"audio": {data, mime_type, excerpt}` or `"audio_error": str`; the result dict gains `"hear"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcp_media.py` (uses the file's existing `frames_server`, `tile_json` helpers):

```python
def test_hear_fetches_an_excerpt_around_each_moment():
    calls = []

    def handler(request):
        calls.append((request.url.path, dict(request.url.params)))
        if request.url.path.endswith("/frames"):
            return httpx.Response(
                200,
                json={
                    "frame_count": 48,
                    "fps": 24.0,
                    "tiles": [
                        tile_json(64, 32, "00:01.0 (frame 24)", 24, 1.0),
                        tile_json(64, 32, "00:00.2 (frame 5)", 5, 0.2),
                    ],
                },
            )
        return httpx.Response(
            200,
            content=b"RIFF" + b"\0" * 64,
            headers={
                "content-type": "audio/wav",
                "x-dw-duration": "2.0",
                "x-dw-excerpt-start": request.url.params["start"],
                "x-dw-excerpt-duration": request.url.params["duration"],
            },
        )

    client = DwClient(transport=httpx.MockTransport(handler))

    result = get_output_frames(client, "x.mp4", at=[1.0, 0.2], hear=1.0)

    audio_calls = [c for c in calls if c[0].endswith("/audio")]
    assert [c[1]["start"] for c in audio_calls] == ["0.5", "0.0"]  # never before 0
    assert [c[1]["duration"] for c in audio_calls] == ["1.0", "1.0"]
    assert all(t["audio"]["mime_type"] == "audio/wav" for t in result["tiles"])
    assert result["tiles"][0]["audio"]["excerpt"]["start"] == 0.5


def test_hear_on_a_mute_clip_keeps_the_frames_and_says_so():
    def handler(request):
        if request.url.path.endswith("/frames"):
            return httpx.Response(
                200,
                json={
                    "frame_count": 48,
                    "fps": 24.0,
                    "tiles": [tile_json(64, 32, "00:01.0 (frame 24)", 24, 1.0)],
                },
            )
        return httpx.Response(404, json={"detail": "x.mp4 carries no soundtrack"})

    client = DwClient(transport=httpx.MockTransport(handler))

    result = get_output_frames(client, "x.mp4", at=[1.0], hear=1.0)

    assert "audio" not in result["tiles"][0]
    assert "no soundtrack" in result["tiles"][0]["audio_error"]


def test_hear_is_refused_without_at():
    client = frames_server([])

    with pytest.raises(DwApiError, match="hear"):
        get_output_frames(client, "x.mp4", count=4, hear=1.0)
```

In `tests/test_mcp_server.py`, beside the `count: 3` tool test at line ~384, add a test that serves a frames body with one tile and an audio body, calls `server.call_tool("get_output_frames", {"name": "shot.mp4", "at": [1.0], "hear": 2.0})`, and asserts the content types in order are `["image", "audio", "text"]` and the text contains `hear: 2.0s around each moment`.

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_mcp_media.py tests/test_mcp_server.py -q -k hear 2>&1 | tail -6`
Expected: `TypeError: get_output_frames() got an unexpected keyword argument 'hear'`.

- [ ] **Step 3: Implement the client**

In `dw_mcp/media.py`, add `hear=None` to `get_output_frames`'s signature (after `max_dimension`), and after the selector check:

```python
if hear is not None:
    if not at:
        raise DwApiError(
            "`hear` takes seconds of soundtrack around each `at` moment - pass `at`"
        )
    if float(hear) <= 0:
        raise DwApiError("`hear` is a positive number of seconds")
```

After `tiles, downscaled_to = _fit_tiles_within_budget(...)`:

```python
    if hear is not None:
        span = float(hear)
        for tile in tiles:
            start = max(0.0, float(tile["seconds"]) - span / 2)
            try:
                audio = get_output_audio(
                    client, name, start=start, duration=span, workspace=workspace
                )
            except DwApiError as e:
                tile["audio_error"] = str(e)
                continue
            tile["audio"] = {
                "data": audio["data"],
                "mime_type": audio["mime_type"],
                "excerpt": audio["excerpt"],
            }
```

Add `"hear": hear` to the returned dict. Note the audio excerpts are *not* counted against `_fit_tiles_within_budget` - each is its own ≤4 MB answer, as `get_output_audio` already enforces.

- [ ] **Step 4: Implement the tool**

In `dw_mcp/server.py`, add `hear: float | None = None,` after `max_dimension`, pass it through, and build parts in order: for each tile, the `ImageContent`, then `AudioContent(type="audio", data=tile["audio"]["data"], mime_type=tile["audio"]["mime_type"])` when `"audio" in tile`. In the text lines, per tile append `  hear: {tile['audio_error']}` when `audio_error` is set, and after the loop `lines.append(f"hear: {result['hear']}s around each moment")` when `result.get("hear")`. Docstring: append to the first paragraph `` `hear=N` adds N seconds of soundtrack around each `at` moment.``

- [ ] **Step 5: Run the budget test and pay for the words**

Run: `venv/bin/python -m pytest tests/test_mcp_server.py -q -k "fits_the_budget or hear or get_output_frames" 2>&1 | tail -6`

If the budget fails, cut in this order until it passes, re-running after each: (1) in `get_output_frames`'s docstring, "(seconds, or "frame:N")" → "(seconds or "frame:N")"; (2) in `get_output_audio`'s docstring, the sentence "The text part reports the track's length and, for an excerpt, exactly what was cut, so a slice is never mistaken for the whole." → "The text part says what was cut."; (3) in `get_output_audio`, "(the audio analogue of `get_output_image`)". Never raise `SURFACE_BUDGET`. Update the measurement comment above the constant with the new figure.

- [ ] **Step 6: Docs and commit**

`docs/MCP.md` row 238: add `hear=None` to the signature and arguments, and the sentence "`hear=N` also returns N seconds of soundtrack centred on each `at` moment, after its image - the way to check a hit point or lip-sync without reconciling two clocks; a mute clip keeps its frames and says `no soundtrack`." `docs/RELEASING.md` Unreleased: add "- `get_output_frames` takes `hear` (soundtrack around each `at` moment) and reports `difference` per seam pair."

```bash
git add dw_mcp/media.py dw_mcp/server.py docs/MCP.md docs/RELEASING.md tests/test_mcp_media.py tests/test_mcp_server.py
git commit -m "feat(mcp): #193 - get_output_frames hear= returns the soundtrack around each moment

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: The video skills map each failure mode to the tool that shows it

**Improvement:** Both skills now say "look for the family's failure modes" and separately name the tools; neither says which tool shows which mode. Stage 4's `dw:judging-output` is the real home; this is the interim, and it must fit under the 12 KiB cap that both skills sit at.

**Files:**
- Modify: `plugins/dw/skills/minimax-h3/SKILL.md:177-188`, `plugins/dw/skills/ltx-2.5/SKILL.md:152-158`
- Test: `tests/test_plugin_skills.py` (existing size cap + phrase pins; one new pin)

- [ ] **Step 1: Find every phrase the tests pin in the two step blocks**

Run: `grep -n -o '"[^"]*"' tests/test_plugin_skills.py | grep -i -E 'frames|audio|url|portrait|drift|voice|storyboard|softness|scene cut|silent'`
Every string printed that occurs in the current step 4 (H3) / step 5 (LTX) text must survive verbatim, on one line each.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_plugin_skills.py`:

```python
@pytest.mark.parametrize("path", [H3_SKILL, LTX_SKILL], ids=["minimax-h3", "ltx-2.5"])
def test_a_video_skill_maps_its_failure_modes_to_the_tool_that_shows_them(path):
    """Naming the tools and naming the failure modes in separate sentences
    leaves the agent to guess which shows which; the step has to say
    `seams=true` is for a join and `at` is for a moment."""
    text = skill_text(path)
    judge = text[text.index("## Run and judge") :]
    assert "`seams=true`" in judge
    assert "`at`" in judge
```

Run: `venv/bin/python -m pytest tests/test_plugin_skills.py -q -k maps_its 2>&1 | tail -3` - expect 2 FAIL.

- [ ] **Step 3: Rewrite H3 step 4**

Replace lines 177-188 of `plugins/dw/skills/minimax-h3/SKILL.md` with (measure with `wc -c` - the file must stay ≤ 12_288 bytes):

```
4. Judge it yourself: `get_output_frames(count=12)` for a clip's shape,
   `seams=true` (with each shot's start frame) for a cut's joins - a character
   that changes between shots (reference the same portraits everywhere), a
   portrait imposing its framing on every shot - `at` late in a chain for drift
   sharpening into noise, and `get_output_audio` for a voice-over without
   affect (the reference's delivery came through). Then `get_job` for the
   manifest and its warnings, `get_gallery_metadata` for duration and whether
   audio is present, and hand the user the gallery `url` (`list_gallery`).
   `get_output_image` works only on image steps - the Z-Image portraits and
   boards of `dialogue-short`, `storyboard`, `generated-subject-reference` and
   `music-video`. Also look for a storyboard skipped, every shot the same
   length, one look word on every board softening all of them.
```

- [ ] **Step 4: Rewrite LTX step 5**

Replace lines 152-158 of `plugins/dw/skills/ltx-2.5/SKILL.md` with:

```
5. Judge it yourself: `get_output_frames(count=12)` for a clip's shape,
   `seams=true` for a chained clip's joins, `at` near the end for a scene cut
   where the prompt contradicted the image or softness where the refine pass
   was skipped, and `get_output_audio` for a near-silent soundtrack. Then
   `get_job` for the manifest and its warnings, `get_gallery_metadata` for
   duration, size and whether audio is present, and hand the user the gallery
   `url` (`list_gallery`, or the manifest's file name).
```

- [ ] **Step 5: Run the skill tests**

Run: `wc -c plugins/dw/skills/*/SKILL.md && venv/bin/python -m pytest tests/test_plugin_skills.py -q 2>&1 | tail -4`
Expected: both files ≤ 12_288 bytes; all pass. If a pinned phrase fails, it wrapped - rejoin it on one line and take the bytes from a later line in the same step.

- [ ] **Step 6: Commit**

```bash
git add plugins/dw/skills tests/test_plugin_skills.py
git commit -m "docs(plugin): #193 - the video skills say which evidence tool shows each failure mode

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Full suite and hand-off

- [ ] **Step 1: Run everything**

Run: `venv/bin/python -m pytest -q 2>&1 | tail -3`
Expected: all pass (baseline before this plan: 5256 passed, 16 skipped), no new skips.

- [ ] **Step 2: Add the plan's outcome to the stage-1 plan's hand-off notes**

Append to `docs/superpowers/plans/2026-09-20-output-assessment-stage-1-evidence.md` a short "Review fixes (this plan)" list: the four fixes by task title and the five improvements, one line each, so the stage-2 author knows `difference` and `hear` exist before designing `shots`. Commit as `docs: #193 - stage 1 review fixes landed`.

- [ ] **Step 3: Report**

State the test count, that `SURFACE_BUDGET` and both skill caps still hold (quote the numbers), and which tasks were skipped if any. The branch is then ready for `git merge --no-ff feat/193-evidence` into `develop`; the deploy to lem is a separate hand-off (the `implementer-cycle-*` memories carry the restart recipe).
