# `window_video` and `join_windows` host memory: range reads and a uint8 join (#695)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-08). Plan v1 was approved by Don on 2026-10-08 with the defaults
(Q1 build smaller, Q2 cut the file-backed `safe_get`, Q3 moot). Plan v2
recorded those answers with no change of scope. It was built as one stage,
#780, which shipped and was verified on mps (mini-ai) on 2026-10-08.

## The idea

The 2026-10-07 develop review (structural item 5) found that the overlap
windowing tasks from #601 held the whole source in host RAM:
- `window_video` decoded and stacked the entire source (`load_audio_video`
  → `frames_as_array`) to emit one window. Under `for_each` that meant N
  full decodes. At 1080p, `restore-long`'s 32-window limit (about 140 s)
  is about 21 GB per decode.
- `join_windows` held every window as uint8, plus a full decode of the
  source (only for its frame count and audio), a float32 `joined` array of
  every frame, a float32 copy of each window and a PIL list of the result.
- `argument_media` copied a fetched body (`io.BytesIO(bytearray)`), for a
  2× peak of up to 2 GiB per URL fetch.

## Verdict

**Build smaller.** There was no measured demand: no field report, and no
job in Don's workspaces had run `restore-long`, `window_video` or
`join_windows`. Only the `regression-*` cases had. The case was
structural. `restore-long` exists for long sources and couldn't run on
long HD ones. The fix had no MCP, REST, schema or syntax surface and is
fully reversible.

**Cut, with what brings it back:**
- **The file-backed `safe_get` variant.** A URL `video` already arrives as
  a silent decoded `FrameList` (`_fetch_remote_video`), so a range read
  couldn't help it. A real spill to disk means changing `safe_get` itself,
  which the 2026-10-07 security fixes had just hardened, and it breaks
  about a dozen tests that mock it. *Back when:* a field report of a URL
  media fetch running out of host RAM, or the `local_media_file` path
  (`dw/argument_media.py`) being used for sources near the 1 GiB cap.

## What the plan found against the idea

1. **A frame-range reader alone doesn't bound memory.** `window_video` also
   needs the source's exact frame count and its audio without a full
   decode. The count comes from the header (`video_shape`), with a guard,
   and the audio from `decode_soundtrack`, which reads the audio stream
   only. `extract_audio` was ruled out: it returns s16 cut on rounded
   seconds, which would break the sample-exact tiling.
2. **In `join_windows` the dominant terms were the full source decode and
   the float32/PIL copies,** not only the accumulator.

## What was built

### Stage 1 (#780): range-read windows, uint8 join

Merged to `develop` at e27d2890 (`feat/695-A-range-read-windows`: c6d9d677
code, 5c94e633 tests and docs).

- **`dw/media.py` `read_frame_range(path, start, stop)`** returns
  `(n, h, w, 3)` uint8 from one keyframe seek and one forward decode,
  converted with `rgb24` like the full decode. `read_frames`' seek logic
  is split into the shared helpers `_seek_needed`, `_seek_before` and
  `_landed_at`, so it isn't copied. A seek that lands past the start falls
  back to the top of the file. A read that hits EOF early raises
  `ShortFrameRange`. `count_video_frames` counts by decoding.
- **Frame count.** `total` comes from `video_shape`. If a range read runs
  short, it is retried once on a counted total. If the counts still
  disagree, it refuses, naming both ("header says it has N frames and
  decoding counted M").
- **`window_video`.** A `VideoFileReference` source takes the range path,
  with audio from `decode_soundtrack` → `fit_codec_padding` → the
  `frames_to_samples` cut. Other source types keep the old code. Error and
  refusal text is unchanged.
- **`join_windows`.** A file source (a str path, `asset:`/`output:`, or a
  `VideoFileReference`) is read for its header and soundtrack only. A URL
  source keeps the old load. The output is a uint8 array. Seams are
  blended in float32 over only `overlap` frames, with a carry of the last
  `overlap` unrounded frames, so the result equals the old float32
  accumulator exactly, including when overlap > stride.
- **Tests.** `tests/test_read_frame_range.py` (byte parity against a full
  decode: single keyframe, g=10 ranges straddling keyframes, non-zero
  start pts, the last frames, the EOF guard). `tests/test_window_video.py`
  `TestFileMatchesInMemory`, `TestFileMemory`, `TestHeaderCountMismatch`.
  `tests/test_join_windows.py` `TestUint8Output` (dtype, exact parity for
  3 curves × 8 combinations), `TestFileSourceRoundTrip`, `TestJoinMemory`.
- **Docs.** `docs/TASKS.md` gained a memory paragraph each for
  `window_video` and `join_windows`. The window and join rows of
  `docs/ARCHITECTURE.md` name the new tests. `VideoFileReference`'s stale
  docstring was corrected.

**Verified** (C-F371–C-F375, mps on mini-ai): windows on a 4,089-frame
source match the source frame for frame at the prefix, a mid-file seek and
the end pad; a 7-window join round-trip is 282 frames with shots tiling
exactly and the last sample at 518175; refusals keep their text with no
decoder error; `previous_result:` and URL sources are unchanged. On
memory (C-F373), job peak RSS grew 0.47 MB for a 4,089-frame source and
0.47 MB for a 121-frame one, against roughly 6.4 GB of decoded source
before.

## Deviations from the plan

1. **`dw/writers.py` `frames_for_encoding`** hands a uint8 array to
   `encode_video` as a torch tensor. The plan assumed the writer already
   took uint8, but diffusers' `encode_video` treats an ndarray whose values
   all lie in [0, 1] as floats and multiplies by 255, so a near-black uint8
   join would have been written blown out.
2. **`dw/audio_qc.py` `remeasure_shots_after_mux`** counted frames with
   `len(frames or [])`, which raised on an ndarray. It now checks for None
   first.
3. **`dw/tasks/video_utils.py` `local_video_path`**: `load_audio_video`'s
   local-path validation, split out so `join_windows` can resolve a
   location to a validated path without decoding it.
4. **The `window_video` memory test's audio allowance is 4× the decoded
   track, not 1×.** The existing `decode_soundtrack` holds its chunks,
   their concatenation and a transposed copy at once. Measured: 9.5 MB
   peak against a 13.5 MB bound on 2,000 frames at 64×64; a full float32
   decode would be 98 MB.

## Bounces per stage

| Stage | Bounces |
|---|---|
| #780 | 0 |

The architecture review passed on the first hand-off. No stage comment
names `usage:` figures, so per-stage session cost is not recorded here
(the plan estimated $4–6).

## Known edges

- **Header count disagreement between the two tasks.** `window_video`
  falls back to a decode-counted total when a header count runs short, but
  `join_windows` and the validate-time window-count check use the header
  count only. On a source whose header overstates its frame count, they
  could disagree. No reachable fixture has such a header, so this is
  covered by unit tests only (`TestHeaderCountMismatch`), not over MCP.
- **Child decoder memory.** RSS doesn't show memory a child decoder
  process uses, if there is one.
- **Host-memory projection.** `dw/host_memory_projection.py` was left
  alone. Old `restore-long` history will over-warn until new runs replace
  it; it only ever warns.

## Not built, and why

- **The file-backed `safe_get`** (cut, trigger above).
- **Streaming the join's output to disk.** `gather:` keeps every window
  resident until the join runs (`release_unreferenced_results`), so the
  windows' own uint8 is the floor without a much bigger change.
- **URL and `previous_result:` sources** keep their old paths, by design.
