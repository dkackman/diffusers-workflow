# `pcm_sha256`: a decoded-sample identity check for audio over MCP (#1139)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-02). Don filed the idea on 2026-11-01 from harnest#170. Plan v1 was
designed on #1139, and Don approved it unchanged (Q1 cut, Q2 yes). It was built
as one stage, #1158. The plan comment on #1139 is still the full design. This
file records what shipped.

## The problem

The MCP surface had no way to check whether two audio files are the same.
`compare_outputs` refuses `.wav`. `get_gallery_metadata` returned only derived
levels (duration, peak/mean dBFS, LUFS, true peak, an optional envelope). Two
files with equal levels are not proven identical: a polarity flip, a reordering
inside one envelope bin, or a one-LSB change all give the same levels. This
caused two problems in the suite:
- harnest#170: C-F619 step 3 ("same seed gives identical audio") fell back to
  comparing a 0.05 s envelope plus the five levels.
- C-F080: its sample-level properties had to be checked in pytest instead of
  over MCP.

## Verdict: build smaller

The audio hash shipped. **Cut: `pixel_sha256` for images and video** (Q1).
`compare_outputs` already reports "identical" (infinite PSNR) for images and
video at sampled moments. Image `media` is pinned to an exact key set, and
hashing every video frame would mean a full decode on every metadata call.
*What would bring it back:* a verify that needs exact identity across every
frame of a video, or an image identity check that `compare_outputs` can't
express. Either would be filed as a new idea.

These non-goals stayed out:
- a "how different" measure for audio;
- a hash of the file bytes;
- a hash in `list_gallery` or computed on save.

## Plan history

- **v1 (approved, 2026-11-01):** audio only, one stage.
  - Q1 [scope]: cut image and video hashes.
  - Q2 [none]: hash the native decoded dtype, interleaved `(frames, channels)`,
    little-endian, and trimmed to the container duration the same way the
    LUFS and envelope chunks are. `sample_format` is the numpy dtype name.
  - The plan corrected the proposal. The float32 array behind the levels loses
    precision for `s32` sources. Peak and mean are measured untrimmed while
    LUFS is trimmed. So the hash uses its own native-dtype path.

## What was built

### Stage A (#1158): `pcm_sha256` in `get_gallery_metadata` `media`

Commit `44cc7e00` on `feat/1139-A-pcm-sha256`, merged to `develop` as
`710a0e81` and deployed to lem on 2026-11-02.

- `dw/media.py`: `_AudioAccumulator(..., pcm_hash=False)` and
  `probe_media(..., pcm_hash=False)`. The hash is fed one decoded frame at a
  time over the native `to_ndarray()` dtype and trimmed by the same `length`
  the LUFS/envelope chunks use. It adds three new keys: `pcm_sha256`,
  `sample_format` and `frames`. `sample_rate` and `channels` were already
  reported.
- `dw/server/routes/media.py`: the `gallery_metadata` route always passes
  `pcm_hash=True`. There is no new REST or MCP parameter. `list_gallery
  media=true`, `check_written_media` and `probe_metadata` don't compute the
  hash.
- `dw_mcp/tools_catalog.py` and `docs/MCP.md` each say what is hashed and the
  rule for comparing: two hashes compare only when `sample_format`, `frames`,
  `sample_rate` and `channels` all agree. The OpenAPI snapshot and UI types
  were regenerated for the docstring change. The schema didn't change.
- Tests: `TestPcmHash` in `tests/test_media_info.py` and route cases in
  `tests/test_server.py`. They cover the byte layout, the hash being off by
  default, an extra `LIST` chunk, WAV vs FLAC, packed vs planar, polarity,
  one-LSB, an AAC mp4 soundtrack, a mute video, a damaged track, the envelope
  and levels being unchanged, and `list_gallery` not carrying the hash.

**Deviations recorded at build:**
- On a damaged track, `pcm_sha256`, `sample_format` and `frames` are absent.
  `sample_rate` and `channels` stay, as they did before the stage, because
  they are header fields.
- If the decoded dtype changes mid-stream, the hash is omitted, because no
  single `sample_format` describes it.
- This libav decodes FLAC to packed `s16`. So WAV vs FLAC pins container
  invariance, and packed vs planar is pinned by feeding both ndarray layouts
  to the accumulator.

**Review and verify:** the architecture review passed. The tester verified it
over MCP with no bounces. C-F658 to C-F661 all passed as written:
- wav and mp3 assets: the mp3's `frames` shows the priming trim;
- mp4 soundtrack vs mute mp4 vs jpg;
- envelope bins unchanged at 1.0 s and 0.05 s;
- two seed-1124 MMAudio runs gave equal hashes, seed 1125 gave a different
  one, and the kept asset hashed the same as its source.

| Stage | Bounces | Cost |
|---|---|---|
| A (#1158) | 0 | not recorded |

## Known edges and follow-ups

- **Lossy codecs:** the hash is of this server's decode. It is stable run to
  run on one box but may differ across libav versions. Suite cases compare
  within one server.
- **C-F080's "bit-identical outside a region"** needs a hash of a time range.
  That would be a follow-up with a new parameter and needs its own approval.
- **C-F619 step 3** can now be tightened to compare `pcm_sha256`. That
  amendment is the curator's.
- Found by the tester, outside this feature: `validate_workflow`'s plan
  fingerprint doesn't cover a step-level `seed` (seed 1124 and 1125 inline
  workflows got the same fingerprint). It was noted on #1158 but not filed.
