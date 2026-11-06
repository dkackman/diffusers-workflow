# Audio pass, long narration and long music (Task 21 Step 6, Task 22 Step 6)

2026-11-05. Bench server lem:8766 (worktree at `6a89744a` when measured), cuda:0.
Arms: `bench/arms/audio-pass.json`, `audio-long.json`, `audio-long-prep.json`;
records: `audio-pass.json`, `audio-long.json`, `audio-long-prep.json` (the long
arms' "wrote no video" error text is the driver's, expected for audio-only output).

How it was measured: outputs copied off lem and read on the Mac. ffmpeg for PSNR,
frame hashes, `ebur128` and `silencedetect`; the server's own `assess` route for the
shot probe; `check_script` (whisper-base) and a per-take whisper-small pass, run
locally on the waveforms; `analyze_beats` run locally on each music section; ECAPA
speaker embeddings (`speechbrain/spkrec-ecapa-voxceleb`) for voice drift. No GPU job
was submitted. Claude cannot judge sound by ear: the listening list is in `for-don.md`.

## Task 21: audio pass (#1192)

**Verdict: the mechanism works.** The output plays the source's own frames, and each
track is new audio (correlation with the source track is about 0), so this is not a
copy. Every scripted line is spoken word for word. The levels are sane, with no
clipping. `listen_scale` 0.5 and 0.25 cost less, but this run lost most of the saving
to a first-pass stall. They also put the line 0.7-0.9 s earlier than the line in the
source picture, so lip sync needs Don's eye before either becomes a default. Same seed
throughout, one draw per arm.

Note: the S4 "silent" shots are not silent. Each S4 source shot carries its own H3
track at about -23 LUFS with its line spoken. The pass replaced that track.

### Picture

| arm | PSNR vs source (Y / avg / min frame) | decoded frames |
|---|---|---|
| S4 cold_open ls1 / ls0.5 / ls0.25 / foley | 41.6 / 42.9 / 40.7 dB | all four bit-identical to each other |
| S4 deflect ls1 | 42.2 / 43.8 / 42.3 | |
| S4 react ls1 | 42.7 / 44.0 / 42.5 | |
| S4 button ls1 | 42.4 / 43.9 / 42.2 | |
| S3 span same-line / new-line | inf vs the trimmed span (byte-identical stream); 42.8 / 44.2 / 42.5 vs the S3 take | identical to each other |

About 43 dB is one x264 generation: the S3 trim alone measures the same (42.8 dB)
against the take it was cut from. A VAE round trip would be far lower. For a control,
a wrong-shot pair (cold_open output against the deflect source) measures 14.2 dB. The
picture does not depend on `listen_scale`.

### Track

Levels on each 5.17 s track: `ebur128` integrated loudness (I) and true peak; `assess`
dead air (the longest gap below the line's floor); silence share is the 50 ms windows
under -50 dBFS. Clipped samples: 0 everywhere.

| arm (job) | I LUFS | true peak dBFS | LRA | silent share | dead air (at) | line heard | line span s (source's) |
|---|---|---|---|---|---|---|---|
| cold_open ls1 (`0e13c36a659b`) | -22.8 | -8.0 | 4.4 | 0.50 | 1.2 s (0.0) | exact | 1.22-4.18 (1.14-4.18) |
| cold_open ls0.5 (`a1d731ddbf29`) | -24.1 | -8.1 | 6.3 | 0.44 | 0.9 s (2.0) | exact | 0.46-3.70 |
| cold_open ls0.25 (`c92a26d1c1d3`) | -21.5 | -7.9 | 4.8 | 0.32 | 0.3 s | exact | 0.26-3.20 |
| cold_open foley ls1 (`b51cc152524e`) | -21.5 | -11.1 | 15.2 | 0.59 | 2.9 s (0.0) | no line asked; Whisper heard "Thank you very much." | - |
| deflect ls1 (`adfb95588730`) | -23.9 | -12.1 | 3.6 | 0.34 | 1.05 s (1.0) | exact, then laughter | 0.32-3.42 (0.32-3.56) |
| react ls1 (`9f76f275eb8b`) | -23.8 | -11.9 | 3.1 | 0.19 | 0.8 s (0.0) | exact, then laughter | 0.82-3.34 (0.90-3.38) |
| button ls1 (`a37bd9bc70d1`) | -20.1 | -7.5 | 8.4 | 0.34 | 1.45 s (0.0) | exact, then laughter | 1.48-2.30 (0.56-2.32) |
| S3 seg1 same line (`864375a2cef2`) | -21.5 | -7.6 | 2.8 | 0.51 | none | exact | 1.96-4.46 (1.20-5.14) |
| S3 seg1 new line (`4b6ebb139735`) | -20.2 | -7.3 | 1.5 | 0.43 | none | exact | 1.20-4.12 |
| *sources:* S4 cold_open / deflect / react / button | -24.5 / -23.1 / -22.3 / -23.5 | -9.9 / -5.7 / -11.2 / -9.4 | | | 1.15 / 1.05 / 0.9 / 0.85 s | | |
| *source:* S3 seg1 span | -14.2 | -2.2 | 3.4 | 0.06 | none | | |

- The new S4 tracks sit within about 2 LU of the source tracks, and their dead air is
  the same kind the sources had. The S3 re-voices are 6-7 LU quieter than the S3 take,
  whose bed is loud in the low band (-19.8 vs -29.1 dBFS).
- On deflect and react the line starts within 0.1 s of where the source's line starts.
  Button starts 0.9 s later and is shorter. The S3 same-line re-voice starts 0.76 s
  late and ends 0.68 s early, against a mouth that moves for the source's 3.9 s line.
  Whether that reads as lip-sync drift is Don's call.
- `listen_scale` moves the line: ls1 starts 0.08 s after the source's line, ls0.5
  0.68 s before it, ls0.25 0.88 s before it. That is one seed, but a coarser view of
  the face plausibly loosens sync.
- The foley arm asked for no speech. Whisper heard "Thank you very much." (probably a
  hallucination over laughter), and the first 2.9 s are dead air, though the prompt
  asks for footsteps as he enters.

### listen_scale cost (cold_open, warm pipeline)

| | ls1 (mean of 6 warm ls1 runs) | ls0.5 | ls0.25 |
|---|---|---|---|
| job wall s | 188 | 193.5 | 160.5 |
| encode s | 44.5 | 26.8 | 15.9 |
| first denoise pass s | 13.4 | **61.5** (stall) | **45.3** (stall) |
| steady pass s | 10.2 | 8.7 | 8.7 |
| decode s | 20.1-22.2 | 10.2 | 4.7 |
| wall with a normal first pass (est.) | 188 | ~145 (-23%) | ~129 (-31%) |

Both reduced-scale runs hit the engine's "first denoise pass took 45-62 s" warm-up
warning: a new shape after the ls1 runs. The steady pass falls only 15%, so the
audio tokens and the text dominate a pass. The first ls1 run (`0e13c36a659b`, 312 s)
included a 115.8 s cold load.

**Recommendation:** promote add-sound to `templates/minimax/add-sound` at ls1 if Don's
ear passes it. Keep 0.5 and 0.25 as knobs, not defaults, until a lip-sync read says
the earlier line is harmless.

## Task 22: long narration (S6)

**Verdict: `max_seconds` 60 is not safe. Take 45 is clean, and 45 should be the
default.** At 60, 5 of the 10 takes (two seeds) drop or loop words. At 45, none of the
6 takes does. The join level steps are within the in-take spread. The join pause at
`gap_seconds` 0.3 is at or under a normal sentence pause. At 0.05 the words run
together at two joins. ECAPA finds no voice drift between takes. The narration is
also shorter than designed: the voice speaks at about 3.0 words a second, not the 2.5
assumed. So the 594-word text runs 198 s at take 45, not 240 s. The t60 runs land
just under the 3-minute gate (178-180 s), because words were dropped.

### split_takes

594 words. `split_takes` agrees with the job's step count (5 takes at 60, 6 at 45). It
kept "Dr. Alder" whole, and no sentence was over a limit, so no comma cut was used.
The takes are packed greedily across paragraph breaks. At 60, take 1 ends on the
first sentence of paragraph 2 ("On the night the storm came in, the wind arrived
first."). So a join falls mid-paragraph, while a paragraph break inside a take gets
only a sentence pause.

| limit | words per take | take 1-n end on |
|---|---|---|
| 60 s (150 words) | 125, 137, 127, 143, 62 | "...wind arrived first." / "...nothing to be done." / "...for a long while." / "...away from the stones." / end |
| 45 s (112 words) | 103, 109, 86, 91, 106, 99 | "...stool to wait." / "...break white." / "...he saw the ship." / "...for a long while." / "...while it turned." / end |

### Words heard (per-take whisper-small; whisper-base check_script agrees)

| run (job) | file s | take: words heard / expected (take s) | what is missing |
|---|---|---|---|
| t60 s1166 (`3668ebd2e4de`) | 179.7 | 83/125 (30.8), 137/137, 127/127, 127/144, 62/62 | take 1 says "He trimmed the wick, wiped the salt from the glass" twice (24.2 s and 27.4 s), then stops, dropping the rest of paragraph 1 and paragraph 2's first sentence. Take 4 jumps from "while it turned" to "...every eight seconds the boat turned" (about 153 s): "the beam swept across the water, and every eight seconds it found the boat" is not said |
| t60 s1167 (`c9f1f6145c3c`) | 178.3 | 125/125, 135/137, 98/127 (30.2), 126/144, 62/62 | take 2 stumbles ("does not all wince. This not all wince stands", about 77-80 s) and cuts off at "because there..." (85.6 s, the join). Take 3 stops at "trough of a wa-" (115.7 s), dropping 2 sentences. Take 4 has the same "every eight seconds" skip (about 151 s) |
| t45 s1166 (`111a5623cba0`) | 198.0 | 103/103, 109/109, 86/86, 91/91, 107/107, 99/99 | nothing (numbers transcribed as digits, punctuation only) |
| gap 0.05 s1166 (`c221ffe761a9`) | 178.4 | same speech as t60 s1166, sample for sample | same as t60 s1166 |

The failures are a take that stops early, or a jump between repeated phrases ("every
eight seconds", three times in one sentence). Neither happens at 45, where the same
sentence is spoken whole. Complete takes run at 2.93-3.10 words a second, so a 60 s
take of 125-144 words is 41-48 s of speech, and a 45 s take is 29-37 s. The reference
is 4.55 s (-24.8 LUFS). The limit that bites is the take's speech length, well short
of the ~95 s context `voice-clone` cites.

### Joins

The join is found by its inserted digital silence. The digital gap is 0.241 s at
`gap_seconds` 0.3: the 30 ms crossfade is taken off twice. "Pause" is the whole quiet
run around the join, -45 dBFS at 20 ms. The level step compares 5 s of speech on each
side.

| run | join at s | pause s | level step dB |
|---|---|---|---|
| t60 s1166 | 30.8 / 76.3 / 117.2 / 159.7 | 0.24 / 0.40 / 0.44 / 0.48 | +1.5 / +1.4 / -1.0 / +2.6 |
| t60 s1167 | 40.6 / 85.6 / 115.8 / 158.1 | 0.50 / 0.24 / 0.26 / 0.46 | +0.4 / +0.2 / -1.2 / +1.2 |
| t45 s1166 | 33.3 / 70.0 / 99.3 / 129.6 / 164.9 | 0.46 / 0.46 / 0.52 / 0.44 / 0.50 | +1.0 / +1.2 / +1.5 / -2.6 / +1.2 |
| gap 0.05 s1166 | 30.8 / 76.1 / 116.7 / 158.9 | 0.08 / 0.17 / 0.20 / 0.23 (5 ms read) | as t60 s1166 |

- Floor: a sentence pause inside a take measures a median of 0.52-0.56 s (p90
  0.76-0.79 s). The same 5 s level step taken at in-take pauses measures a median of
  0.6-0.9 dB, p90 1.4-1.8 dB and max 1.7-3.0 dB. Every join step (max 2.6 dB) is
  inside that max. No loudness fix is needed: each file is -24.1 to -24.5 LUFS, true
  peak -4.4 to -5.6 dBFS, no clipping.
- A join at 0.3 is 0.24-0.52 s, at or under the median sentence pause. The 0.24 s
  joins are takes that end and start on speech. At 0.05, joins 1 and 2 have under a
  fifth of a second between sentences. **Recommendation: `gap_seconds` 0.45.** That
  puts the shortest join near 0.39 s and the typical one near the in-take median.
  Don's ear decides.
- The 0.05 arm is the t60 s1166 audio with each gap 0.25 s shorter (the takes match
  sample for sample). So it isolates the gap: same seed, same takes, same draws.

### Voice drift (ECAPA cosine)

| run | take vs reference | adjacent takes | halves within one take (floor) |
|---|---|---|---|
| t60 s1166 | 0.82-0.85 | 0.93-0.95 | 0.91-0.95 |
| t60 s1167 | 0.81-0.84 | 0.92-0.96 | 0.92-0.96 |
| t45 s1166 | 0.79-0.85 | 0.94-0.95 | 0.92-0.96 |

Takes match each other as closely as two halves of one take do. The lowest score is
the shortest final take each time. Take 2 of the two seeds scores 0.976 against each
other. There is no measured drift, so the ear has the last word.

## Task 22: long music (S6)

**Verdict: the gate is not met, and the cause is a template defect.** Each file is
335.5 s (331 s at xf3000), as the arithmetic says. But Music 3 wrote sections of only
26.8-70.1 s for 85 s asked (`audio_duration` 128 is a ceiling, and a 4-line lyric ends
early). `slice_audio` then pads each section to 85 s with digital silence, and the
server warned about it on every run. So the pieces hold 13-57 s silent holes between
sections, and 33-45 s of silence at the end. Music content is 245-246 s at seed 1166
and 179 s at seed 1167, under one track's 300 s. Only one join per seed-1166 piece is
a real music-to-music crossfade.

| run (job) | file s | music s | sections generated s (kept 85) | silent holes s (at) |
|---|---|---|---|---|
| s1166 xf1500 (`52303a4436d4`) | 335.5 | 246 | 52.1, 70.1, >=85, 39.8 | 31.5 (52.1), 13.5 (153.6), 45.2 tail (290.3) |
| s1167 xf1500 (`2fc471b36521`) | 335.5 | 179 | 63.2, 26.8, 38.0, 52.1 | 20.6 (63.1), 57.1 (110.3), 45.8 (205.0), 32.9 tail (302.6) |
| s1166 xf3000 (`3bfee8aad7c5`) | 331.0 | 244.5 | same draw as s1166 | 30.0 (52.1), 12.0 (152.1), 45.2 tail (285.8) |

- Joins: the one real crossfade is s1166 section 3 into 4. Section 3 is cut at 85 s,
  mid-music. The level step there is +0.6 dB at xf1500 (250.5 s) and 0.0 dB at xf3000
  (246 s). Every other join is a section's own ending, then silence, then a fresh
  start. The step across it (+11.7, -10.0 dB at s1166; -3.7, -0.2, -8.1 dB at s1167)
  compares a fade tail with an intro. It is not a seam figure.
- Tempo: `analyze_beats` finds a reliable pulse in only 5 of 12 sections (method
  `onset`). Those read 84.5 and 87.0 BPM (s1166 sections 2 and 3, the same in both
  crossfades) and 63.2 BPM (s1167 section 2), against the 72 BPM asked. The rest fall
  back to loudness peaks (69-105 BPM, unreliable). The beat tracker cannot settle
  tempo continuity on this soft folk. That goes to the ear.
- xf3000 against xf1500: the same seed gives the same sections, so the crossfade
  changes only the one real join (0.0 vs +0.6 dB) and shortens the file by 4.5 s.
  Nothing else can differ.
- Loudness: -18.1 / -18.6 / -18.1 LUFS (silence included). Sample peak -2.1 / -2.8 /
  -1.4 dBFS against `normalize_audio`'s -3 dBFS. That is mp3 encode overshoot, true
  peak up to -1.3 dBFS, with no clipping.

**Defect (template `workflows/templates/minimax/music-long.json`):** the `trim` step
slices each section to `item:seconds` with no regard to the generated length. Any
section Music 3 ends early is padded with digital silence, and the crossfade then
joins silence to the next intro. Fix options:
1. Trim each section to `min(generated, seconds)`, cutting its trailing silence before
   the fade.
2. Write lyrics long enough for 85 s: about 2-3x these 4-line sections. The skill
   should say that the lyric, not `audio_duration`, sets the length.
3. Both, plus a check that refuses (or warns as a finding) a joined piece whose music
   is under the requested total.

The bench result cannot gate S6 music until it is rerun with longer lyrics or the
trim fix.

## Outputs (lem `~/bench-workspace/outputs/`)

| arm | job | file |
|---|---|---|
| ap-s4-cold_open-ls1 | `0e13c36a659b` | `MiniMaxH3AddSound-ap/20261106-002201-e099b1ba/MiniMaxH3AddSound-ap-sound.0-0.0.mp4` |
| ap-s4-deflect-ls1 | `adfb95588730` | `MiniMaxH3AddSound-ap/20261106-002706-a8cc5bf9/MiniMaxH3AddSound-ap-sound.0-0.0.mp4` |
| ap-s4-react-ls1 | `9f76f275eb8b` | `MiniMaxH3AddSound-ap/20261106-003015-3fe5ff5d/MiniMaxH3AddSound-ap-sound.0-0.0.mp4` |
| ap-s4-button-ls1 | `a37bd9bc70d1` | `MiniMaxH3AddSound-ap/20261106-003323-9a93c662/MiniMaxH3AddSound-ap-sound.0-0.0.mp4` |
| ap-s4-cold_open-ls050 | `a1d731ddbf29` | `MiniMaxH3AddSound-ap/20261106-003630-0dd54320/MiniMaxH3AddSound-ap-sound.0-0.0.mp4` |
| ap-s4-cold_open-ls025 | `c92a26d1c1d3` | `MiniMaxH3AddSound-ap/20261106-003943-56ccfd42/MiniMaxH3AddSound-ap-sound.0-0.0.mp4` |
| ap-s4-cold_open-foley-ls1 | `b51cc152524e` | `MiniMaxH3AddSound-ap/20261106-004224-f12f561f/MiniMaxH3AddSound-ap-sound.0-0.0.mp4` |
| ap-s3-seg1-same-line-ls1 | `864375a2cef2` | `MiniMaxH3AddSoundSpan-ap/20261106-004532-3a4001fc/final/MiniMaxH3AddSoundSpan-ap-sound.1-0.0.mp4` |
| ap-s3-seg1-new-line-ls1 | `4b6ebb139735` | `MiniMaxH3AddSoundSpan-ap/20261106-004840-98cb5c66/final/MiniMaxH3AddSoundSpan-ap-sound.1-0.0.mp4` |
| s6-narrator-ref-s1166 | `5623e25a933a` | `BenchS6NarratorReference/20261105-233044-a0358f86/final/BenchS6NarratorReference-narrator.0-0.0.wav` (asset `bench/narrator.wav`) |
| s6-narr-long-240s-t60-s1166 | `3668ebd2e4de` | `BenchS6NarrationLong/20261105-233313-a7732021/final/BenchS6NarrationLong-narration.0-0.0.wav` |
| s6-narr-long-240s-t60-s1167 | `c9f1f6145c3c` | `BenchS6NarrationLong/20261105-233643-19bc78a7/final/BenchS6NarrationLong-narration.0-0.0.wav` |
| s6-narr-long-240s-t45-s1166 | `111a5623cba0` | `BenchS6NarrationLong/20261105-234005-77b3c12f/final/BenchS6NarrationLong-narration.0-0.0.wav` |
| s6-narr-long-240s-gap005-s1166 | `c221ffe761a9` | `BenchS6NarrationLong/20261105-234351-b0a08747/final/BenchS6NarrationLong-narration.0-0.0.wav` |
| s6-music-long-335s-s1166 | `52303a4436d4` | `BenchS6MusicLong/20261105-234715-35bcd6ab/final/BenchS6MusicLong-piece.0-0.0.mp3` |
| s6-music-long-335s-s1167 | `2fc471b36521` | `BenchS6MusicLong/20261105-235710-f02aa4dc/final/BenchS6MusicLong-piece.0-0.0.mp3` |
| s6-music-long-335s-xf3000-s1166 | `3bfee8aad7c5` | `BenchS6MusicLong/20261106-000347-934cbc99/final/BenchS6MusicLong-piece.0-0.0.mp3` |

Sources: S4 shots
`BenchS4Montage/20261104-040255-9813688b/intermediate/BenchS4Montage-shot@{cold_open.2,deflect.3,react.4,button.5}-0.0.mp4`;
S3 take
`BenchS3DialogueWalk/20261104-035218-d2733a3a/final/BenchS3DialogueWalk-chained_keyframe_to_video_audio.0-0.0.mp4`
(frames 0-123).
