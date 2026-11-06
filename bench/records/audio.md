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

## S6 rerun (fixed templates, mini-ai)

2026-11-05. Bench server mini-ai:8766 (Apple M5 Pro, **MPS**, not lem's CUDA; the
server's worktree at `47a9e025`), after 237ee307 (music-long keeps a short section's
own length, `slice_audio` `past_end` "stop", `joined_short`) and 50e036fe
(narration-long defaults to 37 s takes at 3.0 words a second, paragraph-break
preference). Arms `bench/arms/audio-long-2.json`, record `audio-long-2.json` (the
"wrote no video" text is the driver's, as before). Same method as the first round,
run locally on the Mac on copies of the outputs. One draw per arm.

**Verdict: Task 22's gate passes on both halves.**

- **Narration, over 3 minutes and word-complete: passes.** Both seeds come back as
  7 takes, 202.9 s and 201.4 s. All 14 takes are word-complete under whisper-small
  and whisper-base, apart from spelling and homophones (pane/pain, harbour/harbor,
  toward/towards). Five of the six joins land on a paragraph break. The join level
  steps (-1.3 to +1.5 dB) sit inside the in-take spread. ECAPA finds no voice drift.
  So 37 s at 3.0 words a second holds on a second seed, where the first round had
  only one seed at 45.
- **Music, over 5 minutes and continuous: passes.** The pieces are 319.2 s and
  311.5 s at xf1500, and 314.7 s at xf3000. There is no silence of 0.5 s at -50 dBFS
  anywhere. Every join is music into music, and the outro is the song's own ending
  (the last lyric line, then a decay), not a cut.
- **Remaining defect (music):** a section that runs *longer* than `seconds` is cut
  at 85 s wherever that falls, and nothing warns unless the dropped remainder is
  under 5% of the slice. With 10-12 lines, three of the four sections in each draw
  ran past 85 s. The pass counts put the verses at about the 128 s ceiling. In
  s1166 the cut drops verse_1's last three lines. In s1167 the cut lands on
  verse_2's last line. Five of the six joins also drop 7-9 dB, as a section cut
  mid-song runs into the next section's quiet intro. That is inside the in-section
  maximum (6.3-12.7 dB), but well above its median (1.8-2.4 dB).

### Narration t37 (`max_seconds` 37, `words_per_second` 3.0)

`split_takes` gives 7 takes: 103, 109, 77, 88, 72, 83, 62 words. Take 1 ends
mid-paragraph ("...on the wooden stool to wait."). Paragraph 1 is 113 words, over
the 111-word limit, so its last sentence ("Then the light turned...") opens take 2,
and only a sentence pause leads into paragraph 2. The other five joins are paragraph
breaks: the preference worked wherever the limit allowed it.

| run (job) | file s | take s (words a second, by take length) | words heard / expected, small and base |
|---|---|---|---|
| t37 s1166 (`847ccb839875`) | 202.9 | 34.2, 36.7, 26.5, 29.8, 26.7, 29.1, 20.0 (2.70-3.10) | all 7 complete. small: pane/pain, lantern/"lanterns". base: pane/pain, its/it's, harbour/harbor |
| t37 s1167 (`4467e6380276`) | 201.4 | 34.2, 36.7, 26.6, 29.8, 26.4, 27.6, 20.3 (2.73-3.05) | all 7 complete. small: pane/pain, lantern/"lanterns", toward/towards. base: turned/turn, pane/pain, harbour/harbor (x2) |
| *first round:* t45 s1166 (`111a5623cba0`) | 198.0 | 6 takes | all complete |
| *first round:* t60 s1166 / s1167 | 179.7 / 178.3 | 5 takes | 2 of 5 takes lose words in each |

The longest take is 36.7 s of file (109 words), and it is complete on both seeds.
The voice still speaks at about 3.0 words a second on MPS.

**Joins** (found by the inserted digital silence, |sample| <= 2 LSB; pause is the
quiet run at -45 dBFS in 20 ms windows; the level step compares 5 s of speech on each
side):

| run | join at s | digital gap s | pause s | level step dB |
|---|---|---|---|---|
| t37 s1166 | 34.1 / 70.7 / 97.2 / 127.0 / 153.7 / 182.8 | 0.24-0.28 | 0.48 / 0.46 / 0.44 / 0.42 / 0.44 / 0.36 | +1.3 / -0.3 / -0.5 / +0.5 / -0.8 / +1.5 |
| t37 s1167 | 34.0 / 70.7 / 97.3 / 127.0 / 153.4 / 181.0 | 0.24-0.27 | 0.44 / 0.44 / 0.42 / 0.38 / 0.42 / 0.44 | +0.1 / +0.3 / +0.7 / -1.3 / +0.1 / +1.4 |

- Floor: in-take sentence pauses have a median of 0.60 s and 0.56 s (p90 0.78 and
  0.84 s). In-take 5 s level steps have a median of 0.7-0.8 dB, p90 1.6-1.8 dB and
  max 3.0-3.6 dB. Each file is -24.2 and -24.0 LUFS, true peak -4.5 and -4.9 dBFS,
  with no clipping.
- The bench still runs `gap_seconds` 0.3, so every join (0.36-0.48 s) is shorter
  than the median sentence pause, as in the first round. The 0.45 recommendation
  stands for Don's ear.

**Voice drift (ECAPA cosine):**

| run | take vs reference | adjacent takes | halves within one take (floor) |
|---|---|---|---|
| t37 s1166 | 0.81-0.86 | 0.91-0.96 | 0.91-0.95 |
| t37 s1167 | 0.81-0.86 | 0.90-0.96 | 0.90-0.95 |

The same take across the two seeds scores 0.97-0.98. The lowest adjacent pair is
the last join both times (0.91 and 0.90), against the shortest take. That is at the
within-take floor, so there is no measured drift.

### Music, 10-12 lyric lines a section (`bench/workflows/music-long-s6-full.json`)

| run (job) | file s | sections kept s | I LUFS / LRA | true peak dBFS |
|---|---|---|---|---|
| s1166 xf1500 (`ae2e1373cff6`) | 319.18 | 85, 85, 85, 68.69 (`joined_short`) | -18.2 / 11.9 | -2.0 |
| s1167 xf1500 (`f83cf3566727`) | 311.45 | 85, 85 (of 87.34; `slice_trimmed_tail` warned), 85, 60.95 (`joined_short`) | -17.3 / 10.4 | -2.0 |
| s1166 xf3000 (`ef47c7b6684d`) | 314.68 | as s1166 | -18.2 / 11.8 | -2.1 |

**Generated lengths:** the server keeps no intermediate file, so the lengths past
85 s are inferred. Each section's denoise passes are a multiple of 30 (30 steps
times about 4.2 s chunks), which fits all three known lengths (60.95, 68.69 and
87.34 s). On that model: verse_1 is 31 chunks in both draws (about 125-128 s, the
ceiling). chorus_1 is 25 chunks at s1166 (about 100-107 s). verse_2 is 26 chunks at
s1166 (about 104-112 s) and 31 at s1167 (about 125-128 s). chorus_2 ended early in
both draws (68.69 and 60.95 s). So the longer lyric overshoots in the verses
(8 verse lines plus a 2-line pre-chorus or bridge) and undershoots in the last
chorus.

**Silence:** silencedetect at -50 dBFS/0.5 s finds nothing in any piece. At
-40 dBFS/0.2 s it finds only the 0.28-0.33 s lead-in at 0 s, three 0.27-0.33 s dips
inside s1166's verse_1 (43-46 s, between phrases), one 0.27 s dip at 82.0 s, single
0.24 s dips at 196 s and 294 s, and each piece's last 1-2 s of decay. These are not
holes.

**Joins** (the level step compares the 5 s before the crossfade with the 5 s after
it; in-section floor = |5 s step| every 5 s inside sections):

| run | crossfade s | level step dB | in-section floor median / p90 / max |
|---|---|---|---|
| s1166 xf1500 | 83.5-85 / 167-168.5 / 250.5-252 | -8.8 / -8.9 / -9.4 | 2.4 / 6.9 / 12.7 |
| s1167 xf1500 | 83.5-85 / 167-168.5 / 250.5-252 | -0.8 / -7.8 / -7.4 | 1.8 / 4.3 / 6.3 |
| s1166 xf3000 | 82-85 / 164-167 / 246-249 | -10.2 / -4.7 / -7.2 | 2.4 / 6.9 / 12.7 |

The negative steps are a full band cut at 85 s running into the next section's own
sparse intro. Each section opens like a new song. The one flat join (s1167, 85 s)
is verse_1's instrumental tail into chorus_1's opening.

**What is sung at each cut** (whisper-small with word timestamps over the 20 s
before each crossfade; sung-lyric ASR is lossy, so read it as where the voice is,
not as a lyric check):

| run | section, cut at | last words heard before the cut | cut verdict |
|---|---|---|---|
| s1166 | verse_1, 85 | "Salt on the glass, I wipe it clear" ends at 80.8 | **lyric lost:** "The only work that matters here" and the 2-line pre-chorus are never sung |
| s1166 | chorus_1, 168.5 | "Every eight seconds, find them there" ends at 157.7 | complete; about 9 s of band after it, then cut mid-music |
| s1166 | verse_2, 252 | bridge sung by 240.2, then a repeat "Somewhere I'll save, somewhere I'll..." at 249.4 | **cut mid-phrase** (a repeated line, level -20 to -25 dB up to the cut) |
| s1167 | verse_1, 85 | pre-chorus "...comes to the door" ends at 77.5 | complete; band tail cut |
| s1167 | chorus_1, 168.5 | post-chorus repeat "...I'm turning for you" ends at 162.0 | complete. The dropped 2.34 s tail is band only, more than 6 s after the last word: **no phrase cut**, but the section's own ending is replaced by the 400 ms fade |
| s1167 | verse_2, 252 | "Somewhere a sail / a hand / a light..." at 241.8-249.8; "they understand" heard inside the crossfade (250.5-252) | **last line lands on the cut** |
| both | chorus_2, end | "One hundred stairs and twelve more still" ends at 315.6 / 308.8; level -33, then -54 dBFS within 1 s | **a real ending:** the lyric's last line, then a decay |

**Tempo** (`analyze_beats` per section, crossfades excluded; 72 BPM asked): only 4
of 8 xf1500 sections give an onset pulse. s1166 chorus_1 and verse_2 read 84.5 and
87.0 BPM (the same as the first round's s1166 sections 2 and 3). s1167 chorus_1 and
verse_2 read 92.3 and 77.9 BPM. The rest fall back to loudness peaks (83-140 BPM,
unreliable). Steering to 72 changes nothing. Music 3 does not hold 72, and the tempo
moves between sections by 3-14 BPM where it can be read.

**Key** (a whole-section chroma against Krumhansl profiles: crude, indicative only;
D minor asked). s1166 reads F major/D minor (the relative pair), Bb major/G minor,
D major, then D major/minor. s1167 reads F major, D major/B minor, C major (0.90),
then A major (0.80). The clearest key change at a seam is s1167's last join: the
10 s before it reads C major (0.95), and the 10 s after it reads A major (0.81).
That goes to the ear.

**xf3000 against xf1500:** the sections are sample-identical (the first 80 s differ
by 0.0), so only the joins differ. The file is 4.5 s shorter. The steps are -10.2 /
-4.7 / -7.2 dB against -8.8 / -8.9 / -9.4. A longer fade does not hide a band
dropping to an intro, and the intro still arrives.

**Wall time on MPS:** 40.2, 39.1 and 39.2 min per music job (four sections, encode
174-375 s and denoise 168-345 s per section, 0.4 s a pass). 4.1 and 3.8 min per
narration job.

**Fix options for the long-section cut** (not made here):
1. Give `audio_duration` less headroom than 1.5x (for example 1.1x), so a long
   lyric is squeezed rather than overrun. That is untested, and Music 3 may then
   end sections early again.
2. Have `slice_audio` (or music-long) warn when the dropped remainder of a
   `past_end` "stop" slice is loud. The current warning only covers remainders
   under 5% of the slice and 10 s, so a 40 s overrun is silent.
3. Size the lyric per section: here the 8- and 10-line choruses filled 61-107 s,
   while 8 verse lines plus 2 more ran to 104-128 s.

### Outputs (mini-ai `~/bench-workspace/outputs/`)

| arm | job | file |
|---|---|---|
| s6-music-full-335s-s1166 | `ae2e1373cff6` | `BenchS6MusicLongFull/20261106-024659-f8973042/final/BenchS6MusicLongFull-piece.0-0.0.mp3` |
| s6-music-full-335s-s1167 | `f83cf3566727` | `BenchS6MusicLongFull/20261106-032709-655afa0d/final/BenchS6MusicLongFull-piece.0-0.0.mp3` |
| s6-music-full-335s-xf3000-s1166 | `ef47c7b6684d` | `BenchS6MusicLongFull/20261106-040613-0ce22d5b/final/BenchS6MusicLongFull-piece.0-0.0.mp3` |
| s6-narr-long-t37-wps3-s1166 | `847ccb839875` | `BenchS6NarrationLong-t37/20261106-044527-c46db14f/final/BenchS6NarrationLong-t37-narration.0-0.0.wav` |
| s6-narr-long-t37-wps3-s1167 | `4467e6380276` | `BenchS6NarrationLong-t37/20261106-044934-a03a36cc/final/BenchS6NarrationLong-t37-narration.0-0.0.wav` |
