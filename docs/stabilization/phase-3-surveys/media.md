> Survey taken at `9e25c49a` (2026-09-30) for the Phase 3 plan. Line numbers are that commit's: re-verify before relying on them.

# Survey: media + DSP layer (worktree /Users/don/src/dkackman/dw-stabilization, branch stabilization/phase-3-plan)

Read-only survey. Paths relative to the worktree. Line numbers are current HEAD (9e25c49a).

## 0. Headline corrections to the brief

- A partial media layer ALREADY exists: `dw/media_audio.py` (266 lines), `dw/media_frames.py` (367), `dw/media_info.py` (398), `dw/loudness.py` (82). The refactor is a consolidation of these plus the stragglers in `dw/tasks/*`, not a greenfield layer.
- Pyloudnorm and scipy are ALREADY declared dependencies and already used (`dw/loudness.py:18-19`, `dw/tasks/audio_utils.py:14-15,20`). The "hand-rolled" audio_utils parts are the limiter curve, compressor envelope follower, RBJ biquad coefficients, PyAV-based resample, FFT helpers - not loudness metering and not the true-peak oversampling.
- NO ffmpeg subprocess anywhere. `subprocess` in dw/ is only `sysctl` (`dw/__init__.py:394`), `nvidia-smi` (`dw/server/sysinfo.py:38`), `pip` (`dw/server/updater.py:114`). dw_mcp/ has no PyAV/audio I/O at all.
- No bundled workflow uses the `teacache` key (see section 5).
- Functions >=150 lines in dw/ + dw_mcp/ (ast): create_app 3781, build_server 1280, Workflow.run 567, find_loop_bed 362, save_artifact 348, concat_videos 333, pipeline_flux_rf_inversion `__call__` 317, create_step_action 278, serve.main 259, plan.estimate 201, teacache `_create_flux_teacache_forward` 197 (nested `teacache_forward` 186), probe_media 184, attribute_voices 175, worker._handle_execute 165, gallery_frames 165, join_into_song 154. Note: those in the media set are docstring-heavy - see section 4 for body-only sizes.

## 1. PyAV / soundfile / scipy I/O inventory

Declared deps (pyproject.toml): `av>=18.1.0`, `soundfile>=0.14.0`, `imageio`, `imageio-ffmpeg`, `pyloudnorm>=0.2.0`, `scipy>=1.18.1`, `speechbrain`, `demucs`. Not declared: torchaudio (present in venv 2.11.0 only transitively via speechbrain's hard `Requires-Dist: torchaudio>=2.1.0`; demucs lists it only under the `train` extra), librosa (absent), pedalboard (absent). `docs/TASKS.md:1033` and `audio_utils.py:888` state "no torchaudio dependency" as a deliberate position.

### 1a. `av.open` sites (10 in dw/, none in dw_mcp/)

| # | Where | Function | R/W | Kind | What |
|---|---|---|---|---|---|
| 1 | dw/media_audio.py:49 | `media_duration` | R | probe | header duration |
| 2 | dw/media_audio.py:57 | `container_fps` | R | probe | video stream average_rate |
| 3 | dw/media_audio.py:69 | `audio_shape` | R | probe | duration/rate/channels |
| 4 | dw/media_audio.py:110 | `extract_audio` | R | audio decode | -> s16 WAV bytes via `AudioResampler(s16)` (:130) and `wave` (:214) |
| 5 | dw/media_audio.py:242 | `decode_soundtrack` | R | audio decode | -> float32 (ch, n) via `AudioResampler(flt)` (:253) |
| 6 | dw/media_frames.py:37 | `video_shape` | R | probe | w/h/fps/count from header |
| 7 | dw/media_frames.py:319 | `_read_frames` | R | video decode | selected frame indexes |
| 8 | dw/media_info.py:64 | `probe_media` | R | probe + audio decode | header + one decode pass (frame count, levels, LUFS, envelope) |
| 9 | dw/media_info.py:274 | `probe_metadata` | R | probe | header only / demux count (`_demuxed_frame_count` :223) |
| 10 | dw/tasks/video_utils.py:461 | `file_fps` | R | probe | DUPLICATE of media_audio `container_fps` (try/except -> None) |
| 11 | dw/tasks/video_utils.py:539 | `_decode_audio_video` | R | video+audio decode | frames -> PIL, audio -> fltp, `AudioResampler(fltp)` (:550) |
| 12 | dw/tasks/assess.py:137 | `read_media` | R | video+audio decode | gray thumbnails + float audio; imports private `_as_frame_samples` from media_info (:133) |
| 13 | dw/pipeline_processors/chain.py:283 | `_decode_segment` | R | video decode | segment file back to uint8 tensor (`container.decode(video=0)`, rgb24) |

Other PyAV: `av.AudioFrame.from_ndarray` + `AudioResampler` in `audio_utils.resample_waveform` (audio_utils.py:860-874) - the only rate conversion in the repo (write side of resample).

### 1b. Writers
- Video/audio-video encode: `dw/result.py` only, via diffusers `export_to_video`/`export_to_gif`/`encode_video`/`is_av_available` (imports :10-13; calls :906, :908, :1330, :1356, :1364). No direct PyAV encode in dw/.
- Audio encode: `dw/result.py:1738 write_audio` (chunked `soundfile.SoundFile`, vorbis segfault workaround), `normalize_audio` (result.py, torch->numpy, name collides with the task `normalize_audio` in audio_utils), `_audio_write_arguments` (:1373). `wave` for WAV bytes in media_audio.py:214.
- Audio read: `audio_utils.load_audio` (audio_utils.py:430): soundfile.read at :460 (http bytes) and :469 (local path); video extension -> `load_audio_video` (PyAV).
- Loudness/peak metering read side: `dw/loudness.py` (pyloudnorm `Meter`, `scipy.signal.resample_poly` 4x for true peak).

### 1c. Duplicated idioms (consolidation targets)

1. **Decode all audio frames to a float array** - written 5 times, each with its own dtype handling:
   - media_audio.py:253 `decode_soundtrack` (flt, packed -> `.reshape(-1, channels)`)
   - media_audio.py:130/170 `extract_audio` (s16, same layout rule)
   - video_utils.py:550 `_decode_audio_video` (fltp, (ch, n))
   - assess.py:137 `read_media` (native dtype -> `_as_float_samples` :113, then `_as_frame_samples`)
   - media_info.py:142-160 `probe_media` (inline copy of the same u/i -> float32 conversion that `assess._as_float_samples` has; `_as_frame_samples` :306)
   - Plus layout-name idiom `"stereo" if channels == 2 else ("mono" if channels == 1 else stream.layout.name)` twice in media_audio.py (:120-128 and :246-252), and `{1:"mono",2:"stereo"}.get(n, f"{n}c")` in audio_utils.py:866.
2. **Stream-rate/fps/duration from a container header** - `container_fps` (media_audio:53) vs `file_fps` (video_utils:453) vs inline in `video_shape` (media_frames:34), `probe_media` (media_info:64-90), `probe_metadata` (:274-295), `read_media` (assess:137-146), `_decode_audio_video` (video_utils:539-545), plus `_container_duration` (media_audio:33) vs inline `container.duration / av.time_base` (media_info:82, :292) vs `_stream_seconds` (assess:107) vs `audio_stream_seconds` inline (media_info:~89).
3. **dBFS / RMS / peak helpers** - 6 copies: `loudness._dbfs` (:33, imported privately by media_info:14), `audio_utils.level_dbfs` (:1248), `voice_attribution._dbfs` (:322), `assess._db`/`_rms`/`_peak` (:312-322), `loop_bed._db` (:81), `result._peak_dbfs` (:59).
4. **Resample-to-common-rate + warn** - concat_videos.py:177-228 and dissolve_videos.py:281-340 (`_dissolve_audio`) are near-identical blocks (log on one-unique-rate-but-pinned, `sample_rate_mismatch` warning, resample each). Fix: one helper in the media/dsp layer.
5. **Two "fit track to frame grid" implementations** - `audio_utils.fit_audio_to_frames` (:86, pads short join + warns) vs `video_utils._fit_audio_to_frames` (:585, trims/pads codec padding within 0.25 s, torch-or-numpy) plus `_sample_axis` (:623). Different policies, same shape; test imports `_fit_audio_to_frames` privately.
6. **Channel/axis normalisation** - `audio_utils.as_channels_samples` (:57), `media_info._as_frame_samples` (:306), `video_utils._sample_axis` (:623), `audio_utils._matched_channels` (:1415), `join_into_song._with_channels` (:322), `result.normalize_audio`/`_to_tensor`.
7. **`_round`** helper in loop_bed.py:85 and assess.py:335.
8. **Private cross-module imports** that signal mis-placed code: media_info -> loudness `_dbfs`; assess -> media_info `_as_frame_samples`; voice_attribution/audio_transcription -> audio_utils `_waveform_and_rate`; loop_bed -> audio_utils `_as_number`.

## 2. audio_utils.py concern map (2,229 lines) and library replacement

Constants first: DECLICK_MS (:37), SLICE_TRIM_* (:47-54), TONAL/HARMONICITY thresholds (:196-208), GAIN_LOOKS_LIKE_DB_ABOVE (:1043), MATCH_* (:1232-1245), LIMITER_* (:1588-1610), COMPRESS_MODES (:1932), FILTER_KINDS (:2043), `_SPECTRAL_BANDS` (:2153).

| Concern | Functions (line spans) | ~Lines |
|---|---|---|
| Channel/shape plumbing | `as_channels_samples` 57-85, `_matched_channels` 1415-1427, `_waveform_and_rate` 1863-1905, `_warn_on_rate_override` 1906-1931, `_as_number` 508-517, `_as_track` 475-507, `_track_names` 938, `_load_tracks_matching_rate` 951-1005 | ~250 |
| I/O | `load_audio` 430-474 (soundfile + PyAV delegate) | 45 |
| Frame-grid fitting | `fit_audio_to_frames` 86-145, `slice_samples` 146-160 | 75 |
| Slicing | `slice_audio` 518-633, `_warn_on_slice_past_end` 760-795, `_warn_on_slice_trims_tail` 796-841 | ~320 (mostly warnings) |
| Gain / level / match | `gain_audio` 634-759, `level_dbfs` 1248-1271, `match_levels` 1272-1350, `warn_on_level_spread` 1351-1379 | ~280 |
| Fades / declick / crossfade / join | `equal_power_crossfade_join` 161-195, `crossfade_concat` 391-429, `_equal_power_ramps` 1380, `_declick_join` 1386-1414, `fade_audio` 1428-1469, `_fade_curve` 1855, `crossfade_audio` 1006-1042 | ~250 |
| Bleed join + tonal analysis | `_spectral_flatness` 211-235, `_harmonicity` 236-273, `bleed_join` 274-390 | ~180 |
| Resample | `resample_waveform` 842-883, `resample_audio` 884-937 | ~95 |
| Mix / loop | `mix_audio` 1046-1137, `loop_audio` 1138-1231 | ~190 |
| Normalize (peak/LUFS) | `normalize_audio` 1470-1587 | 118 |
| Limiter + LUFS search | `_true_peak_envelope` 1613-1636, `_limiter_curve` 1637-1676, `_limit_at` 1677-1695, `_search_gain` 1696-1741, `_normalize_limited` 1742-1854 | ~240 |
| Compressor/limit/gate | `compress_audio` 1940-2012, `_follow_envelope` 2013-2034, `_time_constant_coef` 2035-2042 | ~105 |
| Biquad filters | `filter_audio` 2046-2086, `_biquad_coefficients` 2087-2119, `_apply_biquad` 2120-2152 | ~105 |
| Analysis | `analyze_audio` 2160-2198, `_spectral_balance` 2199-2229 | ~70 |

A natural split: `dw/audio/` or `dw/dsp.py` (limiter/compressor/biquad/resample/analysis/levels, pure numpy, no events) vs `dw/tasks/audio_*` (task commands that take `audio`, call `check_arguments`, `emit_log`, `_as_track`). Today the task wrappers and the DSP live interleaved in one file.

### Per-primitive build-vs-buy

| Primitive | Current | Candidate library | Dependency status | Behaviour risk / what tests pin |
|---|---|---|---|---|
| Integrated LUFS (BS.1770) | `loudness.integrated_lufs` -> `pyloudnorm.Meter` | already pyloudnorm | PRESENT (declared) | Nothing to replace. `_search_gain` calls it up to 8x per normalize; tests `test_audio_utils.py:656,669,831,838` use abs=0.5 LU. |
| True-peak measure | `loudness.true_peak_dbfs` and `audio_utils._true_peak_envelope` (:1613) both `scipy.signal.resample_poly` 4x | already scipy; pyloudnorm has no true-peak | PRESENT | Two implementations of the same oversampling (`loudness.py:77` whole-array vs `_true_peak_envelope` blocked 2**18 with 32-sample overlap). Merge into one blocked function; safe. `_true_peak_envelope` imported privately by `test_audio_utils.py:1017`. |
| Resample | PyAV `AudioResampler` (swresample) in `resample_waveform` | `scipy.signal.resample_poly` (present) or `soxr` (new) or `torchaudio.functional.resample` (transitive only) | scipy PRESENT; soxr/torchaudio would be new or undeclared | Result differs numerically (swresample default filter vs polyphase Kaiser). Tests `TestResampleAudio` (:463) check lengths/levels, not samples (`test_audio_utils.py:496-498` rel=0.01). `test_media_audio.py:232` patches `AudioResampler.resample` but for the decode path, not this. Note #140: PyAV accepts rate 0 silently - guarded at :852; scipy would need `up/down` via `Fraction`. Low risk, modest win (removes ~30 lines + one PyAV frame hack). Do NOT pick torchaudio: undeclared, in maintenance mode, and the project documents "no torchaudio dependency". |
| RBJ biquad (lp/hp/bp/notch) | `_biquad_coefficients` 33 lines + `_apply_biquad` (lfilter with a pure-Python fallback, 33 lines) | `scipy.signal.iirnotch` (notch = RBJ notch exactly), `scipy.signal.butter(2, ...)` / `iirfilter` (lp/hp match RBJ only at Q=0.7071; filter_audio exposes free `q`), `scipy.signal.sosfilt`; `torchaudio.functional.{lowpass,highpass,bandpass,bandreject}_biquad` (same RBJ formulas, takes Q) | scipy PRESENT; torchaudio transitive | scipy has no RBJ lowpass/highpass with arbitrary Q, so only notch is a drop-in; keep the 30-line coefficient table or use torchaudio's (new undeclared dep, tensor round trip). The scipy-missing fallback is dead code: scipy is a declared dependency; `tests/test_audio_utils.py:1700-1719` (`without_scipy` patching `builtins.__import__`, asserting fallback within 1e-9 of lfilter) pins a now-pointless two-implementation contract. DELETE the fallback and the test together. Real win: ~35 lines. |
| Compressor / limit / gate | `compress_audio` + `_follow_envelope` (per-sample Python loop over `tolist()`, asymmetric attack/release) + `_time_constant_coef` | `pedalboard` (Compressor, Limiter, NoiseGate) - NEW dep, binary wheels, GPL-3.0 (license conflict with Apache-2.0 project - must flag); `torchaudio` has no compressor; scipy cannot do branchy one-pole in C (could use `lfilter` only if attack==release). `numba` would be new. | pedalboard NEW, absent from venv | Algorithm differs (pedalboard compresses with its own smoothing on gain, not on linked peak envelope). Tests pin exact behaviour: `test_audio_utils.py:1584` (compressed == expected_gain abs 1e-5), `:1602` (limit holds at -6 dB 1e-5), `:1621`, `:1640` (==0.1 abs 1e-6). These would fail against pedalboard; they'd have to be rewritten as tolerance tests. Also performance: current loop is Python on ~8M samples for 3 min. Honest call: a maintained drop-in does not match the semantics (linked-channel detector, "limit" = hold at threshold); the defensible "buy" is pedalboard but with a licence and a numeric-behaviour change. Cheaper alternative that keeps the algorithm: vectorise the envelope follower (separate attack/release is expressible with `scipy.signal.lfilter` per state only approximately) or keep and move. Recommend: keep, relocate, leave for a later decision. |
| True-peak look-ahead limiter | `_limiter_curve` 1637-1676 (sliding min via `scipy.ndimage.minimum_filter1d` x2, linear-dB release via cumulative minimum, boxcar), `_limit_at`, `_search_gain` (regula falsi to within `LIMITER_TOLERANCE_LU`=0.1 LU, <=8 passes), `_normalize_limited` | pedalboard `Limiter` (sample-peak, not true-peak, no look-ahead param, no LU search), `pyloudnorm` (no limiter) | no drop-in exists in any library | Already leans on scipy/pyloudnorm for the heavy parts. Tests pin behaviour: `test_audio_utils.py:1007-1011` (constants `LIMITER_LOOKAHEAD_MS==5.0`, RELEASE 150, HOLD 20, MAX_REDUCTION 12, HEAVY 6), `:1017-1025` (`_limiter_curve` + `_true_peak_envelope` private imports, curve value at a point == 1.0), `:825,:892,:924,:927` (reaches target LUFS under true-peak ceiling, searched gain, shortfall, 12 dB cap), CLAUDE.md "audio headroom" bullets. KEEP. Move verbatim to dsp module; expose `_limiter_curve`/`_true_peak_envelope` under their public names and update the 2 test imports. |
| Spectral flatness | `_spectral_flatness` 211 | `scipy.stats.gmean` for the geometric mean; librosa/`scipy.signal` spectrogram-based flatness not equivalent (this is a single-FFT, native-Nyquist-limited variant, #198) | scipy PRESENT; librosa NEW | Small; `_PERIODICITY_*` and thresholds are tuned (`test_audio_utils.py:363`). Keep. |
| Harmonicity (autocorr peak) | `_harmonicity` 236 (FFT autocorrelation) | `scipy.signal.correlate(method="fft")` replaces the hand FFT/irfft block (~8 lines) | PRESENT | Same numbers if padded the same; `_harmonicity` imported privately by tests. Minor win. |
| Spectral balance | `_spectral_balance` 2199 (rfft + Parseval bands) | `scipy.signal.welch` | PRESENT | Different estimator; test asserts band power == rms (abs 0.05 dB, `:1825`) which relies on exact Parseval. Keep. |
| Ducking ramp / equal-power crossfade / fades / mix / loop / slice | numpy 5-15 line curves | none worth buying | - | `join_into_song._ducked` (join_into_song.py:365) is a linear gain ramp; tests pin exact samples (`test_audio_utils.py:122-123,279-302,338,526-529`). Keep. |
| Loudness match (`match_levels`) | numpy peak/rms | pyloudnorm (LUFS) is already an option via `normalize_audio`; `match_levels` intentionally uses peak/rms | - | Leave. |
| Envelope second-by-second (probe) | `media_info._fill_envelope` 323-388 | none | - | Leave; relocate. |

Conclusion for item 2: the honest build-vs-buy deletions are small. Safe: delete `_apply_biquad` pure-Python fallback (+ its test), unify the two true-peak oversamplers, use `scipy.signal.iirnotch` for notch (and optionally `resample_poly` for resample), use `scipy.signal.correlate`/`scipy.stats.gmean` in the tonal tests. Not buyable without behaviour change: the compressor (pedalboard: new dep, GPL-3.0) and the true-peak look-ahead limiter with the LU search (no library). PyAV stays for decode; soundfile stays for file I/O.

## 3. Duplication between other tasks modules

### concat_videos.py <-> dissolve_videos.py (13 shared commits, 51% coupling)
Both load inputs, check sizes, reconcile audio sample rates, match levels, crossfade/join the track, fit to frame grid, then build a shot map. The shared logic, line for line:
- Load: `[load_audio_video(v) if is_video_location(v) else v for v in videos]` - concat:162, dissolve:107.
- Size check `check_same_frame_size` - concat:164, dissolve (after `frames_as_array`).
- Names: `video_names` defined in concat_videos.py:39, imported by dissolve (and used by join_into_song:124) - lives in the wrong module.
- Sample-rate reconcile: concat:177-228 (log / `sample_rate_mismatch` warning / `resample_waveform` list-comp) vs dissolve `_dissolve_audio`:~300-340. Message text is identical except the command name.
- Level matching: `match_track_levels` if `match_levels` else `warn_on_level_spread` - concat:230-233, dissolve `_dissolve_audio` ~340-350.
- Audio join: `crossfade_concat` (dissolve) vs `equal_power_crossfade_join` / `bleed_join` in a loop (concat); both end in `fit_audio_to_frames(joined, sample_rate, total_frames, written_fps, command)`.
- `written_fps = fps or next((v.fps for v in videos if getattr(v,"fps",None)), None)` - concat:375, dissolve:~130.
- Shot map: `shot_record` / `nested_shots` / `measured_num_samples` from `dw/shots.py`; concat inner loop (:255-371) builds `video_shots` with `trimmed_shots`, dissolve `_dissolve_shots` (:170-258) builds with `frame_starts`; both set `source_index` and `overlap_frames`/`hard_cut`.
Suggested extraction: `prepare_inputs(videos, command)` -> (videos, names, clips), `reconcile_rates(videos, waveforms, sample_rate, command)` -> (waveforms, rate), `level_tracks(waveforms, match_levels, dbfs, command)`, and `finish_join(frames, audio, shots, ...)`. Tests reaching these: `test_concat_videos.py:102,612` patch `dw.tasks.concat_videos.load_audio_video` (target name must stay importable in the module or the patch target changes), `test_dissolve_videos.py`, `test_shots.py`, `test_rule_parity.py`.

### loop_bed.py (758 lines) - audio analysis
`_envelope` 229, `_occupied_rate` 251, `_tonal_blocks` 288, `_looped` 311, `_db` 81 duplicate audio_utils concepts: it imports `_spectral_flatness`, `_harmonicity`, `_PERIODICITY_*`, `_as_number`, `_waveform_and_rate`, `crossfade_concat` from audio_utils, and reads the soundtrack via `media_audio.decode_soundtrack` (:108-111, a fast-path that avoids `load_audio_video`, added in #218) with `_waveform_and_rate` fallback (:118, :120). `_read_source` (:89-129) is the I/O seam.

### join_into_song.py (412)
Reuses the same load/fps/size-check prefix as concat (`video_names`, `frames_as_pil_list`, `check_same_frame_size`, `_one_fps` ~ concat's written_fps), plus `_match_loudness` 334 (pyloudnorm via `integrated_lufs`, `MIN_LUFS_SECONDS`), `_ducked` 365, `_placed_song` 387, `_with_channels` 322 (dup of `audio_utils._matched_channels`), `_song_track` 256 (`load_audio` + resample_waveform :304).

### voice_attribution.py (778)
Pure analysis: `_dbfs` 322 (dup), `voiced_mask`/`_Voicing` 336-381, separation via demucs (`_run_separator` 537, `separate_vocals` 558), `_mono_16k` 581 (`resample_waveform` + `.mean`), `embed` 590 (speechbrain ECAPA). Audio DSP is only resample + mono mix; model loading is torch/speechbrain/demucs (keep). `speech_generation.py:72` and `audio_transcription.py:78` do the same "load -> resample to model rate -> mono" as `_mono_16k`; one `to_mono_at(rate)` helper covers all three.

### assess.py (910)
Own decode (`read_media` 123) and own helpers `_db/_rms/_peak/_round` (312-335), `_band_shares` 603 (FFT band energy, similar to `audio_utils._spectral_balance`), `_dead_air` 339, `_seam_audio` 627. Should sit on the shared decode/level helpers rather than its own.

### pair_audio.py (367)
`_fit_to_video` 115-245 (fit policy) duplicates pieces of `fit_audio_to_frames` and `slice`/pad logic; `_Loaded` 97 wraps `load_audio`; `_regridded_drift` 40.

### video_utils.py (637)
Mixed concerns: frame helpers (`get_frame`, `loop_frames`, `frame_grid`, `FrameList` 426) + I/O (`load_audio_video` 471, `_decode_audio_video` 530, `file_fps` 453) + audio fit (`_fit_audio_to_frames` 585, `_sample_axis` 623). `frame_grid`/`_grid_tile`/`_compose_grid` (221-337) overlap `media_frames.contact_sheet`/`_tile`/`_stamped_tile` (94/265/276) - two contact-sheet implementations (one on in-memory frames, one streaming from file).

## 4. Long functions - section outlines (line spans in file; bodies exclude docstring)

### probe_media (dw/media_info.py:19-202; 184 lines, docstring 19-62, body 63-202 = 140)
1. 63-67 open container or return None.
2. 68-90 pick streams, build `info` (kind/fps/width/height/duration/sample_rate/channels/audio_stream_seconds).
3. 91-99 decide `need_frame_count` (frames==0) and whether a decode is needed at all.
4. 100-132 set up accumulators: peak/total/count, `bins` (envelope), `max_envelope_samples`, `lufs_chunks`, `max_lufs_samples`, choose streams.
5. 133-178 the decode loop: video frame count; audio frame -> float conversion (DUP of `assess._as_float_samples`) -> running peak/sum, `_fill_envelope`, LUFS chunk trimming (past duration = codec padding, #277).
6. 179-188 post-loop: merge trailing fragment; except -> return partial `info`.
7. 189-202 fill `frame_count`, `peak/mean/lufs/true_peak` and `envelope`.
Cut plan: `_stream_info(container)` (2), `_AudioAccumulator` class (4+5 audio half: `.add(frame)`, `.finish()` -> levels/LUFS/envelope), shared `to_float32(samples)` (5); probe_media becomes ~45 lines. The accumulator is also what `assess.read_media` and `decode_soundtrack` need.

### concat_videos (dw/tasks/concat_videos.py:59-391; 333 lines, docstring 59-141, body 142-391 = 250)
1. 142-155 validate list; single-input bleed warning.
2. 159-164 names, load inputs, frame lists, same-size check.
3. 165-176 per-input waveforms.
4. 177-228 sample-rate reconcile (log/warn/resample) - 52 lines, DUP with dissolve.
5. 230-233 match levels / level-spread warning.
6. 235-253 accumulator setup (`has_audio`, `silence_channels`).
7. 255-371 per-input loop (117 lines): (a) shot records incl. nested/hard_cut 255-296; (b) silence fill or per-input fit+drift warning 297-346; (c) first-audio vs join branch: bleed_join or equal_power_crossfade_join 347-371.
8. 373-391 written_fps, fit_audio_to_frames, `measured_num_samples`, return.
Cut plan: `_reconcile_rates` (4), `_shot_records_for(video, index, ...)` (7a), `_input_waveform(...)` (7b: silence fill / fit + drift warn), `_join_seam(...)` (7c); loop shrinks to ~30 lines.

### find_loop_bed (dw/tasks/loop_bed.py:397-758; 362 lines, docstring 397-469, body 470-758 = 289)
1. 470-503 coerce 12 numeric args (`_as_number` x12) + `check_arguments` + min>max guard.
2. 505-514 read source, resolve shots, mono mix, duration check.
3. 516-540 search-window bounds validation (5 raise blocks).
4. 542-560 bins: offset, `_envelope`, `_shot_spans`, `bin_shot`, shortest/longest, thresholds, `rejected` tally.
5. 562-583 tonal block scales (15 lines).
6. 586-597 edge-tick guard arrays (`tails/heads/before/after`).
7. 599-666 vectorised window rejection loop over lengths (67 lines): shot-boundary, loud, silent, ticked, tonal, survivors with readings.
8. 670-709 rank survivors, thin overlaps, build `candidates` via `_looped` (27-line loop 683-709).
9. 710-758 sort, `criteria` dict, findings, log, return dict (26 lines).
Cut plan: `_coerce_arguments` (1), `_search_window` (3), `_tonal_scales` (5), `_edge_ticks` (6), `_scan_windows` (7, the big one, returns survivors + rejected), `_rank_and_loop` (8), `_answer` (9). Top level ~60 lines.

### join_into_song (dw/tasks/join_into_song.py:56-209; 154, docstring 56-99, body 100-209 = 110)
1. 100-122 validate args/types. 2. 124-136 load, names, frame lists, fps, song track, channels. 3. 138-158 dialogue loop (`_dialogue_track`, shots). 4. 160-172 song-shot loop. 5. 173-193 cue check, `song_entry`, mix with `_ducked`/`_placed_song`. 6. 195-209 shots measured, log, return. Already under 150 body lines once the docstring is counted out; cut 1-2 (`_validated`) and 3-4 (`_place_shots`) to be under 150 total including docstring.

### attribute_voices (dw/tasks/voice_attribution.py:604-778; 175, docstring 604-653, body 654-778 = 125)
1. 654-660 check args, load waveform. 2. 662-683 parse voices/lines/windows (+ `clip_duration` closure). 3. 683-692 dtype, `stem` closure (separate vocals), `_Voicing`, load encoder. 4. 694-737 embed references, similarity, too-similar warnings (44 lines). 5. 739-755 attribute each line. 6. 756-766 roll up windows. 7. 768-778 return dict. Cut plan: `_embed_references` (4), `_attribute_lines` (5-6), keep 1-3/7 in the command. Closures `clip_duration`/`stem` become module functions (they close over `clips`, `separate`, `device`, `dtype`).

### teacache (dw/teacache.py:89-285) - see section 5; one 197-line factory with a 186-line nested forward.

## 5. teacache.py

- What: 381 lines. A context manager (`teacache_context` :301) that monkey-patches `transformer.forward` (or `_old_forward` when accelerate's AlignDevicesHook is present - lines 340-357 handle the accelerate hook interaction) with `teacache_forward`, a copy of `FluxTransformer2DModel.forward` with a polynomial-rescaled relative-L1 gate (coefficients in `dw/teacache_models.json`). Adapted from ali-vilab/TeaCache via Teriks/dgenerate. Only Flux has a forward factory (`_FORWARD_FACTORIES`); the JSON registry also lists HunyuanVideo, Mochi, LTX-Video, CogVideoX, Lumina2, Wan2.1 "pending" (docs/ACCELERATION.md:155).
- Wiring: `dw/pipeline_processors/pipeline.py:17,566-582` (`configuration.teacache` block); schema `dw/workflow_schema.json:802` (description :738 says cache is "mutually exclusive with teacache").
- Who uses it: NO workflow. `grep '"teacache"'` over workflows/, dw/workflows/, plugins/, prompts/ finds nothing; `workflows/templates/step-caching.json:18` uses `"cache": {"type": "first_block"}`. Only docs (docs/ACCELERATION.md:126-182, docs/WORKFLOW_GUIDE.md), `tests/test_teacache.py` (300 lines, exercises accelerate-hook composition, duplicate-timestep error, forward restore; imports private `_create_flux_teacache_forward`).
- Diffusers equivalent in installed 0.41.0.dev0 (`venv/lib/python3.14/site-packages/diffusers/hooks/`): `first_block_cache.py` (`FirstBlockCacheConfig`, `apply_first_block_cache`; its docstring :199 says it "builds on the ideas of TeaCache", simpler), `mag_cache.py` (`MagCacheConfig`, `apply_mag_cache`), `taylorseer_cache.py` (`TaylorSeerCacheConfig`, `apply_taylorseer_cache`), `faster_cache.py`, `sea_cache.py`, `pyramid_attention_broadcast.py`, `text_kv_cache.py`. No class named TeaCache upstream (grep found only the docstring mention). dw already routes `configuration.cache.type` in {first_block, faster, text_kv, mag, taylorseer} through `config_objects.py:229-280`.
- Verdict for the plan: behaviour is not identical (TeaCache rescales by a fitted polynomial; FBC uses raw first-block L1; MagCache uses calibrated magnitude ratios with flux preset `"mag_ratios": "flux"` per ACCELERATION.md:75), but it is Flux-only, unreferenced by any shipped workflow, and ACCELERATION.md:182 already steers users to `first_block`/`mag`. Deleting `dw/teacache.py`, `teacache_models.json`, `tests/test_teacache.py`, the pipeline.py branch, the schema `teacache` property and the doc sections removes ~381+300+JSON lines and the 197-line function; users lose the `rel_l1_thresh` knob with Flux-specific speed table. Needs a product decision (breaking change to a documented workflow key; bump/release-note).

## 6. Test coupling

String `patch("dw....")` targets (counts):
- `dw.result.is_av_available` 13, `dw.result.export_to_video` 13, `dw.result.encode_video` 13 (test_result.py, test_concat_videos.py:172-174, test_join_into_song.py:394 ...) - tied to the result.py encode imports, not to the media layer.
- `dw.media_info.probe_media` 7 (all in tests/test_result.py: 1988, 2036, 2059, 2070, 2160, 2189, 2266) - patched at the source module because result.py imports it lazily inside the function (`result.py:181`). If probe moves, the lazy-import site and these 7 targets move together.
- `dw.tasks.voice_attribution._load_embedder` 6, `.embed` 6, `.load_audio` 2, `.emit_warning` 2 (tests/test_voice_attribution.py).
- `dw.result.warn_if_written_above_full_scale` 2.
- `dw.media_audio.av.open` 2 (test_media_audio.py:300,316), `dw.media_frames.av.open` 1 (test_media_frames.py:451).
- Also string: `dw.tasks.concat_videos.load_audio_video` (test_concat_videos.py:102,612), `dw.tasks.video_utils.load_audio_video` (test_find_loop_bed.py:530, monkeypatch).
Patching the library rather than a dw module (survive a move, but assert decode counts): `patch.object(av.container.InputContainer, "decode", counting_decode)` at test_media_audio.py:123,256; test_media_frames.py:258; test_server.py:1654,1685,1694,5988; test_media_info.py:404 (decode boom), :471 (`demux` boom); `patch.object(AudioResampler, "resample", ...)` test_media_audio.py:232; `monkeypatch.setattr(av, "open", counting_open)` test_admission.py:640. These pin "decode once / header only" behaviour (Phase 2b B9) - the new media layer must keep a single `av.open` per call or these counters change meaning.
Other monkeypatch targets: `assess_module.emit_warning` (test_assess.py:568,876), `media_frames._read_frames` (:406) and `video_shape` (:287).

Private names tests import (must be renamed/re-exported or tests edited):
- audio_utils: `_PERIODICITY_MAX_HZ`, `_PERIODICITY_MIN_HZ`, `_as_track`, `_declick_join`, `_harmonicity`, `_limiter_curve`, `_true_peak_envelope`; plus attribute access `audio_utils._biquad_coefficients`, `audio_utils._apply_biquad`, `audio_utils.LIMITER_*` (test_audio_utils.py:1007-1011, :1700-1719).
- video_utils `_fit_audio_to_frames`; loop_bed `_occupied_rate`; media_frames `_read_frames`; voice_attribution `_load_embedder`, `_load_separator`; teacache `_create_flux_teacache_forward`.
Importing files (modules -> test files):
- audio_utils: test_audio_utils, test_join_into_song, test_mix_audio, test_concat_videos, test_find_loop_bed, test_rule_parity, test_shots, test_dissolve_videos, test_assemble_and_score_target_lufs, test_assessment_rules, test_task_domains, test_security_ssrf, test_shaping_logs (13).
- media_info: test_slice_preflight, test_shot_span_preflight, test_dissolve_frame_errors, test_video_size_errors, test_media_info (5-6). media_audio: test_media_audio, test_find_loop_bed. media_frames: test_media_frames, test_server. loudness: test_join_into_song, test_assemble_and_score_target_lufs, test_audio_utils. video_utils: test_catalog_structure, test_video_utils, test_rule_parity, test_security_ssrf. concat_videos: test_assess, test_concat_videos, test_shots. dissolve_videos: test_assess, test_rule_parity, test_dissolve_videos, test_shots. loop_bed: test_find_loop_bed. join_into_song: test_join_into_song, test_plugin_skills. voice_attribution: test_voice_attribution. assess: test_assess, test_assessment_rules. pair_audio: test_pair_audio, test_pair_audio_fit, test_video_utils, test_shots, test_modular_output_properties. teacache: test_teacache.
Numeric pins to respect when swapping DSP: compressor/limit/gate exactness (test_audio_utils.py:1584,1602,1621,1640), LUFS targets abs=0.5 (:656,669,831,838,892), limiter constants and curve (:1007-1025), equal-power/fade/declick exact samples (:122-123,279-302,338,526-529), biquad scipy-vs-python parity (:1716), spectral Parseval (:1804-1825), `test_plugin_skills.py` pins skill numbers to diffusers symbols (plugin text that quotes LIMITER/-3 dBFS must stay consistent).

## 7. Suggested plan inputs (derived)

1. Create `dw/media/` (or `dw/media.py`): `open_media` context, one `decode_audio(path, fmt, start, duration)` generator, `to_float32`, `stream_info`, `container_duration`, `container_fps`; fold media_audio, media_frames, the probe half of media_info, and video_utils' `file_fps`/`_decode_audio_video`, assess `read_media`, chain `_decode_segment` onto it. Keep one `av.open` per call (tests count them).
2. Create `dw/dsp.py` (pure numpy/scipy; no events/check_arguments): resample, true-peak + limiter + gain search, compressor, biquad, levels (`dbfs`, rms, peak), spectral flatness/harmonicity/balance, crossfade/fade curves. `audio_utils.py` keeps only task commands (~40% of today's size).
3. Deletions with no behaviour change: `_apply_biquad` fallback + test; duplicate `_dbfs` x5; duplicate true-peak oversampler; dup rate-reconcile block; `file_fps`.
4. Needs a decision: delete TeaCache (unused, Flux-only, diffusers ships first_block/mag/taylorseer); pedalboard is NOT recommended (GPL-3.0, semantics differ, tests pin exact numbers).
