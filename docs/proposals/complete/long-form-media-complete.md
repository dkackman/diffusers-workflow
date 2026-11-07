# Long-form media: one playbook, measured levers, quality-first defaults (#1199)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-06). Don approved the design (`docs/proposals/long-form-media.md`) and the
plan (`docs/superpowers/plans/2026-11-03-long-form-media.md`) on 2026-11-03. Unlike
most features, it was not built through the harnest implementer. Don's own session
built it on `feat/long-form-media` with subagents, and benched every lever on a second
`dw.serve` on lem (port 8766, workspace `~/bench-workspace`), plus mini-ai (MPS,
port 8766) for LTX-2.5 and the S6 audio rerun. The full working log, every ruling (R1-R44)
and every deferred minor are in the SDD ledger,
`.superpowers/sdd/2026-11-03-long-form-media/progress.md`. The records are under
`bench/records/`. This file records what shipped, what the bench found, what lost,
and what is still open.

## The problem

An agent asked for a minute, five minutes, an episode or a music video should treat
the 15-second clip as a render detail. It should block the piece, choose a length and
a continuity per scene, get past the clip ceiling, and handle what goes wrong during
the run, without the requester knowing what a seam, a guide or an anchor is. Before
this work, each chain lever was an unmeasured opt-in, 18 ideas on the subject were parked, and
nothing told an agent which lever to use for which shot.

## Verdict, in one paragraph

**The default chain changed and became shot-aware. The structural rewrites lost.**
`templates/minimax/chained-segments` now defaults to `guide` continuity (a
`breaking-change`). A `shot` preset picks the drift counters: `static` gets anchor
0.25, `static_long` gets keystone slot `last`, and `moving` and `dialogue` get nothing.
That changed the two-minute static take from a mosaic by 40 s to a usable take at
59 min of wall time. Several new behaviours shipped: a per-segment check with a bounded
re-roll, `duration_seconds`, repair in place, an audio-only pass, long narration and
long music, OTIO export, and a `long-form` playbook skill. Latent carry, hybrid
windows, the late capstone, SelfLift, TST, hand-over backoff, sharpest-of-N, guide
`look_hold` and the anti-burn LoRA were all measured and lost (or tied) as defaults.
The windowed refine seam pass won on seams, but it is not a burn fix and costs about
36 min per minute of take, so it stays dark behind the experimental flag.

## How it was measured (the bench)

- **Server and driver.** A second `dw.serve` runs on lem, managed by
  `scripts/bench_server.sh` and described in `bench/README.md`. The driver
  (`bench/driver.py`) submits the arms in `bench/arms/*.json` and measures every
  output through the server's own `compare` and `assess` routes, so the tester can
  repeat any measure over MCP. Verdicts come from `bench/verdict.py`.
- **Scenes** (`bench/workflows/`, `bench/scenes.json`):
  - S1 static portrait, locked off;
  - S2 moving arc around a statue;
  - S3 dialogue walk with a held voice;
  - S4 a four-shot montage under one bed;
  - S5 an LTX-2.5 courier chain;
  - S6 narration and music.

  S1 and S2 reuse #1175's `drift-chain` scenes, so its numbers are a known baseline.
- **Bench settings.** The bench runs 6 segments (26.4 s). The check in
  `noise-floor.md`, *Cheap-setting check*, showed that 6 segments reproduce #1175's
  10-segment ordering on every row. Two-minute takes (2,880 frames, 28 seams) were
  run for the static case.
- **Measures.** Lower is better for all of them:
  - seam jump and look (`analyze_seams`);
  - the soundtrack level step;
  - texture drift `|ln(band / band@segment 1)|` on a subject-free crop, in fine, mid
    and coarse bands (ruling R8).

  The level step counts only on scenes with a held voice or a music bed (S3, S4, S6).
  On S1 and S2 it is room tone. The same-seed rerun moved S1's level step by 4 dB with
  the picture unchanged, so S1/S2 were read on picture.
- **Noise floor (R35, superseding R26).** The floor is
  `combined_floor(seed floor, same-seed rerun gap)`. The rerun gap is the maximum over
  8 pairs: look 0.438, level 6.2 dB mean and 8.8 dB max, jump 0.124, texture 0.068.
  `verdict()` returns `wins` only when the primary measure beats the control past the
  floor and nothing else is worse past it. `bench/segment1.py` labels every record
  `clean` or `drifted` by a hash of its segment 1. A read counts only clean against
  clean.
- **Determinism (`determinism.md`).** Same-seed runs on a two-card server diverged
  because of `cudnn_benchmark` autotuning while the other card's worker was busy.
  Single-card runs were 16/16 bit-exact. With `cudnn_benchmark: false` and
  `cudnn_deterministic: true` it was 14/14 canonical, at no measurable cost on H3
  (ruling R36). lem's `~/.diffusers_helper/settings.json` carries that setting now.
  The file is shared with the main server (see *Follow-ups*).
- **Ear and eye.** Claude cannot judge sound, so every timbre or sync call went to
  Don with job ids (`bench/records/for-don.md`). The eye checks used contact sheets and
  side-by-side exports to Don's Mac.

## What shipped

### Defaults, and the measurements that chose them

| Default | Where | Evidence |
|---|---|---|
| **`continuity: guide`** for every H3 chain shot type (was `last_frame`; `breaking-change`) | `minimax/chained-segments` (18df52f4, R33) | `counters.md` *Defaults by shot type*. Every guide-vs-last_frame read was mixed or favoured guide. last_frame counters froze or held back the S2 camera (`242358078691`, `a3d6a044fd6b`). Clean last_frame controls: `look_hold` ties its own control on texture at both seeds. |
| **`shot`** preset: `static`, `static_long`, `moving` (default), `dialogue` | `minimax/chained-segments` (23196f7b, d555a373) | Below. `shot` is a `presets` selector, so any variable the caller passes still wins. |
| `shot: static` = guide + `anchor: segment_1` + `anchor_strength: 0.25`, for takes up to about 30 s | same | Seed 1166 `0a40c41af97c` and seed 2 `93dd359dc11e` against controls `b40e855eb553`/`894911aed6c7`: fine texture 0.367 vs 0.536 and 0.325 vs 0.455 (floor 0.081). At 120 s (`25f6bd1da507` vs `8d1614ced27a`, clean/clean) it is a strict win on all 8 picture measures, by 3-23x the floor. It delays the burn: crust from about 42 s instead of 24 s. Don: "both degrade, 0.25 less" (2026-11-04). 0.5 dissolves the face by 62 s (`107859a31bc1`). |
| `shot: static_long` = guide + `keystone: segment_1` + `keystone_slot: last`, anchor off, for static takes past about 30 s | same (43a0fb87 makes `keystone_slot` a stable key) | `structural.md` *Keystone slot*. Crust stays at 1.1-1.5x segment 1 for 120 s with no reset pop, at seed 1166 (`3232cabd16b1`) and seed 2 (`94c5c0b219b3`), and again without the `<Picture 2>` prompt clause (`f55bd445faa5`, `00ed8fb067ca`). Plain keystone gives a 1-6.3x sawtooth (`c4ec5a055464`). The template's own 120 s run is `aa80458ab4f7`: fine drift 0.121, wall 59.2 min. The cost is about +9 min against plain keystone, plus a pose return every 4.25 s. Don (2026-11-06): "not a loop". |
| `shot: moving` and `shot: dialogue` = guide, no counters | same | On S2 every guide counter loses seam look (anchor at every strength 2.03-3.50 vs 1.29, floor 0.07), and every last_frame counter holds the camera back. On S3 every counter is inside the floor or loses. |
| **`look_hold: true`** on the LTX chain (`breaking-change`) | `ltx2/chained-segments` (c8fe45bb) | `ltx.md`. Seam look wins on both S5 seeds (1.72→1.17, 1.68→1.22) and passes the strict rule under S5's own floor. It does not hold an LTX orbit back (`9331cd20fa39`, `64641b316eed`) and costs nothing warm. No difference is visible by eye. Don ok'd it 2026-11-06. The H3 rule "never `look_hold` on a moving camera" is H3's only. |
| LTX long take cap about 15-20 s, then cut to a fresh still | `ltx-2.5` and `long-form` skills | At 36 s the rider changes by about 30 s on both seeds, with streaks from 20 s. With `look_hold`, segment 7 swaps the courier for a man on a moped. |
| Audio pass at `listen_scale` 1 | `minimax/add-sound` (483f4487) | `audio.md`. PSNR 42 dB against the source, which is one x264 generation. Every line is spoken word for word. Don: "very good lip sync". 0.5 is acceptable (about 23% faster). 0.25 fails lip sync. |
| Narration takes of 37 s at 3.0 words a second, `gap_seconds` 0.3 | `narration-long`, `split-takes` (50e036fe) | At 60 s, 5 of 10 takes dropped or looped words. 45 s was clean on one seed. 37 s was word-complete on two seeds (`847ccb839875`, `4467e6380276`, about 200 s each). Don: the 0.3 joins are fine ("a titch rushed on occasion"). At 0.05 the +1.5/+2.6 dB take-to-take step becomes audible, so do not go below about 0.3. |
| Music sections trimmed to `min(generated, seconds)`, 8 s end snap, 1.5 s crossfade | `minimax/music-long` (237ee307, d1d19120) | Before the fix: 13-57 s silent holes (`52303a4436d4`, `2fc471b36521`). After: 311-319 s continuous. The snap moved all six cuts 0.6-7.0 s to a quiet point, none mid-word (`2179e0c0c9df`, `d51816c55389`). Don: the drops are ok, the ending is good, and 1.5 s beats 3 s. |
| Static takes over about 30 s are cut into shots, unless `static_long`. Any take drifts past about 35 s. | `long-form` skill | `counters.md` *Round 3*, and drill ask 1 (`ab7f54f73b2f`, a 90 s dialogue chain that passed every check and drifted anyway). |

`shot` and the template's own variables are independent of the `tier` dial (Task 20,
below).

### Engine

All new keys are off by default, so every workflow that does not set one is
bit-identical to before (R3).

- **`presets`** (`dw/presets.py`, 845ec3bb, 420fef50). A top-level
  `{selector: {value: {variable: value}}}` block. The chosen bundle fills the variables
  the caller did not pass, and an explicit argument always wins. `effective_variables()`
  is the one owner. Validate, the estimate, the fingerprint and every merge site see
  bundle values, and bundle values are checked like caller values.
- **`chain.duration_seconds`** (0f075b35, 315bfaf6). The plan derives the segment
  count and the last segment's frame snap. The stitched video and its audio are trimmed
  to exactly that length. Given beside `segments`, the duration wins and the run logs it
  (R18). It is refused with `match_audio`, `guide_backoff` and `carry_pick`.
- **Per-segment check and bounded re-roll** (`chain_check.py`; 4bdf6d38, d37c3192,
  2fddfbe3). `chain.check` (true or ceilings) and `chain.retries` (0-3). The ceilings:
  - jump 4.0 and look 5.5, both set from the noise floor's P99 with about 2x margin
    (7979b0c9);
  - `cut_ratio` 4.0: the largest frame step inside a segment over its own typical step.
    Good segments peak at 2.98. The broken cuts read 4.15 and 17.25 (`6668a913863c`);
  - texture 2.0, **warn-only** (R38). A texture failure is drift in the carry, and a
    fresh seed inherits it.

  A re-roll uses the seed + attempt × 1000003. On exhaustion the chain keeps the
  closest attempt, records `check_failed` on its shot, warns `chain_check_failed`
  naming the measure, and finishes. The estimate quotes `(1 + retries)` as the worst
  case. Segment 1 and a `hard_cut` seam are never checked. Jump and look gave no false
  positives on 22 good seams (`check.md`).
- **Chain counter keys**, all measured. Only the first two are defaults, through `shot`:
  - `anchor_strength` and the H3 call argument `latent_anchor_strength` (#1190,
    f109d379). 0 is bit-identical to no anchor, and 1 to the full match;
  - `keystone: segment_1` (#1187, e8c2dac9) and `keystone_slot` (`first`, `last`, or a
    `{frame, enter_below_sigma}` slot whose sigma gate is experimental; 4ef3cd03,
    43a0fb87, 22ca7f82). `last` with `match_audio` is refused where the short last
    segment cannot take it;
  - `guide_backoff` (#1189, b4d74843);
  - `carry_pick: {sharpest: N}` (#1032, e6a91c47, 3ebe9b73);
  - `look_hold` with pickup, creep, texture and precompensate (#1191/#1033/#1067,
    `chain_look.py`; 283967c9, 1f5b3e36). It is applied once per segment, before the
    carry is extracted (R2).
- **`save_segments`/`keep_segments`** take `variable:` references. Kept segment files
  are named on the take's shots (`segment_file`).
- **Repair in place** (`dw/tasks/chain_repair.py`: `chain_repair_cut`,
  `chain_repair_splice`, `check_take_shots`; dcc58770..cf6999d7). It re-renders one
  segment with a head guide from segment k-1 and a tail guide from segment k+1, and
  splices it into the take itself. Only the repaired shots and the shots after them
  change. A take without shots is refused before the render.
- **Audio pass** (`DwH3AudioPassStep`, `DwH3ReleaseVideoStep`; 037f216e, 4a31b372).
  t2va is given `source_video` with no mask and no `hold_audio`. Every video row is the
  clip's clean latent, and only the audio is generated. The source keeps its own frame
  rate: it is retimed to 24 fps for conditioning only, and the output is the source's
  frames at the source's rate (R25, the keep-source-frame-rate rule).
- **Text and audio tasks:**
  - `split_takes` (`dw/tasks/text_takes.py`): sentence boundaries, a paragraph-break
    preference, abbreviations kept whole;
  - `slice_audio` moved to `dw/tasks/audio_slice.py` and gained `past_end: "stop"`,
    `end_snap_seconds`, and the `joined_short` and `slice_dropped_loud_tail` warnings.
- **`export_timeline`** (`dw/tasks/timeline_export.py`, #1197; 1e77425c, 224fb272).
  It writes a joined piece as an OpenTimelineIO `.otioz` bundle (`application/x-otioz`),
  with one clip per shot, gaps filled, and an audio track when the file has one.
- **`encode_h3_latents`** (13b0b276, 969ab437..7f16ee06) and streamed
  **`decode_h3_latents`** (fb6a9e11). A long take goes into one H3 latent clip by clip,
  bit-identical to a whole encode. The decode is piece by piece. The header frame count
  is bounded by duration × fps and a maximum take length before any allocation
  (security review, R32).
- **Host-RAM gate on committed memory** (c441f92e..6495878e, a follow-up to #869). The
  old figure, about 122 GB, counted the mmapped checkpoint page cache. The gate now
  projects RssAnon + RssShmem + the device-mapped pages, with a margin of
  max(10%, 2 GB). The dispatcher reads `/proc/<pid>/status` only, never `smaps`, under
  its lock. Confirmed live: two H3 jobs ran together at 42-51 GiB each, about 91 of
  125.7 GiB (`speed.md`).
- **Validation fixes found on the way:**
  - an argument the server refuses at submit now fails validate too (edbbe984);
  - Music 3 templates refuse an empty lyric at validate (688f2921);
  - an animatic past the 4 GiB frame budget is refused at validate (1bdd8439);
  - the variables pass skips key conventions at every depth (285a27fe, from the
    drill's `'Image' object is not subscriptable`, `620dc103b35c`);
  - Music 3 quotes scale with `audio_duration` (b0afbd8b);
  - measured cost rows for `repair-segment`, `narration-*` and `music-long` (9e86841b).

### Templates

| Template | What it is |
|---|---|
| `minimax/chained-segments` | Guide default, `shot` presets, `keystone_slot`, `duration_seconds`, `keep_segments`. Cost row 13.5 min (R34, measured guide, C-F311). |
| `minimax/chain-video-continuity`, `chain-matched-to-audio`, `chain-matched-and-aligned`, `music-video` | Gained `keep_segments`, so a repair can find the segments. |
| `minimax/repair-segment` (new) | `chain_take` (or `chain_output`), `segment`, its own prompt and seed. 175-frame render, head 22, tail 39. Peak VRAM 18.92 GiB against the template's estimate of 19.95 (`46d9c4f4b1d7`). |
| `minimax/add-sound` (new) | `source_video` plus a prompt that has the sound written in. `listen_scale` 1, about 3.1 min. |
| `split-takes`, `narration-long`, `narration-from-takes` (new) | `narration-long` takes text in and runs `split_takes`, then the child, in one call (R22, R24). `narration-from-takes` speaks a given takes list. |
| `minimax/music-long` (new) | Sections of `{name, caption, lyrics, seconds, audio_duration}`, trimmed, snapped, faded and crossfaded. It warns when a section's lyric is too long for its seconds. |
| `minimax/music` | Refuses an empty lyric. The quote scales with `audio_duration`. |
| `ltx2/chained-segments` | `look_hold` variable, default true. |

### Agent layer

- **`plugins/dw/skills/long-form/`** (ed88cd3b, then fixes). `SKILL.md` covers the
  per-scene decision: one clip, cuts, chain or bridge, plus length, `shot`, cost with
  the retry worst case, run and judge, and repair. The references are `levers.md` (every
  lever's default, knob or cut status, with the record it came from),
  `run-and-repair.md`, `audio.md` and `quality-and-speed.md`.
- **Pointers.** `minimax-h3`, `ltx-2.5`, `minimax-music3`, `storyboard`,
  `script-to-video` and `series-episodes` point to `long-form` and keep only rules
  specific to their family (4ad4e14b). The shot document gains
  `<project>/scenes {id, shots, length_seconds, continuity}`. The H3 skill routes a
  silent clip to `add-sound` (279086b1).
- **#1193 claims.** All five were probed on two seeds and none held
  (`docs/proposals/audits/2026-11-longvideos-claims.md`). Each is recorded under
  *Probed and not a rule* (95b5e218).
- **Cold drill (`drill.md`).** Three fresh sessions ran a 90 s walk-and-talk, a 3-minute
  four-scene piece and a 5-minute narration over boards. Asks 1 and 2 read `long-form`
  first. The fixes (ceb6147b): any take drifts past about 35 s, a saved copy has no cost
  row, the lyric (not the ceiling) sets a track's length, and narration always clones
  from a kept reference line. After the fixes, each rerun planned what the playbook
  says.

### Experimental, shipped dark

These sit behind `DW_EXPERIMENTAL_REFINE_WINDOWS=1`, which is set only on the bench
server. Without it they are refused at load or validate. No template or skill offers
them.

- `experimental_refine_windows` (`h3_windows.py`): the temporal windowed refine.
- `experimental_latent_carry` (chain key).
- `experimental_capstone` (chain key).
- The sigma-gated `keystone_slot` frame form.
- `experimental_selflift` (`h3_selflift*.py`).
- `experimental_tst` (`h3_tst.py`).

## Measured and lost

| Idea | Verdict | Numbers |
|---|---|---|
| #1189 hand-over backoff (17 frames) | Cut. The key stays. | S1 texture 0.462 vs 0.536 (inside the 0.081 floor). S2 look 1.43 vs 1.29 (loses, floor 0.07). S3 texture 0.337 vs 0.067 (loses, floor 0.23). |
| #1032 sharpest of last 4 | Cut. The key stays as a knob. | S1 0.533 vs 0.569 on a 0.036 floor, no visible change (`ae932366ea0e`). Froze the S2 arc (`9713480482bc`). Lost on LTX S5 too (seed-2 jump 1.25→1.81). |
| #1191 `look_hold` on guide | Cut. On last_frame it is a knob only, never with a camera move. | Guide S1 look 1.31 vs 1.46 is inside the 0.31 floor; S2 loses (1.56 vs 1.29). On last_frame it ties its own clean control on texture at both seeds and held the S2 camera back on both seeds (front view to 13 s, against the control reaching the statue's back). |
| #1033 / #1067 seam grade | Knob, inside `look_hold`'s `pickup` (last_frame only). | Measured in the last_frame `look_hold` rows. It steadies colour at the seams and leaves the burn. |
| Anti-burn LoRA (`Imagineer13/H3-Anti-Burn-LoRA` 0.6, every segment; #1151's trial) | Cut as a chain lever. | Its low texture is blur: the S1 face dissolves by frame 277 (`049d652e1648`), the S2 statue smears (`9124d417577d`), and S3 jump 2.05 vs 1.52 (`c50ec5961418`). |
| #1187 plain keystone at 26 s | Superseded by slot `last`. | It ties at 26 s (`ab59d503fd4d`), and every seam gets worse with anchor 0.5 (`e5c7b013538f`, `08e32162985c`). At 120 s it resets the burn, but as a visible pop at 39-111 s, crusted 40-60% of the time (`e4c7c19a99cd`, `4cae30688ed4`, `c4ec5a055464`). Kept as `keystone_slot: first`. |
| #1187 late capstone (the moving-camera candidate) | Not built (spike 45115d6a stays dark). | S2, 10 segments: gate 0.7 ties every measure. Gate 0.9 is worse past the floor on one measure per seed (look max 4.52 vs 2.71 at 1166, level max 35.9 vs 24.5 dB at seed 2). The cost is unmeasurable. `repair.md` *Step 4*. |
| Keystone frame slots (f34, f34 gated at sigma 0.9) and guide-side variants (guide_noise 0.4, guide_frames 39) | Closed. | f34 resets every segment but snaps the pose (motion step 12-25 against a p99 of 11.8, `726bb72c3908`). Gated 0.9 flashes 3 frames and burns to 13.6x (`16f116c27981`). guide_noise 0.4 raised the reset threshold (`60d9f28c6865`). guide_frames 39 adds 18 min and is still a sawtooth (`d5cb657fb9e8`). |
| #1150 / #1044 latent carry | Not a default. The flag stays dark. | Without the anchor it still burns, crusty (`20ce09d60040`). With anchor 0.25 it is "soft and masky" (Don), but that is the anchor's softness. S2 seams get worse on both seeds (`d93adb33249e`, `d901c8c82784`). Under the keystone it ties on numbers, and by eye its face holds better (`4766787e793d`, `4cae30688ed4`). It is about 5% faster (it skips the re-encode: 465 vs 757 s). |
| #1030 decode once, #1150's VRAM-saving arm | Not measured separately. | The only saving the latent-carry A/B saw is the skipped re-encode, about 5 min per two-minute take. Not worth a feature on its own (`structural.md` *The call (Step 3)*). |
| #1188 hybrid windows, Steps 5-6 | Not built. | The chain keystone answered Step 6's anchor question in chain form, and slot `last` then held the burn for 120 s. Step 5's remaining value is the blended finish, and Step 4's refine already gives the seam pass at about 36 min per minute of take. Don answers the #1188 commenter himself. |
| Windowed refine seam pass (Step 4) | Won on seams. Not promoted. | 0.3 with `{frames: 40-42, overlap: 12}`, rebased past 15 s. It wins seam jump on S1/S2/S3 (look too on S2) and nothing got worse. Drift against segment 1 is unchanged, so it is not a burn fix. At 120 s seams win only to 95 s (jump mean 0.94→0.73). It costs about 36 min per minute of take (71.5 min at 120 s, half of it decode) at a 16.1 GiB peak with the streamed decode. Don saw no shimmer and no lost motion. Promotion needs the feature stages and dropping the redundant internal decode. |
| #1195 SelfLift | Close. | Lift step 2 paints brush-stroke artifacts on every take. Lift step 1 re-stages the shot (S1 PSNR 18.0/21.7 dB) and saves about 6% of wall time (11% of denoise). |
| #1196 TST | Close. | At tau 0.2 per head there is no measured or visible effect (S1 PSNR 31-35 dB against dense). It costs 5-6% over dense, and dense is already 22-28% slower than the sparse default. tau 0 is bit-identical to dense and still costs 3%. |
| #980 FastH3 Omni | Out of long-form scope, stays parked. | It is a 68 GB Ref2VA transformer, and the long-form chains are t2va/fl2va. |
| #1193 LongVideos claims | None held. | Audit doc. Leaving the portrait off a continued shot costs identity, which backs the chains.md advice. The claim-5 hum: Don heard no drone and no music, so it stays "probed, not a rule". |
| LTX `extend`, `carry_pick`, check on S5 | `extend` is a knob. `carry_pick` loses. The check is free and never fired. | `extend` wins jump at 2.1x wall time, but on seed 2 it keeps an empty road. |
| Audio pass re-voice of an on-screen speaker | Not a use. | Don: "sounds dubbed". The picture is the source's own pixels, so no mouth is drawn for a new line. Use it only for an off-screen or voice-over line. |

## Tiers (Task 20)

> **PLACEHOLDER: tier work is still in progress. To be filled by the controller
> from `bench/records/tiers.md` once Task 20 lands.** Expected contents: the
> `tier` variable (`draft`/`standard`/`best`) and its bundles per template, the
> quality-per-minute measurements behind each, the 2-minute cost-rule check, the
> host-RAM cost of `best`, the cost rows and `cost_drivers`, whether `check`/`retries`
> sit in any bundle (see *Open questions*), Don's int8-at-standard call, and the LTX
> templates' exclusion.

## Don's rulings

- **2026-11-03:** the design and plan are approved with all defaults. Kandinsky 6 is
  out. Both lem cards go to the bench and the main server stops (R9). The Task 17
  rework comes from the #1188 correction: test the handoff and the blended refine first
  and the anchors last, and window 1 runs to full steps for clean anchors. No comments
  on #1188.
- **2026-11-04:**
  - Approved the arch-ratchet module raise, the compact-listing budget bumps (final
    `COMPACT_BUDGET` 13_900, for `minimax/add-sound`), and the Resolve import ask (Don
    runs it).
  - "Everything is yours." 0.19.0 (#1198) goes first.
  - Static default: guide + anchor 0.25, for shots up to about 30 s ("both degrade,
    0.25 less").
  - #1200 filed (cinematography moves).
  - Card use changed several times (R40, R41, and 2026-11-06).
- **2026-11-05:**
  - Latent carry first look: "cleaner by the end but quite soft and masky". The music
    was lost (later traced to a seed property).
  - The windowed refine at 120 s shows no visible shimmer or lost motion.
  - mini-ai may be used as a bench (R43).
- **2026-11-06 (the review walkthrough):**
  1. The pose return is "not a loop", so slot `last` is the static long-take default.
  2. The LTX cap is ok, and `look_hold` goes on in the LTX chain (no answer was given on
     look_hold; Don followed the recommendation).
  3. Audio pass: A passes ("very good lip sync"), B1 (0.5) is ok, B2 (0.25) fails lip
     sync, and C (re-voice) reads as voice-over, so it ships as `minimax/add-sound` at
     scale 1 (483f4487) and an on-screen speaker is never re-voiced.
  4. Narration: keep gap 0.3. At 0.05 the joins get louder, so do not go below about
     0.3.
  5. Music: the drops are ok, the ending is good, and the key change is not jarring.
     1.5 s beats 3 s crossfades. **The gain-ramp and beat-aligned-join follow-ups are
     dropped.**
  6. Hum: no drone and no music, so it stays "probed, not a rule".

  Also: #980 is out of long-form scope. Raise the ratchets at their final counts
  (modules 259, `test_dw_patch_targets` 324, R44).

## Accepted limits

- **Static.** `static_long` returns the subject to the opening pose at every segment's
  end (every 4.25 s, eased, no snap) and costs about +17% wall time. It is static only:
  it pins every segment's end to segment 1's frame.
- **Moving camera.** It has no counter. The S2 sky still blows out and the scene is
  re-invented by the last segment of a 41 s take. Moving shots are cut at about 30 s.
- **Two-minute face reads are by eye only.** The face crop moves 0.64 on a same-seed
  rerun alone.
- **The re-roll fixes a seed's accident, not a prompt that asks for the break, and not
  a carry that already holds it.** With `retries: 3` on the made-to-fail segment every
  attempt broke (`ccce7adc9aaf`). Texture failures warn without a re-roll.
- **The check measures picture only.** It does not see an audio dropout.
- **Repair limits.** A repair of segment k cannot clean a break that reached segment
  k+1's opening (`repair.md`). A second repair of the same take is not supported, and a
  take copied away without its manifest is refused.
- **LTX takes cap at about 15-20 s.**
- **Audio pass.** `listen_scale` 0.25 loses lip sync. A re-voice of an on-screen
  speaker reads as dubbed.
- **Music.** Joins into a sparse intro drop 7-9 dB (Don: ok). A section whose lyric
  overruns its seconds loses lines. The template warns, and the skill sizes the lyric
  by the measured word rate.
- **MPS.** The default guide chain at 960x544 projects 18.34 GB against a 24 GB Mac's
  17.8 GB working set and is refused. The guidance is to lower the canvas (768x448
  fits), not `num_frames`.
- **Bench workspace jobs** live on the bench servers (lem:8766, mini-ai:8766). lem's
  firewall drops LAN traffic to 8766, so it is reachable only through an ssh tunnel.

## Open questions

- **Re-roll default against the Global Constraint.** The plan says re-roll is on at
  `standard` and `best`. R3 left the engine default off and gave the default to Task
  20, and the tier work so far keeps `check`/`retries` out of every bundle. Without a
  ruling the constraint is unmet. Needs a ruling at Task 20.
- **#1044 latent carry.** Its face hold under the keystone (88-120 s clip,
  `4cae30688ed4`) waits on Don's eye. The slot `last` default, which runs without the
  carry, may make the question moot.
- **#1194 performer replacement (Task 16) was never run.** There is no record and no
  verdict.
- **#1197 Resolve import of an exported `.otioz`.** Owed by Don. The bench S4 montage
  (`6d33a88d24de`) is exported as `ae2ecd79da44` (2026-11-06; 4 clips of 124 frames at
  24 fps on one video and one audio track, the joined film bundled under `media/`), in
  `~/Desktop/dw-review/7-resolve/`, REVIEW.md step 7.
- **TST tau sweep.** Don's call. The spike found no gain at tau 0.2.
- **R36.** lem's shared `settings.json` has cudnn deterministic. Task 29's restore
  decides whether to keep or revert it for the main server.

## Follow-ups

These are drafted for filing at hand-off. None is filed yet.

- A CUDA OOM in `load_components` surfaces as "Adapter 'turbo' not present":
  diffusers logs the OOM and skips the component.
- The job queue does not survive a server restart. Queued jobs vanish without notice,
  as on the 2026-11-05 lem power-off.
- LTX on MPS renders near-silent segments (-89 dBFS, segment 3 of every seed-2 run).
  CUDA does not (`645d8600ac92`). A segment silence check is needed for the chain.
- `plan.estimate` under-quotes `narration-long`: the child's roll-up in `dw/plan.py`.
- The MPS Music 3 cost row is 2.8x faster than measured.
- The scaled H3 chain estimate over-quotes: 210 min against 71 min real in the drill
  (`ab7f54f73b2f`).
- A drift counter for moving shots: a regular reset that does not pin the pose.
- The step cache is not shared between two servers on one workspace.
- int8 H3 holds about 63-69 GB of anonymous memory. Two RAM gates on one host
  OOM-killed each other seven times.
- A braced `*_dtype` loses its braces when a template is composed as a child workflow
  (`1fbbc89cf0eb`, found during Task 20).
- An engine default of `cudnn_benchmark: false`, after timing a conv-heavy family
  (R36, not done in Task 19).
- Promoting the windowed refine seam pass (a feature with its own stages).
- Repairing two adjacent segments together (a break that spilled into k+1).
- The deferred minors from the ledger, collected in one issue.

## Cost and bounces

| Task | Fix rounds | Note |
|---|---|---|
| 1 bench server | 1 | |
| 2 driver | 1 | |
| 3 scenes | 1 | |
| 4 noise floor | 1 | |
| 6 anchor strength | 0 | |
| 7 backoff | 0 | |
| 8 keystone | 0 | |
| 9 carry_pick | 1 | |
| 10 look_hold | 1 | |
| 12 duration_seconds | 1 | R18 |
| 13 check | 0 | Later cut_ratio, R38 |
| 14 repair | 3 | |
| 15 shot presets | 2 | R33, R34 |
| 17 refine windows | 1 + 2 (4b) | Plus a security fix |
| 18 presets | 1 | Two Criticals |
| 21 audio pass | 1 | R25 |
| 22 narration/music | 2 | Plus three template fixes after the bench |
| 27 export_timeline | 0 | |
| RAM gate | 2 | |
| TST/SelfLift | 1 | |
| latent carry | 1 | |

GPU time was dominated by the two-minute static takes (52-72 min each) and the
latent-carry and keystone rounds (about 9 h on one card). The final whole-branch review
ran 3 code reviewers and one template/skill reviewer, and its fixes landed in
97fe365d..d1e2b614. Two lem power events (2026-11-04 power loss; 2026-11-05 power-key
shutdown) each lost the in-memory queue.

## Attribution

Every source that shaped a shipped behaviour is credited in `docs/ATTRIBUTIONS.md`,
with its licence and whether code or only the concept was taken. The sources include
NerdNotes, geiru_context_suite, Kinoloom and the H3MD Director. The LongVideos
(Smite79) claims are paraphrased from #1193. That repository's licence forbids
redistribution, so none of its text is copied.
