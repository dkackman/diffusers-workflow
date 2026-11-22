# Ref2VA environment swap: MiniMax-H3 probe (#1518 stage A, #1545)

**Status: complete. Environment swap confirmed at both seeds; `retention_analysis` not shown to be operative (the seeds disagree). Prompts were recorded before any job ran.**
Frame reads are an agent's (claude-opus-5-5, anthropic).

## Source of the concept

A Reddit post (cited in #1518) described editing a clip with H3's Ref2VA by
passing the clip as a video reference and writing a structured prompt that
marks the environment and weather `fully_replaced` while motion and camera are
`fully_preserved`. No code from the post is copied. The post's exact prompt
text was not available to this session, so arm A was composed from the
structure #1518 describes (subject definitions, an `Edit <Video 1> by
replacing the original environment and weather with ...` summary, a
retention analysis, a CHANGE / PRESERVE / consistency description, and
soundtrack lines).

## Setup

- lem, RTX 3090 24 GB, workspace `default`, 2026-11-22 (UTC), develop @ 834bec82.
- `get_diffusers_state`: diffusers 0.44.0.dev0 @ ed32672706fec20ddac3a090c5a8290b3fd313cb, idle.
- Workflows: `bench/workflows/env-swap-a.json`, `-b.json`, `-c.json`. Shared:
  ref2va, a single `MiniMaxH3VideoReference` (`variable:source`) and nothing
  else (no image reference, no `guides`, no `source_video`), turbo ref2v LoRA
  (`minimax_h3_ref2v_turbo_8step_v1.0_768p`), int4 SDNQ, sparse attention,
  9 steps, video_shift 12, audio_shift 3, 124 frames, 960x544 - the settings
  of `workflows/templates/minimax/composable-references.json`. Only `prompt`
  differs between arms (and `seed` across seeds); `tests/test_bench_env_swap.py`
  pins that.
- Validate's estimate for an arm: 16.1 min (basis `inherited` from
  `templates/minimax/reference-to-video`, scaled x4 for a whole-clip video
  reference, low confidence).

### Source clip

`asset:bench/1518/env-swap-source.mp4` (default workspace), kept from job
`507c71743ec2`, run `20261116-134453-4a489844`
(`templates/minimax/video-with-audio/20261116-134453-4a489844/final/MiniMaxH3-text_to_video_audio.0-0.0.mp4`,
note "#1260 stage A (#1379): H3 tracking, seed 42"). 960x544, 24 fps, 124
frames (5.17 s), audio 32 kHz stereo. Content: a man in a dark overcoat, flat
cap and dark trousers walks left to right across a wet cobbled square at
blue-hour dusk, past a stone tiered fountain, a red telephone box and an iron
lamp post, lit brick shopfronts behind, light mist; the camera tracks right
with him (the phone box moves from right of frame at frame 0 to left-centre by
frame 123).

The source soundtrack is near-silent: mean -55.3 dBFS, peak -34.4 dBFS,
integrated -53.0 LUFS (`near_silent`). Sync judging below accounts for that.

## Prompts (verbatim, written before any job ran)

### A - structured edit

```text
subject_definitions:
<Video 1> is the source video: a wide tracking shot of a man in a dark overcoat, a flat cap and dark trousers walking from left to right across a wet cobbled town square at blue-hour dusk, past a stone tiered fountain, a red telephone box and an iron lamp post, with lit brick shopfronts behind him and a light mist low over the cobbles; the camera tracks right at his pace.

summary:
[reference generation] Edit <Video 1> by replacing the original environment and weather with a narrow mountain village lane of dark timber chalets under deep fresh snow, in heavy, active snowfall at dusk, while the man, his walk and the tracking camera stay exactly as they are in <Video 1>.

retention_analysis:
<Video 1>: attribute_transfer
environment: fully_replaced
weather: fully_replaced
lighting: partially_modified
motion: fully_preserved
camera: fully_preserved

detailed_description:
The target video is in realistic photographic style. [Shot 1]
CHANGE: The cobbled square, the stone fountain, the red telephone box, the iron lamp post and the brick shopfronts of <Video 1> are gone. In their place is a narrow mountain village lane under deep fresh snow, lined with dark timber chalets whose small windows glow warm orange, snow heaped on their roofs and window ledges, pine-covered slopes rising behind them into low grey cloud. Heavy snow falls steadily through the whole clip, large flakes drifting down across the frame from the first frame to the last and settling on the man's cap and shoulders. The light is a cold blue-grey winter dusk, with warm window light falling across the snow.
PRESERVE: The man is the same man as in <Video 1>, with the same face, dark overcoat, flat cap and dark trousers. He walks from left to right with exactly the pace, stride, posture and arm swing of <Video 1>, and reaches the same positions in frame at the same moments. The camera tracks right with him exactly as in <Video 1>, at the same framing, height and speed. Do not improvise or add movements: no new gestures, no turn of the head, no stop, no change of direction, and no camera move that <Video 1> does not have.
Consistency: the snowfall, the lane and the light stay continuous across the whole clip with no flicker or cut; the man's identity and clothing are identical in every frame; his feet meet the snow where his feet met the cobbles in <Video 1>.

overall_soundscape:
preserve the original paired soundtrack of <Video 1>.

non_diegetic_music:
preserve the original paired soundtrack of <Video 1>.
```

### B - one plain paragraph (comparison)

```text
A wide tracking shot of a man in a dark overcoat, a flat cap and dark trousers walking from left to right along a narrow mountain village lane under deep fresh snow at dusk, in heavy, active snowfall. Dark timber chalets with warm orange windows and snow heaped on their roofs line the lane, pine-covered slopes rise behind them into low grey cloud, and large flakes fall steadily across the frame from the first frame to the last, settling on his cap and shoulders. The camera tracks right at his pace in a cold blue-grey winter light.
```

### C - A without `retention_analysis`

```text
subject_definitions:
<Video 1> is the source video: a wide tracking shot of a man in a dark overcoat, a flat cap and dark trousers walking from left to right across a wet cobbled town square at blue-hour dusk, past a stone tiered fountain, a red telephone box and an iron lamp post, with lit brick shopfronts behind him and a light mist low over the cobbles; the camera tracks right at his pace.

summary:
[reference generation] Edit <Video 1> by replacing the original environment and weather with a narrow mountain village lane of dark timber chalets under deep fresh snow, in heavy, active snowfall at dusk, while the man, his walk and the tracking camera stay exactly as they are in <Video 1>.

detailed_description:
The target video is in realistic photographic style. [Shot 1]
CHANGE: The cobbled square, the stone fountain, the red telephone box, the iron lamp post and the brick shopfronts of <Video 1> are gone. In their place is a narrow mountain village lane under deep fresh snow, lined with dark timber chalets whose small windows glow warm orange, snow heaped on their roofs and window ledges, pine-covered slopes rising behind them into low grey cloud. Heavy snow falls steadily through the whole clip, large flakes drifting down across the frame from the first frame to the last and settling on the man's cap and shoulders. The light is a cold blue-grey winter dusk, with warm window light falling across the snow.
PRESERVE: The man is the same man as in <Video 1>, with the same face, dark overcoat, flat cap and dark trousers. He walks from left to right with exactly the pace, stride, posture and arm swing of <Video 1>, and reaches the same positions in frame at the same moments. The camera tracks right with him exactly as in <Video 1>, at the same framing, height and speed. Do not improvise or add movements: no new gestures, no turn of the head, no stop, no change of direction, and no camera move that <Video 1> does not have.
Consistency: the snowfall, the lane and the light stay continuous across the whole clip with no flicker or cut; the man's identity and clothing are identical in every frame; his feet meet the snow where his feet met the cobbles in <Video 1>.

overall_soundscape:
preserve the original paired soundtrack of <Video 1>.

non_diegetic_music:
preserve the original paired soundtrack of <Video 1>.
```

## Soundtrack mechanism

A video reference conditions the generation; it cannot copy its audio into the
output. The `overall_soundscape` / `non_diegetic_music` lines ask for the
original soundtrack, but the only ways dw carries a clip's audio through are
`hold_audio` or a guide's `audio: true`. This probe instead muxes the source
audio onto each output with the existing video/audio tasks and judges sync.

## Runs

Every arm ran on lem's `default` workspace through MCP as an inline workflow
(the server's checkout does not carry this branch), with the same definition
as the committed file. Arm B's first run passed its prompt as an `arguments`
override of an inline copy whose `prompt` default was empty; the text is
identical to `env-swap-b.json`'s. The seed-2002 runs are `rerun_job` of the
seed-1001 jobs with `arguments: {"seed": 2002}`.

| Arm | Seed | Job | Run | Card | Wall | Encode | Denoise | First pass | Pass | Decode |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 1001 | `8624df4b7aca` | `20261122-073914-68b30788` | cuda:0 | 1004 s | 77.6 s | 894.8 s | 368.7 s | 75.2 s | 20.1 s |
| B | 1001 | `968307188b3d` | `20261122-075639-7e65256f` | cuda:1 | 1136 s | 75.6 s | 1028.5 s | 388.6 s | 89.4 s | 21.1 s |
| C | 1001 | `f51e05720963` | `20261122-081708-baf989e4` | cuda:0 | 709 s | 77.1 s | 604.5 s | 77.4 s | 75.3 s | 19.9 s |
| A | 2002 | `11f0d3ac7b8f` | `20261122-081716-db817948` | cuda:1 | 867 s | 76.4 s | 767.5 s | 98.1 s | 95.5 s | 19.9 s |
| B | 2002 | `701a135d37e1` | `20261122-082857-5ab03aab` | cuda:0 | 700 s | 76.4 s | 601.4 s | 80.9 s | 74.2 s | 19.7 s |
| C | 2002 | `80ce117593d1` | `20261122-083143-042d5739` | cuda:1 | 857 s | 76.7 s | 758.1 s | 99.0 s | 93.5 s | 19.2 s |

8 passes each. Wall is job start to finish. Load 4.8 / 4.4 s for the two
seed-1001 A/B runs, 4.5 s for C 1001, and cached (0 s) for the seed-2002
runs. Host memory peak 23846 MB (A, B 1001) up to 24681 MB (C 2002).

Against the baselines the plan names: a pass costs 74-95 s, which is the
whole-clip video-reference class (97.6 / 95.4 s/pass, 891-1228 s wall), not
the image-only B0s (22.0 s/pass, 386 s wall). The cuda:1 runs after the first
sit at 93-95 s/pass, and the cuda:0 runs at 74-75 s/pass. The two seed-1001
runs that went first on each card paid a first pass of about 370-390 s
against about 75-99 s for every later run, so a cold card adds about 5 min to
the first job. Walls run 700-1136 s. Validate's 16.1 min estimate (966 s)
sits inside that range.

### Gate: A and B at seed 1001

Frames 0 / 31 / 62 / 92 / 123 (0 / 25 / 50 / 75 / 100 %) of each output read
against the same frames of the source. Positions are the man's centre in a
384-px-wide frame, read by eye (about +/-10 px).

| Frame | Source | A | B |
|---|---|---|---|
| 0 | x~60, mid-stride, fountain behind, phone box right | x~58, mid-stride; chalet lane, deep snow, lamp post right of centre | x~68, standing upright, feet together, not mid-stride |
| 31 | x~115, mid-stride | x~118, mid-stride | x~133, starting to walk |
| 62 | x~165, mid-stride under the lamp post | x~170, mid-stride, beside the lamp post | x~185, mid-stride |
| 92 | behind the phone box (occluded at x~205) | x~212, mid-stride (no box to hide him) | x~218, mid-stride |
| 123 | x~240, mid-stride | x~255, mid-stride | x~245, mid-stride |

Camera: in the source the background slides about 150 px left over the clip
(the phone box from x~300 to x~150). In A the lamp post slides from x~222 to
x~75 (about 147 px): the same track. In B the right-hand chalet slides about
65-100 px: the same direction, a slower track.

**A, per criterion (seed 1001):**
- **Motion: holds, 5 of 5.** Position, stride phase and camera track agree
  with the source on every sampled frame; frame 92 places him where the source
  hides him behind the phone box. No added gesture, stop or turn seen.
- **Environment: replaced, 5 of 5.** No cobbles, fountain, phone box or brick
  shopfronts in any frame; a snow-covered lane of timber chalets, snow on
  roofs, ground and fence rails in every frame. One source object survives
  restyled: an iron lamp post, where the source had one, though CHANGE named
  it as gone. *Falling* snow is not legible in the stills (the air is hazy;
  no distinct flakes at 960 px on frame 62), so "heavy, active snowfall" is
  not confirmed; snow *on the ground* is throughout.
- **Identity: holds.** Dark overcoat, flat cap (now snow-topped), dark
  trousers and build match in every frame. The face is too small in both clips
  to judge features beyond that.

**B, per criterion (seed 1001):** environment replaced in all 5 frames (the
same kind of chalet lane); identity holds (coat, cap, trousers). Motion is
looser: at frame 0 he stands still rather than mid-stride, he runs ahead of
the source's position at frames 31-62, and the camera track is slower. So the
video reference alone carries most of the walk and track, and the structured
prompt tightens it (A's start pose and track match; B's don't).

**Gate verdict: pass.** A matches on 5 of 5 frames, the environment is
replaced on all 5 with snow throughout (lying snow; falling snow not legible),
and identity holds. The full matrix ran.

### Soundtrack: source audio muxed on

Job `42ababbdbb83`, run `EnvSwapMux/20261122-081603-1ed4da86`: two
`pair_audio` steps (`fit: "video"`) put the source clip's own track
(`asset:bench/1518/env-swap-source.mp4`) on A's and B's outputs
(`EnvSwapMux-mux_a.0-0.0.mp4`, `EnvSwapMux-mux_b.1-0.0.mp4`). Both run
124 frames / 5.167 s with a 5.167 s audio stream, no trim or pad warning.
Levels are the source's (mean -55.35 dBFS, peak -34.5 dBFS, `near_silent`);
the 1 s envelope is flat to within 3 dB (RMS -56.7 to -53.8 dBFS) with no
transient that stands out, so there are no footfalls or other events to sync
against. Sync is therefore judged by timing only: the track is the source's
length, and the picture keeps the source's timing frame for frame (motion
table above), so any event in the source track would land where it did in
the source. A real sync test needs a source clip with audible footsteps.

The generated outputs' own tracks: A mean -51.93 dBFS, peak -28.96 dBFS
(warned "quiet overall, not empty"); B mean -56.04 dBFS (warned near-silent).
The "preserve the original paired soundtrack" lines produce a quiet ambience
like the source's, not a copy of it.

### Full matrix: C at 1001, then A, B and C at 2002

The same five frames, read the same way (the man's x position in a 384-px
frame).

| Frame | A 2002 | B 2002 | C 1001 | C 2002 |
|---|---|---|---|---|
| 0 | x~62, mid-stride | x~62, mid-stride | x~62, mid-stride | x~62, mid-stride |
| 31 | x~122, mid-stride | x~128, mid-stride | x~118, mid-stride | x~122, mid-stride |
| 62 | x~178, mid-stride | x~180, mid-stride | x~172, mid-stride | x~178, mid-stride |
| 92 | x~215, mid-stride | x~222, mid-stride | x~212, mid-stride | x~215, mid-stride |
| 123 | x~255, mid-stride | x~250, **standing, feet together** | x~250, mid-stride | x~255, mid-stride |

- **A 2002.**
  - Motion holds on 5 of 5 frames, and the camera tracks right (the chalets
    slide left).
  - The environment is replaced on all 5 frames: a snowy chalet lane with no
    fountain, phone box, cobbles or shopfronts. Snow lies everywhere, but no
    falling flakes are legible.
  - Identity holds: cap, overcoat and trousers.
  - **A passes the gate's criteria at both seeds.**
- **B 2002.**
  - The environment is replaced on all 5 frames: a fence-lined chalet lane.
  - Identity holds.
  - The walk matches through frame 92, but at frame 123 he has stopped,
    standing with his feet together. At seed 1001 B departed the other way: he
    stood still at frame 0.
- **C 1001.**
  - Motion and camera match the source on 5 of 5 frames (the lamp post slides
    from x~222 to x~45), and identity holds.
  - The environment is changed to a snowy chalet village, but **two source
    objects survive, restyled under snow**: the stone tiered fountain, at
    left-centre in every frame, and the iron lamp post. A at 1001 kept only
    the lamp post.
- **C 2002.**
  - Near-identical to A 2002 frame for frame: the same chalets, the same lane,
    the same positions. No fountain survives.

### Verdicts (a claim holds only when both seeds agree)

| Claim | Seed 1001 | Seed 2002 | Verdict |
|---|---|---|---|
| A swaps the environment and keeps motion, camera and identity | yes (5/5) | yes (5/5) | **Confirmed** |
| Snow in every frame | lying snow yes, falling not legible | the same | **Confirmed for lying snow only**: heavy falling snow is not shown |
| `retention_analysis` is operative (A differs from C, in A's favour) | A removes the fountain and C keeps it | A and C near-identical | **Not shown**: the seeds disagree |
| The structured prompt holds motion tighter than one paragraph (A vs B) | B stands at frame 0 | B stops at frame 123 | **Confirmed, narrowly**: at both seeds B leaves the walk on one sampled frame and A doesn't |
| The plain paragraph alone also swaps the environment (B) | yes | yes | **Confirmed** |
| The source soundtrack survives the swap | (see Soundtrack) | | **Not by generation**: the outputs carry a quiet ambience of their own; `pair_audio` puts the source track back on length-exact |

No job was refused and none failed. Every warning was a level warning on the
output's own track: A 1001 and C 1001 were "quiet overall, not empty"; A
2002, B 1001, B 2002 and C 2002 were near-silent (-54.6 to -57.2 dBFS mean).
These are expected, since the source's track is itself near-silent room tone.
The A and B 1001 warnings are quoted above. C 1001's, verbatim: "env_swap:
EnvSwapC-env_swap.0-0.0.mp4 decodes at a mean level of -52.20 dBFS but peaks
at -29.06 dBFS: quiet overall, not empty. Expected for an ambience-only shot;
a concern only if this was meant to carry speech or music."

## Conclusion

An environment swap with H3's Ref2VA works with the clip as the only
reference. The man's walk, the camera's track and his clothing carry over,
and the setting is replaced. The structured edit prompt (arm A) did that at
both seeds. A plain paragraph (arm B) also swaps the setting, but lets the
walk drift at one end of the clip. The `retention_analysis` block made no
difference that held across seeds: at 1001, the arm without it (C) kept the
fountain, and at 2002, C and A came out nearly the same. So keep the block if
you like, but don't rely on it. CHANGE / PRESERVE does the work that can be
seen. Lying snow came through in every frame, but heavy *falling* snow was
not legible in the stills of any arm. Asking for "preserve the original
paired soundtrack" makes a quiet ambience like the source's, not the source
track itself, so put the source track back with a `pair_audio` task.
