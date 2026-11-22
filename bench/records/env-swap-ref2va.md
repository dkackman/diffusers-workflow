# Ref2VA environment swap: MiniMax-H3 probe (#1518 stage A, #1545)

**Status: draft - prompts recorded before any job ran; results below are filled in as the jobs finish.**
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

_Pending._
