# The animatic recipe

Part of the `storyboard` skill; step 4 says when to read this. An animatic
is boards held for a time with a voice and a music bed under them. It tests
the cut's pace and order for the price of a few seconds of CPU.

## 1. The boards

Boards at one size, kept with `keep_output(name, asset_name="boards/<scene>-NN")`
with no extension: the engine appends the kept file's own, and Klein writes
JPEG, so each is an `asset:boards/<scene>-NN.jpg`. A name ending in another
extension is refused. A mixed-size set is refused by the task, naming both
sizes.

## 2. A scratch voice

`templates/generate-speech` speaks one line as a wav. Run it per line, or
once for a whole read; it is a scratch voice for timing, not the final
performance. Keep the wav as an asset.

## 3. A music bed

`templates/minimax/music` makes a track from a prompt and lyrics (instrumental
if the lyrics are empty), by `dw:minimax-music3`'s rules. Make it at least
as long as the animatic: `animatic` trims a longer track and pads a shorter
one with silence, either way with a warning. To lay voice and bed together,
mix them first with `mix_audio` into one asset.

## 4. The animatic

Run `templates/animatic`:

```
validate_workflow(name="templates/animatic", arguments={
  "boards": ["asset:boards/s1-01.jpg", "asset:boards/s1-02.jpg", ...],
  "durations": [2.0, 1.5, 3.0],
  "soundtrack": "asset:audio/s1-scratch.wav",
  "fps": 24})
```

`durations` is a list with one duration per board, in seconds, as long as
`boards`; repeat the number to hold every board alike. Each board gets
`max(1, round(duration * fps))` frames, at most 600 seconds and 36000 frames
in total. The output's audio is the soundtrack trimmed or padded to the
clip's length. A voice line only lines up with its board if the soundtrack
was cut to the same durations: the template does not time boards to audio.
Read the finished clip with `get_output_frames`.

## 5. Iterate

Change a duration or swap a board and re-run. When the timing reads right,
the durations become each shot's `duration` in the shot document
(`script-to-video`'s `references/shot-document.md`).
