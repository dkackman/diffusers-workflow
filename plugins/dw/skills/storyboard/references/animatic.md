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

A read longer than about 40 s of speech is a narration in takes under one
cloned voice: `long-form`'s `references/audio.md`, *Narration longer than
one take*, which also says how to make a reference voice when the user has
none. For a voice in step with the boards, pass one take per board (each at
most about 111 words, 37 s) to `templates/narration-from-takes`, and plan
each board's duration as its words / 3.0 s plus `gap_seconds` (the voice's
measured rate), then retime by ear in step 5.

## 3. A music bed

`templates/minimax/music` makes a track from a prompt and lyrics, by
`dw:minimax-music3`'s rules. Empty lyrics are refused at run time: an
instrumental is a tag-only body (`[intro]`, `[instrumental]`, `[outro]`, one
per line), and a bed past about 90 s is `templates/minimax/music-long`
sections (`long-form`'s `references/audio.md`). Make it at least
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
in total. The limit that bites first is memory: all the frames held at once
must fit in 4 GiB (frames x width x height x 3 bytes). At 1344x768 and 24
fps that is about 57 s. `validate_workflow` refuses an animatic of asset
boards past it, naming the `fps` that fits. A board holds still, so a long
animatic lowers `fps`: 4 fps holds 5 minutes at 1344x768. The
output's audio is the soundtrack trimmed or padded to the
clip's length. A voice line only lines up with its board if the soundtrack
was cut to the same durations: the template does not time boards to audio.
Read the finished clip with `get_output_frames`.

## 5. Iterate

Change a duration or swap a board and re-run. When the timing reads right,
the durations become each shot's `duration` in the shot document
(`script-to-video`'s `references/shot-document.md`).
