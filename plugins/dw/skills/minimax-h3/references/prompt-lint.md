# H3 prompt lint

Part of the `minimax-h3` skill; `script-to-video` sends a compiled shot
here. A checklist an agent runs by eye over a drafted prompt before
spending a render; nothing in the engine enforces it. The idea of linting a
shot document comes from Picky (GPL-3.0): idea credit only, no text or code
copied.

Flag, then fix in the shot document and recompile; do not patch only the
prompt:

1. **A missing `overall_soundscape`.** Every H3 prompt carries one, even a
   shot with no dialogue: silence is a stated choice, not an absence.
2. **An overfull beat.** More than one visible state in a shot's span (a
   person enters, sits, speaks and leaves). Split it into shots or cut to
   the one state the `beat` names.
3. **A cut with no new information.** A `cut` whose next shot repeats the
   framing, place and state of the one before. Merge the two or change the
   `shot_size`, `move` or `location`.
4. **A speaker without a voice description,** or one described differently
   from the last shot. Repeat it verbatim.
5. **A `must_not` the prompt contradicts,** such as text in frame asked for
   while `must_not` rules it out.
6. **A `duration` off the grid.** H3 runs on its `17*n+5` frame grid; the
   prompt's pacing should fit the snapped length, not the script's.
7. **A camera word outside the closed set.** Take `shot_size`, `move`,
   `lens` and `light` from `cinematography/references/vocabulary.md`, in the
   caption words its H3 table gives.

A shot that passes is not thereby good: look at the frames
(`get_output_frames`) as usual.
