# Repairing and upscaling the user's footage

Part of the `ltx-2.5` skill: the restore, upscale, refine and face-repair
templates that take a clip the user supplies.

## One defect each

`templates/ltx2/restore-deblur` for spatial defocus,
`templates/ltx2/restore-decompression` for low-bitrate artefacts,
`templates/ltx2/restore-colorize` for black-and-white or washed-out footage.
All three come back at the source's size and frame rate (`frame_rate`
overrides the rate), with no soundtrack.

Colorize colours only what the prompt names: look at a source frame
(`get_output_frames`) and write both prompt panels for that clip, naming its
colours - the default is an example and leaves an unlike clip grey. Reliable
on 24 fps 16:9; a low-rate or portrait source may stay grey even so: check a
frame of the result; if it is grey, this route doesn't take that source.

Each inverts one defect and no other - neither upscales or removes motion
blur or grain - so name the defect and let the user correct you.

## Upscale or sharpen

`templates/ltx2/upscale-clip` (IC-LoRA re-render), `width`/`height` the
render size, or `refine-clip` (its own latents, no LoRA), `width`/`height`
the working size. Both fit the source first and restore after: output
exactly 2x the source (2x the kept region under crop) at its length, any
aspect ratio letterboxed and cut back (`fit`), a short source held and
trimmed. Soundtrack kept, silent source refused.

Same size and length, no 2x: `templates/ltx2/refine-in-place` (`strength`
0-4, default 2).

All three write at the source's frame rate (`frame_rate` overrides the rate,
never the frame count).

## A small face

A small, smeared face in a wide shot, the rest left alone:
`templates/ltx2/face-repair` (`strength` 0-4, default 2; describe the person
in `prompt` to hold identity at 3-4). A near face comes back untouched; the
source's soundtrack, size and frame rate are kept (`frame_rate` overrides the
rate, never the frame count).

## Longer than one window

Footage longer than `num_frames`: `templates/ltx2/restore-long` restores it
window by window. Its `windows` list needs
`ceil(source_frames / (num_frames - overlap))` entries, `index` 0 up - read
`source_frames` from `get_gallery_metadata`, keep `num_frames` 121 and
`overlap` 16 (105 new frames per window), and let `validate_workflow` name
any entry to add or drop. At most 32 windows. Written at the source's frame
rate with its soundtrack (`frame_rate` overrides the rate, never the frame
count).
