# GYRE

*One shape, nine worlds.*

A spiral holds the exact centre of a square frame. Every 5.17 seconds the
material of it is replaced on a hard cut, moving outward through scale:

> an iris → a fingerprint → a fern fiddlehead → a nautilus → a dew-strung orb
> web → a whirlpool → a hurricane → a galaxy → and back to an eye, with the
> galaxy reflected in it

The cut is the whole idea. Your eye tracks a shape it believes it is still
watching while the universe underneath it is swapped out. Each world brings its
own sound — surf, wind, water, the hum of orbit — and those change at the cuts
too, while a single unbroken score runs under all of them.

## Running it

Three passes, in order. Each one is cheap to review before committing to the
next, which is the point of splitting them.

```bash
python -m dw.run workflows/gyre/GyreStills.json  --output_dir ./outputs/gyre
python -m dw.run workflows/gyre/GyreFrames.json  --output_dir ./outputs/gyre
python -m dw.run workflows/gyre/GyreFilm.json    --output_dir ./outputs/gyre
```

Those write into `./outputs/gyre`. Passes 2 and 3, and every workflow below,
also *read* from `assets/` - the nine source stills, the nine registered frames,
the nine generated shots and the score. **`assets/` is not in the repository.**
It is about 30 MB of generated media kept outside the tree, so restore it from
wherever you archived it before running anything but `GyreStills.json`.

| Pass | What it does | Cost |
| ---- | ------------ | ---- |
| [`GyreStills.json`](GyreStills.json) | Z-Image Turbo paints the nine source stills at 1024x1024; MiniMax-Music3 writes two candidate scores | ~10 min |
| [`GyreFrames.json`](GyreFrames.json) | Registration. `recenter_crop` re-frames each still to 704x704 around its own spiral's centre | seconds |
| [`GyreFilm.json`](GyreFilm.json) | MiniMax-H3 animates each frame into a 124-frame shot, `stabilize_video` holds each one still, `concat_videos` cuts them together, and the score is mixed over the nine worlds' own sound | ~40 min on a 3090 |

[`GyreDissolve.json`](GyreDissolve.json) is the same edit with the cuts softened
into cross-dissolves - which, because the spirals are registered, transforms one
subject into the next rather than fading between them.
[`GyreAssemble.json`](GyreAssemble.json) re-cuts the film from the shots already
written, without regenerating them - the edit is a minute, the generation is an
hour, and they should not be tied together.
[`GyreReshoot.json`](GyreReshoot.json) re-generates a single shot when one of
them misses, leaving the other eight alone.

[`GyreStillsFix.json`](GyreStillsFix.json) and [`GyreStillsFix2.json`](GyreStillsFix2.json)
are the re-shoots of the three stills the first pass missed. They are kept
because the three stills workflows together reproduce the nine source stills
*exactly*, and `GyreFrames.json`'s centres are hand-measured on those exact
images — regenerate them differently and its numbers no longer mean anything.
They do not rebuild `assets/` by themselves, though: which of their takes became
which `src_*.jpg` was picked by eye and never written down, so the archived copy
is the only authority. `GyreStillsFix.json` also reads one of pass 1's stills
back out of `assets/` under its original run name, so it cannot run from a bare
tree either.

Run [`GyreReshoot.json`](GyreReshoot.json) as a single-shot probe before
committing to the full film whenever a parameter changes — it is the cheapest way
to find out that something is wrong.

## The two things that make it work

**Registration.** A generative model will not put a subject on an exact pixel
because the prompt asked it to, and across nine unrelated prompts it never puts
them in the same place twice. `GyreFrames.json` measures each spiral's centre
once and re-frames around it, so all nine agree on where the centre is and how
big it is. Without this the cuts are nine pictures in a row; with it they are
one thing changing.

**Stabilization.** H3 drifts. Given a pinned keyframe and a prompt asking for a
locked camera, the probe shot still slid 38px down a 704px frame over 124 frames.
That is invisible inside a shot and glaring across a cut, where the subject snaps
back to centre. `stabilize_video` takes it to 2px.

Both are documented, with what was tried and abandoned, in [NOTES.md](NOTES.md).

## Notes on the prompts

The H3 prompts are hand-written Context-IR, in the format the
[built-in enhancer](../minimax/MiniMaxH3EnhancePrompt.json) produces: the I2VA
instruction line, then `integrated_multimodal_description`,
`overall_soundscape` and `non_diegetic_music`. `non_diegetic_music` is `None.`
in all nine, because the music is not H3's job here — the score is laid on
afterwards, and asking H3 for music as well only muddies the soundscape that is
wanted.

Each prompt opens on `<Picture 1>` and describes what is actually in that frame
before it describes any motion, which is what keeps the generated first frame
faithful to the registered still.
