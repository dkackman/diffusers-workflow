# One reference per character

A group reference picture imposes its composition on every shot that uses it,
and a prompt line saying the framing is not reused has not been enough to stop
that. Give each character their own reference instead.

1. Generate or upload the group picture as an asset.
2. For each character, run `recenter_crop` (`list_tasks`; `center_x`,
   `center_y`, `crop`, `width`, `height`, `fill`) centred on that character,
   so each output holds one person.
3. Reference the crops, one per character, in the shot's reference list
   (`keep_output` makes each an `asset:`), never the group picture.
4. Judge the first frames (`get_output_frames`) for the old composition.

## Probed and not a rule

Tested on our server (Ref2VA turbo, 960x544, two seeds per arm,
October 2026; the record is `docs/proposals/audits/2026-10-h3-aiskills-probes.md` in the
diffusers-workflow repo, #1059), and not a rule:

- Refuted, but weakly: a style-only image reference didn't pull its person
  or composition into the shot. A watercolour of an old man at a window,
  prompted as "a style reference only", left no man, pose or window on
  either seed (row P3). The medium didn't carry over either, so H3 ignored
  the reference under that phrasing; this says nothing about a style
  reference H3 does follow.

Tested on our server (Ref2VA turbo, 960x544, two seeds per arm, November
2026; the record is `docs/proposals/audits/2026-11-longvideos-claims.md`,
#1193), and not a rule:

- Refuted: a reference picture the prompt never names isn't drawn as
  another person. With a bearded man's portrait as an unnamed `<Picture 2>`,
  both seeds showed only the named woman, and his face didn't blend into
  hers (claim 2). Her similarity to her own portrait was 0.06-0.11 lower
  with the unused picture there, on both seeds; that is recorded, not
  ruled on.
- Refuted: a full-length portrait doesn't copy its clothes back on when the
  prompt dresses the person otherwise. The walker's olive-coat picture and a
  face-and-hair crop of it both gave the red sweater and jeans the prompt
  asked for, on both seeds, with no coat, and the crop held identity no
  better (claim 3). The prompt said twice that the clothes are not taken
  from the picture and marked her `partially_preserved`; a prompt without
  those words, or a picture with a bold print, wasn't run.
