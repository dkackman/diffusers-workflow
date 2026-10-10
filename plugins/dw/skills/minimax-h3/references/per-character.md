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
