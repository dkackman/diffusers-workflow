# MiniMax Music 3: loudness

Part of the `minimax-music3` skill; step 4 of *Run and judge* says when to
read this.

`get_gallery_metadata`'s `peak_dbfs` is a single sample and doesn't say
how loud the song reads end to end - `integrated_lufs` (BS.1770,
whole-track) is the field for that, and what `normalize_audio`'s
`target_lufs` targets when a mix must match another track by ear, not
by peak. A master louder than its peak allows (-16 streaming): add
`limit: true`; `limiter_heavy` means lower the target.
