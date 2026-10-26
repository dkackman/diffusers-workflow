# Series episodes: one loudness across episodes

Part of the `series-episodes` skill; its **normalize** beat and *Run and
judge* say when to read this - picking the series `target_lufs` before
episode 1, and comparing finished episodes.

## Picking the series target

`assemble-and-score`'s `balanced` step normalizes the mixed world sound and
score together to a -3 dBFS peak ceiling, always enforced. A shared peak
ceiling does not mean a shared loudness: a sparse, dialogue-only episode and
a dense, score-heavy one can both sit at -3 dBFS peak and still read as very
different volumes.

`assemble-and-score`'s `target_lufs` variable (passed through to
`normalize_audio`) gains the mix toward a measured loudness before the
ceiling is applied. Gain down succeeds only while the mix's own peak is below
the ceiling; a single loud peak (a laugh track, a sting) caps gain up the
same way, and can pull a gain-down episode lower still.

Match episodes **downward** anyway: pick one series `target_lufs` that every
episode can reach - in practice, about the quietest episode the ceiling holds
back - and pass that same value on every episode's `assemble-and-score` run.

- A `target_lufs_capped` warning on an episode means that episode's own peak
  sets the series' ceiling; lower the series `target_lufs` rather than
  accepting the mismatch.
- Only when the series should sit louder than its most dynamic episode allows
  (e.g. -16 for streaming), pass `limit: true` beside the same `target_lufs`
  on every episode: -3 dBFS becomes a true-peak ceiling a look-ahead limiter
  holds, so the laugh is limited rather than setting the gain.
- A `limiter_heavy` warning on an episode means the series target is
  squashing it audibly - lower the series `target_lufs` rather than living
  with it.

## Comparing finished episodes

`get_gallery_metadata` on each episode's final file, for duration and
loudness, so a level mismatch between episodes shows up before a viewer
notices it. Its `peak_dbfs` is a single sample and does not say how loud the
episode reads as a whole; `integrated_lufs` (BS.1770, whole-track) is the
field that answers that, and is what to compare across episodes.

A peak-normalized episode with one loud outlier (a studio-audience laugh, a
sting) can sit 1-2 LU quieter than its neighbors even at the same -3 dBFS
peak ceiling. That gap is `target_lufs`'s to close, matched downward as
above; `limit: true` for a series that must sit louder.
