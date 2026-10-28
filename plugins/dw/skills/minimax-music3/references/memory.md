# MiniMax Music 3: checking memory before a long run

Part of the `minimax-music3` skill; step 4 of *Before anything* says when to
read this - before a long track or the music video, anything near the card's
ceiling.

`get_memory` with the server idle, and read a `live: true` reading's
`gpu_memory_allocated_mb`. Only a live reading is the worker's own and only
live readings compare:

- `info: null` means nothing is resident: go ahead.
- A `live: false` reading with a populated `info` is cached from another
  moment. It reads low while a job is loading a model, so ask again once idle
  rather than trusting it.
- A non-trivial idle figure is what an earlier run left behind, coming off
  the ~22 GB these templates need. With the server idle, `clear_memory`
  clears it (refused while a job is queued or running). It also drops the
  step cache, so the next run, including a seeded rerun, is cold and
  regenerates. Re-read `get_memory` afterwards to confirm.

Don't retry into a failed attempt without clearing first: a failed attempt is
what leaves weight resident, so an immediate retry starts from less than the
attempt that failed had.
