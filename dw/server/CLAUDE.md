# dw/server

Guidance for the HTTP server package. Each module's docstring says what it owns; `docs/ARCHITECTURE.md` is the map of the rules that hold across them. See docs/SERVER.md.

## Worker pool

`JobManager.slots` holds one `WorkerSlot` per `--devices` entry, each with its own `WorkerManager`; a dispatcher thread (`_run_loop`, `_next_dispatch`) hands queued jobs to free slots that fit them (`check_fits`). `_current_job_id`, `_worker_lock` and `worker_manager` are compatibility names for the first slot and the oldest running job; new code reads `slots`.
