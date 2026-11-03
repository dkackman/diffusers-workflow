# Long-form bench server

A second `dw.serve` on lem for benching long video takes, so the main server (the automated tester's, port 8765) is never disturbed.

| | |
|---|---|
| Port | 8766 (`http://lem:8766`) |
| Card | `cuda:1` (`BENCH_DEVICES` overrides); the main server runs on `cuda:0` only (systemd drop-in `~/.config/systemd/user/dw-serve.service.d/devices.conf`) |
| Workspace | `/home/don/bench-workspace` |
| Code | worktree `/home/don/dw-bench`, detached at `origin/<branch>`; the main checkout is never touched |
| Token | `BENCH_TOKEN`, else `/home/don/.dw-bench-token` (mode 600) |
| Log | `/home/don/dw-bench.log` |

## Use

```bash
# first time (the worktree does not exist yet): from any lem shell
ssh lem 'git -C ~/diffusers-workflow fetch origin feat/long-form-media && git -C ~/diffusers-workflow worktree add --detach ~/dw-bench origin/feat/long-form-media'
# afterwards
ssh lem 'bash ~/dw-bench/scripts/bench_server.sh deploy feat/long-form-media'
ssh lem 'bash ~/dw-bench/scripts/bench_server.sh start'    # waits up to 120 s for /api/health
ssh lem 'bash ~/dw-bench/scripts/bench_server.sh status'
ssh lem 'bash ~/dw-bench/scripts/bench_server.sh stop'     # SIGTERM, SIGKILL after 30 s
```

A code change needs `stop`, `deploy`, `start`: the server runs persistent workers.

## Host RAM rule

The two servers do not share the host-RAM hold. lem has 125 GB; see the Task 1 report for the measured budget. If a bench job and a main-server job could together exceed it, run the bench only while the main server is idle.
