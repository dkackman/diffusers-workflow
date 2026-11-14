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

The two servers do not share the host-RAM hold. lem has 125 GB (122 GB available idle), but no recent H3 chain with a projected `host_ram_need` was on the main server to confirm that both servers' largest jobs fit together. Until that is measured, run the bench only while the main server is idle (`curl localhost:8765/api/jobs?status=running` on lem).

They share no step cache either: an arms file goes to one server (`bench/driver.py --server`) for its whole life, including measure and rerun, because another server's run is a cache miss.

## Reaching it from the Mac

lem's firewall passes 8765 but not 8766 (a rule needs sudo: `sudo ufw allow 8766/tcp`, if ufw is the filter). Until then, tunnel: `ssh -f -N -L 18766:127.0.0.1:8766 lem`, then use `http://127.0.0.1:18766`.

## Measures

`python bench/driver.py measure <record>` re-measures a record's succeeded runs; `report` prints them as a table. Each run's `measures`:

| Key | From | Meaning |
|---|---|---|
| `seam_jump_mean`, `seam_jump_max` | `analyze_seams` `seams[].jump_ratio` | Picture jump at each seam |
| `seam_look_mean`, `seam_look_max` | `seams[].look_ratio` | Look change at each seam |
| `seam_level_step_db_mean`, `seam_level_step_db_max` | `seams[].level_step_db` | Audio level step at each seam, absolute |
| `texture_<band>_seg_last` | compare's `texture_bands` on the scene crop | `\|ln(last/first)\|` of the band's luma energy, first moment to last (fine, mid, coarse) |
| `split_<m>_total` | `analyze_seams` `segment_looks` | End-to-end change of look measure `<m>` (saturation, lightness, contrast, detail): the last segment's last second minus the first segment's first second |
| `split_<m>_seams` | `segment_looks` | Sum over seams of the next segment's first second minus this one's last second |
| `split_<m>_segments` | `segment_looks` | Sum over segments of their last second minus their first second |
| `split_<m>_seam_share` | | `seams / (\|seams\| + \|segments\|)`; left out when both are 0 |
| `split_<m>_seam_sign` | | Fraction of seams whose step has the sign of `split_<m>_seams`: how consistent the seam half is |
| `wall_minutes` | the job | Run time |

`total = seams + segments` exactly, because both halves use the same endpoint windows. A segment missing a measure is skipped and the seam step bridges it. A take with no `segment_looks` (one shot, no picture, or a server before #1324) gets no `split_*` keys.

The split says *where* a measure changes, not whether a viewer sees the change: it informs a viewing verdict and doesn't replace one (#1324: the "progressively glossy" levelled carry was missed by the measures). The concept is from [ComfyUI-Continuity](https://github.com/roadmaus/ComfyUI-Continuity) (MIT; no code adapted).
