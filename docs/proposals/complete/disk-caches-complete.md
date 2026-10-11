# List and clear dw's own disk caches (#828)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-10-11). Filed after lem's HF disk filled to 100% on 2026-10-11, with no
way to see how much of the disk was dw's own data. Plan v1 and v2 were designed
on #828, and Don approved v2 with his Q4 and Q5 answers folded into the stages.
It was built as stages #832 (A), #833 (B) and #834 (C). The plan comment on #828
stays the full design. This file records what shipped.

## The problem

dw's runs leave rebuildable data on disk that outlives them:
- Inductor's compiled kernels: `TORCHINDUCTOR_CACHE_DIR`, ~300 MB on lem.
- Triton's compiled and autotuned kernels: SDNQ's `triton.jit`/autotune, in
  `~/.triton/cache`, 7.1 GB on lem.
- #815's quantized weights: `DW_QUANTIZED_CACHE_DIR`, ~288 GB on lem by the
  end of stage C.

Nothing reported their size, and nothing cleared them short of a shell on the
server.

## Verdict: build, in three stages, with no new MCP tool

- **A registry**, `dw/disk_caches.py`, with three members:
  - `inductor`;
  - `triton`, registered where Triton already keeps it (dw sets no `TRITON_*`
    variable);
  - `quantized`.
- **REST:**
  - `GET /api/caches`;
  - `DELETE /api/caches/{name}`;
  - `DELETE /api/caches/{name}/entries/{entry}`.
- **A Caches panel** on the server page.
- **MCP reads only:** `get_server_info` gains `caches`. There is no
  `clear_cache` tool, and the tool count stays 62 (Don's review of v1, Q1).

## What shipped

- **Registry** (`dw/disk_caches.py`). Each member has a root, a default root,
  `enabled`, a `cost_of_clearing` sentence, and (for `quantized`) entries.
  `scan()` reports bytes, files, `last_used`, `clearable` with a reason, and
  the volume's `disk_free`/`disk_total`. The walk never follows symlinks, and a
  missing root reads as 0.
  - **Triton's root** is resolved without importing triton:
    `TRITON_CACHE_DIR`, else `$TRITON_HOME/.triton/cache`, else
    `~/.triton/cache`. `enabled` comes from `find_spec`. dw never creates the
    directory.
  - **Ownership marker:** a `.dw-cache` file. dw writes it into a root that is
    empty or new. It also writes it into a populated root that exactly equals
    that member's default; this is the grandfathering rule. Only
    `~/.triton/cache` is grandfathered for Triton, not `$TRITON_HOME` (Q5). A
    root without the marker is `clearable: false`, and a clear of it is a 400.
  - **Clear:** each child is renamed into `.clearing-<pid>-<hex>`, and that
    directory is then `rmtree`d. The root and its marker are kept. Every path
    goes through `validate_path(..., base_dir=root)`.
- **Routes** (`dw/server/routes/system.py`):
  - 404 for an unknown cache name, and the response names the valid ones;
  - 400 for a root that isn't clearable, or a malformed entry id (checked
    against the key pattern before any path is built);
  - 409 while a job is running or queued, and for `quantized` also while a
    model download is active.

  The response models are in `api_models.py`, with the regenerated
  `openapi.json` and `api-schema.ts`.
- **MCP:** `get_server_info`'s `caches` gives each member's name, bytes, files,
  enabled, clearable and disk_free, plus `entry_count` for `quantized`. It is
  omitted when `/api/caches` fails.
- **UI:** the Caches panel lists each cache. Its Clear button opens a confirm
  that quotes the cost, and is disabled with the reason when the cache can't be
  cleared. `quantized` has an expandable entry table with a delete button per
  entry, and a low-disk warning row.
- **Quantized member** (stage C):
  - Entries come from `<key>/entry.json`. An unreadable entry is listed with
    `unreadable: true` and can still be deleted.
  - `low_disk` uses #826's `quantized_cache.floor_bytes`.
  - The key check and the default root live only in `dw/quantized_cache.py`,
    and `disk_caches` calls them.
  - A cache that is turned off still lists and clears what is on disk.
- **Docs:** `SERVER.md`, `MCP.md`, `ACCELERATION.md`, `QUANTIZATION.md`,
  `REMOTE.md`, `SECURITY.md`, and the `ARCHITECTURE.md` row *dw disk caches*.

## Deviations from the plan

- **No `SURFACE_BUDGET` raise.** The docstring clause measured 16 tokens, which
  still fits under the existing ceiling. The measurement is recorded in
  `tests/test_mcp_server.py`.
- **A missing root lists `clearable: false`** ("root does not exist (nothing to
  clear)"). On mini-ai, `triton` reads `enabled: false, bytes: 0`.
- **Roots are re-read per request**, not fixed at server start. The marker is
  also written when a default root first appears after start (#837 below).
- **An entry's `component` is its `subfolder`,** because `entry.json` has no
  component field.
- **`DELETE /api/caches/..%2Fx` answers 404 or 405, not only 404.** The decoded
  `/` matches no clear route, so nothing is touched either way.

## Question 2 answered: clearing under a warm worker needs no restart

This was checked on lem, on one warm worker (`cuda:1`) running the MiniMax H3
`video-with-audio` template, with no `clear_memory` between runs. After
`DELETE triton` freed 7.08 GB (149,915 files), a new shape denoised and
succeeded. After `DELETE inductor` freed 60 MB, the same shape and then another
new shape succeeded too. The designed restart fallback wasn't needed and wasn't
built.

`triton` did not refill under the warm worker. The worker keeps its kernels in
memory, so the cache only fills again on a cold start. This is expected, not a
defect.

## Deferred

- **Eviction:** clearing is manual only. If the registry shows `triton` growing
  again, the first follow-up is a size cap or LRU for it. A second full-disk
  incident with the quantized cache on would also reopen this.
- **A `clear_cache` MCP tool:** adding one later is additive and needs its own
  approval.
- **Caching the scan:** no cache was added. If the walk over ~20k Triton
  entries makes `get_server_info` slow, cache the scan for 30 s.

## Stages and bounces

| Stage | What | Bounces |
|---|---|---|
| #832 A | registry, `inductor`/`triton`, list and clear routes, MCP `caches` | 0. Verified on mps first time (C-F389, C-F390). |
| #833 B | the UI Caches panel | 0 tester bounces. Don's browser check found that an empty cache's confirm said "1 KB", fixed in `d83a3793`. The REST steps on lem were run by the lead, because the tester is MCP-only. |
| #834 C | `quantized` member, entry delete, `low_disk` | 1 architecture-review bounce: the key pattern and the default root were each defined twice. Fixed in `096bc1fd`. Then verified (C-F391-C-F393). |

**After the stages:** #837. On lem, `triton` turned `clearable: false` after
its first run. The marker was only written at server start, and Triton's root
didn't exist yet then. Fixed in `16959b02`: a default root that appears after
start is now marked. When this was written, #837 was still waiting for the
tester's verification.

**Process note:** the plan was decomposed twice, which filed a duplicate stage
set (#829-#831, closed `not planned`). Stage A was built under #829's number
and handed off under #832.

The stage comments give no cost figures. The plan estimated ~$8-14.
