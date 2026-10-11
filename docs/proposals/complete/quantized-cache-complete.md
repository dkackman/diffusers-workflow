# On-disk cache of on-the-fly-quantized components (#815)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-10-11). The idea came from a load-time profile during the H3 speed A/B on
2026-10-10. Plan v1-v3 were designed on #815; stage 0's measurements and Don's
six decisions made v4, which Don approved. It was built as stages #825 (0),
#826 (A) and #827 (B). The plan comment on #815 stays the full design; this is
the record of what shipped.

## The problem

Every cold load of an SDNQ on-the-fly component (a fresh worker, a
`clear_memory`, a config change) read the bf16 weights, copied them to the GPU,
quantized them and copied them back, and an identical config gave identical
weights each time. The issue's profile put the H3 transformer at 46.7 s, ~34 s
of it data movement rather than quantization math.

## Verdict: build smaller

Built for **SDNQ only**, **opt-in**, in the two families that quantize on the
fly: MiniMax H3 (modular `load_components`) and LTX-2.5 (per-component
`from_pretrained`).

Cut, with what would bring each back:
- **bitsandbytes, torchao, quanto:** partial serialization, some via pickled
  tensor subclasses, so each needs its own security review. Trigger: a field
  report of slow cold loads from a template using one.
- **GGUF:** already a file format; nothing to gain.
- **Cache-aware `plan.downloads_required`:** trigger: a run that downloads bf16
  weights a hit never reads.
- **Listing and clearing over MCP, REST or the UI:** moved to #828 (Don,
  decision 5). Until it lands, clearing is `rm -rf "$DW_QUANTIZED_CACHE_DIR"`.
- **Reordering pruning ahead of the cache (Q5):** stays no. The entry holds the
  untruncated component; a hit takes ~1-9 s, so time doesn't trigger a revisit.

## Plan history

- **v1-v3:** design, a measurement gate (stage 0 must show a ~30 s H3 saving),
  cache-before-pruning, and the key.
- **v4:** folded in stage 0's numbers and Don's decisions:
  1. off by default, `DW_QUANTIZED_CACHE=1` opts in;
  2. `DW_QUANTIZED_CACHE_DIR`, default `<HF_HOME>/dw-quantized`;
  3. a free-space floor of `min(50 GiB, 10%)` per save (`skipped: low_disk`);
  4. the key's commit comes from a snapshot verified complete on disk
     (`skipped: unpinned_snapshot` otherwise);
  5. listing/clearing moved to #828;
  6. #825's three script fixes folded into stage A.

  Don's review fixed one thing in the decompose: the start-up low-disk warning's
  margin is **the largest entry already in the cache** (zero when empty), not a
  hard-coded H3 size, because model sizes don't belong in engine code.

## What was built

### Stage 0 (#825): measurement script

`scripts/measure_quantized_cache.py` (03b2424d), run by Don's local session on
lem (RTX 3090) with `dw.serve` stopped. sdnq 0.2.7, diffusers 0.41.0.dev0,
transformers 5.19.0, torch 2.14.1+cu130. Seconds:

| component | weights | fresh warm | fresh cold | cached warm | cached cold | GB on disk |
|---|---|---:|---:|---:|---:|---:|
| H3 transformer | int4 | 33.1 | 52.4 | 1.6 | 8.1 | 18.88 |
| H3 text encoder | int4 | 31.1 | 50.2 | 1.9 | 9.0 | 20.88 |
| H3 VAE | int8 | 3.3 | 10.1 | 0.5 | 1.6 | 3.15 |
| **H3 total** | | **67.5** | **112.7** | **4.0** | **18.7** | **42.91** |

**63.5 s saved per cold load with the page cache warm, 93.9 s cold**, against
the ~30 s gate.

### Stage A (#826): the cache module and the H3 modular path

- `dw/quantized_cache.py`: the env switch and root (`configure_at_startup`,
  called from `dw/serve.py` before workers spawn), the key, snapshot
  resolution, the free-space floor and start-up disk line, atomic save with the
  two-worker race, verified load (a corrupt entry is deleted and the load falls
  back to a fresh quantize), the version sweep, and `quantized_cache` events.
- The hook in `dw/pipeline_processors/components.py`: look up each SDNQ
  component, register hits through `update_components` (keeping `load_id`),
  `load_components` for the misses only, then save each miss before
  `sdnq_optimize`, LoRA and pruning.
- Snapshot resolution goes through `huggingface_hub.try_to_load_from_cache`
  (offline), not a hand-written `models--` layout (the architecture review's
  build-vs-buy finding).
- **Two key bugs found in acceptance, both in the quantizer config:**
  - `SDNQConfig.to_dict()` returns its own lists, and sdnq appends the modules
    it skips to `modules_to_not_convert` during the load, so the save key never
    matched a later lookup key. Fixed with a deep-copied `config_snapshot`.
  - `SDNQConfig.post_init` dedupes lists with `list(set(...))`, so their order
    follows each process's hash seed: the transformer's key differed between
    the two cards' workers. Fixed by sorting the set-valued arguments
    (`SET_ARGUMENTS`), pinned by a test across four `PYTHONHASHSEED`s.
- #825's three script fixes (argv match excluding its own ancestors, narrowed
  `allow_patterns`, fsync before eviction and page-touching inside the timed
  region).
- Docs: `QUANTIZATION.md` ("Cached on-the-fly"), `ACCELERATION.md`,
  `WORKER_GUIDE.md`, `SERVER.md` (both env vars), and the *Quantized component
  cache* row in `ARCHITECTURE.md`.
- lem opted in through its `.env` (Don).

**Accepted on lem** (M-F128-M-F134, M-F139, M-F140): an H3 run with all three
components hitting loaded in 72.3 s against 173.8 s for the miss-and-save run,
with per-component hits of 0.4-1.1 s; frames identical across hit and miss
runs; a text-encoder dtype change misses only that component; a LoRA applies
on a hit; the two-worker race gives one `saved` and one "already saved by
another worker".

### Stage B (#827): the per-component path (LTX-2.5)

- `quantized_cache.from_pretrained`, called from `load_component`'s
  `model_name` branch (the path `load_optional_component` takes). It shares
  `_lookup` with the modular path, so the two can't drift. A hit is placed on
  the config's `return_device` (`cuda` for the LTX transformer) and skips
  `download_watch`; a miss loads under it as before.
- **One bug found in acceptance:** transformers 5's `save_pretrained` writes a
  model in its checkpoint's original layout by reversing the class's weight
  conversion mapping, and `load_sdnq_model` reads names as they are. Gemma 4's
  vision-tower mapping made every LTX text-encoder entry fail to load, get
  deleted and re-save ~13 GB per run. Fixed in `_save_model` by passing
  `save_original_format=False` when `save_pretrained` accepts it. H3's
  Qwen3-VL has no such mapping and was never affected.
- The LTX template's "First load" wording now says the cache removes the
  quantize cost after the first load.

**Accepted on lem** (M-F135-M-F137, M-F141): loading 33.9 s and 38.0 s on a
hit against 61.6 s on the run that re-saved the text encoder; identical frames
across runs; mean step on a hit within 1.2× of the miss run; H3 still hits.
Entry sizes reported: transformer ~11.2 GB, Gemma text encoder ~13.0 GB on
disk.

## Bounces

| stage | architecture review | tester |
|---|---|---|
| #825 (0) | — | 0 |
| #826 (A) | 1 (build vs. buy: hand-written hub-cache layout) | 4: one environmental (lem not yet opted in), three on M-F128 (the two key bugs above; the first fix exposed the second) |
| #827 (B) | 0 | 1 (Gemma 4 entries saved in the original layout) |

Cost per stage isn't recorded here: the stage comments name no `usage:`
figures. The plan estimated A at $6-9 and B at $3-5.

## Deferred

- **The LTX rows of stage 0's table** (fresh/cached, warm/cold, measured with
  the fixed script and `dw.serve` stopped). lem's LTX snapshot is complete
  (`da0b6909`), so it can run whenever lem is idle. Until then the template's
  "about 10 minutes" fresh-quantize figure is the old, unmeasured one.
- **Listing, clearing and eviction:** #828.
- **Reporting the cache's on/off and free-space state** in `get_server_info`
  or `get_health`, so a consumer can check opt-in directly rather than infer it
  from events (the tester's note on #815; a natural fit for #828).
- **mps/cpu coverage:** mini-ai isn't opted in, so only the off arm (M-F139)
  ran there.
- A possible simplification from B's architecture review: if
  `save_sdnq_model` forwards keyword arguments to `save_pretrained`, the
  temporary `save_pretrained` override in `_save_model` can go.

## Risks that remain

- A stale hit gives silently wrong weights. The key carries a commit from a
  complete snapshot plus the sdnq, diffusers, transformers and torch versions,
  and the sweep deletes old-version entries at start. The residual is an sdnq
  default that changes without a version bump.
- Disk: ~43 GB per H3 config and ~24 GB per LTX config, with no eviction between
  version bumps beyond the floor.
- Opt-in means only lem exercises it; unit tests carry the rest.
