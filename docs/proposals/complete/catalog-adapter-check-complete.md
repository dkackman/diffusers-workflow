# Catalog-backed H3 adapter partition check (#870)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-14). Plan v2 approved by Don as "build smaller" on 2026-10-14;
one stage, #888 (A), verified on lem.

## The report

On the H3 quality matrix (lem, workspace `h3-quality`), the fal Realism
People LoRA (`fal/MiniMax-H3-Realism-People-LoRA`,
`h3-realism-people-t2v-i2v-r2v.safetensors`) was put on a Ref2VA
(`transformer_ref`) step to fix faces. It was inert and cost a full render
to find out. `dw/adapter_compatibility.py` decided from the filename tokens
`ref2v` / `fl2v`; this name has neither (its `r2v` even suggests reference
support), so validate only warned "names neither". The file's safetensors
`__metadata__` says `base_model: minimax-h3-fl2va`. On FL2VA/I2VA it works
(R3: 4/5 faces).

The ask was to read that header during validation and prefer `base_model`
over the filename.

## Decisions

- **Plan v1: build smaller.** The catalog already said it:
  `loras/minimax-h3/realism-people.json` had `"workflow": "t2va"`,
  `"status": "proven"`. The check just never asked the catalog. Wire the
  catalog in; cut the header read, because:
  1. it is I/O on the hottest validation path (every validate, admit,
     `for_each` member and composed child), which is free and offline today;
  2. a local-path read during validate is a directory-existence oracle
     (`plan.py` refuses exactly that);
  3. a ranged Hub GET breaks "`lora_hub.py` is the only place dw searches
     the Hub";
  4. `__metadata__` is uploader-written: it shouldn't turn a warning into a
     refusal;
  5. only one vendor key (`base_model`) was evidence.
- **Q1:** a catalog entry that disagrees with the filename token warns,
  never refuses. **Q2:** the revision is ignored. **Q3:** not needed (if a
  `save_lora` pre-fill comes back, trust `base_model` only).
- **Build stop on v1 (#888, 2026-10-14):** v1 assumed the catalog was
  layered per workspace. It isn't: `server_lora_library(state)` is one
  library shared by every workspace, so one agent-written trial entry would
  refuse a file server-wide.
- **Don's ruling (a) → plan v2:** keep the one shared catalog, add no
  layering. **Only a `proven` entry can refuse**; a `trial` or `rejected`
  entry that conflicts with the step's partition warns, naming the entry
  and its status. No new evidence requirement on `proven`. Scope narrower
  than v1.
- **Non-goals held:** no header reads, no Hub request from validation, no
  non-H3 base-model checks, no change to the filename rule, no per-workspace
  catalogs, no MCP/REST/syntax change.

## What was built

**Stage A, #888** (server + data + docs), `feat/870-A-catalog-adapter-check`
(27674293, 687e288a), merged to `develop` as `9a210585`:

- **`adapter_compatibility.catalog_index(entries)`**: `(model_name,
  weight_name) -> (entry, workflows, status)`, revision ignored. With no
  index the result is the old rule.
- **Partition, not workflow** (`_catalog_partition`): all of
  `{t2va, fl2va}` is `transformer`, `ref2va` is `transformer_ref`, a mix
  decides nothing.
- **Status order:** a `proven` conflict is an error naming the entry
  ("catalog entry 'minimax-h3/realism-people' (proven) lists it for 't2va,
  fl2va', which denoises against 'transformer'…"); a `trial`/`rejected`
  conflict is a warning naming the entry and status; an agreeing entry
  clears "names neither" but a `trial` one never lifts a filename-token
  error; a `proven` entry that disagrees with a token is a warning naming
  both.
- **Callers:** `deps.lora_index(state)` is built from the shared
  `catalog_entries(state)` (moved from `routes/loras.py` to `deps.py`, so
  `list_loras` and validation read one function), cached on
  `state.lora_indexes` keyed on the listing's (name, path, mtime). Passed by
  `admit_for` (`/api/validate`, submit, rerun, enhance-admit) and the save
  path in `routes/library.py`. The CLI `dw.validate`, the worker and
  `warn_adapters` pass none: the server is stricter than the worker, never
  looser.
- **Data:** `realism-people.json` `workflow` is `["t2va","fl2va"]`,
  `use_when` says "not reference-to-video", new `evidence` note (#870); it
  stays `proven`.
- **Docs:** the module docstring, `docs/ARCHITECTURE.md`, `docs/SERVER.md`,
  `docs/LORAS.md` (a `proven` entry's `workflow` is enforced on the server;
  a `trial` one only warns; promoting is what makes it refuse), and the
  `minimax-h3` skill's `checkpoints-and-loras.md` ("validate refuses it on
  `ref2va`").
- **Tests:** `tests/test_h3_adapters.py` (`test_a_proven_*`,
  `test_a_weaker_*`, `test_no_index_is_todays_rule`, null/mixed fallback,
  `arguments.<name>` path, `test_one_build_per_listing_for_every_member`,
  `test_a_trial_entry_warns_and_promoting_it_refuses`); the
  `workflows/**/*.json` sweep also runs with the packaged catalog.

**Deviations from the plan**, both accepted at verify:
- Two entries covering the same file: the index keeps the stronger status
  (proven > trial > rejected).
- A `trial` conflict on a no-token file gives the entry's warning *in place
  of* "names neither", not both.

## Bounces per stage

| Stage | Bounces | Cause |
|---|---|---|
| A, #888 | 0 (1 build stop) | The build stopped on v1's per-workspace catalog assumption; re-planned as v2 (Don's ruling (a)), re-decomposed and re-specced. Verified first time on v2. |

No stage comment names `usage:` figures, so per-stage cost is not recorded.
The plan estimated ~$3-5.

## Verification notes

- Verified over MCP on lem `develop` @ f73088a5: M-F142-M-F145, M-F147,
  M-F152 and M-F153 pass. M-F142(c): `run_workflow` with Realism on
  `reference-to-video` is refused at `arguments.lora_weight_name` with no
  job queued.
- M-F144(b): the `ref2v` turbo on `fl2va` is refused by the proven catalog
  message (`minimax-h3/turbo-reference (proven)`), not the filename message;
  v2 lets a proven entry decide.
- M-F146 (v1 workspace layering) contradicts v2 and was not run; retiring it
  is dkackman/harnest#122.
- mini-ai (MPS) was not checked, and its restart is Don's.

## Deferred

- **Reading safetensors headers**, reopened only by a second render lost to
  a LoRA the catalog doesn't cover, or a run of uncatalogued H3 trials that
  keep saying "names neither". It would go into `save_lora` /
  `recommend_loras` (which already read Hub headers via `lora_hub.py`) to
  *pre-fill* an entry's `workflow`, trusting `base_model` only; the
  validator stays free.
- **Edges with no shipped entry** (a `proven` entry against its own token, a
  mixed list, a `rejected` entry with a `workflow`): unit-tested only, since
  probing them over MCP needs a `save_lora` write to the shared catalog.
- **CLI `dw.validate` stays on the filename rule** (documented; the safe
  direction).
