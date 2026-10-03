# LoRA catalog and recommender - design

Date: 2026-10-03. Status: approved in conversation, pending spec review.

## Intent

An agent (or Don) asks "I want a voxel-style image on Qwen-Image 2.1 - are
there LoRAs to consider?" and gets back entries it can use directly: the
repo, the weight file, the trigger word, the scale, and the evidence behind
the entry.

Two parts, built in order:

1. **The catalog** - LoRAs that have been used to good effect (and the ones
   that failed), keyed by base model, read through `list_loras` and written
   through `save_lora`.
2. **The recommender** - an opt-in `recommend_loras` tool that searches the
   Hugging Face Hub for adapters of that exact base model matching a style
   request. It runs only when called, and its Hub results are trials, never
   recommendations.

Why: LoRAs are the cheapest quality lever dw has (the 2026-10-03 H3 eval,
#585: Realism People at 0.7 was the best arm for +6 s), but what made a
trial succeed - trigger word, scale, base partition, stacking - currently
lands in memories and issues where a later session or an MCP agent never
sees it. The H3 skill is at its byte cap, so the skills cannot hold it
either. CLAUDE.md already says model knowledge belongs in `plugins/dw/` and
the catalog; this is that rule applied to adapters.

Success: `list_loras(model="qwen-image-2.1")` returns the catalog's entries
for `Qwen/Qwen-Image-2.1`; `recommend_loras(model="qwen-image-2.1",
query="voxel style")` returns catalog matches first, then Hub candidates
filtered to that exact base, safetensors-only, each carrying a step-ready
`as_lora` block and its warnings. No GPU time, no weight downloads.

## 1. The catalog as a library kind

A fourth `dw/library.py` kind, `loras`, beside workflows, prompts and assets,
reusing `LibraryPath` unchanged:

- Search path: the workspace's LoRA library (writable) shadows the shipped
  top-level `loras/` (read-only examples). Like the prompt library, the
  writable LoRA library belongs to the workspace root and a named workspace
  points back at it (`Workspace.loras`, following `Workspace.prompts`), so a
  LoRA promoted in one workspace is seen from all of them.
- One JSON file per LoRA, grouped by family folder:
  `loras/qwen-image/voxel-style.json`, `loras/minimax-h3/realism-people.json`.
- The kind's three strategies: name -> `<name>.json`; a new
  `validate_lora_path` in `dw/security.py` (modelled on
  `validate_prompt_path`, and added to the CodeQL sanitizer model alongside
  it); listing is the JSON walk `_JSON_KINDS` already does.
- The server reads the catalog and the worker never does, so the kind adds
  no worker environment variable.

### Entry format

Field names match a step's `loras` entry where they overlap, so an entry's
`model_name`/`weight_name`/`revision`/`scale.default` drop straight into a
workflow:

```json
{
  "model_name": "prithivMLmods/Qwen-Image-2.1-Voxel-Style",
  "weight_name": "<file>.safetensors",
  "revision": "<pinned sha>",
  "base_models": ["Qwen/Qwen-Image-2.1"],
  "workflow": null,
  "description": "Blocky voxel / Minecraft-like 3D look",
  "use_when": "The request asks for voxel, blocky 3D, or an isometric game-asset style",
  "trigger": "Voxel Style",
  "scale": { "default": 0.9, "range": [0.7, 1.0] },
  "stacks_with": [],
  "status": "trial",
  "evidence": [],
  "license": "apache-2.0",
  "tags": ["style", "voxel", "3d"]
}
```

(Illustrative - the weight file and sha are read off the Hub when the entry
is written.)

| Field | Required | Meaning |
| --- | --- | --- |
| `model_name` | yes | Hub repo id (repo-id pattern) |
| `weight_name` | no | Weight file; required when the repo holds more than one `.safetensors` |
| `revision` | no | Pinned commit sha; the shipped entries pin one |
| `base_models` | yes, non-empty | Exact base repo ids the LoRA is known to work on |
| `workflow` | no | Partition constraint, the `adapter_compatibility.py` vocabulary (`t2va`, `fl2va`, `ref2va`) |
| `description` | yes | What it does, one line |
| `use_when` | yes | Prose an agent matches a request against |
| `trigger` | no | Trigger word or phrase the prompt must contain |
| `scale` | no | `{default, range}`: `default` is the scale to start a trial at, `range` the span that has been tested |
| `stacks_with` | no | Catalog names it was tested stacked with |
| `status` | yes | `proven` / `trial` / `rejected` |
| `evidence` | proven: >= 1 | `[{job?, issue?, note}]`; a `rejected` entry's note says why |
| `license` | no | The Hub card's license id |
| `tags` | no | Free tags, used by `tag=` filtering and query ranking |

Matching is **exact** on `base_models`: no aliases, no family prefix
matching. A LoRA trained on one base is listed under another only after it
has been tried there and the entry edited. This is the same failure
`dw/adapter_compatibility.py` exists for - a LoRA on the wrong base or
partition usually loads without error and is quietly worse.

The schema is `dw/lora_catalog_schema.json`, checked on `PUT` and pinned by
a test over every shipped entry. The `proven`-needs-evidence rule is a
schema `if/then`.

### Seed entries

From what has already been run, with ids and files read from the existing
templates and #585:

- `minimax-h3/realism-people` - proven, `workflow: t2va`, scale 0.7, stacks
  with the 8-step turbo, evidence #585 jobs. Note the framing-tightening
  observation.
- The two lightx2v H3 turbo adapters - proven, one per partition.
- The LTX-2.5 IC-LoRAs the templates use - proven.
- Flux In-Context LoRA set and XLabs Realism - proven, from
  `workflows/templates/lora-styles.json` / `lora.json`.
- FastH3 (drozbay), Acc PDD (alibaba-pai), HyperFlow - `rejected`, each with
  the reason from #585 (the `.diff` full-weight keys; custom head-bank /
  time-embedding loaders).

## 2. Read/write surface

### HTTP (`dw/server/routes/loras.py`)

| Route | Does |
| --- | --- |
| `GET /api/loras?model=&workflow=&status=&tag=` | Filtered list of full entries |
| `GET /api/loras/{name}` | One entry |
| `PUT /api/loras/{name}` | Schema-validate, write to the writable root; saving over a shipped entry writes a copy |
| `DELETE /api/loras/{name}` | Workspace entries only (`ReadOnlyLibraryError` otherwise) |
| `GET /api/lora-schema` | The schema - its own route so an entry named `schema` cannot shadow it |
| `GET /api/loras/recommend?model=&query=&limit=` | Section 3 |

`recommend` is registered before `{name:path}` so it is not read as a name.

### Resolving `model`

`model` is either a Hub repo id or a catalog workflow name. A workflow name
resolves to the `from_pretrained_arguments.model_name` of each pipeline step,
plus the H3 `workflow=` partition where the step declares one. `dw/lora_catalog.py`
owns this. When nothing matches, the response carries the resolved repo ids
(`"resolved": [...]`) so the agent sees exactly what was looked up. An
unknown workflow name and a malformed repo id are 400s naming the value.

`workflow=` filtering: an entry with a `workflow` matches only steps of that
partition; an entry with none matches every step of its base.

### MCP (`dw_mcp/loras.py`)

- `list_loras(model?, status?, tag?)` - full entries (small, a handful per
  model), so no `get_lora`. Description leads with what to act on: proven
  and rejected LoRAs for a base model or workflow; `use_when` says when,
  `trigger`/`scale` say how; points at the `LORAS.md` *LoRA catalog*
  section.
- `save_lora(name, entry)` - promotion. `WRITES` annotation, like
  `save_prompt`.
- Delete stays on HTTP.

`SURFACE_BUDGET` in `tests/test_mcp_server.py` is raised by the measured
cost of the three new tools (this section's two plus section 3's), with a
dated comment in the style of the existing ones. Descriptions stay terse
and point at the guide.

### Skills

The H3 and LTX skills replace inline LoRA notes that the catalog now holds
with a one-line pointer to `list_loras`, keeping the rules that are not
catalog facts (the partition rule, step counts). `tests/test_plugin_skills.py`
caps still hold.

## 3. The recommender

`recommend_loras(model, query, limit=8)` -> `GET /api/loras/recommend`. The
only place dw searches the Hub on its own. It runs only when called, spends
no GPU and downloads no weights, so it takes no `acknowledged_cost`; its
description says it queries the Hub.

1. **Resolve** `model` to base repo ids (section 2).
2. **Catalog first.** Matching catalog entries, labelled `source: "catalog"`,
   ranked by `query` terms found in `use_when`/`tags`/`description`
   (case-insensitive term count), `proven` before `trial`. `rejected` entries
   are not offered here (they are used in step 5).
3. **Hub search.** `HfApi.list_models(filter="base_model:adapter:<repo>",
   search=<query>, sort="downloads")` per base repo. Hub `search` matches the
   repo id only, so a multi-word query also runs per word (words under three
   characters and the word "style" skipped); results are merged and
   de-duplicated, ordered by downloads. The exact base comes from the filter.
4. **Inspect** the top `limit` candidates, no weight download:
   - files: `model_info(files_metadata=True)`. A repo with no `.safetensors`
     is dropped (pickle `.bin` only). More than one `.safetensors`: listed,
     `weight_name` unset, warning `multiple_weights`.
   - header: `get_safetensors_metadata` (a range read). Key prefixes
     classify `format`: `diffusers` (`transformer.`/`unet.` with
     `lora_A`/`lora_B`), `kohya` (`lora_unet_`/`lora_te`), `full_weight`
     (`.diff`/`.diff_b` keys - the FastH3 failure, warning
     `will_not_load`), else `unknown`.
   - card: `license`, `gated`, `instance_prompt` (-> `trigger`),
     `downloads`, `likes`, `lastModified`, `sha`.
5. **Rejected suppressed.** A Hub repo the catalog marks `rejected` comes
   back as `status: "rejected"` with the entry's reason, never as a
   candidate.

Each Hub candidate:

```json
{
  "source": "hub",
  "status": "candidate",
  "model_name": "...", "trigger": "...", "license": "...", "gated": false,
  "downloads": 0, "likes": 0, "last_modified": "...", "format": "diffusers",
  "as_lora": { "model_name": "...", "weight_name": "...", "revision": "<sha>", "scale": 1.0 },
  "warnings": ["no_license", "gated", "multiple_weights", "will_not_load", "unknown_format", "stale"]
}
```

`stale`: `lastModified` older than the base repo's own `lastModified`
(trained on an earlier revision of the base). `as_lora` is omitted when
`weight_name` cannot be chosen.

The response says in words that Hub results are trials; a trial that works
is promoted with `save_lora` (`status: "trial"` -> `"proven"`, its job in
`evidence`).

**Failure.** Hub calls are bounded by a timeout (the `plan.py`
`SIZE_LOOKUP_TIMEOUT` pattern). Hub unreachable, rate-limited or timed out:
the catalog results are still returned, with `hub_error` naming what failed.
A per-candidate inspection failure drops to that candidate's `warnings`, not
the whole call. Repo ids returned by the Hub are validated against the
repo-id pattern before reuse; no URL is constructed from Hub data; the
server's own HF token is used, so gated repos appear with `gated` set.

## 4. Modules, tests, docs

- `dw/lora_catalog.py` - schema loading, `model` resolution, matching,
  ranking. No network.
- `dw/lora_hub.py` - Hub search, inspection, format classification. The
  only module that imports `HfApi` for this.
- `dw/server/routes/loras.py`, registered in `dw/server/app.py`.
- `dw_mcp/loras.py`, tools registered in `dw_mcp/server.py`.

Tests (all Hub calls mocked; no network in the suite):

- every shipped entry validates; `proven` without evidence is refused
- the `loras` kind: shadowing, copy-on-save over a shipped entry, delete of a
  shipped entry refused, named workspace sees the root's library
- `validate_lora_path` refuses traversal
- `model` resolution from a workflow name, including an H3 step's partition;
  unknown workflow and malformed repo id are 400s
- exact match: an entry for one base is not listed for another
- `workflow` filter: a `ref2va` entry is not listed for a `t2va` step
- format classification over fixture headers (diffusers, kohya, `.diff`,
  unknown)
- a pickle-only repo is dropped; multiple weights leave `weight_name` unset
- a catalog-`rejected` repo comes back rejected, not as a candidate
- Hub failure returns catalog results plus `hub_error`
- the MCP surface budget test, raised
- `recommend` is not shadowed by `{name:path}`

Docs:

- `docs/LORAS.md` - a *LoRA catalog* section (entry format, statuses,
  promotion) and a *Finding LoRAs on the Hub* section (what
  `recommend_loras` filters and what its warnings mean).
- `docs/MCP.md` - the three tools.
- `docs/ARCHITECTURE.md` - rows for "LoRA catalog matching is exact on base
  repo" and "the Hub is searched only by `recommend_loras`", each naming its
  test.
- `docs/SECURITY.md` - `validate_lora_path`.

## Out of scope (follow-ups)

- A trial template: same seed and prompt, with and without the LoRA (and two
  scales), side by side.
- A UI LoRA page over the HTTP routes.
- Response caching for `recommend`.
- Access probing (#186) for gated candidates.
- A periodic check that pinned catalog repos still exist.
