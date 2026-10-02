# dw_mcp assessment (2026-10-01, develop 2b2b82ce)

The engine stabilization (gates 0-4) measured `dw_mcp` but never read it
module by module; the 2026-09-28 assessment's whole verdict was "correctly a
thin HTTP client; only the shared-session defect (B6)". This is that read:
all 18 modules (4,597 lines), against the owners the stabilization put in
`dw/`.

## Verdict

Mostly true. Nothing in `dw_mcp` parses reference prefixes, walks a
definition, does cost or VRAM math, computes run versions or handles sample
rates: those all come from the server. But `dw_mcp` cannot import `dw`
(it stays torch-free), so every rule it does know is a second copy, and it
knows about twenty. All of them agree with their owners today and only one
is pinned by a test; `tests/test_mcp_*.py` run against `httpx.MockTransport`,
so they test `dw_mcp`'s idea of the server, not the server. The one real
bug is the stabilization's pattern in miniature: a fix landed in one of two
twin functions.

## Findings

| # | Severity | Finding | Evidence | Fix |
| --- | --- | --- | --- | --- |
| M1 | bug, medium | #389's per-call `workspace` reached `media._remote_root` but not its twin `assets._remote_roots`. A mounted session pinned to `ep4` calling `upload_asset(file_path="<default root>/assets/x.wav", workspace="default")` is confined to ep4's roots and refused. Fails closed. | `dw_mcp/assets.py:52`, `:353`; `dw_mcp/media.py:494` | Pass `workspace` through; merge the two root/confine pairs (M8) |
| M2 | structure, medium | `dw` imports `dw_mcp`: `dw/run.py` is a client of `dw.serve` through `dw_mcp.client`, and keeps a third `TERMINAL_STATUSES`. `dw/media.py:37` says the packages do not import each other. No seam-map row. | `dw/run.py:17`, `:36` | Either a map row naming the edge, or move the HTTP client to a torch-free module both import |
| M3 | duplication, medium | Image and frame handling copied from the server: `_crop_box` clones `dw/media_frames.resolve_crop_box`; the one-selector check and `_fit` repeat `dw/server/routes/media.py`; `MIN_DIMENSION`, `MAX_RETURNED_BYTES` twin server constants. `get_output_image` downloads and decodes the whole file client-side; `_fit_tiles_within_budget` re-encodes tiles the server already made. | `dw_mcp/media.py:96`, `:136`, `:274`, `:364` | A server image route with `max_dimension`/`crop`/`max_bytes` and a byte budget on `/frames`; `media.py` becomes call-and-reshape, and the UI can use the same routes |
| M4 | duplication, medium | `get_gallery_metadata` writes audio-QC thresholds (-0.5 / 0.0 / -40 dBFS) owned by `dw/audio_qc.py`, and the Music 3 "within 0.2 s of the ceiling" rule owned by `plugins/dw/skills/minimax-music3/SKILL.md` - model knowledge outside plugins and the catalog. | `dw_mcp/catalog.py:304-361` | The metadata route emits its findings and next step, as assess already does |
| M5 | duplication, low | Twin constants, none pinned except `MAX_DECODE_PIXELS`: upload extensions and size cap (`routes/assets.py`), `LOOPBACK_HOSTS` (`server/netinfo.py`), `TERMINAL_STATUSES` (`job_record.py`), `DEFAULT_WORKSPACE` (`workspace.py`), `ASSESSMENT_PROBES` (`server/assess.py`; redundant, the server returns the same 400). Three owner comments name `dw/server/app.py` or `dw/security.py`, which no longer own them. | `dw_mcp/assets.py:18-31`, `client.py:17`, `:130`, `diagnose.py:15-18`, `media.py:426` | One test (tests may import `dw`) asserting each twin equals its owner; delete `ASSESSMENT_PROBES`; fix the comments |
| M6 | duplication, low | Engine word lists in tool text: shapes and traits, reserved prompt-text prefixes (`references.RESERVED_TEXT`), reserved workspace names, estimate bases, `move_job` directions, job states. A test checks some words appear, not that the lists match. | `dw_mcp/server.py:65`, `tools_catalog.py:26`, `:251`, `tools_authoring.py:36`, `:76`, `:192`, `tools_jobs.py:201` | Pin the rendered descriptions against the owners' tuples |
| M7 | misplaced logic, low-medium | Work the server should own: `save_workflow` patch mode is GET + merge + PUT, not atomic, so a UI save in between is lost; `delete_output(job_id=)` derives the run dir; the acknowledgement body strips null repos because admission types `downloads: List[str]`; the 409 re-acknowledge body is rebuilt client-side; `workspaces.server_info` re-derives directories `/api/server` already scopes. | `dw_mcp/authoring.py:96`, `media.py:477`, `diagnose.py:57`, `client.py:426`, `workspaces.py:195` | A `PATCH` workflow route; `DELETE /api/jobs/{id}/run`; the server tolerates nulls and sends the body in its 409; delete `server_info`'s re-derivation |
| M8 | structure, low | Duplicates inside `dw_mcp`: the root/confine pairs in `media.py` and `assets.py` (source of M1); three field-projection helpers; the base64-size formula three times; two loopback sets that differ; two copies of the name/path/inline alias parsing; the acknowledgement gate in six places while the map's *Spending needs consent* row names only `diagnose.py`. `diagnose.py` is mostly queue operations. | `dw_mcp/media.py:523`, `assets.py:118`, `:164`, `authoring.py:39`, `diagnose.py:104` | Consolidate each pair; widen the map row to every gated tool |
| M9 | defects, low | `get_output_frames` with `hear` budgets frames and audio separately, so a reply can reach about 2x the cap; `_upload_inline` decodes before checking the size cap; `_probe` matches "401" in text, not `status_code`; a bad `DW_MCP_MAX_WAIT_SECONDS` fails the import; `get_gallery_metadata` reads `job["id"]` unguarded. | `dw_mcp/media.py:317`, `:341`, `assets.py:414`, `__main__.py:112`, `diagnose.py:27`, `catalog.py:312` | Each a one-line fix with a test |

Complexity over 10 (ruff C901): `media.get_output_frames` 16,
`client._format_detail` 13, `assets._remote_roots` 11,
`tools_media.get_output_frames` 11. None is over the ratchet's limit.

## What is already sound

- `dw_mcp` stays torch-free and only `server.py` and `tools_*.py` import the
  MCP SDK, both pinned (`docs/ARCHITECTURE.md`, *MCP*).
- B6, the shared session pin on the mounted surface, is documented as
  deliberate and warns on change; a per-call `workspace` reaches every tool
  but M1's.
- Drift that is caught today: `MAX_DECODE_PIXELS`
  (`tests/test_security_decoder_bombs.py`), the tool listing over the real
  mount (`tests/test_server_mcp.py`), symlink confinement through the real
  app (`tests/test_security_symlinks.py`).

## Proposed pass

Three stages, each small. The ratchet already covers `dw_mcp`, so none needs
new tooling.

1. **Fixes and pins.** M1, M9; the twin-constant test and word-list pin
   (M5, M6); stale comments; the M2 map row; widen the consent row (M8).
   No server change.
   Done 2026-10-01. M1 fixed. M9 fixed except `_upload_inline`'s decode
   order, left as it is: the base64 text is already in memory as the tool
   call's argument, so checking first saves nothing. `ASSESSMENT_PROBES`
   deleted; the server's 400 reaches the caller. `tests/test_mcp_twins.py`
   pins every M5 constant (and `dw.run`'s copies) and every M6 word list
   to its owner, and checks every tool that takes `acknowledged_cost`
   refuses without it. The map has rows for the copies, for `dw.run` as a
   client, and the consent rule over all seven gated tools.
2. **Move logic to the server.** M3, M4, M7: new or widened routes, then
   `dw_mcp` calls them. The UI is the second consumer, so this stage is
   shared with the UI pass.
3. **Consolidate inside `dw_mcp`.** M8's pairs, once stage 2 has removed the
   code some of them guard.
