# Bulk download of a job's outputs: export_job reports the zip's real gating (#592)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-05). Plan v2 was approved by Don on 2026-10-05 (Q1 and Q2 kept
their defaults). Stage 1, #595, shipped the same day at `develop` @
`69a3ac8a` and was verified on its first pass.

## The idea

A field report from an agent job on mini-ai: bringing 41 shots and 7 music
cues back to the Mac meant reading each job's file list and curling every
file. The issue asked for a "fetch everything from this job" call, and
whether it should extend `export_job` or be a new tool, and how large
binary transfer should work over MCP.

## Verdict

**Build smaller: one stage, no new tool.** The bulk download already
existed. `POST /api/jobs/{id}/export` builds `exports/<id>/` (outputs,
inputs, assets, workflow, manifest, job record, README) and returns
`zip_url` (`/exports/<id>.zip`), and `/exports` is ungated by design, beside
`/outputs` (`dw/server/routes/files.py`, pinned in
`tests/test_security_auth.py`, stated in `docs/SERVER.md`).

What broke it was one field. The export response set
`auth_required = bool(state.api_token)`: whether the *server* has a token,
not whether the *zip URL* needs one. `dw_mcp/exports.py` then told the agent
"do not fetch it; hand open_url to the person". On any token-bearing server
(mini-ai, lem) an agent was told not to fetch a URL that needs no token,
which left the per-file curl loop the report describes. #353 had introduced
the flag on the premise that the zip sat behind the token; the code said it
didn't.

**Binary transfer over MCP itself: don't build.** A tool result lands in
the agent's context, so base64 video would swamp it (#203 capped inline
upload at 4 MB for the same reason), and MCP has no channel to the client's
disk (`download_output` over a mounted endpoint writes on the server).
Over MCP, big files travel by URL; the tool's job is to hand over one URL
that works.

## Decisions (Don, 2026-10-05)

- **Q1. Keep `/exports/*.zip` ungated?** Yes. It serves nothing `/outputs`
  doesn't already serve ungated. Don required the test pinning "token
  configured → `auth_required` false → unauthenticated GET returns 200",
  so the field can't drift from the route again.
- **Q2. Keep `auth_required` (always false today)?** Yes: no shape change,
  and it guards a future gating of `/exports`.
- **Q3. Surface removal:** none found.

## What was built (stage 1, #595)

- **Server** (`dw/server/routes/jobs.py`): the export response sets
  `auth_required = False`, with a comment tying it to `files.py`'s ungated
  list. Response shape unchanged, so the OpenAPI dump didn't move.
- **MCP `next`** (`dw_mcp/exports.py`): when `auth_required` is false,
  `next` says to fetch `open_url` with whatever HTTP you have (it needs no
  token; prefix a relative URL with the server address), and unpack into
  `exports/` under the working directory without pre-creating the job-id
  folder. Only an agent without HTTP is told to give the user `open_url`.
  The `true` branch is unchanged, as the forward guard.
- **MCP description** (`dw_mcp/tools_jobs.py`): "do NOT fetch it" is gone.
  It shrank from 1429 to 1313 characters, so `SURFACE_BUDGET` was left
  alone.
- **Skills:** `script-to-video` and `series-episodes` end their delivery
  sections with "take the project home": `export_job` once per job, then
  fetch each zip into `exports/`. `minimax-music3/references/keeping-a-run.md`
  lost its stale "can't attach the token" wording. `minimax-h3` already
  keyed on the field; `minimax-music3/SKILL.md` and `ltx-2.5` had no export
  wording.
- **Docs:** the `export_job` row in `docs/MCP.md`; a paragraph in
  `docs/REMOTE.md` saying the zip needs no token and an agent with HTTP
  fetches it itself.
- **Tests:**
  - `tests/test_server_exports.py::…test_with_a_token_the_zip_is_not_auth_required_and_opens_without_one`:
    with a token, `auth_required` is false and a token-less GET of
    `zip_url` returns 200 with `<id>/manifest.json` inside; a token-less
    export POST is still 401 (Don's Q1).
  - `tests/test_mcp_exports.py`: both `next` branches.
  - `tests/test_mcp_server.py::test_export_job_description_says_to_fetch_the_ungated_zip`
    replaces `…_is_auth_aware`.

### Deviations from the plan

- **No plugin version bump.** `plugins/dw/README.md` ties the plugin's
  version to the engine's, bumped only by the release script. The tester's
  plugin tree follows `origin/develop`, so the skill text was live without
  one.
- The no-HTTP fallback reads "give the user open_url to open" rather than
  the old "hand open_url to the person".

## Verified

On lem (token set) at `69a3ac8a`, cases C-F180–C-F182: `export_job` of a
finished job returns `auth_required: false` and the fetch `next`; unknown
id, running *and queued* jobs, and a repeat without `overwrite` are all
refused as before; `overwrite=true` replaces the export identically; the
served description and skills no longer steer away from fetching. The
token-less GET of the zip was not checked over MCP (the tester has no HTTP
client); it rests on the pinned unit test above.

Left behind: no MCP tool reaches `exports/`, so each run of C-F180/C-F181
leaves an `exports/<id>/` on the test server.

## Bounces per stage

- **Stage 1 (#595): 0.** The architecture review passed and the tester
  verified on the first pass.

No `usage:` figures were recorded on the stage, so cost is left out (the
plan estimated about $3–4).

## Deferred

- **A multi-job bundle** (`export_jobs([...])` or a series zip): a new tool,
  a new naming scheme under `exports/`, a UI flow and new symlink cases,
  about 4 stages and pressure on the surface budget. **Bring it back** when
  a field report shows a project spread over enough jobs that one zip per
  job is the friction; `script-to-video`'s one-job-per-shot film is the
  likely trigger.
- **An outputs-only zip** (no `assets/` and `inputs/` copies, so no doubled
  server disk): `POST /api/gallery/archive` does name-based zips but is
  gated, has no job filter and no MCP tool. **Bring it back** on a report
  of server disk pressure from `exports/`.
- **Removing an export over MCP:** no tool reaches `exports/` today, so
  exports accumulate on the server. Not in scope; worth a fix if the
  outputs-only zip above comes back, or if the test server's `exports/`
  grows.
