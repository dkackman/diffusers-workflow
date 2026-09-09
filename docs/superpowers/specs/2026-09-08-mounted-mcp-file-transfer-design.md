# Mounted MCP file transfer: design

The MCP tool surface is built once (`dw_mcp.server.build_server`) and served two
ways: the stdio `dw-mcp` on the user's machine, and mounted at `/mcp` inside
`dw.serve --mcp` on the GPU box. Two tools move files between "here" and the
engine - `upload_asset` and `download_output` - and both assume "here" is the
user's machine. Over the mounted transport it is the GPU box, so `upload_asset`
cannot see the file the agent names and `download_output` writes into the
server's working directory. On 2026-09-08 a cold agent, told to upload a voice
reference, tried to ssh to the GPU box to get around it.

## Goal

An agent on either transport can put a small input file into the workspace's
asset library and get an `asset:` reference back, with the tool itself saying
what works on this transport when a form does not. A file the engine's decoders
can read is not refused at the upload gate.

## Non-goals

- Large uploads over MCP. Anything past the inline cap is told to use the HTTP
  route; the browser UI and `curl` already cover it.
- Downloading media inline over MCP. `get_output_image` / `get_output_text`
  and the gallery `url` remain the way content reaches the conversation.
- Changing the stdio behaviour of either tool, except for the shared additions
  below.

## Global constraints

- One tool name per job. `upload_asset` gains a second form rather than a
  sibling tool, so a skill can say "upload_asset" on both transports.
- The MCP server knows which transport it is on: `build_server(client,
  mounted=False)`; `dw/server/mcp_mount.build_mcp_app` passes `mounted=True`.
  Nothing else consults it.
- Refusals explain. A form that cannot work on this transport returns a
  `DwApiError` whose message names the form that does; it never falls through
  to a bare "No such file".
- Inline content is bounded: 16 MiB decoded (`MAX_INLINE_UPLOAD_BYTES`). A
  voice reference, a portrait or a short clip fits; a video of any length does
  not, and the message says so and gives the route.
- The extension gate matches the decoders. What `POST /api/uploads` accepts is
  what `load_audio` / the H3 audio reference can read.

## 1. `upload_asset`, two forms

### Signature

```
upload_asset(file_path: str | None = None,
             content_base64: str | None = None,
             filename: str | None = None) -> dict
```

Exactly one of `file_path` or `content_base64`. `filename` is required with
`content_base64` (it carries the extension the gate checks) and ignored with
`file_path` (the basename is used, as now).

### Rules

- `file_path` form: unchanged when `mounted=False`. When `mounted=True` it is
  refused before touching the filesystem: "This MCP endpoint runs on the
  engine's machine, so `file_path` names a path there, not where you are. Pass
  the file as `content_base64` with `filename` (up to 16 MiB), or POST the raw
  bytes to `<base_url>/api/uploads?filename=<name>&workspace=<ws>` with the
  same bearer token." `<base_url>` and `<ws>` are filled from the client, so
  the message is a command the agent can run.
- `content_base64` form, both transports: decode (refuse on padding/alphabet
  error with a plain message), check the decoded size against
  `MAX_INLINE_UPLOAD_BYTES`, check `filename`'s extension against
  `ALLOWED_UPLOAD_EXTENSIONS`, then `client.post_bytes("/api/uploads", body,
  params={"filename": filename})` exactly as the path form does. Same return
  shape: `reference`, `url`, plus `bytes` (decoded size) so the agent can see
  the file arrived whole.
- The docstring is set per transport in `build_server` before registration
  (`fn.__doc__ = ...`), so a mounted session reads the caveat in the tool
  listing rather than discovering it on failure. Both docstrings name the
  extension list and the inline cap.
- Annotation stays `WRITES`.

### Module

`dw_mcp/assets.py`: `upload_asset(client, file_path=None, content_base64=None,
filename=None, mounted=False)`; the path form's body moves into a private
`_post_upload(client, body, filename)` shared by both. `MAX_INLINE_UPLOAD_BYTES`
beside `MAX_UPLOAD_BYTES`. `dw_mcp/server.py` passes `mounted` through and
selects the docstring.

## 2. `download_output` when mounted

### Rules

- `mounted=False`: unchanged.
- `mounted=True`: the tool is still registered (a hidden tool cannot explain
  itself) but refuses before contacting the server, returning a `DwApiError`
  whose message gives the gallery `url` for `name` - the same one `list_gallery`
  reports, with the workspace selector - and a one-line `curl -H
  "Authorization: Bearer <token>" -o <basename> <url>`. The token is not
  echoed; the placeholder is literal.
- The docstring's existing transport paragraph is kept and moved to the front
  of the mounted variant.

This closes the 2026-09-08 memory note: a cold agent no longer drops mp4s into
the server's working directory.

## 3. The extension gate follows the decoders

### Today

`dw/security.py` `ALLOWED_AUDIO_EXTENSIONS = {.wav .mp3 .flac .ogg}`; the same
set is copied into `dw_mcp/assets.py`. `dw/tasks/audio_utils.load_audio` reads
audio extensions with `soundfile` and routes video extensions through PyAV.
diffusers' `MiniMaxH3AudioReference.from_file` decodes with PyAV and takes
`.m4a`, `.aac`, `.opus`. A phone recording (`.m4a`) is refused at the gate
though the reference it was meant for would have decoded it.

### Rules

- `ALLOWED_AUDIO_EXTENSIONS` gains `.m4a`, `.aac`, `.opus`.
- `load_audio` tries `soundfile` first and, on `soundfile.LibsndfileError` or
  an extension libsndfile does not handle, decodes through PyAV the way the
  video branch already does, returning the same `(samples, channels)` float32
  and sample rate. The audio tasks therefore accept everything the gate does.
- `dw_mcp` does not import `dw` (it is installable alone), so the copy stays,
  and a test asserts the two sets are equal.

## 4. Tests

- `tests/test_mcp_assets.py`: the `content_base64` form posts decoded bytes to
  `/api/uploads` with the given filename and returns `reference`/`url`/`bytes`;
  over the cap is refused naming the route; bad base64 is refused plainly;
  both forms given, or neither, is refused; `filename` without an allowed
  extension is refused before any request; the `file_path` form is refused
  when `mounted=True` and the message contains the base URL, the workspace and
  `content_base64`.
- `tests/test_mcp_server.py`: `build_server(client, mounted=True)` registers
  `upload_asset` whose description contains the transport caveat and
  `build_server(client)` does not; `download_output` mounted refuses with the
  gallery url and a `curl` line and makes no HTTP call.
- `tests/test_server_mcp.py`: a round trip through the mounted app - a small
  wav as `content_base64` lands in the selected workspace's `assets/` and
  `asset:<name>` resolves in a validated workflow.
- `tests/test_assets.py` / audio task tests: `.m4a` passes the gate;
  `load_audio` decodes an `.m4a` fixture (generated in the test with PyAV, no
  binary checked in) to the same shape and rate as its wav twin; the two
  extension sets are equal.

## 5. Docs

- docs/MCP.md: a "Where files live" section - the two transports, which
  machine each tool's paths mean, the inline form and its cap, the `curl`
  forms for both directions. Replaces the scattered sentences in the two tool
  docstrings' current prose.
- docs/SERVER.md: the widened audio list where uploads are described.
- `plugins/dw/skills/minimax-h3/SKILL.md` "Run and judge" step 5 gains one
  sentence: a recorded voice reference reaches the server with `upload_asset`
  (`content_base64` over a `dw.serve --mcp` endpoint); the skill already
  explains the export-zip fetch in the other direction. Version-pinned by the
  existing size test only; no numbers are stated.
- `.remember` / memory: the 2026-09-08 workaround note becomes "fixed in
  <commit>".

## Sequence for the plan

1. `mounted` flag through `build_server` and `mcp_mount`; per-transport
   docstrings; `file_path` refusal when mounted (section 1, first bullet).
   Ships alone if needed - it would have prevented the ssh attempt.
2. Extension gate and `load_audio` fallback (section 3).
3. `content_base64` form (section 1, rest).
4. `download_output` mounted refusal (section 2).
5. Tests alongside each step; docs last.

Do not restart the GPU box's `dw.serve` to pick this up while a drill job is
running - a restart kills the worker mid-job.
