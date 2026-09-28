# Releasing

## Unreleased

This project has no standing release-notes file - GitHub auto-generates
notes from commits at tag time (see below). This section is a scratch pad
for items a branch's author wants the next release note to name; clear it
when a release ships.

### 0.5.0

<!-- Drafted by the release agent, model claude-opus-5-5 via the anthropic provider, from v0.4.0..f272188 (origin/develop). -->

Most of this range landed on `develop` without PRs, so the auto-generated
notes name only the Mac PR (#470) and a deploy-script PR (#476). Paste this
section into the GitHub release body once the tag has published (`gh release
edit v0.5.0 --notes-file ...`).

**Breaking and behaviour changes**

- `templates/ltx2/two-stage` no longer has `full_width`/`full_height`. They
  never changed the output, which is always exactly 2x `width`/`height`. A
  call that still passes them is refused as an unknown variable (#506).
- Every LTX-2.5 template refuses a `width` or `height` that isn't a multiple
  of 32 at validate and at `run_workflow`, before the pipeline loads (#505).
- The MiniMax-H3 Ref2VA VRAM ceiling is now projected per step, after
  `for_each` expansion, and adds 1 GB for each non-null reference. At
  1344x768 on a 24 GB card the limits are 243/209/175/141 frames with 1/2/3/4
  references. Calls that used to pass are now refused, for example 209
  frames with 3 references. A `for_each` refusal names the member
  (`Member 'shot@...'`). Eight more Ref2VA templates declare the ceiling
  (#479/#501).
- A workflow with no `vram_estimate` gets a `vram_projection_inherited`
  warning, not a refusal, when the catalog template that loads the same
  pipeline would be over its ceiling (#502).
- The VRAM ceiling is checked against the serving device's own cost entries
  (or its measured capacity), not every CUDA card in the catalog. A 64 GB
  Mac is no longer refused against the RTX 3090 entry (#470).
- `asset:`, `prompt:` and `output:` references written directly into a step
  are resolved at validate and at submission. A stored workflow with a
  literal reference to a missing file now fails validation (#494).
- `validate_workflow` refuses a `for_each` item's reference that resolves to
  null, which the run already refused (#478). It also refuses a frame-size
  mismatch between the `asset:`/`output:`/path inputs of `concat_videos` and
  `dissolve_videos` (#504).
- A `transcribe_audio` step with `timestamps` must set `content_type:
  application/json` (#498).
- A `weight_name` (a LoRA's, an IP adapter's) must be a relative file inside
  the model repo. An absolute path, a backslash, a drive or a `.`/`..`
  segment is refused at validate. Media path and glob refusals no longer
  list the server's directories.
- `previous_result:<step>.<key>` for a key that no result carries is an
  error. It used to run zero iterations and save nothing.
- `templates/minimax/video-with-audio-768p` runs its turbo LoRA at its
  trained strength. It had been running at 16x, so its output changes (#468).
- On macOS, parallel checkpoint loading is off by default, and so is SDNQ
  quantized matmul on MPS. Group-offload CUDA streams are dropped, with a
  warning, when the onload device isn't CUDA or XPU. An explicit
  `HF_ENABLE_PARALLEL_LOADING` still wins (#470).
- A `for_each` whose members produce the same shot name now warns
  `shot_name_collision` (#508). `assess_output` has a new `shot_dead_air`
  finding for silent gaps inside a shot, and `concat_videos` marks its joins
  as hard cuts, so `seam_frame_jump` no longer fires on a deliberate cut
  (#465/#466).

**New**

- Apple Silicon (MPS): the CUDA templates, LTX-2.5 and `minimax/music`
  included, run unchanged. Memory figures and the chip name are real, and
  torch's CPU-fallback warning is shown (#470).
- The `attribute_voices` task names which reference voice sings or speaks
  each line or window, by timbre. It uses demucs separation and ECAPA
  embeddings, and `demucs` is now a dependency. `list_tasks` marks the
  read-only probes with an `assessment` flag (#485/#494). The
  `minimax-music3` skill points to it for multi-singer songs (#495).
- `normalize_audio(limit=true)`: a true-peak look-ahead limiter that reaches
  `target_lufs` past a loud transient. It warns `limiter_heavy` past 6 dB of
  reduction and `target_lufs_capped` (`limited: true`) past 12 dB
  (#474/#496).
- `assemble-and-score` takes `target_lufs` (#467) and `limit` (#497, still
  open). The `series-episodes` skill says to match episodes downward to one
  series target. **Unfinished (#497):** with `limit: true` the mix holds
  -3 dBTP, but the AAC-muxed film can land up to about 1 dB above it.
- `transcribe_audio(timestamps="segment"|"word")` returns `{text, chunks}`.
  Unset, it still returns plain text (#483).
- `templates/ltx2/extend-clip` takes a `clip` to extend an existing clip.
  The opening isn't generated when one is given (#446).
- `save_workflow`, `delete_workflow`, `upload_asset`, `delete_asset` and
  `list_assets` take `workspace=` for one call. Their replies name the
  workspace they acted on (#463).
- A LoRA with `model_name: null` is switched off and warns `lora_disabled`.
  It used to validate clean and then fail after the model load (#469).
- `get_guide(section=...)` reaches a `###` subsection and returns its
  `parent_section` (#503).
- `upload_asset`'s refusals give a working `curl` to `POST /api/uploads`
  and state the 200 MB limit. The guide explains that `for_each`
  item-level `from_previous_result` must name the member, as
  `slice@<entry>` (#481/#482).
- A running job's `manifest.json` is rewritten after each step, so finished
  `for_each` members show up before the run ends (#480).
- `SECURITY.md`: report vulnerabilities privately, not as issues.

**Fixes**

- `templates/ltx2/extend-clip` frees the opening's pipeline before the
  extension loads, so it no longer holds two LTX-2.5 stacks on the GPU
  (#523).
- `concat_videos`/`dissolve_videos` load a `{"location": ...}` entry in
  `videos` (#510).
- A modular step whose outputs include latents no longer crashes at save
  (#507). `pair_audio` unwraps a pipeline's batch of one video.
- `transcribe_audio` works on clips over 30 s, and a non-Whisper model on a
  long clip still returns text.
- `segment` works again: the SAM2 `KeyError` is fixed, and the
  GroundingDINO query is written the way it scores. Dead default input URLs
  in several templates are replaced (#470).
- `upload_asset(file_path=...)` accepts the writable shared asset library
  (#448).
- `host_memory_job_peak_rss_mb` is a running max (#457).
- `dw.serve` exits on SIGTERM with an MCP client still connected (#477).
- The `shot_dead_air` finding names the room-tone remedy. The
  `normalize_audio` warning no longer says that gaining down always succeeds
  (#491/#492). The docs describe `dead_air_floor_dbfs` correctly (#519).
- The `minimax-h3` skill carries the 24-shot field report's prompting rules
  (#484).

### 0.4.0

The auto-generated notes for this range are a single merge line, since the
work landed on `develop` without PRs. Paste this section into the GitHub
release body once the tag has published (`gh release edit v0.4.0
--notes-file ...`).

**Breaking and behaviour changes**

- `download_output` over a `dw.serve --mcp` endpoint refuses a call with no
  `destination`. It used to write into the server's own directory (#353).
- Untrusted workflows are refused in more cases (#409-#413):
  - a `*_type` that doesn't resolve to a class, or that isn't a kind a
    workflow constructs: a diffusers or transformers model, pipeline,
    scheduler, tokenizer or processor, a quantization config, an auto
    factory, a diffusers reference/condition type or an attention processor.
    A plain `torch` class such as `torch.nn.Linear` is now refused;
  - `constant:` walks through `_` names or out of the allowed packages;
  - URLs with backslashes;
  - `text/html` and `text/xml` result types;
  - media hosts that aren't globally routable, including 100.64/10 (CGNAT,
    and so Tailscale);
  - more than 5 redirects;
  - images over 50M pixels.

  Listings and export zips drop symlinks that escape their root.
  `--trust-workflows` lifts all of these.
- `run_workflow` validates the caller's `arguments` when it queues the job
  (#414/#415). `validate_workflow(arguments={})` checks a run with no values
  supplied, not just the document (#364).
- A fractional value for an int variable is refused (#338), and so is a
  still image passed as a video argument (#347).
- `templates/minimax/music` normalizes to -3 dBFS instead of -1, so its output
  is quieter (#362).
- Every response carries `X-Content-Type-Options: nosniff` and
  `X-Frame-Options: DENY`. Active document types under `/outputs` and
  `/inputs` are served with `Content-Security-Policy: sandbox`.
- A validate-time probe reads only a literal media path that the run itself
  would be allowed to read.
- A dict or list passed to a string-typed variable is refused (#433).
  `templates/ltx2/keyframes` takes `first_image`/`last_image` as plain
  strings, not `{"location": ...}` (#431/#433).
- `loop_frames` returns float32 frames in [0, 1] instead of uint8, the shape
  `LTX2ReferenceCondition` needs; a keyframe condition still wants
  `frames_as_array`. `ltx2/reference-sheet`'s default asset is now
  `asset:reference_sheet.jpg` (#444).
- `validate_workflow` refuses a `components` name the pipeline doesn't
  register; `duration_head` is gone from the in-context LTX-2 templates
  (#442).
- A `{"media_type": "image"}` reference on a video argument loads as a
  one-frame still (#443).
- `pair_audio fit: "video"` always fits, and warns on any nonzero gap
  (#428/#429). `concat_videos` and `dissolve_videos` pad a short joined
  track to the frame grid, warning (`joined_audio_padded_to_frames`) only
  when the pad is a frame or more; a residual the AAC mux trims off is
  logged, or warned as `joined_audio_short_after_mux` from a frame up.
  `media.shots` is measured against the file as written (#426/#435/#454).
  Neither warns about resampling inputs that agree to a pinned
  `sample_rate` (#453).
- New warnings: `match_levels_near_silent` (#434), and `shot_span_overrun`
  from the probes plus a validate-time check (#425).
- Error text changed: `delete_workspace` (#437/#438), the sub-workflow path
  refusal names the places it looked (#422), and `/outputs/asset:...` misses
  name the asset without server paths.

**New**

- The `assess_output` tool and `GET /api/gallery/{name}/assess`, plus the
  probe tasks `analyze_shots`, `analyze_seams` and `analyze_sync_drift`
  (#387/#388).
- A joined video records its shot boundaries (`media.shots`).
  `get_output_frames(seams=true)` uses them, so it no longer needs
  `boundaries` (#385).
- Run versions (`v<N>`):
  - `list_gallery` returns `run_id`/`version` and filters by `folder` and
    `version`;
  - `output:<wf>/v<N>/<file>` references;
  - `wait_for_job` returns `run_version`;
  - export zips download as `<wf>-vN-<job>.zip`.
- `list_gallery(media=true)` adds durations, and `output:` names work in
  gallery reads (#356).
- `DW_PUBLIC_URL` adds absolute URLs to gallery and export responses.
  `export_job` also returns `auth_required` and `open_url` (#353).
- A `grade` task for images and video: exposure, contrast, saturation and
  temperature/tint (#349).
- The `templates/minimax/shots-batch` H3 template (#352).
- Every generative template takes a `seed` argument (#351).
- `normalize_audio(target_lufs)`, and `integrated_lufs` plus true peak in
  media metadata (#361).
- `gain_audio` with no region gains the whole track (#395).
- `world_fade_out_ms` on `assemble-and-score` (#339).
- Download progress shows in `phase_detail` (#343). `phase_stall` events
  now read as informational (#357).
- `workflow`, `inline_workflow` and `prompt` also accept a JSON string. A
  mistyped workflow name gets suggestions from the catalog (#397).
- Host caches are released when each job ends (#368), and the skills point
  at `clear_memory`.
- `get_job_events(kinds=...)` and `?kinds=` on the event-log route; a kind
  matches an event's `event` or its `kind`, so `["phase_stall"]` selects
  one warning type (#436).
- `get_memory` reports the step cache's `entries` and `retained_bytes`
  (#418).
- `get_output_image` and `/outputs` resolve `asset:` references (#445), and
  `get_output_frames(seams=true)` works on linked assets (#430).
- Compact `assess_output` lists each finding once (#427). Shots are named by
  their source when joined inputs already carry shots (#432).
- A task-only workflow's run history counts, so its estimate can quote
  `basis: observed` (#439). The Music 3 hint no longer shows on video
  (#441).

**Fixes**

- The step cache's retained-byte count no longer only grows (#418).
- `templates/ltx2/keyframes` (#431), `restore-decompression` (#442) and
  `reference-sheet` (#444) run with their own defaults again.
- Joined audio and shot maps stay on the frame grid through repeated joins
  (#423, #426, #428, #435).

Releases are cut by pushing a `v<semver>` tag. CI does the rest.

Before merging `develop` into `master`, run `scripts/preflight.sh` and get it
passing. It covers more than CI: ruff over the whole repo rather than
`dw dw_mcp tests`, the real-model integration tests (`pytest -m
integration`), and the UI's Playwright e2e tests, none of which CI runs.

```bash
scripts/release.sh 0.38.0
scripts/release.sh 0.38.0-alpha.1 "UI front end"   # optional tag message
scripts/release.sh 0.38.0 --next 0.39.0-alpha.1     # and reopen develop
```

The script bumps `pyproject.toml` (the single source of the version —
`dw.__version__` reads it at runtime) and sets the same version in
`plugins/dw/.claude-plugin/plugin.json`, so an installed plugin names the
engine it was written against; it commits just those two files, pushes
master, tags the bump commit `v0.38.0`, and pushes the tag. It refuses
a malformed version, a branch other than master, an existing tag, or a
dirty index (unstaged changes elsewhere are fine — the release commit
is path-limited to those two files).

Before it bumps anything it runs the integration tests, the gate CI's
accelerator-less runners cannot, and refuses to release when they fail or
when the machine has no CUDA or MPS device - there they would skip and
pass having run nothing. Cut a release from the Mac or lem, with the venv
active. The tests run against the working tree, so unstaged changes are
part of what they check.

`--next <version>` finishes the release on the other branch: it merges
`master` back into `develop` (a fast-forward when nothing landed there
since the release PR), sets `<version>` in the same two files, commits
`chore: open <version> on develop` and pushes `develop`, all in a temporary
worktree, so it works while `develop` is checked out elsewhere. Without it, do that by hand, or `develop` goes on reporting the
previous pre-release.

CI runs on every push to `develop` as well as `master` - the agent loop
pushes `develop` directly, with no PR - so a failure shows up against the
commit that caused it, not first on the release PR.

By hand, the equivalent is:

```bash
# 1. Bump the version in pyproject.toml:
#    version = "0.38.0"
# 2. Set the same version in plugins/dw/.claude-plugin/plugin.json
git commit -m "release 0.38.0" -- pyproject.toml plugins/dw/.claude-plugin/plugin.json

# 3. Tag the bump commit and push
git tag -a v0.38.0 -m "release 0.38.0"
git push origin master v0.38.0
```

The tag must point at a commit whose pyproject already declares the
same version — the release job checks and refuses a mismatch.

The tag triggers the full CI chain: backend tests, UI lint/type-check/
unit tests, then the wheel build (SPA compiled into the package via
`scripts/build_dist.sh`). Only if all of that passes does the `release`
job run — it verifies the tag matches the pyproject version, then
creates a GitHub release named after the tag with auto-generated notes
and the wheel + sdist attached.

Note on pre-release numbering: Python packaging normalizes semver-style
pre-releases, so a `0.38.0-alpha.1` version builds a wheel named
`0.38.0a1`. The tag, pyproject, and release stay in the semver form;
only the wheel filename and pip metadata show the normalized one.

A pre-release tag like `v0.38.0-rc1` is marked as a pre-release on
GitHub. Tags that aren't `v` + semver (or that don't match the declared
versions) fail the release job before anything is published.

After the GitHub release, the `pypi` job publishes the same artifacts to
PyPI via [trusted publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC — no token stored anywhere). One-time setup on pypi.org under
*Publishing*: add a trusted publisher for project `diffusers-workflow`
with owner `dkackman`, repository `diffusers-workflow`, workflow
`ci.yml`, environment `pypi` (use "add a pending publisher" before the
first release, since the project won't exist yet). Pre-release versions
are hidden from plain `pip install`; they need `pip install --pre`.

Note: released `diffusers` from PyPI may lag the newest model pipelines
this project targets — a PyPI install can need
`pip install git+https://github.com/huggingface/diffusers` on top.

To rebuild artifacts without releasing, run the CI workflow manually
(`workflow_dispatch`) — the wheel job uploads `dist/*` as a workflow
artifact.
