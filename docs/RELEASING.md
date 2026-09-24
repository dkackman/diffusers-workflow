# Releasing

## Unreleased

This project has no standing release-notes file - GitHub auto-generates
notes from commits at tag time (see below). This section is a scratch pad
for items a branch's author wants the next release note to name; clear it
when a release ships.

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

Releases are cut by pushing a `v<semver>` tag. CI does the rest.

Before merging `develop` into `master`, run `scripts/preflight.sh` and get it
passing. It covers more than CI: ruff over the whole repo rather than
`dw dw_mcp tests`, and the UI's Playwright e2e tests, which CI doesn't run.

```bash
scripts/release.sh 0.38.0
scripts/release.sh 0.38.0-alpha.1 "UI front end"   # optional tag message
```

The script bumps `pyproject.toml` (the single source of the version —
`dw.__version__` reads it at runtime) and sets the same version in
`plugins/dw/.claude-plugin/plugin.json`, so an installed plugin names the
engine it was written against; it commits just those two files, pushes
master, tags the bump commit `v0.38.0`, and pushes the tag. It refuses
a malformed version, a branch other than master, an existing tag, or a
dirty index (unstaged changes elsewhere are fine — the release commit
is path-limited to those two files).

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
