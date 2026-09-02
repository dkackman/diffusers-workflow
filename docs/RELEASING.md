# Releasing

Releases are cut by pushing a `v<semver>` tag. CI does the rest.

```bash
scripts/release.sh 0.38.0
scripts/release.sh 0.38.0-alpha.1 "UI front end"   # optional tag message
```

The script bumps `pyproject.toml` (the single source of the version —
`dw.__version__` reads it at runtime), commits just that file, pushes
master, tags the bump commit `v0.38.0`, and pushes the tag. It refuses
a malformed version, a branch other than master, an existing tag, or a
dirty index (unstaged changes elsewhere are fine — the release commit
is path-limited to pyproject.toml).

By hand, the equivalent is:

```bash
# 1. Bump the version in pyproject.toml:
#    version = "0.38.0"
git commit -m "release 0.38.0" -- pyproject.toml

# 2. Tag the bump commit and push
git tag -a v0.38.0 -m "release 0.38.0"
git push origin master v0.38.0
```

The tag must point at a commit whose pyproject already declares the
same version — the release job checks and refuses a mismatch.

The tag triggers the full CI chain: backend tests, UI lint/type-check/
unit tests, the desktop shell's Rust tests, then the wheel build (SPA compiled into the package via
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

Alongside the wheel, the `desktop` job builds the native installers -
a signed and notarized `.dmg` on `macos-14`, an `.msi` on
`windows-latest`, and an `.AppImage` on `ubuntu-22.04` - and the release
job attaches all three. Each runner first fetches the bundled Python
runtime and `uv` with `desktop/scripts/fetch-runtime.sh`, which verifies
both against pinned SHA256 hashes.

The shell pins the wheel to its own version, so the installers and the
PyPI release always come from one tag. Two sets of secrets are involved:
`TAURI_SIGNING_PRIVATE_KEY` (Tauri's own update signing, unrelated to OS
code signing, so the unsigned Windows build still auto-updates) and the
`APPLE_*` pair used to sign and notarize the macOS build. Windows code
signing is not yet configured; adding it is a matter of adding secrets
and a step, not restructuring the job.

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
