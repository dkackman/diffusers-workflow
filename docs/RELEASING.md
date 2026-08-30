# Releasing

Releases are cut by pushing a `v<semver>` tag. CI does the rest.

```bash
# 1. Bump the version in BOTH places (they must match the tag):
#    - pyproject.toml        -> version = "0.38.0"
#    - dw/__init__.py        -> __version__ = "0.38.0"
git commit -am "release 0.38.0"

# 2. Tag and push
git tag v0.38.0
git push origin master v0.38.0
```

The tag triggers the full CI chain: backend tests, UI lint/type-check/
unit tests, then the wheel build (SPA compiled into the package via
`scripts/build_dist.sh`). Only if all of that passes does the `release`
job run — it verifies the tag matches both declared versions, then
creates a GitHub release named after the tag with auto-generated notes
and the wheel + sdist attached.

A pre-release tag like `v0.38.0-rc1` is marked as a pre-release on
GitHub. Tags that aren't `v` + semver (or that don't match the declared
versions) fail the release job before anything is published.

To rebuild artifacts without releasing, run the CI workflow manually
(`workflow_dispatch`) — the wheel job uploads `dist/*` as a workflow
artifact.
