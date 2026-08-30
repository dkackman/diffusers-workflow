# Releasing

Releases are cut by pushing a `v<semver>` tag. CI does the rest.

```bash
# 1. Bump the version - pyproject.toml is the single source
#    (dw.__version__ reads it at runtime):
#    version = "0.38.0"
git commit -am "release 0.38.0"

# 2. Tag the bump commit and push
git tag -a v0.38.0 -m "release 0.38.0"
git push origin master v0.38.0
```

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

To rebuild artifacts without releasing, run the CI workflow manually
(`workflow_dispatch`) — the wheel job uploads `dist/*` as a workflow
artifact.
