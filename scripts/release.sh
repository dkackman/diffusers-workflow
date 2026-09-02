#!/usr/bin/env bash
# Cut a release: bump pyproject.toml, commit, push, tag the bump commit,
# push the tag. CI does the rest (see docs/RELEASING.md).
#
#   scripts/release.sh 0.38.0
#   scripts/release.sh 0.38.0-alpha.1 "UI front end"
#
# The optional message annotates the tag (and shows up in git show/describe);
# without one the tag says "release <version>".
set -euo pipefail

cd "$(dirname "$0")/.."

version="${1:-}"
version="${version#v}" # a leading v is forgiven - the tag adds it back
message="${2:-release $version}"

if [ -z "$version" ]; then
    echo "usage: $0 <semver> [tag message]    e.g. $0 0.38.0 or $0 0.38.0-alpha.1 \"UI front end\"" >&2
    exit 1
fi

# The same shape the CI release job enforces
if ! echo "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$'; then
    echo "error: '$version' is not <major>.<minor>.<patch>[-prerelease]" >&2
    exit 1
fi

tag="v$version"

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "master" ]; then
    echo "error: releases are cut from master (currently on '$branch')" >&2
    exit 1
fi

if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    echo "error: tag $tag already exists" >&2
    exit 1
fi

# The release commit is path-limited to pyproject.toml, so a dirty working
# tree is fine - but staged changes would ride along with a plain commit,
# so refuse an index that isn't clean
if ! git diff --cached --quiet; then
    echo "error: staged changes in the index - commit or unstage them first" >&2
    exit 1
fi

current=$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml)
if [ "$current" != "$version" ]; then
    # python3 rather than sed -i, which spells in-place differently on macOS
    python3 - "$version" <<'EOF'
import re, sys

path = "pyproject.toml"
with open(path, encoding="utf-8") as f:
    text = f.read()
text, n = re.subn(
    r'^version = ".*"$',
    f'version = "{sys.argv[1]}"',
    text,
    count=1,
    flags=re.MULTILINE,
)
if n != 1:
    sys.exit("error: no version line found in pyproject.toml")
with open(path, "w", encoding="utf-8") as f:
    f.write(text)
EOF
    echo "pyproject.toml: $current -> $version"
fi

# The desktop shell pins the wheel to its own version, so the two must be
# released together - a shell at a version PyPI has never seen would try to
# install a wheel that does not exist. Cargo takes the semver form as-is;
# pip normalizes it (0.4.0-alpha.10 -> 0.4.0a10) and an exact pin still
# matches, so no second spelling of the version is needed.
desktop_manifest="desktop/Cargo.toml"
desktop_current=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$desktop_manifest" | head -1)
if [ "$desktop_current" != "$version" ]; then
    python3 - "$version" "$desktop_manifest" <<'EOF'
import re, sys

version, path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    text = f.read()
text, n = re.subn(
    r'^version = ".*"$', f'version = "{version}"', text, count=1, flags=re.MULTILINE
)
if n != 1:
    sys.exit(f"error: no version line found in {path}")
with open(path, "w", encoding="utf-8") as f:
    f.write(text)
EOF
    echo "$desktop_manifest: $desktop_current -> $version"
fi

# Refresh the lockfile so the bump does not leave CI with a dirty tree
if command -v cargo >/dev/null; then
    (cd desktop && cargo metadata --format-version 1 >/dev/null)
fi

release_paths="pyproject.toml $desktop_manifest"
if [ -f desktop/Cargo.lock ]; then
    release_paths="$release_paths desktop/Cargo.lock"
fi

# shellcheck disable=SC2086
if ! git diff --quiet -- $release_paths; then
    # shellcheck disable=SC2086
    git commit -m "release $version" -- $release_paths
else
    echo "pyproject.toml and $desktop_manifest already at $version - tagging HEAD"
fi

git push origin master
git tag -a "$tag" -m "$message"
git push origin "$tag"

echo
echo "$tag pushed - CI takes it from here:"
echo "  https://github.com/dkackman/diffusers-workflow/actions"
