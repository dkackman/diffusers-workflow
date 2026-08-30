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

if ! git diff --quiet -- pyproject.toml; then
    git commit -m "release $version" -- pyproject.toml
else
    echo "pyproject.toml already at $version and committed - tagging HEAD"
fi

git push origin master
git tag -a "$tag" -m "$message"
git push origin "$tag"

echo
echo "$tag pushed - CI takes it from here:"
echo "  https://github.com/dkackman/diffusers-workflow/actions"
