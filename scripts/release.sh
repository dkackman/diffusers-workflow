#!/usr/bin/env bash
# Cut a release: bump pyproject.toml, commit, push, tag the bump commit,
# push the tag. CI does the rest (see docs/RELEASING.md).
#
#   scripts/release.sh 0.38.0
#   scripts/release.sh 0.38.0-alpha.1 "UI front end"
#   scripts/release.sh 0.38.0 --next 0.39.0-alpha.1
#
# The optional message annotates the tag (and shows up in git show/describe);
# without one the tag says "release <version>".
#
# --next <version> also does the step after a release: merge master back
# into develop and open <version> there, so the two branches don't diverge
# on the bump (the 0.4.0 release did this by hand, and beta.7 never did it
# at all - develop still said beta.6 afterwards). It leaves develop checked
# out.
set -euo pipefail

cd "$(dirname "$0")/.."

next=""
args=()
while [ $# -gt 0 ]; do
    case "$1" in
    --next)
        next="${2:-}"
        shift 2 || true
        ;;
    --next=*)
        next="${1#--next=}"
        shift
        ;;
    *)
        args+=("$1")
        shift
        ;;
    esac
done
version="${args[0]:-}"
version="${version#v}" # a leading v is forgiven - the tag adds it back
message="${args[1]:-release $version}"
next="${next#v}"

if [ -z "$version" ]; then
    echo "usage: $0 <semver> [tag message] [--next <semver>]    e.g. $0 0.38.0 --next 0.39.0-alpha.1" >&2
    exit 1
fi

# The same shape the CI release job enforces
is_semver() {
    echo "$1" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$'
}
for v in "$version" ${next:+"$next"}; do
    if ! is_semver "$v"; then
        echo "error: '$v' is not <major>.<minor>.<patch>[-prerelease]" >&2
        exit 1
    fi
done
if [ -n "$next" ] && [ "$next" = "$version" ]; then
    echo "error: --next must differ from the version being released" >&2
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

# set_version <version>: pyproject.toml and the plugin manifest to <version>
plugin_path="plugins/dw/.claude-plugin/plugin.json"
set_version() {
    local version="$1"
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

    # The plugin's version is the engine's (tests/test_plugin_skills.py holds
    # them equal), and it is the key the plugin cache is stored under. Written
    # unconditionally so a plugin.json that drifted while pyproject.toml already
    # sat at the target version is still brought into line
    plugin_current=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$plugin_path")
    if [ "$plugin_current" != "$version" ]; then
        python3 - "$version" "$plugin_path" <<'EOF'
import json, sys

path = sys.argv[2]
with open(path, encoding="utf-8") as f:
    plugin = json.load(f)
plugin["version"] = sys.argv[1]
with open(path, "w", encoding="utf-8") as f:
    json.dump(plugin, f, indent=4, ensure_ascii=False)
    f.write("\n")
EOF
        echo "$plugin_path: $plugin_current -> $version"
    fi
}

set_version "$version"

if ! git diff --quiet -- pyproject.toml "$plugin_path"; then
    git commit -m "release $version" -- pyproject.toml "$plugin_path"
else
    echo "pyproject.toml and $plugin_path already at $version and committed - tagging HEAD"
fi

git push origin master
git tag -a "$tag" -m "$message"
git push origin "$tag"

echo
echo "$tag pushed - CI takes it from here:"
echo "  https://github.com/dkackman/diffusers-workflow/actions"

if [ -n "$next" ]; then
    echo
    git fetch origin develop
    git checkout develop
    git merge --ff-only origin/develop
    # A fast-forward when nothing landed on develop since the release branch
    # was cut (the usual case under a freeze); a merge commit otherwise
    git merge --no-edit master
    set_version "$next"
    git commit -m "chore: open $next on develop" -- pyproject.toml "$plugin_path"
    git push origin develop
    echo
    echo "develop merged master and opened $next (now checked out)"
fi
