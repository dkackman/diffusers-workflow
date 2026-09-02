#!/usr/bin/env bash
# Re-sign the bundled Python runtime and uv with the Developer ID identity.
#
#   scripts/sign-macos-runtime.sh
#
# python-build-standalone and uv ship ad-hoc signed on arm64 - enough for
# macOS to execute them, but notarization rejects an ad-hoc signature. Tauri
# signs the app binary and seals Contents/Resources by hash; it does not
# re-sign the Mach-O files living in there, so without this step the whole
# submission is rejected.
#
# Runs before `tauri build`, so the bundler copies files that are already
# signed and its outer signature seals them as they are.
#
# The interpreter gets entitlements the app binary does not need: a
# provisioned venv loads pip-installed libraries signed by nobody, which
# library validation would otherwise refuse. See python.entitlements.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$(uname)" != "Darwin" ]; then
    echo "not macOS - nothing to sign"
    exit 0
fi

IDENTITY="${APPLE_SIGNING_IDENTITY:-}"
if [ -z "$IDENTITY" ]; then
    echo "APPLE_SIGNING_IDENTITY is not set - skipping (an unsigned build" >&2
    echo "still runs locally; it will not notarize)" >&2
    exit 0
fi

RESOURCES="src-tauri/resources"
ENTITLEMENTS="$PWD/src-tauri/python.entitlements"
test -f "$ENTITLEMENTS" || { echo "error: $ENTITLEMENTS not found" >&2; exit 1; }

# Every Mach-O file under resources/, deepest first: codesign requires the
# nested code to be signed before whatever contains it.
mach_o_files() {
    # `file --mime-type` pads its output, so ask per file with -b and get
    # back the bare type. Deepest paths first, by counting separators.
    find "$RESOURCES" -type f -print0 |
        while IFS= read -r -d '' f; do
            if [ "$(file --mime-type -b "$f")" = "application/x-mach-binary" ]; then
                printf '%s %s\n' "$(printf '%s' "$f" | tr -cd '/' | wc -c | tr -d ' ')" "$f"
            fi
        done |
        sort -rn |
        cut -d' ' -f2-
}

count=0
while IFS= read -r target; do
    [ -n "$target" ] || continue
    codesign --force --timestamp --options runtime \
        --entitlements "$ENTITLEMENTS" \
        --sign "$IDENTITY" \
        "$target"
    count=$((count + 1))
done < <(mach_o_files)

echo "signed $count bundled binaries with $IDENTITY"

# Fail loudly here rather than at notarization, which reports it far less
# clearly and only after an upload
while IFS= read -r target; do
    [ -n "$target" ] || continue
    if codesign -dv "$target" 2>&1 | grep -q "Signature=adhoc"; then
        echo "error: $target is still ad-hoc signed" >&2
        exit 1
    fi
done < <(mach_o_files)

echo "verified: no ad-hoc signatures remain"
