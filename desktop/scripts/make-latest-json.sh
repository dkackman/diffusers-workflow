#!/usr/bin/env bash
# Build the updater manifest Tauri's updater fetches.
#
#   scripts/make-latest-json.sh <version> <artifact-dir> <output>
#
# Tauri emits an update archive and a detached .sig per platform, but not
# the manifest that ties them together - that is on us. The endpoint in
# tauri.conf.json points at latest.json on the GitHub release, so this runs
# in the release job once all three platforms' artifacts are downloaded.
#
# A missing platform is omitted rather than fatal: a release that built on
# two of three runners should still update those two.
set -euo pipefail

VERSION="${1:?usage: make-latest-json.sh <version> <artifact-dir> <output>}"
ARTIFACTS="${2:?}"
OUTPUT="${3:?}"
REPO="${GITHUB_REPOSITORY:-dkackman/diffusers-workflow}"
BASE="https://github.com/$REPO/releases/download/v$VERSION"

python3 - "$VERSION" "$ARTIFACTS" "$OUTPUT" "$BASE" <<'PY'
import json, os, sys
from datetime import datetime, timezone

version, artifacts, output, base = sys.argv[1:5]

# The updater's platform keys, and the archive suffix Tauri produces for
# each. The .sig sits beside the archive with the same name.
TARGETS = {
    "darwin-aarch64": ".app.tar.gz",
    "windows-x86_64": ".msi.zip",
    "linux-x86_64": ".AppImage.tar.gz",
}

found = {}
for root, _, files in os.walk(artifacts):
    for name in files:
        for platform, suffix in TARGETS.items():
            if name.endswith(suffix):
                sig = os.path.join(root, name + ".sig")
                if not os.path.exists(sig):
                    print(f"warning: {name} has no .sig - skipping", file=sys.stderr)
                    continue
                with open(sig, encoding="utf-8") as handle:
                    found[platform] = {
                        "signature": handle.read().strip(),
                        "url": f"{base}/{name}",
                    }

if not found:
    sys.exit("error: no signed update archives found - was "
             "bundle.createUpdaterArtifacts enabled?")

manifest = {
    "version": version,
    "notes": f"See https://github.com/{os.environ.get('GITHUB_REPOSITORY', '')}"
             f"/releases/tag/v{version}",
    "pub_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "platforms": found,
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)

print(f"latest.json covers: {', '.join(sorted(found))}")
PY
