#!/usr/bin/env bash
# Build a distributable wheel: the SPA is built with npm and copied into the
# dw.server package (where default_ui_dir finds it in an install), the guides
# dw.server serves are copied into dw/docs/, the example catalog into
# dw/examples/, and the wheel is assembled around them.
set -euo pipefail
cd "$(dirname "$0")/.."

(cd ui && npm run build)

rm -rf dw/server/ui
cp -r ui/dist dw/server/ui

# The guides dw.server serves at /api/guides. A checkout reads the repo's
# docs/ directly (dw/server/guides.py prefers it), so this copy only matters
# to an install - but without it an installed server has no guides at all.
# Every doc goes in: read_guide only ever opens the names in
# dw.server.guides.GUIDES, so the extras are inert.
rm -rf dw/docs
mkdir -p dw/docs
cp docs/*.md dw/docs/

# The example catalog an installed server offers beside the workspace's own
# (dw/serve.py packaged_examples_dir), as the Docker image carries it in
# /opt/dw/examples. A checkout serves its own workflows/ regardless. Only
# tracked files go in: workflows/*/assets/ holds generated media (gitignored).
rm -rf dw/examples
mkdir -p dw/examples
git ls-files -z -- workflows prompts loras | while IFS= read -r -d '' f; do
    mkdir -p "dw/examples/$(dirname "$f")"
    cp "$f" "dw/examples/$f"
done

python -m pip show build >/dev/null 2>&1 || python -m pip install build
python -m build

echo ""
echo "Artifacts in dist/. dw/server/ui/, dw/docs/ and dw/examples/ are build"
echo "products (gitignored); remove or rebuild them - a checkout serves ui/dist,"
echo "the repo's own docs/ and its own workflows/ regardless."
