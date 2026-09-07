#!/usr/bin/env bash
# Build a distributable wheel: the SPA is built with npm and copied into the
# dw.server package (where default_ui_dir finds it in an install), the guides
# dw.server serves are copied into dw/docs/, and the wheel is assembled around
# them.
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

python -m pip show build >/dev/null 2>&1 || python -m pip install build
python -m build

echo ""
echo "Artifacts in dist/. dw/server/ui/ and dw/docs/ are build products"
echo "(gitignored); remove or rebuild them - a checkout serves ui/dist and"
echo "the repo's own docs/ regardless."
