#!/usr/bin/env bash
# Build a distributable wheel: the SPA is built with npm, copied into the
# dw.server package (where default_ui_dir finds it in an install), and the
# wheel is assembled around it.
set -euo pipefail
cd "$(dirname "$0")/.."

(cd ui && npm run build)

rm -rf dw/server/ui
cp -r ui/dist dw/server/ui

python -m pip show build >/dev/null 2>&1 || python -m pip install build
python -m build

echo ""
echo "Artifacts in dist/. dw/server/ui/ is a build product (gitignored);"
echo "remove it or rebuild it - a checkout serves ui/dist first regardless."
