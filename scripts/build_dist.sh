#!/usr/bin/env bash
# Build a distributable wheel: the SPA is built with npm and copied into the
# dw.server package (where default_ui_dir finds it in an install), the guides
# dw_mcp serves are copied into dw/docs/, and the wheel is assembled around
# them.
set -euo pipefail
cd "$(dirname "$0")/.."

(cd ui && npm run build)

rm -rf dw/server/ui
cp -r ui/dist dw/server/ui

# The guides dw_mcp serves. A checkout reads the repo's docs/ directly, so this
# copy only matters to an install - but without it an installed dw-mcp has no
# guides at all
rm -rf dw/docs
mkdir -p dw/docs
python - <<'GUIDES'
import shutil
from pathlib import Path

from dw_mcp.guides import GUIDES

for file_name in sorted({f for f, _ in GUIDES.values()}):
    shutil.copy(Path("docs") / file_name, Path("dw/docs") / file_name)
    print(f"docs/{file_name} -> dw/docs/{file_name}")
GUIDES

python -m pip show build >/dev/null 2>&1 || python -m pip install build
python -m build

echo ""
echo "Artifacts in dist/. dw/server/ui/ and dw/docs/ are build products"
echo "(gitignored); remove or rebuild them - a checkout serves ui/dist and"
echo "the repo's own docs/ regardless."
