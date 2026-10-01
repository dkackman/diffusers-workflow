#!/usr/bin/env bash
# Local pre-merge checks: ruff (format + fix, scoped to the Python packages so
# docs/*.md code blocks are left alone), pytest (unit, then the
# real-model integration tests, which skip without an accelerator), the
# architecture ratchet (scripts/arch_metrics.py --check against
# docs/stabilization/baseline.json), and the UI's own
# preflight (check, lint, format, build, unit and e2e tests). Runs every step
# and lists the ones that failed. Run it from anywhere:
#
#   scripts/preflight.sh
#
# ruff format and ruff check --fix rewrite files in place; review the diff.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

failures=()

run_step() {
    local name="$1"
    shift
    echo "==> $name"
    if ! "$@"; then
        failures+=("$name")
    fi
}

run_step "ruff format" ruff format dw dw_mcp tests scripts
run_step "ruff check" ruff check dw dw_mcp tests scripts --fix
run_step "pytest" python -m pytest
run_step "pytest integration" python -m pytest -m integration -n0
run_step "architecture ratchet" python scripts/arch_metrics.py --check docs/stabilization/baseline.json
# e2e starts the fixture server on the same interpreter pytest just used,
# wherever its venv lives (a worktree, .venv) - see ui/playwright.config.ts
export DW_E2E_PYTHON="${DW_E2E_PYTHON:-$(command -v python)}"
run_step "ui preflight" bash -c 'cd ui && npm run preflight'

echo
if [ ${#failures[@]} -eq 0 ]; then
    echo "All preflight checks passed."
    exit 0
else
    echo "Preflight failures:"
    for f in "${failures[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
