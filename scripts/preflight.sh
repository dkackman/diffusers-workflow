#!/usr/bin/env bash
# Local pre-merge checks: ruff (format + fix), pytest, and the UI's own
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

run_step "ruff format" ruff format .
run_step "ruff check" ruff check . --fix
run_step "pytest" python -m pytest
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
