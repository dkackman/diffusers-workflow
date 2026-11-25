#!/usr/bin/env bash
# Refresh Claude Code's copy of the dw plugin from this checkout.
#
# Claude loads an installed plugin from a cached copy
# (~/.claude/plugins/cache/diffusers-workflow/dw/<version>), not from
# plugins/dw/. `claude plugin update` only copies when the version in
# plugins/dw/.claude-plugin/plugin.json changes, so this script reinstalls
# instead: that copies the current files even when the version hasn't changed.
# Restart open sessions (or run /reload-plugins) afterwards.
set -euo pipefail

plugin="dw@diffusers-workflow"
scope="${1:-user}"

# Re-check marketplace.json and pick up plugins added to it.
claude plugin marketplace update diffusers-workflow

if claude plugin uninstall --keep-data -s "$scope" "$plugin"; then
    :
else
    echo "($plugin was not installed at $scope scope; installing fresh)"
fi
claude plugin install -s "$scope" "$plugin"

echo "Done. Restart open Claude sessions, or run /reload-plugins, to load it."
