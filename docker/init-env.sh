#!/usr/bin/env bash
# Write docker/.env for docker/compose.yaml from this host's native layout:
# the same workspace, Hugging Face cache, compile caches and settings a
# native dw.serve uses, so the two can be switched between (docker/switch.sh).
# Each value can be overridden by exporting it first:
#
#   HF_HOME=/mnt/models bash docker/init-env.sh
#
# Refuses to overwrite an existing docker/.env unless --force.
set -euo pipefail
cd "$(dirname "$0")"

if [ -e .env ] && [ "${1:-}" != "--force" ]; then
  echo "docker/.env exists - edit it, or rerun with --force" >&2
  exit 1
fi

# The defaults dw resolves natively (dw/__init__.py, dw/disk_caches.py,
# dw/settings.py, dw/workspace.py), absolute. A symlinked directory stays as
# its link path: docker mounts the target there, so the path dw records is
# the one a native run records too.
cache="${XDG_CACHE_HOME:-$HOME/.cache}"
workspace="${DW_WORKSPACE:-$HOME/diffusers-workspace}"
hf_home="${HF_HOME:-$cache/huggingface}"
inductor="${TORCHINDUCTOR_CACHE_DIR:-$cache/dw/torchinductor}"
triton="${TRITON_CACHE_DIR:-${TRITON_HOME:-$HOME}/.triton/cache}"
settings="${DIFFUSERS_HELPER_ROOT:-$HOME/.diffusers_helper}"
env_dir="${DW_ENV_DIR:-$HOME/.local/share/dw-docker/env}"

# docker would create a missing source as root; create them as this user
mkdir -p "$workspace" "$hf_home" "$inductor" "$triton" "$settings" "$env_dir"

cat > .env <<EOF
# Written by docker/init-env.sh on $(date +%Y-%m-%d). Paths are mounted into
# the container at the same absolute path.
DW_UID=$(id -u)
DW_GID=$(id -g)
DW_WORKSPACE=$workspace
HF_HOME=$hf_home
TORCHINDUCTOR_CACHE_DIR=$inductor
TRITON_CACHE_DIR=$triton
DIFFUSERS_HELPER_ROOT=$settings
# The diffusers overlay venv - the container's own, not shared with native
DW_ENV_DIR=$env_dir

# 0.0.0.0 publishes the port to the LAN (then keep a token)
DW_BIND=127.0.0.1
DW_HOST_PORT=8765
# Unset: the container generates one and saves it in DIFFUSERS_HELPER_ROOT
#DW_API_TOKEN=
#HF_TOKEN=hf_...
# head (the default) | image | a commit hash
#DW_DIFFUSERS_REF=head
# dw.serve's arguments (default --mcp)
#DW_SERVE_ARGS=--mcp --devices cuda:0,cuda:1
# How long an MCP wait_for_job may block (the systemd unit's value)
DW_MCP_MAX_WAIT_SECONDS=1800
EOF
echo "wrote docker/.env:"
grep -v '^#' .env | grep -v '^$'
