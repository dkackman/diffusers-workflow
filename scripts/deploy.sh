#!/usr/bin/env bash
# Deploy a branch of diffusers-workflow to this box and restart dw.serve.
#
#   scripts/deploy.sh [branch] [--force]
#
# Run ON the server box (lem), from anywhere:
#   ssh lem ~/diffusers-workflow/scripts/deploy.sh develop
#
# What it does, in order, stopping at the first failure:
#   1. fetch; check out <branch> (default: the current branch); fast-forward
#      pull. A dirty checkout or a non-fast-forward is an error, never a
#      reset - a deploy must not lose anything.
#   2. `pip install -e .` only when pyproject.toml changed in the pull;
#      `npm run build` in ui/ (after `npm ci` if package-lock.json changed)
#      only when the pull touched ui/. A checkout serves the SPA from
#      ui/dist, which is gitignored, so without this a UI change is pulled
#      but never shown.
#   3. refuse to restart while a job is running (health says current_job),
#      unless --force. Agent cycles are serialized, so in practice the only
#      running job is the caller's own.
#   4. restart. With the systemd user unit installed (scripts/dw-serve.service)
#      that is `systemctl --user restart dw-serve` and the log is the journal.
#      Without it: SIGTERM the running dw.serve (never SIGKILL; a stuck one is
#      a bug to report, so after 30 s this fails loudly instead of
#      escalating), then start the new server as its own window of the
#      `dw-serve` screen session (created if absent), stdout+stderr appended
#      to ~/dw-serve.log.
#   5. poll /api/health until it answers ok (up to 120 s), then print the
#      deployed commit and the server's reported version.
#
# Environment overrides, all optional:
#   DW_DIR (checkout, default ~/diffusers-workflow), DW_TOKEN (default xyz),
#   DW_PORT (8765), DW_WORKSPACE (~/diffusers-workspace), DW_HOST (0.0.0.0),
#   DW_NODE_BIN (the directory holding npm, when it is not on a
#   non-interactive PATH and not in one of the places find_npm looks).
set -euo pipefail

DW_DIR="${DW_DIR:-$HOME/diffusers-workflow}"
DW_TOKEN="${DW_TOKEN:-xyz}"
DW_PORT="${DW_PORT:-8765}"
DW_HOST="${DW_HOST:-0.0.0.0}"
DW_WORKSPACE="${DW_WORKSPACE:-$HOME/diffusers-workspace}"
LOG="$HOME/dw-serve.log"
SCREEN_SESSION="dw-serve"
HEALTH="http://localhost:$DW_PORT/api/health"

branch=""
force=0
for a in "$@"; do
  case "$a" in
    --force) force=1 ;;
    -*) echo "deploy: unknown flag $a" >&2; exit 2 ;;
    *) branch="$a" ;;
  esac
done

ts() { date '+%H:%M:%S'; }
say() { echo "[deploy $(ts)] $*"; }
health() { curl -s -m 5 -H "Authorization: Bearer $DW_TOKEN" "$HEALTH" 2>/dev/null; }
server_pids() { pgrep -f 'python -m dw\.serve' || true; }

# `ssh lem deploy.sh` runs a non-interactive shell, and a per-user node
# install is usually put on PATH by ~/.bashrc *after* its "not interactive,
# stop here" guard - so npm that works at a prompt is missing here. Look in
# the usual per-user places rather than depend on the caller's shell
find_npm() {
  command -v npm >/dev/null 2>&1 && return 0
  local dir
  for dir in "${DW_NODE_BIN:-}" "$HOME/.local/node/bin" "$HOME/.volta/bin" \
             "$HOME/.local/share/fnm/aliases/default/bin" "$HOME/.local/bin" /usr/local/bin; do
    if [ -n "$dir" ] && [ -x "$dir/npm" ]; then
      PATH="$dir:$PATH"; export PATH
      say "npm not on PATH; using $dir"
      return 0
    fi
  done
  # nvm is a shell function, not a directory on PATH
  if [ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]; then
    # shellcheck disable=SC1091
    . "${NVM_DIR:-$HOME/.nvm}/nvm.sh" >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 \
      && { say "npm from nvm: $(command -v npm)"; return 0; }
  fi
  say "npm not found (PATH=$PATH); set DW_NODE_BIN to the directory holding npm"
  exit 1
}

cd "$DW_DIR"
[ -z "$branch" ] && branch="$(git branch --show-current)"

# 1. checkout + fast-forward
if [ -n "$(git status --porcelain)" ]; then
  say "checkout is dirty; refusing to deploy over uncommitted changes:"; git status --short; exit 1
fi
say "fetching origin"
git fetch -q origin
before="$(git rev-parse HEAD)"
git checkout -q "$branch"
git pull -q --ff-only origin "$branch"
after="$(git rev-parse HEAD)"
say "on $branch at $(git rev-parse --short "$after") (was $(git rev-parse --short "$before"))"

# 2. deps and the UI bundle, only when they changed
changed=""
[ "$before" = "$after" ] || changed="$(git diff --name-only "$before" "$after")"
if echo "$changed" | grep -qx 'pyproject.toml'; then
  say "pyproject.toml changed; reinstalling"
  venv/bin/pip install -q -e .
fi
if echo "$changed" | grep -q '^ui/' || [ ! -d ui/dist ]; then
  find_npm
  if echo "$changed" | grep -qx 'ui/package-lock.json' || [ ! -d ui/node_modules ]; then
    say "ui/package-lock.json changed; npm ci"
    (cd ui && npm ci --silent --no-audit --no-fund)
  fi
  say "ui/ changed; building the SPA"
  (cd ui && npm run build --silent)
fi

# 3. don't yank a running job
h="$(health || true)"
if [ -n "$h" ] && echo "$h" | grep -q '"current_job":"' && [ "$force" -ne 1 ]; then
  say "a job is running ($(echo "$h" | grep -o '"current_job":"[^"]*"')); waiting up to 5 min for it, or pass --force"
  for _ in $(seq 1 60); do
    sleep 5
    h="$(health || true)"
    echo "$h" | grep -q '"current_job":"' || break
  done
  echo "$h" | grep -q '"current_job":"' && { say "still running; not restarting"; exit 1; }
fi

# 4. restart - systemd unit when installed, screen otherwise
if systemctl --user cat dw-serve >/dev/null 2>&1; then
  say "restarting via systemd user unit dw-serve"
  systemctl --user restart dw-serve
  LOG_HINT="journalctl --user -u dw-serve -n 200"
  tail_log() { journalctl --user -u dw-serve -n 20 --no-pager; }
else
LOG_HINT="$LOG"
tail_log() { tail -n 20 "$LOG"; }
pids="$(server_pids)"
if [ -n "$pids" ]; then
  say "stopping dw.serve (pid $pids)"
  kill -TERM $pids
  for _ in $(seq 1 30); do
    [ -z "$(server_pids)" ] && break
    sleep 1
  done
  if [ -n "$(server_pids)" ]; then
    say "dw.serve did not exit within 30 s of SIGTERM (pid $(server_pids)); not escalating - report this"; exit 1
  fi
else
  say "no dw.serve running"
fi

# start the new one in its own screen window
cmd="cd $DW_DIR && source venv/bin/activate && exec python -m dw.serve --host $DW_HOST --port $DW_PORT --mcp --token $DW_TOKEN --workspace $DW_WORKSPACE --examples-dir $DW_DIR/workflows >> $LOG 2>&1"
if screen -ls | grep -q "\.${SCREEN_SESSION}[[:space:]]"; then
  screen -S "$SCREEN_SESSION" -X screen -t serve bash -c "$cmd"
else
  screen -dmS "$SCREEN_SESSION" -t serve bash -c "$cmd"
fi
say "started; log: $LOG"
fi

# 5. wait for health
for i in $(seq 1 120); do
  h="$(health || true)"
  if echo "$h" | grep -q '"status":"ok"'; then
    say "healthy after ${i}s: $(echo "$h" | grep -o '"version":"[^"]*"') mcp=$(echo "$h" | grep -o '"mcp":[a-z]*' | cut -d: -f2)"
    say "deployed $branch @ $(git rev-parse --short HEAD)"
    exit 0
  fi
  [ -z "$(server_pids)" ] && { say "dw.serve exited during startup; last log lines ($LOG_HINT):"; tail_log; exit 1; }
  sleep 1
done
say "no healthy answer within 120 s; last log lines ($LOG_HINT):"; tail_log; exit 1
