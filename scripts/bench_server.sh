#!/usr/bin/env bash
# A second dw.serve on lem for the long-form bench. Runs ON lem:
#   ssh lem 'bash ~/dw-bench/scripts/bench_server.sh {start|stop|status|deploy <branch>}'
# Port 8766, card cuda:1, workspace ~/bench-workspace. The main server (8765) is untouched.
# Token: $BENCH_TOKEN, else the file ~/.dw-bench-token. Cards: $BENCH_DEVICES (default cuda:1).
set -euo pipefail

MAIN="$HOME/diffusers-workflow"
TREE="$HOME/dw-bench"
PORT=8766
LOG="$HOME/dw-bench.log"

token() {
    if [ -n "${BENCH_TOKEN:-}" ]; then printf '%s' "$BENCH_TOKEN"
    elif [ -r "$HOME/.dw-bench-token" ]; then tr -d '\n' < "$HOME/.dw-bench-token"
    else echo "no BENCH_TOKEN and no ~/.dw-bench-token" >&2; exit 1; fi
}

pid() { pgrep -f -- "dw.serve .*--port $PORT" | grep -v "^$$\$" | head -1 || true; }

health() { curl -sf "http://127.0.0.1:$PORT/api/health" -H "Authorization: Bearer $(token)"; }

cmd_deploy() {
    local branch="${1:?usage: deploy <branch>}"
    git -C "$MAIN" fetch origin "$branch"
    if [ ! -d "$TREE" ]; then
        git -C "$MAIN" worktree add --detach "$TREE" "origin/$branch"
    else
        git -C "$TREE" checkout --detach "origin/$branch"
    fi
    git -C "$TREE" log -1 --format='deployed %h %s'
}

cmd_start() {
    if [ -n "$(pid)" ]; then echo "already running (pid $(pid))"; return 0; fi
    local tok; tok="$(token)"
    mkdir -p "$HOME/bench-workspace"
    cd "$TREE"
    nohup "$MAIN/venv/bin/python" -m dw.serve --host 0.0.0.0 --port "$PORT" \
        --devices "${BENCH_DEVICES:-cuda:1}" --token "$tok" \
        --workspace "$HOME/bench-workspace" --examples-dir "$TREE/workflows" \
        > "$LOG" 2>&1 &
    disown
    for _ in $(seq 120); do
        if health; then echo; return 0; fi
        sleep 1
    done
    echo "no health after 120 s; see $LOG" >&2
    return 1
}

cmd_stop() {
    local p; p="$(pid)"
    [ -z "$p" ] && { echo "not running"; return 0; }
    kill -TERM "$p"
    for _ in $(seq 30); do
        kill -0 "$p" 2>/dev/null || { echo "stopped"; return 0; }
        sleep 1
    done
    kill -KILL "$p" 2>/dev/null || true
    # the persistent workers are children; drop any left holding the bench card
    echo "killed (SIGKILL after 30 s)"
}

cmd_status() {
    git -C "$TREE" log -1 --format='worktree %h %s' 2>/dev/null || echo "no worktree at $TREE"
    health || echo "bench server not answering on :$PORT"
}

case "${1:-}" in
    deploy) shift; cmd_deploy "$@" ;;
    start) cmd_start ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    *) echo "usage: $0 {start|stop|status|deploy <branch>}" >&2; exit 2 ;;
esac
