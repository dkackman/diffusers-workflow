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
    else echo "no bench token: set BENCH_TOKEN or create ~/.dw-bench-token" >&2; exit 1; fi
}

# every descendant pid of $1, deepest first
descendants() {
    local c
    for c in $(pgrep -P "$1" || true); do descendants "$c"; echo "$c"; done
}

pid() { pgrep -f -- "dw.serve .*--port $PORT" | grep -v "^$$\$" | head -1 || true; }

health() {
    local tok; tok="$(token)" || return 1
    curl -sf "http://127.0.0.1:$PORT/api/health" -H "Authorization: Bearer $tok"
}

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
    local spid=$!
    disown
    for _ in $(seq 120); do
        if health; then echo; return 0; fi
        kill -0 "$spid" 2>/dev/null || { echo "bench server exited; see $LOG" >&2; return 1; }
        sleep 1
    done
    echo "no health after 120 s; see $LOG" >&2
    return 1
}

cmd_stop() {
    local p; p="$(pid)"
    [ -z "$p" ] && { echo "not running"; return 0; }
    # collect the server's workers first: a killed server orphans them
    local all; all="$(descendants "$p") $p"
    kill -TERM $all 2>/dev/null || true
    local q alive
    for _ in $(seq 30); do
        alive=""
        for q in $all; do kill -0 "$q" 2>/dev/null && alive="$alive $q"; done
        [ -z "$alive" ] && break
        sleep 1
    done
    if [ -n "$alive" ]; then
        kill -KILL $alive 2>/dev/null || true
        echo "killed (SIGKILL after 30 s):$alive"
    else
        echo "stopped (server $p and its workers)"
    fi
    echo "gpu 1 compute apps (pid, used_memory):"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 1 || true
}

cmd_status() {
    git -C "$TREE" log -1 --format='worktree %h %s' 2>/dev/null || echo "no worktree at $TREE"
    health || echo "bench server not answering on :$PORT"
}

main() {
    case "${1:-}" in
        deploy) shift; cmd_deploy "$@" ;;
        start) cmd_start ;;
        stop) cmd_stop ;;
        status) cmd_status ;;
        *) echo "usage: $0 {start|stop|status|deploy <branch>}" >&2; exit 2 ;;
    esac
}

# everything above is parsed before main runs, so deploy can check out over this file
main "$@"; exit
