#!/usr/bin/env bash
# One-line installer for macOS and Linux (Windows: bootstrap.ps1):
#   curl -LsSf <url>/bootstrap.sh | bash -s -- [--torch-index URL|pypi|cpu] [--version X] [dw setup flags...]
#
# Installs uv, a Python 3.12 venv at <root>/venv, the torch build that fits this
# machine, diffusers-workflow[server,mcp] and diffusers main, records the choice
# in <root>/install.json, then hands off to `dw setup`. Re-running it is the
# upgrade path: the recorded torch index is reused and torch is pinned.
#
# DW_INSTALL_ROOT       install root (default ~/.local/share/diffusers-workflow)
# DW_BOOTSTRAP_WHEEL    install this wheel instead of the PyPI release
# DW_BOOTSTRAP_DIFFUSERS_SHA  use this commit instead of asking GitHub for main's (tests)
# DW_BOOTSTRAP_DRY_RUN  print each command (prefixed "+ ") instead of running it
# DW_BOOTSTRAP_TTY      test-only, honoured only with DW_BOOTSTRAP_DRY_RUN: 0 or 1
#                       says whether /dev/tty is available. A real run ignores it
#                       and probes the terminal.
set -eu

# Everything runs inside main, called on the last line, so a download cut short
# runs nothing (and bash reading the script from a pipe never half-executes it).
main() {
  ROOT="${DW_INSTALL_ROOT:-$HOME/.local/share/diffusers-workflow}"
  VENV="$ROOT/venv"
  PY="$VENV/bin/python"
  RECORD="$ROOT/install.json"
  DRY="${DW_BOOTSTRAP_DRY_RUN:-}"
  CPU_URL="https://download.pytorch.org/whl/cpu"
  CU130_URL="https://download.pytorch.org/whl/cu130"

  TORCH_INDEX=""
  VERSION=""
  PASS=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --torch-index) TORCH_INDEX="${2:?--torch-index needs a value}"; shift 2 ;;
      --torch-index=*) TORCH_INDEX="${1#*=}"; shift ;;
      --version) VERSION="${2:?--version needs a value}"; shift 2 ;;
      --version=*) VERSION="${1#*=}"; shift ;;
      *) PASS+=("$1"); shift ;;
    esac
  done

  run() {
    if [ -n "$DRY" ]; then
      echo "+ $*"
    else
      "$@"
    fi
  }

  if [ -z "$DRY" ]; then mkdir -p "$ROOT" 2>/dev/null || true; fi
  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v uv >/dev/null 2>&1; then
    if [ -n "$DRY" ]; then
      echo "+ curl -LsSf https://astral.sh/uv/install.sh | sh"
    else
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
  fi

  # --- torch choice. The detection below is copied from install.sh (the macOS pin,
  # the nvidia-smi probe and the driver thresholds): change them together.
  MACOS=false
  [ "$(uname -s)" = "Darwin" ] && MACOS=true

  detect_torch_index() {
    if $MACOS; then
      echo pypi
      return
    fi
    local driver_cuda=""
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
      driver_cuda=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*' | grep -o '[0-9]*$' || true)
      driver_cuda=${driver_cuda:-12}
    fi
    if [ -z "$driver_cuda" ]; then
      echo "$CPU_URL"
    elif [ "$driver_cuda" -ge 13 ]; then
      echo "$CU130_URL"
    else
      echo pypi
    fi
  }

  recorded_torch_index() {
    [ -f "$RECORD" ] || return 0
    sed -n 's/.*"torch_index"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$RECORD" | head -n 1
  }

  if [ -z "$TORCH_INDEX" ]; then
    TORCH_INDEX="$(recorded_torch_index)"
  fi
  if [ -z "$TORCH_INDEX" ]; then
    TORCH_INDEX="$(detect_torch_index)"
  fi
  case "$TORCH_INDEX" in
    cpu) TORCH_INDEX="$CPU_URL" ;;
    cu130) TORCH_INDEX="$CU130_URL" ;;
  esac
  echo "torch index: $TORCH_INDEX"

  # --- venv
  EXISTING=false
  [ -x "$PY" ] && EXISTING=true
  if ! $EXISTING; then
    run uv venv --seed --python 3.12 "$VENV"
  fi

  # --- torch, torchvision and torchaudio from one index so the builds match
  TORCH_PKG="torch"
  $MACOS && TORCH_PKG="torch<2.15" # torch 2.15 + the fp4-fp8 MPS autoload breaks scalar ** mps_tensor (#1395)
  if [ "$TORCH_INDEX" = "pypi" ]; then
    run uv pip install --python "$PY" "$TORCH_PKG" torchvision torchaudio
  else
    run uv pip install --python "$PY" --index-url "$TORCH_INDEX" "$TORCH_PKG" torchvision torchaudio
  fi

  # --- diffusers-workflow, with torch pinned to what is now installed so the
  # resolver cannot swap the build
  CONSTRAINTS="$ROOT/torch-constraints.txt"
  if [ -n "$DRY" ]; then
    echo "+ write $CONSTRAINTS: torch==<installed>"
  else
    torch_version=$("$PY" -c 'import importlib.metadata as m; print(m.version("torch"))')
    printf 'torch==%s\n' "$torch_version" >"$CONSTRAINTS"
  fi

  if [ -n "${DW_BOOTSTRAP_WHEEL:-}" ]; then
    REQ="${DW_BOOTSTRAP_WHEEL}[server,mcp]"
  elif [ -n "$VERSION" ]; then
    REQ="diffusers-workflow[server,mcp]==$VERSION"
  else
    REQ="diffusers-workflow[server,mcp]"
  fi
  UPGRADE=()
  $EXISTING && UPGRADE=(-U)
  if [ "$TORCH_INDEX" = "pypi" ]; then
    run uv pip install --python "$PY" "${UPGRADE[@]+"${UPGRADE[@]}"}" -c "$CONSTRAINTS" "$REQ"
  else
    # Both indexes weighed: uv takes a package from the first index that
    # carries it, and the PyTorch index's requests is below dw's floor
    run uv pip install --python "$PY" "${UPGRADE[@]+"${UPGRADE[@]}"}" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match -c "$CONSTRAINTS" "$REQ"
  fi

  # --- diffusers main, from a GitHub archive (no git needed). Forced because main
  # keeps one dev version string for a release cycle; --no-deps so torch is left
  # alone, then a plain install adds what the new commit needs (dw/server/updater.py).
  if [ -n "${DW_BOOTSTRAP_DIFFUSERS_SHA:-}" ]; then
    SHA="$DW_BOOTSTRAP_DIFFUSERS_SHA" # hermetic override (tests); same 40-hex check below
  else
    SHA=$(curl -fsSL https://api.github.com/repos/huggingface/diffusers/commits/main 2>/dev/null |
      sed -n 's/.*"sha"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1 || true)
  fi
  case "$SHA" in
    *[!0-9a-f]*) SHA="" ;;
  esac
  [ "${#SHA}" -eq 40 ] || SHA=""
  if [ -n "$SHA" ]; then
    run uv pip install --python "$PY" --reinstall --no-deps "https://github.com/huggingface/diffusers/archive/$SHA.zip"
    run uv pip install --python "$PY" diffusers
  else
    echo "warning: could not read diffusers main's commit; using the released diffusers (the updater can move it to main later)" >&2
    run uv pip install --python "$PY" diffusers
  fi

  # macOS extras pyproject cannot express (copied from install.sh)
  if $MACOS; then
    run uv pip install --python "$PY" fp4-fp8-for-torch-mps fluidtop
  fi

  # --- install.json
  # Under `curl | sh` stdin is the pipe, so dw setup's prompts would see no
  # terminal: hand it /dev/tty unless --yes (take every default) was passed.
  TTY_NOTE=""
  WANT_TTY=true
  for a in ${PASS[@]+"${PASS[@]}"}; do [ "$a" = "--yes" ] && WANT_TTY=false; done
  # DW_BOOTSTRAP_TTY (test hook, dry runs only): 0 = /dev/tty unavailable,
  # 1 = available, unset = probe it. A real run always probes.
  TTY_HOOK=""
  [ -n "$DRY" ] && TTY_HOOK="${DW_BOOTSTRAP_TTY:-}"
  case "$TTY_HOOK" in
    0) HAVE_TTY=false ;;
    1) HAVE_TTY=true ;;
    *) if (: </dev/tty) 2>/dev/null; then HAVE_TTY=true; else HAVE_TTY=false; fi ;;
  esac
  if $WANT_TTY && $HAVE_TTY; then TTY_NOTE=" </dev/tty"; fi
  if [ -n "$SHA" ]; then SHA_JSON="\"$SHA\""; else SHA_JSON="null"; fi
  if [ -n "$DRY" ]; then
    echo "+ write $RECORD: {\"torch_index\": \"$TORCH_INDEX\", \"version\": \"<installed>\", \"diffusers_commit\": $SHA_JSON}"
    echo "+ exec $VENV/bin/dw setup ${PASS[*]+"${PASS[*]}"}${TTY_NOTE}"
    exit 0
  fi
  DW_VERSION=$("$PY" -c 'import importlib.metadata as m; print(m.version("diffusers-workflow"))')
  printf '{"torch_index": "%s", "version": "%s", "diffusers_commit": %s}\n' \
    "$TORCH_INDEX" "$DW_VERSION" "$SHA_JSON" >"$RECORD"

  if [ -n "$TTY_NOTE" ]; then
    exec "$VENV/bin/dw" setup ${PASS[@]+"${PASS[@]}"} </dev/tty
  fi
  exec "$VENV/bin/dw" setup ${PASS[@]+"${PASS[@]}"}
}

main "$@"
