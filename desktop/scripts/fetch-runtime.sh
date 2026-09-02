#!/usr/bin/env bash
# Fetch the Python interpreter and uv that get bundled into the installer.
#
#   scripts/fetch-runtime.sh aarch64-apple-darwin
#
# These are the only two things the installer actually carries; every
# Python package is fetched at provisioning time. Both are pinned by
# version AND by SHA256 - an installer that silently picked up a different
# interpreter than the one that was tested is not a reproducible build.
#
# To move to a newer runtime, bump the versions below and replace the
# hashes with the published ones:
#   uv   https://github.com/astral-sh/uv/releases -> <asset>.sha256
#   pbs  https://github.com/astral-sh/python-build-standalone/releases -> SHA256SUMS
set -euo pipefail

TARGET="${1:?usage: fetch-runtime.sh <rust target triple>}"

UV_VERSION="0.12.9"
# Interpreter floor for the project is 3.10; 3.12 is what CI tests on
PBS_TAG="20260901"
PBS_PYTHON="3.12.14"

case "$TARGET" in
  aarch64-apple-darwin)
    UV_ASSET="uv-aarch64-apple-darwin.tar.gz"
    UV_SHA="301f72afaf54060f92da7016cb0115bd077f43a9c8e39c1d8170a0bac80fd398"
    PBS_SHA="3ee3ee547cedfeb7c2b16b2b7156039f7b470bb8f857e226fd3d2eb11db83c76"
    ;;
  x86_64-pc-windows-msvc)
    UV_ASSET="uv-x86_64-pc-windows-msvc.zip"
    UV_SHA="ddbfcee1ac615a0499f6aa97b5ec8ebdf3ee4a7714a48055ec2ba0030e3cf810"
    PBS_SHA="e90c1b6419da3bd812dd73bb3de40287a21abf153438147639ec5e20375ea93f"
    ;;
  x86_64-unknown-linux-gnu)
    UV_ASSET="uv-x86_64-unknown-linux-gnu.tar.gz"
    UV_SHA="ec7a99cd05e0cd7f80243f135ce1361c76835cb0ee60055d14d20eba8eba1460"
    PBS_SHA="936c246dfdbbfa7cb22dd01814a21f582a892689fae96b06071a5e433baffa22"
    ;;
  *)
    echo "error: no pinned runtime for target '$TARGET'" >&2
    exit 1
    ;;
esac

PBS_ASSET="cpython-${PBS_PYTHON}+${PBS_TAG}-${TARGET}-install_only.tar.gz"
UV_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${UV_ASSET}"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${PBS_ASSET}"

cd "$(dirname "$0")/.."
RESOURCES="src-tauri/resources"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

verify() {
  # shasum is present on macOS and Linux runners; sha256sum is not on macOS
  local actual
  actual=$(shasum -a 256 "$1" | cut -d' ' -f1)
  if [ "$actual" != "$2" ]; then
    echo "error: checksum mismatch for $1" >&2
    echo "  expected $2" >&2
    echo "  actual   $actual" >&2
    exit 1
  fi
}

echo "Fetching uv $UV_VERSION for $TARGET"
curl -fsSL "$UV_URL" -o "$WORK/$UV_ASSET"
verify "$WORK/$UV_ASSET" "$UV_SHA"

echo "Fetching CPython $PBS_PYTHON ($PBS_TAG) for $TARGET"
curl -fsSL "$PBS_URL" -o "$WORK/$PBS_ASSET"
verify "$WORK/$PBS_ASSET" "$PBS_SHA"

rm -rf "$RESOURCES/python"
mkdir -p "$RESOURCES"

# python-build-standalone unpacks to a top-level python/ directory
tar -xzf "$WORK/$PBS_ASSET" -C "$RESOURCES"
test -d "$RESOURCES/python" || { echo "error: expected $RESOURCES/python" >&2; exit 1; }

case "$UV_ASSET" in
  *.zip) unzip -qo "$WORK/$UV_ASSET" -d "$WORK/uv-extract" ;;
  *)     mkdir -p "$WORK/uv-extract" && tar -xzf "$WORK/$UV_ASSET" -C "$WORK/uv-extract" ;;
esac
# The archive nests the binary one directory deep on some targets
UV_BIN=$(find "$WORK/uv-extract" -name 'uv' -o -name 'uv.exe' | head -1)
test -n "$UV_BIN" || { echo "error: no uv binary in $UV_ASSET" >&2; exit 1; }
if [ "${TARGET}" = "x86_64-pc-windows-msvc" ]; then
  cp "$UV_BIN" "$RESOURCES/uv.exe"
else
  cp "$UV_BIN" "$RESOURCES/uv"
  chmod +x "$RESOURCES/uv"
fi

echo "Runtime ready in $RESOURCES"
