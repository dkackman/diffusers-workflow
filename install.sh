#!/bin/bash

# 100% credit to chia and chia dev team - https://github.com/Chia-Network/chia-blockchain

set -o errexit


UBUNTU=false
DEBIAN=false
MACOS=false
if [ "$(uname)" = "Linux" ]; then
  #LINUX=1
  if command -v apt-get >/dev/null; then
    if command -v lsb_release >/dev/null; then
      OS_ID=$(lsb_release -is)
      if [ "$OS_ID" = "Debian" ]; then
        DEBIAN=true
      else
        UBUNTU=true
      fi
    fi
  fi
elif [ "$(uname)" = "Darwin" ]; then
  MACOS=true
fi

# Check for non 64 bit ARM64/Raspberry Pi installs
if [ "$(uname -m)" = "armv7l" ]; then
  echo ""
  echo "WARNING:"
  echo "diffusers-workflow requires a 64 bit OS and this is 32 bit armv7l"
  echo "Exiting."
  exit 1
fi

UBUNTU_PRE_20=0
UBUNTU_20=0
UBUNTU_21=0
UBUNTU_22=0

if $UBUNTU; then
  if command -v lsb_release >/dev/null; then
    LSB_RELEASE=$(lsb_release -rs)
    # In case Ubuntu minimal does not come with bc
    if ! command -v bc > /dev/null 2>&1; then
      sudo apt install bc -y
    fi
    # Mint 20.04 responds with 20 here so 20 instead of 20.04
    if [ "$(echo "$LSB_RELEASE<20" | bc)" = "1" ]; then
      UBUNTU_PRE_20=1
    elif [ "$(echo "$LSB_RELEASE<21" | bc)" = "1" ]; then
      UBUNTU_20=1
    elif [ "$(echo "$LSB_RELEASE<22" | bc)" = "1" ]; then
      UBUNTU_21=1
    else
      UBUNTU_22=1
    fi
  fi
fi

# You can specify preferred python version by exporting `INSTALL_PYTHON_VERSION`
# e.g. `export INSTALL_PYTHON_VERSION=3.8`
INSTALL_PYTHON_PATH=
PYTHON_MAJOR_VER=
PYTHON_MINOR_VER=

find_python() {
  set +e
  unset BEST_VERSION
  for V in 314 3.14 313 3.13 312 3.12 311 3.11 310 3.10; do
    if command -v python$V >/dev/null; then
      if [ "$BEST_VERSION" = "" ]; then
        BEST_VERSION=$V
      fi
    fi
  done

  if [ -n "$BEST_VERSION" ]; then
    INSTALL_PYTHON_VERSION="$BEST_VERSION"
    INSTALL_PYTHON_PATH=python${INSTALL_PYTHON_VERSION}
    PY3_VER=$($INSTALL_PYTHON_PATH --version | cut -d ' ' -f2)
    PYTHON_MAJOR_VER=$(echo "$PY3_VER" | cut -d'.' -f1)
    PYTHON_MINOR_VER=$(echo "$PY3_VER" | cut -d'.' -f2)
  fi
  set -e
}

if [ "$INSTALL_PYTHON_VERSION" = "" ]; then
  echo "Searching available python executables..."
  find_python
else
  echo "Python $INSTALL_PYTHON_VERSION is requested"
  INSTALL_PYTHON_PATH=python${INSTALL_PYTHON_VERSION}
  PY3_VER=$($INSTALL_PYTHON_PATH --version | cut -d ' ' -f2)
  PYTHON_MAJOR_VER=$(echo "$PY3_VER" | cut -d'.' -f1)
  PYTHON_MINOR_VER=$(echo "$PY3_VER" | cut -d'.' -f2)
fi

if ! command -v "$INSTALL_PYTHON_PATH" >/dev/null; then
  echo "${INSTALL_PYTHON_PATH} was not found"
  exit 1
fi

if [ "$PYTHON_MAJOR_VER" -ne "3" ] || [ "$PYTHON_MINOR_VER" -lt "10" ] || [ "$PYTHON_MINOR_VER" -ge "15" ]; then
  echo "diffusers-workflow requires Python version >= 3.10 and <= 3.14" >&2
  echo "Current Python version = $INSTALL_PYTHON_VERSION" >&2
  # If Arch, direct to Arch Wiki
  if type pacman >/dev/null 2>&1 && [ -f "/etc/arch-release" ]; then
    echo "Please see https://wiki.archlinux.org/title/python#Old_versions for support." >&2
  fi

  exit 1
fi
echo "Python version is $INSTALL_PYTHON_VERSION"

# Delete the venv folder if present
if [ -d "venv" ]; then
  rm -rf ./venv
fi

# Create the venv and add soft link to activate
$INSTALL_PYTHON_PATH -m venv venv
if [ ! -f "activate" ]; then
  ln -s venv/bin/activate .
fi

# Activate the virtual environment
# shellcheck disable=SC1091
. ./activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch first - platform/GPU-aware, so the resolver below sees it
# satisfied and leaves it alone. torchaudio is deliberately absent: nothing
# in dw imports it.
#
# PyPI's wheels are CPU-only on Linux (no CUDA), but are the correct choice
# on macOS (MPS support is built in there). So on Linux we probe for an
# actual working NVIDIA driver via nvidia-smi and, if found, point pip at
# the CUDA build index instead of letting it fall through to the CPU wheel.
if $MACOS; then
  echo "macOS detected - installing torch with MPS support"
  pip install torch torchvision
else
  CUDA_AVAILABLE=false
  # `command -v` alone only proves the binary exists; a stale/broken driver
  # can leave nvidia-smi installed but failing, so also require it to run.
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    CUDA_AVAILABLE=true
  fi

  if $CUDA_AVAILABLE; then
    echo "CUDA detected - installing GPU-enabled torch (cu130)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
  else
    echo "No CUDA GPU detected - installing CPU-only torch"
    pip install torch torchvision
  fi
fi

# Everything else resolves from pyproject.toml - the single source of the
# dependency list (bitsandbytes included, via its linux platform marker).
# Editable install with the server + dev extras.
pip install -e ".[server,dev]"

# Diffusers from GitHub (releases lag the newest model pipelines) - after
# the resolver, so it isn't replaced by the released version
pip install --upgrade git+https://github.com/huggingface/diffusers

# Platform-specific extras pyproject can't express
if $MACOS; then
  # FP8/FP4 dtype support for MPS (auto-activates via torch.backends)
  pip install fp4-fp8-for-torch-mps fluidtop
fi

# Web UI dependencies - optional, only needed to build/run the SPA served by dw-serve
if command -v npm >/dev/null; then
  echo ""
  echo "Installing web UI dependencies..."
  (cd ui && npm install)
else
  echo ""
  echo "npm was not found - skipping web UI dependency install."
  echo "Install Node.js/npm and run 'npm install' in the ui/ folder to build the web UI."
fi

echo ""
echo "Installation complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  . ./activate"
echo ""
if $MACOS; then
  echo "Note: Running on Apple Silicon with MPS (Metal Performance Shaders) backend."
  echo "Some features (bitsandbytes quantization, flash_attn) are not available on macOS."
else
  echo "Note: Some LLM workflows (e.g., Phi mini-instruct) require flash_attn,"
  echo "which requires the CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit"
fi
