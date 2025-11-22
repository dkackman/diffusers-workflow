#!/bin/bash

# 100% credit to chia and chia dev team - https://github.com/Chia-Network/chia-blockchain

set -o errexit


UBUNTU=false
DEBIAN=false
if [ "$(uname)" = "Linux" ]; then
  #LINUX=1
  if command -v apt-get >/dev/null; then
    OS_ID=$(lsb_release -is)
    if [ "$OS_ID" = "Debian" ]; then
      DEBIAN=true
    else
      UBUNTU=true
    fi
  fi
fi

# Check for non 64 bit ARM64/Raspberry Pi installs
if [ "$(uname -m)" = "armv7l" ]; then
  echo ""
  echo "WARNING:"
  echo "The diffusers helper requires a 64 bit OS and this is 32 bit armv7l"
  echo "Exiting."
  exit 1
fi

UBUNTU_PRE_20=0
UBUNTU_20=0
UBUNTU_21=0
UBUNTU_22=0

if $UBUNTU; then
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

# You can specify preferred python version by exporting `INSTALL_PYTHON_VERSION`
# e.g. `export INSTALL_PYTHON_VERSION=3.8`
INSTALL_PYTHON_PATH=
PYTHON_MAJOR_VER=
PYTHON_MINOR_VER=

find_python() {
  set +e
  unset BEST_VERSION
  for V in 312 3.12 311 3.11 310 3.10; do
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

if [ "$PYTHON_MAJOR_VER" -ne "3" ] || [ "$PYTHON_MINOR_VER" -lt "7" ] || [ "$PYTHON_MINOR_VER" -ge "13" ]; then
  echo "The diffusers helper requires Python version >= 3.7 and  <= 3.12.0" >&2
  echo "Current Python version = $INSTALL_PYTHON_VERSION" >&2
  # If Arch, direct to Arch Wiki
  if type pacman >/dev/null 2>&1 && [ -f "/etc/arch-release" ]; then
    echo "Please see https://wiki.archlinux.org/title/python#Old_versions for support." >&2
  fi

  exit 1
fi
echo "Python version is $INSTALL_PYTHON_VERSION"

# delete the venv folder if present
if [ -d "venv" ]; then
  rm ./venv -rf
fi

# create the venv and add soft link to activate
$INSTALL_PYTHON_PATH -m venv venv
if [ ! -f "activate" ]; then
  ln -s venv/bin/activate .
fi

# shellcheck disable=SC1091
. ./activate

python -m pip install --upgrade pip

pip install wheel setuptools
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install git+https://github.com/huggingface/diffusers
# pip install diffusers[torch]
pip install peft transformers accelerate safetensors controlnet_aux sentencepiece torchsde bitsandbytes torchao gguf kornia ftfy
pip install aiohttp matplotlib opencv-python concurrent-log-handler qrcode protobuf imageio imageio-ffmpeg beautifulsoup4 soundfile jsonschema black

# to use some LLM workflows like Phi mini-instruct you'll need to install the following
# they however need the cuda dev toolkit to be installed
# https://developer.nvidia.com/cuda-toolkit

# pip install flash_attn