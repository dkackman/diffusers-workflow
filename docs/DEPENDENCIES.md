# Dependencies

## Installation

### Linux / macOS (Recommended)

```bash
bash ./install.sh
source ./activate
```

### Windows

```powershell
.\install.ps1
.\venv\scripts\activate
```

The install scripts handle everything: Python version detection (3.10-3.14), virtual environment creation, PyTorch, diffusers, and all dependencies.

### Manual Installation

If you prefer manual control:

```bash
python3 -m venv venv
source venv/bin/activate

pip install torch torchvision
pip install git+https://github.com/huggingface/diffusers

pip install -r requirements.txt
```

## Platform-Specific Dependencies

**All platforms:** peft, transformers, accelerate, safetensors, controlnet_aux, sentencepiece, torchsde, torchao, gguf, kornia, ftfy, sdnq

**Linux/Windows (CUDA):** bitsandbytes, kernels

**macOS (MPS):** fp4-fp8-for-torch-mps (FP8/FP4 dtype support for Metal)

## Optional

**flash_attn** — Improved attention performance on CUDA. Requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit):

```bash
pip install flash_attn
```

## Specifying Python Version

```bash
export INSTALL_PYTHON_VERSION=3.13
bash ./install.sh
```

## Troubleshooting

**Package conflicts:** Re-run the install script — it recreates the venv from scratch.

**CUDA not detected:** PyTorch auto-detects CUDA. Verify with `python -c "import torch; print(torch.cuda.is_available())"`.

**MPS not detected:** Requires Apple Silicon. Verify with `python -c "import torch; print(torch.backends.mps.is_available())"`.
