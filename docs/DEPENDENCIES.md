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

**All platforms (core ML):** peft, transformers, accelerate, safetensors, controlnet_aux, sentencepiece, torchsde, torchao, gguf, kornia, ftfy, sdnq, spandrel, facexlib (spandrel + facexlib back the `upscale` and `restore_faces` tasks)

**All platforms (utilities):** av, aiohttp, matplotlib, opencv-python-headless, concurrent-log-handler, qrcode, protobuf, imageio, imageio-ffmpeg, beautifulsoup4, soundfile, jsonschema, black, python-dotenv

**Linux (CUDA):** bitsandbytes

**Windows (CUDA):** bitsandbytes, kernels

**macOS (MPS):** fp4-fp8-for-torch-mps (FP8/FP4 dtype support for Metal)

## Optional

**flash_attn** — Improved attention performance on CUDA. Requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit):

```bash
pip install flash_attn
```

**piexif** — Embeds generation metadata as EXIF `UserComment` when a step's `result.embed_metadata` is `true` and the content type is JPEG/WebP (PNG embedding uses Pillow's `PngInfo` and needs nothing extra). Without it, saving falls back to a logged warning and no embedded metadata:

```bash
pip install piexif
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
