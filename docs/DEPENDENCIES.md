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

pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers
```

## Where the Dependency List Lives

`pyproject.toml` is the single source: its `dependencies` list (plus the
`server` and `dev` extras) is what the published wheel declares, what
`requirements.txt` and `requirements-test.txt` resolve (they are one-line
pointers, `-e .[server]` and `-e .[dev]`), and what the install scripts
and CI install. To add or change a dependency, edit pyproject.toml -
every install path picks it up.

Things pyproject can't express stay in the scripts: the PyTorch CUDA
index, diffusers from git HEAD, and the macOS/Windows-specific extras
below. bitsandbytes carries a `sys_platform == 'linux'` marker in
pyproject; install.ps1 installs it explicitly on Windows.

## Platform-Specific Dependencies

**All platforms (core ML):** peft, transformers, accelerate, safetensors, controlnet_aux, sentencepiece, torchsde, torchao, optimum-quanto, gguf, kornia, ftfy, sdnq, spandrel, facexlib (spandrel + facexlib back the `upscale` and `restore_faces` tasks)

**All platforms (utilities):** fastapi, uvicorn (the `dw.serve` HTTP server and web UI), av, aiohttp, matplotlib, opencv-python-headless, concurrent-log-handler, qrcode, protobuf, imageio, imageio-ffmpeg, beautifulsoup4, soundfile, jsonschema, black, python-dotenv

**Linux (CUDA):** bitsandbytes

**Windows (CUDA):** bitsandbytes, kernels

**macOS (MPS):** fp4-fp8-for-torch-mps (FP8/FP4 dtype support for Metal), fluidtop

## Optional

**flash_attn** — Improved attention performance on CUDA. Requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit):

```bash
pip install flash_attn
```

**piexif** (installed by default) — Embeds generation metadata as EXIF `UserComment` when a step's `result.embed_metadata` is `true` and the content type is JPEG/WebP (PNG embedding uses Pillow's `PngInfo` and needs nothing extra). Without it, saving falls back to a logged warning and no embedded metadata.

## Specifying Python Version

```bash
export INSTALL_PYTHON_VERSION=3.13
bash ./install.sh
```

## Troubleshooting

**Package conflicts:** Re-run the install script — it recreates the venv from scratch.

**CUDA not detected:** PyTorch auto-detects CUDA. Verify with `python -c "import torch; print(torch.cuda.is_available())"`.

**MPS not detected:** Requires Apple Silicon. Verify with `python -c "import torch; print(torch.backends.mps.is_available())"`.
