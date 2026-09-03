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

pip install torch torchvision torchaudio  # add --index-url https://download.pytorch.org/whl/cu130 for CUDA on Linux
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

**Hugging Face authentication (401/403 downloading a model):** Most workflows under `workflows/` (Flux, LTX-2, MiniMax...) point at **gated** models on the Hub — repos that require the owner to approve your account before you can download them. A run against one of these fails with an actionable error naming the repo and `huggingface-cli login` (mapped from the Hub's 401/403 in `load_component()`, `dw/pipeline_processors/pipeline.py`) — request access on the model's page (e.g. [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)) and then log in once, locally:

```bash
huggingface-cli login
```

`workflows/sd15.json` uses an ungated model and needs no login - start there if you just want to confirm the install works.

**Package conflicts:** Re-run the install script — it recreates the venv from scratch.

**CUDA not detected:** `install.sh` probes for a working `nvidia-smi` on Linux and installs the CUDA build of torch (cu130) only when it finds one; otherwise (or on a manual install with plain `pip install torch`) you get PyPI's CPU-only Linux wheel. Verify with `python -c "import torch; print(torch.cuda.is_available())"`.

**MPS not detected:** Requires Apple Silicon. Verify with `python -c "import torch; print(torch.backends.mps.is_available())"`.
