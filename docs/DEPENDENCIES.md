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

The install scripts need Python 3.12-3.14. Each deletes and recreates `venv/`, installs PyTorch for the detected GPU, then dw with the `server` and `dev` extras (`pip install -e ".[server,dev]"`), diffusers from git HEAD, the platform extras below, and the web UI's npm packages when `npm` is on the `PATH`.

### Docker (Linux, NVIDIA)

A container image with torch, dw and its dependencies built in, sharing the
workspace and caches with a native install on the same host:
[DOCKER.md](DOCKER.md).

### Manual Installation

```bash
python3 -m venv venv
source venv/bin/activate

pip install torch torchvision torchaudio  # add --index-url https://download.pytorch.org/whl/cu130 for CUDA 13 on Linux
pip install -r requirements.txt           # -e .[server]; requirements-test.txt is -e .[dev]
pip install --force-reinstall --no-deps git+https://github.com/huggingface/diffusers
pip install diffusers                     # adds any dependency the git commit needs
```

dw does not import torchaudio, but diffusers' MiniMax H3 blocks need it to resample an audio reference, so install it from the same index as torch.

## Where the Dependency List Lives

`pyproject.toml` is the single source. Its `dependencies` list and its `server`, `mcp` and `dev` extras are what the wheel declares, what `requirements.txt` and `requirements-test.txt` point at, and what the install scripts and CI install. To add or change a dependency, edit `pyproject.toml`. Refresh the version floors from a working venv with `python scripts/refresh_dep_floors.py`.

The scripts handle what `pyproject.toml` cannot express: the PyTorch index, diffusers from git HEAD, and the platform extras below.

`constraints-openapi.txt` pins the FastAPI and Pydantic that `ui/src/lib/generated/openapi.json` is generated under. CI installs them with `pip -c`, and `scripts/dump_openapi.py` refuses to write under any other versions.

## Notable Dependencies

The full list is in `pyproject.toml`. These are the ones with behaviour worth knowing:

- **spandrel, facexlib** back the `upscale` and `restore_faces` tasks. On first use, `restore_faces` has facexlib download its face detection and parsing weights (about 110 MB) from facexlib's GitHub releases into the facexlib package directory. It is the one model download that bypasses the Hugging Face cache and dw's download manager; while the weights are missing, a validate's `plan.downloads_required` lists them as URL entries.
- **comfy-kitchen** (`<0.3`) is Sol-Attn sparse attention for MiniMax H3's generated-video rows (#931). Its compiled CUDA/HIP kernels ship for Linux and Windows; elsewhere it installs an eager-only build.
- **kernels** fetches Hub-hosted compiled kernels at run time, for example NATTEN's `na3d` for LTX-2.5's diffusion decoder.
- **speechbrain, demucs** back `generate_speech`'s `speaker_embedding` and the `attribute_voices` task. Both are imported lazily, and their weights download only when a call uses them.
- **piexif** embeds generation metadata as EXIF `UserComment` when a step's `result.embed_metadata` is `true` and the content type is JPEG or WebP. PNG embedding uses Pillow's `PngInfo`.
- **fastapi, uvicorn** (the `server` extra) run `dw.serve` and the web UI. **mcp** (the `mcp` extra, also in `dev`) runs `dw_mcp`.

## Platform-Specific Dependencies

**Linux:** bitsandbytes, through its `sys_platform == 'linux'` marker in `pyproject.toml`.

**Windows:** `install.ps1` installs bitsandbytes and kernels explicitly.

**macOS (MPS):** `install.sh` adds fp4-fp8-for-torch-mps (FP8/FP4 dtypes for Metal; autoloads on every torch import, overrides mm/linear/copy on the MPS dispatch key and sets `PYTORCH_ENABLE_MPS_FALLBACK=1`) and fluidtop (a GPU/power TUI; needs sudo). bitsandbytes and flash_attn are not available on macOS.

## Optional

**flash_attn** speeds up attention on CUDA, and some LLM workflows (for example Phi mini-instruct) require it. It needs the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit):

```bash
pip install flash_attn
```

## Specifying Python Version

```bash
export INSTALL_PYTHON_VERSION=3.13
bash ./install.sh
```

Without it, `install.sh` takes the newest of `python3.14`, `python3.13` and `python3.12` on the `PATH`. `install.ps1` always runs plain `python`; `INSTALL_PYTHON_VERSION` only has to match that interpreter's version.

## Troubleshooting

**Hugging Face authentication (401/403 downloading a model):** Most workflows under `workflows/` (Flux, LTX-2, MiniMax...) use **gated** models, which need the repo owner to approve your account. A run against one fails with an error naming the repo and `hf auth login` (mapped from the Hub's 401/403 in `load_component()`, `dw/pipeline_processors/components.py`). Request access on the model's page (e.g. [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)), then log in once:

```bash
hf auth login
```

`workflows/templates/text-to-image.json` uses an ungated model and needs no login, so use it to confirm an install works.

**Package conflicts:** Re-run the install script; it recreates the venv from scratch.

**CUDA not detected:** The install scripts run `nvidia-smi` and read the CUDA version the driver supports. A CUDA 13 driver (>= 580) gets torch from the cu130 index, an older driver gets PyPI's wheel with its bundled CUDA 12 runtime, and no working `nvidia-smi` gets the CPU-only build. A manual `pip install torch` on Linux gets PyPI's wheel. Verify with `python -c "import torch; print(torch.cuda.is_available())"`.

**MPS not detected:** Requires Apple Silicon. Verify with `python -c "import torch; print(torch.backends.mps.is_available())"`.
