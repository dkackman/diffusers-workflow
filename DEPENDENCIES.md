# Dependency Management Guide

## Overview

This project uses **install.sh** as the primary installation method. The requirements files serve as documentation and alternative installation paths.

## Installation Methods

### Method 1: Using install.sh (Recommended)

The bash script handles everything including venv creation and CUDA-enabled PyTorch:

```bash
./install.sh
```

This will:
1. Check Python version (3.10-3.12 required)
2. Create/recreate virtual environment
3. Install PyTorch with CUDA 13.0 support
4. Install diffusers from git (latest)
5. Install all dependencies
6. Create activation symlink

### Method 2: Manual pip install

If you prefer manual control:

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with CUDA support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 3. Install diffusers (choose one)
pip install diffusers  # Stable release
# OR
pip install git+https://github.com/huggingface/diffusers  # Latest from git

# 4. Install other dependencies
pip install -r requirements.txt
```

### Method 3: From frozen requirements

To reproduce exact versions from a working environment:

```bash
pip install -r requirements-frozen.txt
```

## Requirements Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Main dependencies with minimum versions (clean, maintainable) |
| `requirements-frozen.txt` | Exact versions from working environment (reproducibility) |
| `requirements-test.txt` | Test dependencies only |
| `requirements-clean.txt` | Same as requirements.txt (kept for reference) |

## Key Dependencies

### Added to match install.sh

The following packages were added to requirements.txt to match install.sh:

- **gguf** - GGUF model format support
- **kornia** - Computer vision transformations
- **ftfy** - Text fixing utilities
- **black** - Code formatter (development)

### Installation Order Matters

install.sh installs in this specific order:

1. **PyTorch** (with CUDA support)
2. **Diffusers** (from git for latest features)
3. **Model libraries** (peft, transformers, accelerate, etc.)
4. **Utilities** (aiohttp, matplotlib, opencv, etc.)

This order ensures compatibility and proper dependency resolution.

## CUDA Support

### PyTorch with CUDA 13.0

Both install.sh and requirements.txt expect CUDA 13.0:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### NVIDIA Libraries

The following NVIDIA packages are installed with PyTorch+CUDA:
- nvidia-cublas-cu12
- nvidia-cuda-cupti-cu12
- nvidia-cuda-nvrtc-cu12
- nvidia-cuda-runtime-cu12
- nvidia-cudnn-cu12
- nvidia-cufft-cu12
- nvidia-curand-cu12
- nvidia-cusolver-cu12
- nvidia-cusparse-cu12
- nvidia-nccl-cu12
- nvidia-nvjitlink-cu12
- nvidia-nvtx-cu12

## Optional Components

### Flash Attention

For improved performance with attention mechanisms:

```bash
# Requires CUDA dev toolkit installed
# https://developer.nvidia.com/cuda-toolkit
pip install flash_attn
```

## Dependency Updates

### Updating requirements.txt

If you add new dependencies to install.sh:

1. Add to install.sh pip install line
2. Add to requirements.txt with minimum version
3. Test with `./install.sh`
4. Regenerate frozen requirements: `pip freeze > requirements-frozen.txt`

### Checking for Updates

```bash
# Check outdated packages
pip list --outdated

# Update specific package
pip install --upgrade <package-name>

# Update all (use with caution)
pip install --upgrade -r requirements.txt
```

## Troubleshooting

### Issue: CUDA not found

**Solution**: Install CUDA-enabled PyTorch explicitly:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Issue: Package conflicts

**Solution**: Use install.sh which installs in proper order:
```bash
./install.sh
```

### Issue: Out of memory during installation

**Solution**: Install packages one at a time or increase swap space.

### Issue: Different Python version

**Solution**: Specify Python version before running install.sh:
```bash
export INSTALL_PYTHON_VERSION=3.11
./install.sh
```

## Version Compatibility

| Component | Minimum | Tested | Notes |
|-----------|---------|--------|-------|
| Python | 3.10 | 3.11 | 3.7-3.12 supported |
| PyTorch | 2.4.1 | 2.4.1 | CUDA 13.0 |
| Diffusers | 0.31.0 | latest | Install from git recommended |
| Transformers | 4.47.0 | 4.47.0 | |
| CUDA | 13.0 | 13.0 | Required for GPU |

## Best Practices

1. **Use install.sh** for initial setup
2. **Keep requirements-frozen.txt** updated for reproducibility
3. **Test after updates** using the test suite
4. **Document changes** in this file when adding dependencies
5. **Use virtual environments** always

## Maintenance Commands

```bash
# Regenerate frozen requirements
pip freeze > requirements-frozen.txt

# Check what's installed
pip list

# Show package details
pip show <package-name>

# Verify installation
python -m dw.test
```

## Summary

- ✅ **install.sh** is the primary installation method
- ✅ **requirements.txt** is now synchronized with install.sh
- ✅ Missing packages (gguf, kornia, ftfy, black) added
- ✅ **requirements-frozen.txt** preserves exact working versions
- ✅ Clear documentation for both methods
