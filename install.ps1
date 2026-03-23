$ErrorActionPreference = "Stop"

# Check for 64-bit Windows installation
if (-not [Environment]::Is64BitOperatingSystem) {
    Write-Error "diffusers-workflow requires a 64-bit Windows installation"
    Exit 1
}

# Check for Visual C++ Runtime DLLs
$vcRuntime = Get-Item -ErrorAction SilentlyContinue "$env:windir\System32\msvcp140.dll"
if (-not $vcRuntime.Exists) {
    $vcRuntimeUrl = "https://visualstudio.microsoft.com/downloads/#microsoft-visual-c-redistributable-for-visual-studio-2019"
    Write-Error "Unable to find Visual C++ Runtime DLLs"
    Write-Output "Download and install the Visual C++ Redistributable for Visual Studio 2019 package from: $vcRuntimeUrl"
    Exit 1
}

# Check for Python
try {
    $pythonVersion = (python --version).split(" ")[1]
}
catch {
    Write-Error "Unable to find python"
    $pythonUrl = "https://docs.python.org/3/using/windows.html#installation-steps"
    Write-Output "Note the check box during installation of Python to install the Python Launcher for Windows."
    Write-Output "Install Python from: $pythonUrl"
    Exit 1
}

# Check for supported Python version (3.10 - 3.13)
$supportedPythonVersions = "3.14", "3.13", "3.12", "3.11", "3.10"
if ($env:INSTALL_PYTHON_VERSION) {
    $pythonVersion = $env:INSTALL_PYTHON_VERSION
}
else {
    $pythonVersion = $null
    foreach ($version in $supportedPythonVersions) {
        try {
            $pver = (python --version).split(" ")[1]
            $result = $pver.StartsWith($version)
        }
        catch {
            $result = $false
        }
        if ($result) {
            $pythonVersion = $version
            break
        }
    }
}

if (-not $pythonVersion) {
    $supportedPythonVersions = ($supportedPythonVersions | ForEach-Object { "Python $_" }) -join ", "
    Write-Error "No usable Python version found, supported versions are: $supportedPythonVersions"
    Write-Output "diffusers-workflow requires Python version >= 3.10 and <= 3.14"
    Exit 1
}

# Print Python version
Write-Output "Python version is: $pythonVersion"

# Remove the venv if it exists
if (Test-Path -Path ".\venv" -PathType Container) {
    Remove-Item -LiteralPath ".\venv" -Recurse -Force
}

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\scripts\activate

# Upgrade pip
python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Install build tools
pip install wheel setuptools
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Install PyTorch - use standard install (CUDA auto-detected on Windows)
pip install torch torchvision
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Install Diffusers from GitHub (latest version)
pip install --upgrade git+https://github.com/huggingface/diffusers
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Install core ML dependencies
pip install peft transformers accelerate safetensors controlnet_aux sentencepiece torchsde bitsandbytes torchao gguf kornia ftfy kernels
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Install utility dependencies
pip install aiohttp matplotlib opencv-python-headless concurrent-log-handler qrcode protobuf imageio imageio-ffmpeg beautifulsoup4 soundfile jsonschema black python-dotenv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output ""
Write-Output "Installation complete!"
Write-Output ""
Write-Output "To activate the virtual environment, run:"
Write-Output "  .\venv\scripts\activate"
Write-Output ""
Write-Output "Note: Some LLM workflows (e.g., Phi mini-instruct) require flash_attn,"
Write-Output "which requires the CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit"
