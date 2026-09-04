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

# Check for supported Python version (3.10 - 3.14)
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

# Install PyTorch first - GPU-aware, so the resolver below sees it satisfied
# and leaves it alone. torchaudio comes with it, from the same index so its
# build matches: dw does not import it, but diffusers' MiniMax H3 blocks use it
# to resample an audio reference that is not already at the audio VAE's rate,
# and without it those pipelines fail at the first setup block.
#
# PyPI's Windows wheel bundles a CUDA 12.x runtime; pytorch.org's cu130 index
# carries a CUDA 13 build that needs a newer driver (>= 580). So we probe for a
# working NVIDIA driver via nvidia-smi and, if found, read the CUDA version it
# supports: a CUDA 13-capable driver gets the cu130 build, an older one stays on
# PyPI's CUDA 12 wheel rather than a build its driver cannot load (which would
# silently fall back to the CPU).
$driverCuda = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    # Finding the binary only proves it exists; a stale or broken driver can
    # leave nvidia-smi installed but failing, so require it to run too.
    try {
        $smiOutput = (& nvidia-smi 2>$null) -join "`n"
        if ($LASTEXITCODE -eq 0) {
            $cudaMatch = [regex]::Match($smiOutput, 'CUDA Version:\s*(\d+)')
            if ($cudaMatch.Success) {
                $driverCuda = [int]$cudaMatch.Groups[1].Value
            }
            else {
                $driverCuda = 12
            }
        }
    }
    catch {
        $driverCuda = $null
    }
}

if ($null -eq $driverCuda) {
    Write-Output "No CUDA GPU detected - installing CPU-only torch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}
elseif ($driverCuda -ge 13) {
    Write-Output "CUDA $driverCuda driver detected - installing GPU-enabled torch (cu130)"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
}
else {
    Write-Output "CUDA $driverCuda driver detected - installing PyPI torch (bundled CUDA 12 runtime)"
    pip install torch torchvision torchaudio
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Everything else resolves from pyproject.toml - the single source of the
# dependency list. Editable install with the server + dev extras.
pip install -e ".[server,dev]"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Diffusers from GitHub (releases lag the newest model pipelines) - after
# the resolver, so it isn't replaced by the released version
pip install --upgrade git+https://github.com/huggingface/diffusers
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Windows/CUDA-specific: bitsandbytes' pyproject marker is linux-only
pip install bitsandbytes kernels
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Web UI dependencies - optional, only needed to build/run the SPA served by dw-serve
if (Get-Command npm -ErrorAction SilentlyContinue) {
    Write-Output ""
    Write-Output "Installing web UI dependencies..."
    Push-Location ui
    npm install
    $npmExitCode = $LASTEXITCODE
    Pop-Location
    if ($npmExitCode -ne 0) { exit $npmExitCode }
}
else {
    Write-Output ""
    Write-Output "npm was not found - skipping web UI dependency install."
    Write-Output "Install Node.js/npm and run 'npm install' in the ui\ folder to build the web UI."
}

Write-Output ""
Write-Output "Installation complete!"
Write-Output ""
Write-Output "To activate the virtual environment, run:"
Write-Output "  .\venv\scripts\activate"
Write-Output ""
Write-Output "Note: Some LLM workflows (e.g., Phi mini-instruct) require flash_attn,"
Write-Output "which requires the CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit"
