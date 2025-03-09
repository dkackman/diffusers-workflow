$ErrorActionPreference = "Stop"

# Check for 64-bit Windows installation
if (-not [Environment]::Is64BitOperatingSystem) {
    Write-Error "chiaSWARM requires a 64-bit Windows installation"
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

# Check for supported Python version
$supportedPythonVersions = "3.12", "3.11", "3.10"
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
    Exit 1
}

# Print Python version
Write-Output "Python version is: $pythonVersion"

# remove the venv if it exists
if (Test-Path -Path ".\venv" -PathType Container) {
    Remove-Item -LiteralPath ".\venv" -Recurse -Force
}

python -m venv venv

.\venv\scripts\activate 

python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

pip install wheel setuptools
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

pip install git+https://github.com/huggingface/diffusers
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# pip install diffusers[torch]
# if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

pip3 install peft transformers accelerate safetensors controlnet_aux sentencepiece mediapipe torchsde bitsandbytes torchao gguf kornia ftfy
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

pip3 install aiohttp matplotlib opencv-python concurrent-log-handler qrcode protobuf imageio imageio-ffmpeg beautifulsoup4 soundfile jsonschema black
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "done"
exit

# to use some LLM workflows like Phi mini-instruct you'll need to install the following
# they however need the cuda dev toolkit to be installed
# https://developer.nvidia.com/cuda-toolkit

# pip install flash_attn
