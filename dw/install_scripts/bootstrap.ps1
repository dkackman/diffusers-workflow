# One-line installer for Windows (macOS and Linux: bootstrap.sh):
#   & ([scriptblock]::Create((irm <url>/bootstrap.ps1))) [-TorchIndex URL|pypi|cpu] [-Version X] [dw setup flags...]
#
# Same steps as bootstrap.sh: uv, a Python 3.12 venv at <root>\venv, the torch build
# that fits this machine, diffusers-workflow[server,mcp], diffusers main, install.json,
# then `dw setup`. Re-running it is the upgrade path.
#
# DW_INSTALL_ROOT       install root (default %LOCALAPPDATA%\diffusers-workflow)
# DW_BOOTSTRAP_WHEEL    install this wheel instead of the PyPI release
# DW_BOOTSTRAP_DRY_RUN  print each command (prefixed "+ ") instead of running it
param(
    [string]$TorchIndex = "",
    [string]$Version = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$SetupArgs = @()
)
$ErrorActionPreference = "Stop"

$base = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME "AppData\Local" }
$root = if ($env:DW_INSTALL_ROOT) { $env:DW_INSTALL_ROOT } else { Join-Path $base "diffusers-workflow" }
$venv = Join-Path $root "venv"
$py = Join-Path $venv "Scripts\python.exe"
$record = Join-Path $root "install.json"
$dry = [bool]$env:DW_BOOTSTRAP_DRY_RUN
$cpuUrl = "https://download.pytorch.org/whl/cpu"
$cu130Url = "https://download.pytorch.org/whl/cu130"

function Invoke-Step {
    param([string]$Exe, [string[]]$StepArgs)
    if ($dry) {
        Write-Output ("+ $Exe " + ($StepArgs -join " "))
        return
    }
    & $Exe @StepArgs
    # A throw, never `exit`: run as a script block, `exit` would close the
    # PowerShell window this was typed into, and the error with it
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($StepArgs[0..1] -join ' ') failed (code $LASTEXITCODE)" }
}

if (-not $dry) { New-Item -ItemType Directory -Force -Path $root | Out-Null }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    if ($dry) {
        Write-Output "+ irm https://astral.sh/uv/install.ps1 | iex"
    }
    else {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $env:Path = "$HOME\.local\bin;$env:Path"
    }
}

# --- torch choice. The detection below is copied from install.ps1 (the nvidia-smi
# probe and the driver thresholds): change them together.
function Get-DetectedTorchIndex {
    $driverCuda = $null
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        try {
            $smiOutput = (& nvidia-smi 2>$null) -join "`n"
            if ($LASTEXITCODE -eq 0) {
                $cudaMatch = [regex]::Match($smiOutput, 'CUDA Version:\s*(\d+)')
                if ($cudaMatch.Success) { $driverCuda = [int]$cudaMatch.Groups[1].Value }
                else { $driverCuda = 12 }
            }
        }
        catch { $driverCuda = $null }
    }
    if ($null -eq $driverCuda) { return $cpuUrl }
    if ($driverCuda -ge 13) { return $cu130Url }
    return "pypi"
}

if (-not $TorchIndex -and (Test-Path $record)) {
    try { $TorchIndex = [string](Get-Content $record -Raw | ConvertFrom-Json).torch_index } catch { $TorchIndex = "" }
}
if (-not $TorchIndex) { $TorchIndex = Get-DetectedTorchIndex }
if ($TorchIndex -eq "cpu") { $TorchIndex = $cpuUrl }
if ($TorchIndex -eq "cu130") { $TorchIndex = $cu130Url }
Write-Output "torch index: $TorchIndex"

# --- venv
$existing = Test-Path $py
if (-not $existing) {
    Invoke-Step uv @("venv", "--seed", "--python", "3.12", $venv)
}

# --- an upgrade: end the server setup's task runs before pip touches the venv,
# since uv cannot replace a running dw-serve.exe; `dw setup` starts it again.
# Only a task setup's manifest records is ended ("not running" is fine)
$helper = if ($env:DIFFUSERS_HELPER_ROOT) { $env:DIFFUSERS_HELPER_ROOT } else { Join-Path $HOME ".diffusers_helper" }
$manifest = Join-Path $helper "setup-manifest.json"
$ourTask = $false
if (Test-Path $manifest) {
    try {
        $entries = @((Get-Content $manifest -Raw | ConvertFrom-Json).entries)
        $ourTask = [bool]($entries | Where-Object { $_.kind -eq "service" -and $_.key -eq "dw-serve" })
    }
    catch { $ourTask = $false }
}
if ($ourTask) {
    if ($dry) { Write-Output "+ schtasks /End /TN dw-serve" }
    else {
        # Windows PowerShell 5.1 makes redirected native stderr a terminating
        # error under Stop, and a task that is already gone writes to it
        $ErrorActionPreference = "Continue"
        try { & schtasks /End /TN dw-serve 2>&1 | Out-Null } catch { }
        $ErrorActionPreference = "Stop"
        $global:LASTEXITCODE = 0
    }
}

# --- an upgrade: nothing may still be running from the venv. Claude Desktop and
# Claude Code run its dw-mcp.exe, which uv cannot replace; stop before anything
# is changed. The task ended above gets a few seconds to go
if ($existing -and -not $dry) {
    $running = @()
    foreach ($attempt in 1..10) {
        $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
                $_.Path -and $_.Path.StartsWith($venv, [StringComparison]::OrdinalIgnoreCase)
            })
        if (-not $running) { break }
        Start-Sleep -Seconds 1
    }
    if ($running) {
        $names = ($running | ForEach-Object { $_.ProcessName } | Sort-Object -Unique) -join ", "
        throw "Still running from ${venv}: $names. Close Claude Desktop and Claude Code, then run this again."
    }
}

# --- torch, torchvision and torchaudio from one index so the builds match
$torchArgs = @("pip", "install", "--python", $py)
if ($TorchIndex -ne "pypi") { $torchArgs += @("--index-url", $TorchIndex) }
Invoke-Step uv ($torchArgs + @("torch", "torchvision", "torchaudio"))

# --- diffusers-workflow, with torch pinned to what is now installed
$constraints = Join-Path $root "torch-constraints.txt"
if ($dry) {
    Write-Output "+ write ${constraints}: torch==<installed>"
}
else {
    $torchVersion = (& $py -c "import importlib.metadata as m; print(m.version('torch'))").Trim()
    Set-Content -Path $constraints -Value "torch==$torchVersion"
}
if ($env:DW_BOOTSTRAP_WHEEL) { $req = "$($env:DW_BOOTSTRAP_WHEEL)[server,mcp]" }
elseif ($Version) { $req = "diffusers-workflow[server,mcp]==$Version" }
else { $req = "diffusers-workflow[server,mcp]" }
$installArgs = @("pip", "install", "--python", $py)
if ($existing) { $installArgs += "-U" }
# uv takes a package only from the first index that carries it, and the PyTorch
# index's requests is below dw's floor. first-match goes on to PyPI only when
# that index has no version that fits; it never lets the newest of both win
if ($TorchIndex -ne "pypi") {
    $installArgs += @("--extra-index-url", $TorchIndex, "--index-strategy", "unsafe-first-match")
}
Invoke-Step uv ($installArgs + @("-c", $constraints, $req))

# --- diffusers main, from a GitHub archive (no git needed)
$sha = $null
try {
    if ($env:DW_BOOTSTRAP_DIFFUSERS_SHA) { $candidate = $env:DW_BOOTSTRAP_DIFFUSERS_SHA }
    else { $candidate = [string](Invoke-RestMethod https://api.github.com/repos/huggingface/diffusers/commits/main).sha }
    if ($candidate -cmatch '^[0-9a-f]{40}$') { $sha = $candidate }
}
catch { $sha = $null }
if ($sha) {
    Invoke-Step uv @("pip", "install", "--python", $py, "--reinstall", "--no-deps", "https://github.com/huggingface/diffusers/archive/$sha.zip")
}
else {
    Write-Warning "could not read diffusers main's commit; using the released diffusers (the updater can move it to main later)"
}
Invoke-Step uv @("pip", "install", "--python", $py, "diffusers")

# Windows/CUDA-specific: bitsandbytes' pyproject marker is linux-only (copied from install.ps1)
Invoke-Step uv @("pip", "install", "--python", $py, "bitsandbytes", "kernels")

# --- install.json
$shaJson = if ($sha) { "`"$sha`"" } else { "null" }
$dw = Join-Path $venv "Scripts\dw.exe"
if ($dry) {
    Write-Output "+ write ${record}: {`"torch_index`": `"$TorchIndex`", `"version`": `"<installed>`", `"diffusers_commit`": $shaJson}"
    Write-Output ("+ exec $dw setup " + ($SetupArgs -join " "))
    return
}
$dwVersion = (& $py -c "import importlib.metadata as m; print(m.version('diffusers-workflow'))").Trim()
Set-Content -Path $record -Value "{`"torch_index`": `"$TorchIndex`", `"version`": `"$dwVersion`", `"diffusers_commit`": $shaJson}"
& $dw setup @SetupArgs
if ($LASTEXITCODE -ne 0) { throw "dw setup failed (code $LASTEXITCODE)" }
