# sference CLI installer (Windows)
# Usage: powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/s-ference/sference/main/install.ps1 | iex"
#
# Optional environment:
#   $env:SFERENCE_CLI_VERSION = "0.0.7"   pin PyPI version (default: latest)
#   $env:SFERENCE_NO_UV_BOOTSTRAP = "1"    fail instead of installing uv when missing

$ErrorActionPreference = "Stop"

$Package = "sference-cli"
$Binary = "sference"
$UvInstallUrl = "https://astral.sh/uv/install.ps1"
$DefaultBinDir = Join-Path $env:USERPROFILE ".local\bin"

function Write-Info([string]$Message) {
    Write-Host $Message
}

function Write-Success([string]$Message) {
    Write-Host $Message -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host $Message -ForegroundColor Yellow
}

function Ensure-Path {
    if (-not ($env:PATH -split ';' | Where-Object { $_ -eq $DefaultBinDir })) {
        $env:PATH = "$DefaultBinDir;$env:PATH"
    }
}

function Ensure-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return
    }
    if ($env:SFERENCE_NO_UV_BOOTSTRAP -eq "1") {
        throw "uv is not installed. Install uv (https://docs.astral.sh/uv/) or unset SFERENCE_NO_UV_BOOTSTRAP."
    }
    Write-Info "Installing uv (Python toolchain manager)..."
    Invoke-Expression ((Invoke-WebRequest -UseBasicParsing $UvInstallUrl).Content)
    Ensure-Path
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv installation failed"
    }
}

function Install-WithUv {
    $versionSpec = ""
    if ($env:SFERENCE_CLI_VERSION) {
        $versionSpec = "==$($env:SFERENCE_CLI_VERSION)"
    }
    Write-Info "Installing $Package$versionSpec with uv..."
    uv tool install "$Package$versionSpec" --force
}

function Install-WithPipx {
    if (-not (Get-Command pipx -ErrorAction SilentlyContinue)) {
        return $false
    }
    $versionSpec = ""
    if ($env:SFERENCE_CLI_VERSION) {
        $versionSpec = "==$($env:SFERENCE_CLI_VERSION)"
    }
    Write-Info "Installing $Package$versionSpec with pipx..."
    pipx install "$Package$versionSpec" --force
    return $true
}

function Install-WithPip {
    $python = $null
    foreach ($candidate in @("python", "py")) {
        if (-not (Get-Command $candidate -ErrorAction SilentlyContinue)) {
            continue
        }
        try {
            & $candidate -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $python = $candidate
                break
            }
        } catch {
            continue
        }
    }
    if (-not $python) {
        return $false
    }
    $versionSpec = ""
    if ($env:SFERENCE_CLI_VERSION) {
        $versionSpec = "==$($env:SFERENCE_CLI_VERSION)"
    }
    Write-Info "Installing $Package$versionSpec with $python -m pip --user..."
    & $python -m pip install --user "$Package$versionSpec"
    return $true
}

function Warn-Path {
    if ($env:PATH -split ';' | Where-Object { $_ -eq $DefaultBinDir }) {
        return
    }
    Write-Host ""
    Write-Warn "Add $DefaultBinDir to your PATH if sference is not found."
    Write-Host '  setx PATH "$env:USERPROFILE\.local\bin;$env:PATH"'
    Write-Host ""
}

function Verify-Install {
    Ensure-Path
    if (-not (Get-Command $Binary -ErrorAction SilentlyContinue)) {
        Warn-Path
        throw "$Binary was installed but is not on PATH."
    }
    try {
        $installedVersion = & $Binary --version 2>$null
        if ($installedVersion) {
            Write-Success "Installed $installedVersion"
        } else {
            Write-Success "Installed $Binary to $((Get-Command $Binary).Source)"
        }
    } catch {
        Write-Success "Installed $Binary to $((Get-Command $Binary).Source)"
    }
}

function Print-GetStarted {
    Write-Host ""
    Write-Info "Get started:"
    Write-Host "  sference auth login              # Authenticate via browser"
    Write-Host "  sference auth login --api-key    # Authenticate with an API key"
    Write-Host "  sference batch submit --help     # Submit a batch job"
    Write-Host "  sference --help                  # See all commands"
    Write-Host ""
}

Write-Host ""
Write-Host "  ╔══════════════════════════════════════╗"
Write-Host "  ║           sference CLI               ║"
Write-Host "  ╚══════════════════════════════════════╝"
Write-Host ""

Ensure-Path

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    if ($env:SFERENCE_NO_UV_BOOTSTRAP -eq "1") {
        Write-Warn "uv not found; trying pipx or pip..."
    } else {
        Ensure-Uv
    }
}

if (Get-Command uv -ErrorAction SilentlyContinue) {
    Install-WithUv
} elseif (-not (Install-WithPipx)) {
    if (-not (Install-WithPip)) {
        throw "Could not install $Package. Install uv (https://docs.astral.sh/uv/) or Python 3.12+, then retry."
    }
}

Write-Host ""
Verify-Install
Print-GetStarted
