# to install, run:
# powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1

param(
  [string]$Root,
  [string]$Venv,
  [string]$Wheelhouse,
  [string]$VcRedist,
  [string]$CudnnDir
)

$ErrorActionPreference = 'Stop'

# 1. Get script directory
$scriptDir = $null
if ($MyInvocation.MyCommand.Path) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
elseif ($PSCommandPath) { $scriptDir = Split-Path -Parent $PSCommandPath }
else { $scriptDir = (Get-Location).Path }

# 2. Set Root and Defaults
if (-not $PSBoundParameters.ContainsKey('Root')) {
  $Root = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..'))
} else {
  $Root = [System.IO.Path]::GetFullPath($Root)
}

if (-not $Venv)       { $Venv       = Join-Path $Root '.venv' }
if (-not $Wheelhouse) { $Wheelhouse = Join-Path $Root 'vendor\wheelhouse' }
if (-not $VcRedist)   { $VcRedist   = Join-Path $Root 'vendor\etc\VC_redist.x64.exe' }
if (-not $CudnnDir)   { $CudnnDir   = Join-Path $Root 'vendor\cudnn' }

Write-Host "--------------------------------------------------"
Write-Host "Faster-Whisper-API Hybrid Installer"
Write-Host "--------------------------------------------------"
Write-Host "Root:       $Root"
Write-Host "Wheelhouse: $Wheelhouse"
Write-Host "--------------------------------------------------"

if (-not (Test-Path $Root)) { throw "Project root not found: $Root" }

# 3. Python Check
Write-Host "Checking Python version..."
& python -V

# 4. Create venv
if (Test-Path $Venv) {
    Write-Host "venv already exists. Skipping creation."
} else {
    Write-Host "Creating venv..."
    python -m venv $Venv
}
$pip = Join-Path $Venv 'Scripts\pip.exe'
if (-not (Test-Path $pip)) { throw "venv creation failed." }

$reqFile = Join-Path $Root 'requirements.txt'

# 5. Install Dependencies (Hybrid: Offline First -> Online Fallback)
Write-Host "Step 5: Installing Dependencies..."

# 5-1. pip 업그레이드
try {
    Write-Host "  - Upgrading pip (Offline attempt)..."
    & $pip install --no-index --find-links "$Wheelhouse" --upgrade pip 2>$null
    if ($LASTEXITCODE -ne 0) { throw "failed" }
} catch {
    Write-Warning "  - Offline pip upgrade failed. Trying Online..."
    & $pip install --upgrade pip
}

# 5-2. 메인 패키지 설치
try {
    Write-Host "  - Attempting OFFLINE installation from wheelhouse..."
    # --no-index: 인터넷 연결 X
    & $pip install --no-index --find-links "$Wheelhouse" -r "$reqFile"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  - Offline installation SUCCESS!" -ForegroundColor Green
    } else {
        throw "Offline install returned error code."
    }
}
catch {
    Write-Warning "  - Offline installation failed (missing files or build error)."
    Write-Warning "  - Falling back to ONLINE installation..."
    Write-Host "  - Downloading missing packages from PyPI (keeping local wheels priority)..."
    
    # --no-index 제거: 로컬에 없거나 로컬 파일에 문제가 있다면 인터넷 통해 다운로드
    & $pip install --find-links "$Wheelhouse" -r "$reqFile"
}

# 6. Install cuDNN Libraries (Critical for GPU)
$cudnnBin = Join-Path $CudnnDir "bin"
$venvScripts = Join-Path $Venv "Scripts"

if (Test-Path $cudnnBin) {
    Write-Host "Found cuDNN libraries. Copying to venv..."
    Get-ChildItem -Path $cudnnBin -Filter "*.dll" | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination $venvScripts -Force
    }
} else {
    Write-Warning "cuDNN folder not found at $cudnnBin. Check vendor/cudnn."
}

# 7. Check VC++ Runtime
function Test-VcRedistInstalled {
  $key = "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
  if (Test-Path $key) { return ((Get-ItemProperty $key).Installed -eq 1) }
  return $false
}
if (-not (Test-VcRedistInstalled)) {
  if (Test-Path $VcRedist) {
    Write-Host "Installing VC++ Redistributable..."
    Start-Process -FilePath $VcRedist -ArgumentList "/install","/quiet","/norestart" -Wait
  }
}

# 8. Logs & FFmpeg
$logDir = Join-Path $Root 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path "$logDir" | Out-Null }

$localBin = Join-Path $Root "bin"
$ffmpegExe = if (Test-Path "$localBin\ffmpeg.exe") { "$localBin\ffmpeg.exe" } else { "ffmpeg" }

try {
    & $ffmpegExe -version | Select-Object -First 1 | Write-Host
} catch {
    Write-Warning "FFmpeg not found."
}

Write-Host "--------------------------------------------------"
Write-Host "Installation Complete."
Write-Host "& `"$Venv\Scripts\uvicorn.exe`" app.main:app --host 0.0.0.0 --port 8000 --workers 1"
Write-Host "--------------------------------------------------"