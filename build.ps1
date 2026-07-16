param(
    [ValidateSet("cpu", "gpu")]
    [string]$Mode = "cpu"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root "venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Run install.bat $Mode before building."
}

& $Python -m pip install --requirement (Join-Path $Root "requirements-dev.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install build dependencies." }
if ($Mode -eq "gpu") {
    & $Python -m pip install --requirement (Join-Path $Root "requirements-gpu.txt")
    if ($LASTEXITCODE -ne 0) { throw "Could not install GPU dependencies." }
}

$GpuCollectArgs = @()
if ($Mode -eq "gpu") {
    $GpuCollectArgs = @(
        "--collect-binaries", "nvidia.cublas",
        "--collect-binaries", "nvidia.cudnn"
    )
}

Push-Location $Root
try {
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --windowed `
        --onedir `
        --name "Pascribe" `
        --collect-all "faster_whisper" `
        --collect-binaries "ctranslate2" `
        --hidden-import "pystray._win32" `
        @GpuCollectArgs `
        "pascribe.py"
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }
}
finally {
    Pop-Location
}

Write-Host "Build complete: dist\Pascribe\Pascribe.exe" -ForegroundColor Green
