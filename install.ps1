param(
    [ValidateSet("cpu", "gpu")]
    [string]$Mode = "gpu",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root "venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

function Find-Python {
    $launcher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($launcher) {
        return @{ Command = $launcher.Source; Arguments = @("-3") }
    }
    $python = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($python) {
        return @{ Command = $python.Source; Arguments = @() }
    }
    throw "Python was not found. Install 64-bit Python 3.11 or newer, then run install.bat again."
}

Write-Host "=== Pascribe Desktop Setup ===" -ForegroundColor Cyan
Write-Host "Install mode: $Mode"

$Python = Find-Python
$PythonCommand = $Python.Command
$PythonArguments = $Python.Arguments
$Version = & $PythonCommand @PythonArguments -c "import struct, sys; print('%d.%d (%d-bit)' % (sys.version_info.major, sys.version_info.minor, struct.calcsize('P') * 8)); raise SystemExit(0 if sys.version_info >= (3, 11) and struct.calcsize('P') == 8 else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Pascribe requires 64-bit Python 3.11 or newer. Detected: $Version"
}
Write-Host "Python $Version"

if ($Recreate -and (Test-Path $Venv)) {
    Remove-Item -Recurse -Force $Venv
}
if (-not (Test-Path $VenvPython)) {
    & $PythonCommand @PythonArguments -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Could not prepare pip." }

$Requirements = if ($Mode -eq "gpu") { "requirements-gpu.txt" } else { "requirements.txt" }
& $VenvPython -m pip install --requirement (Join-Path $Root $Requirements)
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

& $VenvPython -c "import faster_whisper, sounddevice, numpy, keyboard, pystray, PIL, pyperclip; print('Dependency check passed')"
if ($LASTEXITCODE -ne 0) { throw "Installed dependencies could not be imported." }

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Run Pascribe with run.bat. Open Diagnostics from the tray menu after launch."
