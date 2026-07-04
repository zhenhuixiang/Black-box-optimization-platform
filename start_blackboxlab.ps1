param(
  [int]$Port = 0,
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

if ($Port -le 0) {
  if ($env:LILO_PORT) {
    $Port = [int]$env:LILO_PORT
  } else {
    $Port = 8765
  }
}

$env:PYTHONUNBUFFERED = "1"
$LogFile = Join-Path $ProjectDir "server.start.log"

function Test-PythonCandidate {
  param(
    [string]$Exe,
    [string[]]$ExtraArgs = @()
  )

  try {
    $testArgs = @($ExtraArgs) + @("-c", "import sys, numpy; print(sys.executable)")
    $output = & $Exe @testArgs 2>&1
    if ($LASTEXITCODE -eq 0) {
      return [pscustomobject]@{
        Exe = $Exe
        Args = $ExtraArgs
        Path = (($output | Select-Object -Last 1) -as [string])
      }
    }
  } catch {
    return $null
  }
  return $null
}

$candidates = @()
if ($env:PYTHON) {
  $candidates += [pscustomobject]@{ Exe = $env:PYTHON; Args = @() }
}
$candidates += [pscustomobject]@{ Exe = "python"; Args = @() }
$candidates += [pscustomobject]@{ Exe = "py"; Args = @("-3") }

@(
  "C:\ProgramData\anaconda3\python.exe",
  "$env:USERPROFILE\anaconda3\python.exe",
  "$env:USERPROFILE\miniconda3\python.exe",
  "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
  "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
) | ForEach-Object {
  if ($_ -and (Test-Path $_)) {
    $candidates += [pscustomobject]@{ Exe = $_; Args = @() }
  }
}

$python = $null
foreach ($candidate in $candidates) {
  $python = Test-PythonCandidate -Exe $candidate.Exe -ExtraArgs $candidate.Args
  if ($python) {
    break
  }
}

if (-not $python) {
  "Python with NumPy was not found. Install Python/Anaconda or add it to PATH." | Tee-Object -FilePath $LogFile
  Read-Host "Press Enter to close"
  exit 1
}

$serverArgs = @("server.py", "$Port")
if (-not $NoBrowser) {
  $serverArgs += "--open"
}

Set-Content -Path $LogFile -Value ""
Write-Host "Starting BlackBoxLab..."
Write-Host "Project: $ProjectDir"
Write-Host "Port: $Port"
Write-Host "Python: $($python.Path)"
Write-Host ""
Write-Host "A browser window should open automatically after the backend is ready."
Write-Host "Keep this window open while using BlackBoxLab. Press Ctrl+C here to stop."
Write-Host ""

try {
  $allArgs = @($python.Args) + $serverArgs
  & $python.Exe @allArgs 2>&1 | ForEach-Object {
    Write-Host $_
    Add-Content -Path $LogFile -Value $_ -Encoding UTF8
  }
  $exitCode = $LASTEXITCODE
} catch {
  $_ | Tee-Object -FilePath $LogFile -Append
  $exitCode = 1
}

if ($exitCode -ne 0) {
  Write-Host ""
  Write-Host "BlackBoxLab failed to start. Full log:"
  Write-Host "------------------------------------------------------------"
  Get-Content -Path $LogFile -ErrorAction SilentlyContinue
  Write-Host "------------------------------------------------------------"
  Write-Host ""
  Write-Host "You can also try running this command manually:"
  Write-Host "`"$($python.Path)`" `"$ProjectDir\server.py`" $Port --open"
  Write-Host ""
  Read-Host "Press Enter to close"
  exit $exitCode
}
