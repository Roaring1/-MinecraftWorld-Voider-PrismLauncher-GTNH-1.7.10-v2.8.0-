# build_voider.ps1
# GTNH World Voider — Windows build script (PowerShell 5.1 compatible)
# Creates/uses .venv, installs deps, sanity-checks imports, and builds ONEDIR + ONEFILE.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --------------------------- CONFIG ------------------------------------------
$AppName        = 'Voider-GTNH'
$EntryPy        = 'main_voider.py'
$IconIco        = 'voider.ico'           # optional; ignored if missing
$VenvDir        = '.venv'
$DoSkipOneDir   = $false                  # $true to skip the ONEDIR pass
$OneFileConsole = $false                  # $true to build console ONEFILE (for debugging)
$CleanFirst     = $false                  # $true to delete build/dist/*.spec first
# -----------------------------------------------------------------------------


# --------------------------- Helpers -----------------------------------------
function Ensure-Folder { param([Parameter(Mandatory=$true)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
  }
}

function Write-Section { param([string]$Text)
  Write-Host "`n=== $Text ===" -ForegroundColor Cyan
}

function Tail-If { param([string]$Path, [int]$Lines=140, [string]$Label="")
  if (Test-Path -LiteralPath $Path) {
    if ($Label) { Write-Host "---- $Label ----" -ForegroundColor DarkCyan }
    Get-Content -LiteralPath $Path -Tail $Lines
  }
}

function Clean-BuildOutputs {
  Write-Section "Cleaning build artifacts"
  Remove-Item -Recurse -Force @('build','dist') -ErrorAction SilentlyContinue | Out-Null
  Get-ChildItem -Filter '*.spec' -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
  }
}

function Run-Logged {
  param(
    [Parameter(Mandatory=$true)][string]$Exe,
    [Parameter(Mandatory=$true)][string[]]$ArgList,
    [Parameter(Mandatory=$true)][string]$StdOut,
    [Parameter(Mandatory=$true)][string]$StdErr
  )
  $ArgList = @($ArgList | Where-Object { $_ -ne $null -and $_ -ne '' })
  Write-Host ("`n>>> {0} {1}" -f $Exe, ($ArgList -join ' ')) -ForegroundColor Cyan
  $p = Start-Process -FilePath $Exe -ArgumentList $ArgList `
       -NoNewWindow -RedirectStandardOutput $StdOut -RedirectStandardError $StdErr -PassThru
  $p.WaitForExit()
  return $p.ExitCode
}

function Get-CommandPath([string]$Name) {
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Path } else { return $null }
}

# --------------------------- Paths/Logs --------------------------------------
# PS 5.1-safe: use $MyInvocation.MyCommand.Path (works for dot/run)
$ScriptPath = $MyInvocation.MyCommand.Path
if (-not $ScriptPath) { $ScriptPath = (Get-Location).Path }   # rare fallback
$Here = Split-Path -Path $ScriptPath -Parent
Set-Location -Path $Here

$PyExe        = Join-Path $VenvDir 'Scripts\python.exe'
$LogsDir      = 'build-logs'
$TmpDir       = '.build_tmp'

$OneDirOut    = Join-Path $LogsDir 'onedir-out.txt'
$OneDirErr    = Join-Path $LogsDir 'onedir-err.txt'
$OneFileOut   = Join-Path $LogsDir 'onefile-out.txt'
$OneFileErr   = Join-Path $LogsDir 'onefile-err.txt'

Ensure-Folder $LogsDir
Ensure-Folder $TmpDir

if ($CleanFirst) { Clean-BuildOutputs }

if (-not (Test-Path -LiteralPath $EntryPy)) {
  throw "Entry script not found: $EntryPy (run from the repo root or edit `$EntryPy in this script)."
}

# --------------------------- venv + deps -------------------------------------
if (-not (Test-Path -LiteralPath $VenvDir)) {
  Write-Section "Creating virtual environment ($VenvDir)"
  $pyCmd = Get-CommandPath 'py'
  if (-not $pyCmd) { $pyCmd = Get-CommandPath 'python' }
  if (-not $pyCmd) { throw "No 'py' or 'python' on PATH. Install Python 3.10+ first." }
  & $pyCmd -m venv $VenvDir
}

if (-not (Test-Path -LiteralPath $PyExe)) {
  throw "Virtual env python not found: $PyExe"
}

Write-Section "Installing/Updating dependencies"
& $PyExe -m pip install --upgrade pip setuptools wheel | Out-Null
if (Test-Path -LiteralPath 'requirements-build.txt') {
  & $PyExe -m pip install -r requirements-build.txt | Out-Null
} elseif (Test-Path -LiteralPath 'requirements.txt') {
  & $PyExe -m pip install -r requirements.txt "PyInstaller>=6.6,<7" | Out-Null
} else {
  & $PyExe -m pip install "PyInstaller>=6.6" "Pillow>=10.3" "NBT==1.5.1" | Out-Null
}

# --------------------------- sanity check ------------------------------------
Write-Section "Sanity import check (tkinter, Pillow, nbt)"
$sanityPy = @'
import sys
try:
    import tkinter, PIL, PIL.Image, PIL.ImageTk, nbt
    print("Imports OK:", tkinter.TkVersion, getattr(PIL, "__version__", "?"))
except Exception as e:
    print("Sanity import failed:", e)
    sys.exit(2)
'@
$sanityPath = Join-Path $TmpDir 'sanity.py'
Set-Content -Path $sanityPath -Value $sanityPy -Encoding UTF8 -Force
& $PyExe $sanityPath | Tee-Object -FilePath (Join-Path $LogsDir 'sanity.txt')

# --------------------------- Tcl/Tk discovery (optional) ---------------------
Write-Section "Discovering Tcl/Tk root (optional)"
$tclPy = @'
import sys, os, os.path as p
try:
    import tkinter
    cands = [p.join(sys.base_prefix, "tcl")]
    try:
        tkdir = p.dirname(tkinter.__file__)
        cands.append(p.normpath(p.join(tkdir, "..", "tcl")))
    except Exception:
        pass
    for c in cands:
        if p.exists(c):
            print(c)
            break
    else:
        print("")
except Exception:
    print("")
'@
$tclPathPy = Join-Path $TmpDir 'tclpath.py'
Set-Content -Path $tclPathPy -Value $tclPy -Encoding UTF8 -Force
$tclRoot = (& $PyExe $tclPathPy) -join ''
if ($tclRoot) { Write-Host "Tcl root: $tclRoot" -ForegroundColor Green } else { Write-Host "Tcl root not explicitly found (PyInstaller usually bundles Tk automatically)." -ForegroundColor Yellow }
$addData = @()
if ($tclRoot -and (Test-Path -LiteralPath $tclRoot)) { $addData = @('--add-data', "$tclRoot;tcl") }

# --------------------------- clean stage -------------------------------------
Write-Section "Preparing build folders"
Clean-BuildOutputs
Ensure-Folder $LogsDir
Ensure-Folder $TmpDir

# --------------------------- Sidecar: chunkmap_ui ----------------------------
$ChunkArgs = @('--hidden-import','chunkmap_ui')   # try to bundle as a real module
# Ship the file/package alongside, so robust importer in main_voider.py can find it in _MEIPASS / onedir dir.
if (Test-Path -LiteralPath 'chunkmap_ui.py') {
  $cp = (Resolve-Path -LiteralPath 'chunkmap_ui.py').Path
  $ChunkArgs += @('--add-data', "$cp;.")
  Write-Host "Will include sidecar: chunkmap_ui.py" -ForegroundColor DarkGreen
} elseif (Test-Path -LiteralPath 'chunkmap_ui' -PathType Container) {
  $cp = (Resolve-Path -LiteralPath 'chunkmap_ui').Path
  # Add entire folder to a same-named folder inside bundle
  $ChunkArgs += @('--add-data', "$cp;chunkmap_ui")
  Write-Host "Will include sidecar package folder: chunkmap_ui\\" -ForegroundColor DarkGreen
} else {
  Write-Host "Warning: chunkmap_ui not found next to the builder; relying on hidden-import only." -ForegroundColor Yellow
}

# --------------------------- PyInstaller args --------------------------------
$AddPaths = @('--paths', $Here)  # ensure local folder is on analysis path

$common = @(
  '-m','PyInstaller',
  '--noconfirm',
  '--log-level','DEBUG',
  '--collect-submodules','tkinter',
  '--hidden-import','_tkinter',
  '--collect-all','PIL',
  '--collect-all','nbt',
  '--hidden-import','PIL.Image',
  '--hidden-import','PIL.ImageTk',
  '--hidden-import','PIL.PngImagePlugin',
  '--name', $AppName
) + $AddPaths + $ChunkArgs + $addData

$iconArgs = @()
if (Test-Path -LiteralPath $IconIco) { $iconArgs = @('--icon', $IconIco) }

# --------------------------- PASS A: ONEDIR ----------------------------------
$onedirExe = $null
if (-not $DoSkipOneDir) {
  Write-Section "PASS A — ONEDIR (diagnostic)"
  $rc1 = Run-Logged -Exe $PyExe `
    -ArgList ($common + @('--onedir','--console') + $iconArgs + @($EntryPy)) `
    -StdOut $OneDirOut -StdErr $OneDirErr

  $onedirExe = Join-Path -Path ("dist\{0}" -f $AppName) -ChildPath ("{0}.exe" -f $AppName)
  if (Test-Path -LiteralPath $onedirExe) {
    Write-Host "✅ ONEDIR built: $onedirExe" -ForegroundColor Green
  } else {
    Write-Host "`n❌ ONEDIR failed (exit $rc1). Showing tails:" -ForegroundColor Yellow
    Tail-If $OneDirOut 120 'STDOUT'
    Tail-If $OneDirErr 120 'STDERR'
    throw "Stopped after ONEDIR failure."
  }
} else {
  Write-Host "Skipping ONEDIR pass (per setting)." -ForegroundColor DarkYellow
}

# --------------------------- PASS B: ONEFILE ---------------------------------
Write-Section "PASS B — ONEFILE (final)"
$wf = @('--windowed')
if ($OneFileConsole) { $wf = @('--console') }

$rc2 = Run-Logged -Exe $PyExe `
  -ArgList ($common + @('--onefile') + $wf + $iconArgs + @($EntryPy)) `
  -StdOut $OneFileOut -StdErr $OneFileErr

$onefileExe = Join-Path -Path 'dist' -ChildPath ("{0}.exe" -f $AppName)
if (Test-Path -LiteralPath $onefileExe) {
  $final = Resolve-Path -LiteralPath $onefileExe
  Write-Host "`n✅ ONEFILE built: $final" -ForegroundColor Green
} else {
  Write-Host "`n❌ ONEFILE failed (exit $rc2). Showing tails:" -ForegroundColor Yellow
  Tail-If $OneFileOut 140 'STDOUT'
  Tail-If $OneFileErr 140 'STDERR'
  if ($onedirExe) {
    Write-Host "`nYou can still use ONEDIR: $onedirExe" -ForegroundColor Yellow
  } else {
    Write-Host "`nONEDIR was not built; set `$DoSkipOneDir = `$false and re-run." -ForegroundColor Yellow
  }
}

# --------------------------- Post info ---------------------------------------
Write-Section "Done"
Write-Host "Artifacts:"
if ($onedirExe -and (Test-Path -LiteralPath $onedirExe)) { Write-Host " • ONEDIR : $onedirExe" }
if (Test-Path -LiteralPath $onefileExe) { Write-Host " • ONEFILE: $onefileExe" }
Write-Host "`nTip: To verify the EXE integrity, compute a hash:" -ForegroundColor DarkGray
if (Test-Path -LiteralPath $onefileExe) {
  Write-Host "  certutil -hashfile `"$onefileExe`" SHA256" -ForegroundColor DarkGray
}
