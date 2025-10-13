# GTNH Voider — Builder (hardened, PS-safe; no here-strings; temp .py helpers)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---- settings ---------------------------------------------------------------
$AppName   = 'Voider-GTNH'
$EntryPy   = 'main_voider.py'
$IconIco   = 'voider.ico'      # optional
$VenvDir   = '.venv'
$LogsDir   = 'build-logs'
$TmpDir    = '.build_tmp'

$PyExe     = Join-Path $VenvDir 'Scripts\python.exe'
$OneDirOut = Join-Path $LogsDir 'onedir-out.txt'
$OneDirErr = Join-Path $LogsDir 'onedir-err.txt'
$OneFileOut= Join-Path $LogsDir 'onefile-out.txt'
$OneFileErr= Join-Path $LogsDir 'onefile-err.txt'

function Ensure-Folder { param([string]$Path) if (-not (Test-Path -LiteralPath $Path)) { New-Item -Type Directory -Force -Path $Path | Out-Null } }
function Clean-BuildOutputs {
  if (Test-Path -LiteralPath 'build') { Remove-Item -Recurse -Force 'build' -ErrorAction SilentlyContinue }
  if (Test-Path -LiteralPath 'dist')  { Remove-Item -Recurse -Force 'dist'  -ErrorAction SilentlyContinue }
  Get-ChildItem -Filter '*.spec' -ErrorAction SilentlyContinue | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue }
  Ensure-Folder $LogsDir
  Ensure-Folder $TmpDir
}

# --- make sure staging dirs exist BEFORE writing helper scripts --------------
Ensure-Folder $LogsDir
Ensure-Folder $TmpDir

# ---- locate/create venv + deps ---------------------------------------------
function Get-PyLauncher {
  $cands = @('py', 'py.exe')
  foreach ($c in $cands) {
    $p = (Get-Command $c -ErrorAction SilentlyContinue)
    if ($p) { return $p.Path }
  }
  return $null
}
function Get-SystemPython {
  $p = (Get-Command python -ErrorAction SilentlyContinue)
  if ($p) { return $p.Path }
  return $null
}

if (-not (Test-Path -LiteralPath $VenvDir)) {
  $pyLauncher = Get-PyLauncher
  $sysPy = Get-SystemPython
  if ($pyLauncher) {
    & $pyLauncher -3 -m venv $VenvDir
  } elseif ($sysPy) {
    & $sysPy -m venv $VenvDir
  } else {
    throw "No Python found. Install Python 3 (with the Windows launcher) first."
  }
}
if (-not (Test-Path -LiteralPath $PyExe)) {
  throw "Virtual env looks broken: $PyExe not found."
}

# Upgrade tooling + deps
& $PyExe -m pip install --upgrade pip setuptools wheel | Out-Null
& $PyExe -m pip install --upgrade PyInstaller Pillow NBT | Out-Null

# ---- helper: sanity imports -------------------------------------------------
$sanityLines = @(
  'import sys',
  'try:',
  '    import tkinter, PIL, PIL.Image, PIL.ImageTk, nbt',
  '    print("Imports OK:", tkinter.TkVersion, getattr(PIL, "__version__", "?"))',
  'except Exception as e:',
  '    print("Sanity import failed:", e)',
  '    sys.exit(2)'
)
$sanityPath = Join-Path $TmpDir 'sanity.py'
$sanityLines | Set-Content -Encoding UTF8 -NoNewline $sanityPath
& $PyExe $sanityPath | Out-Host

# ---- helper: discover tcl root (tkinter data) ------------------------------
$tclLines = @(
  'import sys, os, os.path as p',
  'try:',
  '    import tkinter',
  '    base = sys.base_prefix',
  '    tcl  = p.join(base, "tcl")',
  '    print(tcl if p.exists(tcl) else "")',
  'except Exception:',
  '    print("")'
)
$tclPathPy = Join-Path $TmpDir 'tclpath.py'
$tclLines | Set-Content -Encoding UTF8 -NoNewline $tclPathPy
$tclRoot = (& $PyExe $tclPathPy) -join ''   # flatten any CRLFs
$addData = @()
if ($tclRoot -and (Test-Path -LiteralPath $tclRoot)) {
  # Use PyInstaller's SOURCE:DEST syntax; put whole tcl dir under top-level 'tcl'
  $addData = @('--add-data', ("{0}:{1}" -f $tclRoot, 'tcl'))
}

# ---- clean staging (after helpers exist) -----------------------------------
Clean-BuildOutputs
Ensure-Folder $LogsDir
Ensure-Folder $TmpDir

function Run-Logged {
  param([string]$Exe,[string[]]$ArgList,[string]$StdOut,[string]$StdErr)
  $ArgList = @($ArgList | Where-Object { $_ -ne $null -and $_ -ne '' })
  Write-Host ("`n>>> {0} {1}" -f $Exe, ($ArgList -join ' ')) -ForegroundColor Cyan
  $p = Start-Process -FilePath $Exe -ArgumentList $ArgList `
       -NoNewWindow -RedirectStandardOutput $StdOut -RedirectStandardError $StdErr -PassThru
  $p.WaitForExit(); return $p.ExitCode
}

# ---- common pyinstaller args -----------------------------------------------
$common = @(
  '-m','PyInstaller',
  '--noconfirm',
  '--clean',
  '--log-level','DEBUG',
  '--name',$AppName,
  '--collect-submodules','tkinter',
  '--hidden-import','_tkinter',
  '--collect-all','PIL',
  '--collect-all','nbt',
  '--hidden-import','PIL.Image',
  '--hidden-import','PIL.ImageTk',
  '--hidden-import','PIL.PngImagePlugin'
) + $addData

$iconArgs = @()
if (Test-Path -LiteralPath $IconIco) { $iconArgs = @('--icon', $IconIco) }

# ---- PASS A: ONEDIR + console (diagnostic) ---------------------------------
$rc1 = Run-Logged -Exe $PyExe `
       -ArgList ($common + @('--onedir','--console') + $iconArgs + @($EntryPy)) `
       -StdOut $OneDirOut -StdErr $OneDirErr

$onedirExe = Join-Path -Path 'dist' -ChildPath (Join-Path $AppName ($AppName + '.exe'))
if (-not (Test-Path -LiteralPath $onedirExe)) {
  Write-Host "`n❌ ONEDIR failed (exit $rc1). Tails:" -ForegroundColor Yellow
  if (Test-Path $OneDirOut) { Write-Host "---- STDOUT ----" -ForegroundColor DarkCyan; Get-Content $OneDirOut -Tail 120 }
  if (Test-Path $OneDirErr) { Write-Host "---- STDERR ----" -ForegroundColor Magenta ; Get-Content $OneDirErr -Tail 120 }
  throw "Stopped after ONEDIR failure."
}
Write-Host "✅ ONEDIR built: $onedirExe" -ForegroundColor Green

# ---- PASS B: ONEFILE + windowed (final) ------------------------------------
$rc2 = Run-Logged -Exe $PyExe `
       -ArgList ($common + @('--onefile','--windowed','--disable-windowed-traceback') + $iconArgs + @($EntryPy)) `
       -StdOut $OneFileOut -StdErr $OneFileErr

$onefileExe = Join-Path -Path 'dist' -ChildPath ($AppName + '.exe')
if (Test-Path -LiteralPath $onefileExe) {
  $final = Resolve-Path $onefileExe
  Write-Host "`n✅ ONEFILE built: $final" -ForegroundColor Green
} else {
  Write-Host "`n❌ ONEFILE failed (exit $rc2). Tails:" -ForegroundColor Yellow
  if (Test-Path $OneFileOut) { Write-Host "---- STDOUT ----" -ForegroundColor DarkCyan; Get-Content $OneFileOut -Tail 140 }
  if (Test-Path $OneFileErr) { Write-Host "---- STDERR ----" -ForegroundColor Magenta ; Get-Content $OneFileErr -Tail 140 }
  Write-Host "`nUse ONEDIR for now: $onedirExe" -ForegroundColor Yellow
}
