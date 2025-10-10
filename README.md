# GTNH-Voider — Minecraft World Voider (GTNH 1.7.10)
# I made this for GregTech: New Horizons, v2.8.0.

**What it does:** Re-runnable tool that turns already-generated chunks into air-only while **keeping biomes**. Works on classic Anvil chunks (1.7.10 era). New, unexplored areas will still generate normally.

**UI / performance (this release):** Python Tkinter with **LOD tiling**, **batch drawing**, and **throttling**. This felt adequate in testing, but **large maps can feel laggy**.

---

## Option A — Download & run (Windows, recommended)

1) Download the latest `GTNH-Voider.exe` from **Releases**.  
2) **Close** Minecraft / Prism / MCEdit.  
3) Double-click the EXE. A **Status & Controls** window opens first.  
4) Click **Discover…** (auto-scan) or **Browse…** and choose your world (folder with `level.dat`).  
5) Click **Chunk Selector…**  
   * Pick a **dimension** (Overworld / Nether / etc.).  
   * **LMB** to include chunks (drag for box). Switch to **Exclude** to mark chunks you want to keep.  
   * Toggle **JourneyMap** imagery (day/topo/night/auto), adjust **Overlay α**, grid, axis, “selected only”, and **Snap 32×32** if you want region-aligned boxes.  
   * Click **Save & Close** — your selection is written to `{world}/.voider-selection.json`.  
   * Per-dimension view settings are saved to `{world}/.voider-ui.json`.  
6) Back in **Status & Controls**, review options (e.g., Dry run, Include already void, Keep/Skip Entities/TileEntities, Workers).  
7) Click **Start Voiding**. Progress and per-file results stream into the **Log**. Use **Save Log…** to export.

> **Defaults/behavior:**  
> • **Biomes are preserved**.  
> • **Blocks become air** in every 16×16×256 chunk section you included.  
> • **Entities/TileEntities are removed by default** (enable **Keep Entities / Keep TileEntities** to carry them over).  
> • Already-empty chunks are **skipped** unless **Include already void** is enabled.

---

## Option B — Run from source (Python)

> Minimal, transparent steps. Replace the path example with your actual folder.

1) In **PowerShell**, go to the project folder:  
```powershell
cd "<path to this folder>"    # e.g. C:\Users\You\Downloads\GTNH-Voider
```

2) Create an isolated environment (one-time):  
```powershell
py -3.10 -m venv .venv
```

3) Install the required packages (one-time):  
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
.\.venv\Scripts\python.exe -m pip install pillow nbt
```

4) Run the app:  
```powershell
.\.venv\Scripts\python.exe .\main_voider.py
```

*The GUI opens first (same flow as Option A). The **Chunk Selector** requires `chunkmap_ui.py` in the same folder.*

---

## Build your own EXE (optional)

> In PowerShell, the backtick `` ` `` means “command continues on the next line.”

```powershell
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onefile --windowed `
  --name GTNH-Voider `
  --hidden-import chunkmap_ui `
  --hidden-import PIL.Image --hidden-import PIL.ImageTk `
  --hidden-import PIL.PngImagePlugin --hidden-import PIL._tkinter_finder `
  .\main_voider.py
```

**Why the hidden imports?** Pillow/Tkinter bits are loaded dynamically; listing them ensures they’re bundled, so the GUI and imagery work in the EXE.  
**Where’s the EXE?** `dist\GTNH-Voider.exe`

---

## Advanced (CLI flags the GUI maps to)

*Filtering & scope:*  
- `--only region,DIM-1/region` · `--exclude DIM7/region`  
- `--bbox minX,minZ,maxX,maxZ`  
- `--center-x N --center-z N --radius-blocks R`

*Safety & backups:*  
- `--backup` (ZIP the world once before changes)  
- `--per-file-backup` (write `.bak` next to each `.mca`)  
- `--wait-for-unlock N` (wait up to N seconds if `session.lock` exists)

*Entities & control:*  
- `--keep-entities` · `--keep-tileentities`  
- `--skip-if-entities` · `--skip-if-tileentities`  
- `--include-void` (rewrite already-empty chunks)  
- `--mark-void` (add a tiny “Voider” tag to chunk NBT)  
- `--workers N` (parallel region processing)  
- `--dry-run` (report only; no file changes)

*Spawn platform (optional, CLI only):*  
- `--platform-x X --platform-z Z --platform-y 64 --platform-size 3 --platform-id 7 --platform-meta 0`  
  Places a small odd-sized platform at (X,Z,Y) in the target chunk when it’s being voided.

---

## Notes

* **World discovery:** “Discover…” scans common launcher locations (Vanilla `.minecraft`, Prism/MultiMC, ATLauncher, Technic, CurseForge) and caches the list in `%LOCALAPPDATA%\GTNHVoider\worlds-cache.json`.  
* **Selection & UI settings:** persisted in the world folder as `.voider-selection.json` and `.voider-ui.json`.  
* **JourneyMap overlay:** day/topo/night/auto layers; per-chunk or mosaic tiles are supported; imagery is resized with Pillow for clean overlays.  
* **Performance:** LOD tiling skips heavy imagery when zoomed out; the minimap decimates very large sets of present chunks.  
* **Theme:** dark UI with readable contrast; system fonts if available.  

---

## Troubleshooting

* **“.venv\Scripts\python.exe not found”**  
  Find where `Scripts` actually lives, then use that path:  
  ```powershell
  Get-ChildItem -Recurse -Directory -Filter "Scripts" | Select-Object FullName
  ```

* **Path has spaces?**  
  Quote the path and use the PowerShell **call operator** `&`:  
  ```powershell
  & "D:\Some Folder\.venv\Scripts\python.exe" -m PyInstaller ...
  ```

* **“World appears open”**  
  Close Minecraft/launchers. If needed, run with `--wait-for-unlock N` to wait for `session.lock`.

* **“No module named nbt / Pillow” (source mode)**  
  ```powershell
  .\.venv\Scripts\python.exe -m pip install nbt pillow
  ```

* **“Access is denied” when rebuilding EXE**  
  Close the old EXE (and any antivirus scan), then rebuild.

---
