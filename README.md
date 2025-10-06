
# README.md

````markdown
# Minecraft World Voider (GTNH 1.7.10)
# I made this for Gregtech: New horizons, v2.8.0.

**What it does:** re-runnable tool that turns already-generated chunks into air-only while **keeping biomes**. Works on classic Anvil chunks (1.7.10 era). New, unexplored areas will still generate normally.

---

## Option A — Download & run (Windows, recommended)
1) Download the latest `MinecraftWorldVoider-1.7.10.exe` from **Releases**.  
2) **Close** Minecraft / Prism / MCEdit.  
3) Double-click the EXE → pick your world → **Run**.  
   * Default makes a zip backup and waits for `session.lock`.  
   * Progress + final count will show in the log.

---

## Option B — Run from source (Python)
1) Install Python 3.  
2) In a terminal (PowerShell):  
   ```powershell
   py -3 -m pip install -r requirements.txt
   py void_rerunnable_1710_pro.py
````

The GUI opens; pick your world and **Run**.

The script uses the **NBT** library to read/write classic chunk data. 

---

## Build your own EXE (optional)

```powershell
py -3 -m pip install pyinstaller
pyinstaller --onefile --windowed --noconfirm --name "MinecraftWorldVoider-1.7.10" void_rerunnable_1710_pro.py
```

(General usage & spec-file behavior: PyInstaller docs.) 

---

## Notes

* **Safety:** close the game/launchers first. The app waits for `session.lock` and will warn before forcing.
* **Biomes stay.** Sections are rewritten to air (Blocks/Data/SkyLight/BlockLight), preserving the 256-byte `Biomes` array.
* **Re-runnable:** by default it skips chunks that are already empty; enable “Rewrite already-void chunks” if you want to refresh a spawn platform, etc.
* **Area filters:** you can target a radius or bbox; dimensions are selectable.
* **Theme:** the GUI follows Windows light/dark and uses the system accent color (via `DwmGetColorizationColor`).
* **Readability:** control/field colors are chosen to meet WCAG AA contrast (≥4.5:1 for text, ≥3:1 for UI). 

---

## Troubleshooting

* **“World appears open”** → make sure Minecraft/Prism/MCEdit are closed; wait or “Force proceed” if you’re sure.
* **“Access is denied” when rebuilding EXE** → close the old EXE (and any antivirus scan), then rebuild.
* **“No module named NBT”** (source mode) → `py -3 -m pip install NBT`. 

