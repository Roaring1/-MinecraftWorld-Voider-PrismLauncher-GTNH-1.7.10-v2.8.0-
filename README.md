# Minecraft World Voider (GTNH 1.7.10)

Re-runnable tool that turns **already-generated** chunks into air (biomes kept).
Designed for 1.7.10 Anvil worlds (classic Sections: Blocks/Data/SkyLight/BlockLight + 256-byte Biomes).

---

## Requirements (Windows)

* **Python 3** installed (includes Tk/tkinter GUI support).

---

## Install

Open **PowerShell** and run:

```bat
py -3 -m pip install --upgrade NBT
```

---

## Run — GUI (recommended)

1. **Close** Minecraft/Prism/MCEdit.
2. **Double-click** `void_rerunnable_1710_pro.py`.
3. Click **Browse…** → select your world folder (must contain `level.dat`).
4. Click **Scan Dims** → **Run**.

> If the `NBT` package is missing, a window will tell you exactly what to install.

---

## Run — CLI (optional)

```bat
py -3 ".\void_rerunnable_1710_pro.py" "C:\Path\To\World" --backup --workers 4 --wait-for-unlock 20
```

Common flags:

* `--backup` : make a zip of the world first
* `--workers N` : process region files in parallel
* `--wait-for-unlock S` : wait up to S seconds for `session.lock` to clear
* Area filters: `--bbox minX,minZ,maxX,maxZ` or `--center-x X --center-z Z --radius-blocks R`

---

## Notes

* This **voids existing chunks** only. New, unexplored chunks will still generate normally.
* Re-run the tool any time to void **newly generated** areas.
* If you see `session.lock`, the world is open—close the game or use `--wait-for-unlock`.

---

## Optional: Build a one-file EXE

```bat
py -3 -m pip install pyinstaller
pyinstaller --onefile --windowed void_rerunnable_1710_pro.py
```
