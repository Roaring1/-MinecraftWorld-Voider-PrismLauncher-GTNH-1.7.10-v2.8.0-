# GTNH World Voider 2.0.0

A safety-first chunk voiding and biome-maintenance tool for Minecraft 1.7.x worlds, especially GregTech: New Horizons 2.8.x instances managed by Prism Launcher.

## Safety model

- **Dry run is the default.** Writing requires `--write` or unchecking **Dry run** in the GUI.
- **Verified full-world ZIP backup is the default before writes.** Use `--no-backup` only if you deliberately accept the risk.
- Region edits use copy-on-write and atomic replacement.
- A region that changes during processing is rejected rather than overwritten.
- Headless mode refuses an implicit whole-world operation: provide a selection or `--all-chunks`.
- Selection JSON is schema- and type-validated.

Always close Minecraft before writing. `session.lock` normally remains after shutdown, so its presence alone is not a reliable activity signal.

## Features

- Provider-safe biome behavior for Overworld and modded dimensions.
- Visual include/exclude chunk selector with JourneyMap overlays.
- Rules by dimension for blocks, entities, and biome IDs.
- Optional entity/tile-entity preservation or safety skips.
- Spawn platforms with correct 1.7 Anvil indexing, metadata, extended block IDs, and cross-chunk placement.
- GUI and validated headless CLI.
- Windows and Linux/Nobara packaging.

## Linux / Nobara

### Run from the checkout

```bash
sudo dnf install python3 python3-pip python3-tkinter
chmod +x run_voider.sh
./run_voider.sh
```

The launcher creates `.venv-linux` and installs only Pillow and NBT.

### Install for the current user

```bash
sudo dnf install python3 python3-pip python3-tkinter rsync desktop-file-utils
chmod +x install_linux.sh
./install_linux.sh
```

This installs a desktop entry and `~/.local/bin/gtnh-world-voider`.

### Build native Linux bundles

```bash
sudo dnf install python3 python3-pip python3-tkinter binutils
chmod +x build_linux.sh
./build_linux.sh
```

Artifacts appear in `dist/` as one-file and onedir builds plus SHA-256 sums.

## Windows

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-build.txt
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\build_voider.ps1
```

## GUI workflow

1. Launch the tool and select a world containing `level.dat`.
2. Open **Chunk Selector** and save include/exclude marks.
3. Keep **Dry run** enabled and review the count.
4. Close Minecraft.
5. Disable **Dry run** only when the preview is correct.
6. Keep the default verified backup enabled.

## Headless workflow

Preview a saved selection:

```bash
python3 main_voider.py '/path/to/world' --headless --selection selection.json
```

Write after reviewing the preview:

```bash
python3 main_voider.py '/path/to/world' --headless --selection selection.json --write
```

Explicitly authorize all chunks after filters:

```bash
python3 main_voider.py '/path/to/world' --headless --all-chunks --only DIM-1/region --dry-run
```

The default selection path is `<world>/.voider-selection.json` when present.

## Read-only verification

```bash
python3 main_voider.py '/path/to/world' --verify-world --report world-audit.json
```

This checks sector ranges and overlap, NBT readability, and stored chunk coordinates without opening Tk or modifying the world.

## Tests

```bash
python3 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-runtime.txt
PYTHON=.venv-test/bin/python ./test.sh
```

The regression suite creates synthetic real-NBT Anvil regions and tests parsing, platform encoding, rules, atomic replacement, backups, and malformed selections.

## Release

See `PUBLISH_COMMANDS.md`. `prepare_release.sh` creates source/Linux-source archives and hashes without publishing anything.

## Dependencies

- Python 3.10+
- Tkinter for GUI use
- Pillow 10.3+
- NBT 1.5.1
- PyInstaller 6.6+ only for binary builds

## License

Copyright © 2026 Roaring1. All rights reserved. See `LICENSE`.
