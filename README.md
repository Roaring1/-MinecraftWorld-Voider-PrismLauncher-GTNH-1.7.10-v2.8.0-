# GTNH World Voider 2.1.0

A safety-first chunk voiding and biome-maintenance tool for Minecraft 1.7.x worlds, especially GregTech: New Horizons 2.8.x instances managed by Prism Launcher.

## Safety model

- **Dry run is the default.** Writing requires `--write` or unchecking **Dry run** in the GUI.
- **A complete, verified world ZIP is the default before writes.** The archive includes a per-file SHA-256 manifest, CRC verification, and a stable world-inventory check. Use `--no-backup` only when you deliberately accept the recovery risk.
- The tool holds a per-world in-process mutex and the operating-system lock on `session.lock` for the entire backup/write transaction. If safe locking cannot be established, writing fails closed.
- Region edits are made in temporary copies and atomically replaced only after the chunk pass succeeds and the source file's device, inode, size, and modification time still match.
- Malformed chunk NBT, coordinate mismatches, and region-worker failures abort the run instead of being reported as successful skips. Cancellation prevents a working region copy from being committed; earlier region commits remain recoverable from the verified ZIP.
- Biome reset uses the same lock and verified backup, stages all selected region files before replacing any original, verifies the result, and restores committed originals if verification fails.
- Headless mode refuses an implicit whole-world operation: provide a non-empty selection or explicitly pass `--all-chunks`.
- A selection authorizes only the dimension tags it names. Dimensions omitted from the selection are not scanned. An empty or invalid selection fails closed.
- `--all-chunks` explicitly overrides a saved selection and authorizes every discovered dimension (still subject to `--only`, `--exclude`, bounds, and rules).
- The GUI also requires a valid saved chunk selection before starting; it will not silently interpret a missing or malformed file as “all chunks.”
- Selection JSON is schema- and type-validated.

Always close Minecraft or stop the dedicated server before writing. The held session lock is an additional coordination barrier, not permission to edit an active world. Do not use `--no-backup` for valuable worlds.

## Features

- Provider-safe biome behavior for Overworld and modded dimensions, including GTNH's 16-bit `Biomes16v2` arrays.
- Biome rules and the visual selector understand legacy 8-bit biome IDs and GTNH 16-bit IDs (0–65535).
- Unknown root and `Level`-level mod-owned chunk metadata is retained when a chunk is rewritten; terrain-dependent arrays, tile entities, and scheduled block ticks are rebuilt or safely skipped.
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

Run the Nobara/Fedora diagnostic afterward:

```bash
~/.local/share/gtnh-world-voider/doctor_linux.sh
```

The native Linux release bundle needs no Python installation. Extract it and run:

```bash
./install_native_linux.sh
```

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

A selection file scopes work to the dimension keys it contains. For example, a file containing only `region` will never process `DIM-1/region` or modded dimensions. An empty selection is rejected. To deliberately include every discovered dimension, use `--all-chunks`; that flag overrides any saved selection file.

## Client and dedicated-server worlds

Both client saves and dedicated-server worlds use the same Minecraft 1.7 Anvil region format and are supported. Select the directory containing `level.dat`—for a typical server this is `server/world`, not necessarily the directory containing the server JAR. You may also select the dedicated-server instance root: the tool reads `server.properties`, resolves `level-name`, and rejects paths that escape the instance directory.

The tool discovers and processes the Overworld `region/` directory and nested dimensions such as `DIM-1/region`, `DIM1/region`, and modded dimension folders. Inspect a world without changing it:

```bash
gtnh-world-voider '/path/to/server/world' --inspect-world --report inventory.json
```

On Linux, world discovery also scans `~/CrucibleServers` and resolves each Crucible server's `server.properties` / `level-name` layout. The selected working path shown in the tool is the actual world folder, not the server root.

## Read-only verification

```bash
python3 main_voider.py '/path/to/world' --verify-world --report world-audit.json
```

This checks sector ranges and overlap, NBT readability, and stored chunk coordinates without opening Tk or modifying the world.

## Named presets and write verification

Use `--preset` to apply a reviewed group of processing options. A void operation intentionally removes terrain and block entities; the preset only controls entity records and whether TileEntity-bearing chunks are skipped. Mod-owned root and `Level` tags are preserved when safe, but this is not a general migration tool for arbitrary mod state.

- `safe-void`: skip chunks containing entities or tile entities, and mark rewritten chunks.
- `full-void`: void eligible chunks even when they contain entities or tile entities.
- `preserve-data`: preserve entity records and leave any chunk containing TileEntities untouched. TileEntities are never copied without their backing blocks.
- `manual`: use individual flags exactly as supplied (default).

```bash
python3 main_voider.py '/path/to/world' --headless --all-chunks \
  --preset safe-void --dry-run --report preview.json
```

GUI write runs always perform a dry-run preflight and ask for confirmation with the candidate count. After a write, the world verifier runs automatically. Headless users can disable only the latter with `--no-post-verify`; doing so is not recommended.

## Tests

```bash
python3 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-runtime.txt
PYTHON=.venv-test/bin/python ./test.sh
```

The regression suite creates synthetic real-NBT Anvil regions and tests client/server layouts, cross-process and same-process write locks, transactional biome reset and rollback, SHA-256 backup manifests, mod-tag retention, GTNH `Biomes16v2` handling, native Linux installation, parsing, platform encoding, rules, atomic replacement, verification, and malformed selections. The suite does not replace validation against a freshly generated GTNH server world.

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
