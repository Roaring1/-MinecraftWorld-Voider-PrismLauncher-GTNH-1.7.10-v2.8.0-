# Changelog

## Unreleased — world-safety hardening

### Safety and transaction handling
- Hold an in-process per-world mutex and the operating-system session lock across each write transaction; fail closed when the lock mechanism is unavailable.
- Verify full-world backups with an inventory, CRC checks, per-file SHA-256 manifest, and source-stability checks.
- Exclude the transient root `session.lock` from backup contents while recording the omission in the manifest, allowing Windows backups to run while the write guard is held.
- Stage biome reset across all selected region files, verify the result, and roll back committed region files if verification fails.
- Treat malformed NBT, chunk-coordinate mismatches, region-worker exceptions, and cancellation as failed/incomplete runs rather than successful skips.
- Prevent orphan TileEntities by skipping TileEntity-bearing chunks when preservation is requested instead of copying their block entities into empty terrain.
- Preserve unknown root and Level metadata tags, including GTNH's POSTEA and V, when rebuilding chunks.

### GTNH / Crucible compatibility
- Recognize GTNH's 512-byte little-endian Biomes16v2 array and 16-bit biome IDs in rules and the chunk selector.
- Preserve Biomes16v2 during voiding when present; remove it during provider-safe non-Overworld biome reset.
- Resolve dedicated-server roots through server.properties / level-name and discover Linux CrucibleServers layouts.
- Add regressions for extended IDs, metadata preservation, rollback, lock reentrancy, backup manifests, and server-root path traversal.
- Fail closed on selection files with no dimension scopes; selection runs are limited to named dimensions, and explicit `--all-chunks` overrides saved selections.
- Keep headless CLI failures non-interactive so a Tk modal cannot stall a command-line run or CI; use a writable file handle for the Windows durability flush.

## 2.1.0 — 2026-07-21

### Added
- Named `manual`, `safe-void`, `full-void`, and `preserve-data` processing presets in the CLI and GUI.
- GUI write runs now perform a full dry-run preflight and show the exact candidate chunk count before confirmation.
- Post-write region verification is enabled by default and checks sector allocation, overlap, NBT readability, and chunk coordinates.
- Headless JSON reports include the selected preset and complete post-write verification result.
- Added read-only `--inspect-world` inventory with client/server classification, dimensions, region size, and lock status.
- Added operating-system `session.lock` detection and refusal to write while a client or dedicated server holds the world open.
- Added a native Linux installer and a Nobara/Fedora diagnostic script.

### Maintenance
- Added regression coverage for presets, valid/invalid world verification, dedicated-server dimension layouts, held server locks, and native Linux installation.
- Removed safe unused imports, a bare exception, and other correctness-oriented lint findings without reformatting legacy code wholesale.
- Fixed the Linux release workflow to build with Ubuntu's matched Python/Tk packages instead of a Tk-less setup-python interpreter.
- Added strict release artifact existence checks and a self-contained native Linux bundle.

## 2.0.0 — r12 hardening release

### Correctness
- Fixed decompressed NBT parsing throughout the voider, biome reset, rules engine, entity checks, and chunk selector.
- Fixed legacy Anvil section indexing and platform metadata/extended-ID output.
- Fixed multi-chunk platform placement and unaligned region-sector growth.
- Made persisted rules effective during GUI and CLI runs.

### Safety
- Dry run is now the default; writing requires `--write` or unchecking Dry run in the GUI.
- Verified world ZIP backup is now the default before writes; `--no-backup` is explicit and unsafe.
- Region edits are copy-on-write and atomically replace originals only after success.
- Concurrent region-file changes abort replacement.
- Selection JSON is strictly validated; headless runs require a selection or explicit `--all-chunks`.
- Backup names are unique, ZIP-tested, and SHA-256 reported.

### Linux / Nobara
- Added focused Prism Launcher discovery, native run/install/build scripts, and desktop integration.
- Added headless CLI operation that does not import Tk.

### Engineering
- Added real-NBT synthetic Anvil regression tests.
- Added Linux and Windows CI/release workflows.
- Split runtime and build dependencies.
