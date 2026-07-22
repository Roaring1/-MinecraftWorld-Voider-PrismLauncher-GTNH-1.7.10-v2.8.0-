# Changelog

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
