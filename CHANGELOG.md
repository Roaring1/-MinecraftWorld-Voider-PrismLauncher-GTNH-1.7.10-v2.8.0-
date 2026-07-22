# Changelog

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
