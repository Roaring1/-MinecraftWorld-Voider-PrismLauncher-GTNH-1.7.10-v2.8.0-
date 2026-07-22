#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$ROOT"
"$PYTHON" -m py_compile main_voider.py chunkmap_ui.py
PYTHONPATH=. "$PYTHON" -m unittest discover -s tests -v
