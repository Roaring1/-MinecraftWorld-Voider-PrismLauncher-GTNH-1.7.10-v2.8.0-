#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="${GTNH_VOIDER_VENV:-$ROOT/.venv-linux}"

if ! command -v "$PYTHON" >/dev/null; then
  echo "Python 3 is required." >&2; exit 2
fi
NEEDS_GUI=1
for arg in "$@"; do
  [[ "$arg" == "--headless" || "$arg" == "--verify-world" || "$arg" == "--inspect-world" ]] && NEEDS_GUI=0
done
if [[ "$NEEDS_GUI" == 1 ]] && ! "$PYTHON" -c 'import tkinter' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Tkinter is missing. On Nobara/Fedora run:
  sudo dnf install python3-tkinter
For headless mode, invoke main_voider.py --headless directly after installing runtime requirements.
EOF
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -r "$ROOT/requirements-runtime.txt"
fi
exec "$VENV/bin/python" "$ROOT/main_voider.py" "$@"
