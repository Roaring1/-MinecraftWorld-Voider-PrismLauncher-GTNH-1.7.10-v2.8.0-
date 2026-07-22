#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
VENV="${BUILD_VENV:-$ROOT/.venv-build-linux}"

if ! "$PYTHON" -c 'import tkinter' >/dev/null 2>&1; then
  echo 'Tkinter is required. On Nobara/Fedora: sudo dnf install python3-tkinter' >&2
  exit 2
fi
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements-build.txt
PYTHONPATH=. "$VENV/bin/python" -m unittest discover -s tests -v
rm -rf build dist
COMMON=(--noconfirm --clean --name Voider-GTNH --hidden-import chunkmap_ui --collect-all PIL --collect-all nbt --add-data "chunkmap_ui.py:." --add-data "image.png:.")
"$VENV/bin/python" -m PyInstaller "${COMMON[@]}" --onedir --console main_voider.py
mv dist/Voider-GTNH dist/Voider-GTNH-linux-onedir
"$VENV/bin/python" -m PyInstaller "${COMMON[@]}" --onefile --windowed main_voider.py
(
  cd dist
  tar -czf Voider-GTNH-linux-x86_64-onedir.tar.gz Voider-GTNH-linux-onedir
)
PORTABLE="dist/GTNH-World-Voider-linux-x86_64"
rm -rf "$PORTABLE"
mkdir -p "$PORTABLE"
cp dist/Voider-GTNH "$PORTABLE/Voider-GTNH"
cp install_native_linux.sh uninstall_linux.sh image.png README.md LICENSE VERSION "$PORTABLE/"
chmod +x "$PORTABLE/Voider-GTNH" "$PORTABLE/install_native_linux.sh" "$PORTABLE/uninstall_linux.sh"
(
  cd dist
  tar -czf GTNH-World-Voider-linux-x86_64.tar.gz GTNH-World-Voider-linux-x86_64
  sha256sum Voider-GTNH Voider-GTNH-linux-x86_64-onedir.tar.gz GTNH-World-Voider-linux-x86_64.tar.gz > SHA256SUMS-linux.txt
)
echo "Linux artifacts are in $ROOT/dist"
