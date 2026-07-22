#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/gtnh-world-voider"
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
BIN_DIR="$HOME/.local/bin"

if ! python3 -c 'import tkinter' >/dev/null 2>&1; then
  echo 'Tkinter is required. Run: sudo dnf install python3-tkinter' >&2
  exit 2
fi
mkdir -p "$APP_HOME" "$APP_DIR" "$BIN_DIR"
rsync -a --delete \
  --exclude '.git' --exclude '.venv*' --exclude 'build' --exclude 'dist' \
  "$SRC/" "$APP_HOME/"
python3 -m venv "$APP_HOME/.venv-linux"
"$APP_HOME/.venv-linux/bin/python" -m pip install --upgrade pip
"$APP_HOME/.venv-linux/bin/python" -m pip install -r "$APP_HOME/requirements-runtime.txt"
ln -sfn "$APP_HOME/run_voider.sh" "$BIN_DIR/gtnh-world-voider"
cat > "$APP_DIR/gtnh-world-voider.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=GTNH World Voider
Comment=Provider-safe chunk voiding for Minecraft 1.7.x / GTNH
Exec=$APP_HOME/run_voider.sh
Icon=$APP_HOME/image.png
Terminal=false
Categories=Game;Utility;
StartupNotify=true
EOF
chmod +x "$APP_HOME/run_voider.sh" "$BIN_DIR/gtnh-world-voider"
command -v update-desktop-database >/dev/null && update-desktop-database "$APP_DIR" || true
echo "Installed GTNH World Voider to $APP_HOME"
echo "CLI launcher: $BIN_DIR/gtnh-world-voider"
