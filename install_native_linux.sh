#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_BIN="${1:-$SRC/Voider-GTNH}"
APP_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/gtnh-world-voider"
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
BIN_DIR="$HOME/.local/bin"

if [[ ! -x "$SOURCE_BIN" ]]; then
  echo "Native executable not found: $SOURCE_BIN" >&2
  echo "Run this script from the extracted Linux release bundle." >&2
  exit 2
fi
mkdir -p "$APP_HOME" "$APP_DIR" "$BIN_DIR"
install -m 0755 "$SOURCE_BIN" "$APP_HOME/Voider-GTNH"
[[ -f "$SRC/image.png" ]] && install -m 0644 "$SRC/image.png" "$APP_HOME/image.png"
ln -sfn "$APP_HOME/Voider-GTNH" "$BIN_DIR/gtnh-world-voider"
cat > "$APP_DIR/gtnh-world-voider.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=GTNH World Voider
Comment=Safety-first Minecraft 1.7.x world voiding
Exec=$APP_HOME/Voider-GTNH
Icon=$APP_HOME/image.png
Terminal=false
Categories=Game;Utility;
StartupNotify=true
EOF
chmod 0644 "$APP_DIR/gtnh-world-voider.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$APP_DIR" || true
printf 'Installed native GTNH World Voider to %s\n' "$APP_HOME"
printf 'Launch from the app menu or run: %s\n' "$BIN_DIR/gtnh-world-voider"
