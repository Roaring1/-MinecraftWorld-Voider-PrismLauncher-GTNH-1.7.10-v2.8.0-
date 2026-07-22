#!/usr/bin/env bash
set -euo pipefail
APP_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/gtnh-world-voider"
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
rm -rf -- "$APP_HOME"
rm -f -- "$HOME/.local/bin/gtnh-world-voider" "$APP_DIR/gtnh-world-voider.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$APP_DIR" || true
echo "GTNH World Voider removed."
