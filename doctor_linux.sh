#!/usr/bin/env bash
set -euo pipefail

APP_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/gtnh-world-voider"
BIN="$HOME/.local/bin/gtnh-world-voider"
fail=0

check() {
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then
    printf 'PASS  %s\n' "$label"
  else
    printf 'FAIL  %s\n' "$label"
    fail=1
  fi
}

printf 'GTNH World Voider — Nobara/Fedora doctor\n'
printf 'System: %s\n' "$(uname -a)"
check 'Python 3' command -v python3
check 'Tkinter GUI support' python3 -c 'import tkinter'
check 'Installed application directory' test -d "$APP_HOME"
check 'CLI launcher' test -x "$BIN"
check 'Runtime virtual environment' test -x "$APP_HOME/.venv-linux/bin/python"
if [[ -x "$APP_HOME/.venv-linux/bin/python" ]]; then
  check 'NBT runtime dependency' "$APP_HOME/.venv-linux/bin/python" -c 'from nbt import nbt'
  check 'Pillow runtime dependency' "$APP_HOME/.venv-linux/bin/python" -c 'import PIL'
  check 'Application import' env PYTHONPATH="$APP_HOME" "$APP_HOME/.venv-linux/bin/python" -c 'import main_voider'
fi

printf '\nKnown client save roots:\n'
printf '  %s\n' "$HOME/.minecraft/saves"
printf '  %s\n' "$HOME/.local/share/PrismLauncher/instances"
printf '  %s\n' "$HOME/.var/app/org.prismlauncher.PrismLauncher/data/PrismLauncher/instances"
printf '\nDedicated servers are supported by selecting the world directory containing level.dat.\n'

exit "$fail"
