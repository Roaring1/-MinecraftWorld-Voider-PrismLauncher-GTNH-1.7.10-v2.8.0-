#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VERSION="$(tr -d '[:space:]' < VERSION)"
OUT="$ROOT/release-out/v$VERSION"
rm -rf "$OUT"
mkdir -p "$OUT/stage/GTNH-World-Voider-$VERSION"

PYTHON="${PYTHON:-python3}"
"$PYTHON" -m py_compile main_voider.py chunkmap_ui.py
PYTHONPATH=. "$PYTHON" -m unittest discover -s tests -v

tar -cf - --exclude='./.git' --exclude='./.venv*' --exclude='*/__pycache__' \
  --exclude='./build' --exclude='./dist' --exclude='./release-out' . \
  | tar -xf - -C "$OUT/stage/GTNH-World-Voider-$VERSION"
(
  cd "$OUT/stage"
  tar -czf "$OUT/GTNH-World-Voider-$VERSION-source.tar.gz" "GTNH-World-Voider-$VERSION"
  zip -qr "$OUT/GTNH-World-Voider-$VERSION-source.zip" "GTNH-World-Voider-$VERSION"
)
rm -rf "$OUT/stage"
(
  cd "$OUT"
  sha256sum ./* > SHA256SUMS.txt
)
echo "Prepared release files in $OUT"
cat "$OUT/SHA256SUMS.txt"
