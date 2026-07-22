# Publish 2.0.0 to GitHub

Run these commands later from a clean clone of the repository. They do not expose tokens in shell history; authenticate with `gh auth login` if needed.

## 1. Copy in and review this release

```bash
cd /path/to/-MinecraftWorld-Voider-PrismLauncher-GTNH-1.7.10-v2.8.0-
git status --short
git switch main
git pull --ff-only origin main
```

Copy the delivered `code/` contents into the clone, then:

```bash
chmod +x run_voider.sh install_linux.sh uninstall_linux.sh build_linux.sh test.sh prepare_release.sh
git diff --check
git status --short
```

## 2. Test and prepare archives

```bash
python3 -m venv .venv-release
.venv-release/bin/python -m pip install -r requirements-runtime.txt
PYTHON=.venv-release/bin/python ./test.sh
PYTHON=.venv-release/bin/python ./prepare_release.sh
cat release-out/v2.0.0/SHA256SUMS.txt
```

On Nobara, install Tk first if you also want GUI/build validation:

```bash
sudo dnf install python3-tkinter rsync desktop-file-utils binutils
./build_linux.sh
```

## 3. Commit and push `main`

```bash
git add --all
git commit -m "release: GTNH World Voider 2.0.0"
git push origin main
```

## 4. Tag the exact pushed commit

```bash
git tag -a v2.0.0 -m "GTNH World Voider 2.0.0"
git push origin v2.0.0
```

The tag starts the GitHub release workflow. Wait for it to finish:

```bash
gh run list --workflow release.yml --limit 5
gh run watch
```

## 5. Create or update the GitHub Latest release

If the workflow has not created the release:

```bash
gh release create v2.0.0 \
  --repo Roaring1/-MinecraftWorld-Voider-PrismLauncher-GTNH-1.7.10-v2.8.0- \
  --title "GTNH World Voider 2.0.0" \
  --notes-file CHANGELOG.md \
  --latest \
  release-out/v2.0.0/*
```

If `v2.0.0` already exists, upload/replace local source artifacts and mark it Latest:

```bash
gh release upload v2.0.0 \
  --repo Roaring1/-MinecraftWorld-Voider-PrismLauncher-GTNH-1.7.10-v2.8.0- \
  --clobber release-out/v2.0.0/*
gh release edit v2.0.0 \
  --repo Roaring1/-MinecraftWorld-Voider-PrismLauncher-GTNH-1.7.10-v2.8.0- \
  --title "GTNH World Voider 2.0.0" \
  --notes-file CHANGELOG.md \
  --latest
```

## 6. Verify what users will download

```bash
gh release view v2.0.0 \
  --repo Roaring1/-MinecraftWorld-Voider-PrismLauncher-GTNH-1.7.10-v2.8.0- \
  --json url,isLatest,name,tagName,assets
gh release download v2.0.0 --dir /tmp/gtnh-voider-release-check
(cd /tmp/gtnh-voider-release-check && sha256sum -c SHA256SUMS.txt)
```

Do not use `--no-backup` for first live-world testing. Test a copied world first.
