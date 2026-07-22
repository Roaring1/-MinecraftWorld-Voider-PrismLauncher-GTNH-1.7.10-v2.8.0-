#!/usr/bin/env python3
# 10/12/2025 — Chunk Map UI for GTNH Voider — FINAL+flow polish
# - Safer: thread-safe UI logging (marshal via .after), robust wheel bindings
# - JourneyMap overlay: real alpha via RGBA pre-multiply + smarter caching
# - Spawn marker + per-dimension quick stats (present / include / exclude)
# - Undo/Redo, Snap 32×32, Selected-only, Save/Load, Help, Tools, NEI auto, Biome rule scan
# - Public API unchanged: launch_chunk_selector(world_path, initial_dim=None, preload=None, limit_tags=None)
#
# Requires: Pillow (PIL), NBT
#   pip install Pillow NBT

from __future__ import annotations
import os, sys, math, json, time, glob, struct, string, traceback, threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import deque

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import webbrowser

try:
    from PIL import Image, ImageTk  # type: ignore
except Exception as _pil_e:
    raise SystemExit(
        "Pillow (PIL) is required for chunkmap_ui.\nInstall with: pip install Pillow\n\nDetails: %r" % (_pil_e,)
    )

# Resolve main_voider lazily. Importing it here creates a cycle when
# main_voider is the application entry point and imports this module.
def _get_main_module():
    import importlib
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "reset_biomes") and hasattr(main, "RulesEditorDialog"):
        return main
    return importlib.import_module("main_voider")

# ---------- Help / texts ----------
HELP_LONG = (
    "Chunk Selector — Full Help\n\n"
    "Goal: Pick chunks to INCLUDE (green) or EXCLUDE (red) per dimension. Save writes {world}/.voider-selection.json.\n\n"
    "Basic Controls:\n"
    " • Left click a cell to toggle a single chunk in the current mode.\n"
    " • Drag with left mouse to select a box of chunks.\n"
    " • Hold Ctrl while clicking/dragging to invert the target set (Include↔Exclude) for that gesture.\n"
    " • Right/Middle mouse drag to pan. Mouse wheel zooms in/out.\n"
    " • Press R to toggle Snap 32×32 (align rectangles to .mca region boundaries).\n\n"
    "Legend toggles:\n"
    " • Existing: draw filled squares for generated chunks (present in region files).\n"
    " • Include / Exclude: show green/red outlines for your selections.\n"
    " • Axis: world axes; label at (0,0). A gold ✚ marks spawn.\n"
    " • Grid: chunk grid; thicker lines at region boundaries (every 32 chunks).\n\n"
    "JourneyMap overlay:\n"
    " • If JourneyMap cache is found, choose a layer (day/topo/night/underground/auto) and adjust α to blend imagery.\n"
    " • Imagery is skipped when too zoomed out for speed.\n\n"
    "Undo/Redo: Ctrl+Z / Ctrl+Y.  Save: Ctrl+S.\n\n"
    "Session Safety:\n"
    " • Close Minecraft before writing. session.lock normally persists after shutdown, so its presence alone is not proof the game is open.\n\n"
    "Tools menu:\n"
    " • Reset Biomes (All / Non-Overworld) — provider-safe policy: Overworld missing biomes → Plains (ID=1),\n"
    "   other dims → remove Biomes so the provider recomputes.\n"
    " • Import NEI Dump (Auto) — scans the instance for a 'Dump ID Map'.\n"
    " • Open Rules Editor — opens the main app’s rules editor if available.\n\n"
    "Files:\n"
    " • Selection: {world}/.voider-selection.json\n"
    " • UI state:  {world}/.voider-ui.json\n\n"
    "Repo:\n"
    " • https://github.com/Roaring1/-MinecraftWorld-Voider-PrismLauncher-GTNH-1.7.10-v2.8.0-\n"
)

GITHUB_REPO = "https://github.com/Roaring1/-MinecraftWorld-Voider-PrismLauncher-GTNH-1.7.10-v2.8.0-"

TT = {
    "existing": ("Show/hide known (already generated) chunks.",
                 "Existing: filled squares for chunks present in region files. Toggle for clarity with overlays."),
    "include": ("Mark clicks/drags as INCLUDE (green).",
                "Include mode: chunks you click/drag become INCLUDED. Ctrl flips the mode for that gesture."),
    "exclude": ("Mark clicks/drags as EXCLUDE (red).",
                "Exclude mode: chunks you click/drag become EXCLUDED. Ctrl flips the mode."),
    "axis": ("Show world axes (0,0) + spawn marker.",
             "Axis lines pass through (0,0). A gold ✚ marks spawn from level.dat."),
    "grid": ("Chunk grid; thick lines at region edges.",
             "Grid: thin = chunks, thick = region boundary (32×32 per .mca). Use Grid Scale to adjust density."),
    "jm": ("Toggle JourneyMap overlay from cache.",
           "Reads PNG tiles/chunks from your JourneyMap cache. ‘auto’ picks a sensible layer."),
    "layer": ("Choose JM layer (day/topo/night/underground/auto).",
              "Default is ‘auto’ if present, else the first available layer. Remembered per dimension."),
    "alpha": ("Overlay transparency (0–100%).",
              "Applies real alpha to the overlay imagery (Tk 8.6+) by pre-multiplying the PNG alpha."),
    "selonly": ("Hide non-selected chunks to focus.",
                "Selected-only: show only chunks you’ve included/excluded, plus grid/imagery."),
    "snap": ("Snap rectangles to 32×32 chunk regions.",
             "Snap 32×32 aligns large selections to .mca file boundaries for efficient processing."),
    "clear": ("Clear Include/Exclude marks for this dimension.",
              "Removes selection overlays (does not modify world files)."),
    "save": ("Save selection to .voider-selection.json.",
             "Writes selection under the world folder so main tool can pick it up."),
    "load": ("Load selection JSON from disk.",
             "Load a prior selection; replaces current. Use Undo if you did it by accident."),
    "done": ("Save selection & close.",
             "Writes selection, persists UI, then closes."),
    "copylog": ("Copy the log text to clipboard.",
                "Copies the log pane content so you can paste into issues or chat."),
}

SECTOR_BYTES = 4096

# ---------- Region helpers ----------
def _read_header(fp) -> Tuple[List[int], List[int]]:
    fp.seek(0)
    hdr = fp.read(8192)
    if len(hdr) != 8192:
        raise IOError("bad region header size")
    offsets = [struct.unpack(">I", hdr[i*4:(i+1)*4])[0] for i in range(1024)]
    stamps  = [struct.unpack(">I", hdr[4096+i*4:4096+(i+1)*4])[0] for i in range(1024)]
    return offsets, stamps

def _parse_loc(entry:int) -> Tuple[int,int]:
    return (entry >> 8) & 0xFFFFFF, entry & 0xFF

def _region_coords_from_name(path:Path) -> Tuple[int,int]:
    # r.<rx>.<rz>.mca
    try:
        _, rx, rz = path.stem.split(".")
        return int(rx), int(rz)
    except Exception:
        raise ValueError(f"Bad region filename: {path.name}")

def _iter_present_chunks_in_region(mca_path:Path) -> List[Tuple[int,int]]:
    """Return absolute chunk (cx,cz) that exist in this region file (not empty)."""
    present = []
    rx, rz = _region_coords_from_name(mca_path)
    with open(mca_path, "rb") as fp:
        fsize = fp.seek(0, os.SEEK_END)
        fp.seek(0)
        offsets, _ = _read_header(fp)
        for idx, entry in enumerate(offsets):
            if entry == 0: continue
            off, cnt = _parse_loc(entry)
            if off == 0 or cnt == 0: continue
            if (off * SECTOR_BYTES) + 5 > fsize: continue
            cx_local = idx % 32
            cz_local = idx // 32
            cx_abs   = rx*32 + cx_local
            cz_abs   = rz*32 + cz_local
            present.append((cx_abs, cz_abs))
    return present

def _find_region_dirs(world:Path) -> List[Path]:
    out=[]
    for p in world.rglob("region"):
        if p.is_dir() and p.parent.name not in ("entities","poi"):
            out.append(p)
    out.sort()
    return out

def _dim_tag(world:Path, rdir:Path) -> str:
    return str(rdir.relative_to(world)).replace("\\","/")

# ---------- Friendly dim names ----------
DIM_FRIENDLY_MAP = {
    "region": "Overworld",
    "DIM-1/region": "Nether",
    "DIM1/region": "The End",
    "DIM7/region": "Twilight Forest",
    "DIM112/region": "Twilight Dream",
    "DIM-112/region": "Twilight Cave",
    "DIM64/region": "Moon (Galacticraft)",
    "DIM65/region": "Mars (Galacticraft)",
    "DIM-28/region": "Asteroids",
    "DIM2/region": "Underdark",
    "DIM3/region": "Deep Dark / Mining",
    "DIM20/region": "Last Millennium",
    "DIM-37/region": "Compact Machines",
    "DIM-38/region": "Eldritch",
    "DIM-100/region": "Outer Lands",
    "DIM6/region": "Betweenlands",
    "DIM-343800852/region": "Tropics",
}
def _friendly_dim_name(tag:str) -> str:
    if tag in DIM_FRIENDLY_MAP: return DIM_FRIENDLY_MAP[tag]
    if tag.lower().endswith("/region") and tag.upper().startswith("DIM"):
        stem = tag[:-len("/region")]
        return f"{stem} (mod dimension)"
    return tag

# ---------- NEI auto-discovery + parsing ----------
def _try_load_nei_dump(path: Optional[Path]) -> Dict[str, str]:
    out = {}
    if not path or not path.exists(): return out
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s: continue
            if "Name:" in s:
                nm = s.split("Name:", 1)[1]
                nm = nm.split("ID:")[0].strip().strip(".")
                key = nm.replace("tile.", "").replace("item.", "").replace(" ", "_").lower()
                out[key] = nm
    except Exception:
        pass
    return out

def _find_nei_dump_auto(world: Path) -> Optional[Path]:
    roots = []
    parts = list(world.resolve().parts)
    for i in range(len(parts), 1, -1):
        cand = Path(*parts[:i])
        if cand.exists(): roots.append(cand)
    app = os.environ.get("APPDATA")
    if app:
        roots.append(Path(app)/".minecraft")
        roots.append(Path(app)/"PrismLauncher")
        roots.append(Path(app)/"MultiMC"); roots.append(Path(app)/".multimc")
        roots.append(Path(app)/"ATLauncher"); roots.append(Path(app)/".atlauncher")
        roots.append(Path(app)/"gdlauncher"); roots.append(Path(app)/"gdlauncher_next")
    best: Optional[Path] = None
    best_score = -1
    tried=set()
    for root in roots:
        try: root = root.resolve()
        except Exception: continue
        if not root.exists(): continue
        key = str(root)
        if key in tried: continue
        tried.add(key)
        for p in root.rglob("*.txt"):
            nm = p.name.lower()
            s = nm.replace("_", " ").replace("-", " ")
            if ("dump" in s and "id" in s and "map" in s) or ("id" in s and "map" in s):
                sz = p.stat().st_size
                score = max(0, 2_000_000 - min(2_000_000, sz))  # prefer smaller (<2MB)
                if score > best_score:
                    best, best_score = p, score
    return best

# ---------- Selection model with Undo/Redo ----------
class Selection:
    def __init__(self):
        self.by_dim: Dict[str, Dict[str, Set[Tuple[int,int]]]] = {}
        self._undo: deque = deque()
        self._redo: deque = deque()
        self._cap = 50

    def ensure_dim(self, dim:str):
        if dim not in self.by_dim:
            self.by_dim[dim] = {"include": set(), "exclude": set()}

    def _push_undo(self, dim:str):
        snap = (dim,
                frozenset(self.by_dim.get(dim, {}).get("include", set())),
                frozenset(self.by_dim.get(dim, {}).get("exclude", set())))
        self._undo.append(snap)
        if len(self._undo) > self._cap: self._undo.popleft()
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo: return False
        dim, inc, exc = self._undo.pop()
        self._redo.append((dim,
                           frozenset(self.by_dim.get(dim, {}).get("include", set())),
                           frozenset(self.by_dim.get(dim, {}).get("exclude", set()))))
        self.by_dim.setdefault(dim, {"include": set(), "exclude": set()})
        self.by_dim[dim]["include"] = set(inc)
        self.by_dim[dim]["exclude"] = set(exc)
        return True

    def redo(self) -> bool:
        if not self._redo: return False
        dim, inc, exc = self._redo.pop()
        self._push_undo(dim)
        self.by_dim.setdefault(dim, {"include": set(), "exclude": set()})
        self.by_dim[dim]["include"] = set(inc)
        self.by_dim[dim]["exclude"] = set(exc)
        return True

    def add(self, dim:str, coord:Tuple[int,int], mode:str):
        self.ensure_dim(dim)
        self._push_undo(dim)
        if mode == "include":
            self.by_dim[dim]["include"].add(coord)
            self.by_dim[dim]["exclude"].discard(coord)
        elif mode == "exclude":
            self.by_dim[dim]["exclude"].add(coord)
            self.by_dim[dim]["include"].discard(coord)

    def add_rect(self, dim:str, x1:int, z1:int, x2:int, z2:int, mode:str):
        self.ensure_dim(dim)
        self._push_undo(dim)
        minx, maxx = (x1, x2) if x1<=x2 else (x2, x1)
        minz, maxz = (z1, z2) if z1<=z2 else (z2, z1)
        target = self.by_dim[dim]["include"] if mode=="include" else self.by_dim[dim]["exclude"]
        other  = self.by_dim[dim]["exclude"] if mode=="include" else self.by_dim[dim]["include"]
        for cx in range(minx, maxx+1):
            for cz in range(minz, maxz+1):
                target.add((cx,cz))
                other.discard((cx,cz))

    def to_simple_json(self) -> Dict[str, Dict[str, List[List[int]]]]:
        out: Dict[str, Dict[str, List[List[int]]]] = {}
        for dim, d in self.by_dim.items():
            out[dim] = {
                "include": sorted([list(t) for t in d["include"]]),
                "exclude": sorted([list(t) for t in d["exclude"]]),
            }
        return out

    def from_simple_json(self, data:Dict):
        self.by_dim.clear()
        if not data: return
        mapping = data["by_dim"] if "by_dim" in data and isinstance(data["by_dim"], dict) else data
        for dim, d in mapping.items():
            inc = set(tuple(x) for x in d.get("include", []))
            exc = set(tuple(x) for x in d.get("exclude", []))
            self.by_dim[dim] = {"include": inc, "exclude": exc}
        self._undo.clear(); self._redo.clear()

# ---------- Theme ----------
def _detect_theme_colors() -> Tuple[Dict[str,str], bool]:
    DARK = {
        "BG": "#0d1117",
        "Panel": "#0f141b",
        "Text": "#e6edf3",
        "Subtle": "#9aa6b2",
        "InputBG": "#0b0f14",
        "InputText": "#e6edf3",
        "Border": "#2b3138",
        "GridMajor": "#2e3a46",
        "GridMinor": "#202a33",
        "Axis": "#8b949e",
        "Accent": "#4aa3ff",
        "Present": "#274861",
        "IncOutline": "#3fb950",
        "ExcOutline": "#f85149",
        "VB": "#12161c",
        "HelpBG": "#0f141b",
        "Warn": "#f2cc60",
        "Spawn": "#f0c400",
    }
    return DARK, False

# ---------- JourneyMap tile source ----------
class JMTileSource:
    LAYERS_ORDER = ("day", "topo", "night", "underground")
    AUTO_LAYER = "auto"

    def __init__(self, world: Path, dimtag: str, max_index: int = 500000, master: Optional[tk.Misc] = None):
        self.world = world
        self.dimtag = dimtag
        self.master = master or tk._default_root
        self.chunk_index: Dict[str, Dict[Tuple[int, int], Path]] = {}
        self.tile_index:  Dict[str, Dict[Tuple[int, int], Path]] = {}
        self.layers: List[str] = []
        from collections import OrderedDict
        self.cache: "OrderedDict[Tuple[str,int,int,int,int,int], tk.PhotoImage]" = OrderedDict()
        self.cache_limit = 4096
        self._build_index(max_index=max_index)

    def _guess_roots(self) -> List[Path]:
        cands: List[Path] = []
        try:
            mc = self.world.parents[1] if len(self.world.parents) >= 2 else None
            if mc and (mc / "journeymap").exists():
                cands.append(mc / "journeymap")
            inst = self.world.parents[2] if len(self.world.parents) >= 3 else None
            if inst and (inst / "journeymap").exists():
                cands.append(inst / "journeymap")
            inst2 = self.world.parents[3] if len(self.world.parents) >= 4 else None
            if inst2 and (inst2 / "journeymap").exists():
                cands.append(inst2 / "journeymap")
        except Exception:
            pass
        app = os.environ.get("APPDATA")
        if app:
            p1 = Path(app) / ".minecraft" / "journeymap"
            p2 = Path(app) / "journeymap"
            if p1.exists(): cands.append(p1)
            if p2.exists(): cands.append(p2)
        out, seen = [], set()
        for p in cands:
            try:
                s = str(p.resolve())
            except Exception:
                s = str(p)
            if s not in seen:
                seen.add(s); out.append(Path(s))
        return out

    def _dim_dirs_under(self, jm_root: Path) -> List[Path]:
        base = jm_root / "data"
        worldish: List[Path] = []
        for mode in ("sp", "mp"):
            mdir = base / mode
            if not mdir.exists(): continue
            w_exact = mdir / self.world.name
            if w_exact.exists(): worldish.append(w_exact)
            for wdir in mdir.iterdir():
                if wdir.is_dir() and wdir != w_exact:
                    worldish.append(wdir)

        target_dim = "DIM0" if self.dimtag == "region" else (
            self.dimtag[:-len("/region")] if self.dimtag.endswith("/region") else self.dimtag
        )

        results: List[Path] = []
        for wdir in worldish:
            strict = list(wdir.glob(target_dim + "*"))
            if strict:
                results.extend([d for d in strict if d.is_dir()])
            else:
                results.extend([d for d in wdir.glob("DIM*") if d.is_dir()])

        out, seen = [], set()
        for d in results:
            try:
                s = str(d.resolve())
            except Exception:
                s = str(d)
            if s not in seen:
                seen.add(s); out.append(Path(s))
        return out

    def _parse_chunk_name(self, stem: str) -> Optional[Tuple[int, int]]:
        try:
            if stem.startswith("tile_"):
                s = stem[5:]
                parts = s.split("_")
                x = next((p for p in parts if p.startswith("x") and p[1:].lstrip("-").isdigit()), None)
                z = next((p for p in parts if p.startswith("z") and p[1:].lstrip("-").isdigit()), None)
                if x and z:
                    return int(x[1:]), int(z[1:])
            toks = stem.split(".")
            def grab(label: str) -> Optional[int]:
                if label in toks:
                    i = toks.index(label)
                    if i + 1 < len(toks) and toks[i+1].lstrip("-").isdigit():
                        return int(toks[i+1])
                return None
            cx = grab("x"); cz = grab("z")
            if cx is not None and cz is not None:
                return cx, cz
            if "z" in toks and "x" in toks:
                zidx = toks.index("z"); xidx = toks.index("x")
                if zidx + 1 < len(toks) and toks[zidx+1].lstrip("-").isdigit() and \
                   xidx + 1 < len(toks) and toks[xidx+1].lstrip("-").isdigit():
                    return int(toks[xidx+1]), int(toks[zidx+1])
        except Exception:
            return None
        return None

    def _parse_tile_name(self, stem: str) -> Optional[Tuple[int, int]]:
        try:
            if "," in stem:
                a, b = stem.split(",", 1)
                if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
                    return int(a), int(b)
        except Exception:
            return None
        return None

    def _build_index(self, max_index: int):
        from collections import defaultdict
        chunk_b: Dict[str, Dict[Tuple[int, int], Path]] = defaultdict(dict)
        tile_b:  Dict[str, Dict[Tuple[int, int], Path]] = defaultdict(dict)
        dim_dirs: List[Path] = []
        for r in self._guess_roots():
            dim_dirs += self._dim_dirs_under(r)
        total = 0
        for ddir in dim_dirs:
            if total >= max_index: break
            had_layer = False
            for layer in self.LAYERS_ORDER:
                ldir = ddir / layer
                if not ldir.exists(): continue
                had_layer = True
                for png in ldir.rglob("*.png"):
                    stem = png.stem
                    cxcz = self._parse_chunk_name(stem)
                    if cxcz:
                        if cxcz not in chunk_b[layer]:
                            chunk_b[layer][cxcz] = png; total += 1
                            if total >= max_index: break
                        continue
                    tile = self._parse_tile_name(stem)
                    if tile:
                        if tile not in tile_b[layer]:
                            tile_b[layer][tile] = png; total += 1
                            if total >= max_index: break
                if total >= max_index: break
            if total >= max_index: break
            if not had_layer:
                for png in ddir.rglob("*.png"):
                    stem = png.stem
                    cxcz = self._parse_chunk_name(stem)
                    if cxcz:
                        if cxcz not in chunk_b[self.AUTO_LAYER]:
                            chunk_b[self.AUTO_LAYER][cxcz] = png; total += 1
                            if total >= max_index: break
                        continue
                    tile = self._parse_tile_name(stem)
                    if tile:
                        if tile not in tile_b[self.AUTO_LAYER]:
                            tile_b[self.AUTO_LAYER][tile] = png; total += 1
                            if total >= max_index: break

        found_layers: List[str] = [ly for ly in self.LAYERS_ORDER if (chunk_b.get(ly) or tile_b.get(ly))]
        if chunk_b.get(self.AUTO_LAYER) or tile_b.get(self.AUTO_LAYER):
            found_layers.append(self.AUTO_LAYER)
        self.chunk_index = {ly: chunk_b[ly] for ly in found_layers}
        self.tile_index  = {ly:  tile_b[ly]  for ly in found_layers}
        self.layers = found_layers

    def has_any(self) -> bool:
        return any((self.chunk_index.get(ly) or self.tile_index.get(ly)) for ly in self.layers)

    def available_layers(self) -> List[str]:
        return list(self.layers)

    def _cache_put(self, key, img):
        self.cache[key] = img
        self.cache.move_to_end(key)
        if len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)

    def _tile_chunks_per_side(self, path: Path) -> Optional[int]:
        try:
            im = Image.open(path)
            w = im.width
            if w % 16 != 0: return None
            return w // 16
        except Exception:
            return None

    def get_photoimage(self, layer: str, cx: int, cz: int, pw: int, ph: int, alpha_pct: int, master: tk.Misc) -> Optional[tk.PhotoImage]:
        """
        Return a Tk PhotoImage sized to (pw, ph) for chunk (cx,cz) on given layer.
        Applies alpha multiplier [0..100].
        """
        pw = max(1, int(pw)); ph = max(1, int(ph))
        key = (layer, cx, cz, pw, ph, int(alpha_pct))
        if key in self.cache:
            img = self.cache[key]
            self.cache.move_to_end(key)
            return img

        def _apply_rgba_alpha(pil: Image.Image) -> Image.Image:
            a = max(0, min(100, alpha_pct)) / 100.0
            if a >= 0.999: return pil
            if pil.mode != "RGBA":
                pil = pil.convert("RGBA")
            r, g, b, al = pil.split()
            # multiply alpha channel
            al = al.point(lambda p: int(p * a))
            return Image.merge("RGBA", (r, g, b, al))

        def _to_photo(pil: Image.Image) -> tk.PhotoImage:
            if pil.mode not in ("RGB", "RGBA"):
                pil = pil.convert("RGBA")
            if pil.width != pw or pil.height != ph:
                pil = pil.resize((pw, ph), Image.NEAREST)
            return ImageTk.PhotoImage(pil, master=master)

        # 1) Per-chunk imagery
        if layer in self.chunk_index:
            p = self.chunk_index[layer].get((cx, cz))
            if p and p.exists():
                try:
                    pil = Image.open(p)
                    pil = _apply_rgba_alpha(pil)
                    photo = _to_photo(pil)
                    self._cache_put(key, photo)
                    return photo
                except Exception:
                    pass

        # 2) Tiled imagery (big tile containing many chunks)
        if layer in self.tile_index and self.tile_index[layer]:
            any_tile_path = next(iter(self.tile_index[layer].values()))
            cpt = self._tile_chunks_per_side(any_tile_path) or 32
            tx = cx // cpt if cx >= 0 else -((-cx - 1) // cpt + 1)
            tz = cz // cpt if cz >= 0 else -((-cz - 1) // cpt + 1)
            p = self.tile_index[layer].get((tx, tz))
            if p and p.exists():
                try:
                    pil = Image.open(p)
                    if pil.mode not in ("RGB", "RGBA"):
                        pil = pil.convert("RGBA")
                    rel_x = cx - tx * cpt
                    rel_z = cz - tz * cpt
                    if 0 <= rel_x < cpt and 0 <= rel_z < cpt:
                        x0 = rel_x * 16
                        y0 = rel_z * 16
                        crop = pil.crop((x0, y0, x0 + 16, y0 + 16))
                        crop = _apply_rgba_alpha(crop)
                        photo = _to_photo(crop)
                        self._cache_put(key, photo)
                        return photo
                except Exception:
                    pass
        return None

# ---------- Biomes rule scan (optional “Select by Biome ID rule…”) ----------
def _get_chunk_biomes(raw: bytes) -> Optional[bytes]:
    try:
        from nbt import nbt
        nb = nbt.NBTFile(buffer=__import__("io").BytesIO(raw))
        lvl = nb["Level"]
        if "Biomes" in lvl and isinstance(lvl["Biomes"], nbt.TAG_Byte_Array):
            b = bytes(lvl["Biomes"].value)
            if len(b) == 256: return b
    except Exception:
        pass
    return None

def _read_chunk_payload(fp, sector_off:int) -> Optional[bytes]:
    fp.seek(sector_off * SECTOR_BYTES)
    lb = fp.read(4)
    if len(lb) < 4: return None
    (length,) = struct.unpack(">I", lb)
    comp_b = fp.read(1)
    if not comp_b: return None
    comp = comp_b[0]
    data = fp.read(max(0, length - 1))
    if comp == 2:
        import zlib
        return zlib.decompress(data)
    elif comp == 1:
        import gzip
        return gzip.decompress(data)
    elif comp == 3:
        return data
    return None

def _chunks_matching_biomes(mca: Path, wanted: Set[int]) -> Set[Tuple[int,int]]:
    hits=set()
    try:
        rx, rz = _region_coords_from_name(mca)
        with open(mca, "rb") as fp:
            offsets, _ = _read_header(fp)
            for idx, entry in enumerate(offsets):
                if entry == 0: continue
                off, cnt = _parse_loc(entry)
                if off == 0 or cnt == 0: continue
                raw = _read_chunk_payload(fp, off)
                if not raw: continue
                bio = _get_chunk_biomes(raw)
                if not bio: continue
                if any(b in wanted for b in bio):
                    cx_local = idx % 32
                    cz_local = idx // 32
                    cx_abs   = rx*32 + cx_local
                    cz_abs   = rz*32 + cz_local
                    hits.add((cx_abs, cz_abs))
    except Exception:
        pass
    return hits

# ---------- Canvas view ----------
class ChunkCanvas(ttk.Frame):
    """
    Chunk selector canvas with:
      • Debounced redraws
      • JourneyMap imagery with real alpha
      • Present/Include/Exclude overlays
      • Spawn marker + Minimap
    """
    REDRAW_MS = 16
    LOD_IMAGE_MIN_PX = 12
    MINIMAP_MAX_POINTS = 30000

    def __init__(self, master, owner, world: Path, dimtag: str,
                 present_chunks: Set[Tuple[int, int]],
                 selection: "Selection",
                 colors: Dict[str, str],
                 settings: Dict,
                 jm_source: "JMTileSource",
                 spawn_chunk: Optional[Tuple[int,int]] = None):
        super().__init__(master)
        self.owner = owner
        self.world = world
        self.dimtag = dimtag
        self.present = present_chunks
        self.sel = selection
        self.colors = colors
        self.settings = settings
        self.jm = jm_source
        self.spawn_chunk = spawn_chunk

        self.scale = float(settings.get("scale", 24.0))  # px per chunk
        if present_chunks:
            minx = min(c[0] for c in present_chunks)
            minz = min(c[1] for c in present_chunks)
        else:
            minx = minz = -8
        self.cam_x = float(settings.get("cam_x", minx))
        self.cam_z = float(settings.get("cam_z", minz))

        self.use_jm = tk.BooleanVar(value=bool(settings.get("show_jm", True)) and self.jm.has_any())
        default_layer = settings.get("jm_layer")
        if not default_layer or default_layer not in self.jm.available_layers():
            default_layer = self.jm.available_layers()[0] if self.jm.available_layers() else ""
        self.jm_layer = tk.StringVar(value=default_layer)
        self.jm_alpha = tk.DoubleVar(value=float(settings.get("jm_alpha", 1.0)))
        if not (0.0 <= self.jm_alpha.get() <= 1.0): self.jm_alpha.set(1.0)

        self.show_present = tk.BooleanVar(value=bool(settings.get("show_present", True)))
        self.show_inc = tk.BooleanVar(value=bool(settings.get("show_inc", True)))
        self.show_exc = tk.BooleanVar(value=bool(settings.get("show_exc", True)))
        self.show_grid = tk.BooleanVar(value=bool(settings.get("show_grid", True)))
        self.show_axis = tk.BooleanVar(value=bool(settings.get("show_axis", True)))
        self.view_selected_only = tk.BooleanVar(value=bool(settings.get("sel_only", False)))
        self.snap_region = tk.BooleanVar(value=bool(settings.get("snap_region", False)))

        self.canvas = tk.Canvas(self, bg=colors["VB"], highlightthickness=0, bd=0, cursor="crosshair")
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self._xscroll)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self._yscroll)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_lmb_down)
        self.canvas.bind("<B1-Motion>", self._on_lmb_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_lmb_up)
        self.canvas.bind("<Button-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<Button-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        # Wheel (Windows/macOS)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        # Wheel (X11)
        self.canvas.bind("<Button-4>", self._on_wheel_up)
        self.canvas.bind("<Button-5>", self._on_wheel_down)
        self.canvas.bind("<Motion>", self._on_motion)

        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_rect = None
        self._pan_anchor = None
        self.cursor_lbl = ttk.Label(self, text="", style="TLabel")
        self.cursor_lbl.grid(row=2, column=0, sticky="w")

        self._build_legend_help()
        self._build_minimap()

        self._image_refs: List[tk.PhotoImage] = []
        self._redraw_after_id: Optional[str] = None
        self._request_redraw()

    # ----- legend / help with tooltips -----
    def _build_legend_help(self):
        f = ttk.Frame(self)
        f.grid(row=3, column=0, sticky="we", padx=6, pady=(4, 2))

        def swatch(color: str):
            c = tk.Canvas(f, width=14, height=14, bd=0, highlightthickness=1,
                          highlightbackground=self.colors["Border"])
            c.create_rectangle(1, 1, 13, 13, fill=color, outline="")
            return c

        s1 = swatch(self.colors["Present"]); s1.grid(row=0, column=0, padx=(0, 4))
        cb_present = ttk.Checkbutton(f, text="Existing", variable=self.show_present, command=self._request_redraw)
        cb_present.grid(row=0, column=1, padx=(0, 12))
        _Tip(cb_present, *TT["existing"])

        s2 = swatch(self.colors["IncOutline"]); s2.grid(row=0, column=2, padx=(0, 4))
        cb_inc = ttk.Checkbutton(f, text="Include", variable=self.show_inc, command=self._request_redraw)
        cb_inc.grid(row=0, column=3, padx=(0, 12))
        _Tip(cb_inc, *TT["include"])

        s3 = swatch(self.colors["ExcOutline"]); s3.grid(row=0, column=4, padx=(0, 4))
        cb_exc = ttk.Checkbutton(f, text="Exclude", variable=self.show_exc, command=self._request_redraw)
        cb_exc.grid(row=0, column=5, padx=(0, 12))
        _Tip(cb_exc, *TT["exclude"])

        s4 = swatch(self.colors["Axis"]); s4.grid(row=0, column=6, padx=(0, 4))
        cb_axis = ttk.Checkbutton(f, text="Axis + Spawn", variable=self.show_axis, command=self._request_redraw)
        cb_axis.grid(row=0, column=7, padx=(0, 12))
        _Tip(cb_axis, *TT["axis"])

        s5 = swatch(self.colors["GridMajor"]); s5.grid(row=0, column=8, padx=(0, 4))
        cb_grid = ttk.Checkbutton(f, text="Grid", variable=self.show_grid, command=self._request_redraw)
        cb_grid.grid(row=0, column=9, padx=(0, 12))
        _Tip(cb_grid, *TT["grid"])

        cb_jm = ttk.Checkbutton(f, text="JourneyMap", variable=self.use_jm, command=self._request_redraw)
        cb_jm.grid(row=0, column=10, padx=(8, 6))
        _Tip(cb_jm, *TT["jm"])

        ttk.Label(f, text="Layer:").grid(row=0, column=11, padx=(6, 4))
        self.layer_combo = ttk.Combobox(f, state="readonly", width=12,
                                        values=self.owner.jm_layers_for_dim(self.dimtag))
        if self.jm_layer.get() and self.jm_layer.get() in self.owner.jm_layers_for_dim(self.dimtag):
            self.layer_combo.set(self.jm_layer.get())
        elif self.owner.jm_layers_for_dim(self.dimtag):
            self.layer_combo.set(self.owner.jm_layers_for_dim(self.dimtag)[0])
        self.layer_combo.bind("<<ComboboxSelected>>",
                              lambda _e=None: (self.jm_layer.set(self.layer_combo.get()), self._request_redraw()))
        self.layer_combo.grid(row=0, column=12, padx=(0, 12))
        _Tip(self.layer_combo, *TT["layer"])

        ttk.Label(f, text="Overlay α").grid(row=0, column=13, padx=(10, 4))
        self._alpha_pct = tk.IntVar(value=int(round(self.jm_alpha.get() * 100)))
        def _alpha_changed(_=None):
            self.jm_alpha.set(max(0.0, min(1.0, self._alpha_pct.get() / 100.0)))
            self._request_redraw()
        sld = tk.Scale(f, from_=0, to=100, orient="horizontal", showvalue=1, resolution=5,
                       length=140, variable=self._alpha_pct, command=_alpha_changed)
        sld.grid(row=0, column=14, padx=(0, 8))
        _Tip(sld, *TT["alpha"])

        cb_sel = ttk.Checkbutton(f, text="Selected only", variable=self.view_selected_only, command=self._request_redraw)
        cb_sel.grid(row=0, column=15, padx=(0, 12))
        _Tip(cb_sel, *TT["selonly"])

        cb_snap = ttk.Checkbutton(f, text="Snap 32×32 (R)", variable=self.snap_region, command=self._request_redraw)
        cb_snap.grid(row=0, column=16)
        _Tip(cb_snap, *TT["snap"])

        ttk.Label(f, text="Grid scale").grid(row=0, column=17, padx=(14, 4))
        self.grid_scale = tk.DoubleVar(value=float(self.settings.get("grid_scale", 1.0)))
        gs = tk.Scale(f, from_=0.0, to=2.0, orient="horizontal", showvalue=0, resolution=0.1,
                      length=120, variable=self.grid_scale, command=lambda _=None: self._request_redraw())
        gs.grid(row=0, column=18)
        _Tip(gs, "Density of grid lines (left=lighter, right=denser).",
             "Grid density can help declutter at different zoom levels.")

        # Stats line (per-dim)
        self._stats_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._stats_var).grid(row=5, column=0, sticky="w", padx=6)

        # Inline “Legend & Controls Help”
        helpbox = tk.Text(self, height=3, wrap="word", bg=self.colors["HelpBG"], fg=self.colors["Text"], bd=0)
        helpbox.insert("1.0",
            "Controls: LMB select (drag for box). Ctrl inverts. RMB/MMB pan. Wheel zoom. R toggles snap-to-32×32. "
            "Selected-only focuses on Include/Exclude. Press F1 for a full help sheet.")
        helpbox.configure(state="disabled")
        helpbox.grid(row=4, column=0, sticky="we", padx=6, pady=(2, 6))

    def _build_minimap(self):
        self.minimap = tk.Canvas(self, width=180, height=140, bg=self.colors["Panel"],
                                 bd=0, highlightthickness=1, highlightbackground=self.colors["Border"])
        self.minimap.grid(row=3, column=1, sticky="e", padx=(6, 10))
        self.minimap.bind("<Button-1>", self._mini_click)
        _Tip(self.minimap, "Mini-map of present chunks. Click to recenter.",
             "Mini-map shows a compressed footprint of generated chunks. Click to jump the camera center.")

    # ----- coord transforms -----
    def scr_to_chunk(self, sx: float, sy: float) -> Tuple[int, int]:
        cx = math.floor(self.cam_x + sx / self.scale)
        cz = math.floor(self.cam_z + sy / self.scale)
        return cx, cz

    def chunk_to_scr(self, cx: int, cz: int) -> Tuple[float, float]:
        sx = (cx - self.cam_x) * self.scale
        sy = (cz - self.cam_z) * self.scale
        return sx, sy

    # ----- scrolling / zoom -----
    def _xscroll(self, *args):
        if args[0] == "moveto":
            frac = float(args[1])
            self.cam_x = -2048 + frac * 4096
        elif args[0] == "scroll":
            units = int(args[1])
            self.cam_x += units * (self.canvas.winfo_width() / self.scale) * 0.1
        self._request_redraw()

    def _yscroll(self, *args):
        if args[0] == "moveto":
            frac = float(args[1])
            self.cam_z = -2048 + frac * 4096
        elif args[0] == "scroll":
            units = int(args[1])
            self.cam_z += units * (self.canvas.winfo_height() / self.scale) * 0.1
        self._request_redraw()

    def _on_resize(self, _): self._request_redraw()
    def _on_wheel(self, ev):
        now = time.time()
        if now - getattr(self, "_last_wheel", 0.0) < 0.04: return
        self._last_wheel = now
        self._zoom_at(ev.x, ev.y, 1 if ev.delta > 0 else -1)
    def _on_wheel_up(self, ev): self._zoom_at(ev.x, ev.y, +1)
    def _on_wheel_down(self, ev): self._zoom_at(ev.x, ev.y, -1)

    def _zoom_at(self, sx: int, sy: int, direction: int):
        old_scale = self.scale
        factor = 1.15 if direction > 0 else (1 / 1.15)
        new_scale = min(128.0, max(6.0, old_scale * factor))
        if abs(new_scale - old_scale) < 1e-6: return
        before_cx = self.cam_x + sx / old_scale
        before_cz = self.cam_z + sy / old_scale
        self.scale = new_scale
        self.cam_x = before_cx - sx / new_scale
        self.cam_z = before_cz - sy / new_scale
        self._request_redraw()

    # ----- drag/select -----
    def _on_pan_start(self, ev):
        self._pan_anchor = (ev.x, ev.y, self.cam_x, self.cam_z)

    def _on_pan_move(self, ev):
        if not self._pan_anchor: return
        ax, ay, ox, oz = self._pan_anchor
        dx = (ev.x - ax) / self.scale
        dz = (ev.y - ay) / self.scale
        self.cam_x = ox - dx
        self.cam_z = oz - dz
        self._request_redraw()

    def _on_lmb_down(self, ev):
        self.drag_start = self.scr_to_chunk(ev.x, ev.y)
        if self.drag_rect is not None:
            self.canvas.delete(self.drag_rect)
            self.drag_rect = None

    def _on_lmb_drag(self, ev):
        if not self.drag_start: return
        cx1, cz1 = self.drag_start
        cx2, cz2 = self.scr_to_chunk(ev.x, ev.y)
        if self.snap_region.get():
            def snap32(v: int) -> int:
                return (v // 32) * 32 if v >= 0 else -(((-v - 1) // 32 + 1) * 32)
            cx1s, cz1s = snap32(cx1), snap32(cz1)
            cx2s = snap32(cx2 + (1 if cx2 >= cx1 else -1))
            cz2s = snap32(cz2 + (1 if cz2 >= cz1 else -1)) - (1 if cz2 >= cz1 else -1)
            cx1, cz1, cx2, cz2 = cx1s, cz1s, cx2s, cz2s

        x1, y1 = self.chunk_to_scr(min(cx1, cx2), min(cz1, cz2))
        x2, y2 = self.chunk_to_scr(max(cx1, cx2) + 1, max(cz1, cz2) + 1)
        if self.drag_rect is not None:
            self.canvas.coords(self.drag_rect, x1, y1, x2, y2)
        else:
            self.drag_rect = self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=self.colors["Accent"], width=2, dash=(4, 2)
            )

    def _on_lmb_up(self, ev):
        if not self.drag_start: return
        cx1, cz1 = self.drag_start
        cx2, cz2 = self.scr_to_chunk(ev.x, ev.y)
        self.drag_start = None
        if self.drag_rect is not None:
            self.canvas.delete(self.drag_rect)
            self.drag_rect = None

        if self.snap_region.get():
            def snap32(v: int) -> int:
                return (v // 32) * 32 if v >= 0 else -(((-v - 1) // 32 + 1) * 32)
            cx1, cz1 = snap32(cx1), snap32(cz1)
            cx2 = snap32(cx2 + (1 if cx2 >= cx1 else -1)) - (1 if cx2 >= cx1 else -1)
            cz2 = snap32(cz2 + (1 if cz2 >= cz1 else -1)) - (1 if cz2 >= cz1 else -1)

        mode = self.owner.mode_var.get()
        if (abs(cx2 - cx1) <= 0) and (abs(cz2 - cz1) <= 0):
            if (ev.state & 0x0004):  # Ctrl flips set
                mode = "exclude" if mode == "include" else "include"
            self.sel.add(self.dimtag, (cx1, cz1), mode)
        else:
            self.sel.add_rect(self.dimtag, cx1, cz1, cx2, cz2, mode)
        self._request_redraw()

    def _on_motion(self, ev):
        cx, cz = self.scr_to_chunk(ev.x, ev.y)
        bx, bz = cx * 16, cz * 16
        self.cursor_lbl.config(text=f"Chunk ({cx},{cz}) — Blocks ({bx},{bz})..({bx+15},{bz+15})")

    # ----- drawing -----
    def _visible_chunk_bounds(self) -> Tuple[int, int, int, int]:
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        cx_min = math.floor(self.cam_x)
        cz_min = math.floor(self.cam_z)
        cx_max = math.floor(self.cam_x + w / self.scale)
        cz_max = math.floor(self.cam_z + h / self.scale)
        return cx_min - 1, cz_min - 1, cx_max + 1, cz_max + 1

    def _draw_grid(self):
        if not self.show_grid.get(): return
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        cx0, cz0, cx1, cz1 = self._visible_chunk_bounds()

        base_step = 1
        if self.scale < 16: base_step = 2
        if self.scale < 10: base_step = 4
        if self.scale < 8:  base_step = 8
        s = max(1, int(base_step * (1.0 + self.owner.grid_scale.get())))

        for cx in range(cx0, cx1 + 1):
            if cx % s != 0: continue
            sx, _ = self.chunk_to_scr(cx, cz0)
            col = self.colors["GridMajor"] if cx % 32 == 0 else self.colors["GridMinor"]
            self.canvas.create_line(sx, 0, sx, h, fill=col)
        for cz in range(cz0, cz1 + 1):
            if cz % s != 0: continue
            _, sy = self.chunk_to_scr(cx0, cz)
            col = self.colors["GridMajor"] if cz % 32 == 0 else self.colors["GridMinor"]
            self.canvas.create_line(0, sy, w, sy, fill=col)

    def _draw_axis_spawn(self):
        if not self.show_axis.get(): return
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        sx0, sy0 = self.chunk_to_scr(0, 0)
        self.canvas.create_line(sx0, 0, sx0, h, fill=self.colors["Axis"])
        self.canvas.create_line(0, sy0, w, sy0, fill=self.colors["Axis"])
        self.canvas.create_text(sx0 + 6, sy0 + 12, text="(0,0)", fill=self.colors["Axis"], anchor="nw")
        # Spawn marker (gold ✚)
        if self.spawn_chunk:
            scx, scz = self.spawn_chunk
            x1, y1 = self.chunk_to_scr(scx, scz)
            x2, y2 = self.chunk_to_scr(scx+1, scz+1)
            cx = (x1+x2)/2; cy = (y1+y2)/2
            self.canvas.create_line(cx-6, cy, cx+6, cy, fill=self.colors["Spawn"], width=2)
            self.canvas.create_line(cx, cy-6, cx, cy+6, fill=self.colors["Spawn"], width=2)

    def _draw_present_lod(self):
        cx0, cz0, cx1, cz1 = self._visible_chunk_bounds()
        fill = self.colors["Present"]
        layer = self.layer_combo.get() if (self.use_jm.get() and self.jm.has_any()) else None
        alpha_pct = int(round(self.jm_alpha.get() * 100))

        self._image_refs.clear()

        if self.view_selected_only.get():
            self.sel.ensure_dim(self.dimtag)
            inc = self.sel.by_dim[self.dimtag]["include"]
            exc = self.sel.by_dim[self.dimtag]["exclude"]
            consider = [c for c in self.present if (c in inc or c in exc)]
        else:
            consider = [c for c in self.present]

        use_images = bool(layer) and (self.scale >= self.LOD_IMAGE_MIN_PX)

        for (cx, cz) in consider:
            if not (cx0 <= cx <= cx1 and cz0 <= cz <= cz1): continue

            x1, y1 = self.chunk_to_scr(cx, cz)
            x2, y2 = self.chunk_to_scr(cx + 1, cz + 1)
            pw = int(max(1, x2 - x1))
            ph = int(max(1, y2 - y1))

            drawn = False
            if use_images:
                img = self.jm.get_photoimage(layer, cx, cz, pw, ph, alpha_pct, master=self)
                if img:
                    self.canvas.create_image(x1, y1, image=img, anchor="nw")
                    self._image_refs.append(img)
                    drawn = True

            if not drawn and self.show_present.get():
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="")

    def _draw_selection(self):
        self.sel.ensure_dim(self.dimtag)
        inc = self.sel.by_dim[self.dimtag]["include"]
        exc = self.sel.by_dim[self.dimtag]["exclude"]
        cx0, cz0, cx1, cz1 = self._visible_chunk_bounds()

        if self.show_inc.get():
            for (cx, cz) in inc:
                if cx0 <= cx <= cx1 and cz0 <= cz <= cz1:
                    x1, y1 = self.chunk_to_scr(cx, cz)
                    x2, y2 = self.chunk_to_scr(cx + 1, cz + 1)
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline=self.colors["IncOutline"], width=2)
        if self.show_exc.get():
            for (cx, cz) in exc:
                if cx0 <= cx <= cx1 and cz0 <= cz <= cz1:
                    x1, y1 = self.chunk_to_scr(cx, cz)
                    x2, y2 = self.chunk_to_scr(cx + 1, cz + 1)
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline=self.colors["ExcOutline"], width=2)

        # update stats line
        self._stats_var.set(f"Present: {len(self.present)}  |  Include: {len(inc)}  |  Exclude: {len(exc)}")

    def _draw_minimap(self):
        c = self.minimap
        c.delete("all")
        if not self.present: return

        pts = list(self.present)
        n = len(pts)
        if n > self.MINIMAP_MAX_POINTS:
            step = max(1, n // self.MINIMAP_MAX_POINTS)
            pts = pts[::step]

        xs = [x for (x, _) in pts]
        zs = [z for (_, z) in pts]
        minx, maxx = min(xs), max(xs)
        minz, maxz = min(zs), max(zs)
        w = int(c.cget("width"))
        h = int(c.cget("height"))
        dx = max(1, maxx - minx + 1)
        dz = max(1, maxz - minz + 1)
        scale = min((w - 8) / dx, (h - 8) / dz)

        sz = max(1, int(scale))
        for (cx, cz) in pts:
            x = 4 + int((cx - minx) * scale)
            y = 4 + int((cz - minz) * scale)
            c.create_rectangle(x, y, x + sz, y + sz, fill=self.colors["Present"], outline="")

        v0x, v0z, v1x, v1z = self._visible_chunk_bounds()
        x1 = 4 + int((v0x - minx) * scale)
        y1 = 4 + int((v0z - minz) * scale)
        x2 = 4 + int((v1x - minx) * scale)
        y2 = 4 + int((v1z - minz) * scale)
        c.create_rectangle(x1, y1, x2, y2, outline=self.colors["Accent"])

    def _mini_click(self, ev):
        if not self.present: return
        xs = [x for (x, _) in self.present]
        zs = [z for (_, z) in self.present]
        minx, maxx = min(xs), max(xs)
        minz, maxz = min(zs), max(zs)
        w = int(self.minimap.cget("width"))
        h = int(self.minimap.cget("height"))
        dx = max(1, maxx - minx + 1)
        dz = max(1, maxz - minz + 1)
        scale = min((w - 8) / dx, (h - 8) / dz)
        cx = minx + (ev.x - 4) / scale
        cz = minz + (ev.y - 4) / scale
        self.cam_x = float(cx) - (self.canvas.winfo_width() / self.scale) / 2
        self.cam_z = float(cz) - (self.canvas.winfo_height() / self.scale) / 2
        self._request_redraw()

    # ----- redraw plumbing -----
    def _request_redraw(self):
        if self._redraw_after_id is not None: return
        self._redraw_after_id = self.after(self.REDRAW_MS, self._do_redraw)

    def _do_redraw(self):
        self._redraw_after_id = None
        self._redraw_all()

    def _redraw_all(self):
        self.canvas.delete("all")
        self._draw_present_lod()
        self._draw_grid()
        self._draw_axis_spawn()
        self._draw_selection()
        self._draw_minimap()

    # ----- export settings -----
    def export_settings(self) -> Dict:
        return {
            "scale": self.scale,
            "cam_x": self.cam_x,
            "cam_z": self.cam_z,
            "show_jm": self.use_jm.get(),
            "jm_layer": self.layer_combo.get() if self.layer_combo.winfo_exists() else self.jm_layer.get(),
            "jm_alpha": float(self.jm_alpha.get()),
            "show_present": self.show_present.get(),
            "show_inc": self.show_inc.get(),
            "show_exc": self.show_exc.get(),
            "show_grid": self.show_grid.get(),
            "show_axis": self.show_axis.get(),
            "sel_only": self.view_selected_only.get(),
            "snap_region": self.snap_region.get(),
            "grid_scale": float(self.grid_scale.get()),
        }

# ---------- Tooltip helper ----------
class _Tip:
    def __init__(self, widget, short_text: str, long_text: Optional[str] = None, delay_ms: int = 450):
        self.widget = widget
        self.short = short_text
        self.long = long_text or short_text
        self.delay = delay_ms
        self._id = None
        self._tw = None
        self._expanded = [False]  # mutable cell

        root = widget.winfo_toplevel()
        root.bind_all("<KeyPress-Shift_L>", self._expand_on, add="+")
        root.bind_all("<KeyPress-Shift_R>", self._expand_on, add="+")
        root.bind_all("<KeyRelease-Shift_L>", self._expand_off, add="+")
        root.bind_all("<KeyRelease-Shift_R>", self._expand_off, add="+")
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")
        widget.bind("<ButtonPress>", self._leave, add="+")

    def _expand_on(self, _e=None):
        self._expanded[0] = True
        if self._tw: self._refresh()

    def _expand_off(self, _e=None):
        self._expanded[0] = False
        if self._tw: self._refresh()

    def _enter(self, _e=None): self._schedule()
    def _leave(self, _e=None): self._unschedule(); self._hide()
    def _schedule(self):
        self._unschedule()
        self._id = self.widget.after(self.delay, self._show)
    def _unschedule(self):
        if self._id is not None:
            self.widget.after_cancel(self._id); self._id = None
    def _text(self) -> str:
        return self.long if self._expanded[0] else self.short
    def _refresh(self):
        try:
            for child in self._tw.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(text=self._text())
        except Exception:
            pass
    def _show(self):
        if self._tw: return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tw = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        lbl = tk.Label(tw, text=self._text(), justify="left",
                       background="#161a22", foreground="#e6edf3",
                       relief="solid", borderwidth=1,
                       font=("Segoe UI", 9), padx=6, pady=4)
        lbl.pack()
    def _hide(self):
        if self._tw:
            self._tw.destroy()
            self._tw = None

# ---------- Main window ----------
class ChunkMapWindow(tk.Toplevel):
    def __init__(self, root, world:Path, colors:Dict[str,str], is_light:bool,
                 dims:List[Path], initial_dim:str, preload:Optional[Dict], ui_settings:Dict):
        super().__init__(root)
        self.world = world
        self.colors = colors
        self.is_light = is_light
        self.title("Voider — Chunk Selector")
        self.geometry("1200x820")
        self.minsize(920, 680)

        # session.lock warning
        self._warn_frame = ttk.Frame(self)
        self._warn_lbl = ttk.Label(self._warn_frame, text="", foreground=colors["Warn"])
        self._warn_btn = ttk.Button(self._warn_frame, text="Recheck", command=self._check_lock)
        self._warn_lbl.pack(side="left", padx=(0,8)); self._warn_btn.pack(side="left")
        self._warn_frame.pack(fill="x", padx=10, pady=(8,0))
        self._check_lock()

        self.selection = Selection()
        if preload:
            self.selection.from_simple_json(preload)

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background=colors["BG"])
        style.configure("TLabelframe", background=colors["BG"], foreground=colors["Text"])
        style.configure("TLabel", background=colors["BG"], foreground=colors["Text"])
        style.configure("TButton", background=colors["Panel"], foreground=colors["Text"])
        style.configure("Accent.TButton", background=colors["Accent"], foreground="#0b0e14" if not is_light else "#000")
        style.configure("TCombobox", fieldbackground=colors["InputBG"], foreground=colors["InputText"])

        # menu bar
        self._build_menubar()

        # top controls
        top = ttk.Frame(self); top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Dimension:").pack(side="left")
        self.dim_combo = ttk.Combobox(top, state="readonly", width=44)
        self.dim_combo.pack(side="left", padx=6)
        self.mode_var = tk.StringVar(value="include")
        self.btn_inc = ttk.Radiobutton(top, text="Include", value="include", variable=self.mode_var)
        self.btn_exc = ttk.Radiobutton(top, text="Exclude", value="exclude", variable=self.mode_var)
        self.btn_inc.pack(side="left", padx=(12,4)); self.btn_exc.pack(side="left")
        self.btn_clear = ttk.Button(top, text="Clear", command=self._clear_current); self.btn_clear.pack(side="left", padx=(12,4))
        self.btn_load  = ttk.Button(top, text="Load…", command=self._load_json);    self.btn_load.pack(side="left", padx=4)
        self.btn_save  = ttk.Button(top, text="Save",  command=self._save_json);     self.btn_save.pack(side="left", padx=4)
        self.btn_done  = ttk.Button(top, text="Save & Close", style="Accent.TButton", command=self._done); self.btn_done.pack(side="right")
        self.btn_help  = ttk.Button(top, text="Help (F1)", command=self._show_help); self.btn_help.pack(side="right", padx=(0,8))
        self.btn_copylog = ttk.Button(top, text="Copy Log", command=self._copy_log); self.btn_copylog.pack(side="right", padx=(0,8))
        self.btn_savelog = ttk.Button(top, text="Save Log…", command=self._save_log); self.btn_savelog.pack(side="right", padx=(0,8))

        _Tip(self.btn_inc, *TT["include"])
        _Tip(self.btn_exc, *TT["exclude"])
        _Tip(self.btn_clear, *TT["clear"])
        _Tip(self.btn_load, *TT["load"])
        _Tip(self.btn_save, *TT["save"])
        _Tip(self.btn_done, *TT["done"])
        _Tip(self.btn_copylog, *TT["copylog"])

        # hint bar
        helpf = ttk.Frame(self); helpf.pack(fill="x", padx=10)
        ttk.Label(helpf, text="LMB: select (drag box) • Ctrl flips set • RMB/MMB: pan • Wheel: zoom • R: snap 32×32 • Ctrl+S: save • Ctrl+Z/Y: undo/redo • F1: help").pack(anchor="w")

        # dimension list
        self.dim_to_dir: Dict[str, Path] = {}
        self.dim_order: List[str] = []
        shown=[]
        for rdir in dims:
            tag = _dim_tag(world, rdir)
            label = f"{_friendly_dim_name(tag)} — {tag}"
            shown.append(label)
            self.dim_to_dir[label] = rdir
            self.dim_order.append(tag)
        shown.sort(key=lambda s: s.lower())
        self.dim_combo["values"] = shown
        if initial_dim:
            cand = [s for s in shown if s.endswith(f"— {initial_dim}")]
            if cand: self.dim_combo.set(cand[0])
        if not self.dim_combo.get() and shown:
            self.dim_combo.set(shown[0])

        # JM sources
        self._jm_by_dim: Dict[str, JMTileSource] = {}
        for rdir in dims:
            tag = _dim_tag(world, rdir)
            self._jm_by_dim[tag] = JMTileSource(world, tag, master=self)

        # canvas holder
        self.canvas_holder = ttk.Frame(self); self.canvas_holder.pack(fill="both", expand=True, padx=10, pady=10)
        self.current_canvas: Optional[ChunkCanvas] = None
        self.ui_settings = ui_settings or {}

        # log area (bottom)
        lf = ttk.Labelframe(self, text="Log"); lf.pack(fill="both", padx=10, pady=(0,10))
        self.log = tk.Text(lf, height=8, wrap="word", bg=self.colors["HelpBG"], fg=self.colors["Text"], bd=0)
        self.log.pack(fill="both", expand=True)

        self.grid_scale = tk.DoubleVar(value=float(self.ui_settings.get("grid_scale_global", 1.0)))

        # bind menubar accelerators
        self.bind("<F1>", lambda _e=None: self._show_help())
        self.bind("<Control-s>", lambda _e=None: self._save_json())
        self.bind("<Control-S>", lambda _e=None: self._save_json())
        self.bind("<Control-z>", lambda _e=None: self._undo())
        self.bind("<Control-y>", lambda _e=None: self._redo())
        self.bind("<Key-r>", lambda _e=None: self._toggle_snap())

        self.dim_combo.bind("<<ComboboxSelected>>", self._on_dim_changed)
        self._on_dim_changed()

    # ----- menubar -----
    def _build_menubar(self):
        m = tk.Menu(self)
        # File
        fm = tk.Menu(m, tearoff=0)
        fm.add_command(label="Save (Ctrl+S)", command=self._save_json)
        fm.add_command(label="Load…", command=self._load_json)
        fm.add_separator()
        fm.add_command(label="Save & Close", command=self._done)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.destroy)
        m.add_cascade(label="File", menu=fm)

        # Edit
        em = tk.Menu(m, tearoff=0)
        em.add_command(label="Undo (Ctrl+Z)", command=self._undo)
        em.add_command(label="Redo (Ctrl+Y)", command=self._redo)
        em.add_separator()
        em.add_command(label="Select by Biome ID rule…", command=self._select_by_biome_rule)
        m.add_cascade(label="Edit", menu=em)

        # Tools
        tm = tk.Menu(m, tearoff=0)
        tm.add_command(label="Reset Biomes (All Dims, incl. Overworld)", command=self._reset_biomes_all)
        tm.add_command(label="Reset Biomes (All Non-Overworld Dims)", command=self._reset_biomes_non_over)
        tm.add_separator()
        tm.add_command(label="Import NEI Dump (Auto)", command=self._import_nei_auto)
        tm.add_command(label="Open Rules Editor", command=self._open_rules_editor)
        m.add_cascade(label="Tools", menu=tm)

        # Help
        hm = tk.Menu(m, tearoff=0)
        hm.add_command(label="Full Help (F1)", command=self._show_help)
        hm.add_command(label="Open GitHub Repo", command=lambda: webbrowser.open(GITHUB_REPO))
        m.add_cascade(label="Help", menu=hm)

        self.config(menu=m)

    # ----- JM helpers for per-dim layers -----
    def jm_layers_for_dim(self, tag:str) -> List[str]:
        src = self._jm_by_dim.get(tag)
        return src.available_layers() if src else []

    # ----- IO helpers -----
    def _scan_present(self, rdir:Path) -> Set[Tuple[int,int]]:
        present=set()
        files = [p for p in rdir.iterdir() if p.suffix==".mca" and p.name.startswith("r.")]
        for mca in files:
            try:
                present.update(_iter_present_chunks_in_region(mca))
            except Exception:
                pass
        return present

    # ----- actions -----
    def _on_dim_changed(self, *_):
        for c in self.canvas_holder.winfo_children(): c.destroy()
        label = self.dim_combo.get()
        rdir = self.dim_to_dir.get(label)
        present = self._scan_present(rdir) if rdir else set()
        tag = _dim_tag(self.world, rdir) if rdir else "region"
        per_dim = self.ui_settings.get("per_dim", {}).get(tag, {})
        jm = self._jm_by_dim.get(tag)
        if jm is None:
            jm = JMTileSource(self.world, tag, master=self)
            self._jm_by_dim[tag] = jm
        spawn = _try_spawn_center(self.world)
        self.current_canvas = ChunkCanvas(self.canvas_holder, self, self.world, tag, present, self.selection, self.colors, per_dim, jm, spawn_chunk=spawn)
        self.current_canvas.pack(fill="both", expand=True)

    def _clear_current(self):
        label = self.dim_combo.get()
        if not label: return
        rdir = self.dim_to_dir.get(label)
        dimtag = _dim_tag(self.world, rdir)
        self.selection.ensure_dim(dimtag)
        self.selection.by_dim[dimtag]["include"].clear()
        self.selection.by_dim[dimtag]["exclude"].clear()
        if self.current_canvas: self.current_canvas._redraw_all()

    def _load_json(self):
        p = filedialog.askopenfilename(title="Load selection JSON", filetypes=[("JSON","*.json"), ("All","*.*")])
        if not p: return
        try:
            data=json.loads(Path(p).read_text(encoding="utf-8"))
            self.selection.from_simple_json(data)
            if self.current_canvas: self.current_canvas._redraw_all()
            self._log(f"Loaded selection from {p}")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _save_json(self):
        try:
            data = self.selection.to_simple_json()
            default_path = self.world/".voider-selection.json"
            Path(default_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._save_ui_settings()
            self._log(f"Saved selection to {default_path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _save_ui_settings(self):
        try:
            per_dim = self.ui_settings.get("per_dim", {})
            label = self.dim_combo.get()
            if label:
                rdir = self.dim_to_dir.get(label)
                tag = _dim_tag(self.world, rdir)
                if self.current_canvas:
                    per_dim[tag] = self.current_canvas.export_settings()
            self.ui_settings["per_dim"] = per_dim
            self.ui_settings["grid_scale_global"] = float(self.grid_scale.get())
            (self.world/".voider-ui.json").write_text(json.dumps(self.ui_settings, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _done(self):
        self._save_json()
        self.destroy()

    def _show_help(self):
        messagebox.showinfo("Chunk Selector — Help", HELP_LONG.replace("{world}", str(self.world)))

    def _copy_log(self):
        try:
            txt = self.log.get("1.0", "end-1c")
            self.clipboard_clear(); self.clipboard_append(txt)
            self._log("Log copied to clipboard.")
        except Exception:
            messagebox.showerror("Copy failed", "Clipboard operation failed.")

    def _save_log(self):
        p = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text","*.txt"), ("All","*.*")])
        if not p: return
        try:
            txt = self.log.get("1.0", "end-1c")
            Path(p).write_text(txt, encoding="utf-8")
            self._log(f"Saved log to {p}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _undo(self):
        if self.selection.undo():
            if self.current_canvas: self.current_canvas._redraw_all()

    def _redo(self):
        if self.selection.redo():
            if self.current_canvas: self.current_canvas._redraw_all()

    def _toggle_snap(self):
        if self.current_canvas:
            self.current_canvas.snap_region.set(not self.current_canvas.snap_region.get())
            self.current_canvas._request_redraw()

    def _open_rules_editor(self):
        try:
            _mv = _get_main_module()
            if hasattr(_mv, "RulesEditorDialog"):
                dims = [_mv._dim_tag(self.world, d) for d in _mv._find_region_dirs(self.world)]
                if hasattr(_mv, "_load_rules"):
                    _mv._GLOBAL_RULES = _mv._load_rules(self.world)
                _mv.RulesEditorDialog(self, self.world, _mv._GLOBAL_RULES, dims, None)
                self._log("Opened Rules Editor.")
                return
            messagebox.showinfo("Open Rules Editor", "Open the main Voider app and click “Rules Editor…”.")
        except Exception as e:
            messagebox.showerror("Rules Editor failed", str(e))

    def _reset_biomes_all(self):
        self._reset_biomes(dims="all_including_overworld")

    def _reset_biomes_non_over(self):
        self._reset_biomes(dims="non_overworld")

    def _reset_biomes(self, dims:str):
        try:
            _mv = _get_main_module()
        except Exception as exc:
            messagebox.showwarning("Not available", f"main_voider could not be loaded:\n{exc}")
            return
        if not hasattr(_mv, "reset_biomes"):
            messagebox.showwarning("Not available", "main_voider.reset_biomes is not available.")
            return
        all_dirs = _find_region_dirs(self.world)
        tags = [_dim_tag(self.world, d) for d in all_dirs]
        if dims == "non_overworld":
            tags = [t for t in tags if t != "region"]
        elif dims == "all_including_overworld":
            pass
        else:
            return
        if not tags:
            messagebox.showinfo("Reset Biomes", "No dimensions found to reset.")
            return
        ok = messagebox.askyesno("Reset Biomes",
                                 "This will remove Level.Biomes from selected dimensions so providers recompute on next load.\n\n"
                                 f"Dimensions:\n  - " + "\n  - ".join(tags) + "\n\nProceed?")
        if not ok: return
        self._log("Starting Reset Biomes…")
        def worker():
            try:
                def prog_cb(done, total, line):
                    pct = int((done/total)*100) if total else 0
                    self._log(f"{pct}% — {line}", replace_line=True)
                changed = _mv.reset_biomes(self.world, dims=tags, progress_cb=prog_cb)
                self._log(f"\nReset biomes in {changed} chunk(s). Providers will refill on next load.")
            except Exception as e:
                self._log(f"[ERROR] Reset biomes failed: {e}")
        threading.Thread(target=worker, daemon=True).start()

    def _import_nei_auto(self):
        p = _find_nei_dump_auto(self.world)
        if not p:
            messagebox.showinfo("NEI Import", "Could not locate a NEI 'Dump ID Map' text file automatically.")
            return
        m = _try_load_nei_dump(p)
        self._log(f"NEI Import: {len(m)} entries from {p}")
        messagebox.showinfo("NEI Import", f"Loaded {len(m)} entries from:\n{p}")

    def _select_by_biome_rule(self):
        dlg = tk.Toplevel(self); dlg.title("Select by Biome ID")
        ttk.Label(dlg, text="Biome IDs (0..255), comma-separated:").pack(anchor="w", padx=10, pady=(10,4))
        ent = ttk.Entry(dlg, width=48); ent.pack(fill="x", padx=10); ent.focus_set()
        mode_var = tk.StringVar(value="include")
        frm = ttk.Frame(dlg); frm.pack(fill="x", padx=10, pady=6)
        ttk.Radiobutton(frm, text="Include", variable=mode_var, value="include").pack(side="left")
        ttk.Radiobutton(frm, text="Exclude", variable=mode_var, value="exclude").pack(side="left", padx=(8,0))
        btns = ttk.Frame(dlg); btns.pack(fill="x", padx=10, pady=(0,10))
        result = {"ok": False}
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="right")
        ttk.Button(btns, text="OK", command=lambda: (result.__setitem__("ok", True), dlg.destroy())).pack(side="right", padx=(0,6))
        dlg.transient(self); dlg.grab_set(); self.wait_window(dlg)
        if not result["ok"]: return
        raw = ent.get().strip()
        try:
            wanted = {int(x.strip()) for x in raw.split(",") if x.strip() != ""}
            wanted = {x for x in wanted if 0 <= x <= 255}
        except Exception:
            messagebox.showerror("Bad input", "Enter integers 0..255 separated by commas.")
            return
        if not wanted: return
        label = self.dim_combo.get()
        if not label: return
        rdir = self.dim_to_dir.get(label)
        if not rdir: return
        files = [p for p in rdir.iterdir() if p.suffix==".mca" and p.name.startswith("r.")]
        self._log(f"Scanning biomes in {len(files)} region file(s)… This may take a minute.")
        def worker():
            try:
                hits:set[Tuple[int,int]] = set()
                for i, mca in enumerate(files, 1):
                    h = _chunks_matching_biomes(mca, wanted)
                    hits |= h
                    self._log("  [{}/{}] {}: +{} match(es)".format(i, len(files), mca.name, len(h)), newline=True)
                dimtag = _dim_tag(self.world, rdir)
                if hits:
                    self.selection._push_undo(dimtag)
                    tgt = self.selection.by_dim.setdefault(dimtag, {"include": set(), "exclude": set()})
                    if mode_var.get()=="include":
                        tgt["include"].update(hits); tgt["exclude"].difference_update(hits)
                    else:
                        tgt["exclude"].update(hits); tgt["include"].difference_update(hits)
                if self.current_canvas: self.current_canvas._redraw_all()
                self._log(f"Biome rule selection complete. {len(hits)} chunk(s) affected.")
            except Exception as e:
                self._log(f"[ERROR] Biome scan failed: {e}")
        threading.Thread(target=worker, daemon=True).start()

    # ----- stats / logs / lock -----
    def _log(self, s: str, replace_line: bool=False, newline: bool=False):
        """
        Thread-safe append to self.log. Always marshal to Tk thread.
        """
        def _do():
            if replace_line:
                try:
                    self.log.delete("end-2l", "end")
                except Exception:
                    pass
            self.log.insert("end", s + ("\n" if (newline or not s.endswith("\n")) else ""))
            self.log.see("end")
        try:
            # If called from Tk thread, run directly; otherwise schedule
            if threading.current_thread() is threading.main_thread():
                _do()
            else:
                self.after(0, _do)
        except Exception:
            pass

    def _check_lock(self):
        lock = self.world/"session.lock"
        if lock.exists():
            self._warn_lbl.configure(text="Close Minecraft before writing. session.lock exists normally and is not an activity detector.")
            self._warn_frame.pack(fill="x", padx=10, pady=(8,0))
        else:
            self._warn_frame.pack_forget()

# ---------- World discovery (optional CLI helper) ----------
def _candidate_roots_windows() -> List[Path]:
    roots=set()
    user = Path(os.environ.get("USERPROFILE", Path.home()))
    appdata = Path(os.environ.get("APPDATA", user / "AppData" / "Roaming"))
    roots.add(appdata/".minecraft"/"saves")
    roots.update(Path(p) for p in glob.glob(str(appdata/"PrismLauncher"/"instances" / "*" / "minecraft" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(appdata/".multimc"/"instances" / "*" / "minecraft" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(appdata/"MultiMC"/"instances" / "*" / "minecraft" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(user/"curseforge"/"minecraft"/"Instances" / "*" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(user/"curseforge"/"minecraft"/"Instances" / "*" / "minecraft" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(appdata/"gdlauncher_next"/"instances" / "*" / ".minecraft" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(appdata/"gdlauncher"/"instances" / "*" / ".minecraft" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(appdata/".atlauncher"/"instances" / "*" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(appdata/".atlauncher"/"instances" / "*" / ".minecraft" / "saves")))
    if os.name != "nt":
        linux_patterns = [
            user/".minecraft"/"saves",
            user/".local"/"share"/"PrismLauncher"/"instances"/"*"/"minecraft"/"saves",
            user/".local"/"share"/"PrismLauncher"/"instances"/"*"/".minecraft"/"saves",
            user/".local"/"share"/"prismlauncher"/"instances"/"*"/"minecraft"/"saves",
            user/".var"/"app"/"org.prismlauncher.PrismLauncher"/"data"/"PrismLauncher"/"instances"/"*"/"minecraft"/"saves",
            user/"Games"/"PrismLauncher"/"instances"/"*"/"minecraft"/"saves",
        ]
        for pattern in linux_patterns:
            roots.update(Path(p) for p in glob.glob(str(pattern)))
    return [r for r in roots if r.exists()]

def _read_world_name(level_dat:Path) -> str:
    try:
        from nbt import nbt
        nd = nbt.NBTFile(filename=str(level_dat))
        data = nd["Data"]
        nm = str(data.get("LevelName").value) if "LevelName" in data else level_dat.parent.name
        printable = set(string.printable)
        nm = "".join(c for c in nm if c in printable).strip()
        return nm or level_dat.parent.name
    except Exception:
        return level_dat.parent.name

def _enumerate_worlds(max_worlds:int=3000, slow_scan_seconds:int=6) -> List[Tuple[str, Path]]:
    worlds=[]
    for root in _candidate_roots_windows():
        try:
            for lvl in Path(root).glob("*/level.dat"):
                worlds.append((_read_world_name(lvl), lvl.parent))
        except Exception:
            pass
        if len(worlds) >= max_worlds:
            return worlds
    deadline = time.time() + slow_scan_seconds
    drives=[]
    if os.name == "nt":
        for c in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            p = Path(f"{c}:/")
            if p.exists(): drives.append(p)
    else:
        drives = [Path("/")]
    for d in drives:
        try:
            for root, dirs, _files in os.walk(d, topdown=True):
                bn = os.path.basename(root).lower()
                if bn in ("windows","program files","program files (x86)","programdata","system volume information","$recycle.bin"):
                    dirs[:] = []; continue
                if bn == "saves":
                    for lvl in Path(root).glob("*/level.dat"):
                        worlds.append((_read_world_name(lvl), lvl.parent))
                        if len(worlds) >= max_worlds: return worlds
                if time.time() > deadline: return worlds
        except Exception:
            pass
    return worlds

def choose_world_folder() -> Optional[str]:
    root = tk.Tk(); root.withdraw()
    win = tk.Toplevel(root)
    win.title("Select a Minecraft World")
    win.geometry("1080x680")

    cols = ("World", "Path")
    tree = ttk.Treeview(win, columns=cols, show="headings")
    tree.heading("World", text="World")
    tree.heading("Path", text="Path")
    tree.column("World", width=360, anchor="w")
    tree.column("Path", width=680, anchor="w")
    tree.pack(fill="both", expand=True, padx=8, pady=8)

    btns = ttk.Frame(win); btns.pack(fill="x", padx=8, pady=(0,8))
    var_sel = {"path": None}

    def on_ok():
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("Pick a world", "Select a world from the list, or use Browse…")
            return
        var_sel["path"] = tree.item(sel[0], "values")[1]
        win.destroy()

    def on_cancel():
        var_sel["path"] = None
        win.destroy()

    def on_browse():
        p = filedialog.askdirectory(title="Pick a world folder (must contain level.dat)")
        if p and (Path(p)/"level.dat").exists():
            var_sel["path"] = p; win.destroy()
        elif p:
            messagebox.showerror("Invalid selection", "That folder does not contain level.dat")

    ttk.Button(btns, text="Browse…", command=on_browse).pack(side="left")
    ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right")
    ttk.Button(btns, text="OK", command=on_ok).pack(side="right", padx=(0,6))
    tree.bind("<Double-1>", lambda _e: on_ok())

    worlds = _enumerate_worlds()
    worlds.sort(key=lambda t: (t[0].lower(), str(t[1]).lower()))
    for nm, p in worlds:
        tree.insert("", "end", values=(nm, str(p)))

    win.bind("<Return>", lambda e: on_ok())
    win.bind("<Escape>", lambda e: on_cancel())

    root.wait_window(win)
    root.destroy()
    return var_sel["path"]

# ---------- Utilities ----------
def _discover_dims(world:Path, limit_tags:Optional[List[str]] = None) -> List[Path]:
    dirs = _find_region_dirs(world)
    if limit_tags:
        want = {t.strip().lower() for t in limit_tags}
        dirs = [d for d in dirs if _dim_tag(world,d).lower() in want]
    return dirs

def _load_preexisting_selection(world:Path, preload:Optional[Dict]) -> Optional[Dict]:
    if preload: return preload
    p = world/".voider-selection.json"
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return None
    return None

def _load_ui_settings(world:Path) -> Dict:
    p = world/".voider-ui.json"
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}

def _try_spawn_center(world:Path) -> Optional[Tuple[int,int]]:
    try:
        from nbt import nbt
        lvl = nbt.NBTFile(filename=str(world/"level.dat"))
        data = lvl["Data"]
        sx = int(data.get("SpawnX", nbt.TAG_Int(0)).value)
        sz = int(data.get("SpawnZ", nbt.TAG_Int(0)).value)
        return sx//16, sz//16
    except Exception:
        return None

# ---------- Public API ----------
def launch_chunk_selector(world_path:str,
                          initial_dim:Optional[str]=None,
                          preload:Optional[Dict]=None,
                          limit_tags:Optional[List[str]]=None) -> Dict[str, Dict[str, List[List[int]]]]:
    """
    Open the UI, block until closed, return selection dict:
      { "<dimtag>": { "include": [[cx,cz],...], "exclude": [[cx,cz],...] }, ... }
    """
    world = Path(world_path)
    if not (world/"level.dat").exists():
        raise SystemExit("level.dat not found — point to a valid world folder.")

    dims = _discover_dims(world, limit_tags=limit_tags)
    if not dims:
        raise SystemExit("No region folders found under this world (or after filtering).")

    colors, is_light = _detect_theme_colors()
    root = tk.Tk()
    try:
        base = tkfont.nametofont("TkDefaultFont")
        family = "Segoe UI Variable" if "Variable" in tkfont.families() else "Segoe UI"
        base.configure(family=family, size=11)
        root.option_add("*Font", base)
    except Exception:
        pass
    root.withdraw()

    preload_data = _load_preexisting_selection(world, preload)
    ui_settings = _load_ui_settings(world)

    if "per_dim" not in ui_settings:
        ui_settings["per_dim"] = {}
    if dims:
        tag0 = _dim_tag(world, dims[0])
        if tag0 not in ui_settings["per_dim"]:
            spawn = _try_spawn_center(world)
            if spawn:
                cx, cz = spawn
                ui_settings["per_dim"][tag0] = {"cam_x": float(cx-32), "cam_z": float(cz-18), "scale": 24.0}

    initial = initial_dim or _dim_tag(world, dims[0])
    win = ChunkMapWindow(root, world, colors, is_light, dims, initial_dim=initial,
                         preload=preload_data, ui_settings=ui_settings)

    win.grab_set()
    root.wait_window(win)

    sel = win.selection.to_simple_json()
    try:
        (world/".voider-selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        (world/".voider-ui.json").write_text(json.dumps(ui_settings, indent=2), encoding="utf-8")
    except Exception:
        pass
    root.destroy()
    return sel

# ---------- CLI (optional) ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Chunk Map UI for GTNH Voider — select chunks to include/exclude.")
    ap.add_argument("world", nargs="?", help="Path to world folder (contains level.dat). If omitted, a world picker opens.")
    ap.add_argument("--dim", help="Initial dimension tag (e.g., region, DIM-1/region, DIM7/region)")
    ap.add_argument("--limit", help="Comma-separated dimension tags to allow")
    args = ap.parse_args()

    world = args.world
    if not world:
        picked = choose_world_folder()
        if not picked: sys.exit(0)
        world = picked

    initial = args.dim
    tags = [t.strip() for t in args.limit.split(",")] if args.limit else None

    try:
        sel = launch_chunk_selector(world, initial_dim=initial, limit_tags=tags)
        print(json.dumps(sel, indent=2))
        print(f"\nSelection saved to: {Path(world)/'.voider-selection.json'}")
    except Exception as e:
        traceback.print_exc()
        try:
            tk.Tk().withdraw()
            messagebox.showerror("ChunkMap UI failed", str(e))
        except Exception:
            pass
        sys.exit(2)
