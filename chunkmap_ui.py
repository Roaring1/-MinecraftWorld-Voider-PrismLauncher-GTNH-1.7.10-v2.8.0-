#!/usr/bin/env python3
# 9/10/2025-2
# Chunk Map UI for GTNH 1.7.x Voider — FINAL
#
# - Top-down chunk selector with per-dimension picker (region/, DIM-1/region, DIM7/region, etc.)
# - JourneyMap imagery overlay (ALWAYS uses Pillow; enabled by default); day/topo/night layers with opacity slider
# - Per-dimension UI persistence (.voider-ui.json): camera, zoom, grid/axis toggles, JM layer & alpha, selected-only, snap-to-32×32
# - Legend (swatches + toggles), grid controls (density slider), HUD cursor, minimap viewport
# - Panning (RMB/MMB drag), zoom (wheel) with throttle, LMB select (click/drag box)
# - Include / Exclude modes; Ctrl inverts target set
# - Saves selection -> {world}/.voider-selection.json  (simple format used by main tool)
# - Public API: launch_chunk_selector(world_path:str, initial_dim:Optional[str], preload:Optional[dict], limit_tags:Optional[list]) -> dict
#
# Notes on JourneyMap:
# - We search for cached tiles beneath JM roots like:
#     <instance>/.minecraft/journeymap/data/{sp|mp}/<world_or_server_id>/DIM*/{day|night|topo|underground}/**/*.png
# - Dimension mapping: "region" => DIM0, "DIM-1/region" => DIM-1, etc.
# - Tile file patterns accepted:
#     x.<chunkX>.z.<chunkZ>.png      OR      tile_x<chunkX>_z<chunkZ>.png
# - Images are resized with Pillow (NEAREST) to exactly fit the chunk cell in the current zoom.
#
# Tested on Python 3.10–3.13, Windows 10+.

import os
import sys
import math
import json
import time
import glob
import struct
import string
import traceback
import re
import threading
from concurrent.futures import ThreadPoolExecutor, Future 
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Set

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

# Pillow is REQUIRED for this UI (always-on high quality imagery)
try:
    from PIL import Image, ImageTk  # type: ignore
except Exception as _pil_e:
    raise SystemExit(
        "Pillow (PIL) is required for chunkmap_ui. "
        "Install with: pip install Pillow\n\nDetails: %r" % (_pil_e,)
    )
_PIL_OK = True

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
            if entry == 0:
                continue
            off, cnt = _parse_loc(entry)
            if off == 0 or cnt == 0:
                continue
            if (off * SECTOR_BYTES) + 5 > fsize:
                continue
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

# ---------- Friendly dim names (GTNH + common) ----------
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
    if tag in DIM_FRIENDLY_MAP:
        return DIM_FRIENDLY_MAP[tag]
    if tag.lower().endswith("/region") and tag.upper().startswith("DIM"):
        stem = tag[:-len("/region")]
        return f"{stem} (mod dimension)"
    return tag

# ---------- Selection model ----------
class Selection:
    """Keep per-dim include/exclude sets of chunk tuples."""
    def __init__(self):
        self.by_dim: Dict[str, Dict[str, Set[Tuple[int,int]]]] = {}

    def ensure_dim(self, dim:str):
        if dim not in self.by_dim:
            self.by_dim[dim] = {"include": set(), "exclude": set()}

    def add(self, dim:str, coord:Tuple[int,int], mode:str):
        self.ensure_dim(dim)
        if mode == "include":
            self.by_dim[dim]["include"].add(coord)
            self.by_dim[dim]["exclude"].discard(coord)
        elif mode == "exclude":
            self.by_dim[dim]["exclude"].add(coord)
            self.by_dim[dim]["include"].discard(coord)

    def add_rect(self, dim:str, x1:int, z1:int, x2:int, z2:int, mode:str):
        self.ensure_dim(dim)
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
    }
    # dark is default; returns (colors, is_light)
    return DARK, False

# ---------- JourneyMap tile source ----------
# 9/10/2025-2
# ---------- JourneyMap tile source ----------
class JMTileSource:
    """
    JourneyMap tile discovery — robust for GTNH 2.8.0 / 1.7.10.

    Supports BOTH:
      • Per-chunk tiles:  x.<cx>.z.<cz>.png   or   tile_x<cx>_z<cz>.png
      • Mosaic tiles:     <tx>,<tz>.png  (each PNG is a tile made of 16×16px chunk sprites)

    Layers: day/topo/night/underground (if present). Falls back to an AUTO layer when
    PNGs are directly under DIM* with no explicit layer dir.

    Rendering: crops per-chunk 16×16 pixels from a mosaic tile, resizes via Pillow,
    and caches PhotoImages with an LRU cache. All PhotoImages are created with the
    provided Tk master so they live in the correct Tcl interpreter.
    """

    LAYERS_ORDER = ("day", "topo", "night", "underground")
    AUTO_LAYER = "auto"

    def __init__(self, world: Path, dimtag: str, max_index: int = 500000, master: Optional[tk.Misc] = None):
        self.world = world
        self.dimtag = dimtag
        self.master = master or tk._default_root  # bind images to caller's root/toplevel
        # Two indices:
        #  - chunk_index[layer][(cx,cz)] = path   (1 PNG per chunk)
        #  - tile_index[layer][(tx,tz)]  = path   (1 PNG per N×N chunks)
        self.chunk_index: Dict[str, Dict[Tuple[int, int], Path]] = {}
        self.tile_index:  Dict[str, Dict[Tuple[int, int], Path]] = {}
        self.layers: List[str] = []
        self.jm_roots = self._guess_roots()
        self.cache: "OrderedDict[Tuple[str,int,int,int,int], tk.PhotoImage]" = OrderedDict()
        self.cache_limit = 4096
        self._build_index(max_index=max_index)

    # ---- discovery roots ----
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
            s = str(p.resolve())
            if s not in seen:
                seen.add(s); out.append(Path(s))
        return out

    def _dim_dirs_under(self, jm_root: Path) -> List[Path]:
        base = jm_root / "data"
        worldish: List[Path] = []
        for mode in ("sp", "mp"):
            mdir = base / mode
            if not mdir.exists():
                continue
            # exact world name, or any world-like dir
            w_exact = mdir / self.world.name
            if w_exact.exists():
                worldish.append(w_exact)
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
            s = str(d.resolve())
            if s not in seen:
                seen.add(s); out.append(Path(s))
        return out

    # ---- filename parsing ----
    def _parse_chunk_name(self, stem: str) -> Optional[Tuple[int, int]]:
        """x.<cx>.z.<cz>.png   or   tile_x<cx>_z<cz>.png  (with extra tokens tolerated)."""
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
        """<tx>,<tz>.png  (tile grid index; each PNG contains a grid of chunk sprites)."""
        try:
            if "," in stem:
                a, b = stem.split(",", 1)
                if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
                    return int(a), int(b)
        except Exception:
            return None
        return None

    # ---- build indices ----
    def _build_index(self, max_index: int):
        """
        Populate:
          self.chunk_index[layer][(cx,cz)] = Path
          self.tile_index[layer][(tx,tz)]  = Path
        Honors explicit layer subdirs (day/topo/night/underground) and a flat 'auto' fallback.
        """
        from collections import defaultdict

        chunk_b: Dict[str, Dict[Tuple[int, int], Path]] = defaultdict(dict)
        tile_b:  Dict[str, Dict[Tuple[int, int], Path]] = defaultdict(dict)

        dim_dirs: List[Path] = []
        for r in self.jm_roots:
            dim_dirs += self._dim_dirs_under(r)

        total = 0
        for ddir in dim_dirs:
            if total >= max_index:
                break

            had_layer = False

            # Pass 1: explicit layer directories
            for layer in self.LAYERS_ORDER:
                ldir = ddir / layer
                if not ldir.exists():
                    continue
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
                if total >= max_index:
                    break

            if total >= max_index:
                break

            # Pass 2: flat files under DIM* → AUTO layer
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

        # finalize layer order
        found_layers: List[str] = [ly for ly in self.LAYERS_ORDER if (chunk_b.get(ly) or tile_b.get(ly))]
        if chunk_b.get(self.AUTO_LAYER) or tile_b.get(self.AUTO_LAYER):
            found_layers.append(self.AUTO_LAYER)

        self.chunk_index = {ly: chunk_b[ly] for ly in found_layers}
        self.tile_index  = {ly:  tile_b[ly]  for ly in found_layers}
        self.layers = found_layers

    # ---- access ----
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
        """
        If a tile image represents a grid of chunk sprites, return how many chunks per side.
        We infer this from width/16; JourneyMap 1.7 often uses 512×512 → 32 chunks/side.
        """
        try:
            im = Image.open(path)
            w = im.width
            if w % 16 != 0:
                return None
            return w // 16
        except Exception:
            return None

    def get_photoimage(
        self,
        layer: str,
        cx: int,
        cz: int,
        pw: int,
        ph: int,
        alpha: Optional[float] = None,   # tolerated extra param from caller; not applied here
        *_, **__,
    ) -> Optional[tk.PhotoImage]:
        """
        Return a Tk PhotoImage for the given chunk cell, sized to (pw, ph).
          - Prefer per-chunk tiles.
          - Otherwise crop a 16×16px sprite from a mosaic tile <tx>,<tz>.png and resize.
        """
        pw = max(1, int(pw)); ph = max(1, int(ph))
        key = (layer, cx, cz, pw, ph)
        if key in self.cache:
            img = self.cache[key]
            self.cache.move_to_end(key)
            return img

        # 1) direct per-chunk tile
        if layer in self.chunk_index:
            p = self.chunk_index[layer].get((cx, cz))
            if p and p.exists():
                try:
                    pil = Image.open(p)
                    if pil.mode not in ("RGB", "RGBA"):
                        pil = pil.convert("RGBA")
                    if pil.width != pw or pil.height != ph:
                        pil = pil.resize((pw, ph), Image.NEAREST)
                    photo = ImageTk.PhotoImage(pil, master=self.master)
                    self._cache_put(key, photo)
                    return photo
                except Exception:
                    pass  # fallthrough

        # 2) mosaic tile: compute (tx,tz), crop 16×16px sprite, resize
        if layer in self.tile_index and self.tile_index[layer]:
            any_tile_path = next(iter(self.tile_index[layer].values()))
            cpt = self._tile_chunks_per_side(any_tile_path) or 32  # default to 32 if unknown
            tx = cx // cpt if cx >= 0 else -((-cx - 1) // cpt + 1)  # floor-div for negatives
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
                        if crop.width != pw or crop.height != ph:
                            crop = crop.resize((pw, ph), Image.NEAREST)
                        photo = ImageTk.PhotoImage(crop, master=self.master)
                        self._cache_put(key, photo)
                        return photo
                except Exception:
                    pass

        return None

# ---------- Canvas view ----------
class ChunkCanvas(ttk.Frame):
    """
    Chunk selector canvas with:
      • Debounced redraws (prevents thrash on mouse move/zoom)
      • LOD: when a chunk cell is small on screen, skip JM imagery and draw fast fills
      • Present/Include/Exclude overlay rendering
      • Minimap with decimation for very large worlds
    Assumes JMTileSource provides: has_any(), available_layers(), get_photoimage(layer, cx, cz, pw, ph, *args)
    """

    # Redraw debounce (ms) and LOD thresholds (in pixels per chunk)
    REDRAW_MS = 16
    LOD_IMAGE_MIN_PX = 12  # if chunk on screen is smaller than this, skip JM imagery
    MINIMAP_MAX_POINTS = 30000  # decimate minimap if present set is huge

    def __init__(self, master, owner, world: Path, dimtag: str,
                 present_chunks: Set[Tuple[int, int]],
                 selection: "Selection",
                 colors: Dict[str, str],
                 settings: Dict,
                 jm_source: "JMTileSource"):
        super().__init__(master)
        self.owner = owner
        self.world = world
        self.dimtag = dimtag
        self.present = present_chunks
        self.sel = selection
        self.colors = colors
        self.settings = settings
        self.jm = jm_source

        # view state
        self.scale = float(settings.get("scale", 24.0))  # px per chunk
        # Camera defaults: center near min present or a sensible default
        if present_chunks:
            minx = min(c[0] for c in present_chunks)
            minz = min(c[1] for c in present_chunks)
        else:
            minx = minz = -8
        self.cam_x = float(settings.get("cam_x", minx))
        self.cam_z = float(settings.get("cam_z", minz))
        self.last_zoom_time = 0.0

        # Render toggles
        self.use_jm = tk.BooleanVar(value=bool(settings.get("show_jm", True)) and self.jm.has_any())
        # layer selection (default to first available if none saved)
        default_layer = settings.get("jm_layer")
        if not default_layer or default_layer not in self.jm.available_layers():
            default_layer = self.jm.available_layers()[0] if self.jm.available_layers() else ""
        self.jm_layer = tk.StringVar(value=default_layer)
        # opacity 0..1; store as 0..100 in UI
        self.jm_alpha = tk.DoubleVar(value=float(settings.get("jm_alpha", 1.0)))
        if self.jm_alpha.get() < 0.0 or self.jm_alpha.get() > 1.0:
            self.jm_alpha.set(1.0)

        self.show_present = tk.BooleanVar(value=bool(settings.get("show_present", True)))
        self.show_inc = tk.BooleanVar(value=bool(settings.get("show_inc", True)))
        self.show_exc = tk.BooleanVar(value=bool(settings.get("show_exc", True)))
        self.show_grid = tk.BooleanVar(value=bool(settings.get("show_grid", True)))
        self.show_axis = tk.BooleanVar(value=bool(settings.get("show_axis", True)))
        self.view_selected_only = tk.BooleanVar(value=bool(settings.get("sel_only", False)))
        self.snap_region = tk.BooleanVar(value=bool(settings.get("snap_region", False)))

        # canvas + scrollbars
        self.canvas = tk.Canvas(self, bg=colors["VB"], highlightthickness=0, bd=0, cursor="crosshair")
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self._xscroll)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self._yscroll)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # events
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_lmb_down)
        self.canvas.bind("<B1-Motion>", self._on_lmb_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_lmb_up)
        self.canvas.bind("<Button-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<Button-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<MouseWheel>", self._on_wheel)        # Windows/macOS
        self.canvas.bind("<Button-4>", self._on_wheel_up)       # Linux
        self.canvas.bind("<Button-5>", self._on_wheel_down)     # Linux
        self.canvas.bind("<Motion>", self._on_motion)

        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_rect = None
        self._pan_anchor = None
        self.cursor_lbl = ttk.Label(self, text="", style="TLabel")
        self.cursor_lbl.grid(row=2, column=0, sticky="w")

        # Legend / toggles / JM controls
        self._build_legend()
        # Minimap
        self._build_minimap()

        # keep PhotoImage refs per draw to avoid GC
        self._image_refs: List[tk.PhotoImage] = []

        # Debounced redraw handle
        self._redraw_after_id: Optional[str] = None

        self._request_redraw()

    # ----- legend / toggles -----
    def _build_legend(self):
        f = ttk.Frame(self)
        f.grid(row=3, column=0, sticky="we", padx=6, pady=(4, 2))

        def swatch(color: str):
            c = tk.Canvas(f, width=14, height=14, bd=0, highlightthickness=1,
                          highlightbackground=self.colors["Border"])
            c.create_rectangle(1, 1, 13, 13, fill=color, outline="")
            return c

        swatch(self.colors["Present"]).grid(row=0, column=0, padx=(0, 4))
        ttk.Checkbutton(f, text="Existing", variable=self.show_present,
                        command=self._request_redraw).grid(row=0, column=1, padx=(0, 12))

        swatch(self.colors["IncOutline"]).grid(row=0, column=2, padx=(0, 4))
        ttk.Checkbutton(f, text="Include", variable=self.show_inc,
                        command=self._request_redraw).grid(row=0, column=3, padx=(0, 12))

        swatch(self.colors["ExcOutline"]).grid(row=0, column=4, padx=(0, 4))
        ttk.Checkbutton(f, text="Exclude", variable=self.show_exc,
                        command=self._request_redraw).grid(row=0, column=5, padx=(0, 12))

        swatch(self.colors["Axis"]).grid(row=0, column=6, padx=(0, 4))
        ttk.Checkbutton(f, text="Axis", variable=self.show_axis,
                        command=self._request_redraw).grid(row=0, column=7, padx=(0, 12))

        swatch(self.colors["GridMajor"]).grid(row=0, column=8, padx=(0, 4))
        ttk.Checkbutton(f, text="Grid", variable=self.show_grid,
                        command=self._request_redraw).grid(row=0, column=9, padx=(0, 12))

        # JourneyMap controls
        ttk.Checkbutton(f, text="JourneyMap", variable=self.use_jm,
                        command=self._request_redraw).grid(row=0, column=10, padx=(8, 6))
        ttk.Label(f, text="Layer:").grid(row=0, column=11, padx=(6, 4))
        self.layer_combo = ttk.Combobox(f, state="readonly", width=12,
                                        values=self.owner.jm_layers_for_dim(self.dimtag))
        if self.jm_layer.get() and self.jm_layer.get() in self.owner.jm_layers_for_dim(self.dimtag):
            self.layer_combo.set(self.jm_layer.get())
        elif self.owner.jm_layers_for_dim(self.dimtag):
            self.layer_combo.set(self.owner.jm_layers_for_dim(self.dimtag)[0])
        self.layer_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e=None: (self.jm_layer.set(self.layer_combo.get()), self._request_redraw())
        )
        self.layer_combo.grid(row=0, column=12, padx=(0, 12))

        ttk.Label(f, text="Overlay α").grid(row=0, column=13, padx=(10, 4))
        # slider 0..100 mapped to jm_alpha 0..1
        self._alpha_pct = tk.IntVar(value=int(round(self.jm_alpha.get() * 100)))

        def _alpha_changed(_=None):
            self.jm_alpha.set(max(0.0, min(1.0, self._alpha_pct.get() / 100.0)))
            self._request_redraw()

        tk.Scale(f, from_=0, to=100, orient="horizontal", showvalue=1, resolution=5,
                 length=140, variable=self._alpha_pct, command=_alpha_changed).grid(row=0, column=14, padx=(0, 8))

        ttk.Checkbutton(f, text="Selected only", variable=self.view_selected_only,
                        command=self._request_redraw).grid(row=0, column=15, padx=(0, 12))
        ttk.Checkbutton(f, text="Snap 32×32", variable=self.snap_region,
                        command=self._request_redraw).grid(row=0, column=16)

        ttk.Label(f, text="Grid scale").grid(row=0, column=17, padx=(14, 4))
        self.grid_scale = tk.DoubleVar(value=float(self.settings.get("grid_scale", 1.0)))
        tk.Scale(f, from_=0.0, to=2.0, orient="horizontal", showvalue=0, resolution=0.1,
                 length=120, variable=self.grid_scale,
                 command=lambda _=None: self._request_redraw()).grid(row=0, column=18)

    def _build_minimap(self):
        self.minimap = tk.Canvas(self, width=180, height=140, bg=self.colors["Panel"],
                                 bd=0, highlightthickness=1, highlightbackground=self.colors["Border"])
        self.minimap.grid(row=3, column=1, sticky="e", padx=(6, 10))
        self.minimap.bind("<Button-1>", self._mini_click)

    # ----- coord transforms -----
    def scr_to_chunk(self, sx: float, sy: float) -> Tuple[int, int]:
        """Convert screen pixel (sx, sy) to chunk coords."""
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
    def _on_wheel(self, ev): self._zoom_at(ev.x, ev.y, 1 if ev.delta > 0 else -1)
    def _on_wheel_up(self, ev): self._zoom_at(ev.x, ev.y, +1)
    def _on_wheel_down(self, ev): self._zoom_at(ev.x, ev.y, -1)

    def _zoom_at(self, sx: int, sy: int, direction: int):
        now = time.time()
        if now - self.last_zoom_time < 0.04:  # throttle
            return
        self.last_zoom_time = now
        old_scale = self.scale
        factor = 1.15 if direction > 0 else (1 / 1.15)
        new_scale = min(128.0, max(6.0, old_scale * factor))
        if abs(new_scale - old_scale) < 1e-6:
            return
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
        if not self._pan_anchor:
            return
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
        if not self.drag_start:
            return
        cx1, cz1 = self.drag_start
        cx2, cz2 = self.scr_to_chunk(ev.x, ev.y)
        if self.snap_region.get():
            # snap rectangle to 32×32 chunk boundaries
            def snap32(v: int) -> int:
                return (v // 32) * 32 if v >= 0 else -(((-v - 1) // 32 + 1) * 32)
            cx1s, cz1s = snap32(cx1), snap32(cz1)
            cx2s, cz2s = snap32(cx2 + (1 if cx2 >= cx1 else -1)), snap32(cz2 + (1 if cz2 >= cz1 else -1)) - (1 if cz2 >= cz1 else -1)
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
        if not self.drag_start:
            return
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
            cx2, cz2 = snap32(cx2 + (1 if cx2 >= cx1 else -1)) - (1 if cx2 >= cx1 else -1), \
                       snap32(cz2 + (1 if cz2 >= cz1 else -1)) - (1 if cz2 >= cz1 else -1)

        mode = self.owner.mode_var.get()
        if (abs(cx2 - cx1) <= 0) and (abs(cz2 - cz1) <= 0):
            if (ev.state & 0x0004):  # Ctrl flips
                mode = "exclude" if mode == "include" else "include"
            self.sel.add(self.dimtag, (cx1, cz1), mode)
        else:
            self.sel.add_rect(self.dimtag, cx1, cz1, cx2, cz2, mode)
        self._request_redraw()

    def _on_motion(self, ev):
        cx, cz = self.scr_to_chunk(ev.x, ev.y)
        bx, bz = cx * 16, cz * 16
        self.cursor_lbl.config(text=f"Chunk ({cx},{cz}) — Blocks ({bx},{bz})..({bx+15},{bz+15})")
        # No redraw on mere motion; keeps things smooth.

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
        if not self.show_grid.get():
            return
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        cx0, cz0, cx1, cz1 = self._visible_chunk_bounds()

        # decimate based on zoom and slider
        base_step = 1
        if self.scale < 16: base_step = 2
        if self.scale < 10: base_step = 4
        if self.scale < 8:  base_step = 8
        s = max(1, int(base_step * (1.0 + self.grid_scale.get())))

        for cx in range(cx0, cx1 + 1):
            if cx % s != 0:
                continue
            sx, _ = self.chunk_to_scr(cx, cz0)
            col = self.colors["GridMajor"] if cx % 32 == 0 else self.colors["GridMinor"]
            self.canvas.create_line(sx, 0, sx, h, fill=col)
        for cz in range(cz0, cz1 + 1):
            if cz % s != 0:
                continue
            _, sy = self.chunk_to_scr(cx0, cz)
            col = self.colors["GridMajor"] if cz % 32 == 0 else self.colors["GridMinor"]
            self.canvas.create_line(0, sy, w, sy, fill=col)

    def _draw_axis(self):
        if not self.show_axis.get():
            return
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        sx0, sy0 = self.chunk_to_scr(0, 0)
        self.canvas.create_line(sx0, 0, sx0, h, fill=self.colors["Axis"])
        self.canvas.create_line(0, sy0, w, sy0, fill=self.colors["Axis"])
        self.canvas.create_text(sx0 + 6, sy0 + 12, text="(0,0)", fill=self.colors["Axis"], anchor="nw")

    def _draw_present_lod(self):
        cx0, cz0, cx1, cz1 = self._visible_chunk_bounds()
        fill = self.colors["Present"]
        layer = self.layer_combo.get() if (self.use_jm.get() and self.jm.has_any()) else None
        alpha = float(self.jm_alpha.get())

        # ImageRefs for this frame
        self._image_refs.clear()

        # If view_selected_only: restrict to selected chunks
        if self.view_selected_only.get():
            self.sel.ensure_dim(self.dimtag)
            inc = self.sel.by_dim[self.dimtag]["include"]
            exc = self.sel.by_dim[self.dimtag]["exclude"]
            consider = [c for c in self.present if (c in inc or c in exc)]
        else:
            consider = [c for c in self.present]

        # LOD: if on-screen cell is tiny, skip JM imagery (fast fill only)
        use_images = bool(layer) and (self.scale >= self.LOD_IMAGE_MIN_PX)

        for (cx, cz) in consider:
            if not (cx0 <= cx <= cx1 and cz0 <= cz <= cz1):
                continue

            x1, y1 = self.chunk_to_scr(cx, cz)
            x2, y2 = self.chunk_to_scr(cx + 1, cz + 1)
            pw = int(max(1, x2 - x1))
            ph = int(max(1, y2 - y1))

            drawn = False
            if use_images:
                img = self.jm.get_photoimage(layer, cx, cz, pw, ph, alpha)
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

    def _draw_minimap(self):
        c = self.minimap
        c.delete("all")
        if not self.present:
            return

        # Decimate if huge
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

        sz = max(1, int(scale))  # size per plotted chunk
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
        if not self.present:
            return
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
        # Coalesce many calls into one draw every REDRAW_MS
        if self._redraw_after_id is not None:
            # already scheduled
            return
        self._redraw_after_id = self.after(self.REDRAW_MS, self._do_redraw)

    def _do_redraw(self):
        self._redraw_after_id = None
        self._redraw_all()

    def _redraw_all(self):
        self.canvas.delete("all")
        self._draw_present_lod()
        self._draw_grid()
        self._draw_axis()
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


# ---------- Main window ----------
class ChunkMapWindow(tk.Toplevel):
    def __init__(self, root, world:Path, colors:Dict[str,str], is_light:bool,
                 dims:List[Path], initial_dim:str, preload:Optional[Dict], ui_settings:Dict):
        super().__init__(root)
        self.world = world
        self.colors = colors
        self.is_light = is_light
        self.title("Voider — Chunk Selector")
        self.geometry("1180x780")
        self.minsize(880, 640)

        # selection
        self.selection = Selection()
        if preload:
            self.selection.from_simple_json(preload)

        # styling
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background=colors["BG"])
        style.configure("TLabelframe", background=colors["BG"], foreground=colors["Text"])
        style.configure("TLabel", background=colors["BG"], foreground=colors["Text"])
        style.configure("TButton", background=colors["Panel"], foreground=colors["Text"])
        style.configure("Accent.TButton", background=colors["Accent"], foreground="#0b0e14" if not is_light else "#000000")
        style.configure("TCombobox", fieldbackground=colors["InputBG"], foreground=colors["InputText"])

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
        self.bind("<F1>", lambda _e=None: self._show_help())

        # hint
        helpf = ttk.Frame(self); helpf.pack(fill="x", padx=10)
        ttk.Label(helpf, text="LMB: select (drag for box) • Ctrl+click toggles set • RMB/MMB: pan • Wheel: zoom • Include=green, Exclude=red").pack(anchor="w")

        # dim list
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

        # JourneyMap sources: pass master=self to keep images in the same interpreter
        self._jm_by_dim: Dict[str, JMTileSource] = {}
        for rdir in dims:
            tag = _dim_tag(world, rdir)
            self._jm_by_dim[tag] = JMTileSource(world, tag, master=self)

        # canvas holder
        self.canvas_holder = ttk.Frame(self); self.canvas_holder.pack(fill="both", expand=True, padx=10, pady=10)
        self.current_canvas: Optional[ChunkCanvas] = None
        self.ui_settings = ui_settings or {}

        self.dim_combo.bind("<<ComboboxSelected>>", self._on_dim_changed)
        self._on_dim_changed()

        for w in (self.dim_combo, self.btn_inc, self.btn_exc, self.btn_clear, self.btn_load, self.btn_save, self.btn_help, self.btn_done):
            w.focus_set()

    def jm_layers_for_dim(self, tag:str) -> List[str]:
        src = self._jm_by_dim.get(tag)
        return src.available_layers() if src else []

    def _scan_present(self, rdir:Path) -> Set[Tuple[int,int]]:
        present=set()
        files = [p for p in rdir.iterdir() if p.suffix==".mca" and p.name.startswith("r.")]
        for mca in files:
            try:
                present.update(_iter_present_chunks_in_region(mca))
            except Exception:
                pass
        return present

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
        self.current_canvas = ChunkCanvas(self.canvas_holder, self, self.world, tag, present, self.selection, self.colors, per_dim, jm)
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
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _save_json(self):
        try:
            data = self.selection.to_simple_json()
            default_path = self.world/".voider-selection.json"
            Path(default_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            messagebox.showinfo("Saved", f"Wrote {default_path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _save_ui_settings(self):
        try:
            per_dim = self.ui_settings.get("per_dim", {})
            label = self.dim_combo.get()
            if label:
                rdir = self.dim_to_dir.get(label)
                tag = _dim_tag(self.world, rdir)
                per_dim[tag] = self.current_canvas.export_settings()
            self.ui_settings["per_dim"] = per_dim
            (self.world/".voider-ui.json").write_text(json.dumps(self.ui_settings, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _done(self):
        self._save_json()
        self._save_ui_settings()
        self.destroy()

    def _show_help(self):
        messagebox.showinfo("Chunk Selector — Help",
                            "• LMB: select chunk (drag to select a box)\n"
                            "• Ctrl+click: invert target (Include/Exclude)\n"
                            "• RMB/MMB: pan    • Wheel: zoom (throttled)\n"
                            "• Legend toggles: imagery, grid, axis, selected-only\n"
                            "• Minimap (bottom-right): click to recenter\n"
                            "• Settings persist in .voider-ui.json per dimension\n"
                            "• Selection is saved to .voider-selection.json\n")

# ---------- World discovery & selection (optional CLI helper) ----------
def _candidate_roots_windows() -> List[Path]:
    roots=set()
    user = Path(os.environ.get("USERPROFILE", Path.home()))
    appdata = Path(os.environ.get("APPDATA", user / "AppData" / "Roaming"))
    # Vanilla
    roots.add(appdata/".minecraft"/"saves")
    # Prism/MultiMC common variants
    roots.update(Path(p) for p in glob.glob(str(appdata/"PrismLauncher"/"instances" / "*" / "minecraft" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(appdata/".multimc"/"instances" / "*" / "minecraft" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(appdata/"MultiMC"/"instances" / "*" / "minecraft" / "saves")))
    # CurseForge / Overwolf
    roots.update(Path(p) for p in glob.glob(str(user/"curseforge"/"minecraft"/"Instances" / "*" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(user/"curseforge"/"minecraft"/"Instances" / "*" / "minecraft" / "saves")))
    # GD/ATLauncher
    roots.update(Path(p) for p in glob.glob(str(appdata/"gdlauncher_next"/"instances" / "*" / ".minecraft" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(appdata/"gdlauncher"/"instances" / "*" / ".minecraft" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(appdata/".atlauncher"/"instances" / "*" / "saves")))
    roots.update(Path(p) for p in glob.glob(str(appdata/".atlauncher"/"instances" / "*" / ".minecraft" / "saves")))
    return [r for r in roots if r.exists()]

def _read_world_name(level_dat:Path) -> str:
    try:
        from nbt import nbt  # optional dependency used by main app anyway
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

    # bounded slow scan
    deadline = time.time() + slow_scan_seconds
    drives=[]
    if os.name == "nt":
        for c in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            p = Path(f"{c}:/")
            if p.exists():
                drives.append(p)
    else:
        drives = [Path("/")]

    for d in drives:
        try:
            for root, dirs, _files in os.walk(d, topdown=True):
                bn = os.path.basename(root).lower()
                if bn in ("windows","program files","program files (x86)","programdata","system volume information","$recycle.bin"):
                    dirs[:] = []
                    continue
                if bn == "saves":
                    for lvl in Path(root).glob("*/level.dat"):
                        worlds.append((_read_world_name(lvl), lvl.parent))
                        if len(worlds) >= max_worlds:
                            return worlds
                if time.time() > deadline:
                    return worlds
        except Exception:
            pass
    return worlds

def choose_world_folder() -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
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
            var_sel["path"] = p
            win.destroy()
        elif p:
            messagebox.showerror("Invalid selection", "That folder does not contain level.dat")

    ttk.Button(btns, text="Browse…", command=on_browse).pack(side="left")
    ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right")
    ttk.Button(btns, text="OK", command=on_ok).pack(side="right", padx=(0,6))

    def on_dbl(_e):
        on_ok()
    tree.bind("<Double-1>", on_dbl)

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
    # Seed first dim to spawn area if we have nothing yet
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
        if not picked:
            sys.exit(0)
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
