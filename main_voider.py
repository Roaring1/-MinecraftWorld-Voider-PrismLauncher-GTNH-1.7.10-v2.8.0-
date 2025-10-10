#!/usr/bin/env python3
# 9/10/2025-1
# main_voider.py — GTNH / 1.7.x Voider (keep biomes) — FINAL
#
# Entry point (GUI-first + CLI):
# - A lightweight status/controls GUI opens immediately on launch (even if CLI args are provided).
# - If --world is omitted, we auto-discover worlds and offer a picker; then open the chunk selector UI.
# - The chunk selector returns include/exclude sets per dimension; we then void accordingly.
#
# Requirements:
#   pip install NBT Pillow
#   Put chunkmap_ui.py in the same folder (it must provide launch_chunk_selector()).
#
# Packaging tip (PyInstaller):
#   pyinstaller --noconfirm --onefile --windowed main_voider.py ^
#     --hidden-import PIL.Image --hidden-import PIL.ImageTk --hidden-import PIL.PngImagePlugin --hidden-import PIL._tkinter_finder
# (Pillow’s internal _tkinter_finder helps Tk bindings be found inside frozen apps.)

import os
import sys
import io
import time
import json
import struct
import zlib
import argparse
import threading
import queue
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set, Callable

# Third-party / sibling
from nbt import nbt
from chunkmap_ui import launch_chunk_selector

# ---------- Region / chunk I/O ----------
SECTOR_BYTES = 4096
ZLIB_ID = 2

def _read_header(fp):
    fp.seek(0)
    hdr = fp.read(8192)
    if len(hdr) != 8192:
        raise IOError(f"Region header bad size: {len(hdr)}")
    offsets = [struct.unpack(">I", hdr[i*4:(i+1)*4])[0] for i in range(1024)]
    stamps  = [struct.unpack(">I", hdr[4096+i*4:4096+(i+1)*4])[0] for i in range(1024)]
    return offsets, stamps

def _parse_loc(entry:int) -> Tuple[int,int]:
    return (entry >> 8) & 0xFFFFFF, entry & 0xFF

def _iter_chunk_locations(fp):
    offsets, _ = _read_header(fp)
    for idx, entry in enumerate(offsets):
        if entry == 0:
            continue
        off, cnt = _parse_loc(entry)
        if off and cnt:
            yield idx, off, cnt

def _read_chunk_payload(fp, sector_off:int) -> Optional[bytes]:
    fp.seek(sector_off * SECTOR_BYTES)
    lb = fp.read(4)
    if len(lb) < 4:
        return None
    (length,) = struct.unpack(">I", lb)
    comp_b = fp.read(1)
    if not comp_b:
        return None
    comp = comp_b[0]
    data = fp.read(length - 1)
    if comp == ZLIB_ID:
        return zlib.decompress(data)
    elif comp == 1:
        import gzip
        return gzip.decompress(data)
    elif comp == 3:
        return data
    return None

def _write_chunk_payload(fp, sector_off:int, sector_cnt:int, raw_nbt:bytes):
    body = bytes([ZLIB_ID]) + zlib.compress(raw_nbt)
    payload = struct.pack(">I", len(body)) + body
    need = (len(payload) + SECTOR_BYTES - 1) // SECTOR_BYTES
    if need > sector_cnt:
        fp.seek(0, os.SEEK_END)
        new_off = fp.tell() // SECTOR_BYTES
        pad = (-fp.tell()) % SECTOR_BYTES
        if pad:
            fp.write(b"\x00"*pad)
        fp.write(payload)
        pad2 = (-len(payload)) % SECTOR_BYTES
        if pad2:
            fp.write(b"\x00"*pad2)
        return new_off, need
    fp.seek(sector_off * SECTOR_BYTES)
    fp.write(payload)
    pad = (-len(payload)) % SECTOR_BYTES
    if pad:
        fp.write(b"\x00"*pad)
    return sector_off, sector_cnt

def _update_header(fp, idx:int, new_off:int, new_cnt:int, stamp:Optional[int]=None):
    if stamp is None:
        stamp = int(time.time())
    fp.seek(idx*4)
    fp.write(struct.pack(">I", ((new_off & 0xFFFFFF) << 8) | (new_cnt & 0xFF)))
    fp.seek(4096 + idx*4)
    fp.write(struct.pack(">I", stamp))

def _region_coords_from_name(path:Path) -> Tuple[int,int]:
    parts = path.stem.split(".")
    if len(parts) != 3:
        raise ValueError(f"Bad region file name: {path.name}")
    _, rx, rz = parts
    return int(rx), int(rz)

def _find_region_dirs(world:Path) -> List[Path]:
    out = []
    for p in world.rglob("region"):
        if p.is_dir() and p.parent.name not in ("entities", "poi"):
            out.append(p)
    out.sort()
    return out

def _dim_tag(world:Path, region_dir:Path) -> str:
    return str(region_dir.relative_to(world)).replace("\\", "/")

# ---------- Void logic ----------
def _has_entities(raw:bytes) -> Tuple[bool,bool]:
    try:
        nb = nbt.NBTFile(fileobj=io.BytesIO(raw))
        lvl = nb["Level"]
        te = ("TileEntities" in lvl and isinstance(lvl["TileEntities"], nbt.TAG_List) and len(lvl["TileEntities"])>0)
        en = ("Entities" in lvl and isinstance(lvl["Entities"], nbt.TAG_List) and len(lvl["Entities"])>0)
        return te, en
    except Exception:
        return (False, False)

def _is_void_chunk(raw:bytes) -> bool:
    try:
        nb = nbt.NBTFile(fileobj=io.BytesIO(raw))
        lvl = nb["Level"]
        if "Sections" not in lvl:
            return False
        sections = lvl["Sections"]
        saw = False
        for sec in sections:
            if not isinstance(sec, nbt.TAG_Compound):
                continue
            if "Blocks" not in sec:
                return False
            saw = True
            if any(b != 0 for b in sec["Blocks"].value):
                return False
        return saw
    except Exception:
        return False

def _get_biomes(raw:bytes) -> Optional[bytes]:
    try:
        nb = nbt.NBTFile(fileobj=io.BytesIO(raw))
        lvl = nb["Level"]
        if "Biomes" in lvl and isinstance(lvl["Biomes"], nbt.TAG_Byte_Array):
            b = bytes(lvl["Biomes"].value)
            if len(b) == 256:
                return b
    except Exception:
        pass
    return None

def _build_void_chunk(cx_abs:int, cz_abs:int,
                      biome_bytes:Optional[bytes],
                      mark_void:bool,
                      platform:Optional[Tuple[int,int,int,int,int,int]],
                      keep_entities:bool,
                      keep_tileentities:bool,
                      source_raw:Optional[bytes]) -> bytes:
    ents = nbt.TAG_List(name="Entities", type=nbt.TAG_Compound)
    tiles= nbt.TAG_List(name="TileEntities", type=nbt.TAG_Compound)
    ticks= nbt.TAG_List(name="TileTicks", type=nbt.TAG_Compound)
    if source_raw and (keep_entities or keep_tileentities):
        try:
            src = nbt.NBTFile(fileobj=io.BytesIO(source_raw))
            lvl = src["Level"]
            if keep_entities and "Entities" in lvl and isinstance(lvl["Entities"], nbt.TAG_List):
                ents = lvl["Entities"]
            if keep_tileentities and "TileEntities" in lvl and isinstance(lvl["TileEntities"], nbt.TAG_List):
                tiles = lvl["TileEntities"]
            if "TileTicks" in lvl and isinstance(lvl["TileTicks"], nbt.TAG_List):
                ticks = lvl["TileTicks"]
        except Exception:
            pass

    root = nbt.NBTFile()
    lvl = nbt.TAG_Compound(name="Level")
    root.tags.append(lvl)
    lvl.tags.extend([
        nbt.TAG_Int(name="xPos", value=cx_abs),
        nbt.TAG_Int(name="zPos", value=cz_abs),
        nbt.TAG_Long(name="LastUpdate", value=0),
        nbt.TAG_Long(name="InhabitedTime", value=0),
        nbt.TAG_Byte(name="TerrainPopulated", value=1),
        nbt.TAG_Byte(name="LightPopulated", value=1),
    ])

    hm = nbt.TAG_Int_Array(name="HeightMap"); hm.value = [0]*256; lvl.tags.append(hm)
    bio = nbt.TAG_Byte_Array(name="Biomes")
    bio.value = list(biome_bytes) if (biome_bytes and len(biome_bytes)==256) else [1]*256
    lvl.tags.append(bio)
    lvl.tags.extend([ents, tiles, ticks])

    sections = nbt.TAG_List(name="Sections", type=nbt.TAG_Compound); lvl.tags.append(sections)
    zeros4096 = [0]*4096; zeros2048 = [0]*2048; skylight_ff = [0xFF]*2048
    sec_blocks = []
    for y in range(16):
        sec = nbt.TAG_Compound()
        sec.tags.append(nbt.TAG_Byte(name="Y", value=y))
        blk = nbt.TAG_Byte_Array(name="Blocks");     blk.value = zeros4096.copy()
        dat = nbt.TAG_Byte_Array(name="Data");       dat.value = zeros2048.copy()
        sky = nbt.TAG_Byte_Array(name="SkyLight");   sky.value = skylight_ff.copy()
        bl  = nbt.TAG_Byte_Array(name="BlockLight"); bl.value  = zeros2048.copy()
        sec.tags.extend([blk, dat, sky, bl])
        sections.tags.append(sec)
        sec_blocks.append(blk)

    if mark_void:
        lvl.tags.append(nbt.TAG_Byte(name="Voider", value=1))

    if platform is not None:
        px, pz, py, size, bid, meta = platform
        center_cx = px // 16; center_cz = pz // 16
        if center_cx == cx_abs and center_cz == cz_abs and 0 <= py <= 255:
            half = size // 2
            x0, z0 = px - half, pz - half
            secY = py // 16; y_in = py % 16
            for dx in range(size):
                for dz in range(size):
                    wx = x0 + dx; wz = z0 + dz
                    if (wx // 16) == cx_abs and (wz // 16) == cz_abs:
                        xl = wx & 15; zl = wz & 15
                        idx = y_in + (zl * 16) + (xl * 256)
                        sec_blocks[secY].value[idx] = bid & 0xFF

    outb = io.BytesIO()
    root.write_file(buffer=outb)
    return outb.getvalue()

def _bbox_ok(cx_abs:int, cz_abs:int, bbox:Optional[Tuple[int,int,int,int]]) -> bool:
    if not bbox: return True
    x1, z1, x2, z2 = bbox
    minx, minz = cx_abs*16, cz_abs*16
    maxx, maxz = minx+15, minz+15
    return not (maxx < x1 or maxz < z1 or minx > x2 or minz > z2)

def _radius_ok(cx_abs:int, cz_abs:int, center:Optional[Tuple[int,int]], r:int) -> bool:
    if not center or r <= 0:
        return True
    cx, cz = center
    minx, minz = cx_abs*16, cz_abs*16
    maxx, maxz = minx+15, minz+15
    dx = 0 if minx <= cx <= maxx else min(abs(cx-minx), abs(cx-maxx))
    dz = 0 if minz <= cz <= maxz else min(abs(cz-minz), abs(cz-maxz))
    return dx*dx + dz*dz <= r*r

def _process_region(mca:Path,
                    bbox:Optional[Tuple[int,int,int,int]],
                    center:Optional[Tuple[int,int]],
                    radius_blocks:int,
                    include_void:bool,
                    platform:Optional[Tuple[int,int,int,int,int,int]],
                    keep_entities:bool,
                    keep_tileentities:bool,
                    skip_if_entities:bool,
                    skip_if_tileentities:bool,
                    mark_void:bool,
                    dry_run:bool,
                    strict:bool,
                    sel_include:Set[Tuple[int,int]],
                    sel_exclude:Set[Tuple[int,int]],
                    cancel_event:Optional[threading.Event]=None) -> Tuple[str,int,int]:
    rx, rz = _region_coords_from_name(mca)
    changed = 0; scanned = 0
    with open(mca, "r+b") as fp:
        for idx, off, cnt in _iter_chunk_locations(fp):
            if cancel_event is not None and cancel_event.is_set():
                break

            cx_local = idx % 32; cz_local = idx // 32
            cx_abs = rx*32 + cx_local; cz_abs = rz*32 + cz_local

            if not _bbox_ok(cx_abs, cz_abs, bbox):
                continue
            if not _radius_ok(cx_abs, cz_abs, center, radius_blocks):
                continue

            # include/exclude filtering before reading
            coord = (cx_abs, cz_abs)
            if coord in sel_exclude:
                continue
            if sel_include and coord not in sel_include:
                continue

            raw = _read_chunk_payload(fp, off)
            if not raw:
                if strict:
                    raise IOError(f"Bad payload at {mca.name}:{idx}")
                continue

            scanned += 1
            te, en = _has_entities(raw)
            if skip_if_entities and en:
                continue
            if skip_if_tileentities and te:
                continue

            already_void = _is_void_chunk(raw)
            platform_needed = False
            if platform:
                px, pz, py, size, bid, meta = platform
                if (cx_abs == px//16) and (cz_abs == pz//16):
                    platform_needed = True

            if already_void and not platform_needed and not include_void:
                continue

            if dry_run:
                changed += 1
                continue

            biome = _get_biomes(raw)
            new = _build_void_chunk(cx_abs, cz_abs, biome,
                                    mark_void, platform if platform_needed else None,
                                    keep_entities, keep_tileentities, raw)
            new_off, new_cnt = _write_chunk_payload(fp, off, cnt, new)
            _update_header(fp, idx, new_off, new_cnt)
            changed += 1
    return (mca.name, scanned, changed)

# ---------- Auto-discovery of worlds ----------
def _get_drive_letters_windows() -> List[Path]:
    import string
    try:
        from ctypes import windll
        bitmask = windll.kernel32.GetLogicalDrives()
        drives = []
        for i, letter in enumerate(string.ascii_uppercase):
            if bitmask & (1 << i):
                drives.append(Path(f"{letter}:/"))
        return drives
    except Exception:
        # fallback typical drives
        return [Path("C:/"), Path("D:/"), Path("E:/"), Path("F:/")]

def _candidate_roots() -> List[Path]:
    roots: List[Path] = []
    appdata = os.environ.get("APPDATA")
    localapp = os.environ.get("LOCALAPPDATA")
    userprofile = os.environ.get("USERPROFILE")

    # Vanilla .minecraft
    if userprofile:
        roots.append(Path(userprofile) / "AppData" / "Roaming" / ".minecraft" / "saves")

    # PrismLauncher / MultiMC
    if appdata:
        prism = Path(appdata)/"PrismLauncher"/"instances"
        multimc = Path(appdata)/"MultiMC"/"instances"
        for inst_root in (prism, multimc):
            if inst_root.exists():
                for inst in inst_root.iterdir():
                    roots.append(inst/".minecraft"/"saves")

    # ATLauncher
    if appdata:
        at = Path(appdata)/"ATLauncher"/"instances"
        if at.exists():
            for inst in at.iterdir():
                roots.append(inst/".minecraft"/"saves")

    # Technic
    if userprofile:
        tech = Path(userprofile)/"AppData"/"Roaming"/".technic"/"modpacks"
        if tech.exists():
            for pack in tech.iterdir():
                roots.append(pack/"saves")

    # CurseForge/Minecraft (old layout sometimes keeps .minecraft copies)
    if userprofile:
        curse = Path(userprofile)/"curseforge"/"minecraft"/"Instances"
        if curse.exists():
            for inst in curse.iterdir():
                roots.append(inst/"saves")

    # Add all drive roots for deep scan
    if os.name == "nt":
        roots.extend(_get_drive_letters_windows())
    else:
        # Unix/macOS: common mount points
        roots.extend([Path("/"), Path.home(), Path("/mnt"), Path("/media"), Path("/Volumes")])

    # Unique existing dirs only
    out: List[Path] = []
    seen = set()
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        if rp.exists() and str(rp) not in seen:
            seen.add(str(rp)); out.append(rp)
    return out

def _looks_like_world_dir(p:Path) -> bool:
    try:
        if not p.is_dir():
            return False
        if not (p/"level.dat").exists():
            return False
        # require at least one region dir with .mca files
        region_dirs = [d for d in p.rglob("region") if d.is_dir() and d.parent.name not in ("entities","poi")]
        for r in region_dirs:
            if any(f.suffix==".mca" and f.name.startswith("r.") for f in r.iterdir()):
                return True
        return False
    except Exception:
        return False

def _scan_for_worlds_under(root:Path, found:Dict[str,str], max_depth:int, cur_depth:int=0, status_cb:Optional[Callable[[str],None]]=None):
    if cur_depth > max_depth:
        return
    try:
        # Fast path: if this looks like a world, record and do NOT descend (saves time)
        if _looks_like_world_dir(root):
            found[str(root)] = str(root)
            return
    except Exception:
        pass
    # Descend a limited set of directories
    try:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            name = child.name.lower()
            # prune noisy/system dirs
            if name in ("windows","program files","program files (x86)","$recycle.bin","system volume information","intel","nvidia","programdata"):
                continue
            if name.startswith(".") and cur_depth > 0:
                continue
            if status_cb and cur_depth <= 2:
                status_cb(f"Scanning: {child}")
            _scan_for_worlds_under(child, found, max_depth, cur_depth+1, status_cb)
    except Exception:
        pass

def _cache_path() -> Path:
    # %LOCALAPPDATA%\GTNHVoider\worlds-cache.json (Windows), else ~/.gtnh_voider_cache.json
    local = os.environ.get("LOCALAPPDATA")
    if local:
        p = Path(local)/"GTNHVoider"
        p.mkdir(parents=True, exist_ok=True)
        return p/"worlds-cache.json"
    return Path.home()/".gtnh_voider_worlds_cache.json"

def discover_minecraft_worlds(force_rescan:bool=False, max_depth:int=4, status_cb:Optional[Callable[[str],None]]=None) -> Dict[str, str]:
    cache_file = _cache_path()
    if not force_rescan and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            # validate entries still exist
            data = {k:v for k,v in data.items() if Path(v).exists() and (Path(v)/"level.dat").exists()}
            if data:
                if status_cb: status_cb("Loaded cached worlds list.")
                return data
        except Exception:
            pass

    found: Dict[str,str] = {}
    # First scan known roots shallowly
    roots = _candidate_roots()
    if status_cb: status_cb(f"Scanning {len(roots)} roots for worlds…")
    for root in roots:
        try:
            if root.is_file():
                continue
            if (root.name.lower() == "saves") and root.exists():
                for w in root.iterdir():
                    if _looks_like_world_dir(w):
                        found[str(w)] = str(w)
            else:
                _scan_for_worlds_under(root, found, max_depth=max_depth, status_cb=status_cb)
        except Exception:
            continue

    # persist cache
    try:
        cache_file.write_text(json.dumps(found, indent=2), encoding="utf-8")
    except Exception:
        pass
    if status_cb: status_cb(f"World scan complete. Found {len(found)}.")
    return found

# ---------- Simple GUI chooser (legacy helper; still used by status UI) ----------
def choose_world_gui(discovered:Dict[str,str]) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        import tkinter.font as tkfont
    except Exception:
        return None

    root = tk.Tk()
    root.title("Select a Minecraft World")
    root.geometry("640x420")
    try:
        import tkinter.font as tkfont
        base = tkfont.nametofont("TkDefaultFont")
        family = "Segoe UI Variable" if "Variable" in tkfont.families() else "Segoe UI"
        base.configure(family=family, size=11)
        root.option_add("*Font", base)
    except Exception:
        pass

    sel_path = {"v": None}

    frm = ttk.Frame(root); frm.pack(fill="both", expand=True, padx=10, pady=10)
    ttk.Label(frm, text="Detected worlds (double-click to choose):").pack(anchor="w")
    cols = ("name","path")
    tree = ttk.Treeview(frm, columns=cols, show="headings", height=14)
    tree.heading("name", text="World")
    tree.heading("path", text="Path")
    tree.column("name", width=220)
    tree.column("path", width=380)
    tree.pack(fill="both", expand=True, pady=(6,6))

    rows = []
    for world_path in sorted(discovered.values(), key=str.lower):
        p = Path(world_path)
        name = p.name
        rows.append((name, world_path))
    for row in rows:
        tree.insert("", "end", values=row)

    def on_ok():
        item = tree.selection()
        if item:
            vals = tree.item(item[0], "values")
            sel_path["v"] = vals[1]
            root.destroy()
        else:
            from tkinter import messagebox
            messagebox.showwarning("Pick a world", "Select a world first (or use Browse…).")

    def on_browse():
        from tkinter import filedialog, messagebox
        d = filedialog.askdirectory(title="Pick a world folder (must contain level.dat)")
        if d and (Path(d)/"level.dat").exists():
            sel_path["v"] = d
            root.destroy()
        elif d:
            messagebox.showerror("Invalid world", "Selected folder does not contain level.dat")

    def on_dbl(_):
        on_ok()

    tree.bind("<Double-1>", on_dbl)

    btns = ttk.Frame(frm); btns.pack(fill="x", pady=(4,0))
    ttk.Button(btns, text="Browse…", command=on_browse).pack(side="left")
    ttk.Button(btns, text="OK", command=on_ok).pack(side="right")
    ttk.Button(btns, text="Cancel", command=root.destroy).pack(side="right", padx=(0,6))

    root.mainloop()
    return sel_path["v"]

# ---------- Backups ----------
def _backup_world_zip(world:Path):
    z = world.with_suffix(".void-backup.zip")
    if z.exists():
        print(f"Backup already exists: {z.name}")
        return z
    with ZipFile(z, "w", ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(world):
            for nm in files:
                p = Path(root)/nm
                zf.write(p, arcname=str(p.relative_to(world)))
    print(f"World backup created: {z.name}")
    return z

def _per_file_backup(world:Path):
    for rdir in _find_region_dirs(world):
        for mca in sorted(p for p in rdir.iterdir() if p.suffix==".mca" and p.name.startswith("r.")):
            bak = mca.with_suffix(mca.suffix + ".bak")
            if not bak.exists():
                bak.write_bytes(mca.read_bytes())

# ---------- Voider orchestration (thread-friendly) ----------
def run_voider(world:Path,
               sel_include_by_dim:Dict[str, Set[Tuple[int,int]]],
               sel_exclude_by_dim:Dict[str, Set[Tuple[int,int]]],
               args,
               *,
               progress_cb:Optional[Callable[[int,int,str],None]]=None,
               status_cb:Optional[Callable[[str],None]]=None,
               cancel_event:Optional[threading.Event]=None) -> Tuple[int,int]:
    """
    Returns: (total_scanned, total_changed)
    Optional callbacks:
      progress_cb(done_files, total_files, last_line)
      status_cb(text)
    cancel_event: if set, aborts work ASAP.
    """
    # Parse platform
    platform = None
    if args.platform_x is not None and args.platform_z is not None:
        size = int(args.platform_size)
        if size <= 0 or size % 2 == 0:
            raise SystemExit("Platform size must be a positive odd number.")
        platform = (args.platform_x, args.platform_z, args.platform_y,
                    size, args.platform_id, args.platform_meta)

    # Dim filtering
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    exclude = [s.strip() for s in args.exclude.split(",")] if args.exclude else None

    region_dirs = _find_region_dirs(world)
    if only:
        region_dirs = [d for d in region_dirs if _dim_tag(world, d) in only]
    if exclude:
        region_dirs = [d for d in region_dirs if _dim_tag(world, d) not in exclude]

    bbox = None
    if args.bbox:
        a = [int(x.strip()) for x in args.bbox.split(",")]
        if len(a) != 4:
            raise SystemExit("bbox must be: minX,minZ,maxX,maxZ")
        x1, z1, x2, z2 = a
        bbox = (min(x1,x2), min(z1,z2), max(x1,x2), max(z1,z2))
    center = (args.center_x, args.center_z) if (args.center_x is not None and args.center_z is not None) else None

    total_scanned = 0
    total_changed = 0

    # Optional backups
    if args.backup and not args.dry_run:
        if status_cb: status_cb("Creating world ZIP backup…")
        _backup_world_zip(world)
    if args.per_file_backup and not args.dry_run:
        if status_cb: status_cb("Creating per-file .bak copies…")
        _per_file_backup(world)

    # Wait for session.lock if asked
    if args.wait_for_unlock and args.wait_for_unlock > 0:
        lock = world/"session.lock"
        start = time.time()
        if status_cb: status_cb(f"Waiting for session.lock (up to {args.wait_for_unlock}s)…")
        while lock.exists():
            if time.time() - start > args.wait_for_unlock:
                print("session.lock still present; aborting to avoid corruption.")
                return (0,0)
            time.sleep(1)

    # Enumerate files up front for progress
    file_list: List[Tuple[str, Path]] = []
    for rdir in region_dirs:
        dimtag = _dim_tag(world, rdir)
        files = sorted(p for p in rdir.iterdir() if p.suffix==".mca" and p.name.startswith("r."))
        for mca in files:
            file_list.append((dimtag, mca))
    total_files = len(file_list)
    if total_files == 0:
        if status_cb: status_cb("No region files found after filters.")
        return (0, 0)

    if status_cb:
        status_cb(f"Processing {total_files} region file(s) across {len(set(d for d,_ in file_list))} dimension(s)…")

    # Worker for a single file
    def _work(dimtag:str, mca:Path):
        inc_set = sel_include_by_dim.get(dimtag, set())
        exc_set = sel_exclude_by_dim.get(dimtag, set())
        nm, scanned, changed = _process_region(
            mca,
            bbox=bbox,
            center=center,
            radius_blocks=int(args.radius_blocks),
            include_void=bool(args.include_void),
            platform=platform,
            keep_entities=bool(args.keep_entities),
            keep_tileentities=bool(args.keep_tileentities),
            skip_if_entities=bool(args.skip_if_entities),
            skip_if_tileentities=bool(args.skip_if_tileentities),
            mark_void=bool(args.mark_void),
            dry_run=bool(args.dry_run),
            strict=False,
            sel_include=inc_set,
            sel_exclude=exc_set,
            cancel_event=cancel_event
        )
        line = f"{'would void' if args.dry_run else 'voided'}: {nm} ({changed} of {scanned} chunks)"
        print(f"  {line}")
        return scanned, changed, dimtag, line

    # Sequential if workers<=1; else parallel
    done_files = 0
    if args.workers and int(args.workers) > 1:
        max_workers = max(1, int(args.workers))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_work, dimtag, mca) for (dimtag, mca) in file_list]
            for fut in as_completed(futs):
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    scanned, changed, dimtag, line = fut.result()
                    total_scanned += scanned; total_changed += changed
                    done_files += 1
                    if progress_cb: progress_cb(done_files, total_files, line)
                except Exception as e:
                    print(f"  [skip] worker exception - {e!r}", file=sys.stderr)
    else:
        # Keep original dim-by-dim banner output
        grouped: Dict[str, List[Path]] = {}
        for dimtag, mca in file_list:
            grouped.setdefault(dimtag, []).append(mca)
        for dimtag, files in grouped.items():
            print(f"Processing {len(files)} region files in '{dimtag}'...")
            for mca in files:
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    scanned, changed, _, line = _work(dimtag, mca)
                    total_scanned += scanned; total_changed += changed
                    done_files += 1
                    if progress_cb: progress_cb(done_files, total_files, line)
                except Exception as e:
                    print(f"  [skip] {mca.name} - {e!r}", file=sys.stderr)

    return total_scanned, total_changed

# ---------- Status GUI (always visible) ----------
class _TeeIO:
    """Redirect stdout/stderr into a queue while still printing to real console if present."""
    def __init__(self, stream, q:queue.Queue, tag:str):
        self._stream = stream
        self._q = q
        self._tag = tag
        self._lock = threading.Lock()
    def write(self, s):
        with self._lock:
            try:
                self._stream.write(s)
            except Exception:
                pass
            # push full lines to UI
            for line in s.splitlines(True):
                self._q.put((self._tag, line))
    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

class VoiderStatusUI:
    def __init__(self, args):
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog
        import tkinter.font as tkfont

        self.args = args
        self.tk = tk.Tk()
        self.tk.title("GTNH Voider — Status & Controls")
        self.tk.geometry("900x620")
        try:
            base = tkfont.nametofont("TkDefaultFont")
            family = "Segoe UI Variable" if "Variable" in tkfont.families() else "Segoe UI"
            base.configure(family=family, size=11)
            self.tk.option_add("*Font", base)
        except Exception:
            pass

        self._q = queue.Queue()
        # Tee stdout/err
        self._orig_out, self._orig_err = sys.stdout, sys.stderr
        sys.stdout = _TeeIO(sys.stdout, self._q, "OUT")
        sys.stderr = _TeeIO(sys.stderr, self._q, "ERR")

        # State
        self.world_path: Optional[str] = args.world
        self.cancel_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        # Top controls
        top = ttk.Frame(self.tk); top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="World:").pack(side="left")
        self.world_var = tk.StringVar(value=self.world_path or "")
        self.world_entry = ttk.Entry(top, textvariable=self.world_var, width=60)
        self.world_entry.pack(side="left", padx=(6,6))
        ttk.Button(top, text="Browse…", command=self._browse_world).pack(side="left", padx=(0,8))
        ttk.Button(top, text="Discover…", command=self._discover_worlds).pack(side="left", padx=(0,8))
        ttk.Button(top, text="Chunk Selector…", command=self._open_selector).pack(side="left")

        # Options row
        opts = ttk.Frame(self.tk); opts.pack(fill="x", padx=10, pady=(0,8))
        self.dry = tk.BooleanVar(value=bool(args.dry_run))
        self.include_void = tk.BooleanVar(value=bool(args.include_void))
        self.keep_entities = tk.BooleanVar(value=bool(args.keep_entities))
        self.keep_tes = tk.BooleanVar(value=bool(args.keep_tileentities))
        self.skip_entities = tk.BooleanVar(value=bool(args.skip_if_entities))
        self.skip_tes = tk.BooleanVar(value=bool(args.skip_if_tileentities))
        self.mark_void = tk.BooleanVar(value=bool(args.mark_void))
        self.workers = tk.IntVar(value=int(args.workers or 1))
        for (label, var) in (
            ("Dry run", self.dry), ("Include already void", self.include_void),
            ("Keep Entities", self.keep_entities), ("Keep TileEntities", self.keep_tes),
            ("Skip if Entities", self.skip_entities), ("Skip if TileEntities", self.skip_tes),
            ("Mark Void tag", self.mark_void),
        ):
            ttk.Checkbutton(opts, text=label, variable=var).pack(side="left", padx=(0,10))
        ttk.Label(opts, text="Workers:").pack(side="left", padx=(10,4))
        self.worker_spin = ttk.Spinbox(opts, from_=1, to=16, width=4, textvariable=self.workers)
        self.worker_spin.pack(side="left")

        # Status + progress
        statf = ttk.Frame(self.tk); statf.pack(fill="x", padx=10, pady=(0,6))
        ttk.Label(statf, text="Status:").pack(side="left")
        self.status_var = tk.StringVar(value="Ready.")
        self.status_lbl = ttk.Label(statf, textvariable=self.status_var)
        self.status_lbl.pack(side="left", padx=(6,0))
        self.progress = ttk.Progressbar(self.tk, mode="determinate")
        self.progress.pack(fill="x", padx=10)

        # Buttons
        btns = ttk.Frame(self.tk); btns.pack(fill="x", padx=10, pady=6)
        self.btn_start = ttk.Button(btns, text="Start Voiding", command=self._start_voiding)
        self.btn_start.pack(side="left")
        self.btn_cancel = ttk.Button(btns, text="Cancel", command=self._cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=(8,0))
        ttk.Button(btns, text="Save Log…", command=self._save_log).pack(side="right")

        # Console dock (toggleable)
        dockh = ttk.Frame(self.tk); dockh.pack(fill="x", padx=10, pady=(4,0))
        ttk.Label(dockh, text="Console (toggle ⇵)").pack(side="left")
        self.console_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(dockh, variable=self.console_visible, command=self._toggle_console).pack(side="left")

        self.console = tk.Text(self.tk, height=16, wrap="none")
        self.console.configure(state="disabled")
        # Console input
        cinf = ttk.Frame(self.tk); cinf.pack(fill="x", padx=10, pady=(0,8))
        ttk.Label(cinf, text="> ").pack(side="left")
        self.console_in = ttk.Entry(cinf)
        self.console_in.pack(side="left", fill="x", expand=True)
        self.console_in.bind("<Return>", self._on_command_enter)
        cinf.pack_forget()  # hidden until console visible
        self._console_frame = (self.console, cinf)

        # Log viewer (always visible)
        logf = ttk.LabelFrame(self.tk, text="Log")
        logf.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.logbox = tk.Text(logf, wrap="none")
        self.logbox.pack(fill="both", expand=True)
        self.logbox.configure(state="disabled")

        # pump queue
        self._pump()

        # If launched with args.world, keep GUI first, but prefill field.
        # We do NOT auto-run; user clicks Start after reviewing options.

    # ---- console helpers ----
    def _toggle_console(self):
        vis = self.console_visible.get()
        if vis:
            self._console_frame[0].pack(fill="both", expand=False, padx=10, pady=(0,4))
            self._console_frame[1].pack(fill="x", padx=10, pady=(0,8))
            self.console_in.focus_set()
        else:
            self._console_frame[0].pack_forget()
            self._console_frame[1].pack_forget()

    def _append_console(self, s:str):
        self.console.configure(state="normal")
        self.console.insert("end", s)
        self.console.see("end")
        self.console.configure(state="disabled")

    def _on_command_enter(self, _ev=None):
        cmd = self.console_in.get().strip()
        self.console_in.delete(0, "end")
        if not cmd:
            return
        self._append_console(f"> {cmd}\n")
        # tiny shorthand commands
        if cmd.lower() in ("help","?"):
            self._append_console("commands: help, rescan, select, start, cancel, dryon, dryoff, workers N\n")
        elif cmd.lower()=="rescan":
            self._discover_worlds()
        elif cmd.lower()=="select":
            self._open_selector()
        elif cmd.lower()=="start":
            self._start_voiding()
        elif cmd.lower()=="cancel":
            self._cancel()
        elif cmd.lower()=="dryon":
            self.dry.set(True)
        elif cmd.lower()=="dryoff":
            self.dry.set(False)
        elif cmd.lower().startswith("workers "):
            try:
                n = int(cmd.split()[1])
                self.workers.set(max(1, min(32, n)))
            except Exception:
                self._append_console("bad workers value\n")
        else:
            self._append_console("unknown command\n")

    # ---- file/selection helpers ----
    def _browse_world(self):
        from tkinter import filedialog, messagebox
        p = filedialog.askdirectory(title="Pick a world folder (must contain level.dat)")
        if p:
            if (Path(p)/"level.dat").exists():
                self.world_var.set(p)
            else:
                messagebox.showerror("Invalid world", "Selected folder does not contain level.dat")

    def _discover_worlds(self):
        def bg():
            self._set_status("Discovering worlds…")
            found = discover_minecraft_worlds(force_rescan=True, status_cb=self._set_status)
            # show picker
            pick = choose_world_gui(found)
            if pick:
                self.world_var.set(pick)
            self._set_status("Discovery done.")
        threading.Thread(target=bg, daemon=True).start()

    def _open_selector(self):
        # Pass through initial_dim/limit if provided via args; they remain optional.
        try:
            sel = launch_chunk_selector(
                self.world_var.get(),
                initial_dim=self.args.initial_dim,
                preload=None,
                limit_tags=[s.strip() for s in self.args.limit_tags.split(",")] if self.args.limit_tags else None
            )
            # nothing to do here; main() will reload selection JSON from world during run.
            self._set_status("Chunk selection saved. Ready to void.")
        except SystemExit:
            self._set_status("Chunk selector aborted.")
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Chunk selector failed", str(e))

    # ---- run/cancel ----
    def _start_voiding(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        wp = self.world_var.get().strip()
        if not wp or not (Path(wp)/"level.dat").exists():
            self._set_status("Pick a valid world first.")
            return
        # refresh args from UI toggles
        self.args.world = wp
        self.args.dry_run = self.dry.get()
        self.args.include_void = self.include_void.get()
        self.args.keep_entities = self.keep_entities.get()
        self.args.keep_tileentities = self.keep_tes.get()
        self.args.skip_if_entities = self.skip_entities.get()
        self.args.skip_if_tileentities = self.skip_tes.get()
        self.args.mark_void = self.mark_void.get()
        self.args.workers = int(self.workers.get())

        self.cancel_event.clear()
        self.btn_start.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.progress.configure(value=0, maximum=100)
        self._set_status("Preparing…")

        def run():
            try:
                world = Path(self.args.world)
                # Load selection JSON generated by chunk UI (if present)
                sel_path = world/".voider-selection.json"
                sel_include: Dict[str, Set[Tuple[int,int]]] = {}
                sel_exclude: Dict[str, Set[Tuple[int,int]]] = {}
                if sel_path.exists():
                    try:
                        data = json.loads(sel_path.read_text(encoding="utf-8"))
                        for dimtag, d in data.items():
                            sel_include[dimtag] = set(tuple(x) for x in d.get("include", []))
                            sel_exclude[dimtag] = set(tuple(x) for x in d.get("exclude", []))
                        print(f"Loaded selection from {sel_path.name}")
                    except Exception as e:
                        print(f"[warn] Failed to read selection JSON: {e!r}")

                # progress aggregator
                prog_lock = threading.Lock()
                totals = {"files_total": 1, "files_done": 0}
                def prog_cb(done, total, line):
                    with prog_lock:
                        totals["files_total"] = total
                        totals["files_done"] = done
                        pct = int((done/total)*100) if total else 0
                    self.progress.configure(value=pct)
                    self._set_status(f"{pct}% — {line}")

                total_scanned, total_changed = run_voider(
                    world, sel_include, sel_exclude, self.args,
                    progress_cb=prog_cb, status_cb=self._set_status, cancel_event=self.cancel_event
                )
                msg = (f"Would void {total_changed} chunk(s) out of {total_scanned} scanned. Biomes preserved."
                       if self.args.dry_run else
                       f"Done. Voided {total_changed} chunk(s) out of {total_scanned} scanned. Biomes preserved.")
                print(msg)
                self._set_status("Idle.")
            finally:
                self.btn_start.configure(state="normal")
                self.btn_cancel.configure(state="disabled")

        self.worker_thread = threading.Thread(target=run, daemon=True)
        self.worker_thread.start()

    def _cancel(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.cancel_event.set()
            self._set_status("Cancelling…")

    # ---- status/log plumbing ----
    def _set_status(self, s:str):
        self.status_var.set(s)

    def _save_log(self):
        from tkinter import filedialog, messagebox
        p = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text","*.txt"), ("All","*.*")])
        if not p:
            return
        try:
            data = self.logbox.get("1.0", "end-1c")
            Path(p).write_text(data, encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _pump(self):
        try:
            while True:
                tag, line = self._q.get_nowait()
                self.logbox.configure(state="normal")
                self.logbox.insert("end", line)
                self.logbox.see("end")
                self.logbox.configure(state="disabled")
                if self.console_visible.get():
                    self._append_console(line)
        except queue.Empty:
            pass
        self.tk.after(60, self._pump)  # ~16 FPS

    def run_forever(self) -> None:
        self.tk.mainloop()

# ---------- CLI / GUI glue ----------
def main() -> bool:
    ap = argparse.ArgumentParser(description="GTNH 1.7.x Voider (keep biomes) — GUI + CLI")
    ap.add_argument("world", nargs="?", help="Path to your world folder (contains level.dat)")
    ap.add_argument("--initial-dim", help="Dimension tag to show first (e.g. DIM-1/region)")
    ap.add_argument("--limit-tags", help="Comma-separated dims to allow (e.g. region,DIM-1/region)")
    ap.add_argument("--only", help="Only these dims (comma) for processing (same namespace as limit-tags)")
    ap.add_argument("--exclude", help="Dimensions to skip (comma)")
    ap.add_argument("--bbox", type=str, help="minX,minZ,maxX,maxZ")
    ap.add_argument("--center-x", type=int, help="Center block X for radius filter")
    ap.add_argument("--center-z", type=int, help="Center block Z for radius filter")
    ap.add_argument("--radius-blocks", type=int, default=0, help="Radius in blocks around center")
    ap.add_argument("--platform-x", type=int, help="Spawn platform X block coord")
    ap.add_argument("--platform-y", type=int, default=64, help="Spawn platform Y")
    ap.add_argument("--platform-z", type=int, help="Spawn platform Z block coord")
    ap.add_argument("--platform-size", type=int, default=3, help="Platform edge length (odd)")
    ap.add_argument("--platform-id", type=int, default=7, help="Block ID of platform")
    ap.add_argument("--platform-meta", type=int, default=0, help="Block meta value")
    ap.add_argument("--backup", action="store_true", help="Backup world zip before voiding")
    ap.add_argument("--per-file-backup", action="store_true", help="Backup each region file before modify")
    ap.add_argument("--dry-run", action="store_true", help="Do not modify files; just report")
    ap.add_argument("--include-void", action="store_true", help="Rewrite already void chunks too")
    ap.add_argument("--wait-for-unlock", type=int, default=0, help="Seconds to wait if session.lock present")
    ap.add_argument("--workers", type=int, default=1, help="Number of worker threads (>=1)")
    ap.add_argument("--skip-if-entities", action="store_true", help="Skip chunks with Entities")
    ap.add_argument("--skip-if-tileentities", action="store_true", help="Skip chunks with TileEntities")
    ap.add_argument("--keep-entities", action="store_true", help="Preserve Entities (copy through)")
    ap.add_argument("--keep-tileentities", action="store_true", help="Preserve TileEntities (copy through)")
    ap.add_argument("--mark-void", action="store_true", help="Mark voided chunks with a Voider tag")
    ap.add_argument("--force-rescan", action="store_true", help="Force re-scan of drives for worlds (ignore cache)")

    args = ap.parse_args()

    # Always show the Status UI first.
    try:
        ui = VoiderStatusUI(args)
    except Exception:
        # fallback: if Tk can't initialize (headless), run legacy CLI
        world_path: Optional[str] = args.world
        if not world_path:
            print("Scanning for worlds… (use --force-rescan to refresh cache)")
            discovered = discover_minecraft_worlds(force_rescan=bool(args.force_rescan))
            world_path = choose_world_gui(discovered)
            if not world_path:
                try:
                    w = input("Enter path to world folder (must contain level.dat), or leave blank to exit: ").strip()
                except EOFError:
                    w = ""
                if not w:
                    return False
                world_path = w

        world = Path(world_path)
        if not (world/"level.dat").exists():
            print("ERROR: level.dat not found in the selected folder.")
            return False

        # Open chunk selection UI
        try:
            sel = launch_chunk_selector(
                str(world),
                initial_dim=args.initial_dim,
                preload=None,
                limit_tags=[s.strip() for s in args.limit_tags.split(",")] if args.limit_tags else None
            )
        except SystemExit:
            return False
        except Exception as e:
            print(f"Chunk selector failed: {e}", file=sys.stderr)
            return False

        # Flatten include/exclude maps
        sel_include: Dict[str, Set[Tuple[int,int]]] = {}
        sel_exclude: Dict[str, Set[Tuple[int,int]]] = {}
        for dimtag, d in sel.items():
            sel_include[dimtag] = set(tuple(x) for x in d.get("include", []))
            sel_exclude[dimtag] = set(tuple(x) for x in d.get("exclude", []))

        # Run voider
        total_scanned, total_changed = run_voider(world, sel_include, sel_exclude, args)
        print(f"{'Would void' if args.dry_run else 'Done.'} Voided {total_changed} chunk(s) out of {total_scanned} scanned. Biomes preserved.")
        return True

    # Normal GUI loop
    ui.run_forever()
    return True

if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except Exception:
        # Show a GUI dialog if possible, and always print traceback
        import traceback
        tb = traceback.format_exc()
        try:
            import tkinter as tk
            from tkinter import messagebox
            tk.Tk().withdraw()
            messagebox.showerror("Voider crashed", tb)
        except Exception:
            pass
        print(tb, file=sys.stderr)
        sys.exit(1)
