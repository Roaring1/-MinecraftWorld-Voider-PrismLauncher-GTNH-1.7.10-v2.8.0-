#!/usr/bin/env python3
# 5/10/2025-16
# GTNH / 1.7.x Re-runnable Voider (keeps biomes) — FINAL
# one-file tool (GUI + CLI) that rewrites classic Anvil chunks (1.2–1.12 era, incl. 1.7.10)
# to pure air with biomes preserved. Safe to re-run; can verify, dry-run, set area filters,
# optional tiny spawn platform, multi-threaded, with Windows-native theming.
#
# DEP: pip install NBT   (only for .py usage; the .exe bundle already includes it)
#
# Refs (kept short; these informed the design/contrast/theme bits):
# - WCAG 2.1 contrast thresholds (normal text 4.5:1; UI components ≥3:1): https://www.w3.org/TR/WCAG21/           # (see also: https://webaim.org/resources/contrastchecker/)
# - Windows / Fluent design & typography (Segoe UI Variable): https://learn.microsoft.com/windows/apps/design/    # typography: https://learn.microsoft.com/windows/apps/design/signature-experiences/typography
# - ttk style maps (state-based colors): https://docs.python.org/3/library/tkinter.ttk.html
# - Windows accent color API: DwmGetColorizationColor: https://learn.microsoft.com/windows/win32/api/dwmapi/nf-dwmapi-dwmgetcolorizationcolor
#
# Notes:
# • CLOSE Minecraft/Prism/MCEdit. The GUI waits ~30s for session.lock; you can force proceed after.
# • By default we skip already-void chunks. You can “Rewrite already-void” to add/refresh a platform, etc.
# • Verified on Python 3.10–3.13 (ttk). NBT lib: twoolie/NBT 1.5.x.

import argparse, io, os, struct, time, zlib, json, threading, concurrent.futures, sys, math
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from zipfile import ZipFile, ZIP_DEFLATED

# ---------- NBT (classic) ----------
try:
    from nbt import nbt  # twoolie/NBT
except Exception:
    nbt = None

SECTOR_BYTES = 4096
ZLIB = 2

# ---------- Region file helpers (spec-accurate) ----------
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
    if comp == ZLIB:
        return zlib.decompress(data)
    elif comp == 1:
        import gzip
        return gzip.decompress(data)
    elif comp == 3:
        return data
    return None

def _write_chunk_payload(fp, sector_off:int, sector_cnt:int, raw_nbt:bytes):
    body = bytes([ZLIB]) + zlib.compress(raw_nbt)
    payload = struct.pack(">I", len(body)) + body
    need = (len(payload) + SECTOR_BYTES - 1) // SECTOR_BYTES
    if need > sector_cnt:
        fp.seek(0, os.SEEK_END)
        new_off = fp.tell() // SECTOR_BYTES
        pad = (-fp.tell()) % SECTOR_BYTES
        if pad: fp.write(b"\x00"*pad)
        fp.write(payload)
        pad2 = (-len(payload)) % SECTOR_BYTES
        if pad2: fp.write(b"\x00"*pad2)
        return new_off, need
    fp.seek(sector_off * SECTOR_BYTES)
    fp.write(payload)
    pad = (-len(payload)) % SECTOR_BYTES
    if pad: fp.write(b"\x00"*pad)
    return sector_off, sector_cnt

def _update_header(fp, idx:int, new_off:int, new_cnt:int, stamp:Optional[int]=None):
    if stamp is None: stamp = int(time.time())
    fp.seek(idx*4); fp.write(struct.pack(">I", ((new_off & 0xFFFFFF)<<8) | (new_cnt & 0xFF)))
    fp.seek(4096 + idx*4); fp.write(struct.pack(">I", stamp))

# ---------- World / dimensions ----------
def _region_coords_from_name(path:Path) -> Tuple[int,int]:
    try:
        _, rx, rz = path.stem.split(".")
        return int(rx), int(rz)
    except Exception:
        raise ValueError(f"Bad region filename: {path.name}")

def _find_region_dirs(world:Path) -> List[Path]:
    out = []
    for p in world.rglob("region"):
        if p.is_dir() and p.parent.name not in ("entities","poi"):
            out.append(p)
    out.sort()
    return out

def _dim_tag(world:Path, rdir:Path) -> str:
    return str(rdir.relative_to(world)).replace("\\","/")

def _select_region_dirs(world:Path, only:Optional[List[str]], exclude:Optional[List[str]]) -> List[Path]:
    dirs = _find_region_dirs(world)
    if only:
        want = set(s.strip().lower() for s in only)
        dirs = [d for d in dirs if _dim_tag(world,d).lower() in want]
    if exclude:
        ban = set(s.strip().lower() for s in exclude)
        dirs = [d for d in dirs if _dim_tag(world,d).lower() not in ban]
    return dirs

# ---------- Classic 1.7.x chunk ops ----------
def _get_biomes(raw:bytes) -> Optional[bytes]:
    try:
        nbf = nbt.NBTFile(fileobj=io.BytesIO(raw))
        lvl = nbf["Level"]
        if "Biomes" in lvl and isinstance(lvl["Biomes"], nbt.TAG_Byte_Array):
            b = bytes(lvl["Biomes"].value)
            return b if len(b)==256 else None
    except Exception:
        pass
    return None

def _has_entities(raw:bytes) -> Tuple[bool,bool]:
    te = en = False
    try:
        nbf = nbt.NBTFile(fileobj=io.BytesIO(raw))
        lvl = nbf["Level"]
        te = ("TileEntities" in lvl and isinstance(lvl["TileEntities"], nbt.TAG_List) and len(lvl["TileEntities"])>0)
        en = ("Entities" in lvl and isinstance(lvl["Entities"], nbt.TAG_List) and len(lvl["Entities"])>0)
    except Exception:
        pass
    return te, en

def _is_void_chunk(raw:bytes) -> bool:
    try:
        nbf = nbt.NBTFile(fileobj=io.BytesIO(raw))
        lvl = nbf["Level"]
        if "Sections" not in lvl: 
            return False
        sections = lvl["Sections"]
        saw_classic = False
        for sec in sections:
            if not isinstance(sec, nbt.TAG_Compound): 
                continue
            if "Blocks" not in sec: 
                return False
            saw_classic = True
            if any(b != 0 for b in sec["Blocks"].value):
                return False
        return saw_classic
    except Exception:
        return False

def _idx_yzx(xl:int, y_in:int, zl:int) -> int:
    # YZX: idx = y + z*16 + x*256
    return y_in + (zl*16) + (xl*256)

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
            if keep_entities and "Entities" in lvl and isinstance(lvl["Entities"], nbt.TAG_List): ents = lvl["Entities"]
            if keep_tileentities and "TileEntities" in lvl and isinstance(lvl["TileEntities"], nbt.TAG_List): tiles = lvl["TileEntities"]
            if "TileTicks" in lvl and isinstance(lvl["TileTicks"], nbt.TAG_List): ticks = lvl["TileTicks"]
        except Exception:
            pass

    root = nbt.NBTFile()
    lvl = nbt.TAG_Compound(name="Level"); root.tags.append(lvl)
    lvl.tags.extend([
        nbt.TAG_Int(name="xPos", value=cx_abs),
        nbt.TAG_Int(name="zPos", value=cz_abs),
        nbt.TAG_Long(name="LastUpdate", value=0),
        nbt.TAG_Long(name="InhabitedTime", value=0),
        nbt.TAG_Byte(name="TerrainPopulated", value=1),
        nbt.TAG_Byte(name="LightPopulated", value=1),
    ])

    hm = nbt.TAG_Int_Array(name="HeightMap"); hm.value = [0]*256; lvl.tags.append(hm)
    bio = nbt.TAG_Byte_Array(name="Biomes"); bio.value = list(biome_bytes if (biome_bytes and len(biome_bytes)==256) else bytes([1]*256)); lvl.tags.append(bio)
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
        px, pz, py, size, block_id, block_meta = platform
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
                        idx = _idx_yzx(xl, y_in, zl)
                        sec_blocks[secY].value[idx] = block_id & 0xFF

    out = io.BytesIO(); root.write_file(buffer=out); return out.getvalue()

# ---------- Area filters ----------
def _bbox_ok(cx_abs:int, cz_abs:int, bbox:Optional[Tuple[int,int,int,int]]) -> bool:
    if not bbox: return True
    x1,z1,x2,z2 = bbox
    minx, minz = cx_abs*16, cz_abs*16
    maxx, maxz = minx+15,  minz+15
    return not (maxx < x1 or maxz < z1 or minx > x2 or minz > z2)

def _radius_ok(cx_abs:int, cz_abs:int, center:Optional[Tuple[int,int]], r:int) -> bool:
    if not center or r <= 0: return True
    cx, cz = center
    minx, minz = cx_abs*16, cz_abs*16
    maxx, maxz = minx+15,  minz+15
    dx = 0 if minx <= cx <= maxx else min(abs(cx - minx), abs(cx - maxx))
    dz = 0 if minz <= cz <= maxz else min(abs(cz - minz), abs(cz - maxz))
    return dx*dx + dz*dz <= r*r

# ---------- Processing ----------
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
                    strict:bool) -> Tuple[str,int,int]:
    rx, rz = _region_coords_from_name(mca)
    changed = 0; scanned = 0
    with open(mca, "r+b") as fp:
        for idx, off, cnt in _iter_chunk_locations(fp):
            cx_local, cz_local = idx % 32, idx // 32
            cx_abs, cz_abs = rx*32 + cx_local, rz*32 + cz_local
            if not _bbox_ok(cx_abs, cz_abs, bbox):   continue
            if not _radius_ok(cx_abs, cz_abs, center, radius_blocks): continue

            raw = _read_chunk_payload(fp, off)
            if not raw:
                if strict: raise IOError(f"Bad payload at {mca.name}:{idx}")
                continue

            scanned += 1
            te, en = _has_entities(raw)
            if skip_if_tileentities and te: continue
            if skip_if_entities and en:     continue

            already_void = _is_void_chunk(raw)
            needs_platform = False
            if platform is not None:
                px, pz, py, size, bid, meta = platform
                if (px // 16) == cx_abs and (pz // 16) == cz_abs:
                    needs_platform = True

            if already_void and not needs_platform and not include_void:
                continue

            if dry_run:
                changed += 1
                continue

            biomes = _get_biomes(raw)
            new_raw = _build_void_chunk(cx_abs, cz_abs, biomes, mark_void,
                                        platform if needs_platform else None,
                                        keep_entities, keep_tileentities, raw)
            new_off, new_cnt = _write_chunk_payload(fp, off, cnt, new_raw)
            _update_header(fp, idx, new_off, new_cnt)
            changed += 1
    return (mca.name, scanned, changed)

# ---------- Backups & lock ----------
def _backup_world(world:Path):
    z = world.with_suffix(".void-backup.zip")
    if z.exists(): return z
    with ZipFile(z, "w", ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(world):
            for name in files:
                fp = Path(root)/name
                zipf.write(fp, arcname=str(fp.relative_to(world)))
    return z

def _backup_mca(mca:Path):
    bak = mca.with_suffix(mca.suffix + ".bak")
    if not bak.exists(): bak.write_bytes(mca.read_bytes())
    return bak

def _wait_for_unlock(world:Path, timeout:int, on_tick=None):
    lock = world/"session.lock"
    start = time.time()
    while lock.exists():
        if timeout and (time.time()-start) > timeout:
            return False
        if on_tick: on_tick(max(0, timeout - int(time.time()-start)))
        time.sleep(1)
    return True

# ---------- Verify & dump ----------
def _dump_chunk(world:Path, dimtag:str, cx:int, cz:int) -> dict:
    rdir = world/Path(dimtag)
    rx, rz = math.floor(cx/32), math.floor(cz/32)
    mca = rdir/f"r.{rx}.{rz}.mca"
    if not mca.exists(): return {"error": f"region file not found: {mca}"}
    idx = (cx % 32) + (cz % 32)*32  # Python % keeps sign-safe positive remainder
    with open(mca, "rb") as fp:
        offsets,_ = _read_header(fp)
        entry = offsets[idx]
        if entry == 0: return {"error":"chunk missing in region"}
        off, cnt = _parse_loc(entry)
        raw = _read_chunk_payload(fp, off)
        if not raw: return {"error":"failed reading chunk payload"}
        out = {"xPos":cx,"zPos":cz,"classic_sections":False,"void":None,"has_entities":None,"has_tileentities":None}
        try:
            nbf = nbt.NBTFile(fileobj=io.BytesIO(raw))
            lvl = nbf["Level"]
            classic = ("Sections" in lvl)
            out["classic_sections"] = classic
            if classic:
                void = True
                for sec in lvl["Sections"]:
                    if "Blocks" not in sec: classic=False; break
                    if any(b != 0 for b in sec["Blocks"].value): void=False; break
                out["classic_sections"] = classic
                out["void"] = (void if classic else None)
            out["has_entities"]     = ("Entities" in lvl and len(lvl["Entities"])>0)
            out["has_tileentities"] = ("TileEntities" in lvl and len(lvl["TileEntities"])>0)
            out["has_biomes_256"]   = ("Biomes" in lvl and isinstance(lvl["Biomes"], nbt.TAG_Byte_Array) and len(lvl["Biomes"].value)==256)
        except Exception as e:
            out["error"] = f"NBT error: {e!r}"
        return out

# ---------- CLI ----------
def run_cli():
    ap = argparse.ArgumentParser(description="Re-run anytime to void newly-generated chunks (1.7.x classic Anvil), preserving biomes.")
    ap.add_argument("world", nargs="?", help="Path to world folder (contains level.dat)")
    ap.add_argument("--backup", action="store_true")
    ap.add_argument("--per-file-backup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-void", action="store_true")
    ap.add_argument("--only", help='Only these region dirs (comma), e.g. "region,DIM-1/region,DIM1/region"')
    ap.add_argument("--exclude", help='Exclude these region dirs (comma).')
    ap.add_argument("--wait-for-unlock", type=int, default=0)
    ap.add_argument("--bbox", type=str, help="minX,minZ,maxX,maxZ")
    ap.add_argument("--center-x", type=int)
    ap.add_argument("--center-z", type=int)
    ap.add_argument("--radius-blocks", type=int, default=0)
    ap.add_argument("--skip-if-entities", action="store_true")
    ap.add_argument("--skip-if-tileentities", action="store_true")
    ap.add_argument("--keep-entities", action="store_true")
    ap.add_argument("--keep-tileentities", action="store_true")
    ap.add_argument("--mark-void", action="store_true")
    ap.add_argument("--platform-x", type=int)
    ap.add_argument("--platform-y", type=int, default=64)
    ap.add_argument("--platform-z", type=int)
    ap.add_argument("--platform-size", type=int, default=3)
    ap.add_argument("--platform-id", type=int, default=7)
    ap.add_argument("--platform-meta", type=int, default=0)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--report-json", type=str)
    ap.add_argument("--dump-chunk", nargs=3, metavar=("DIMTAG","CX","CZ"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    # No world path? Launch GUI.
    if not args.world:
        return False

    if nbt is None:
        print("ERROR: dependency missing. Install with:  pip install NBT")
        sys.exit(2)

    world = Path(args.world)
    if not (world/"level.dat").exists():
        raise SystemExit("level.dat not found — point to the world root.")

    # Lock handling
    if args.wait_for_unlock:
        ok = _wait_for_unlock(world, args.wait_for_unlock)
        if not ok:
            print("session.lock still present; aborting to avoid corruption.")
            sys.exit(3)
    elif (world/"session.lock").exists():
        print("WARNING: session.lock present. Close Minecraft/Prism/MCEdit before running.")

    if args.backup and not args.dry_run:
        z = _backup_world(world); print(f"World backup: {z.name}")

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    exclude = [s.strip() for s in args.exclude.split(",")] if args.exclude else None
    region_dirs = _select_region_dirs(world, only, exclude)
    print(f"Found {len(region_dirs)} region folder(s).")

    bbox = None
    if args.bbox:
        a = [int(x.strip()) for x in args.bbox.split(",")]
        if len(a)!=4: raise SystemExit("bbox must be: minX,minZ,maxX,maxZ")
        x1,z1,x2,z2 = a; bbox = (min(x1,x2),min(z1,z2),max(x1,x2),max(z1,z2))
    center = (args.center_x, args.center_z) if (args.center_x is not None and args.center_z is not None) else None

    platform = None
    if args.platform_x is not None and args.platform_z is not None:
        if args.platform_size <= 0 or args.platform_size % 2 == 0:
            raise SystemExit("Platform size must be a positive odd number.")
        platform = (int(args.platform_x), int(args.platform_z), int(args.platform_y),
                    int(args.platform_size), int(args.platform_id), int(args.platform_meta))

    if args.per_file_backup and not args.dry_run:
        for rdir in region_dirs:
            for mca in sorted(p for p in rdir.iterdir() if p.suffix==".mca" and p.name.startswith("r.")):
                _backup_mca(mca)

    all_mca = []
    for rdir in region_dirs:
        files = sorted(p for p in rdir.iterdir() if p.suffix==".mca" and p.name.startswith("r."))
        print(f"Processing {len(files)} region files in '{_dim_tag(world, rdir)}'...")
        all_mca.extend(files)

    if args.verify_only:
        total_scanned = 0; nonclassic_or_nonvoid = 0
        for mca in all_mca:
            with open(mca, "rb") as fp:
                scanned = 0; bad = 0
                for idx,off,cnt in _iter_chunk_locations(fp):
                    raw = _read_chunk_payload(fp, off)
                    if not raw: continue
                    scanned += 1
                    if not _is_void_chunk(raw): bad += 1
                print(f"verify: {mca.name}: {bad} needing void out of {scanned}")
                total_scanned += scanned; nonclassic_or_nonvoid += bad
        print(f"Verify summary: {nonclassic_or_nonvoid} needing void out of {total_scanned} scanned.")
        if args.report_json:
            Path(args.report_json).write_text(json.dumps({"total_scanned":total_scanned,"needs_void":nonclassic_or_nonvoid}, indent=2))
            print(f"Wrote report: {args.report_json}")
        return True

    total_scanned = 0; total_changed = 0
    def work_one(mca:Path):
        return _process_region(
            mca,
            bbox=bbox, center=center, radius_blocks=int(args.radius_blocks),
            include_void=args.include_void,
            platform=platform,
            keep_entities=args.keep_entities,
            keep_tileentities=args.keep_tileentities,
            skip_if_entities=args.skip_if_entities,
            skip_if_tileentities=args.skip_if_tileentities,
            mark_void=args.mark_void,
            dry_run=args.dry_run,
            strict=args.strict
        )

    workers = max(1, int(args.workers))
    if workers == 1:
        for mca in all_mca:
            n,s,c = work_one(mca)
            print(f"  {'would void' if args.dry_run else 'voided'}: {n} ({c} of {s} chunks)")
            total_scanned += s; total_changed += c
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work_one,m): m for m in all_mca}
            for fut in concurrent.futures.as_completed(futs):
                n,s,c = fut.result()
                print(f"  {'would void' if args.dry_run else 'voided'}: {n} ({c} of {s} chunks)")
                total_scanned += s; total_changed += c

    print(f"{'Would void' if args.dry_run else 'Done. Voided'} {total_changed} chunk(s) out of {total_scanned} scanned. Biomes preserved.")
    return True

# ---------- GUI ----------
def run_gui():
    # ---- Tk imports
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import tkinter.font as tkfont
    import ctypes, ctypes.wintypes
    import platform as _pyplatform
    import winreg

    if nbt is None:
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Missing dependency", "The 'NBT' package is required for the .py version.\nInstall with:\n\npip install NBT")
        return

    # ---- Windows theme helpers (accent + dark mode)
    def get_windows_accent_rgb(default=(74,163,255)):
        try:
            dwm = ctypes.windll.dwmapi
            color = ctypes.wintypes.DWORD()
            opaque = ctypes.wintypes.BOOL()
            if dwm.DwmGetColorizationColor(ctypes.byref(color), ctypes.byref(opaque)) == 0:
                # ARGB (A in high byte)
                c = color.value
                r = (c >> 16) & 0xFF
                g = (c >> 8) & 0xFF
                b = c & 0xFF
                return (r,g,b)
        except Exception:
            pass
        return default

    def is_windows_light_theme():
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                 r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return bool(val)
        except Exception:
            return False  # default to dark

    # ---- Colors (contrast-minded)
    LIGHT = {
        "BG": "#f3f3f3",
        "Panel": "#ffffff",
        "Text": "#1b1b1b",
        "Subtle": "#5a5a5a",
        "InputBG": "#ffffff",
        "InputText": "#1b1b1b",
        "InputDisabledBG": "#e9e9e9",
        "InputDisabledFG": "#6b6b6b",
        "Border": "#d0d0d0",
        "Warn": "#b58900",
        "Err":  "#cc3d3d",
    }
    DARK = {
        "BG": "#0d1117",
        "Panel": "#0f141b",
        "Text": "#e6edf3",
        "Subtle": "#9aa6b2",
        "InputBG": "#0b0f14",
        "InputText": "#e6edf3",
        "InputDisabledBG": "#171b21",
        "InputDisabledFG": "#8a96a3",
        "Border": "#2b3138",
        "Warn": "#ffd166",
        "Err":  "#ff6b6b",
    }
    IS_LIGHT = is_windows_light_theme()
    COLORS = LIGHT if IS_LIGHT else DARK
    ACCENT = get_windows_accent_rgb()
    ACCENT_HEX = f"#{ACCENT[0]:02x}{ACCENT[1]:02x}{ACCENT[2]:02x}"

    # ---- Root & fonts
    root = tk.Tk()
    root.title("GTNH 1.7.x Voider (keep biomes)")
    root.geometry("980x650")
    try:
        # Prefer Segoe UI Variable; fallback to Segoe UI
        base = tkfont.nametofont("TkDefaultFont")
        family = "Segoe UI Variable" if "Variable" in tkfont.families() else "Segoe UI"
        base.configure(family=family, size=11)
        root.option_add("*Font", base)
    except Exception:
        pass

    # ---- ttk theme styling
    style = ttk.Style(root)
    # Use "clam" as baseline; then paint it
    if "clam" in style.theme_names(): 
        style.theme_use("clam")

    # Frames/labels
    style.configure("TFrame", background=COLORS["BG"])
    style.configure("TLabelframe", background=COLORS["BG"], foreground=COLORS["Text"], relief="solid", bordercolor=COLORS["Border"])
    style.configure("TLabelframe.Label", background=COLORS["BG"], foreground=COLORS["Text"])
    style.configure("TLabel", background=COLORS["BG"], foreground=COLORS["Text"])

    # Entry/Combobox with disabled state maps
    style.configure("TEntry", fieldbackground=COLORS["InputBG"], foreground=COLORS["InputText"], bordercolor=COLORS["Border"], lightcolor=COLORS["Border"], darkcolor=COLORS["Border"])
    style.map("TEntry",
              fieldbackground=[("disabled", COLORS["InputDisabledBG"])],
              foreground=[("disabled", COLORS["InputDisabledFG"])])
    style.configure("TCombobox", fieldbackground=COLORS["InputBG"], foreground=COLORS["InputText"], bordercolor=COLORS["Border"])
    style.map("TCombobox",
              fieldbackground=[("disabled", COLORS["InputDisabledBG"])],
              foreground=[("disabled", COLORS["InputDisabledFG"])])

    # Buttons
    style.configure("TButton", background=COLORS["Panel"], foreground=COLORS["Text"], bordercolor=COLORS["Border"])
    style.map("TButton",
              background=[("active", ACCENT_HEX)],
              foreground=[("active", "#0b0e14" if not IS_LIGHT else "#000000")])
    style.configure("Accent.TButton", background=ACCENT_HEX, foreground="#0b0e14" if not IS_LIGHT else "#000000")
    style.map("Accent.TButton",
              background=[("pressed", ACCENT_HEX), ("active", ACCENT_HEX)])

    # Progress
    style.configure("bar.Horizontal.TProgressbar", troughcolor=COLORS["Panel"], background=ACCENT_HEX, bordercolor=COLORS["Border"])

    # ---- Small helpers
    def ellipsize_middle(text:str, max_len:int=100):
        if len(text)<=max_len: return text
        half = (max_len-1)//2
        return text[:half] + "…" + text[-half:]

    class Tooltip:
        def __init__(self, widget, text="", delay=450):
            self.widget = widget; self.text=text; self.delay=delay
            self.tip=None; self.after=None
            widget.bind("<Enter>", self._enter)
            widget.bind("<Leave>", self._leave)
        def _enter(self, _):
            self.after = self.widget.after(self.delay, self._show)
        def _leave(self, _):
            if self.after: self.widget.after_cancel(self.after); self.after=None
            if self.tip: self.tip.destroy(); self.tip=None
        def _show(self):
            if self.tip or not self.text: return
            x = self.widget.winfo_rootx()+12; y=self.widget.winfo_rooty()+self.widget.winfo_height()+6
            self.tip = tk.Toplevel(self.widget); self.tip.wm_overrideredirect(1)
            self.tip.wm_geometry(f"+{x}+{y}")
            frm=ttk.Frame(self.tip); frm.configure(style="TFrame")
            lbl=tk.Label(frm, text=self.text, bg=COLORS["Panel"], fg=COLORS["Text"], bd=1, relief="solid", highlightthickness=0)
            lbl.pack(ipadx=6, ipady=3); frm.pack()

    def log_write(msg:str):
        txt.insert("end", msg+"\n"); txt.see("end"); root.update_idletasks()

    def open_folder(p:Path):
        try:
            os.startfile(str(p))
        except Exception as e:
            messagebox.showerror("Open folder failed", str(e))

    # ---- Discover worlds (PrismLauncher defaults)
    def discover_worlds()->List[Tuple[str,Path]]:
        out=[]
        appdata=os.environ.get("APPDATA")
        if appdata:
            prism=Path(appdata)/"PrismLauncher"/"instances"
            if prism.exists():
                for inst in prism.iterdir():
                    wroot = inst/".minecraft"/"saves"
                    if wroot.exists():
                        for w in wroot.iterdir():
                            if (w/"level.dat").exists(): 
                                # label: LevelName — instance name
                                label = f"{w.name} — {inst.name}"
                                out.append((label, w))
        # Also look at %USERPROFILE%/.minecraft/saves
        home=os.environ.get("USERPROFILE")
        if home:
            saves=Path(home)/"AppData"/"Roaming"/".minecraft"/"saves"
            if saves.exists():
                for w in saves.iterdir():
                    if (w/"level.dat").exists():
                        out.append((f"{w.name} — .minecraft", w))
        # de-dup by path
        seen=set(); uniq=[]
        for label, p in out:
            if str(p) not in seen:
                uniq.append((label,p)); seen.add(str(p))
        return sorted(uniq, key=lambda t: (t[0].lower(), str(t[1]).lower()))

    # ---- Layout
    root.configure(bg=COLORS["BG"])
    P = dict(padx=10, pady=6)

    top = ttk.Labelframe(root, text="World")
    top.pack(fill="x", **P)

    world_path_var = tk.StringVar()
    ent_world = ttk.Entry(top, textvariable=world_path_var)
    ent_world.pack(side="left", fill="x", expand=True, padx=(10,6), pady=8)
    Tooltip(ent_world, "Path to the world folder (must contain level.dat)")

    def pick_world():
        p = filedialog.askdirectory(title="Choose your world folder")
        if p:
            world_path_var.set(p)
            on_world_changed()

    btn_browse = ttk.Button(top, text="Browse…", command=pick_world)
    btn_browse.pack(side="left", padx=(0,10), pady=8)

    # status under entry
    world_status = tk.Label(root, text="", bg=COLORS["BG"], fg=COLORS["Subtle"])
    world_status.pack(anchor="w", padx=20, pady=(0,8))

    # Detected worlds row
    mid = ttk.Frame(root); mid.pack(fill="x", **P)
    ttk.Label(mid, text="Detected worlds:").pack(side="left", padx=(2,8))
    worlds_combo_var = tk.StringVar()
    worlds_combo = ttk.Combobox(mid, textvariable=worlds_combo_var, state="readonly", width=80, values=[])
    worlds_combo.pack(side="left", fill="x", expand=True)
    Tooltip(worlds_combo, "Pick a world discovered on this machine")
    def refresh_worlds():
        worlds = discover_worlds()
        items = [f"{label} — {p}" for (label,p) in worlds]
        worlds_combo["values"] = items
        btn_refresh.configure(state="normal")
        return worlds
    def on_combo_select(_=None):
        s = worlds_combo_var.get()
        if " — " in s:
            path = s.split(" — ")[-1].strip()
            world_path_var.set(path)
            on_world_changed()
    worlds_combo.bind("<<ComboboxSelected>>", on_combo_select)

    btn_refresh = ttk.Button(mid, text="Refresh", command=refresh_worlds)
    btn_refresh.pack(side="left", padx=(8,10))

    # Dimensions frame
    dims_f = ttk.Labelframe(root, text="Dimensions")
    dims_f.pack(fill="both", **P)
    dim_container = ttk.Frame(dims_f); dim_container.pack(fill="x", padx=6, pady=6)
    dims_vars:Dict[Path,tk.BooleanVar]={}

    btn_row = ttk.Frame(dims_f); btn_row.pack(anchor="w", padx=6, pady=(0,6))
    def select_all():
        for v in dims_vars.values(): v.set(True)
    def select_none():
        for v in dims_vars.values(): v.set(False)
    ttk.Button(btn_row,text="Select all",command=select_all).pack(side="left", padx=(0,8))
    ttk.Button(btn_row,text="Select none",command=select_none).pack(side="left")

    # Advanced options (collapsible)
    adv = ttk.Labelframe(root, text="Advanced options")
    adv.pack(fill="x", **P)

    # area filters
    area = ttk.Labelframe(adv, text="Area filters (optional)")
    area.pack(fill="x", padx=6, pady=6)
    cx_var = tk.StringVar(); cz_var = tk.StringVar(); rad_var=tk.StringVar(value="0")
    bbox_minx=tk.StringVar(); bbox_minz=tk.StringVar(); bbox_maxx=tk.StringVar(); bbox_maxz=tk.StringVar()
    row1 = ttk.Frame(area); row1.pack(fill="x", padx=6, pady=4)
    ttk.Label(row1, text="Center X:").pack(side="left"); ttk.Entry(row1,width=10,textvariable=cx_var).pack(side="left", padx=(4,10))
    ttk.Label(row1, text="Z:").pack(side="left"); ttk.Entry(row1,width=10,textvariable=cz_var).pack(side="left", padx=(4,10))
    ttk.Label(row1, text="Radius (blocks):").pack(side="left"); ttk.Entry(row1,width=10,textvariable=rad_var).pack(side="left", padx=(4,10))
    row2 = ttk.Frame(area); row2.pack(fill="x", padx=6, pady=4)
    ttk.Label(row2,text="BBox minX/minZ:").pack(side="left"); ttk.Entry(row2,width=12,textvariable=bbox_minx).pack(side="left", padx=4)
    ttk.Entry(row2,width=12,textvariable=bbox_minz).pack(side="left", padx=(0,10))
    ttk.Label(row2,text="maxX/maxZ:").pack(side="left"); ttk.Entry(row2,width=12,textvariable=bbox_maxx).pack(side="left", padx=4)
    ttk.Entry(row2,width=12,textvariable=bbox_maxz).pack(side="left")

    # platform
    plat = ttk.Labelframe(adv, text="Spawn platform (optional)")
    plat.pack(fill="x", padx=6, pady=6)
    plat_en=tk.BooleanVar(value=False)
    plat_x=tk.StringVar(value="0"); plat_y=tk.StringVar(value="64"); plat_z=tk.StringVar(value="0")
    plat_size=tk.StringVar(value="3"); plat_id=tk.StringVar(value="7"); plat_meta=tk.StringVar(value="0")
    rowp=ttk.Frame(plat); rowp.pack(fill="x", padx=6, pady=4)
    ttk.Checkbutton(rowp,text="Enable",variable=plat_en).pack(side="left", padx=(0,10))
    ttk.Label(rowp,text="X/Y/Z:").pack(side="left")
    ttk.Entry(rowp,width=10,textvariable=plat_x).pack(side="left", padx=4)
    ttk.Entry(rowp,width=6,textvariable=plat_y).pack(side="left", padx=4)
    ttk.Entry(rowp,width=10,textvariable=plat_z).pack(side="left", padx=8)
    ttk.Label(rowp,text="Size:").pack(side="left"); ttk.Entry(rowp,width=6,textvariable=plat_size).pack(side="left", padx=4)
    ttk.Label(rowp,text="Block ID/Meta:").pack(side="left")
    ttk.Entry(rowp,width=6,textvariable=plat_id).pack(side="left", padx=4)
    ttk.Entry(rowp,width=6,textvariable=plat_meta).pack(side="left", padx=4)

    # safety + perf
    saf = ttk.Labelframe(adv, text="Safety & performance")
    saf.pack(fill="x", padx=6, pady=6)
    backup_var=tk.BooleanVar(value=True)
    include_void_var=tk.BooleanVar(value=False)
    wait_var=tk.StringVar(value="30")
    workers_var=tk.StringVar(value="8")
    ttk.Checkbutton(saf,text="Backup world zip",variable=backup_var).pack(side="left", padx=8)
    ttk.Checkbutton(saf,text="Rewrite already-void chunks",variable=include_void_var).pack(side="left", padx=8)
    ttk.Label(saf,text="Wait for session.lock (s):").pack(side="left", padx=(16,4))
    ttk.Entry(saf,width=6,textvariable=wait_var).pack(side="left")
    ttk.Label(saf,text="Workers:").pack(side="left", padx=(16,4))
    ttk.Entry(saf,width=6,textvariable=workers_var).pack(side="left", padx=(0,8))

    # Controls row
    ctl = ttk.Frame(root); ctl.pack(fill="x", **P)
    run_btn = ttk.Button(ctl, text="Run", style="Accent.TButton"); run_btn.pack(side="left", padx=(0,8))
    dry_btn = ttk.Button(ctl, text="Dry run"); dry_btn.pack(side="left", padx=(0,8))
    save_btn = ttk.Button(ctl, text="Save log"); save_btn.pack(side="left", padx=(0,8))
    open_btn = ttk.Button(ctl, text="Open world folder"); open_btn.pack(side="left")
    prog = ttk.Progressbar(ctl, mode="determinate", style="bar.Horizontal.TProgressbar")
    prog.pack(side="right", fill="x", expand=True, padx=(10,6))
    pct_lbl = ttk.Label(ctl, text="0%"); pct_lbl.pack(side="right")

    # Log frame
    log_f = ttk.Labelframe(root, text="Log")
    log_f.pack(fill="both", expand=True, **P)
    txt = tk.Text(log_f, height=14, bg=COLORS["Panel"], fg=COLORS["Text"], insertbackground=COLORS["Text"],
                  relief="flat", bd=1, highlightthickness=1, highlightbackground=COLORS["Border"])
    txt.pack(fill="both", expand=True, padx=6, pady=6)

    # ----- world handling -----
    def set_world_status(ok:bool, msg:str):
        world_status.config(text=msg, fg=(COLORS["Text"] if ok else COLORS["Err"]))

    def scan_dims():
        for c in dim_container.winfo_children(): c.destroy()
        dims_vars.clear()
        wp = Path(world_path_var.get())
        if not (wp/"level.dat").exists():
            set_world_status(False, "level.dat ✗   (pick a valid world)")
            return
        set_world_status(True, "level.dat ✓")
        found=_find_region_dirs(wp)
        if not found:
            ttk.Label(dim_container, text="No region folders found.", foreground=COLORS["Warn"]).pack(anchor="w")
            return
        for p in found:
            var=tk.BooleanVar(value=True)
            dims_vars[p]=var
            row=ttk.Frame(dim_container); row.pack(fill="x")
            count=len([m for m in p.iterdir() if m.suffix==".mca" and m.name.startswith("r.")])
            ttk.Checkbutton(row,text=_dim_tag(wp,p),variable=var).pack(side="left")
            ttk.Label(row,text=f"{count} files").pack(side="right")

    def on_world_changed():
        scan_dims()

    # initial worlds list
    refresh_worlds()

    # ---------- Run pipeline ----------
    def parse_filters():
        bbox=None
        if (bbox_minx.get().strip() or bbox_minz.get().strip() or bbox_maxx.get().strip() or bbox_maxz.get().strip()):
            try:
                x1=int(bbox_minx.get()); z1=int(bbox_minz.get()); x2=int(bbox_maxx.get()); z2=int(bbox_maxz.get())
                bbox=(min(x1,x2),min(z1,z2),max(x1,x2),max(z1,z2))
            except Exception:
                messagebox.showerror("Error","BBox must be integers: minX, minZ, maxX, maxZ")
                return None,None,None
        center=None
        if cx_var.get().strip() and cz_var.get().strip():
            try: center=(int(cx_var.get()), int(cz_var.get()))
            except Exception:
                messagebox.showerror("Error","Center X/Z must be integers."); return None,None,None
        try: radius=int(rad_var.get() or "0")
        except Exception:
            messagebox.showerror("Error","Radius must be an integer."); return None,None,None
        return bbox, center, radius

    def collect_files(wp:Path)->List[Path]:
        selected=[p for p,v in dims_vars.items() if v.get()]
        pool=selected if selected else _find_region_dirs(wp)
        files=[]
        for rdir in pool:
            files.extend(sorted(p for p in rdir.iterdir() if p.suffix==".mca" and p.name.startswith("r.")))
        return files

    def save_log():
        try:
            p=filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text","*.txt")], title="Save log")
            if not p: return
            Path(p).write_text(txt.get("1.0","end"), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def do_run(dry=False):
        wp=Path(world_path_var.get())
        if not (wp/"level.dat").exists():
            messagebox.showerror("Error","Pick a valid world folder (must contain level.dat).")
            return

        # Wait for session.lock (with countdown), then confirm force proceed
        wait_s=0
        try: wait_s=int(wait_var.get() or "0")
        except Exception: wait_s=0
        lock=wp/"session.lock"
        if lock.exists() and wait_s>0:
            def tick_left(sec_left):
                pct_lbl.config(text=f"waiting… {sec_left}s"); root.update_idletasks()
            ok=_wait_for_unlock(wp, wait_s, on_tick=tick_left)
            pct_lbl.config(text="0%")
            if not ok and lock.exists():
                if not messagebox.askyesno("World appears open",
                                           "session.lock is still present.\nProceed anyway? (May corrupt data if Minecraft is open)"):
                    return
        elif lock.exists():
            if not messagebox.askyesno("World appears open",
                                       "session.lock is present. Proceed anyway?"):
                return

        # parse filters & platform
        bbox, center, radius = parse_filters()
        if bbox is None and center is None and radius is None and (bbox_minx.get().strip() or bbox_minz.get().strip() or bbox_maxx.get().strip() or bbox_maxz.get().strip()):
            return
        platform=None
        if plat_en.get():
            try:
                size=int(plat_size.get())
                if size<=0 or size%2==0:
                    messagebox.showerror("Error","Platform size must be a positive odd number."); return
                platform=(int(plat_x.get()), int(plat_z.get()), int(plat_y.get()), size, int(plat_id.get()), int(plat_meta.get()))
            except Exception:
                messagebox.showerror("Error","Platform X/Y/Z, size, id, meta must be integers.")
                return

        files=collect_files(wp)
        if not files: 
            log_write("No region files to process."); 
            return

        # backup
        if backup_var.get() and not dry:
            z=_backup_world(wp); log_write(f"World backup: {z.name}")

        prog.configure(maximum=len(files), value=0); pct_lbl.configure(text="0%")
        changed_total=0; scanned_total=0

        try: workers=max(1,int(workers_var.get() or "1"))
        except Exception: workers=1

        common=dict(
            bbox=bbox, center=center, radius_blocks=radius,
            include_void=bool(include_void_var.get()),
            platform=platform,
            keep_entities=False, keep_tileentities=False,
            skip_if_entities=False, skip_if_tileentities=False,
            mark_void=False,
            dry_run=bool(dry),
            strict=False
        )
        def work_one(mca:Path): 
            return _process_region(mca, **common)

        log_write(f"Processing {len(files)} region files…")
        if workers==1:
            for i,mca in enumerate(files,1):
                try:
                    n,s,c = work_one(mca)
                    log_write(f"[void] {n} ({c} of {s} chunks)")
                    changed_total += c; scanned_total += s
                except Exception as e:
                    log_write(f"[skip] {mca.name} - {e!r}")
                prog.configure(value=i); pct_lbl.configure(text=f"{int(i*100/len(files))}%"); root.update_idletasks()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs={ex.submit(work_one,m):m for m in files}
                done=0
                for fut in concurrent.futures.as_completed(futs):
                    mca=futs[fut]
                    try:
                        n,s,c=fut.result()
                        log_write(f"[void] {n} ({c} of {s} chunks)")
                        changed_total += c; scanned_total += s
                    except Exception as e:
                        log_write(f"[skip] {mca.name} - {e!r}")
                    done+=1; prog.configure(value=done); pct_lbl.configure(text=f"{int(done*100/len(files))}%"); root.update_idletasks()

        messagebox.showinfo("Finished", f"{'Would void' if dry else 'Done.'} Voided {changed_total} / {scanned_total} chunks.\n\nBiomes preserved.")
        log_write(f"[done] Voided {changed_total} / {scanned_total} chunks. Biomes preserved.")

    # wire controls
    run_btn.configure(command=lambda: do_run(False))
    dry_btn.configure(command=lambda: do_run(True))
    save_btn.configure(command=save_log)
    open_btn.configure(command=lambda: open_folder(Path(world_path_var.get())))

    # keyboard mnemonics
    root.bind_all("<Alt-r>", lambda e: run_btn.invoke())
    root.bind_all("<Alt-d>", lambda e: dry_btn.invoke())
    root.bind_all("<Alt-s>", lambda e: save_btn.invoke())
    root.bind_all("<Alt-o>", lambda e: open_btn.invoke())

    # initialize world field if any discovered
    worlds = refresh_worlds()
    if worlds:
        world_path_var.set(str(worlds[0][1]))
        worlds_combo_var.set(f"{worlds[0][0]} — {worlds[0][1]}")
    on_world_changed()

    root.minsize(820, 560)
    root.mainloop()

# ---------- Entrypoint ----------
if __name__ == "__main__":
    ran_cli = run_cli()
    if not ran_cli:
        run_gui()
