#!/usr/bin/env python3
# 5/10/2025-14
# GTNH / 1.7.x Re-runnable Voider (keeps biomes) — GUI + CLI, strict classic Anvil
# - Writes 16 classic Sections with Blocks/Data/SkyLight/BlockLight; preserves 256-byte Biomes
# - Real Region I/O: 8KiB header (offsets+timestamps), per-chunk [length][compression][zlib], 32x32 chunks/region
# - Re-run anytime: skips already-void chunks by default; optional include-void
# - GUI: themed ttk, path box + Browse…, Scan Dims, progress bar, radius/bbox filters, platform pad, config, Auto-run
# - CLI: same features (backup, per-file .bak, workers, verify-only, dump-chunk, lock wait)
# NOTE: Close Minecraft/Prism/MCEdit (or use --wait-for-unlock / GUI wait) to avoid corruption.
#
# Format refs (stable):
# - Region header & chunk payload framing; 32x32 chunks/region. https://c4k3.github.io/wiki.vg/Region_Files.html
# - Classic Anvil chunk Sections & 256-byte Biomes (1.2–1.12). https://minecraft.fandom.com/wiki/Chunk_format

import argparse, io, os, struct, time, zlib, json, threading, concurrent.futures, sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from zipfile import ZipFile, ZIP_DEFLATED

try:
    from nbt import nbt  # twoolie/NBT
except Exception:
    nbt = None

SECTOR_BYTES = 4096
ZLIB = 2

# ---------------- Region primitives (spec-accurate) ----------------

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
        if entry == 0: continue
        off, cnt = _parse_loc(entry)
        if off and cnt: yield idx, off, cnt

def _read_chunk_payload(fp, sector_off:int) -> Optional[bytes]:
    fp.seek(sector_off * SECTOR_BYTES)
    lb = fp.read(4)
    if len(lb) < 4: return None
    (length,) = struct.unpack(">I", lb)
    comp_b = fp.read(1)
    if not comp_b: return None
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

# ---------------- World/dimension helpers ----------------

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

# ---------------- Classic 1.7.x chunk logic ----------------

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
        if "Sections" not in lvl: return False
        sections = lvl["Sections"]
        saw_classic = False
        for sec in sections:
            if not isinstance(sec, nbt.TAG_Compound): continue
            if "Blocks" not in sec: return False  # not classic/palette-era/malformed
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

# ---------------- Area filters ----------------

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

# ---------------- Processing ----------------

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

# ---------------- Backups & lock ----------------

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

def _wait_for_unlock(world:Path, timeout:int):
    lock = world/"session.lock"
    start = time.time()
    while lock.exists():
        if timeout and (time.time()-start) > timeout:
            raise TimeoutError("session.lock still present; abort to avoid corruption.")
        time.sleep(1)

# ---------------- Verify & dump ----------------

def _dump_chunk(world:Path, dimtag:str, cx:int, cz:int) -> dict:
    rdir = world/Path(dimtag)
    rx, rz = cx//32, cz//32
    mca = rdir/f"r.{rx}.{rz}.mca"
    if not mca.exists(): return {"error": f"region file not found: {mca}"}
    idx = (cx % 32) + (cz % 32)*32  # correct for negatives
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

# ---------------- CLI ----------------

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
    ap.add_argument("--autorun", action="store_true", help="(GUI) Auto-run with last settings next launch")
    args = ap.parse_args()

    if not args.world:
        return False  # fall back to GUI

    if nbt is None:
        print("ERROR: dependency missing. Install with:  pip install NBT")
        sys.exit(2)

    world = Path(args.world)
    if not (world/"level.dat").exists():
        raise SystemExit("level.dat not found — point to the world root.")

    if args.wait_for_unlock: _wait_for_unlock(world, args.wait_for_unlock)
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
                print(f"verify: {mca.name}: {bad} non-void/classic out of {scanned}")
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

# ---------------- GUI ----------------

def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    if nbt is None:
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Missing dependency", "The 'NBT' package is required.\nInstall with:\n\npip install NBT")
        return

    CFG = Path(__file__).with_suffix(".config.json")
    def load_cfg()->Dict:
        try:
            if CFG.exists(): return json.load(open(CFG,"r",encoding="utf-8"))
        except Exception: pass
        return {}
    def save_cfg(d:Dict):
        try: json.dump(d, open(CFG,"w",encoding="utf-8"), indent=2)
        except Exception: pass

    def style_setup(root):
        style = ttk.Style(root)
        if "clam" in style.theme_names(): style.theme_use("clam")
        BG="#1b1f24"; FG="#e6edf3"; ACC="#4aa3ff"; WARN="#ffd166"; ERR="#ff6b6b"; OK="#50fa7b"
        style.configure(".", background=BG, foreground=FG)
        style.configure("TFrame", background=BG); style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TEntry", fieldbackground="#0d1117", foreground=FG)
        style.configure("TButton", background="#0d1117", foreground=FG)
        style.configure("Accent.TButton", background=ACC, foreground="#0b0e14")
        style.configure("bar.Horizontal.TProgressbar", troughcolor="#0d1117", background=ACC)
        return {"BG":BG,"FG":FG,"ACC":ACC,"WARN":WARN,"ERR":ERR,"OK":OK}

    def discover_worlds()->List[Path]:
        out=[]
        appdata=os.environ.get("APPDATA")
        if appdata:
            prism=Path(appdata)/"PrismLauncher"/"instances"
            if prism.exists():
                for inst in prism.iterdir():
                    wroot = inst/".minecraft"/"saves"
                    if wroot.exists():
                        for w in wroot.iterdir():
                            if (w/"level.dat").exists(): out.append(w)
        return sorted(out)

    root = tk.Tk(); root.title("GTNH 1.7.x Voider (keep biomes)")
    colors = style_setup(root)
    cfg = load_cfg()

    # top: world path + browse
    frm_top=ttk.Frame(root); frm_top.pack(fill="x", padx=12, pady=10)
    ttk.Label(frm_top, text="World folder (has level.dat):").pack(anchor="w")
    world_var=tk.StringVar(value=cfg.get("last_world",""))
    ent=ttk.Entry(frm_top, textvariable=world_var, width=80); ent.pack(side="left", fill="x", expand=True, padx=(0,8))
    def pick_world():
        p=filedialog.askdirectory(title="Choose your world folder")
        if p: world_var.set(p)
    ttk.Button(frm_top,text="Browse…",command=pick_world).pack(side="left")

    # quick dropdown of detected worlds
    worlds = discover_worlds()
    if worlds:
        frm_quick=ttk.Frame(root); frm_quick.pack(fill="x", padx=12, pady=(0,8))
        ttk.Label(frm_quick,text="Detected worlds:").pack(side="left")
        quick_var=tk.StringVar(value=world_var.get() or str(worlds[0]))
        def change_world(*_): world_var.set(quick_var.get())
        cb=ttk.Combobox(frm_quick,state="readonly",width=70,textvariable=quick_var,values=[str(p) for p in worlds]); cb.pack(side="left", padx=8)
        cb.bind("<<ComboboxSelected>>", change_world)

    # options frame
    frm_opts=ttk.LabelFrame(root,text="Options"); frm_opts.pack(fill="x", padx=12, pady=8)
    backup_var=tk.BooleanVar(value=True)
    dry_var   =tk.BooleanVar(value=False)
    include_void_var=tk.BooleanVar(value=False)
    wait_var  =tk.IntVar(value=0)
    autorun_var=tk.BooleanVar(value=cfg.get("autorun",False))
    ttk.Checkbutton(frm_opts,text="Backup world zip",variable=backup_var).grid(row=0,column=0,sticky="w",padx=6,pady=4)
    ttk.Checkbutton(frm_opts,text="Dry run (no writes)",variable=dry_var).grid(row=0,column=1,sticky="w",padx=6,pady=4)
    ttk.Checkbutton(frm_opts,text="Rewrite already-void chunks",variable=include_void_var).grid(row=0,column=2,sticky="w",padx=6,pady=4)
    ttk.Label(frm_opts,text="Wait for session.lock (s):").grid(row=0,column=3,sticky="e",padx=6)
    ttk.Entry(frm_opts,width=6,textvariable=wait_var).grid(row=0,column=4,sticky="w",padx=6)
    ttk.Checkbutton(frm_opts,text="Auto-run next launch",variable=autorun_var).grid(row=0,column=5,sticky="w",padx=6)

    # area filters
    center_x=tk.StringVar(); center_z=tk.StringVar(); radius_var=tk.StringVar(value="0")
    bbox_var=tk.StringVar()
    ttk.Label(frm_opts,text="Center X/Z:").grid(row=1,column=0,sticky="e",padx=6)
    ttk.Entry(frm_opts,width=10,textvariable=center_x).grid(row=1,column=1,sticky="w")
    ttk.Entry(frm_opts,width=10,textvariable=center_z).grid(row=1,column=2,sticky="w")
    ttk.Label(frm_opts,text="Radius (blocks):").grid(row=1,column=3,sticky="e")
    ttk.Entry(frm_opts,width=8,textvariable=radius_var).grid(row=1,column=4,sticky="w")
    ttk.Label(frm_opts,text="BBox (minX,minZ,maxX,maxZ):").grid(row=2,column=0,sticky="e",padx=6)
    ttk.Entry(frm_opts,width=28,textvariable=bbox_var).grid(row=2,column=1,columnspan=2,sticky="w")

    # platform
    frm_plat=ttk.LabelFrame(root,text="Optional spawn platform"); frm_plat.pack(fill="x", padx=12, pady=8)
    plat_en=tk.BooleanVar(value=False)
    plat_x=tk.StringVar(value="0"); plat_y=tk.StringVar(value="64"); plat_z=tk.StringVar(value="0")
    plat_size=tk.StringVar(value="3"); plat_id=tk.StringVar(value="7"); plat_meta=tk.StringVar(value="0")
    ttk.Checkbutton(frm_plat,text="Enable",variable=plat_en).grid(row=0,column=0,sticky="w",padx=6)
    ttk.Label(frm_plat,text="X/Y/Z:").grid(row=0,column=1,sticky="e")
    ttk.Entry(frm_plat,width=10,textvariable=plat_x).grid(row=0,column=2,sticky="w")
    ttk.Entry(frm_plat,width=6,textvariable=plat_y).grid(row=0,column=3,sticky="w")
    ttk.Entry(frm_plat,width=10,textvariable=plat_z).grid(row=0,column=4,sticky="w")
    ttk.Label(frm_plat,text="Size:").grid(row=0,column=5,sticky="e")
    ttk.Entry(frm_plat,width=6,textvariable=plat_size).grid(row=0,column=6,sticky="w")
    ttk.Label(frm_plat,text="Block ID/Meta:").grid(row=0,column=7,sticky="e")
    ttk.Entry(frm_plat,width=6,textvariable=plat_id).grid(row=0,column=8,sticky="w")
    ttk.Entry(frm_plat,width=6,textvariable=plat_meta).grid(row=0,column=9,sticky="w")

    # dimensions list
    frm_dims=ttk.LabelFrame(root,text="Dimensions to process"); frm_dims.pack(fill="both", padx=12, pady=8)
    dims_container=ttk.Frame(frm_dims); dims_container.pack(fill="x", pady=4)
    dims_vars:Dict[Path,tk.BooleanVar]={}
    def scan_dims():
        for c in dims_container.winfo_children(): c.destroy()
        dims_vars.clear()
        wp = Path(world_var.get())
        if not (wp/"level.dat").exists():
            ttk.Label(dims_container,text="Pick a valid world (level.dat not found)",style="Warn.TLabel").pack(anchor="w")
            return
        found=_find_region_dirs(wp)
        if not found:
            ttk.Label(dims_container,text="No region folders found.",style="Warn.TLabel").pack(anchor="w")
            return
        for p in found:
            var=tk.BooleanVar(value=True)
            dims_vars[p]=var
            row=ttk.Frame(dims_container); row.pack(fill="x")
            ttk.Checkbutton(row,text=_dim_tag(wp,p),variable=var).pack(side="left")
            count=len([m for m in p.iterdir() if m.suffix==".mca" and m.name.startswith("r.")])
            ttk.Label(row,text=f"{count} files").pack(side="right")

    # controls
    frm_ctl=ttk.Frame(root); frm_ctl.pack(fill="x", padx=12, pady=8)
    ttk.Button(frm_ctl,text="Scan Dims",command=scan_dims).pack(side="left")
    workers_var=tk.StringVar(value=str(cfg.get("workers",4)))
    ttk.Label(frm_ctl,text="Workers:").pack(side="left", padx=(10,2))
    ttk.Entry(frm_ctl,width=4,textvariable=workers_var).pack(side="left")
    run_btn=ttk.Button(frm_ctl,text="Run",style="Accent.TButton"); run_btn.pack(side="left", padx=(12,6))
    dry_btn=ttk.Button(frm_ctl,text="Dry Run"); dry_btn.pack(side="left")

    # progress + log
    frm_prog=ttk.Frame(root); frm_prog.pack(fill="x", padx=12, pady=(0,4))
    pbar=ttk.Progressbar(frm_prog,mode="determinate",length=420,style="bar.Horizontal.TProgressbar"); pbar.pack(side="left", fill="x", expand=True)
    pct_lbl=ttk.Label(frm_prog,text="0%"); pct_lbl.pack(side="left", padx=8)
    frm_log=ttk.LabelFrame(root,text="Log"); frm_log.pack(fill="both", expand=True, padx=12, pady=(0,12))
    txt=tk.Text(frm_log, height=16, bg="#0d1117", fg=colors["FG"], insertbackground=colors["FG"], relief="flat"); txt.pack(fill="both", expand=True)
    def log(msg, tag=None):
        txt.insert("end", msg+"\n"); txt.see("end"); root.update_idletasks()

    def collect_files(wp:Path)->List[Path]:
        selected=[p for p,v in dims_vars.items() if v.get()]
        pool=selected if selected else _find_region_dirs(wp)
        files=[]
        for rdir in pool:
            files.extend(sorted(p for p in rdir.iterdir() if p.suffix==".mca" and p.name.startswith("r.")))
        return files

    def parse_filters():
        bbox = None
        if bbox_var.get().strip():
            try:
                a=[int(x.strip()) for x in bbox_var.get().split(",")]
                if len(a)!=4: raise ValueError
                x1,z1,x2,z2=a; bbox=(min(x1,x2),min(z1,z2),max(x1,x2),max(z1,z2))
            except Exception:
                from tkinter import messagebox; messagebox.showerror("Error","BBox must be four integers: minX,minZ,maxX,maxZ"); return None,None,None
        center=None
        if center_x.get().strip() and center_z.get().strip():
            try: center=(int(center_x.get()), int(center_z.get()))
            except Exception:
                from tkinter import messagebox; messagebox.showerror("Error","Center X/Z must be integers."); return None,None,None
        try: radius=int(radius_var.get())
        except Exception:
            from tkinter import messagebox; messagebox.showerror("Error","Radius must be an integer."); return None,None,None
        return bbox, center, radius

    def do_run(dry=False):
        wp=Path(world_var.get())
        if not (wp/"level.dat").exists():
            from tkinter import messagebox; messagebox.showerror("Error","Pick a valid world folder (must contain level.dat)."); return
        save_cfg({"last_world":str(wp),"autorun":False,"workers":int(workers_var.get() or 1)})

        # lock wait or warn
        wait_s=int(wait_var.get() or 0)
        lock=wp/"session.lock"
        if wait_s>0:
            start=time.time()
            while lock.exists() and (time.time()-start)<wait_s:
                log("Waiting for session.lock to disappear…"); time.sleep(1)
        if lock.exists():
            log("WARNING: session.lock present — close Minecraft/Prism/MCEdit to avoid corruption.")

        bbox, center, radius = parse_filters()
        if bbox is None and center is None and radius is None and bbox_var.get().strip():
            return

        platform=None
        if plat_en.get():
            try:
                size=int(plat_size.get())
                if size<=0 or size%2==0:
                    from tkinter import messagebox; messagebox.showerror("Error","Platform size must be a positive odd number."); return
                platform=(int(plat_x.get()), int(plat_z.get()), int(plat_y.get()), size, int(plat_id.get()), int(plat_meta.get()))
            except Exception:
                from tkinter import messagebox; messagebox.showerror("Error","Platform X/Y/Z, size, id, meta must be integers.")
                return

        files=collect_files(wp)
        if not files: log("No region files to process."); return

        if backup_var.get() and not dry:
            z=_backup_world(wp); log(f"World backup: {z.name}")

        pbar.configure(maximum=len(files), value=0); pct_lbl.configure(text="0%")
        changed_total=0; scanned_total=0

        try: workers=max(1,int(workers_var.get()))
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
        def work_one(mca:Path): return _process_region(mca, **common)

        log(f"Processing {len(files)} region files…")
        if workers==1:
            for i,mca in enumerate(files,1):
                try:
                    n,s,c = work_one(mca)
                    log(f"  {'would void' if dry else 'voided'}: {n} ({c} of {s} chunks)")
                    changed_total += c; scanned_total += s
                except Exception as e:
                    log(f"  SKIP (error): {mca.name} - {e!r}")
                pbar.configure(value=i); pct_lbl.configure(text=f"{int(i*100/len(files))}%"); root.update_idletasks()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs={ex.submit(work_one,m):m for m in files}
                done=0
                for fut in concurrent.futures.as_completed(futs):
                    mca=futs[fut]
                    try:
                        n,s,c=fut.result()
                        log(f"  {'would void' if dry else 'voided'}: {n} ({c} of {s} chunks)")
                        changed_total += c; scanned_total += s
                    except Exception as e:
                        log(f"  SKIP (error): {mca.name} - {e!r}")
                    done+=1; pbar.configure(value=done); pct_lbl.configure(text=f"{int(done*100/len(files))}%"); root.update_idletasks()

        log(f"{'Would void' if dry else 'Done. Voided'} {changed_total} chunk(s) out of {scanned_total} scanned. Biomes preserved.")
        from tkinter import messagebox; messagebox.showinfo("Finished", f"{'Would void' if dry else 'Done. Voided'} {changed_total} / {scanned_total} chunks.")

    # widgets used by do_run()
    center_x=tk.StringVar(); center_z=tk.StringVar(); radius_var=tk.StringVar(value="0")
    bbox_var=tk.StringVar()
    plat_en=tk.BooleanVar(value=False)
    plat_x=tk.StringVar(value="0"); plat_y=tk.StringVar(value="64"); plat_z=tk.StringVar(value="0")
    plat_size=tk.StringVar(value="3"); plat_id=tk.StringVar(value="7"); plat_meta=tk.StringVar(value="0")
    backup_var=tk.BooleanVar(value=True)
    dry_var   =tk.BooleanVar(value=False)
    include_void_var=tk.BooleanVar(value=False)
    wait_var  =tk.IntVar(value=0)

    # build option rows (reuse already created frames)
    # (we keep references for vars; labels/entries were created above)

    # initial scan
    if world_var.get():
        scan_dims()
    else:
        if worlds:
            world_var.set(str(worlds[0])); scan_dims()

    root.minsize(820,600); root.mainloop()

# ---------------- Entrypoint ----------------

if __name__ == "__main__":
    ran_cli = run_cli()
    if not ran_cli:
        run_gui()
