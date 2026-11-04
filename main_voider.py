#!/usr/bin/env python3
# 2025-10-12 r11 (FINAL+robust reader + Nether placeholder)
# GTNH / 1.7.x World Voider — provider-safe biomes + rules + docs UI
# - Robust: tolerant region chunk reader (zlib/gzip/raw, junk headers, bad lengths)
# - Provider-safe biome reset (Overworld keep-or-Plains; non-Overworld remove & repop flags)
# - Cosmetic fix: optional Nether placeholder (Hell) biomes to avoid "Ocean" HUD until providers repopulate
# - Fixes prior duplicate defs; same UI/CLI surface as r10 (+ one new flag and one GUI checkbox)

from __future__ import annotations
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
import hashlib
import shutil
import copy
import stat as statmod
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set, Callable

# Third-party / sibling
from nbt import nbt

# Import the GUI only when it is requested. This keeps headless CLI/core use
# functional on servers without Tk and avoids a main_voider <-> chunkmap_ui cycle.
def _load_chunkmap_ui():
    import importlib
    import importlib.util
    try:
        return importlib.import_module("chunkmap_ui")
    except ModuleNotFoundError as original:
        # Only perform the sidecar fallback when chunkmap_ui itself is missing.
        # A missing dependency inside chunkmap_ui (for example tkinter) must not
        # be disguised as a missing sidecar.
        if original.name != "chunkmap_ui":
            raise
        bases = [
            getattr(sys, "_MEIPASS", ""),
            os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else "",
            os.path.dirname(__file__),
            os.getcwd(),
        ]
        for base in filter(None, bases):
            for path in (os.path.join(base, "chunkmap_ui.py"),
                         os.path.join(base, "chunkmap_ui", "__init__.py")):
                if not os.path.exists(path):
                    continue
                spec = importlib.util.spec_from_file_location("chunkmap_ui", path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules["chunkmap_ui"] = module
                return module
        raise original

def launch_chunk_selector(*args, **kwargs):
    module = _load_chunkmap_ui()
    if not hasattr(module, "launch_chunk_selector"):
        raise ImportError("chunkmap_ui.launch_chunk_selector() is missing")
    return module.launch_chunk_selector(*args, **kwargs)


# ---------------- Constants / globals ----------------
SECTOR_BYTES = 4096
PLAINS_ID = 1        # Overworld flat → Plains
HELL_ID   = 8        # Nether base (Hell) — used only as cosmetic placeholder when enabled
GZIP_MAGIC = b"\x1f\x8b\x08"
ZLIB_MAGIC_PREFIX = b"\x78"  # common zlib headers: 0x78 0x01/0x5E/0x9C/0xDA

@dataclass
class RuleSet:
    block_whitelist: List[str] = field(default_factory=list)  # prefer numeric ID strings (1.7)
    block_blacklist: List[str] = field(default_factory=list)
    entity_whitelist: List[str] = field(default_factory=list) # e.g., 'TwilightForest.minotaur' or 'minecraft:zombie'
    entity_blacklist: List[str] = field(default_factory=list)
    biome_whitelist: List[int] = field(default_factory=list)  # biome ids 0..65535
    biome_blacklist: List[int] = field(default_factory=list)
    enabled: bool = True

@dataclass
class AllRules:
    by_dim: Dict[str, RuleSet] = field(default_factory=dict)

_GLOBAL_RULES: AllRules = AllRules()

PRESETS = {
    "manual": {},
    "safe-void": {
        "skip_if_entities": True, "skip_if_tileentities": True,
        "keep_entities": False, "keep_tileentities": False, "mark_void": True,
    },
    "full-void": {
        "skip_if_entities": False, "skip_if_tileentities": False,
        "keep_entities": False, "keep_tileentities": False, "mark_void": True,
    },
    "preserve-data": {
        "skip_if_entities": False, "skip_if_tileentities": True,
        "keep_entities": True, "keep_tileentities": False, "mark_void": True,
    },
}

def _apply_preset(args) -> None:
    """Apply a named, reviewable bundle of existing processing options."""
    name = getattr(args, "preset", "manual")
    try:
        values = PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}") from exc
    for key, value in values.items():
        setattr(args, key, value)

# ---------------- Region I/O ----------------
def _read_header(fp):
    fp.seek(0)
    hdr = fp.read(8192)
    if len(hdr) != 8192:
        raise IOError(f"Region header bad size: {len(hdr)}")
    offsets = [struct.unpack(">I", hdr[i*4:(i+1)*4])[0] for i in range(1024)]
    stamps  = [struct.unpack(">I", hdr[4096+i*4:4096+(i+1)*4])[0] for i in range(1024)]
    return offsets, stamps

def _parse_loc(entry:int)->Tuple[int,int]:
    return (entry >> 8) & 0xFFFFFF, entry & 0xFF

def _iter_chunk_locations(fp):
    offsets, _ = _read_header(fp)
    # file size guard
    here = fp.tell()
    fp.seek(0, os.SEEK_END)
    fsize = fp.tell()
    fp.seek(here)
    for idx, entry in enumerate(offsets):
        if entry == 0:
            continue
        off, cnt = _parse_loc(entry)
        # require sane values and that the record header fits in file
        if off and cnt and (off * SECTOR_BYTES + 5) <= fsize:
            yield idx, off, cnt

# ---------------- Robust/tolerant chunk payload reader ----------------
def _try_decode_with(codec:int, data:bytes) -> Optional[bytes]:
    """
    Return decompressed NBT bytes with the requested codec:
     1=gzip, 2=zlib, 3=uncompressed. Return None if it fails.
    """
    try:
        if codec == 1:
            import gzip
            return gzip.decompress(data)
        elif codec == 2:
            return zlib.decompress(data)
        elif codec == 3:
            # already uncompressed NBT
            return data
    except Exception:
        return None
    return None

def _looks_like_nbt(data:bytes) -> bool:
    # Heuristic: TAG_Compound (0x0A), name length (2 bytes), then name bytes…
    return len(data) > 4 and data[0] == 0x0A and (data[1] == 0x00 or (data[1] == 0x00 and data[2] == 0x00))

def _scan_and_decode(data: bytes) -> Optional[bytes]:
    """
    Brute-force scan of the whole sector window for gzip/zlib/raw-deflate, then
    a last-chance check for already-uncompressed NBT. Slow but robust.
    """
    import gzip
    import zlib
    import io
    from nbt import nbt

    # 1) Any gzip stream anywhere
    pos = 0
    while True:
        pos = data.find(GZIP_MAGIC, pos)
        if pos == -1:
            break
        try:
            return gzip.decompress(data[pos:])
        except Exception:
            pos += 1

    # 2) Try offsets that can begin a valid zlib stream. Trying every byte
    # becomes quadratic on a corrupted, fully allocated region record.
    zlib_starts = [i for i in range(max(0, len(data)-1))
                   if data[i] == 0x78 and ((data[i] << 8) + data[i+1]) % 31 == 0]
    for start in zlib_starts:
        try:
            return zlib.decompress(data[start:])
        except Exception:
            pass

    # 3) Raw DEFLATE has no magic; tolerate only a small junk prefix.
    for start in range(0, min(len(data), 64)):
        try:
            return zlib.decompress(data[start:], wbits=-15)
        except Exception:
            pass

    # 4) Already-uncompressed NBT starting at some offset?
    for start in range(0, len(data) - 3):
        if data[start] == 0x0A and start + 2 < len(data) and data[start+1] == 0x00:
            try:
                nbt.NBTFile(buffer=io.BytesIO(data[start:]))
                return data[start:]
            except Exception:
                pass

    return None

def _read_chunk_payload(fp, sector_off: int, sector_cnt: int) -> Optional[bytes]:
    try:
        fp.seek(sector_off * SECTOR_BYTES)
        lb = fp.read(4)
        if len(lb) != 4:
            return None
        (length_field,) = struct.unpack(">I", lb)
        comp_b = fp.read(1)
        if len(comp_b) != 1:
            return None
        declared_comp = comp_b[0]

        max_body = max(0, sector_cnt * SECTOR_BYTES - 5)

        to_read_decl = 0
        body_decl = b""
        if length_field > 1:
            to_read_decl = min(length_field - 1, max_body)
            body_decl = fp.read(to_read_decl)

        if to_read_decl < max_body:
            body_full = body_decl + fp.read(max_body - to_read_decl)
        else:
            body_full = body_decl

        # Try declared codec on both slices first
        out = _try_decode_with(declared_comp, body_decl) or _try_decode_with(declared_comp, body_full)
        if out is not None:
            return out

        # FALL BACK: scan entire window aggressively
        out = _scan_and_decode(body_full)
        if out is not None:
            return out

        # As a last small hail-mary: if it *said* gzip/zlib, try the other one on the full window
        if declared_comp == 1:
            try: return zlib.decompress(body_full)
            except Exception: pass
        elif declared_comp == 2:
            try:
                import gzip as _gz
                return _gz.decompress(body_full)
            except Exception:
                pass

        return None
    except Exception:
        return None

# ---------------- Low-level write helpers ----------------
def _write_chunk_payload(fp, sector_off:int, sector_cnt:int, raw_nbt:bytes):
    body = bytes([2]) + zlib.compress(raw_nbt)  # write back as zlib
    payload = struct.pack(">I", len(body)) + body
    need = (len(payload) + SECTOR_BYTES - 1) // SECTOR_BYTES
    if need > 0xFF:
        raise ValueError(f"Chunk payload needs {need} sectors; Anvil header limit is 255")
    if need > sector_cnt:
        fp.seek(0, os.SEEK_END)
        pad = (-fp.tell()) % SECTOR_BYTES
        if pad:
            fp.write(b"\x00"*pad)
        new_off = fp.tell() // SECTOR_BYTES
        if new_off > 0xFFFFFF:
            raise ValueError(f"Region file exceeds the 24-bit sector offset limit: {new_off}")
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
    return str(region_dir.relative_to(world)).replace("\\","/")

def _resolve_world_path(path: Path) -> Path:
    """Resolve either a world folder or a dedicated-server instance folder."""
    selected = Path(path).expanduser().resolve()
    if (selected / "level.dat").is_file():
        return selected
    properties = selected / "server.properties"
    if not properties.is_file():
        return selected
    level_name = "world"
    try:
        for line in properties.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == "level-name":
                level_name = value.strip()
                break
    except OSError as exc:
        raise RuntimeError(f"Could not read {properties}: {exc}") from exc
    if not level_name or Path(level_name).is_absolute():
        raise ValueError(f"Unsafe server.properties level-name: {level_name!r}")
    candidate = (selected / level_name).resolve()
    if candidate != selected and selected not in candidate.parents:
        raise ValueError(f"server.properties level-name escapes the server directory: {level_name!r}")
    return candidate

def _world_kind(world:Path) -> str:
    """Classify a selected save without changing how its Anvil data is read."""
    if (world/"server.properties").is_file() or (world.parent/"server.properties").is_file():
        return "dedicated-server"
    parts = {part.lower() for part in world.parts}
    if "saves" in parts or any("prismlauncher" in part for part in parts):
        return "client-save"
    return "minecraft-world"

def _world_lock_status(world:Path) -> Tuple[str, str]:
    """Best-effort detection of an OS-held Minecraft session.lock."""
    lock_path = world/"session.lock"
    if not lock_path.exists():
        return "unlocked", "session.lock is absent"
    try:
        if os.name == "posix":
            import fcntl
            with lock_path.open("a+b") as fp:
                try:
                    fcntl.lockf(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1)
                except (BlockingIOError, PermissionError):
                    return "active", "session.lock is held by another process"
                finally:
                    try: fcntl.lockf(fp.fileno(), fcntl.LOCK_UN, 1)
                    except OSError: pass
            return "unlocked", "session.lock exists but is not OS-locked"
        if os.name == "nt":
            import msvcrt
            with lock_path.open("a+b") as fp:
                if fp.tell() == 0: fp.write(b"\0"); fp.flush()
                fp.seek(0)
                try:
                    msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    return "active", "session.lock is held by another process"
                finally:
                    try: fp.seek(0); msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError: pass
            return "unlocked", "session.lock exists but is not OS-locked"
    except OSError as exc:
        return "unknown", f"could not test session.lock: {exc}"
    return "unknown", "OS lock testing is unavailable on this platform"

def inspect_world(world:Path) -> Dict[str, object]:
    """Return a read-only client/server world inventory."""
    status, detail = _world_lock_status(world)
    dimensions=[]
    total_files=0; total_bytes=0
    for rdir in _find_region_dirs(world):
        files=sorted(rdir.glob("r.*.*.mca"))
        size=sum(p.stat().st_size for p in files)
        dimensions.append({"tag": _dim_tag(world, rdir), "region_files": len(files), "bytes": size})
        total_files += len(files); total_bytes += size
    return {"world": str(world), "kind": _world_kind(world),
            "lock_status": status, "lock_detail": detail,
            "dimensions": dimensions, "region_files": total_files,
            "region_bytes": total_bytes}

def _assert_world_not_active(world:Path) -> None:
    status, detail = _world_lock_status(world)
    if status == "active":
        raise RuntimeError(
            f"World appears to be open by Minecraft or a server ({detail}). "
            "Stop it cleanly before writing."
        )
    if status == "unknown":
        raise RuntimeError(f"Cannot safely establish that the world is offline: {detail}")

_WORLD_WRITE_LOCKS_GUARD = threading.Lock()
_WORLD_WRITE_LOCKS: Dict[str, threading.Lock] = {}


@contextmanager
def _world_write_guard(world: Path):
    """Serialize in-process writers and hold Minecraft's OS session lock.

    POSIX record locks are process-scoped, so they do not serialize two
    threads in the same Python process. The per-world mutex closes that gap;
    the OS lock coordinates with Minecraft and other applications.
    """
    world = Path(world).expanduser().resolve()
    if not world.is_dir() or not (world / "level.dat").is_file():
        raise ValueError(f"Not a Minecraft world folder: {world}")
    key = os.path.normcase(str(world))
    with _WORLD_WRITE_LOCKS_GUARD:
        local_lock = _WORLD_WRITE_LOCKS.setdefault(key, threading.Lock())
    if not local_lock.acquire(blocking=False):
        raise RuntimeError(f"Another Voider write or biome reset is already running for {world}")

    fp = None
    locked = False
    lock_module = None
    try:
        lock_path = world / "session.lock"
        if lock_path.is_symlink():
            raise RuntimeError(f"Refusing symlinked session.lock: {lock_path}")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(lock_path), flags, 0o600)
        fp = os.fdopen(fd, "r+b")
        if os.name == "posix":
            import fcntl as lock_module
            try:
                lock_module.lockf(fp.fileno(), lock_module.LOCK_EX | lock_module.LOCK_NB, 1, 0, os.SEEK_SET)
            except (BlockingIOError, PermissionError) as exc:
                raise RuntimeError(
                    f"World appears to be open by Minecraft or a server (session.lock is locked): {world}"
                ) from exc
            locked = True
        elif os.name == "nt":
            import msvcrt as lock_module
            fp.seek(0, os.SEEK_END)
            if fp.tell() == 0:
                fp.write(b"\0")
                fp.flush()
            fp.seek(0)
            try:
                lock_module.locking(fp.fileno(), lock_module.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    f"World appears to be open by Minecraft or a server (session.lock is locked): {world}"
                ) from exc
            locked = True
        else:
            raise RuntimeError(f"Safe world-write locking is unsupported on {os.name!r}; refusing to write.")
        yield
    finally:
        if fp is not None:
            if locked and lock_module is not None:
                try:
                    if os.name == "posix":
                        lock_module.lockf(fp.fileno(), lock_module.LOCK_UN, 1, 0, os.SEEK_SET)
                    else:
                        fp.seek(0)
                        lock_module.locking(fp.fileno(), lock_module.LK_UNLCK, 1)
                except OSError:
                    pass
            fp.close()
        local_lock.release()

# ---------------- Biomes / flags ----------------
def _nbt_from_raw(raw:bytes) -> Optional[nbt.NBTFile]:
    try:
        return nbt.NBTFile(buffer=io.BytesIO(raw))
    except Exception:
        return None

def _get_biomes(raw:bytes) -> Optional[bytes]:
    nb = _nbt_from_raw(raw)
    if not nb: return None
    try:
        lvl = nb["Level"]
        if "Biomes" in lvl and isinstance(lvl["Biomes"], nbt.TAG_Byte_Array):
            b = bytes(lvl["Biomes"].value)
            if len(b) == 256:
                return b
    except Exception:
        pass
    return None


def _get_biomes16v2(raw: bytes) -> Optional[bytes]:
    """Return GTNH's 256-entry little-endian u16 biome array, if valid."""
    nb = _nbt_from_raw(raw)
    if nb is None or "Level" not in nb:
        return None
    try:
        tag = nb["Level"].get("Biomes16v2")
        if isinstance(tag, nbt.TAG_Byte_Array) and len(tag.value) == 512:
            return bytes(tag.value)
    except Exception:
        pass
    return None


def _get_biome_ids(raw: bytes) -> Set[int]:
    """Read both vanilla 8-bit Biomes and GTNH's 16-bit Biomes16v2 arrays."""
    nb = _nbt_from_raw(raw)
    if nb is None or "Level" not in nb:
        return set()
    level = nb["Level"]
    biome_ids: Set[int] = set()
    try:
        standard = level.get("Biomes")
        if isinstance(standard, nbt.TAG_Byte_Array) and len(standard.value) == 256:
            biome_ids.update(int(value) & 0xFF for value in standard.value)
        extended = level.get("Biomes16v2")
        if isinstance(extended, nbt.TAG_Byte_Array):
            data = bytes(extended.value)
            if len(data) == 512:
                biome_ids.update(
                    data[index] | (data[index + 1] << 8)
                    for index in range(0, len(data), 2)
                )
    except Exception:
        return set()
    return biome_ids

def _has_entities(raw:bytes) -> Tuple[bool,bool]:
    nb = _nbt_from_raw(raw)
    if not nb: return (False, False)
    try:
        lvl = nb["Level"]
        te = ("TileEntities" in lvl and isinstance(lvl["TileEntities"], nbt.TAG_List) and len(lvl["TileEntities"])>0)
        en = ("Entities" in lvl and isinstance(lvl["Entities"], nbt.TAG_List) and len(lvl["Entities"])>0)
        return te, en
    except Exception:
        return (False, False)

def _is_void_chunk(raw:bytes) -> bool:
    """Return true only when every legacy block ID (including Add) is zero."""
    nb = _nbt_from_raw(raw)
    if not nb: return False
    try:
        lvl = nb["Level"]
        if "Sections" not in lvl: return False
        sections = lvl["Sections"]
        saw = False
        for sec in sections:
            if not isinstance(sec, nbt.TAG_Compound):
                continue
            if "Blocks" not in sec or len(sec["Blocks"].value) != 4096:
                return False
            saw = True
            if any(int(b) != 0 for b in sec["Blocks"].value):
                return False
            if "Add" in sec:
                add = bytes(sec["Add"].value)
                if len(add) != 2048 or any(add):
                    return False
        return saw
    except Exception:
        return False

# ----- block scanning (1.7.x) -----
def _nibble_get(arr: Optional[bytes], idx:int)->int:
    if not arr: return 0
    b = arr[idx >> 1]
    return (b & 0x0F) if (idx & 1) == 0 else ((b >> 4) & 0x0F)

def _nibble_set(arr, idx:int, value:int)->None:
    """Set one 4-bit value in a legacy Anvil nibble array."""
    pos = idx >> 1
    old = arr[pos]
    value &= 0x0F
    arr[pos] = ((old & 0xF0) | value) if (idx & 1) == 0 else ((old & 0x0F) | (value << 4))

def _iter_blocks_17(raw:bytes):
    nb = _nbt_from_raw(raw)
    if not nb: return
    lvl = nb["Level"]
    if "Sections" not in lvl: return
    sections = lvl["Sections"]
    for sec in sections:
        if not isinstance(sec, nbt.TAG_Compound): continue
        if "Blocks" not in sec or "Data" not in sec: continue
        ysec = int(sec["Y"].value)
        blocks = bytes(sec["Blocks"].value)           # 4096
        data   = bytes(sec["Data"].value)             # 2048
        add    = bytes(sec["Add"].value) if "Add" in sec else None
        for ly in range(16):
            base_y = ysec*16 + ly
            for lz in range(16):
                for lx in range(16):
                    idx = (ly * 256) + (lz * 16) + lx  # Y-Z-X
                    base = blocks[idx]
                    hi   = _nibble_get(add, idx)
                    full = (hi << 8) | base
                    meta = _nibble_get(data, idx)
                    yield (lx, lz, base_y, full, meta)

def _chunk_has_biome_id(raw:bytes, wanted:Set[int])->bool:
    if not wanted:
        return False
    return bool(_get_biome_ids(raw) & wanted)

def _chunk_entities_flags(raw:bytes, allow:Set[str], deny:Set[str])->Tuple[bool,bool]:
    nb = _nbt_from_raw(raw)
    if not nb: return (False, False)
    try:
        lvl = nb["Level"]
        if "Entities" not in lvl: return (False, False)
        hit_w=hit_b=False
        for e in lvl["Entities"]:
            if not isinstance(e, nbt.TAG_Compound): continue
            eid = str(e.get("id", nbt.TAG_String("")).value).lower()
            if not eid: continue
            if eid in allow or any(eid.endswith(":"+a) or eid.endswith("."+a) for a in allow):
                hit_w=True
            if eid in deny or any(eid.endswith(":"+a) or eid.endswith("."+a) for a in deny):
                hit_b=True
        return (hit_w, hit_b)
    except Exception:
        return (False, False)

def _chunk_blocks_flags(raw:bytes, allow:Set[str], deny:Set[str])->Tuple[bool,bool]:
    allow_num = {int(t) for t in allow if t.isdigit()}
    deny_num  = {int(t) for t in deny  if t.isdigit()}
    hit_w=hit_b=False
    try:
        for (_x,_z,_y,fid,meta) in _iter_blocks_17(raw):
            if fid in allow_num: hit_w=True
            if fid in deny_num:  hit_b=True
            if hit_w and hit_b: break
    except Exception:
        return (False, False)
    return (hit_w, hit_b)

# ---------------- Rules I/O ----------------
def _rules_path(world:Path)->Path: return world/".voider-rules.json"

def _load_rules(world:Path)->AllRules:
    p=_rules_path(world)
    if not p.exists(): return AllRules()
    try:
        raw=json.loads(p.read_text(encoding="utf-8"))
        out=AllRules()
        for dim,d in raw.get("by_dim",{}).items():
            out.by_dim[dim]=RuleSet(
                block_whitelist=d.get("block_whitelist",[]),
                block_blacklist=d.get("block_blacklist",[]),
                entity_whitelist=[s.lower() for s in d.get("entity_whitelist",[])],
                entity_blacklist=[s.lower() for s in d.get("entity_blacklist",[])],
                biome_whitelist=[int(x) for x in d.get("biome_whitelist",[])],
                biome_blacklist=[int(x) for x in d.get("biome_blacklist",[])],
                enabled=bool(d.get("enabled",True)),
            )
        return out
    except Exception:
        return AllRules()

def _save_rules(world:Path, rules:AllRules)->None:
    data={"by_dim":{}}
    for dim,rs in rules.by_dim.items():
        data["by_dim"][dim]={
            "block_whitelist": rs.block_whitelist,
            "block_blacklist": rs.block_blacklist,
            "entity_whitelist": rs.entity_whitelist,
            "entity_blacklist": rs.entity_blacklist,
            "biome_whitelist": rs.biome_whitelist,
            "biome_blacklist": rs.biome_blacklist,
            "enabled": rs.enabled,
        }
    _rules_path(world).write_text(json.dumps(data, indent=2), encoding="utf-8")

def _load_selection(path:Path)->Tuple[Dict[str, Set[Tuple[int,int]]], Dict[str, Set[Tuple[int,int]]]]:
    """Load and strictly validate a chunk-selection JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Selection root must be an object keyed by dimension tag")
    includes: Dict[str, Set[Tuple[int,int]]] = {}
    excludes: Dict[str, Set[Tuple[int,int]]] = {}
    total = 0
    for dimtag, payload in raw.items():
        if not isinstance(dimtag, str) or not dimtag or not isinstance(payload, dict):
            raise ValueError("Every selection entry must be a named dimension object")
        parsed = {}
        for mode in ("include", "exclude"):
            values = payload.get(mode, [])
            if not isinstance(values, list):
                raise ValueError(f"{dimtag}.{mode} must be a list")
            coords: Set[Tuple[int,int]] = set()
            for item in values:
                if (not isinstance(item, (list, tuple)) or len(item) != 2 or
                        any(isinstance(v, bool) or not isinstance(v, int) for v in item)):
                    raise ValueError(f"Invalid chunk coordinate in {dimtag}.{mode}: {item!r}")
                coords.add((item[0], item[1]))
            total += len(coords)
            if total > 5_000_000:
                raise ValueError("Selection exceeds the 5,000,000-coordinate safety limit")
            parsed[mode] = coords
        # Exclusion is the safety-biased winner if malformed input contains both.
        parsed["include"].difference_update(parsed["exclude"])
        includes[dimtag] = parsed["include"]
        excludes[dimtag] = parsed["exclude"]
    return includes, excludes

def _should_void_chunk_by_rules(dimtag:str, raw:bytes, rules:AllRules)->bool:
    rs = rules.by_dim.get(dimtag)
    if not rs or not rs.enabled: return True
    # Biomes
    if rs.biome_whitelist and not _chunk_has_biome_id(raw,set(rs.biome_whitelist)): return False
    if rs.biome_blacklist and _chunk_has_biome_id(raw,set(rs.biome_blacklist)): return False
    # Entities
    w_ent, b_ent = _chunk_entities_flags(raw,set(rs.entity_whitelist), set(rs.entity_blacklist))
    if rs.entity_whitelist and not w_ent: return False
    if b_ent and not w_ent: return False
    # Blocks
    w_blk, b_blk = _chunk_blocks_flags(raw,set(rs.block_whitelist), set(rs.block_blacklist))
    if rs.block_whitelist and not w_blk: return False
    if b_blk and not w_blk: return False
    return True

# ---------------- NEI dump import (auto + manual) ----------------
def _try_load_nei_dump(path: Optional[Path]) -> Dict[str, str]:
    out={}
    if not path or not path.exists(): return out
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line=line.strip()
            if not line: continue
            if "Name:" in line:
                nm=line.split("Name:",1)[1]
                nm=nm.split("ID:")[0].strip().strip(".")
                key=nm.replace("tile.","").replace("item.","").replace(" ","_").lower()
                out[key]=nm
    except Exception:
        pass
    return out

def _nearest_instance_root_from_world(world:Path) -> Path:
    p=world
    for _ in range(7):
        if (p/".minecraft").exists(): return p
        if p.name.lower()==".minecraft": return p.parent
        p=p.parent
    return world.parent.parent

def _auto_find_nei_dump(world:Path) -> Optional[Path]:
    mc_root = world
    while mc_root.name.lower() != ".minecraft" and mc_root.parent != mc_root:
        mc_root = mc_root.parent
    candidates: List[Path] = []
    try:
        for sub in ("config/NEI/dumps", "NEI/dumps", "dumps", "config/nei/dumps"):
            p = mc_root/sub
            if p.exists():
                candidates += [x for x in p.glob("*.txt")]
        if not candidates:
            for p in mc_root.rglob("*.txt"):
                nm=p.name.lower()
                if ("dump" in nm or "id" in nm) and ("nei" in str(p.parent).lower() or "dump" in str(p.parent).lower()):
                    candidates.append(p)
    except Exception:
        pass
    if not candidates: return None
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return candidates[0]

# ---------------- Build new void chunk ----------------
def _build_void_chunk(
    cx_abs: int,
    cz_abs: int,
    biome_bytes: Optional[bytes],
    mark_void: bool,
    platform: Optional[Tuple[int, int, int, int, int, int]],
    keep_entities: bool,
    keep_tileentities: bool,
    source_raw: Optional[bytes],
    populated_flag: int = 1,
    biome16_bytes: Optional[bytes] = None,
) -> bytes:
    """Build a terrain-free 1.7.10 chunk while retaining non-terrain mod metadata."""
    entities = nbt.TAG_List(name="Entities", type=nbt.TAG_Compound)
    tile_entities = nbt.TAG_List(name="TileEntities", type=nbt.TAG_Compound)
    tile_ticks = nbt.TAG_List(name="TileTicks", type=nbt.TAG_Compound)
    biome16_tag = None
    extra_level_tags = []
    extra_root_tags = []

    if source_raw:
        try:
            source = nbt.NBTFile(buffer=io.BytesIO(source_raw))
            source_level = source["Level"]
            if keep_entities and "Entities" in source_level:
                if not isinstance(source_level["Entities"], nbt.TAG_List):
                    raise ValueError("Level.Entities has an unexpected NBT type")
                entities = copy.deepcopy(source_level["Entities"])

            if "Biomes16v2" in source_level:
                candidate = source_level["Biomes16v2"]
                if not isinstance(candidate, nbt.TAG_Byte_Array) or len(candidate.value) != 512:
                    raise ValueError("Level.Biomes16v2 must be a 512-byte array")
                biome16_tag = copy.deepcopy(candidate)

            rewritten_names = {
                "xPos", "zPos", "LastUpdate", "InhabitedTime",
                "TerrainPopulated", "LightPopulated", "Populated",
                "HeightMap", "Biomes", "Biomes16v2", "Entities",
                "TileEntities", "TileTicks", "Sections", "Voider",
            }
            for tag in source_level.tags:
                if getattr(tag, "name", "") not in rewritten_names:
                    extra_level_tags.append(copy.deepcopy(tag))
            for tag in source.tags:
                if getattr(tag, "name", "") != "Level":
                    extra_root_tags.append(copy.deepcopy(tag))
        except Exception as exc:
            raise ValueError(f"Cannot safely preserve source chunk metadata: {exc}") from exc

    if biome16_bytes is not None:
        if len(biome16_bytes) != 512:
            raise ValueError("Biomes16v2 override must contain exactly 512 bytes")
        biome16_tag = nbt.TAG_Byte_Array(name="Biomes16v2")
        biome16_tag.value = list(biome16_bytes)

    root = nbt.NBTFile()
    level = nbt.TAG_Compound(name="Level")
    root.tags.append(level)
    root.tags.extend(extra_root_tags)

    level.tags.extend([
        nbt.TAG_Int(name="xPos", value=cx_abs),
        nbt.TAG_Int(name="zPos", value=cz_abs),
        nbt.TAG_Long(name="LastUpdate", value=0),
        nbt.TAG_Long(name="InhabitedTime", value=0),
        nbt.TAG_Byte(name="TerrainPopulated", value=int(populated_flag)),
        nbt.TAG_Byte(name="LightPopulated", value=int(populated_flag)),
    ])

    heightmap = nbt.TAG_Int_Array(name="HeightMap")
    heightmap.value = [0] * 256
    level.tags.append(heightmap)

    # GTNH's 16-bit array is authoritative when present. Do not add a
    # contradictory vanilla Plains/Hell fallback alongside Biomes16v2.
    if biome16_tag is None and biome_bytes is not None and len(biome_bytes) == 256:
        biome_tag = nbt.TAG_Byte_Array(name="Biomes")
        biome_tag.value = list(biome_bytes)
        level.tags.append(biome_tag)
    if biome16_tag is not None:
        level.tags.append(biome16_tag)

    level.tags.extend([entities, tile_entities, tile_ticks])

    sections = nbt.TAG_List(name="Sections", type=nbt.TAG_Compound)
    level.tags.append(sections)

    zeros4096 = [0] * 4096
    zeros2048 = [0] * 2048
    skylight_full = [0xFF] * 2048
    section_blocks = []
    section_data = []
    section_add = []

    for y in range(16):
        section = nbt.TAG_Compound()
        section.tags.append(nbt.TAG_Byte(name="Y", value=y))
        blocks = nbt.TAG_Byte_Array(name="Blocks")
        blocks.value = zeros4096.copy()
        data = nbt.TAG_Byte_Array(name="Data")
        data.value = zeros2048.copy()
        sky_light = nbt.TAG_Byte_Array(name="SkyLight")
        sky_light.value = skylight_full.copy()
        block_light = nbt.TAG_Byte_Array(name="BlockLight")
        block_light.value = zeros2048.copy()
        section.tags.extend([blocks, data, sky_light, block_light])
        sections.tags.append(section)
        section_blocks.append(blocks)
        section_data.append(data)
        section_add.append(None)

    if mark_void:
        level.tags.append(nbt.TAG_Byte(name="Voider", value=1))

    # Preserve mod-specific Level tags (e.g. GTNH POSTEA and V) that do not
    # encode the removed block/biome/entity data.
    level.tags.extend(extra_level_tags)

    if platform is not None:
        px, pz, py, size, block_id, meta = platform
        center_cx = px // 16
        center_cz = pz // 16
        if center_cx == cx_abs and center_cz == cz_abs and 0 <= py <= 255:
            half = size // 2
            x0, z0 = px - half, pz - half
            section_y = py // 16
            y_in_section = py % 16
            for dx in range(size):
                for dz in range(size):
                    world_x = x0 + dx
                    world_z = z0 + dz
                    if (world_x // 16) == cx_abs and (world_z // 16) == cz_abs:
                        local_x = world_x & 15
                        local_z = world_z & 15
                        index = (y_in_section * 256) + (local_z * 16) + local_x
                        section_blocks[section_y].value[index] = block_id & 0xFF
                        _nibble_set(section_data[section_y].value, index, meta & 0x0F)
                        high = (block_id >> 8) & 0x0F
                        if high:
                            if section_add[section_y] is None:
                                add_tag = nbt.TAG_Byte_Array(name="Add")
                                add_tag.value = [0] * 2048
                                sections[section_y].tags.append(add_tag)
                                section_add[section_y] = add_tag
                            _nibble_set(section_add[section_y].value, index, high)

    output = io.BytesIO()
    root.write_file(buffer=output)
    return output.getvalue()

# ---------------- Backup-biomes helpers ----------------
def _read_biomes_from_region(region_path: Path, idx: int) -> Optional[bytes]:
    try:
        with open(region_path, "rb") as fp:
            offsets, _ = _read_header(fp)
            entry = offsets[idx]
            if entry == 0:
                return None
            off, cnt = _parse_loc(entry)
            raw = _read_chunk_payload(fp, off, cnt)  # tolerant
            if not raw:
                return None
            nb = nbt.NBTFile(buffer=io.BytesIO(raw))
            lvl = nb["Level"]
            if "Biomes" in lvl and isinstance(lvl["Biomes"], nbt.TAG_Byte_Array):
                b = bytes(lvl["Biomes"].value)
                if len(b) == 256:
                    return b
            return None
    except Exception:
        return None

def _copy_biomes_from_backup_if_available(region_path:Path, idx:int)->Optional[bytes]:
    bak = region_path.with_suffix(region_path.suffix+".bak")
    if not bak.exists(): return None
    return _read_biomes_from_region(bak, idx)


def _read_biomes16v2_from_region(region_path: Path, idx: int) -> Optional[bytes]:
    try:
        with region_path.open("rb") as fp:
            offsets, _ = _read_header(fp)
            entry = offsets[idx]
            if entry == 0:
                return None
            off, count = _parse_loc(entry)
            raw = _read_chunk_payload(fp, off, count)
            return _get_biomes16v2(raw) if raw else None
    except Exception:
        return None


def _copy_biomes16v2_from_backup_if_available(region_path: Path, idx: int) -> Optional[bytes]:
    backup = region_path.with_suffix(region_path.suffix + ".bak")
    if not backup.is_file():
        return None
    return _read_biomes16v2_from_region(backup, idx)

# ---------------- Filters ----------------
def _bbox_ok(cx_abs:int, cz_abs:int, bbox:Optional[Tuple[int,int,int,int]])->bool:
    if not bbox: return True
    x1,z1,x2,z2 = bbox
    minx,minz = cx_abs*16, cz_abs*16
    maxx,maxz = minx+15, minz+15
    return not (maxx < x1 or maxz < z1 or minx > x2 or minz > z2)

def _radius_ok(cx_abs:int, cz_abs:int, center:Optional[Tuple[int,int]], r:int)->bool:
    if not center or r<=0: return True
    cx,cz = center
    minx,minz = cx_abs*16, cz_abs*16
    maxx,maxz = minx+15, minz+15
    dx = 0 if minx<=cx<=maxx else min(abs(cx-minx), abs(cx-maxx))
    dz = 0 if minz<=cz<=maxz else min(abs(cz-minz), abs(cz-maxz))
    return dx*dx + dz*dz <= r*r

def _platform_intersects_chunk(platform, cx_abs:int, cz_abs:int)->bool:
    if not platform:
        return False
    px, pz, _py, size, _bid, _meta = platform
    half = size // 2
    plat_x1, plat_x2 = px - half, px + half
    plat_z1, plat_z2 = pz - half, pz + half
    chunk_x1, chunk_x2 = cx_abs * 16, cx_abs * 16 + 15
    chunk_z1, chunk_z2 = cz_abs * 16, cz_abs * 16 + 15
    return not (plat_x2 < chunk_x1 or chunk_x2 < plat_x1 or
                plat_z2 < chunk_z1 or chunk_z2 < plat_z1)

# ---------------- Per-region worker ----------------
def _process_region_inplace(mca:Path, dimtag:str, bbox, center, radius_blocks:int,
                    include_void:bool, platform, keep_entities:bool, keep_tileentities:bool,
                    skip_if_entities:bool, skip_if_tileentities:bool, mark_void:bool,
                    dry_run:bool, strict:bool, sel_include:Set[Tuple[int,int]],
                    sel_exclude:Set[Tuple[int,int]], restore_biomes_from_bak:bool,
                    nether_cosmetic_placeholder:bool,
                    rules:Optional[AllRules]=None,
                    cancel_event:Optional[threading.Event]=None,
                    _region_coords:Optional[Tuple[int,int]]=None,
                    _source_region:Optional[Path]=None)->Tuple[str,int,int]:
    rx, rz = _region_coords or _region_coords_from_name(mca)
    changed=0; scanned=0
    pop_flag = 1 if dimtag == "region" else 0  # Overworld stays populated; others must repopulate
    with open(mca,"r+b") as fp:
        for idx, off, cnt in _iter_chunk_locations(fp):
            if cancel_event is not None and cancel_event.is_set(): break
            cx_local = idx%32; cz_local = idx//32
            cx_abs = rx*32 + cx_local; cz_abs = rz*32 + cz_local
            if not _bbox_ok(cx_abs, cz_abs, bbox): continue
            if not _radius_ok(cx_abs, cz_abs, center, radius_blocks): continue
            coord=(cx_abs, cz_abs)
            if coord in sel_exclude: continue
            if sel_include and coord not in sel_include: continue
            raw = _read_chunk_payload(fp, off, cnt)
            if not raw:
                if strict: raise IOError(f"Bad payload at {mca.name}:{idx}")
                # can't safely void or edit if we can't parse; skip counting as scanned
                continue
            scanned += 1
            parsed = _nbt_from_raw(raw)
            if parsed is None or "Level" not in parsed:
                if strict:
                    raise IOError(f"Unreadable chunk NBT at {mca.name}:{idx}")
                continue
            level = parsed["Level"]
            expected_coords = (cx_abs, cz_abs)
            actual_coords = (
                (int(level["xPos"].value), int(level["zPos"].value))
                if "xPos" in level and "zPos" in level else None
            )
            if actual_coords != expected_coords:
                if strict:
                    raise IOError(
                        f"Chunk coordinate mismatch at {mca.name}:{idx}: "
                        f"found {actual_coords!r}, expected {expected_coords!r}"
                    )
                continue

            if not _should_void_chunk_by_rules(dimtag, raw, rules or _GLOBAL_RULES):
                continue

            te,en = _has_entities(raw)
            if skip_if_entities and en: continue
            # A TileEntity without its backing block is corrupt game state.
            # "Keep TileEntities" therefore means leave the entire chunk intact.
            if (skip_if_tileentities or keep_tileentities) and te: continue

            already_void = _is_void_chunk(raw)
            platform_needed = _platform_intersects_chunk(platform, cx_abs, cz_abs)
            if already_void and not platform_needed and not include_void:
                continue

            # Preserve GTNH's 16-bit biome array when present. Restore it
            # from .mca.bak when requested and missing in the source chunk.
            biome16_src = _get_biomes16v2(raw)
            if biome16_src is None and restore_biomes_from_bak:
                biome16_src = _copy_biomes16v2_from_backup_if_available(_source_region or mca, idx)

            biome_src = _get_biomes(raw)
            if biome_src is None and biome16_src is None and restore_biomes_from_bak:
                biome_src = _copy_biomes_from_backup_if_available(_source_region or mca, idx)

            if dimtag == "region":
                # Overworld: preserve either existing biome representation;
                # write Plains(1) only when neither representation exists.
                if biome16_src is not None:
                    biome_bytes_to_write = None
                else:
                    biome_bytes_to_write = (
                        biome_src if (biome_src is not None and len(biome_src) == 256)
                        else bytes([PLAINS_ID] * 256)
                    )
            else:
                # Non-overworld: preserve either representation; if both are
                # missing, normally omit so the provider can recompute.
                if biome16_src is not None:
                    biome_bytes_to_write = None
                elif biome_src is not None and len(biome_src) == 256:
                    biome_bytes_to_write = biome_src
                elif nether_cosmetic_placeholder and dimtag.startswith("DIM-1/"):
                    biome_bytes_to_write = bytes([HELL_ID] * 256)
                else:
                    biome_bytes_to_write = None

            if dry_run:
                changed += 1
                continue

            new = _build_void_chunk(cx_abs, cz_abs, biome_bytes_to_write, mark_void,
                        platform if platform_needed else None,
                        keep_entities, keep_tileentities, raw,
                        pop_flag, biome16_src)
            new_off, new_cnt = _write_chunk_payload(fp, off, cnt, new)
            _update_header(fp, idx, new_off, new_cnt)
            changed += 1
    return (mca.name, scanned, changed)

def _fsync_directory(path:Path)->None:
    """Best-effort durability barrier after an atomic replacement."""
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass

def _process_region(mca:Path, *args, **kwargs)->Tuple[str,int,int]:
    """Process a copied region and atomically replace the original on success."""
    if kwargs.get("dry_run", False):
        return _process_region_inplace(mca, *args, **kwargs)

    coords = _region_coords_from_name(mca)
    temp = mca.with_name(f".{mca.name}.voider-{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        before = mca.stat()
        source_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if temp.exists():
            temp.unlink()
        shutil.copy2(mca, temp)
        work_kwargs = dict(kwargs)
        work_kwargs["_region_coords"] = coords
        work_kwargs["_source_region"] = mca
        copied = mca.stat()
        copied_signature = (copied.st_dev, copied.st_ino, copied.st_size, copied.st_mtime_ns)
        if copied_signature != source_signature:
            raise RuntimeError(f"Region changed while its working copy was being prepared: {mca}")
        _tmp_name, scanned, changed = _process_region_inplace(temp, *args, **work_kwargs)
        active_cancel = work_kwargs.get("cancel_event")
        if active_cancel is not None and active_cancel.is_set():
            raise InterruptedError(f"Cancelled before committing region {mca.name}")
        if changed:
            current = mca.stat()
            current_signature = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
            if current_signature != source_signature:
                raise RuntimeError(f"Region changed concurrently; refusing replacement: {mca}")
            os.replace(temp, mca)
            _fsync_directory(mca.parent)
        else:
            temp.unlink(missing_ok=True)
        return mca.name, scanned, changed
    except BaseException:
        temp.unlink(missing_ok=True)
        raise

# ---------------- Discovery / cache ----------------
def _get_drive_letters_windows()->List[Path]:
    import string
    try:
        from ctypes import windll
        bitmask = windll.kernel32.GetLogicalDrives()
        drives=[]
        for i,letter in enumerate(string.ascii_uppercase):
            if bitmask & (1<<i):
                drives.append(Path(f"{letter}:/"))
        return drives
    except Exception:
        return [Path("C:/"), Path("D:/"), Path("E:/"), Path("F:/")]

def _candidate_roots()->List[Path]:
    roots: List[Path] = []
    appdata = os.environ.get("APPDATA")
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        roots.append(Path(userprofile)/"AppData"/"Roaming"/".minecraft"/"saves")
    if appdata:
        for inst_root in (Path(appdata)/"PrismLauncher"/"instances",
                          Path(appdata)/"MultiMC"/"instances"):
            if inst_root.exists():
                for inst in inst_root.iterdir():
                    roots.append(inst/".minecraft"/"saves")
        at = Path(appdata)/"ATLauncher"/"instances"
        if at.exists():
            for inst in at.iterdir():
                roots.append(inst/"saves")
    if userprofile:
        tech = Path(userprofile)/"AppData"/"Roaming"/".technic"/"modpacks"
        if tech.exists():
            for pack in tech.iterdir():
                roots.append(pack/"saves")
        curse = Path(userprofile)/"curseforge"/"minecraft"/"Instances"
        if curse.exists():
            for inst in curse.iterdir():
                roots.append(inst/"saves")
    if os.name=="nt":
        roots.extend(_get_drive_letters_windows())
    else:
        home = Path.home()
        roots.extend([
            home/".minecraft"/"saves",
            home/"CrucibleServers",
            home/".local"/"share"/"PrismLauncher"/"instances",
            home/".local"/"share"/"prismlauncher"/"instances",
            home/".var"/"app"/"org.prismlauncher.PrismLauncher"/"data"/"PrismLauncher"/"instances",
            home/".local"/"share"/"MultiMC"/"instances",
            home/".local"/"share"/"multimc"/"instances",
            home/"PrismLauncher"/"instances",
            home/"Games"/"PrismLauncher"/"instances",
            Path("/mnt"), Path("/media"), Path("/run/media"), Path("/Volumes"),
        ])

    out=[]; seen=set()
    for r in roots:
        try: rp=r.resolve()
        except Exception: rp=r
        if rp.exists() and str(rp) not in seen:
            seen.add(str(rp)); out.append(rp)
    return out

def _looks_like_world_dir(p:Path)->bool:
    try:
        if not p.is_dir(): return False
        p = _resolve_world_path(p)
        if not (p/"level.dat").is_file(): return False
        region_dirs=[d for d in p.rglob("region") if d.is_dir() and d.parent.name not in ("entities","poi")]
        for r in region_dirs:
            if any(f.suffix==".mca" and f.name.startswith("r.") for f in r.iterdir()):
                return True
        return False
    except Exception:
        return False

def _scan_for_worlds_under(root:Path, found:Dict[str,str], max_depth:int, cur_depth:int=0, status_cb:Optional[Callable[[str],None]]=None):
    if cur_depth > max_depth: return
    try:
        if _looks_like_world_dir(root):
            candidate = _resolve_world_path(root)
            found[str(candidate)] = str(candidate)
            return
    except Exception:
        pass
    try:
        for child in root.iterdir():
            if not child.is_dir(): continue
            name=child.name.lower()
            if name in ("windows","program files","program files (x86)","$recycle.bin","system volume information","intel","nvidia","programdata"):
                continue
            if name.startswith(".") and cur_depth>0: continue
            if status_cb and cur_depth<=2: status_cb(f"Scanning: {child}")
            _scan_for_worlds_under(child, found, max_depth, cur_depth+1, status_cb)
    except Exception:
        pass

def _cache_path()->Path:
    local=os.environ.get("LOCALAPPDATA")
    if local:
        p=Path(local)/"GTNHVoider"; p.mkdir(parents=True, exist_ok=True)
        return p/"worlds-cache.json"
    return Path.home()/".gtnh_voider_worlds_cache.json"

def discover_minecraft_worlds(force_rescan:bool=False, max_depth:int=4, status_cb:Optional[Callable[[str],None]]=None)->Dict[str,str]:
    cache_file=_cache_path()
    if not force_rescan and cache_file.exists():
        try:
            data=json.loads(cache_file.read_text(encoding="utf-8"))
            data={k:v for k,v in data.items() if Path(v).exists() and (Path(v)/"level.dat").exists()}
            if data:
                if status_cb: status_cb("Loaded cached worlds list.")
                return data
        except Exception:
            pass

    found: Dict[str,str] = {}
    roots=_candidate_roots()
    if status_cb: status_cb(f"Scanning {len(roots)} roots for worlds…")
    for root in roots:
        try:
            if root.is_file(): continue
            if (root.name.lower()=="saves") and root.exists():
                for w in root.iterdir():
                    if _looks_like_world_dir(w): found[str(w)] = str(w)
            else:
                _scan_for_worlds_under(root, found, max_depth=max_depth, status_cb=status_cb)
        except Exception:
            continue

    try: cache_file.write_text(json.dumps(found, indent=2), encoding="utf-8")
    except Exception: pass
    if status_cb: status_cb(f"World scan complete. Found {len(found)}.")
    return found

# ---------------- Provider-safe biome reset (maintenance) ----------------
def _reset_biomes_region_inplace(mca: Path, tag: str, rx: int, rz: int) -> int:
    changed = 0
    with mca.open("r+b") as fp:
        for idx, off, cnt in _iter_chunk_locations(fp):
            raw = _read_chunk_payload(fp, off, cnt)
            if not raw:
                raise IOError(f"Unreadable chunk payload at {mca.name}:{idx}")
            nb = _nbt_from_raw(raw)
            if nb is None or "Level" not in nb:
                raise IOError(f"Unreadable chunk NBT at {mca.name}:{idx}")
            level = nb["Level"]
            expected = (rx * 32 + idx % 32, rz * 32 + idx // 32)
            actual = (
                (int(level["xPos"].value), int(level["zPos"].value))
                if "xPos" in level and "zPos" in level else None
            )
            if actual != expected:
                raise IOError(f"Chunk coordinate mismatch at {mca.name}:{idx}: {actual!r} != {expected!r}")

            if tag == "region":
                current = None
                if "Biomes" in level and isinstance(level["Biomes"], nbt.TAG_Byte_Array):
                    current = bytes(level["Biomes"].value)
                extended = level.get("Biomes16v2")
                has_extended = (
                    isinstance(extended, nbt.TAG_Byte_Array)
                    and len(extended.value) == 512
                )
                if (current is not None and len(current) == 256) or has_extended:
                    continue
                level.tags = [
                    item for item in level.tags
                    if getattr(item, "name", "") not in {"Biomes", "Biomes16v2"}
                ]
                biome_tag = nbt.TAG_Byte_Array(name="Biomes")
                biome_tag.value = [PLAINS_ID] * 256
                level.tags.append(biome_tag)
            else:
                level.tags = [
                    item for item in level.tags
                    if getattr(item, "name", "") not in {"Biomes", "Biomes16v2"}
                ]

            reset_names = {"TerrainPopulated", "LightPopulated", "Populated", "HeightMap", "InhabitedTime"}
            level.tags = [item for item in level.tags if getattr(item, "name", "") not in reset_names]
            heightmap = nbt.TAG_Int_Array(name="HeightMap")
            heightmap.value = [0] * 256
            level.tags.extend([
                nbt.TAG_Byte(name="TerrainPopulated", value=0),
                nbt.TAG_Byte(name="LightPopulated", value=0),
                nbt.TAG_Byte(name="Populated", value=0),
                heightmap,
                nbt.TAG_Long(name="InhabitedTime", value=0),
            ])
            output = io.BytesIO()
            nb.write_file(buffer=output)
            new_off, new_count = _write_chunk_payload(fp, off, cnt, output.getvalue())
            _update_header(fp, idx, new_off, new_count)
            changed += 1
    return changed


def reset_biomes(world: Path, dims: Optional[List[str]] = None, *, progress_cb=None) -> int:
    """Safely reset selected biome metadata with a full verified backup and rollback staging."""
    world = _resolve_world_path(world)
    if not world.is_dir() or not (world / "level.dat").is_file():
        raise ValueError(f"Not a Minecraft world folder: {world}")
    region_dirs = _find_region_dirs(world)
    requested = {item.strip() for item in dims} if dims is not None else None
    if requested is not None:
        available = {_dim_tag(world, directory) for directory in region_dirs}
        unknown = requested - available
        if unknown:
            raise ValueError(f"Unknown dimension tag(s): {', '.join(sorted(unknown))}")
        region_dirs = [directory for directory in region_dirs if _dim_tag(world, directory) in requested]
    files = [
        (_dim_tag(world, directory), path, _region_coords_from_name(path))
        for directory in region_dirs
        for path in sorted(directory.glob("r.*.*.mca"))
    ]
    if not files:
        raise RuntimeError("No region files found in the selected biome-reset scope.")

    with _world_write_guard(world):
        backup = _backup_world_zip(world)
        staged = []
        committed = []
        changed = 0
        try:
            # Stage every output first. A malformed chunk aborts before any
            # original region file is replaced.
            for index, (tag, mca, (rx, rz)) in enumerate(files, 1):
                before = mca.stat()
                signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                temp = mca.with_name(
                    f".{mca.name}.voider-biome-{os.getpid()}-{threading.get_ident()}.tmp"
                )
                temp.unlink(missing_ok=True)
                shutil.copy2(mca, temp)
                try:
                    file_changed = _reset_biomes_region_inplace(temp, tag, rx, rz)
                    staged.append((mca, temp, signature, file_changed))
                    changed += file_changed
                    if progress_cb:
                        progress_cb(index, len(files), f"Staged {tag}: {mca.name} ({file_changed} chunk(s))")
                except BaseException:
                    temp.unlink(missing_ok=True)
                    raise

            for mca, temp, signature, file_changed in staged:
                if not file_changed:
                    temp.unlink(missing_ok=True)
                    continue
                current = mca.stat()
                current_signature = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
                if current_signature != signature:
                    raise RuntimeError(f"Region changed during biome reset; refusing replacement: {mca}")
                rollback = mca.with_name(
                    f".{mca.name}.voider-rollback-{os.getpid()}-{threading.get_ident()}"
                )
                if rollback.exists():
                    raise RuntimeError(f"Refusing to overwrite an existing rollback file: {rollback}")
                try:
                    shutil.copy2(mca, rollback)
                    os.replace(temp, mca)
                except BaseException:
                    rollback.unlink(missing_ok=True)
                    raise
                # Record the replacement before any operation that might fail,
                # including the durability barrier, so the outer exception path
                # can always restore the original region.
                committed.append((mca, rollback))
                _fsync_directory(mca.parent)

            audit = verify_world(world)
            if not audit["ok"]:
                raise RuntimeError(
                    f"Post-reset verifier found {len(audit['errors'])} error(s): "
                    + "; ".join(audit["errors"][:5])
                )
            for _mca, rollback in committed:
                try:
                    rollback.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    print(
                        f"[warning] Biome reset succeeded but could not remove rollback file "
                        f"{rollback}: {cleanup_exc}",
                        file=sys.stderr,
                    )
            return changed
        except BaseException as exc:
            for mca, rollback in reversed(committed):
                try:
                    if rollback.exists():
                        os.replace(rollback, mca)
                        _fsync_directory(mca.parent)
                except OSError as restore_exc:
                    print(f"[critical] rollback failed for {mca}: {restore_exc}", file=sys.stderr)
            for _mca, temp, _signature, _count in staged:
                temp.unlink(missing_ok=True)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError(
                f"Biome reset failed; changes were rolled back where possible. "
                f"Keep the verified backup for recovery: {backup}"
            ) from exc

# ---------------- Backups ----------------
def _backup_world_zip(world:Path):
    """Create a complete, hash-manifested ZIP backup and verify every member."""
    world = Path(world).expanduser().resolve()
    if not world.is_dir() or not (world / "level.dat").is_file():
        raise ValueError(f"Not a Minecraft world folder: {world}")
    manifest_name = ".voider-backup-manifest.json"
    # session.lock is an ephemeral OS coordination file. It must never be
    # restored from a backup, and Windows byte-range locking prevents reading
    # the locked byte while the write guard is held.
    transient_lock_name = "session.lock"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = world.parent / f"{world.name}.void-backup-{stamp}.zip"
    serial = 1
    while destination.exists():
        destination = world.parent / f"{world.name}.void-backup-{stamp}-{serial}.zip"
        serial += 1
    if (world / manifest_name).exists():
        raise RuntimeError(f"World contains reserved backup-manifest path: {manifest_name}")

    def inventory():
        result = {}
        for root, directories, filenames in os.walk(world, followlinks=False):
            root_path = Path(root)
            for name in list(directories):
                item = root_path / name
                if item.is_symlink():
                    raise RuntimeError(f"World backup refuses symlinked directory: {item}")
            for name in filenames:
                item = root_path / name
                if item.is_symlink():
                    raise RuntimeError(f"World backup refuses symlinked file: {item}")
                relative = item.relative_to(world).as_posix()
                if relative == transient_lock_name:
                    continue
                metadata = item.stat(follow_symlinks=False)
                if not statmod.S_ISREG(metadata.st_mode):
                    raise RuntimeError(f"World backup refuses non-regular file: {item}")
                result[relative] = (
                    metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns
                )
        return result

    def digest_stream(stream):
        digest = hashlib.sha256()
        size = 0
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        return digest.hexdigest(), size

    before = inventory()
    if "level.dat" not in before:
        raise RuntimeError("World backup inventory does not contain level.dat")
    temporary = destination.with_name(
        f".{destination.name}.partial-{os.getpid()}-{threading.get_ident()}"
    )
    file_manifest = {}
    try:
        with ZipFile(temporary, "x", ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for relative in sorted(before):
                source = world / Path(relative)
                digest = hashlib.sha256()
                size = 0
                info = zipfile.ZipInfo(relative)
                info.compress_type = ZIP_DEFLATED
                with source.open("rb") as input_stream, archive.open(info, "w") as output_stream:
                    while True:
                        block = input_stream.read(1024 * 1024)
                        if not block:
                            break
                        output_stream.write(block)
                        digest.update(block)
                        size += len(block)
                file_manifest[relative] = {"size": size, "sha256": digest.hexdigest()}
            manifest = {
                "format": "gtnh-world-voider-backup-v1",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "world_name": world.name,
                "file_count": len(file_manifest),
                "files": file_manifest,
            }
            archive.writestr(
                manifest_name,
                json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8"),
                compress_type=ZIP_DEFLATED,
            )

        after = inventory()
        if after != before:
            raise RuntimeError("World files changed during backup; backup is not a consistent snapshot")

        with ZipFile(temporary, "r") as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise IOError(f"Backup CRC verification failed at {bad_member}")
            expected_members = set(before) | {manifest_name}
            names = archive.namelist()
            if len(names) != len(expected_members) or set(names) != expected_members:
                raise IOError("Backup member inventory does not match the world inventory")
            loaded_manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
            if loaded_manifest.get("format") != "gtnh-world-voider-backup-v1":
                raise IOError("Backup manifest format is invalid")
            if set(loaded_manifest.get("files", {})) != set(before):
                raise IOError("Backup manifest file list is incomplete")
            for relative, expected in file_manifest.items():
                with archive.open(relative, "r") as stream:
                    actual_hash, actual_size = digest_stream(stream)
                if actual_size != expected["size"] or actual_hash != expected["sha256"]:
                    raise IOError(f"Backup SHA-256 verification failed for {relative}")
                with (world / Path(relative)).open("rb") as source_stream:
                    source_hash, source_size = digest_stream(source_stream)
                if source_size != actual_size or source_hash != actual_hash:
                    raise RuntimeError(f"World changed while backup was being verified: {relative}")

        if inventory() != before:
            raise RuntimeError("World inventory changed after backup verification")
        # Open read/write so Windows' _commit() path has a writable handle.
        with temporary.open("r+b") as fp:
            os.fsync(fp.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    with destination.open("rb") as stream:
        digest, _size = digest_stream(stream)
    print(f"World backup created and SHA-256 verified: {destination} ({digest})")
    return destination

def _per_file_backup(world:Path):
    for rdir in _find_region_dirs(world):
        for mca in sorted(p for p in rdir.iterdir() if p.suffix==".mca" and p.name.startswith("r.")):
            bak=mca.with_suffix(mca.suffix+".bak")
            if not bak.exists():
                bak.write_bytes(mca.read_bytes())

def _region_snapshot(world:Path)->Dict[Path, Tuple[int,int]]:
    snapshot={}
    for rdir in _find_region_dirs(world):
        for mca in rdir.glob("r.*.*.mca"):
            stat=mca.stat()
            snapshot[mca]=(stat.st_size, stat.st_mtime_ns)
    return snapshot

def verify_world(world:Path)->Dict[str, object]:
    """Read-only audit of sector ranges, overlaps, NBT, and chunk coordinates."""
    report: Dict[str, object] = {"world": str(world), "region_files": 0, "chunks": 0,
                                "valid_chunks": 0, "errors": [], "files": []}
    errors = report["errors"]
    files_report = report["files"]
    for rdir in _find_region_dirs(world):
        for mca in sorted(rdir.glob("r.*.*.mca")):
            info = {"path": str(mca.relative_to(world)), "chunks": 0, "valid": 0, "errors": []}
            file_errors = info["errors"]
            report["region_files"] += 1
            try:
                rx, rz = _region_coords_from_name(mca)
                with mca.open("rb") as fp:
                    offsets, _ = _read_header(fp)
                    fp.seek(0, os.SEEK_END); file_size = fp.tell()
                    if file_size < 2 * SECTOR_BYTES or file_size % SECTOR_BYTES:
                        file_errors.append(
                            f"{mca.name}: region file size is not a positive whole number of sectors: {file_size}"
                        )
                    occupied: Set[int] = set()
                    for idx, entry in enumerate(offsets):
                        if not entry: continue
                        info["chunks"] += 1; report["chunks"] += 1
                        off, count = _parse_loc(entry); label = f"{mca.name}:{idx}"
                        if off < 2 or count < 1 or (off + count) * SECTOR_BYTES > file_size:
                            file_errors.append(f"{label}: invalid sector range off={off} count={count}"); continue
                        sectors = set(range(off, off + count))
                        if occupied & sectors:
                            file_errors.append(f"{label}: overlapping sector allocation"); continue
                        occupied.update(sectors)
                        raw = _read_chunk_payload(fp, off, count)
                        parsed = _nbt_from_raw(raw) if raw else None
                        if parsed is None or "Level" not in parsed:
                            file_errors.append(f"{label}: unreadable NBT payload"); continue
                        level = parsed["Level"]
                        expected = (rx*32 + idx%32, rz*32 + idx//32)
                        actual = ((int(level["xPos"].value), int(level["zPos"].value))
                                  if "xPos" in level and "zPos" in level else None)
                        if actual != expected:
                            file_errors.append(f"{label}: coordinates {actual!r}, expected {expected!r}"); continue
                        info["valid"] += 1; report["valid_chunks"] += 1
            except Exception as exc:
                file_errors.append(f"{mca.name}: {type(exc).__name__}: {exc}")
            errors.extend(f"{info['path']}: {msg}" for msg in file_errors)
            files_report.append(info)
    report["ok"] = not errors
    return report

# ---------------- UI helpers: Tee + Tooltip ----------------
class _TeeIO:
    def __init__(self, stream, q:queue.Queue, tag:str):
        self._stream=stream; self._q=q; self._tag=tag; self._lock=threading.Lock()
    def write(self, s):
        with self._lock:
            try: self._stream.write(s)
            except Exception: pass
            for line in s.splitlines(True):
                self._q.put((self._tag, line))
    def flush(self):
        try: self._stream.flush()
        except Exception: pass

class Tooltip:
    _shift_down=False
    def __init__(self, widget, short:str, extended:str=""):
        self.widget=widget; self.short=short; self.ext=extended or short
        self.tip=None
        widget.bind("<Enter>", self._enter)
        widget.bind("<Leave>", self._leave)
        widget.bind("<Motion>", self._move)
        r = widget.winfo_toplevel()
        r.bind_all("<Shift_L>", lambda e: self._set_shift(True))
        r.bind_all("<KeyRelease-Shift_L>", lambda e: self._set_shift(False))
        r.bind_all("<Shift_R>", lambda e: self._set_shift(True))
        r.bind_all("<KeyRelease-Shift_R>", lambda e: self._set_shift(False))
    def _set_shift(self, on:bool): Tooltip._shift_down=on
    def _enter(self, _):
        import tkinter as tk
        text = self.ext if Tooltip._shift_down else self.short
        if self.tip: return
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.label = tk.Label(self.tip, text=text, justify="left", relief="solid", borderwidth=1,
                              background="#ffffe0", padx=6, pady=4, wraplength=420)
        self.label.pack(ipadx=1)
        self._move(_)
    def _move(self, e):
        if not self.tip: return
        x = e.x_root + 12; y = e.y_root + 12
        self.tip.wm_geometry(f"+{x}+{y}")
    def _leave(self, _):
        if self.tip:
            self.tip.destroy()
            self.tip=None

# ---------------- Help dialog ----------------
class HelpDialog:
    TEXT = (
        "GTNH Voider — Help\n\n"
        "General Flow:\n"
        "  1) Select a world (Browse… or Discover…).\n"
        "  2) (Optional) Chunk Selector to pick Include/Exclude regions per dimension.\n"
        "  3) Configure options. Start.\n\n"
        "Options (hover for short; hold Shift while hovering for more details & examples):\n"
        " • Dry run — Does NOT write files. Shows how many chunks would be voided.\n"
        " • Include already void — Also rewrites chunks that are already empty.\n"
        " • Keep Entities — Copy entity records to the new chunk.\n"
        " • Keep TileEntities — Leave any chunk containing TileEntities untouched.\n"
        " • Skip if Entities/TileEntities — Skip chunks that contain any.\n"
        " • Mark Void tag — Adds Level.Voider=1 to new chunks.\n"
        " • Restore Biomes from .bak — If Biomes are missing, try copying from .mca.bak.\n"
        " • Nether placeholder — When missing in DIM-1, temporarily set Hell biome to avoid 'Ocean' HUD until repop.\n"
        " • Workers — Parallel region writers (4–8 recommended on SSDs).\n\n"
        "Biome Handling:\n"
        " • Overworld (region/) — If Biomes[256] exists, keep it; if missing, write Plains (id=1) to all 256 cells.\n"
        " • Other dimensions — Preserve Biomes when present; if missing, OMIT so the provider refills on next load.\n"
        " • Cosmetic option: Nether placeholder writes Hell (id=8) when missing; provider may still overwrite on repop.\n"
        " • Maintenance: Tools → Reset Biomes (All / Non-Overworld) uses the strict provider-safe policy above.\n\n"
        "Rules (.voider-rules.json):\n"
        " • Whitelists require a match; blacklists block unless whitelisted.\n"
        " • Blocks — Prefer numeric IDs (1.7.x). Entities — 'minecraft:zombie', 'TwilightForest.minotaur'.\n"
        " • Biomes — numeric 0..65535.\n\n"
        "NEI Dump Import:\n"
        " • Auto-scans your instance for a recent NEI “Dump ID Map” and loads it for the Rules Editor.\n\n"
        "Tips:\n"
        " • Always make a backup before bulk changes. Dry run first. 4–8 workers are good on SSDs.\n"
        " • If a mod-dimension shows wrong biomes, run Reset Biomes for that dim and reload the world.\n"
        "Project / Source:\n"
        " • GitHub: https://github.com/Roaring1/-MinecraftWorld-Voider-PrismLauncher-GTNH-1.7.10-v2.8.0-\n"
    )
    def __init__(self, master):
        import tkinter as tk
        from tkinter import ttk
        import webbrowser
        self.top = tk.Toplevel(master)
        self.top.title("Help")
        self.top.geometry("820x640")
        frm = ttk.Frame(self.top); frm.pack(fill="both", expand=True, padx=10, pady=10)
        txt = tk.Text(frm, wrap="word")
        txt.insert("1.0", self.TEXT); txt.configure(state="disabled")
        vs = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vs.set)
        txt.grid(row=0, column=0, sticky="nsew"); vs.grid(row=0, column=1, sticky="ns")
        frm.rowconfigure(0, weight=1); frm.columnconfigure(0, weight=1)
        btns = ttk.Frame(self.top); btns.pack(fill="x", padx=10, pady=(6, 2))
        ttk.Button(btns, text="Open GitHub", command=lambda: webbrowser.open(
            "https://github.com/Roaring1/-MinecraftWorld-Voider-PrismLauncher-GTNH-1.7.10-v2.8.0-"
        )).pack(side="left")
        ttk.Button(btns, text="Close", command=self.top.destroy).pack(side="right")

# ---------------- Rules Editor (with auto NEI) ----------------
class RulesEditorDialog:
    def __init__(self, master, world: Path, rules: AllRules, dims: List[str], nei_autoload: Optional[Dict[str,str]]):
        import tkinter as tk
        from tkinter import ttk
        self.world = world
        self.rules = rules
        self.dims = dims
        self.nei_map: Dict[str,str] = nei_autoload or {}

        self.top = tk.Toplevel(master)
        self.top.title("Voider Rules Editor")
        self.top.geometry("960x640")
        self.top.minsize(820, 520)

        frm = ttk.Frame(self.top); frm.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frm, text="Dimension:").grid(row=0, column=0, sticky="w")
        self.dim_var = tk.StringVar(value=dims[0] if dims else "")
        self.dim_cb = ttk.Combobox(frm, state="readonly", values=dims, textvariable=self.dim_var, width=36)
        self.dim_cb.grid(row=0, column=1, sticky="w", padx=(6,8))
        self.dim_cb.bind("<<ComboboxSelected>>", lambda _e=None: self._load_dim())
        Tooltip(self.dim_cb, "Pick which dimension’s rules to edit.",
                "Example tags: 'region' (Overworld), 'DIM-1/region' (Nether), 'DIM7/region' (Twilight Forest).")

        self.enabled_var = tk.BooleanVar(value=True)
        en = ttk.Checkbutton(frm, text="Enable rules for this dimension", variable=self.enabled_var)
        en.grid(row=0, column=2, sticky="w")
        Tooltip(en, "Toggle rules application for this dimension.",
                "When OFF, the voider ignores all rules for the selected dimension and voids by default behavior only.")

        box = ttk.Labelframe(frm, text="Lookup (optional — NEI Dump ID Map)")
        box.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(8,8))
        for i in range(3): box.columnconfigure(i, weight=1)
        self.search_var = tk.StringVar()
        ttk.Label(box, text="Search:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ent = ttk.Entry(box, textvariable=self.search_var, width=40)
        ent.grid(row=0, column=1, sticky="we", padx=(0,6))
        ttk.Button(box, text="Import NEI dump…", command=self._import_nei).grid(row=0, column=2, sticky="e", padx=6)
        Tooltip(ent, "Type to filter NEI names.",
                "Type part of a name (e.g. 'sapphire ore').")
        self.listbox = tk.Listbox(box, height=8)
        self.listbox.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=6, pady=(0,6))
        box.rowconfigure(1, weight=1)
        self.search_var.trace("w", lambda *_: self._refresh_search())

        def mk_editor(row, label):
            lf = ttk.Labelframe(frm, text=label); lf.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(6,6))
            for i in range(3): lf.columnconfigure(i, weight=1)
            return lf

        blw = mk_editor(2, "Blocks (numeric IDs preferred in 1.7) — one per line")
        bew = mk_editor(3, "Entities (minecraft:zombie, TwilightForest.minotaur) — one per line")
        bmw = mk_editor(4, "Biome IDs (0..65535) — one per line")

        def mk_text(lf, title):
            import tkinter as tk
            from tkinter import ttk
            ttk.Label(lf, text=title).grid(row=0, column=0, sticky="w", padx=6, pady=4)
            txt = tk.Text(lf, height=6)
            vs = ttk.Scrollbar(lf, orient="vertical", command=txt.yview)
            txt.configure(yscrollcommand=vs.set)
            txt.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=(0,6))
            vs.grid(row=1, column=2, sticky="ns", pady=(0,6))
            lf.rowconfigure(1, weight=1)
            return txt

        self.block_whitelist_txt = mk_text(blw, "Whitelist")
        self.block_blacklist_txt = mk_text(blw, "Blacklist")
        self.entity_whitelist_txt = mk_text(bew, "Whitelist")
        self.entity_blacklist_txt = mk_text(bew, "Blacklist")
        self.biome_whitelist_txt = mk_text(bmw, "Whitelist")
        self.biome_blacklist_txt = mk_text(bmw, "Blacklist")

        Tooltip(self.block_whitelist_txt, "Blocks allowed to trigger void when present.",
                "If whitelist is non-empty, at least one must be present for the chunk to be voided.")
        Tooltip(self.block_blacklist_txt, "Blocks that prevent voiding unless whitelisted.",
                "If a chunk contains a blacklisted block and it isn’t whitelisted, the chunk is skipped.")
        Tooltip(self.entity_whitelist_txt, "Entities required to trigger void when present.",
                "Example: 'minecraft:zombie' or 'TwilightForest.minotaur'. Use lowercase.")
        Tooltip(self.entity_blacklist_txt, "Entities that prevent voiding unless whitelisted.",
                "If a chunk contains a blacklisted entity and it isn’t whitelisted, the chunk will be skipped.")
        Tooltip(self.biome_whitelist_txt, "Biomes that allow voiding when present.",
                "Enter numeric biome IDs (0–65535).")
        Tooltip(self.biome_blacklist_txt, "Biomes that prevent voiding.",
                "If any of these biome IDs are present, the chunk is skipped (unless also whitelisted).")

        btns = ttk.Frame(frm); btns.grid(row=5, column=0, columnspan=3, sticky="we", pady=(8,0))
        ttk.Button(btns, text="Save", command=self._save).pack(side="right")
        ttk.Button(btns, text="Close", command=self.top.destroy).pack(side="right", padx=(0,8))

        self._load_dim()
        self._refresh_search()

    def _cur_rules(self) -> RuleSet:
        dim = self.dim_var.get()
        if dim not in self.rules.by_dim:
            self.rules.by_dim[dim] = RuleSet()
        return self.rules.by_dim[dim]

    def _load_dim(self):
        rs = self._cur_rules()
        self.enabled_var.set(rs.enabled)
        def set_text(txt, lines):
            txt.configure(state="normal"); txt.delete("1.0","end")
            for v in lines: txt.insert("end", f"{v}\n")
            txt.configure(state="normal")
        self.block_whitelist_txt.after(0, lambda: None)  # keep tk happy
        set_text(self.block_whitelist_txt, rs.block_whitelist)
        set_text(self.block_blacklist_txt, rs.block_blacklist)
        set_text(self.entity_whitelist_txt, rs.entity_whitelist)
        set_text(self.entity_blacklist_txt, rs.entity_blacklist)
        set_text(self.biome_whitelist_txt, [str(x) for x in rs.biome_whitelist])
        set_text(self.biome_blacklist_txt, [str(x) for x in rs.biome_blacklist])

    def _import_nei(self):
        from tkinter import filedialog, messagebox
        p = filedialog.askopenfilename(title="Pick NEI 'Dump ID Map' text file", filetypes=[("Text","*.txt"),("All","*.*")])
        if not p: return
        try:
            self.nei_map = _try_load_nei_dump(Path(p))
            messagebox.showinfo("NEI Import", f"Loaded {len(self.nei_map)} entries.")
            self._refresh_search()
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    def _refresh_search(self):
        q=(self.search_var.get() or "").strip().lower()
        self.listbox.delete(0,"end")
        if not self.nei_map: return
        items=[]
        for key,disp in self.nei_map.items():
            if q in key or q in disp.lower():
                items.append(disp)
        items.sort()
        for s in items[:800]: self.listbox.insert("end", s)

    def _save(self):
        from tkinter import messagebox
        rs = self._cur_rules()
        rs.enabled = bool(self.enabled_var.get())
        def get_lines(txt)->List[str]:
            v = txt.get("1.0","end").strip().splitlines()
            return [s.strip() for s in v if s.strip()]
        rs.block_whitelist = get_lines(self.block_whitelist_txt)
        rs.block_blacklist = get_lines(self.block_blacklist_txt)
        rs.entity_whitelist = [s.lower() for s in get_lines(self.entity_whitelist_txt)]
        rs.entity_blacklist = [s.lower() for s in get_lines(self.entity_blacklist_txt)]
        def ints(txt):
            out=[]
            for s in get_lines(txt):
                try: out.append(int(s))
                except ValueError: pass
            return out
        rs.biome_whitelist = ints(self.biome_whitelist_txt)
        rs.biome_blacklist = ints(self.biome_blacklist_txt)
        try:
            _save_rules(self.world, self.rules)
            messagebox.showinfo("Saved", f"Wrote {_rules_path(self.world).name}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

# ---------------- Status GUI ----------------
class VoiderStatusUI:
    def __init__(self, args):
        import tkinter as tk
        from tkinter import ttk
        import tkinter.font as tkfont

        self.args = args
        self.tk = tk.Tk()
        self.tk.title("GTNH Voider — Status & Controls")
        self.tk.geometry("980x700")
        try:
            base=tkfont.nametofont("TkDefaultFont")
            family="Segoe UI Variable" if "Variable" in tkfont.families() else "Segoe UI"
            base.configure(family=family, size=11); self.tk.option_add("*Font", base)
        except Exception: pass

        menubar = tk.Menu(self.tk)
        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(label="Reset Biomes (All Dims incl. Overworld)", command=self._menu_reset_all_dims)
        tools.add_command(label="Reset Biomes (All Non-Overworld Dims)", command=self._menu_reset_non_overworld)
        menubar.add_cascade(label="Tools", menu=tools)
        menubar.add_command(label="Help", command=self._open_help)
        self.tk.config(menu=menubar)

        self._q = queue.Queue()
        self._orig_out, self._orig_err = sys.stdout, sys.stderr
        sys.stdout = _TeeIO(sys.stdout, self._q, "OUT")
        sys.stderr = _TeeIO(sys.stderr, self._q, "ERR")

        self.world_path: Optional[str] = args.world
        self.cancel_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.nei_autoload: Dict[str,str] = {}

        top = ttk.Frame(self.tk); top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="World:").pack(side="left")
        self.world_var = tk.StringVar(value=self.world_path or "")
        self.world_entry = ttk.Entry(top, textvariable=self.world_var, width=64)
        self.world_entry.pack(side="left", padx=(6,6))
        b_browse = ttk.Button(top, text="Browse…", command=self._browse_world)
        b_disc   = ttk.Button(top, text="Discover…", command=self._discover_worlds)
        b_select = ttk.Button(top, text="Chunk Selector…", command=self._open_selector)
        b_browse.pack(side="left", padx=(0,8)); b_disc.pack(side="left", padx=(0,8)); b_select.pack(side="left")
        Tooltip(self.world_entry, "Path to the world folder (contains level.dat).",
                "Pick the world directory, e.g. '...\\.minecraft\\saves\\YourWorld'. Use Discover to scan typical roots.")
        Tooltip(b_select, "Open the chunk selection UI.", 
                "Mark Include/Exclude chunk coords per dimension; results saved in .voider-selection.json.")

        opts = ttk.Frame(self.tk); opts.pack(fill="x", padx=10, pady=(0,8))
        self.preset = tk.StringVar(value=getattr(args, "preset", "manual"))
        ttk.Label(opts, text="Preset:").pack(side="left", padx=(0,4))
        self.preset_box = ttk.Combobox(opts, textvariable=self.preset, state="readonly", width=14,
                                       values=tuple(PRESETS))
        self.preset_box.pack(side="left", padx=(0,10))
        self.dry = tk.BooleanVar(value=bool(args.dry_run))
        self.include_void = tk.BooleanVar(value=bool(args.include_void))
        self.keep_entities = tk.BooleanVar(value=bool(args.keep_entities))
        self.keep_tes = tk.BooleanVar(value=bool(args.keep_tileentities))
        self.skip_entities = tk.BooleanVar(value=bool(args.skip_if_entities))
        self.skip_tes = tk.BooleanVar(value=bool(args.skip_if_tileentities))
        self.mark_void = tk.BooleanVar(value=bool(args.mark_void))
        self.restore_biomes = tk.BooleanVar(value=bool(args.restore_biomes))
        self.nether_placeholder = tk.BooleanVar(value=bool(args.nether_cosmetic_placeholder))
        self.post_verify = tk.BooleanVar(value=bool(getattr(args, "post_verify", True)))
        self.workers = tk.IntVar(value=int(args.workers or 1))
        def add_chk(text, var, short, ext):
            b = ttk.Checkbutton(opts, text=text, variable=var); b.pack(side="left", padx=(0,10)); Tooltip(b, short, ext)
            return b
        add_chk("Dry run", self.dry, "Simulate only; no files changed.",
                "Report how many chunks WOULD be voided. Use this first.")
        add_chk("Include already void", self.include_void, "Also rewrite chunks that are already empty.",
                "Useful to re-tag voided chunks or rebuild platform.")
        add_chk("Keep Entities", self.keep_entities, "Copy Level.Entities from source.",
                "Use with caution; can preserve spawns or players.")
        add_chk("Keep TileEntities", self.keep_tes, "Leave chunks with TileEntities untouched.",
                "TileEntities require their backing blocks. Enabling this skips those chunks rather than copying orphaned chests or machines into empty terrain.")
        add_chk("Skip if Entities", self.skip_entities, "Skip chunks that contain any Entities.",
                "Safety.")
        add_chk("Skip if TileEntities", self.skip_tes, "Skip chunks that contain any TileEntities.",
                "Safety.")
        add_chk("Mark Void tag", self.mark_void, "Add Level.Voider=1 marker to new chunks.",
                "Allows future tools to detect voids fast.")
        add_chk("Restore Biomes from .bak", self.restore_biomes, "Copy Biomes[256] from .mca.bak when missing.",
                "When rewriting chunks, if Biomes are absent and a .bak exists, copy from the .bak.")
        add_chk("Nether placeholder", self.nether_placeholder, "Cosmetic Hell biomes in DIM-1 when missing.",
                "When enabled, missing Nether biomes are temporarily written as Hell (id=8) to avoid 'Ocean' HUD; repop flags are still cleared so providers may overwrite.")
        add_chk("Verify after write", self.post_verify, "Audit the world after modifications.",
                "Checks sector ranges, overlap, NBT readability, and stored chunk coordinates after every write run.")
        wlbl = ttk.Label(opts, text="Workers:"); wlbl.pack(side="left", padx=(10,4))
        self.worker_spin = ttk.Spinbox(opts, from_=1, to=32, width=4, textvariable=self.workers)
        self.worker_spin.pack(side="left")
        Tooltip(self.worker_spin, "How many region files to process in parallel.", "4–8 on SSDs, lower on HDDs. 1 = sequential.")

        statf = ttk.Frame(self.tk); statf.pack(fill="x", padx=10, pady=(0,6))
        ttk.Label(statf, text="Status:").pack(side="left")
        self.status_var = tk.StringVar(value="Ready.")
        self.status_lbl = ttk.Label(statf, textvariable=self.status_var); self.status_lbl.pack(side="left", padx=(6,0))
        self.progress = ttk.Progressbar(self.tk, mode="determinate"); self.progress.pack(fill="x", padx=10)

        btns = ttk.Frame(self.tk); btns.pack(fill="x", padx=10, pady=6)
        self.btn_start = ttk.Button(btns, text="Start Voiding", command=self._start_voiding); self.btn_start.pack(side="left")
        self.btn_cancel = ttk.Button(btns, text="Cancel", command=self._cancel, state="disabled"); self.btn_cancel.pack(side="left", padx=(8,0))
        ttk.Button(btns, text="Help", command=self._open_help).pack(side="right", padx=(0,8))
        ttk.Button(btns, text="Rules Editor…", command=self._open_rules).pack(side="right", padx=(0,8))
        ttk.Button(btns, text="Save Log…", command=self._save_log).pack(side="right")
        ttk.Button(btns, text="Copy Log", command=self._copy_log).pack(side="right", padx=(0,8))

        dockh = ttk.Frame(self.tk); dockh.pack(fill="x", padx=10, pady=(4,0))
        ttk.Label(dockh, text="Console (toggle ⇵)").pack(side="left")
        self.console_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(dockh, variable=self.console_visible, command=self._toggle_console).pack(side="left")

        self.console = tk.Text(self.tk, height=16, wrap="none"); self.console.configure(state="disabled")
        cinf = ttk.Frame(self.tk); cinf.pack(fill="x", padx=10, pady=(0,8))
        ttk.Label(cinf, text="> ").pack(side="left")
        self.console_in = ttk.Entry(cinf); self.console_in.pack(side="left", fill="x", expand=True)
        self.console_in.bind("<Return>", self._on_command_enter)
        cinf.pack_forget()
        self._console_frame = (self.console, cinf)

        logf = ttk.LabelFrame(self.tk, text="Log"); logf.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.logbox = tk.Text(logf, wrap="none"); self.logbox.pack(fill="both", expand=True)
        self.logbox.configure(state="disabled")

        if self.world_var.get().strip():
            self._prime_nei(Path(self.world_var.get().strip()))

        self._pump()

    # ---- Menu handlers ----
    def _run_biome_reset(self, world: Path, dims: Optional[List[str]], description: str):
        from tkinter import messagebox
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Operation already running", "Wait for the current world operation to finish.")
            return
        if dims is not None and not dims:
            messagebox.showinfo("No dimensions", "There are no non-Overworld dimensions to reset.")
            return
        scope = "all detected dimensions" if dims is None else ", ".join(dims)
        approved = messagebox.askyesno(
            "Confirm biome reset",
            f"{description}\n\nWorld: {world}\nScope: {scope}\n\n"
            "This changes chunk biome/population metadata. A full verified ZIP backup "
            "will be created first. The Minecraft client/server must be stopped. Continue?",
            parent=self.tk,
        )
        if not approved:
            return

        self.btn_start.configure(state="disabled")
        self.btn_cancel.configure(state="disabled")
        self._set_status("Preparing biome reset and verified backup…")

        def run():
            try:
                def progress(done, total, line):
                    pct = int((done / total) * 100) if total else 0
                    print(f"{pct}% — {line}")
                    self._set_progress(pct)
                    self._set_status(f"{description}: {pct}% — {line}")

                count = reset_biomes(world, dims, progress_cb=progress)
                print(f"Biome reset complete: changed {count} chunk(s).")
                self._set_status(f"Biome reset complete: {count} chunk(s) changed.")
            except Exception as exc:
                print(f"[error] Biome reset failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                self._set_status(f"Biome reset failed: {exc}")
            finally:
                def restore_controls():
                    self.btn_start.configure(state="normal")
                    self.btn_cancel.configure(state="disabled")
                self._ui_call(restore_controls)

        self.worker_thread = threading.Thread(target=run, daemon=True)
        self.worker_thread.start()

    def _menu_reset_all_dims(self):
        from tkinter import messagebox
        selected = self.world_var.get().strip()
        try:
            world = _resolve_world_path(Path(selected)) if selected else None
        except Exception as exc:
            messagebox.showerror("Invalid world", str(exc), parent=self.tk)
            return
        if world is None or not (world / "level.dat").is_file():
            messagebox.showwarning("Pick a world", "Select a valid world or dedicated-server instance first.", parent=self.tk)
            return
        self.world_var.set(str(world))
        self._run_biome_reset(world, None, "Reset biome metadata in all dimensions")

    def _menu_reset_non_overworld(self):
        from tkinter import messagebox
        selected = self.world_var.get().strip()
        try:
            world = _resolve_world_path(Path(selected)) if selected else None
        except Exception as exc:
            messagebox.showerror("Invalid world", str(exc), parent=self.tk)
            return
        if world is None or not (world / "level.dat").is_file():
            messagebox.showwarning("Pick a world", "Select a valid world or dedicated-server instance first.", parent=self.tk)
            return
        self.world_var.set(str(world))
        dims = [
            _dim_tag(world, directory)
            for directory in _find_region_dirs(world)
            if _dim_tag(world, directory) != "region"
        ]
        self._run_biome_reset(world, dims, "Reset biome metadata outside the Overworld")

    # ---- Buttons / actions ----
    def _open_help(self): HelpDialog(self.tk)

    def _open_rules(self):
        from tkinter import messagebox
        selected = self.world_var.get().strip()
        try:
            world = _resolve_world_path(Path(selected)) if selected else None
        except Exception as exc:
            messagebox.showerror("Invalid world", str(exc))
            return
        if world is None or not (world/"level.dat").is_file():
            messagebox.showwarning("Pick a world", "Select a valid world or dedicated-server instance first.")
            return
        self.world_var.set(str(world))
        dims = [_dim_tag(world, d) for d in _find_region_dirs(world)]
        rules = _load_rules(world)
        RulesEditorDialog(self.tk, world, rules, dims, self.nei_autoload)

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
        self.console.configure(state="normal"); self.console.insert("end", s); self.console.see("end"); self.console.configure(state="disabled")

    def _on_command_enter(self, _ev=None):
        cmd=self.console_in.get().strip(); self.console_in.delete(0,"end")
        if not cmd: return
        self._append_console(f"> {cmd}\n")
        if cmd.lower() in ("help","?"):
            self._append_console("commands: help, rescan, select, start, cancel, dryon, dryoff, workers N\n")
        elif cmd.lower()=="rescan": self._discover_worlds()
        elif cmd.lower()=="select": self._open_selector()
        elif cmd.lower()=="start":  self._start_voiding()
        elif cmd.lower()=="cancel": self._cancel()
        elif cmd.lower()=="dryon":  self.dry.set(True)
        elif cmd.lower()=="dryoff": self.dry.set(False)
        elif cmd.lower().startswith("workers "):
            try: n=int(cmd.split()[1]); self.workers.set(max(1,min(32,n)))
            except Exception: self._append_console("bad workers value\n")
        else:
            self._append_console("unknown command\n")

    def _browse_world(self):
        from tkinter import filedialog, messagebox
        p=filedialog.askdirectory(title="Pick a world folder or dedicated-server instance")
        if p:
            try:
                world = _resolve_world_path(Path(p))
                if not (world / "level.dat").is_file():
                    raise ValueError("Selected folder has neither level.dat nor a valid server.properties level-name world.")
                self.world_var.set(str(world))
                self._prime_nei(world)
            except Exception as exc:
                messagebox.showerror("Invalid world", str(exc))

    def _discover_worlds(self):
        def bg():
            self._set_status("Discovering worlds…")
            found=discover_minecraft_worlds(force_rescan=True, status_cb=self._set_status)
            def finish():
                pick=choose_world_gui(found)
                if pick:
                    self.world_var.set(pick)
                    self._prime_nei(Path(pick))
                self._set_status("Discovery done.")
            self.tk.after(0, finish)
        threading.Thread(target=bg, daemon=True).start()

    def _prime_nei(self, world_path:Path):
        try:
            nei_path=_auto_find_nei_dump(world_path)
            if nei_path:
                self.nei_autoload=_try_load_nei_dump(nei_path)
                print(f"NEI dump autoloaded: {nei_path.name} ({len(self.nei_autoload)} entries)")
            else:
                print("NEI dump not found for this instance (optional).")
        except Exception as e:
            print(f"[warn] NEI autoload failed: {e!r}")

    def _open_selector(self):
        try:
            launch_chunk_selector(
                self.world_var.get(),
                initial_dim=self.args.initial_dim,
                preload=None,
                limit_tags=[s.strip() for s in self.args.limit_tags.split(",")] if self.args.limit_tags else None
            )
            self._set_status("Chunk selection saved. Ready to void.")
        except SystemExit:
            self._set_status("Chunk selector aborted.")
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Chunk selector failed", str(e))

    def _start_voiding(self):
        if self.worker_thread and self.worker_thread.is_alive(): return
        wp=self.world_var.get().strip()
        try:
            wp = str(_resolve_world_path(Path(wp))) if wp else ""
        except Exception as exc:
            self._set_status(f"Invalid world: {exc}")
            return
        if not (Path(wp) / "level.dat").is_file():
            self._set_status("Pick a valid world folder or server instance first.")
            return
        self.world_var.set(wp)
        self.args.world = wp
        self.args.dry_run = self.dry.get()
        self.args.include_void = self.include_void.get()
        self.args.keep_entities = self.keep_entities.get()
        self.args.keep_tileentities = self.keep_tes.get()
        self.args.skip_if_entities = self.skip_entities.get()
        self.args.skip_if_tileentities = self.skip_tes.get()
        self.args.mark_void = self.mark_void.get()
        self.args.workers = int(self.workers.get())
        self.args.restore_biomes = self.restore_biomes.get()
        self.args.nether_cosmetic_placeholder = self.nether_placeholder.get()
        self.args.post_verify = self.post_verify.get()
        self.args.preset = self.preset.get()
        _apply_preset(self.args)

        self.cancel_event.clear()
        self.btn_start.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.progress.configure(value=0, maximum=100)
        self._set_status("Preparing…")

        def run():
            try:
                world = Path(self.args.world)
                rules_local = _load_rules(world)

                sel_path = world/".voider-selection.json"
                if not sel_path.is_file():
                    raise RuntimeError(
                        "No saved chunk selection exists. Open Chunk Selector and save a selection; "
                        "whole-world processing is available only through the explicit CLI --all-chunks option."
                    )
                try:
                    sel_include, sel_exclude = _load_selection(sel_path)
                except Exception as exc:
                    raise RuntimeError(f"Refusing to run with an invalid selection file: {exc}") from exc
                self.args.selection_scope = set(sel_include) | set(sel_exclude)
                if not self.args.selection_scope:
                    raise RuntimeError(
                        "The saved selection contains no dimension scopes. Save a selection or explicitly "
                        "authorize all chunks through the headless CLI."
                    )
                print(
                    f"Loaded selection from {sel_path.name}; authorized dimensions: "
                    + ", ".join(sorted(self.args.selection_scope))
                )

                prog_lock=threading.Lock()
                def prog_cb(done, total, line):
                    with prog_lock:
                        pct=int((done/total)*100) if total else 0
                    self._set_progress(pct); self._set_status(f"{pct}% — {line}")

                if not self.args.dry_run:
                    preview_args = copy.copy(self.args)
                    preview_args.dry_run = True
                    preview_args.backup = False
                    preview_scanned, preview_changed = run_voider(
                        world, sel_include, sel_exclude, preview_args,
                        progress_cb=prog_cb, status_cb=self._set_status, rules=rules_local,
                        cancel_event=self.cancel_event,
                    )
                    if self.cancel_event.is_set():
                        print("Cancelled during preflight preview.")
                        return
                    approved = self._ask_yes_no(
                        "Confirm world modification",
                        f"Preflight scanned {preview_scanned} chunks and would modify "
                        f"{preview_changed}.\n\nA verified backup will be made first. Continue?",
                    )
                    if not approved:
                        print("Write cancelled after preflight preview.")
                        self._set_status("Write cancelled.")
                        return

                total_scanned, total_changed = run_voider(
                    world, sel_include, sel_exclude, self.args,
                    progress_cb=prog_cb, status_cb=self._set_status, rules=rules_local,
                    cancel_event=self.cancel_event,
                )
                if not self.args.dry_run and total_changed and self.args.post_verify:
                    self._set_status("Verifying written world…")
                    audit = verify_world(world)
                    audit_path = world/".voider-last-verification.json"
                    audit_path.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
                    if not audit["ok"]:
                        raise RuntimeError(f"Post-write verification found {len(audit['errors'])} error(s); see {audit_path}")
                    print(f"Post-write verification passed; report: {audit_path}")
                msg = (f"Would void {total_changed} chunk(s) out of {total_scanned} scanned. Biomes provider-safe."
                       if self.args.dry_run else
                       f"Done. Voided {total_changed} chunk(s) out of {total_scanned} scanned. Biomes provider-safe.")
                print(msg); self._set_status("Idle.")
            except Exception as exc:
                print(f"[error] Voider operation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                self._set_status(f"Voider operation failed: {exc}")
            finally:
                self._set_running(False)

        self.worker_thread = threading.Thread(target=run, daemon=True); self.worker_thread.start()

    def _cancel(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.cancel_event.set(); self._set_status("Cancelling…")

    def _ui_call(self, callback)->None:
        if threading.current_thread() is threading.main_thread():
            callback()
        else:
            self.tk.after(0, callback)

    def _ask_yes_no(self, title:str, message:str)->bool:
        """Synchronously ask from a worker while creating widgets on Tk's thread."""
        from tkinter import messagebox
        done = threading.Event()
        answer = {"value": False}
        def ask():
            try:
                answer["value"] = bool(messagebox.askyesno(title, message, parent=self.tk))
            finally:
                done.set()
        self.tk.after(0, ask)
        done.wait()
        return answer["value"]

    def _set_status(self, s:str)->None:
        self._ui_call(lambda: self.status_var.set(s))

    def _set_progress(self, value:int)->None:
        self._ui_call(lambda: self.progress.configure(value=value))

    def _set_running(self, running:bool)->None:
        def apply():
            self.btn_start.configure(state="disabled" if running else "normal")
            self.btn_cancel.configure(state="normal" if running else "disabled")
        self._ui_call(apply)

    def _save_log(self):
        from tkinter import filedialog, messagebox
        p=filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text","*.txt"),("All","*.*")])
        if not p: return
        try:
            data=self.logbox.get("1.0","end-1c"); Path(p).write_text(data, encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _copy_log(self):
        data=self.logbox.get("1.0","end-1c")
        try:
            self.tk.clipboard_clear(); self.tk.clipboard_append(data); self.tk.update()
            print("Log copied to clipboard.")
        except Exception as e:
            print(f"[warn] Clipboard copy failed: {e!r}")

    def _pump(self):
        try:
            while True:
                tag,line=self._q.get_nowait()
                self.logbox.configure(state="normal"); self.logbox.insert("end", line); self.logbox.see("end"); self.logbox.configure(state="disabled")
                if self.console_visible.get(): self._append_console(line)
        except queue.Empty:
            pass
        self.tk.after(60, self._pump)

    def run_forever(self)->None: self.tk.mainloop()

# ---------------- World chooser ----------------
def choose_world_gui(discovered:Dict[str,str]) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import ttk
        import tkinter.font as tkfont
    except Exception:
        return None
    root = tk.Tk(); root.title("Select a Minecraft World"); root.geometry("680x460")
    try:
        base=tkfont.nametofont("TkDefaultFont")
        family="Segoe UI Variable" if "Variable" in tkfont.families() else "Segoe UI"
        base.configure(family=family, size=11); root.option_add("*Font", base)
    except Exception: pass
    sel_path={"v":None}
    frm=ttk.Frame(root); frm.pack(fill="both", expand=True, padx=10, pady=10)
    ttk.Label(frm, text="Detected worlds (double-click to choose):").pack(anchor="w")
    cols=("name","path"); tree=ttk.Treeview(frm, columns=cols, show="headings", height=14)
    tree.heading("name", text="World"); tree.heading("path", text="Path")
    tree.column("name", width=240); tree.column("path", width=420)
    tree.pack(fill="both", expand=True, pady=(6,6))
    rows=[]
    for world_path in sorted(discovered.values(), key=str.lower):
        p=Path(world_path); name=p.name; rows.append((name, world_path))
    for row in rows: tree.insert("", "end", values=row)
    def on_ok():
        item = tree.selection()
        if item:
            vals = tree.item(item[0], "values"); sel_path["v"]=vals[1]; root.destroy()
        else:
            from tkinter import messagebox; messagebox.showwarning("Pick a world","Select a world first (or use Browse…).")
    def on_browse():
        from tkinter import filedialog, messagebox
        d = filedialog.askdirectory(title="Pick a world folder (must contain level.dat)")
        if d and (Path(d)/"level.dat").exists():
            sel_path["v"]=d; root.destroy()
        elif d:
            messagebox.showerror("Invalid world","Selected folder does not contain level.dat")
    tree.bind("<Double-1>", lambda _e: on_ok())
    btns=ttk.Frame(frm); btns.pack(fill="x", pady=(4,0))
    ttk.Button(btns, text="Browse…", command=on_browse).pack(side="left")
    ttk.Button(btns, text="OK", command=on_ok).pack(side="right")
    ttk.Button(btns, text="Cancel", command=root.destroy).pack(side="right", padx=(0,6))
    root.mainloop()
    return sel_path["v"]

# ---------------- Orchestration ----------------
def _run_voider_unlocked(world:Path,
               sel_include_by_dim:Dict[str, Set[Tuple[int,int]]],
               sel_exclude_by_dim:Dict[str, Set[Tuple[int,int]]],
               args,
               *,
               progress_cb:Optional[Callable[[int,int,str],None]]=None,
               status_cb:Optional[Callable[[str],None]]=None,
               rules:Optional[AllRules]=None,
               cancel_event:Optional[threading.Event]=None)->Tuple[int,int]:
    """
    Returns: (total_scanned, total_changed)
    """
    cancel_event = cancel_event or threading.Event()
    if cancel_event.is_set():
        raise InterruptedError("Voider operation was already cancelled.")
    platform=None
    if args.platform_x is not None and args.platform_z is not None:
        size=int(args.platform_size)
        if size<=0 or size%2==0: raise SystemExit("Platform size must be a positive odd number.")
        if not 0 <= int(args.platform_y) <= 255: raise SystemExit("Platform Y must be in 0..255.")
        if not 0 <= int(args.platform_id) <= 4095: raise SystemExit("Platform block ID must be in 0..4095.")
        if not 0 <= int(args.platform_meta) <= 15: raise SystemExit("Platform metadata must be in 0..15.")
        platform=(args.platform_x, args.platform_z, args.platform_y, size, args.platform_id, args.platform_meta)

    only=[s.strip() for s in args.only.split(",")] if args.only else None
    exclude=[s.strip() for s in args.exclude.split(",")] if args.exclude else None

    region_dirs = _find_region_dirs(world)
    selection_scope = getattr(args, "selection_scope", None)
    if selection_scope is not None:
        selection_scope = set(selection_scope)
        available_tags = {_dim_tag(world, directory) for directory in region_dirs}
        unknown_tags = selection_scope - available_tags
        if unknown_tags:
            raise ValueError(
                "Selection refers to missing dimension(s): " + ", ".join(sorted(unknown_tags))
            )
        if not selection_scope:
            raise ValueError(
                "Selection authorizes no dimensions. Use an explicit selection or --all-chunks."
            )
        region_dirs = [
            directory for directory in region_dirs
            if _dim_tag(world, directory) in selection_scope
        ]
    if only:
        region_dirs = [d for d in region_dirs if _dim_tag(world, d) in only]
    if exclude:
        region_dirs = [d for d in region_dirs if _dim_tag(world, d) not in exclude]

    bbox=None
    if args.bbox:
        a=[int(x.strip()) for x in args.bbox.split(",")]
        if len(a)!=4: raise SystemExit("bbox must be: minX,minZ,maxX,maxZ")
        x1,z1,x2,z2=a; bbox=(min(x1,x2), min(z1,z2), max(x1,x2), max(z1,z2))
    center=(args.center_x, args.center_z) if (args.center_x is not None and args.center_z is not None) else None

    total_scanned=0; total_changed=0

    backup_path = None
    if not args.dry_run and args.backup:
        if status_cb: status_cb("Creating world ZIP backup…")
        before_backup = _region_snapshot(world)
        backup_path = _backup_world_zip(world)
        if _region_snapshot(world) != before_backup:
            raise RuntimeError(f"World region files changed during backup; verified archive retained at {backup_path}")
    if args.per_file_backup and not args.dry_run:
        if status_cb: status_cb("Creating per-file .bak copies…")
        _per_file_backup(world)

    if args.wait_for_unlock and args.wait_for_unlock>0:
        print("[warning] --wait-for-unlock is deprecated: session.lock normally persists after shutdown. "
              "Concurrent region changes are detected directly instead.")

    file_list: List[Tuple[str, Path]] = []
    for rdir in region_dirs:
        dimtag=_dim_tag(world, rdir)
        files=sorted(p for p in rdir.iterdir() if p.suffix==".mca" and p.name.startswith("r."))
        for mca in files: file_list.append((dimtag, mca))
    total_files=len(file_list)
    if total_files==0:
        if status_cb: status_cb("No region files found after filters.")
        return (0,0)
    if status_cb: status_cb(f"Processing {len(total_files)} region file(s) across {len(set(d for d,_ in file_list))} dimension(s)…")

    def _work(dimtag:str, mca:Path):
        inc_set=sel_include_by_dim.get(dimtag,set()); exc_set=sel_exclude_by_dim.get(dimtag,set())
        nm, scanned, changed = _process_region(
            mca, dimtag=dimtag, bbox=bbox, center=center, radius_blocks=int(args.radius_blocks),
            include_void=bool(args.include_void), platform=platform,
            keep_entities=bool(args.keep_entities), keep_tileentities=bool(args.keep_tileentities),
            skip_if_entities=bool(args.skip_if_entities), skip_if_tileentities=bool(args.skip_if_tileentities),
            mark_void=bool(args.mark_void), dry_run=bool(args.dry_run), strict=True,
            sel_include=inc_set, sel_exclude=exc_set, restore_biomes_from_bak=bool(args.restore_biomes),
            nether_cosmetic_placeholder=bool(args.nether_cosmetic_placeholder),
            rules=rules,
            cancel_event=cancel_event
        )
        line=f"{'would void' if args.dry_run else 'voided'}: {nm} ({changed} of {scanned} chunks)"
        print(f"  {line}")
        return scanned, changed, dimtag, line

    done_files = 0
    failures = []
    if args.workers and int(args.workers) > 1:
        max_workers = max(1, int(args.workers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_paths = {
                executor.submit(_work, dimtag, mca): mca
                for dimtag, mca in file_list
            }
            for future in as_completed(future_paths):
                mca = future_paths[future]
                try:
                    scanned, changed, _dimtag, line = future.result()
                    total_scanned += scanned
                    total_changed += changed
                    done_files += 1
                    if progress_cb:
                        progress_cb(done_files, total_files, line)
                except InterruptedError:
                    continue
                except Exception as exc:
                    failures.append(f"{mca}: {type(exc).__name__}: {exc}")
                    if cancel_event is not None:
                        cancel_event.set()
    else:
        grouped: Dict[str, List[Path]] = {}
        for dimtag, mca in file_list:
            grouped.setdefault(dimtag, []).append(mca)
        for dimtag, files in grouped.items():
            print(f"Processing {len(files)} region files in '{dimtag}'...")
            for mca in files:
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    scanned, changed, _dimtag, line = _work(dimtag, mca)
                    total_scanned += scanned
                    total_changed += changed
                    done_files += 1
                    if progress_cb:
                        progress_cb(done_files, total_files, line)
                except InterruptedError:
                    break
                except Exception as exc:
                    failures.append(f"{mca}: {type(exc).__name__}: {exc}")
                    if cancel_event is not None:
                        cancel_event.set()
                    break

    if failures:
        detail = "; ".join(failures[:5])
        extra = f" Verified recovery backup: {backup_path}" if backup_path else " No backup was created."
        raise RuntimeError(
            f"Voider did not complete: {len(failures)} region failure(s). {detail}.{extra}"
        )
    if cancel_event is not None and cancel_event.is_set():
        extra = f" Verified recovery backup: {backup_path}" if backup_path else " No backup was created."
        raise RuntimeError(
            f"Voider was cancelled. Previously committed region files may have changed;{extra}"
        )
    return total_scanned, total_changed

def run_voider(world:Path,
               sel_include_by_dim:Dict[str, Set[Tuple[int,int]]],
               sel_exclude_by_dim:Dict[str, Set[Tuple[int,int]]],
               args,
               *,
               progress_cb:Optional[Callable[[int,int,str],None]]=None,
               status_cb:Optional[Callable[[str],None]]=None,
               rules:Optional[AllRules]=None,
               cancel_event:Optional[threading.Event]=None)->Tuple[int,int]:
    """Run a read-only preview or a write protected by a held session.lock."""
    world = _resolve_world_path(Path(world))
    if args.dry_run:
        return _run_voider_unlocked(
            world, sel_include_by_dim, sel_exclude_by_dim, args,
            progress_cb=progress_cb, status_cb=status_cb, rules=rules,
            cancel_event=cancel_event,
        )
    with _world_write_guard(world):
        return _run_voider_unlocked(
            world, sel_include_by_dim, sel_exclude_by_dim, args,
            progress_cb=progress_cb, status_cb=status_cb, rules=rules,
            cancel_event=cancel_event,
        )

# ---------------- CLI / GUI ----------------
def main()->bool:
    ap=argparse.ArgumentParser(description="GTNH 1.7.x Voider (provider-safe biomes) — GUI + CLI")
    ap.add_argument("world", nargs="?", help="Path to your world folder (contains level.dat)")
    ap.add_argument("--initial-dim", help="Dimension tag to show first (e.g. DIM-1/region)")
    ap.add_argument("--limit-tags", help="Comma-separated dims to allow (e.g. region,DIM-1/region)")
    ap.add_argument("--only", help="Only these dims (comma) for processing")
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
    backup_group=ap.add_mutually_exclusive_group()
    backup_group.add_argument("--backup", dest="backup", action="store_true", default=True,
                              help="Create a verified world ZIP before writing (default)")
    backup_group.add_argument("--no-backup", dest="backup", action="store_false",
                              help="Skip the default world ZIP backup (unsafe)")
    ap.add_argument("--per-file-backup", action="store_true", help="Backup each region file before modify")
    mode_group=ap.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                            help="Preview only; do not modify files (default)")
    mode_group.add_argument("--write", dest="dry_run", action="store_false",
                            help="Enable world modifications")
    ap.add_argument("--include-void", action="store_true", help="Rewrite already void chunks too")
    ap.add_argument("--wait-for-unlock", type=int, default=0, help="Seconds to wait if session.lock present")
    ap.add_argument("--workers", type=int, default=1, help="Number of worker threads (>=1)")
    ap.add_argument("--skip-if-entities", action="store_true", help="Skip chunks with Entities")
    ap.add_argument("--skip-if-tileentities", action="store_true", help="Skip chunks with TileEntities")
    ap.add_argument("--keep-entities", action="store_true", help="Preserve Entities (copy through)")
    ap.add_argument("--keep-tileentities", action="store_true",
                    help="Preserve chunks with TileEntities intact by skipping them")
    ap.add_argument("--mark-void", action="store_true", help="Mark voided chunks with a Voider tag")
    ap.add_argument("--preset", choices=tuple(PRESETS), default="manual",
                    help="Named option bundle: safe-void, full-void, preserve-data, or manual")
    ap.add_argument("--force-rescan", action="store_true", help="Force re-scan of drives for worlds (ignore cache)")

    # Maintenance / behavior flags
    ap.add_argument("--reset-biomes-dims", help="Comma-separated dim tags to reset. Use 'all' for every dim (provider-safe).")
    ap.add_argument("--restore-biomes", action="store_true", help="Attempt to copy Biomes from .mca.bak when missing")

    # Cosmetic safety: Nether placeholder toggle
    ap.add_argument("--nether-cosmetic-placeholder", dest="nether_cosmetic_placeholder", action="store_true", default=True,
                    help="(Default ON) When missing in DIM-1, write Hell (id=8) biomes to avoid 'Ocean' HUD. Providers may overwrite.")
    ap.add_argument("--no-nether-cosmetic-placeholder", dest="nether_cosmetic_placeholder", action="store_false",
                    help="Disable Nether placeholder biomes (DIM-1).")

    ap.add_argument("--rules-editor", action="store_true", help="Open Rules Editor immediately (GUI)")
    ap.add_argument("--headless", action="store_true", help="Run without Tk or the chunk selector")
    ap.add_argument("--selection", type=str, help="Selection JSON to use in headless mode")
    ap.add_argument("--all-chunks", action="store_true",
                    help="Explicitly process all chunks after other filters (headless safety gate)")
    ap.add_argument("--verify-world", action="store_true",
                    help="Read-only audit of region sectors, NBT, and coordinates")
    ap.add_argument("--inspect-world", action="store_true",
                    help="Read-only client/server world inventory and lock status")
    ap.add_argument("--report", type=str, help="Write a JSON verification/run report")
    verify_group=ap.add_mutually_exclusive_group()
    verify_group.add_argument("--post-verify", dest="post_verify", action="store_true", default=True,
                              help="Verify region integrity after writes (default)")
    verify_group.add_argument("--no-post-verify", dest="post_verify", action="store_false",
                              help="Skip post-write verification")

    args=ap.parse_args()
    _apply_preset(args)
    if args.world:
        try:
            args.world = str(_resolve_world_path(Path(args.world)))
        except Exception as exc:
            ap.error(str(exc))

    if args.inspect_world:
        if not args.world: ap.error("--inspect-world requires a world path")
        world=_resolve_world_path(Path(args.world))
        if not (world/"level.dat").is_file(): ap.error(f"level.dat not found under {world}")
        rendered=json.dumps(inspect_world(world), indent=2)
        print(rendered)
        if args.report: Path(args.report).expanduser().write_text(rendered+"\n", encoding="utf-8")
        return True

    if args.verify_world:
        if not args.world: ap.error("--verify-world requires a world path")
        world=_resolve_world_path(Path(args.world))
        if not (world/"level.dat").is_file(): ap.error(f"level.dat not found under {world}")
        report=verify_world(world); rendered=json.dumps(report, indent=2)
        print(rendered)
        if args.report: Path(args.report).expanduser().write_text(rendered+"\n", encoding="utf-8")
        return bool(report["ok"])

    if args.headless:
        if not args.world:
            ap.error("--headless requires a world path")
        world=_resolve_world_path(Path(args.world))
        if not (world/"level.dat").is_file():
            ap.error(f"level.dat not found under {world}")
        selection_path = Path(args.selection).expanduser().resolve() if args.selection else world/".voider-selection.json"
        if args.all_chunks:
            # Explicit --all-chunks always overrides a persisted selection file.
            # This makes the operator's all-world authorization unambiguous.
            sel_include, sel_exclude = {}, {}
            args.selection_scope = {
                _dim_tag(world, directory) for directory in _find_region_dirs(world)
            }
            print(
                "--all-chunks explicitly authorizes every discovered dimension "
                "after --only/--exclude and coordinate filters."
            )
        elif selection_path.is_file():
            try:
                sel_include, sel_exclude = _load_selection(selection_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                ap.error(f"invalid selection file {selection_path}: {exc}")
            args.selection_scope = set(sel_include) | set(sel_exclude)
            if not args.selection_scope:
                ap.error(
                    f"selection file {selection_path} contains no dimension scopes; "
                    "use --all-chunks to explicitly authorize the entire world"
                )
            print(
                f"Loaded validated selection: {selection_path} "
                f"(dimensions: {', '.join(sorted(args.selection_scope))})"
            )
        else:
            ap.error(
                "headless mode requires a non-empty --selection, an existing non-empty "
                ".voider-selection.json, or explicit --all-chunks"
            )
        rules=_load_rules(world)
        scanned, changed=run_voider(world, sel_include, sel_exclude, args, rules=rules)
        verb="Would void" if args.dry_run else "Voided"
        print(f"{verb} {changed} chunk(s) out of {scanned} scanned.")
        audit = None
        if not args.dry_run and changed and args.post_verify:
            audit = verify_world(world)
            if not audit["ok"]:
                print(f"ERROR: post-write verification found {len(audit['errors'])} error(s).", file=sys.stderr)
        if args.report:
            result={"world": str(world), "mode": "dry-run" if args.dry_run else "write",
                    "scanned_chunks": scanned, "changed_chunks": changed,
                    "selection": str(selection_path) if selection_path.exists() else None,
                    "all_chunks": bool(args.all_chunks),
                    "selection_scope": sorted(getattr(args, "selection_scope", [])),
                    "backup_enabled": bool(args.backup),
                    "preset": args.preset, "post_write_verification": audit}
            Path(args.report).expanduser().write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        return audit is None or bool(audit["ok"])

    if args.reset_biomes_dims:
        world_path: Optional[str] = args.world
        if not world_path:
            print("Scanning for worlds… (use --force-rescan to refresh cache)")
            discovered=discover_minecraft_worlds(force_rescan=bool(args.force_rescan))
            world_path=choose_world_gui(discovered)
            if not world_path:
                return False

        world=_resolve_world_path(Path(world_path))
        if not (world/"level.dat").is_file():
            print("ERROR: selected folder is not a world or recognized server instance."); return False

        if args.reset_biomes_dims.strip().lower()=="all":
            dims=[_dim_tag(world, d) for d in _find_region_dirs(world)]
        else:
            dims=[s.strip() for s in args.reset_biomes_dims.split(",")]

        def prog_cb(done,total,line):
            pct=int((done/total)*100) if total else 0
            print(f"{pct}% — {line}")
        n=reset_biomes(world, dims, progress_cb=prog_cb)
        print(f"Reset biomes (provider-safe) in {n} chunk(s).")
        return True

    try:
        ui=VoiderStatusUI(args)
        if args.rules_editor and args.world and (Path(args.world)/"level.dat").exists():
            dims=[_dim_tag(Path(args.world), d) for d in _find_region_dirs(Path(args.world))]
            rules=_load_rules(Path(args.world))
            nei_path=_auto_find_nei_dump(Path(args.world))
            nei_map=_try_load_nei_dump(nei_path) if nei_path else {}
            RulesEditorDialog(ui.tk, Path(args.world), rules, dims, nei_map)
    except Exception:
        world_path: Optional[str] = args.world
        if not world_path:
            print("Scanning for worlds… (use --force-rescan to refresh cache)")
            discovered=discover_minecraft_worlds(force_rescan=bool(args.force_rescan))
            world_path=choose_world_gui(discovered)
            if not world_path:
                try: w=input("Enter path to world folder (must contain level.dat), or leave blank to exit: ").strip()
                except EOFError: w=""
                if not w: return False
                world_path=w

        world=Path(world_path)
        if not (world/"level.dat").exists():
            print("ERROR: level.dat not found in the selected folder."); return False

        try:
            sel=launch_chunk_selector(
                str(world), initial_dim=args.initial_dim, preload=None,
                limit_tags=[s.strip() for s in args.limit_tags.split(",")] if args.limit_tags else None
            )
        except SystemExit:
            return False
        except Exception as e:
            print(f"Chunk selector failed: {e}", file=sys.stderr); return False

        sel_include: Dict[str, Set[Tuple[int,int]]] = {}
        sel_exclude: Dict[str, Set[Tuple[int,int]]] = {}
        for dimtag,d in sel.items():
            sel_include[dimtag]=set(tuple(x) for x in d.get("include",[]))
            sel_exclude[dimtag]=set(tuple(x) for x in d.get("exclude",[]))
        args.selection_scope = set(sel_include) | set(sel_exclude)
        if not args.selection_scope:
            print("ERROR: Chunk Selector returned no authorized dimensions; refusing to process the world.", file=sys.stderr)
            return False

        rules=_load_rules(world)

        total_scanned, total_changed = run_voider(world, sel_include, sel_exclude, args, rules=rules)
        print(f"{'Would void' if args.dry_run else 'Done.'} Voided {total_changed} chunk(s) out of {total_scanned} scanned. Biomes provider-safe.")
        return True

    ui.run_forever(); return True

if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except Exception:
        import traceback
        tb=traceback.format_exc()
        # Headless/CLI failures must remain non-interactive. A modal Tk
        # message box can block subprocesses and CI runners indefinitely.
        if "--headless" not in sys.argv:
            try:
                import tkinter as tk
                from tkinter import messagebox
                tk.Tk().withdraw(); messagebox.showerror("Voider crashed", tb)
            except Exception:
                pass
        print(tb, file=sys.stderr)
        sys.exit(1)
