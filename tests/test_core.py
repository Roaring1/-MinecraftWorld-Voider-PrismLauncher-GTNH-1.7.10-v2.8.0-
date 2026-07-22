from __future__ import annotations

import io
import json
import os
import subprocess
import struct
import sys
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path

import main_voider as mv


def make_region(path: Path, raw_nbt: bytes, rx: int = 0, rz: int = 0, idx: int = 0) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    mca = path / f"r.{rx}.{rz}.mca"
    body = bytes([2]) + mv.zlib.compress(raw_nbt)
    payload = struct.pack(">I", len(body)) + body
    sectors = (len(payload) + mv.SECTOR_BYTES - 1) // mv.SECTOR_BYTES
    header = bytearray(8192)
    struct.pack_into(">I", header, idx * 4, (2 << 8) | sectors)
    struct.pack_into(">I", header, 4096 + idx * 4, 1)
    with mca.open("wb") as fp:
        fp.write(header)
        fp.write(payload)
        fp.write(b"\0" * ((-len(payload)) % mv.SECTOR_BYTES))
    return mca


def read_level(raw: bytes):
    return mv.nbt.NBTFile(buffer=io.BytesIO(raw))["Level"]


class NibbleTests(unittest.TestCase):
    def test_nibble_round_trip(self):
        values = [0] * 8
        mv._nibble_set(values, 2, 0xA)
        mv._nibble_set(values, 3, 0x5)
        self.assertEqual(values[1], 0x5A)
        self.assertEqual(mv._nibble_get(bytes(values), 2), 0xA)
        self.assertEqual(mv._nibble_get(bytes(values), 3), 0x5)


class PlatformTests(unittest.TestCase):
    def test_platform_uses_anvil_index_metadata_and_add(self):
        raw = mv._build_void_chunk(
            0, 0, bytes([1] * 256), True,
            (2, 3, 65, 1, 300, 7), False, False, None, 1,
        )
        level = read_level(raw)
        section = next(s for s in level["Sections"] if int(s["Y"].value) == 4)
        index = 1 * 256 + 3 * 16 + 2
        self.assertEqual(section["Blocks"].value[index] & 0xFF, 300 & 0xFF)
        self.assertEqual(mv._nibble_get(bytes(section["Data"].value), index), 7)
        self.assertEqual(mv._nibble_get(bytes(section["Add"].value), index), 1)

    def test_platform_crosses_chunk_boundaries(self):
        platform = (15, 15, 64, 3, 7, 0)
        expected = {(0, 0), (1, 0), (0, 1), (1, 1)}
        got = {(x, z) for x in range(2) for z in range(2)
               if mv._platform_intersects_chunk(platform, x, z)}
        self.assertEqual(got, expected)

    def test_generated_void_chunk_is_parseable_and_detected(self):
        raw = mv._build_void_chunk(0, 0, bytes([8] * 256), True, None,
                                   False, False, None, 0)
        self.assertTrue(mv._is_void_chunk(raw))
        self.assertEqual(mv._get_biomes(raw), bytes([8] * 256))


class RegionIOTests(unittest.TestCase):
    def test_unaligned_append_offset_is_computed_after_padding(self):
        with tempfile.TemporaryFile() as fp:
            fp.write(b"\0" * (8192 + 123))
            raw = os.urandom(7000)
            new_off, new_count = mv._write_chunk_payload(fp, 2, 1, raw)
            self.assertEqual(new_off, 3)
            self.assertGreater(new_count, 1)

    def test_atomic_processing_replaces_original_and_leaves_no_temp(self):
        with tempfile.TemporaryDirectory() as td:
            region_dir = Path(td) / "region"
            raw = mv._build_void_chunk(0, 0, bytes([1] * 256), True, None,
                                       False, False, None, 1)
            mca = make_region(region_dir, raw)
            old_inode = mca.stat().st_ino
            name, scanned, changed = mv._process_region(
                mca, dimtag="region", bbox=None, center=None, radius_blocks=0,
                include_void=True, platform=None, keep_entities=False,
                keep_tileentities=False, skip_if_entities=False,
                skip_if_tileentities=False, mark_void=True, dry_run=False,
                strict=True, sel_include=set(), sel_exclude=set(),
                restore_biomes_from_bak=False,
                nether_cosmetic_placeholder=False,
            )
            self.assertEqual((name, scanned, changed), (mca.name, 1, 1))
            self.assertNotEqual(mca.stat().st_ino, old_inode)
            self.assertEqual(list(region_dir.glob("*.tmp")), [])
            with mca.open("rb") as fp:
                _, off, count = next(mv._iter_chunk_locations(fp))
                self.assertIsNotNone(mv._read_chunk_payload(fp, off, count))

    def test_explicit_rules_prevent_rewrite(self):
        with tempfile.TemporaryDirectory() as td:
            region_dir = Path(td) / "region"
            raw = mv._build_void_chunk(0, 0, bytes([1] * 256), True, None,
                                       False, False, None, 1)
            mca = make_region(region_dir, raw)
            rules = mv.AllRules(by_dim={
                "region": mv.RuleSet(biome_whitelist=[99], enabled=True)
            })
            _, scanned, changed = mv._process_region(
                mca, dimtag="region", bbox=None, center=None, radius_blocks=0,
                include_void=True, platform=None, keep_entities=False,
                keep_tileentities=False, skip_if_entities=False,
                skip_if_tileentities=False, mark_void=False, dry_run=False,
                strict=True, sel_include=set(), sel_exclude=set(),
                restore_biomes_from_bak=False,
                nether_cosmetic_placeholder=False, rules=rules,
            )
            self.assertEqual(scanned, 1)
            self.assertEqual(changed, 0)


class BackupTests(unittest.TestCase):
    def test_world_backups_are_unique_and_valid(self):
        with tempfile.TemporaryDirectory() as td:
            world = Path(td) / "World"
            world.mkdir()
            (world / "level.dat").write_bytes(b"level")
            first = mv._backup_world_zip(world)
            second = mv._backup_world_zip(world)
            self.assertNotEqual(first, second)
            for archive in (first, second):
                with zipfile.ZipFile(archive) as zf:
                    self.assertIsNone(zf.testzip())
                    self.assertIn("level.dat", zf.namelist())


class PresetAndVerificationTests(unittest.TestCase):
    def test_safe_void_preset(self):
        args = Namespace(
            preset="safe-void", skip_if_entities=False,
            skip_if_tileentities=False, keep_entities=True,
            keep_tileentities=True, mark_void=False,
        )
        mv._apply_preset(args)
        self.assertTrue(args.skip_if_entities)
        self.assertTrue(args.skip_if_tileentities)
        self.assertFalse(args.keep_entities)
        self.assertFalse(args.keep_tileentities)
        self.assertTrue(args.mark_void)

    def test_world_verifier_accepts_valid_region(self):
        with tempfile.TemporaryDirectory() as td:
            world = Path(td) / "World"
            world.mkdir()
            (world / "level.dat").write_bytes(b"test")
            raw = mv._build_void_chunk(0, 0, bytes([1] * 256), True, None,
                                       False, False, None, 1)
            make_region(world / "region", raw)
            report = mv.verify_world(world)
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["valid_chunks"], 1)

    def test_world_verifier_rejects_wrong_coordinates(self):
        with tempfile.TemporaryDirectory() as td:
            world = Path(td) / "World"
            world.mkdir()
            (world / "level.dat").write_bytes(b"test")
            raw = mv._build_void_chunk(9, 9, bytes([1] * 256), True, None,
                                       False, False, None, 1)
            make_region(world / "region", raw)
            report = mv.verify_world(world)
            self.assertFalse(report["ok"])
            self.assertTrue(any("coordinates" in error for error in report["errors"]))


class SelectionAndHeadlessTests(unittest.TestCase):
    def test_selection_validation_and_exclude_wins(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "selection.json"
            path.write_text(json.dumps({
                "region": {"include": [[1, 2], [3, 4]], "exclude": [[1, 2]]}
            }), encoding="utf-8")
            include, exclude = mv._load_selection(path)
            self.assertEqual(include["region"], {(3, 4)})
            self.assertEqual(exclude["region"], {(1, 2)})

    def test_selection_rejects_boolean_coordinate(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "selection.json"
            path.write_text('{"region":{"include":[[true,2]],"exclude":[]}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                mv._load_selection(path)

    def test_headless_dry_run_without_tk(self):
        with tempfile.TemporaryDirectory() as td:
            world = Path(td) / "World"
            region_dir = world / "region"
            world.mkdir()
            (world / "level.dat").write_bytes(b"test")
            raw = mv._build_void_chunk(0, 0, bytes([1] * 256), True, None,
                                       False, False, None, 1)
            make_region(region_dir, raw)
            proc = subprocess.run(
                [sys.executable, str(Path(mv.__file__)), str(world), "--headless",
                 "--all-chunks", "--include-void", "--dry-run"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("Would void 1 chunk(s)", proc.stdout)

    def test_headless_write_runs_post_verifier_and_report(self):
        with tempfile.TemporaryDirectory() as td:
            world = Path(td) / "World"
            region_dir = world / "region"
            world.mkdir()
            (world / "level.dat").write_bytes(b"test")
            raw = mv._build_void_chunk(0, 0, bytes([1] * 256), True, None,
                                       False, False, None, 1)
            make_region(region_dir, raw)
            report_path = Path(td) / "run-report.json"
            proc = subprocess.run(
                [sys.executable, str(Path(mv.__file__)), str(world), "--headless",
                 "--all-chunks", "--include-void", "--write", "--preset",
                 "safe-void", "--report", str(report_path)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["preset"], "safe-void")
            self.assertTrue(report["post_write_verification"]["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
