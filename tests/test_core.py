from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest import mock
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


class ClientServerCompatibilityTests(unittest.TestCase):
    def test_dedicated_server_world_and_dimensions_are_detected(self):
        with tempfile.TemporaryDirectory() as td:
            server = Path(td) / "server"
            world = server / "world"
            world.mkdir(parents=True)
            (server / "server.properties").write_text("level-name=world\n", encoding="utf-8")
            (world / "level.dat").write_bytes(b"test")
            raw_overworld = mv._build_void_chunk(0, 0, bytes([1] * 256), True, None,
                                                 False, False, None, 1)
            raw_nether = mv._build_void_chunk(0, 0, bytes([8] * 256), True, None,
                                              False, False, None, 0)
            make_region(world / "region", raw_overworld)
            make_region(world / "DIM-1" / "region", raw_nether)
            inventory = mv.inspect_world(world)
            self.assertEqual(inventory["kind"], "dedicated-server")
            self.assertEqual(inventory["region_files"], 2)
            self.assertEqual({d["tag"] for d in inventory["dimensions"]},
                             {"region", "DIM-1/region"})
            self.assertTrue(mv.verify_world(world)["ok"])

    @unittest.skipUnless(os.name == "posix", "POSIX lock test")
    def test_active_server_lock_blocks_writes(self):
        with tempfile.TemporaryDirectory() as td:
            world = Path(td) / "world"
            world.mkdir()
            (world / "level.dat").write_bytes(b"test")
            lock_path = world / "session.lock"
            lock_path.write_bytes(b"lock")
            holder = subprocess.Popen([
                sys.executable, "-c",
                "import fcntl,sys,time; f=open(sys.argv[1],'r+b'); "
                "fcntl.lockf(f.fileno(),fcntl.LOCK_EX); print('locked',flush=True); time.sleep(10)",
                str(lock_path),
            ], stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(holder.stdout.readline().strip(), "locked")
                status, _ = mv._world_lock_status(world)
                self.assertEqual(status, "active")
                with self.assertRaises(RuntimeError):
                    mv._assert_world_not_active(world)
            finally:
                holder.terminate()
                holder.wait(timeout=5)
                if holder.stdout is not None:
                    holder.stdout.close()


class LinuxInstallerTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "Linux installer test")
    def test_native_installer_creates_launcher_and_desktop_entry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = root / "bundle"
            home = root / "home"
            xdg = home / ".local" / "share"
            bundle.mkdir(parents=True)
            home.mkdir()
            script = Path(mv.__file__).with_name("install_native_linux.sh")
            installer = bundle / script.name
            installer.write_bytes(script.read_bytes())
            installer.chmod(0o755)
            binary = bundle / "Voider-GTNH"
            binary.write_text("#!/usr/bin/env bash\necho native-ok\n", encoding="utf-8")
            binary.chmod(0o755)
            (bundle / "image.png").write_bytes(b"png")
            env = os.environ.copy()
            env.update({"HOME": str(home), "XDG_DATA_HOME": str(xdg)})
            proc = subprocess.run([str(installer)], env=env, text=True,
                                  capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            launcher = home / ".local" / "bin" / "gtnh-world-voider"
            desktop = xdg / "applications" / "gtnh-world-voider.desktop"
            self.assertTrue(launcher.is_symlink())
            self.assertTrue(os.access(launcher.resolve(), os.X_OK))
            self.assertIn("Exec=", desktop.read_text(encoding="utf-8"))


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



class HardeningRegressionTests(unittest.TestCase):
    def _world(self, root: Path, name: str = "VOIDER_HARDENING_FIXTURE_DISPOSABLE_DO_NOT_IMPORT") -> Path:
        world = root / name
        world.mkdir(parents=True)
        (world / "level.dat").write_bytes(b"synthetic test fixture, not a playable save")
        return world

    @staticmethod
    def _write_args(**overrides):
        values = dict(
            platform_x=None, platform_z=None, platform_size=3, platform_y=64,
            platform_id=7, platform_meta=0, only=None, exclude=None, bbox=None,
            center_x=None, center_z=None, radius_blocks=0, dry_run=False,
            backup=True, per_file_backup=False, wait_for_unlock=0,
            include_void=True, keep_entities=False, keep_tileentities=False,
            skip_if_entities=False, skip_if_tileentities=False, mark_void=True,
            restore_biomes=False, nether_cosmetic_placeholder=True, workers=1,
        )
        values.update(overrides)
        return Namespace(**values)

    def test_void_detection_includes_extended_block_ids(self):
        raw = mv._build_void_chunk(0, 0, bytes([1] * 256), True, None,
                                   False, False, None, 1)
        parsed = mv.nbt.NBTFile(buffer=io.BytesIO(raw))
        section = parsed["Level"]["Sections"][0]
        add = mv.nbt.TAG_Byte_Array(name="Add")
        add.value = [0] * 2048
        mv._nibble_set(add.value, 0, 1)
        section.tags.append(add)
        output = io.BytesIO()
        parsed.write_file(buffer=output)
        self.assertFalse(mv._is_void_chunk(output.getvalue()))

    def test_backup_manifest_hashes_each_world_file(self):
        with tempfile.TemporaryDirectory(prefix="voider-backup-fixture-") as td:
            world = self._world(Path(td))
            payload = b"important world data"
            (world / "region.dat").write_bytes(payload)
            archive_path = mv._backup_world_zip(world)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertIsNone(archive.testzip())
                manifest = json.loads(archive.read(".voider-backup-manifest.json"))
                self.assertEqual(manifest["format"], "gtnh-world-voider-backup-v1")
                self.assertEqual(manifest["file_count"], 2)
                self.assertEqual(
                    manifest["files"]["region.dat"]["sha256"],
                    hashlib.sha256(payload).hexdigest(),
                )
                self.assertEqual(archive.read("region.dat"), payload)

    @unittest.skipUnless(os.name == "posix", "POSIX record-lock test")
    def test_write_guard_holds_session_lock_across_processes(self):
        with tempfile.TemporaryDirectory(prefix="voider-lock-fixture-") as td:
            world = self._world(Path(td))
            snippet = "\n".join([
                "import pathlib,sys",
                "import main_voider as mv",
                "world=pathlib.Path(sys.argv[1])",
                "try:",
                "    with mv._world_write_guard(world):",
                "        raise SystemExit(0)",
                "except RuntimeError as exc:",
                "    print(exc)",
                "    raise SystemExit(23)",
            ])
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(mv.__file__).resolve().parent)
            with mv._world_write_guard(world):
                proc = subprocess.run(
                    [sys.executable, "-c", snippet, str(world)],
                    env=env, text=True, capture_output=True, check=False,
                )
            self.assertEqual(proc.returncode, 23, proc.stdout + proc.stderr)
            self.assertIn("session.lock is locked", proc.stdout)

    def test_server_instance_root_resolves_level_name_and_rejects_escape(self):
        with tempfile.TemporaryDirectory(prefix="voider-server-layout-") as td:
            server = Path(td) / "server"
            world = server / "h"
            world.mkdir(parents=True)
            (server / "server.properties").write_text("level-name=h\n", encoding="utf-8")
            (world / "level.dat").write_bytes(b"fixture")
            self.assertEqual(mv._resolve_world_path(server), world.resolve())
            (server / "server.properties").write_text("level-name=../outside\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                mv._resolve_world_path(server)

    def test_biome_reset_is_backed_up_staged_and_verified(self):
        with tempfile.TemporaryDirectory(prefix="VOIDER_HARDENING_FIXTURE_DISPOSABLE_DO_NOT_IMPORT_") as td:
            server = Path(td) / "SERVER_INSTANCE_DO_NOT_IMPORT"
            world = server / "h"
            world.mkdir(parents=True)
            (server / "server.properties").write_text("level-name=h\n", encoding="utf-8")
            (world / "level.dat").write_bytes(b"synthetic disposable fixture")
            raw = mv._build_void_chunk(0, 0, None, False, None, False, False, None, 1)
            make_region(world / "region", raw)
            changed = mv.reset_biomes(server, ["region"])
            self.assertEqual(changed, 1)
            self.assertTrue(mv.verify_world(world)["ok"])
            with world.joinpath("region", "r.0.0.mca").open("rb") as fp:
                _name, off, count = next(mv._iter_chunk_locations(fp))
                chunk = mv._read_chunk_payload(fp, off, count)
            level = read_level(chunk)
            self.assertEqual(bytes(level["Biomes"].value), bytes([mv.PLAINS_ID] * 256))
            self.assertEqual(int(level["TerrainPopulated"].value), 0)
            self.assertEqual(int(level["LightPopulated"].value), 0)
            backups = list(server.glob("h.void-backup-*.zip"))
            self.assertEqual(len(backups), 1)
            with zipfile.ZipFile(backups[0]) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn(".voider-backup-manifest.json", archive.namelist())
                self.assertIn("region/r.0.0.mca", archive.namelist())

    def test_biome_reset_aborts_before_commit_on_corrupt_chunk(self):
        with tempfile.TemporaryDirectory(prefix="voider-biome-abort-") as td:
            world = self._world(Path(td))
            region = make_region(world / "region", b"not valid NBT")
            original = region.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "verified backup"):
                mv.reset_biomes(world, ["region"])
            self.assertEqual(region.read_bytes(), original)
            self.assertTrue(list(world.parent.glob(f"{world.name}.void-backup-*.zip")))
            self.assertEqual(list(region.parent.glob(".*.tmp")), [])

    def test_voider_worker_failure_is_reported_and_original_is_untouched(self):
        with tempfile.TemporaryDirectory(prefix="voider-worker-failure-") as td:
            world = self._world(Path(td))
            region = make_region(world / "region", b"not valid NBT")
            original = region.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "Voider did not complete"):
                mv.run_voider(world, {}, {}, self._write_args())
            self.assertEqual(region.read_bytes(), original)
            backups = list(world.parent.glob(f"{world.name}.void-backup-*.zip"))
            self.assertEqual(len(backups), 1)
            with zipfile.ZipFile(backups[0]) as archive:
                self.assertIsNone(archive.testzip())

    def test_keep_tileentities_means_skip_not_orphan(self):
        with tempfile.TemporaryDirectory(prefix="voider-tileentity-fixture-") as td:
            world = self._world(Path(td))
            raw = mv._build_void_chunk(0, 0, bytes([1] * 256), False, None,
                                       False, False, None, 1)
            parsed = mv.nbt.NBTFile(buffer=io.BytesIO(raw))
            section = parsed["Level"]["Sections"][0]
            section["Blocks"].value[0] = 1
            tile_entities = mv.nbt.TAG_List(name="TileEntities", type=mv.nbt.TAG_Compound)
            tile_entities.tags.append(mv.nbt.TAG_Compound())
            parsed["Level"].tags = [
                tag for tag in parsed["Level"].tags if getattr(tag, "name", "") != "TileEntities"
            ]
            parsed["Level"].tags.append(tile_entities)
            output = io.BytesIO()
            parsed.write_file(buffer=output)
            region = make_region(world / "region", output.getvalue())
            original = region.read_bytes()
            result = mv._process_region(
                region, dimtag="region", bbox=None, center=None, radius_blocks=0,
                include_void=True, platform=None, keep_entities=False,
                keep_tileentities=True, skip_if_entities=False,
                skip_if_tileentities=False, mark_void=True, dry_run=False,
                strict=True, sel_include=set(), sel_exclude=set(),
                restore_biomes_from_bak=False,
                nether_cosmetic_placeholder=False,
            )
            self.assertEqual(result[2], 0)
            self.assertEqual(region.read_bytes(), original)

    def test_preserve_data_preset_never_orphans_tileentities(self):
        args = self._write_args(
            preset="preserve-data", skip_if_entities=False,
            skip_if_tileentities=False, keep_entities=False,
            keep_tileentities=True, mark_void=False,
        )
        mv._apply_preset(args)
        self.assertTrue(args.keep_entities)
        self.assertTrue(args.skip_if_tileentities)
        self.assertFalse(args.keep_tileentities)

    def test_cancelled_write_stops_before_backup_or_region_change(self):
        with tempfile.TemporaryDirectory(prefix="voider-cancel-fixture-") as td:
            world = self._world(Path(td))
            raw = mv._build_void_chunk(0, 0, bytes([1] * 256), True, None,
                                       False, False, None, 1)
            region = make_region(world / "region", raw)
            original = region.read_bytes()
            cancelled = __import__("threading").Event()
            cancelled.set()
            with self.assertRaisesRegex(InterruptedError, "already cancelled"):
                mv.run_voider(world, {}, {}, self._write_args(), cancel_event=cancelled)
            self.assertEqual(region.read_bytes(), original)
            self.assertEqual(list(world.parent.glob(f"{world.name}.void-backup-*.zip")), [])



    def test_write_guard_rejects_same_process_reentrancy(self):
        with tempfile.TemporaryDirectory(prefix="voider-thread-lock-") as td:
            world = self._world(Path(td))
            with mv._world_write_guard(world):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with mv._world_write_guard(world):
                        self.fail("nested same-process writer must never enter")

    def test_empty_biome_dimension_list_does_not_expand_to_all(self):
        with tempfile.TemporaryDirectory(prefix="voider-empty-dims-") as td:
            world = self._world(Path(td))
            raw = mv._build_void_chunk(0, 0, None, False, None, False, False, None, 1)
            region = make_region(world / "region", raw)
            original = region.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "No region files"):
                mv.reset_biomes(world, [])
            self.assertEqual(region.read_bytes(), original)
            self.assertEqual(list(world.parent.glob(f"{world.name}.void-backup-*.zip")), [])

    def test_biome_reset_rolls_back_committed_regions_when_postverify_fails(self):
        with tempfile.TemporaryDirectory(prefix="voider-biome-rollback-") as td:
            world = self._world(Path(td))
            region = make_region(
                world / "region",
                mv._build_void_chunk(0, 0, None, False, None, False, False, None, 1),
                rx=0,
                rz=0,
                idx=0,
            )
            original = region.read_bytes()
            with mock.patch.object(
                mv, "verify_world",
                return_value={"ok": False, "errors": ["synthetic verifier rejection"]},
            ):
                with self.assertRaisesRegex(RuntimeError, "rolled back"):
                    mv.reset_biomes(world, ["region"])
            self.assertEqual(region.read_bytes(), original)
            self.assertEqual(list(region.parent.glob(".*.voider-rollback-*")), [])
            backups = list(world.parent.glob(f"{world.name}.void-backup-*.zip"))
            self.assertEqual(len(backups), 1)
            with zipfile.ZipFile(backups[0]) as archive:
                self.assertIsNone(archive.testzip())

    @unittest.skipUnless(hasattr(os, "symlink"), "Symlinks unsupported")
    def test_backup_refuses_symlinks_in_world_tree(self):
        with tempfile.TemporaryDirectory(prefix="voider-symlink-backup-") as td:
            root = Path(td)
            world = self._world(root)
            outside = root / "outside.txt"
            outside.write_text("must not be followed", encoding="utf-8")
            link = world / "external-link.txt"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("Filesystem does not permit symlink creation")
            with self.assertRaisesRegex(RuntimeError, "symlinked file"):
                mv._backup_world_zip(world)
            self.assertEqual(list(root.glob(f"{world.name}.void-backup-*.zip")), [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
