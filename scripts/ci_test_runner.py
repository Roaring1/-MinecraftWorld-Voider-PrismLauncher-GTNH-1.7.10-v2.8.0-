from __future__ import annotations
import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Iterator

def _iter_cases(suite: unittest.TestSuite) -> Iterator[unittest.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_cases(item)
        else:
            yield item

def main() -> int:
    parser = argparse.ArgumentParser(description="Run each unit test in an isolated subprocess with a timeout.")
    parser.add_argument("--timeout", type=float, default=90.0, help="Maximum seconds allowed per test.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    root = Path(__file__).resolve().parents[1]
    tests_dir = root / "tests"
    sys.path.insert(0, str(tests_dir))
    sys.path.insert(0, str(root))
    import test_core
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_core)
    cases = list(_iter_cases(suite))
    if not cases:
        print("ERROR: no test cases were discovered.", file=sys.stderr)
        return 2
    env = os.environ.copy()
    paths = [str(root), str(tests_dir)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    failures = 0
    timeouts = 0
    for index, case in enumerate(cases, 1):
        test_id = case.id()
        print(f"[{index}/{len(cases)}] START {test_id}", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, "-u", "-m", "unittest", test_id, "-v"],
                cwd=root, env=env, text=True, capture_output=True, check=False,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            timeouts += 1
            failures += 1
            print(f"[{index}/{len(cases)}] TIMEOUT after {args.timeout:.1f}s: {test_id}",
                  file=sys.stderr, flush=True)
            if exc.stdout:
                print(exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode(errors="replace"))
            if exc.stderr:
                print(exc.stderr if isinstance(exc.stderr, str)
                      else exc.stderr.decode(errors="replace"), file=sys.stderr)
            continue
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
        if result.returncode:
            failures += 1
            print(f"[{index}/{len(cases)}] FAIL exit={result.returncode}: {test_id}",
                  file=sys.stderr, flush=True)
        else:
            print(f"[{index}/{len(cases)}] PASS {test_id}", flush=True)
    print(f"Isolated test summary: {len(cases) - failures} passed, "
          f"{failures} failed, {timeouts} timed out.", flush=True)
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
