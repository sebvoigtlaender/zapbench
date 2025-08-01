#!/usr/bin/env python3
"""Test runner for ZAPBench compatible with Google's absltest framework.

This script runs tests using direct Python execution instead of pytest,
ensuring compatibility with absl flags and Google's testing infrastructure.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py infer_test         # Run specific test
    python run_tests.py ts_forecasting     # Run tests in directory
    python run_tests.py --verbose          # Verbose output
    python run_tests.py --fast             # Skip slow tests
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def find_test_files(pattern: str = "") -> List[Path]:
  """Find all test files matching the pattern."""
  project_root = Path(__file__).parent

  if not pattern:
    # Find all test files
    return list(project_root.glob("**/*_test.py"))

  # Check if pattern is a specific test file
  if pattern.endswith("_test.py") or pattern.endswith("_test"):
    test_name = pattern if pattern.endswith(".py") else f"{pattern}.py"
    matches = list(project_root.glob(f"**/{test_name}"))
    if matches:
      return matches

  # Check if pattern is a directory
  dir_path = project_root / pattern
  if dir_path.is_dir():
    return list(dir_path.glob("**/*_test.py"))

  # Pattern matching in filenames
  return list(project_root.glob(f"**/*{pattern}*_test.py"))


def run_test(
    test_file: Path, verbose: bool = False, fast: bool = False
) -> tuple[bool, float]:
  """Run a single test file using direct Python execution."""
  start_time = time.time()

  # Prepare environment
  env = os.environ.copy()
  env["PYTHONPATH"] = str(Path(__file__).parent)

  # Prepare command (no special flags for fast mode - just use subprocess timeout)
  cmd = [sys.executable, str(test_file)]

  try:
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=not verbose,
        text=True,
        timeout=300 if not fast else 60,  # 5min normal, 1min fast
    )

    duration = time.time() - start_time
    success = result.returncode == 0

    if not success and not verbose:
      print(f"FAIL {test_file.relative_to(Path.cwd())}")
      print(f"   STDOUT: {result.stdout}")
      print(f"   STDERR: {result.stderr}")
    elif success:
      print(f"PASS {test_file.relative_to(Path.cwd())} ({duration:.1f}s)")

    return success, duration

  except subprocess.TimeoutExpired:
    duration = time.time() - start_time
    print(
        f"TIMEOUT {test_file.relative_to(Path.cwd())} (timeout after"
        f" {duration:.1f}s)"
    )
    return False, duration
  except Exception as e:
    duration = time.time() - start_time
    print(f"ERROR {test_file.relative_to(Path.cwd())}: {e}")
    return False, duration


def main():
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
      "pattern", nargs="?", default="", help="Test file pattern or directory"
  )
  parser.add_argument(
      "-v", "--verbose", action="store_true", help="Verbose output"
  )
  parser.add_argument(
      "-f", "--fast", action="store_true", help="Fast mode (reduced timeouts)"
  )
  parser.add_argument(
      "--list", action="store_true", help="List test files without running"
  )

  args = parser.parse_args()

  # Find test files
  test_files = find_test_files(args.pattern)

  if not test_files:
    print(f"No test files found for pattern: '{args.pattern}'")
    return 1

  # Sort for consistent ordering
  test_files.sort()

  if args.list:
    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
      print(f"  {test_file.relative_to(Path.cwd())}")
    return 0

  print(f"Running {len(test_files)} test files...")
  print(f"Mode: {'Fast' if args.fast else 'Normal'}")
  print("-" * 60)

  # Run tests
  total_start = time.time()
  passed = 0
  failed = 0
  total_duration = 0

  for test_file in test_files:
    success, duration = run_test(test_file, args.verbose, args.fast)
    total_duration += duration

    if success:
      passed += 1
    else:
      failed += 1

  # Summary
  total_time = time.time() - total_start
  print("-" * 60)
  print(f"Results: {passed} passed, {failed} failed ({total_time:.1f}s total)")

  if failed > 0:
    print(f"\nFAIL {failed} tests failed")
    return 1
  else:
    print(f"\nPASS All {passed} tests passed!")
    return 0


if __name__ == "__main__":
  sys.exit(main())
