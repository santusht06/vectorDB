"""
VectorForge Master Test Suite & Validation Script

Runs unit tests, integration tests, API tests, and benchmark verification.
"""

import subprocess
import sys


def run_step(name: str, cmd: list[str]) -> bool:
    print(f"\n========================================================")
    print(f" STEP: {name}")
    print(f" COMMAND: {' '.join(cmd)}")
    print(f"========================================================")
    result = subprocess.run(cmd, text=True)
    if result.returncode == 0:
        print(f"SUCCESS: {name}")
        return True
    else:
        print(f"FAILED: {name} (exit code {result.returncode})")
        return False


def main():
    steps = [
        ("Unit Tests", [sys.executable, "-m", "pytest", "tests/unit", "-v"]),
        ("Integration Tests", [sys.executable, "-m", "pytest", "tests/integration", "-v"]),
        ("API Tests", [sys.executable, "-m", "pytest", "tests/api", "-v"]),
        ("Generate Synthetic Dataset", [sys.executable, "scripts/generate_dataset.py", "-n", "1000", "-d", "128"]),
        ("Build Indexes", [sys.executable, "scripts/build_indexes.py"]),
        ("Generate Ground Truth", [sys.executable, "scripts/generate_ground_truth.py"]),
        ("Run Benchmarks", [sys.executable, "scripts/benchmark.py"]),
    ]

    failed = []
    for name, cmd in steps:
        success = run_step(name, cmd)
        if not success:
            failed.append(name)

    print("\n========================================================")
    print(" SUMMARY")
    print("========================================================")
    if not failed:
        print("ALL STEPS PASSED SUCCESSFULLY! VectorForge is fully verified.")
        sys.exit(0)
    else:
        print(f"FAILED STEPS: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
