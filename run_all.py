"""Run all portfolio demonstrations."""

import runpy
from pathlib import Path


def main():
    demo_files = sorted(Path("demos").glob("*.py"))

    for demo_file in demo_files:
        print(f"\nRunning {demo_file.name}")
        print("-" * 60)
        runpy.run_path(str(demo_file), run_name="__main__")


if __name__ == "__main__":
    main()
