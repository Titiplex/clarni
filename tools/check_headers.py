"""Compile each public header independently with GCC/Clang or MSVC."""
from pathlib import Path
import argparse
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--msvc", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="clarni-headers-") as directory:
        source = Path(directory) / "header.cpp"
        for header in sorted((root / "headers").glob("*.h")):
            source.write_text(f'#include "{header.name}"\n', encoding="utf-8")
            if args.msvc:
                flags = ["/nologo", "/std:c++latest", "/Zs", "/EHsc", "/I" + str(root / "headers")]
            else:
                flags = ["-std=c++23", "-fsyntax-only", "-I" + str(root / "headers")]
            subprocess.run([args.compiler, *flags, str(source)], check=True, cwd=directory)
            print("PASS", header.name, flush=True)


if __name__ == "__main__":
    main()
