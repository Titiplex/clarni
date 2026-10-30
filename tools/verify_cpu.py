"""Build the CPU library and tests using GCC, without requiring CMake."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--build", default="build-direct")
parser.add_argument("--sanitize", action="store_true")
parser.add_argument("--examples", action="store_true", help="Build and run examples (MNIST requires --mnist-data)")
parser.add_argument("--mnist-data", type=Path, help="Optional directory containing raw MNIST IDX files")
parser.add_argument("--no-leak-check", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
build = root / args.build
build.mkdir(parents=True, exist_ok=True)


def run(command):
    environment = os.environ.copy()
    if args.no_leak_check:
        environment["ASAN_OPTIONS"] = environment.get("ASAN_OPTIONS", "") + ":detect_leaks=0"
    return subprocess.run(command, cwd=root, text=True, capture_output=True, env=environment)


flags = ["-std=c++23", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Iheaders", "-Itests"]
if args.sanitize:
    flags = [flag for flag in flags if flag != "-O2"]
    flags += ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
objects = []
for source in sorted((root / "src").glob("*.cpp")) + sorted((root / "src/gpu").glob("*stub.cpp")):
    obj = build / (source.stem + ".o")
    result = run(["g++", *flags, "-c", str(source), "-o", str(obj)])
    if result.returncode:
        print(result.stdout + result.stderr)
        sys.exit(result.returncode)
    if result.stderr:
        print(result.stderr)
    objects.append(str(obj))
library = build / "libML.a"
if library.exists():
    library.unlink()
subprocess.run(["ar", "rcs", str(library), *objects], check=True)
records = []
cases = [(source.stem, source) for source in sorted((root / "ML/tests").glob("test_*.cpp"))]
cases.append(("test_residency_mock", root / "tests/test_residency.cpp"))
for name, source in cases:
    sources = [str(source), str(library)]
    if name == "test_dispatch":
        sources = [str(source), "src/matrix.cpp", "src/compute_device.cpp"]
    executable = build / name
    defines = ["-DCLARNI_EXPECT_CUDA=0"] if name == "test_device" else []
    if name == "test_residency_mock":
        sources = [str(source), str(root / "tests/mock_resident_backend.cpp"), str(library)]
        defines = ["-DCLARNI_MOCK_GPU=1"]
    compiled = run(["g++", *flags, *defines, *sources, "-o", str(executable)])
    if compiled.returncode:
        print(compiled.stdout + compiled.stderr)
        sys.exit(compiled.returncode)
    tested = run([str(executable)])
    print(name + ":\n" + tested.stdout + tested.stderr, flush=True)
    records.append({"test": name, "returncode": tested.returncode, "output": tested.stdout + tested.stderr})
if args.examples:
    for source in sorted((root / "examples").glob("*.cpp")):
        name = "example_" + source.stem
        executable = build / name
        compiled = run(["g++", *flags, str(source), str(library), "-o", str(executable)])
        if compiled.returncode:
            print(compiled.stdout + compiled.stderr)
            sys.exit(compiled.returncode)
        if compiled.stderr:
            print(compiled.stderr)
        command = [str(executable)]
        if source.stem == "mnist":
            if args.mnist_data is None:
                print(name + ": SKIP (supply --mnist-data to run)", flush=True)
                continue
            command += ["--data", str(args.mnist_data.resolve()), "--output", str(build / "mnist-run"),
                        "--train-limit", "300", "--validation", "60", "--test-limit", "100", "--epochs", "1"]
        tested = run(command)
        print(name + ":\n" + tested.stdout + tested.stderr, flush=True)
        records.append({"test": name, "returncode": tested.returncode, "output": tested.stdout + tested.stderr})
(build / "results.json").write_text(json.dumps(records, indent=2))
sys.exit(any(item["returncode"] not in (0, 77) for item in records))
