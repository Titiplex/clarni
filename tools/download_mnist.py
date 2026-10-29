"""Download raw MNIST IDX files from the CVDF mirror; Python standard library only.

MD5 values are the conventional MNIST archive integrity checksums, not signatures.
Source: https://github.com/cvdfoundation/mnist
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import os
import tempfile
import urllib.request

RESOURCES = [
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873", 47040016),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432", 60008),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3", 7840016),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c", 10008),
]


def prepare(destination: Path, force: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename, expected_md5, raw_size in RESOURCES:
        url = "https://storage.googleapis.com/cvdf-datasets/mnist/" + filename
        with urllib.request.urlopen(url, timeout=60) as response:
            compressed = response.read(20 * 1024 * 1024 + 1)
        if len(compressed) > 20 * 1024 * 1024:
            raise ValueError(f"Oversized MNIST archive: {filename}")
        if hashlib.md5(compressed).hexdigest() != expected_md5:
            raise ValueError(f"MNIST archive checksum mismatch: {filename}")
        raw = gzip.decompress(compressed)
        if len(raw) != raw_size:
            raise ValueError(f"MNIST decompressed size mismatch: {filename}")
        path = destination / filename.removesuffix(".gz")
        if path.exists():
            if path.read_bytes() == raw:
                print(f"Unchanged: {path}", flush=True)
                continue
            if not force:
                raise FileExistsError(f"Different existing file: {path}; use another folder or --force")
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=destination)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(raw)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(f"Verified and extracted: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args.destination, args.force)
