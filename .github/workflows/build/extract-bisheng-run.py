#!/usr/bin/env python3
"""Extract a CANN cann-bisheng-compiler .run payload.

The package's own makeself wrapper cannot extract it: the tar payload
contains a self-referential hardlink (toolkit/toolchain/hcc -> itself) that
makes plain tar exit non-zero. This script decompresses the gzip payload
directly, skips that entry, and extracts the rest.

Usage: extract-bisheng-run.py <cann-bisheng-compiler_*.run> <dest-dir>
"""
import gzip
import io
import os
import sys
import tarfile


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dest = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    off = data.find(b"\x1f\x8b")
    if off == -1:
        raise SystemExit(f"no gzip payload in {src}")
    raw = gzip.decompress(data[off:])
    tio = tarfile.open(fileobj=io.BytesIO(raw), mode="r:")
    os.makedirs(dest, exist_ok=True)
    errors = []
    for member in tio:
        if member.islnk() and member.linkname == member.name:
            print("SKIP self-referential hardlink:", member.name)
            continue
        try:
            tio.extract(member, dest)
        except Exception as exc:  # report and continue; ccec presence is checked later
            errors.append(f"{member.name}: {exc}")
    print(f"extracted {len(tio.getmembers())} entries, {len(errors)} errors")
    for error in errors[:20]:
        print("WARN", error)


if __name__ == "__main__":
    main()
