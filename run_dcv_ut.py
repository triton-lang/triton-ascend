#!/usr/bin/env python3
"""Minimal lit stand-in for the DynamicCVPipeline UT tree.

Real `lit` is unavailable here (no python `lit` package, no `llvm-lit` binary), so this
walks the same `// RUN:` lines lit would, substituting the tokens those tests use:
  %s  -> the .mlir file itself (shared by `triton-opt` and `FileCheck`)
  %t  -> a per-file scratch path (the `-o %t` + `FileCheck %s < %t` pair shares it)
  triton-opt / FileCheck -> the freshly built binary and the LLVM FileCheck

Path-independent: the repo root is taken from $TRITON_ASCEND_ROOT, else the current
directory, else the default checkout; `triton-opt`/`FileCheck` are discovered from that
root / the local toolchain. Run it from anywhere.

Only for local verification; not a replacement for the lit suite in CI.
"""
import glob
import os
import re
import subprocess
import sys

REPO_CANDIDATES = [
    os.environ.get("TRITON_ASCEND_ROOT", ""),
    os.getcwd(),
    os.path.expanduser("~/workspace/triton-ascend"),
]
REPO = next(
    (os.path.abspath(p) for p in REPO_CANDIDATES
     if p and os.path.isdir(f"{p}/third_party/ascend/unittest")), None)
if REPO is None:
    sys.exit("cannot locate the triton-ascend checkout: set TRITON_ASCEND_ROOT or cd into it")

TRITON_OPT = os.environ.get("TRITON_OPT") or next(
    iter(sorted(glob.glob(f"{REPO}/build/cmake.*/bin/triton-opt")) + ["triton-opt"]))


def find_filecheck():
    if os.environ.get("FILECHECK"):
        return os.environ["FILECHECK"]
    for pat in ("~/.triton/llvm/*/bin/FileCheck", "~/workspace/**/bin/FileCheck"):
        hits = sorted(glob.glob(os.path.expanduser(pat), recursive=True))
        if hits:
            return hits[-1]
    return "FileCheck"


FILECHECK = find_filecheck()
UT_ROOT = f"{REPO}/third_party/ascend/unittest/Conversion/General/DynamicCVPipeline"
DIRS = ["PlanComputeBlock", "StandardizeOp", "SplitDataflow", "ComputeBlockOpt"]

files = sorted(
    f
    for d in DIRS
    for f in glob.glob(f"{UT_ROOT}/{d}/**/*.mlir", recursive=True)
)
passed, failed, skipped = 0, [], 0
for f in files:
    runs = [ln.split("RUN:", 1)[1].strip()
            for ln in open(f, encoding="utf-8", errors="replace")
            if ln.startswith("// RUN")]
    if not runs:
        skipped += 1
        continue
    scratch = f"/tmp/ut_{os.path.basename(f)}.tmp"
    for run in runs:
        cmd = run.replace("%s", f).replace("%t", scratch)
        cmd = re.sub(r"\btriton-opt\b", TRITON_OPT, cmd)
        cmd = re.sub(r"\bFileCheck\b", FILECHECK, cmd)
        p = subprocess.run(["bash", "-c", cmd], cwd=UT_ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.returncode == 0:
            passed += 1
        else:
            failed.append((f, run, p.stdout.decode(errors="replace")))
    if os.path.exists(scratch):
        os.remove(scratch)

print(f"RUN-line checks: {passed} passed, {len(failed)} failed, {skipped} files skipped (no RUN)")
for f, run, out in failed:
    print(f"\n--- FAIL {os.path.relpath(f, UT_ROOT)}\n    RUN: {run}\n    {out[:1500]}")
sys.exit(1 if failed else 0)
