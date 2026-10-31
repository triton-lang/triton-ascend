#!/usr/bin/env python3
"""Dump `triton-opt` output for every DCV UT file into a directory.

Purpose: prove that a pass change is behaviour-neutral by comparing whole-file pass
output between two builds (`dump_dcv_ut.py build/eq_before`, rebuild, dump again,
`diff -rq`). Stronger than counting FileCheck passes: it catches any output drift,
including in checks the tests do not assert.

Usage: dump_dcv_ut.py <outdir>
Reuses the `// RUN:` lines of run_dcv_ut.py but drops the `| FileCheck` half.
Path-independent: repo root from $TRITON_ASCEND_ROOT, else the current directory,
else the default checkout; `triton-opt` from $TRITON_OPT or `build/cmake.*/bin/`.
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
UT_ROOT = f"{REPO}/third_party/ascend/unittest/Conversion/General/DynamicCVPipeline"
DIRS = ["PlanComputeBlock", "StandardizeOp", "SplitDataflow", "ComputeBlockOpt"]

if len(sys.argv) != 2:
    sys.exit("usage: dump_dcv_ut.py <outdir>")
outdir = os.path.abspath(sys.argv[1])
os.makedirs(outdir, exist_ok=True)

files = sorted(
    f
    for d in DIRS
    for f in glob.glob(f"{UT_ROOT}/{d}/**/*.mlir", recursive=True)
)
written = 0
failed: list[tuple[str, str]] = []
for f in files:
    dst = os.path.join(outdir, os.path.relpath(f, UT_ROOT).replace("/", "__"))
    runs = [ln.split("RUN:", 1)[1].strip()
            for ln in open(f, encoding="utf-8", errors="replace")
            if ln.startswith("// RUN")]
    if not runs:
        continue
    # Keep the producer half of the pipeline; `%t` becomes the dump path so a
    # RUN line that already writes `-o %t` lands there as well.
    cmd = runs[0].split("|")[0].strip().replace("%s", f).replace("%t", dst)
    cmd = re.sub(r"\btriton-opt\b", TRITON_OPT, cmd)
    if " -o " not in cmd:
        cmd += f" -o {dst}"
    p = subprocess.run(["bash", "-c", cmd], cwd=UT_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode == 0:
        written += 1
    else:
        failed.append((f, p.stdout.decode(errors="replace")))

print(f"dumped {written} files to {outdir}, {len(failed)} failed")
for f, out in failed:
    print(f"\n--- FAIL {os.path.relpath(f, UT_ROOT)}\n    {out[:1500]}")
sys.exit(1 if failed else 0)
