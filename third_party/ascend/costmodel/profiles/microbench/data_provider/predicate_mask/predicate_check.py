#!/usr/bin/env python3
"""Build/run the fixed-mode CCE probes, or summarize saved SYS_CNT counters.

Hardware runs require an explicitly selected idle device. Generated objects,
logs and metadata go to a fresh output directory, never into this directory.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
CASES = [("simd", mode) for mode in range(18)] + [("simt", mode) for mode in range(5)]
FIELDS = "route mode repeat c1 cm c2 mismatch sentinel distinct reset_threads rterr outsent".split()


def slope(row):
    """Validate host gates and compute unrounded cycles per source step."""
    simd = row["route"] == "simd"
    c1, cm, c2 = (int(row[key]) for key in ("c1", "cm", "c2"))
    if c2 <= c1 or abs(cm - (c1 + c2) / 2) / (c2 - c1) >= 0.03:
        raise ValueError("nonpositive slope or failed midpoint linearity gate")
    if any(int(row[key]) != 0 for key in ("mismatch", "sentinel", "rterr", "outsent")):
        raise ValueError("runtime/readback gate failed")
    if int(row["distinct"]) < (8 if simd else 32):
        raise ValueError("nontrivial-output gate failed")
    if not simd and int(row["mode"]) and int(row["reset_threads"]) < 16:
        raise ValueError("SIMT reset-arm gate failed")
    # SIMD: I=200/400/600, K=20, four vector steps per iteration.
    # SIMT: I=30/60/90, K=100, 1024 threads, eight state updates.
    return (c2 - c1) / (400 * 20 * 4 if simd else 60 * 100 * 1024 * 8)


def summarize(filename):
    values = {}
    with filename.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["route"], int(row["mode"]), int(row["repeat"])
            if key[:2] not in CASES or key[2] < 0 or key in values:
                raise ValueError(f"invalid/duplicate case: {key}")
            values[key] = slope(row)
    repeats = sorted({key[2] for key in values})
    expected = {(route, mode, repeat) for route, mode in CASES for repeat in repeats}
    if not repeats or set(values) != expected:
        raise ValueError("incomplete result matrix")
    print(f"Validated {len(values)} rows; units are hardware SYS_CNT cycles.")

    def show(label, route, terms):
        samples = [sum(weight * values[route, mode, repeat] for mode, weight in terms) for repeat in repeats]
        print(f"{label:36s} {min(samples):.8f} .. {max(samples):.8f}")

    print("SIMD: cycles / 64-element step; residuals are layout-dependent, not latency.")
    for base, chains in ((0, 4), (6, 1), (12, 2)):
        show(f"{chains}-chain select", "simd", [(base + 1, 1), (base, -1)])
        show(f"{chains}-chain compare", "simd", [(base + 2, 1), (base + 1, -1)])
        for offset, name in ((3, "and"), (4, "or"), (5, "xor")):
            show(f"{chains}-chain {name} residual", "simd", [(base + offset, 1), (base + 2, -2), (base + 1, 1)])
            show(f"{chains}-chain full {name}+2cmp+select", "simd", [(base + offset, 1), (base, -1)])
    print("SIMT: cycles / element update; pairs do not isolate individual operations.")
    for mode, name in ((1, "cmp+select"), (2, "extra cmp+and"), (3, "extra cmp+or"), (4, "extra cmp+xor")):
        show(name, "simt", [(mode, 1), (0 if mode == 1 else 1, -1)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", type=Path, help="CPU-only CSV check and paired differences")
    parser.add_argument("--toolkit", type=Path, default=os.environ.get("ASCEND_TOOLKIT_HOME"))
    parser.add_argument("--template-include", type=Path)
    parser.add_argument("--device", type=int, help="idle physical device index, inspected by the caller")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output", type=Path, help="new directory; default: unique temporary directory")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    if not args.toolkit or not args.template_include:
        parser.error("--toolkit and --template-include are required for building")
    if not args.build_only and (args.device is None or args.device < 0):
        parser.error("hardware runs require --device with an inspected idle device index")
    toolkit, include = args.toolkit.resolve(), args.template_include.resolve()
    out = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="predicate-cce-"))
    if args.output:
        out.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={out}", flush=True)

    def sha(filename):
        return hashlib.sha256(filename.read_bytes()).hexdigest()

    commands = []

    def call(command, log, **kwargs):
        commands.append(command)
        with log.open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180, **kwargs)

    compiler = toolkit / "bin/ccec"
    sources = [
        HERE / (stem + suffix) for stem in ("predicate_ops", "predicate_ops_simt") for suffix in (".cce", "_host.cpp")
    ]
    metadata = {
        "started": datetime.datetime.now().astimezone().isoformat(), "physical_device": args.device, "build_only":
        args.build_only, "compiler_version": subprocess.check_output([str(compiler), "--version"],
                                                                     text=True), "source_sha256":
        {source.name: sha(source)
         for source in sources}, "compiler_sha256": sha(compiler), "commands": commands, "object_sha256": {}
    }
    for stem in ("predicate_ops", "predicate_ops_simt"):
        call([
            "g++", "-O2", "-std=c++17",
            str(HERE / f"{stem}_host.cpp"), "-o",
            str(out / f"{stem}_host"), f"-I{toolkit}/x86_64-linux/pkg_inc", f"-I{toolkit}/include",
            f"-L{toolkit}/lib64", "-lruntime", "-lascendcl"
        ], out / f"{stem}_host_build.log")
    for route, mode in CASES:
        stem = "predicate_ops" + ("_simt" if route == "simt" else "")
        directory = out / f"{route}_m{mode}"
        directory.mkdir()
        obj = directory / f"{stem}.o"
        call([
            str(compiler), "-c", "-std=c++17", "-O2", "--cce-aicore-only", "--cce-aicore-arch=dav-c310",
            "-DREG_REGISTER_SIZE=256", f"-DFIXED_MODE={mode}", f"-I{include}",
            str(HERE / f"{stem}.cce"), "-o",
            str(obj)
        ], directory / "build.log")
        relocations = subprocess.check_output(["readelf", "-r", str(obj)], text=True)
        if any(not section.startswith((".rela.debug", ".rel.debug"))
               for section in re.findall(r"Relocation section '([^']+)'", relocations)):
            raise RuntimeError(f"unresolved device relocations: {obj}")
        metadata["object_sha256"][f"{route}_m{mode}"] = sha(obj)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.build_only:
        print("Build complete; no device was used.")
        return

    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
    # Deliberately exclude simulator libraries for physical measurements.
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(toolkit / "lib64"), "/usr/local/Ascend/driver/lib64/driver", "/usr/local/Ascend/driver/lib64/common"])
    with (out / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for route, mode in CASES:
            stem = "predicate_ops" + ("_simt" if route == "simt" else "")
            directory = out / f"{route}_m{mode}"
            for repeat in range(2):
                log = directory / f"physical_r{repeat}.log"
                call([str(out / f"{stem}_host"), str(mode)], log, cwd=directory, env=env)
                match = re.search(r"^RESULT (.+)$", log.read_text(), re.MULTILINE)
                if not match:
                    raise RuntimeError(f"missing RESULT: {log}")
                fields = dict(re.findall(r"(\w+)=([^\s]+)", match[1]))
                if int(fields["mode"]) != mode:
                    raise RuntimeError(f"mode mismatch: {log}")
                row = {"route": route, "repeat": repeat, **fields}
                slope(row)
                writer.writerow({key: row.get(key, "") for key in FIELDS})
                stream.flush()
    summarize(out / "results.csv")


if __name__ == "__main__":
    main()
