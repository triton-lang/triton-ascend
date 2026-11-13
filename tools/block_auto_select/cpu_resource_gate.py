"""CPU-only resource gate for an already lowered Block candidate.

This is not a performance selector or a numerical correctness check. The input
must be the candidate's own TTAdapter IR; no fallback compilation is performed.
The supported first path is pinned A5 SIMD compilation with static UB buffers.
No Torch, Triton driver, device discovery, loading or launch API is imported.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

PASS_MARKER = "IR Dump After PlanMemory (hivm-plan-memory)"
OVERFLOW = re.compile(r"ub overflow, requires (\d+) bits while (\d+) bits available", re.I)
TARGET = re.compile(r'hacc\.target\s*=\s*#hacc\.target<"([^"\n]+)">')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_final_ub(ir, target, *, allow_i1=False, alignment_bits=8):
    """Read final constant UB addresses, rather than guessing from input size.

    The backend is the resource authority. This additionally checks the final
    allocated address extent for the narrow static-buffer syntax we understand.
    Unknown/dynamic forms are reported, never interpreted as zero allocation.
    """
    if type(alignment_bits) is not int or alignment_bits < 8 or alignment_bits % 8:
        raise ValueError("invalid UB alignment")
    if set(TARGET.findall(ir)) != {target}:
        raise ValueError("final IR target does not match requested target")
    limits = {int(v) for v in re.findall(r'"UB_SIZE",\s*(\d+)\s*:\s*i\d+', ir)}
    if len(limits) != 1 or min(limits) <= 0:
        raise ValueError("missing or ambiguous backend UB capacity")
    limit = limits.pop()
    constants = {}
    buffers = []
    function = None
    for line in ir.splitlines():
        match = re.match(r"\s*func\.func\s+(?:private\s+)?@([^\s(]+)", line)
        if match:
            function = match.group(1)
            constants = {}
        match = re.match(r"\s*(%[\w.$-]+) = llvm\.mlir\.constant\((-?\d+) : (?:i\d+|index)\)", line)
        if match:
            constants[match.group(1)] = int(match.group(2))
        if "#hivm.address_space<ub>" not in line:
            continue
        if "memref.alloc" in line:
            raise ValueError("unresolved UB allocation remains in final IR")
        if "hivm.hir.pointer_cast" not in line:
            continue
        match = re.search(
            r"hivm\.hir\.pointer_cast\((%[\w.$-]+)\)\s*:\s*"
            r"memref<((?:\d+x)*)(bf16|[if]\d+),\s*#hivm\.address_space<ub>>", line)
        if not match or match.group(1) not in constants or function is None:
            raise ValueError("UB pointer has an unproved offset, shape or layout")
        offset = constants[match.group(1)]
        dimensions = [int(v) for v in match.group(2).split("x") if v]
        element_bits = 16 if match.group(3) == "bf16" else int(match.group(3)[1:])
        conservative = element_bits == 1 and allow_i1
        if offset < 0 or (not conservative and (element_bits < 8 or element_bits % 8)):
            raise ValueError("unsupported UB element width or negative address")
        if any(v <= 0 for v in dimensions):
            raise ValueError("nonpositive UB extent")
        bits = math.prod(dimensions) * (8 if conservative else element_bits)
        bits = (bits + alignment_bits - 1) // alignment_bits * alignment_bits
        buffers.append({
            "function": function, "offset_bytes": offset, "size_bits": bits, "end_bits": offset * 8 + bits,
            "conservative_i1": conservative
        })
    if not buffers:
        # A pure Cube module can legitimately use no UB. Require every
        # definition/declaration to be explicitly AIC and no UB-typed value;
        # absence of an allocation alone is not proof for AIV/mixed modules.
        functions = [line for line in ir.splitlines() if re.match(r"\s*func\.func\s", line)]
        pure_cube = functions and all(
            re.search(r"hivm\.func_core_type\s*=\s*#hivm\.func_core_type<AIC>", line) for line in functions)
        if pure_cube and not re.search(r"address_space\s*<\s*ub\s*>", ir):
            return {
                "capacity_bits": limit, "address_extent_bits": 0, "buffers": [], "evidence":
                "backend final module has only explicitly AIC functions and no UB values"
            }
        raise ValueError("no verifiable static UB allocation in final IR")
    return {
        "capacity_bits": limit, "address_extent_bits": max(b["end_bits"] for b in buffers), "conservative_upper_bound":
        any(b["conservative_i1"] for b in buffers), "buffers": buffers, "evidence":
        "backend final HIVM constant UB addresses"
    }


def assess(returncode, stdout, stderr, artifact, final_ir, target):
    result = {"resource_checked": False, "numerical_correctness_verified": False, "performance_verified": False}
    overflow = OVERFLOW.findall(stdout + "\n" + stderr)
    if returncode != 0:
        result.update(status="UB_REJECTED" if overflow else "COMPILE_REJECTED",
                      overflow=[{"required_bits": int(a), "available_bits": int(b)} for a, b in overflow])
        return result
    if not artifact.is_file() or artifact.read_bytes()[:4] != b"\x7fELF":
        return dict(result, status="RESOURCE_UNCONFIRMED", reason="final ELF artifact missing")
    if PASS_MARKER not in stderr or not final_ir.is_file():
        return dict(result, status="RESOURCE_UNCONFIRMED", reason="backend allocation evidence missing")
    try:
        usage = inspect_final_ub(final_ir.read_text(), target)
    except ValueError as error:
        return dict(result, status="RESOURCE_UNCONFIRMED", reason=str(error))
    if usage["address_extent_bits"] > usage["capacity_bits"]:
        return dict(result, status="UB_REJECTED", reason="final UB address exceeds backend capacity", ub=usage)
    return dict(result, status="UB_RESOURCE_CHECKED", resource_checked=True, ub=usage,
                artifact={"path": str(artifact), "sha256": sha256(artifact)}, final_ir_sha256=sha256(final_ir),
                proof="full backend compilation and allocation check")


def compile_candidate(request, output, timeout=240):
    """Compile a request containing pinned compiler/IR/options/target identity.

    Caller supplies the CANN host compiler environment. Reusing this record is
    valid only for the same IR, target, options and exact returned binary. A
    bare factor is not a reusable certificate. Only fresh output dirs allowed.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    compiler = Path(request["compiler"]).resolve()
    ir = Path(request["ir"]).resolve()
    target = request["target"]
    identity = {"compiler_sha256": sha256(compiler), "ir_sha256": sha256(ir)}
    for key, value in identity.items():
        if request[key] != value:
            raise ValueError(f"{key} changed; regenerate the request")
    if set(TARGET.findall(ir.read_text())) != {target}:
        raise ValueError("input IR target mismatch")
    if not target.startswith("Ascend950"):
        raise ValueError("this first resource gate supports the pinned A5 backend only")
    if set(re.findall(r'parallel_mode\s*=\s*"([^"]+)"', ir.read_text())) != {"simd"}:
        raise ValueError("this first resource gate requires the SIMD compilation path")
    options = list(request["options"])
    for option in options:
        if not option.startswith("--") or option.startswith(("--target", "--save-temps", "--mlir-print", "--output")):
            raise ValueError(f"unsupported option in compilation request: {option}")
    copied_ir = output / "candidate.ttadapter"
    copied_ir.write_bytes(ir.read_bytes())
    if sha256(copied_ir) != identity["ir_sha256"]:
        raise ValueError("input IR changed while preparing compilation")
    command = [
        str(compiler),
        str(copied_ir), f"--target={target}", *options, "--mlir-disable-threading",
        "--mlir-print-ir-after=hivm-plan-memory", "--enable-print-memory-allocated-size",
        f"--save-temps={output / 'temps'}", "-o",
        str(output / "kernel")
    ]
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    environment = os.environ.copy()
    environment.update(ASCEND_RT_VISIBLE_DEVICES="", ASCEND_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment["PATH"]
    started = time.monotonic()
    stdout_path, stderr_path = output / "stdout.log", output / "stderr.log"
    timed_out = False
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment, stdin=subprocess.DEVNULL,
                                   start_new_session=True, close_fds=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
    result = assess(returncode, stdout_path.read_text(errors="replace"), stderr_path.read_text(errors="replace"),
                    output / "kernel.o", output / "temps/module.hivm.opt.mlir", target)
    if timed_out:
        result.update(status="COMPILE_TIMEOUT", resource_checked=False)
    result.update(identity=identity, candidate=request.get("candidate"), target=target,
                  compile_seconds=time.monotonic() - started, returncode=returncode, command=command,
                  stderr_sha256=sha256(stderr_path), cpu_backend_only=True, device_executed=False,
                  requested_mapping_verified=False)
    # This layer verifies resources, not the frontend factor mapping or guards.
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    result = compile_candidate(json.loads(args.request.read_text()), args.output, args.timeout)
    print(json.dumps({k: result.get(k) for k in ("candidate", "status", "resource_checked", "ub")}, indent=2))
    return 0 if result["resource_checked"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
