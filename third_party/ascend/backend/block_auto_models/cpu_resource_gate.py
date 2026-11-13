"""Static allocation inspection shared by in-process CV/VV selection."""
import hashlib
import math
from pathlib import Path
import re

PASS_MARKER = "IR Dump After PlanMemory (hivm-plan-memory)"
OVERFLOW = re.compile(r"ub overflow, requires (\d+) bits while (\d+) bits available", re.I)
TARGET = re.compile(r'hacc\.target\s*=\s*#hacc\.target<"([^"\n]+)">')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
