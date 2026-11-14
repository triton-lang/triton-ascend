"""Static allocation inspection shared by in-process CV/VV selection."""
from .native_model import reject
import math
import re
from .. import block_auto_artifacts

sha256 = block_auto_artifacts.file_digest

TARGET = re.compile(r'hacc\.target\s*=\s*#hacc\.target<"([^"\n]+)">')


def inspect_final_ub(ir, target, *, allow_i1=False, alignment_bits=8):
    """Read final constant UB addresses, rather than guessing from input size.

    The backend is the resource authority. This additionally checks the final
    allocated address extent for the narrow static-buffer syntax we understand.
    Unknown/dynamic forms are reported, never interpreted as zero allocation.
    """
    reject(
        type(alignment_bits) is not int or alignment_bits < 8 or alignment_bits % 8, 'invalid UB alignment', ValueError)
    reject(set(TARGET.findall(ir)) != {target}, 'final IR target does not match requested target', ValueError)
    limit = spec(ir, "UB_SIZE", "missing or ambiguous backend UB capacity")
    constants = {}
    end_bits, conservative_bound = None, False
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
        reject('memref.alloc' in line, 'unresolved UB allocation remains in final IR', ValueError)
        if "hivm.hir.pointer_cast" not in line:
            continue
        match = re.search(
            r"hivm\.hir\.pointer_cast\((%[\w.$-]+)\)\s*:\s*"
            r"memref<((?:\d+x)*)(bf16|[if]\d+),\s*#hivm\.address_space<ub>>", line)
        reject(not match or match.group(1) not in constants or function is None,
               'UB pointer has an unproved offset, shape or layout', ValueError)
        offset = constants[match.group(1)]
        dimensions = [int(v) for v in match.group(2).split("x") if v]
        element_bits = 16 if match.group(3) == "bf16" else int(match.group(3)[1:])
        conservative = element_bits == 1 and allow_i1
        reject(offset < 0 or (not conservative and (element_bits < 8 or element_bits % 8)),
               'unsupported UB element width or negative address', ValueError)
        reject(any((v <= 0 for v in dimensions)), 'nonpositive UB extent', ValueError)
        bits = math.prod(dimensions) * (8 if conservative else element_bits)
        bits = (bits + alignment_bits - 1) // alignment_bits * alignment_bits
        end_bits = max(end_bits or 0, offset * 8 + bits)
        conservative_bound |= conservative
    if end_bits is None:
        # A pure Cube module can legitimately use no UB. Require every
        # definition/declaration to be explicitly AIC and no UB-typed value;
        # absence of an allocation alone is not proof for AIV/mixed modules.
        functions = [line for line in ir.splitlines() if re.match(r"\s*func\.func\s", line)]
        pure_cube = functions and all(
            re.search(r"hivm\.func_core_type\s*=\s*#hivm\.func_core_type<AIC>", line) for line in functions)
        reject(not pure_cube or re.search(r"address_space\s*<\s*ub\s*>", ir),
               "no verifiable static UB allocation in final IR", ValueError)
        end_bits = 0
    return dict(capacity_bits=limit, address_extent_bits=end_bits, conservative_upper_bound=conservative_bound)


def spec(ir, key, error=None):
    values = {int(v) for v in re.findall(r'"' + key + r'",\s*(\d+)\s*:\s*i\d+', ir)}
    if len(values) != 1 or min(values) <= 0:
        raise ValueError(error or "missing or ambiguous target specification: " + key)
    return values.pop()


def inspect_vv_ub(ir, target, metadata, command, files):
    mode = metadata.get("parallel_mode")
    reject(
        metadata.get('mix_mode') != 'aiv' or mode not in ('simd', 'mix_simd_simt'),
        'only AIV SIMD and MIX resource proofs are supported', ValueError)
    functions = [line for line in ir.splitlines() if re.match(r"\s*func\.func\s", line)]
    entries = [line for line in functions if "hacc.entry" in line]
    expected = "MIX" if mode == "mix_simd_simt" else "SIMD"
    reject(
        len(entries) != 1 or 'hivm.vf_mode<' + expected + '>' not in entries[0],
        'compile metadata and final IR vector modes differ', ValueError)
    usage = inspect_final_ub(ir, target, allow_i1=True, alignment_bits=spec(ir, "UB_ALIGN_SIZE"))
    usage["backend_usable_bits"] = usage["capacity_bits"]
    if mode == "mix_simd_simt":
        reject(
            len(entries) != 1 or 'hivm.vf_mode<MIX>' not in entries[0] or (not all(
                ('hivm.func_core_type<AIV>' in line for line in functions))),
            'MIX metadata does not describe a single AIV MIX entry', ValueError)
        dynamic = metadata.get("shared_mem_dynamic_size")
        reject(type(dynamic) is not int or dynamic <= 0, 'missing runtime dynamic UB size', ValueError)
        limit = dynamic * 8
        full = usage["capacity_bits"]
        reject(not full - spec(ir, 'MAXIMUM_D_CACHE_SIZE') <= limit <= full - spec(ir, 'MINIMAL_D_CACHE_SIZE'),
               'MIX dynamic UB conflicts with D-cache reservation', ValueError)
        explicit = [int(v.split("=", 1)[1]) for v in command if v.startswith("--simt-vf-dynamic-size=")]
        reject(len(explicit) > 1, 'ambiguous compile-time dynamic UB size', ValueError)
        if explicit:
            compiled = explicit[0] * 1024
        else:
            evidence = files.get("backend_defaults")
            reject(not evidence, 'missing hash-bound compiler dynamic-UB default', ValueError)
            from pathlib import Path
            reject(sha256(evidence['path']) != evidence['sha256'], 'compiler dynamic-UB default changed', ValueError)
            text = Path(evidence["path"]).read_text()
            defaults = re.findall(r"--simt-vf-dynamic-size=<int>[^\n]*Default is (\d+)\.", text)
            reject(len(defaults) != 1, 'unknown compiler dynamic-UB default', ValueError)
            compiled = int(defaults[0]) * 1024
        reject(compiled != dynamic, 'compile, resource proof and runtime UB capacities differ', ValueError)
        usage.update(capacity_bits=limit, reserved_cache_bits=full - limit)
    if usage["address_extent_bits"] > usage["capacity_bits"]:
        reject(usage.get('conservative_upper_bound'),
               'i1 conservative UB upper bound exceeds capacity; actual overflow unconfirmed', ValueError)
        raise ValueError("final UB address exceeds capacity")
    usage["evidence"] = "hash-bound backend allocation, aligned extents and effective runtime capacity"
    return usage
