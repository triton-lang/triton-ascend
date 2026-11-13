"""Static SIMD/MIX UB checks, including conservative i1 allocation bounds."""
import re

from .cpu_resource_gate import inspect_final_ub
from .artifact_io import read


def spec(ir, key):
    values = {int(v) for v in re.findall(r'"' + key + r'",\s*(\d+)\s*:\s*i\d+', ir)}
    if len(values) != 1 or min(values) <= 0:
        raise ValueError("missing or ambiguous target specification: " + key)
    return values.pop()


def inspect_vv_ub(ir, target, metadata, proof, files):
    mode = metadata.get("parallel_mode")
    if metadata.get("mix_mode") != "aiv" or mode not in ("simd", "mix_simd_simt"):
        raise ValueError("only AIV SIMD and MIX resource proofs are supported")
    functions = [line for line in ir.splitlines() if re.match(r"\s*func\.func\s", line)]
    entries = [line for line in functions if "hacc.entry" in line]
    expected = "MIX" if mode == "mix_simd_simt" else "SIMD"
    if len(entries) != 1 or "hivm.vf_mode<" + expected + ">" not in entries[0]:
        raise ValueError("compile metadata and final IR vector modes differ")
    usage = inspect_final_ub(ir, target, allow_i1=True, alignment_bits=spec(ir, "UB_ALIGN_SIZE"))
    usage["backend_usable_bits"] = usage["capacity_bits"]
    if mode == "mix_simd_simt":
        if (len(entries) != 1 or "hivm.vf_mode<MIX>" not in entries[0]
                or not all("hivm.func_core_type<AIV>" in line for line in functions)):
            raise ValueError("MIX metadata does not describe a single AIV MIX entry")
        dynamic = metadata.get("shared_mem_dynamic_size")
        if type(dynamic) is not int or dynamic <= 0:
            raise ValueError("missing runtime dynamic UB size")
        limit = dynamic * 8
        full = usage["capacity_bits"]
        if not full - spec(ir, "MAXIMUM_D_CACHE_SIZE") <= limit <= full - spec(ir, "MINIMAL_D_CACHE_SIZE"):
            raise ValueError("MIX dynamic UB conflicts with D-cache reservation")
        if proof.get("metadata", {}).get("sha256") != files["metadata"]["sha256"]:
            raise ValueError("MIX proof does not bind runtime metadata")
        command = proof.get("command", [])
        explicit = [int(v.split("=", 1)[1]) for v in command if v.startswith("--simt-vf-dynamic-size=")]
        if len(explicit) > 1:
            raise ValueError("ambiguous compile-time dynamic UB size")
        if explicit:
            compiled = explicit[0] * 1024
        else:
            evidence = files.get("backend_defaults")
            if not evidence or evidence["sha256"] != proof.get("default_option_evidence_sha256"):
                raise ValueError("missing hash-bound compiler dynamic-UB default")
            from pathlib import Path
            text = Path(evidence["path"]).read_text()
            defaults = re.findall(r"--simt-vf-dynamic-size=<int>[^\n]*Default is (\d+)\.", text)
            if len(defaults) != 1:
                raise ValueError("unknown compiler dynamic-UB default")
            compiled = int(defaults[0]) * 1024
        if compiled != dynamic or proof.get("ub", {}).get("capacity_bits") != limit:
            raise ValueError("compile, resource proof and runtime UB capacities differ")
        usage.update(capacity_bits=limit, reserved_cache_bits=full - limit)
    if usage["address_extent_bits"] > usage["capacity_bits"]:
        if usage.get("conservative_upper_bound"):
            raise ValueError("i1 conservative UB upper bound exceeds capacity; actual overflow unconfirmed")
        raise ValueError("final UB address exceeds capacity")
    usage["evidence"] = "hash-bound backend allocation, aligned extents and effective runtime capacity"
    return usage
