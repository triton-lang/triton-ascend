"""In-process CPU selection of compiled Ascend block-batching candidates.

No device timings or tensor values enter this module. Compiler artifacts are
loaded from the Triton cache, bound to their binary, and checked before the
published CV/VV models run. Unsupported coverage retains the original kernel.
"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from .block_auto_models.backend_cost_model import Unsupported, read_cost_table
from .block_auto_models.cpu_resource_gate import TARGET, inspect_final_ub, sha256

_MODELS = Path(__file__).with_name("block_auto_models")
_EXCLUDED = frozenset(
    ("UB_REJECTED", "COMPILE_REJECTED", "FRONTEND_REJECTED", "RUNTIME_REJECTED", "NO_MERGE_OPPORTUNITY"))


def _implementation_identity():
    digest = hashlib.sha256()
    paths = [Path(__file__), *_MODELS.rglob("*")]
    cv = Path(__file__).with_name("block_auto_cv.py")
    if cv.is_file():
        paths.append(cv)
    for path in sorted(p for p in paths if p.is_file() and "__pycache__" not in p.parts):
        digest.update(path.relative_to(Path(__file__).parent).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


VERSION = "jit-cv-vv-model-v1-" + _implementation_identity()


@lru_cache(maxsize=1)
def _cost_table():
    data = _MODELS / "data"
    provenance = json.loads((data / "provenance.json").read_text())
    path = data / "AVECostModelInfo.def"
    if provenance.get("schema_version") != 1 or sha256(path) != provenance.get("sha256"):
        raise Unsupported("packaged compiler cost table identity mismatch")
    if provenance.get("target_arch") != "Ascend950PR_9579" or not provenance.get("source_revision"):
        raise Unsupported("packaged compiler cost table has no supported target/version")
    table = read_cost_table(path.read_text())
    if not table:
        raise Unsupported("packaged compiler cost table is empty")
    return table


def _contract_for_ttir(contract, ttir):
    """Bind actual host scalars to the candidate's post-constexpr SSA ABI."""
    grid = contract.get("grid")
    if not isinstance(grid, (tuple, list)) or len(grid) != 3 or any(type(n) is not int or n <= 0 for n in grid):
        raise Unsupported("launch grid must have three positive integer extents")
    signature = next((line for line in ttir.splitlines() if "tt.func public @" in line), None)
    if signature is None:
        raise Unsupported("candidate public TTIR signature is missing")
    names = re.findall(r"%([\w.$-]+):", signature)
    names = [name for name in names if not name.startswith("__hacc_original_grid_")]
    arguments = contract.get("arguments", {})
    order = contract.get("argument_order", list(arguments))
    if not isinstance(arguments, dict) or len(order) != len(set(order)) or set(order) != set(arguments):
        raise Unsupported("launch argument order is incomplete")
    if len(names) != len(order):
        raise Unsupported("launch arguments differ from the post-constexpr TTIR ABI")
    bound = {}
    for name, source in zip(names, order):
        item = arguments[source]
        if not isinstance(item, dict) or item.get("kind") not in ("tensor", "scalar"):
            raise Unsupported("unsupported host launch argument: " + str(source))
        if item["kind"] == "scalar":
            value = item.get("value")
            if type(value) not in (int, bool, float) or isinstance(value, float) and not math.isfinite(value):
                raise Unsupported("unknown/nonfinite host scalar: " + str(source))
        bound[name] = dict(item)
    return dict(contract, grid=list(grid), arguments=bound, argument_order=names)


def _candidate_key(factor, axis):
    return "unmerged" if factor == 1 else f"a{axis}_f{factor}"


def _allocation_target(item, requested):
    """Validate the one supported frontend-to-backend target alias.

    The normal Ascend compiler canonicalizes this device to --target=Ascend950.
    Keep both IR texts unchanged: the hash-bound frontend, captured command and
    final IR must prove that exact mapping before allocation inspection uses it.
    """
    observed = set(TARGET.findall(item["final_ir"]))
    if observed == {requested}:
        return requested
    if requested != "Ascend950PR_9579" or observed != {"Ascend950"}:
        raise Unsupported("final IR target does not match the supported device target")
    evidence = item["metadata"].get("block_auto_evidence", {})
    command = evidence.get("command", [])
    targets = [
        argument.partition("=")[2]
        for argument in command
        if isinstance(argument, str) and argument.startswith("--target=")
    ]
    if (evidence.get("target") != requested or targets != ["Ascend950"] or "--target" in command
            or set(TARGET.findall(item.get("ttadapter", ""))) != {requested}):
        raise Unsupported("canonical Ascend950 target lacks matching frontend/compile evidence")
    item["target_binding"] = dict(requested=requested, compiler="Ascend950",
                                  evidence="hash-bound TTAdapter and captured compiler target option")
    return "Ascend950"


def _resource_check(item):
    metadata = item["metadata"]
    target = metadata.get("target", {}).get("arch")
    if target != "Ascend950PR_9579":
        raise Unsupported("cost model target is outside the validated A5 profile")
    allocation_target = _allocation_target(item, target)
    if metadata.get("mix_mode") == "aiv":
        from .block_auto_models.vv_resource_gate import inspect_vv_ub
        evidence = metadata.get("block_auto_evidence", {})
        proof = dict(command=evidence.get("command", []), metadata=item["files"]["metadata"])
        if metadata.get("parallel_mode") == "mix_simd_simt":
            dynamic = metadata.get("shared_mem_dynamic_size")
            if type(dynamic) is not int or dynamic <= 0:
                raise Unsupported("MIX runtime UB capacity is unavailable")
            proof["ub"] = dict(capacity_bits=dynamic * 8)
            defaults = item["files"].get("backend_defaults")
            if defaults:
                proof["default_option_evidence_sha256"] = defaults["sha256"]
        usage = inspect_vv_ub(item["final_ir"], allocation_target, metadata, proof, item["files"])
    else:
        usage = inspect_final_ub(item["final_ir"], allocation_target)
    if usage["address_extent_bits"] > usage["capacity_bits"]:
        raise Unsupported("final allocated UB address exceeds the compiler capacity")
    item.update(resource_checked=True, ub_usage=usage)


def _optimizer_tools():
    from .utils import _get_bisheng_path
    optimizer = Path(_get_bisheng_path()).resolve()
    link = optimizer.with_name("llvm-link")
    if not link.is_file():
        found = shutil.which("llvm-link")
        if not found:
            raise Unsupported("installed LLVM decoder is unavailable")
        link = Path(found).resolve()
    if not optimizer.is_file():
        raise Unsupported("installed A5 optimizer is unavailable")
    return optimizer, link


def _optimized_evidence(item):
    """Prepare and cache the same executable-bound LLVM used offline.

    These are native compiler invocations owned by JIT compilation, with device
    visibility disabled. No Python script, manifest, or timing run is involved.
    """
    from triton.runtime.cache import get_cache_manager
    from .block_auto_models.vv_optimized_ir import prepare, verified_vectors
    from .block_auto_models.vv_mixed_model import meta_path

    files = dict(item["files"])
    optimizer, link = _optimizer_tools()
    meta = meta_path(item["final_ir"])
    libdevice = Path(files["compiler"]["path"]).parent.parent.parent / "lib/libdevice.10.bc"
    identity = {
        "model": VERSION,
        "binary": item["binary_sha256"],
        "llvm": files["llvm"]["sha256"],
        "compiler": item["compiler_sha256"],
        "optimizer": sha256(optimizer),
        "llvm_link": sha256(link),
        "meta": sha256(meta),
        "libdevice": sha256(libdevice),
    }
    cache = get_cache_manager(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())
    cached = cache.get_file("block-auto-optimized.json")
    if cached:
        files.update(json.loads(Path(cached).read_text()))
        # Validate cached optimized artifacts even when the binary is unchanged.
        verified_vectors(item["llvm"], files)
        return files
    directory = getattr(cache, "cache_dir", None)
    if not directory:
        raise Unsupported("optimized evidence requires a local compiler cache")
    output = Path(directory) / ("optimized-" + uuid.uuid4().hex)
    candidate = dict(files=files)
    prepare(candidate, output, optimizer, link)
    verified_vectors(item["llvm"], files)
    cache.put(json.dumps(files, sort_keys=True), "block-auto-optimized.json", binary=False)
    return files


@lru_cache(maxsize=1)
def _instruction_model():
    from .block_auto_models.vv_skill_model import DATA, InstructionModel
    provenance = json.loads((DATA / "provenance.json").read_text())
    for name, expected in provenance["sha256"].items():
        if sha256(DATA / name) != expected:
            raise Unsupported("packaged VV instruction asset changed: " + name)
    return InstructionModel()


def _rank_vv(candidates, contract, table):
    from .block_auto_models.vv_skill_model import analyze_candidate
    from .block_auto_models.vv_contention_model import choose
    from .block_auto_models.vv_mixed_model import attach_mixed_evidence
    from .block_auto_models.v38_runtime_policy import rejection

    rows = {}
    for item in candidates:
        key = item["key"]
        row = dict(factor=item["factor"], axis=item["axis"], resource_checked=item.get("resource_checked", False),
                   status=item.get("status", "STATIC_ANALYSIS_UNSUPPORTED"), error=item.get("error"))
        if row["resource_checked"]:
            try:
                floor = contract.get("runtime_core_floor", contract.get("core_floor"))
                if floor is None:
                    counts = re.findall(r'"VECTOR_CORE_COUNT",\s*(\d+)\s*:\s*i32', item["final_ir"])
                    if len(counts) != 1:
                        raise Unsupported("compiled vector core floor is unavailable")
                    floor = int(counts[0])
                elif isinstance(floor, str) and floor.isdecimal():
                    floor = int(floor)
                refusal = rejection(contract["grid"], item["axis"], item["factor"], floor, item["metadata"])
                if refusal:
                    row.update(status="RUNTIME_REJECTED", error=refusal, runtime_core_floor=floor)
                    rows[key] = row
                    continue
                files = _optimized_evidence(item)
                if item["metadata"].get("parallel_mode") == "mix_simd_simt":
                    attach_mixed_evidence(dict(files=files))
                lib = Path(files["compiler"]["path"]).parent.parent.parent / "lib/libdevice.10.bc"
                library = dict(path=str(lib), sha256=sha256(lib)) if "@pow_fp32(" in item["llvm"] else None
                row.update(
                    analyze_candidate(item["llvm"], item["ttir"], item["final_ir"], item["metadata"],
                                      _contract_for_ttir(contract, item["ttir"]), item["axis"], item["factor"], table,
                                      model=_instruction_model(), bandwidth="ini", diagnostic=False,
                                      library_evidence=library, mixed_evidence=files, optimized_evidence=files,
                                      lowered_entry=True))
            except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
                row.update(status="STATIC_ANALYSIS_UNSUPPORTED", error=f"{type(error).__name__}: {error}")
        rows[key] = row
    scored = {
        key: row
        for key, row in rows.items()
        if row.get("status") == "STATIC_COST_ESTIMATED" and row.get("resource_checked") is True
        and type(row.get("score")) in (int, float) and math.isfinite(row["score"]) and row["score"] >= 0
    }
    complete = bool(rows) and all(key in scored or row["status"] in _EXCLUDED for key, row in rows.items())
    best = choose(scored) if complete else None
    return dict(selected_key=best or "unmerged", model_complete=bool(complete and best),
                reason="LOWEST_STATIC_SCORE" if best else "BASELINE_FALLBACK_INCOMPLETE_MODEL", candidates=rows)


def select_candidates(candidates, contract):
    """Return (factor, report), preserving F1 for unconfirmed model coverage."""
    report = dict(model=VERSION, selected_key="unmerged", selected_axis=-1, model_complete=False,
                  reason="BASELINE_FALLBACK_INCOMPLETE_MODEL", candidates={}, device_timing_used=False)
    if not isinstance(candidates, (tuple, list)) or not candidates or len(candidates) > 3:
        report["reason"] = "INVALID_CANDIDATE_FAMILY"
        return 1, report
    baseline = [item for item in candidates if item.get("factor") == 1]
    if len(baseline) != 1 or baseline[0].get("axis") != -1:
        raise ValueError("automatic selection requires one explicit F1 baseline")
    try:
        table = _cost_table()
        from .block_auto_artifacts import load_evidence
        prepared = []
        keys = set()
        for candidate in candidates:
            factor, axis = candidate["factor"], candidate["axis"]
            if type(factor) is not int or factor not in (1, 2, 4) or type(axis) is not int or axis not in (-1, 0, 1, 2):
                raise Unsupported("invalid candidate factor or axis")
            excluded = candidate.get("status") in _EXCLUDED and candidate.get("kernel") is None
            if factor > 1 and axis < 0 and not excluded:
                raise Unsupported("merged candidate has no resolved compiler axis")
            key = _candidate_key(factor, axis)
            if key in keys:
                raise Unsupported("duplicate compiled candidate")
            keys.add(key)
            item = dict(candidate, key=key, resource_checked=False)
            if excluded:
                prepared.append(item)
                continue
            try:
                item.update(load_evidence(candidate["kernel"]))
                _resource_check(item)
                _contract_for_ttir(contract, item["ttir"])
            except (ValueError, KeyError, OSError) as error:
                item.update(status="RESOURCE_UNCONFIRMED", error=f"{type(error).__name__}: {error}")
            prepared.append(item)
        first = next(item for item in prepared if item["factor"] == 1)
        report["artifact_target_bindings"] = {
            item["key"]: item["target_binding"]
            for item in prepared
            if "target_binding" in item
        }
        if not first.get("resource_checked"):
            report.update(
                reason="BASELINE_RESOURCE_UNCONFIRMED", candidates={
                    item["key"]: {
                        name: item[name]
                        for name in ("factor", "axis", "status", "error", "resource_checked")
                        if name in item
                    }
                    for item in prepared
                })
            return 1, report
        if first["metadata"].get("mix_mode") == "aiv":
            decision = _rank_vv(prepared, contract, table)
        else:
            from .block_auto_cv import rank_cv
            # All candidates have the same source ABI, but their hidden grid
            # parameters may differ. The CV adapter binds each candidate TTIR.
            decision = rank_cv(prepared, contract, table, VERSION)
        report.update(decision)
        selected = next((item for item in prepared if item["key"] == report["selected_key"]), first)
        if selected["factor"] != 1 and not selected.get("resource_checked"):
            raise Unsupported("selected candidate has no confirmed resource evidence")
        report.update(selected_key=selected["key"], selected_axis=selected["axis"])
        return selected["factor"], report
    except (ValueError, KeyError, OSError) as error:
        report.update(reason="BASELINE_FALLBACK_INCOMPLETE_MODEL", error=f"{type(error).__name__}: {error}")
        return 1, report
