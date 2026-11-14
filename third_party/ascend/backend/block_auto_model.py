"""CPU resource checks and CV/VV candidate ranking."""
import math
from pathlib import Path
import re
import subprocess
from .block_auto_models.native_model import reject
from .block_auto_models.cpu_resource_gate import TARGET, inspect_final_ub, inspect_vv_ub, sha256

_EXCLUDED = frozenset(
    ("UB_REJECTED", "COMPILE_REJECTED", "FRONTEND_REJECTED", "RUNTIME_REJECTED", "NO_MERGE_OPPORTUNITY"))

VERSION = "jit-cv-vv-native-v1"


def _contract_for_ttir(contract, ttir):
    # prepare_launch owns host validation; only remap the normalized TTIR ABI.
    signature = next(line for line in ttir.splitlines() if "tt.func public @" in line)
    names = [n for n in re.findall(r"%([\w.$-]+):", signature) if not n.startswith("__hacc_original_grid_")]
    arguments = contract["arguments"]
    order = contract.get("argument_order", list(arguments))
    reject(len(names) != len(order) or set(order) != set(arguments), "launch/TTIR ABI mismatch")
    reject(any(v["kind"] not in ("tensor", "scalar") for v in arguments.values()), "unsupported host argument")
    return dict(contract, arguments={n: arguments[s] for n, s in zip(names, order)}, argument_order=names)


def _resource_check(item):
    metadata = item["metadata"]
    target = metadata.get("target", {}).get("arch")
    reject(target != "Ascend950PR_9579", "cost model target is outside the validated A5 profile")
    targets = set(TARGET.findall(item["final_ir"]))
    reject(len(targets) != 1 or not targets <= {"Ascend950", target}, "unexpected allocation target")
    target = targets.pop()
    usage = (inspect_vv_ub(item["final_ir"], target, metadata,
                           metadata.get("block_auto_evidence", {}).get("command", []), item["files"])
             if metadata.get("mix_mode") == "aiv" else inspect_final_ub(item["final_ir"], target))
    reject(usage["address_extent_bits"] > usage["capacity_bits"], "final allocated UB address exceeds capacity")
    item.update(resource_checked=True, ub_usage=usage)


def _optimized_llvm(item):
    from .block_auto_models.vv_compilation import optimized_llvm
    return optimized_llvm(item)


def _rank_vv(candidates, contract):
    from .block_auto_models.native_model import analyze
    from .block_auto_models.native_model import rejection
    rows = {}
    for item in candidates:
        row = dict(factor=item["factor"], axis=item["axis"], resource_checked=item.get("resource_checked", False),
                   status=item.get("status", "STATIC_ANALYSIS_UNSUPPORTED"), error=item.get("error"))
        rows[item["key"]] = row
        if not row["resource_checked"]:
            continue
        try:
            floor = contract.get("runtime_core_floor", contract.get("core_floor"))
            if floor is None:
                counts = re.findall(r'"VECTOR_CORE_COUNT",\s*(\d+)\s*:\s*i32', item["final_ir"])
                reject(len(counts) != 1, "compiled vector core floor is unavailable")
                floor = int(counts[0])
            elif isinstance(floor, str) and floor.isdecimal():
                floor = int(floor)
            refusal = rejection(contract["grid"], item["axis"], item["factor"], floor, item["metadata"])
            if refusal:
                row.update(status="RUNTIME_REJECTED", error=refusal)
                continue
            optimized = _optimized_llvm(item)
            lib = Path(item["files"]["compiler"]["path"]).parent.parent.parent / "lib/libdevice.10.bc"
            library = dict(path=str(lib), sha256=sha256(lib)) if "@pow_fp32(" in item["llvm"] else None
            row.update(
                analyze({"aiv": optimized}, item["ttir"], item["final_ir"], item["metadata"], item["launch_contract"],
                        item["axis"], item["factor"], library=library, mixed=item["files"], raw=item["llvm"]))
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
            row.update(status="STATIC_ANALYSIS_UNSUPPORTED", error=f"{type(error).__name__}: {error}")
    scored = {
        k: r
        for k, r in rows.items()
        if r["status"] == "STATIC_COST_ESTIMATED" and r["resource_checked"] and type(r.get("score")) in (
            int, float) and math.isfinite(r["score"]) and r["score"] >= 0
    }
    complete = bool(rows) and all(k in scored or r["status"] in _EXCLUDED for k, r in rows.items())
    best = min(scored, key=lambda k: (scored[k]["score"], scored[k]["factor"], k)) if complete and scored else None
    return dict(selected_key=best or "unmerged", model_complete=bool(complete and best), candidates=rows,
                reason="LOWEST_STATIC_SCORE" if best else "BASELINE_FALLBACK_INCOMPLETE_MODEL")


def select_candidates(candidates, contract):
    from .block_auto_artifacts import load_evidence
    report = dict(model=VERSION, selected_key="unmerged", selected_axis=-1, model_complete=False,
                  reason="BASELINE_FALLBACK_INCOMPLETE_MODEL", candidates={}, device_timing_used=False)
    try:
        prepared = {}
        for candidate in candidates:
            factor, axis = candidate["factor"], candidate["axis"]
            excluded = candidate.get("status") in _EXCLUDED and candidate.get("kernel") is None
            key = "unmerged" if factor == 1 else f"a{axis}_f{factor}"
            item = prepared[key] = dict(candidate, key=key, resource_checked=False)
            if not excluded:
                try:
                    item.update(load_evidence(candidate["kernel"]))
                    _resource_check(item)
                    item["launch_contract"] = _contract_for_ttir(contract, item["ttir"])
                except (ValueError, KeyError, OSError, StopIteration) as error:
                    item.update(status="RESOURCE_UNCONFIRMED", error=f"{type(error).__name__}: {error}")
        first = prepared["unmerged"]
        if not first["resource_checked"]:
            report.update(
                reason="BASELINE_RESOURCE_UNCONFIRMED", candidates={
                    k: {n: v[n]
                        for n in ("factor", "axis", "status", "error", "resource_checked")
                        if n in v}
                    for k, v in prepared.items()
                })
            return 1, report
        if first["metadata"].get("mix_mode") == "aiv":
            decision = _rank_vv(list(prepared.values()), contract)
        else:
            from .block_auto_cv import rank_cv
            decision = rank_cv(list(prepared.values()), contract)
        report.update(decision)
        selected = prepared[report["selected_key"]]
        report.update(selected_key=selected["key"], selected_axis=selected["axis"])
        return selected["factor"], report
    except (ValueError, KeyError, OSError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return 1, report
