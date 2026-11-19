# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""JIT/cache callbacks for the native CPU selector; no device execution."""
import json
import subprocess


def _native(action, contract, **inputs):
    from triton._C.libtriton import ascend
    contract = dict(contract, argument_order=contract.get("argument_order", list(contract.get("arguments", {}))))
    try:
        return json.loads(
            ascend.select_block_candidates(json.dumps(dict(action=action, contract=contract, **inputs),
                                                      allow_nan=False)))
    except RuntimeError as error:
        raise ValueError(str(error)) from error


def candidate_plan(contract, *, probe, axis=-1):
    """Ask the frontend about the axes requested by the native CV planner."""
    requested = _native("plan", contract, axis=axis)["probes"]
    reports = []
    for item in requested:
        try:
            reports.append(dict(item, report=probe(item["axis"], item["factor"])))
        except ValueError:
            continue
    return _native("plan", contract, axis=axis, reports=reports)


def select_candidates(candidates, contract):
    from .block_auto_artifacts import load_evidence
    from .block_auto_models.vv_compilation import optimized_llvm
    try:
        items = []
        for candidate in candidates:
            item = {k: candidate[k] for k in ("factor", "axis", "status", "error") if k in candidate}
            items.append(item)
            if candidate.get("kernel") is not None:
                try:
                    item.update(load_evidence(candidate["kernel"]))
                except (ValueError, KeyError, OSError) as error:
                    item.update(status="RESOURCE_UNCONFIRMED", error=str(error))
        # Resource checks need only allocation IR and metadata, not the LLVM bodies.
        inputs = [{k: "" if k.startswith("llvm") else v for k, v in item.items()} for item in items]
        prepared = _native("prepare", contract, candidates=inputs)["candidates"]
        for item, state in zip(items, prepared):
            item.update(state)
            if state["needs_llvm"]:
                try:
                    item["optimized_llvm"] = optimized_llvm(item)
                except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
                    item.update(status="STATIC_ANALYSIS_UNSUPPORTED", error=str(error))
        inputs = [{
            k: v
            for k, v in item.items()
            if k not in ("metadata", "files", "final_ir", "meta_library") and not (k == "llvm" and item.get("vv"))
        }
                  for item in items]
        report = _native("rank", contract, candidates=inputs)
        return report.pop("factor"), report
    except (ValueError, KeyError, OSError, TypeError) as error:
        return 1, dict(model="jit-cv-vv-native-v1", selected_key="unmerged", selected_axis=-1, model_complete=False,
                       reason="BASELINE_FALLBACK_INCOMPLETE_MODEL", candidates={}, device_timing_used=False,
                       error=str(error))
