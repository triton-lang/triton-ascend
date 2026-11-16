# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""JIT/cache callbacks for the native CPU selector; no device execution."""
import json


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
    from .block_auto_artifacts import load_resources, load_analysis
    try:
        resources = [load_resources(candidate) for candidate in candidates]
        prepared = _native("prepare", contract, candidates=resources)["candidates"]
        if len(prepared) != len(resources):
            raise ValueError("resource checks did not return one state per candidate")
        inputs = [load_analysis(item, state) for item, state in zip(resources, prepared)]
        from .block_batching_worker import BatchCompileError, _attempt
        try:
            report = _attempt("select", dict(contract=contract, candidates=inputs), timeout=30).payload["report"]
        except BatchCompileError as error:
            if error.kind != "Timeout":
                raise
            raise ValueError("static selection exceeded its 30 second budget") from error
        return report.pop("factor"), report
    except (ValueError, KeyError, OSError, TypeError) as error:
        return 1, dict(model="jit-cv-vv-native-v1", selected_key="unmerged", selected_axis=-1, model_complete=False,
                       reason="BASELINE_FALLBACK_INCOMPLETE_MODEL", candidates={}, device_timing_used=False,
                       error=str(error))
