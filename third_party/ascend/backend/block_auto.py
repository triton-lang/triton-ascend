# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""JIT/cache callbacks for the native CPU selector; no device execution."""
import json
import math


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


def _probe(backend, frozen, options, env_vars, axis, factor):
    from .block_batching_worker import BatchCompileError, BatchDecline, selection_probe
    values = dict(options, block_batching_factor=factor, block_batching_axis=axis, block_auto_collect=False,
                  enable_graph_optimize=False, debug=False)
    try:
        report = selection_probe(backend, frozen, values, env_vars)
    except ValueError as error:
        # candidate_plan treats ValueError as an unsupported axis/factor. Only a
        # modeled worker decline is such an outcome.
        raise BatchCompileError(str(error), stage="selection_probe") from error
    if isinstance(report, BatchDecline):
        raise ValueError(report.reason)
    return report


def compile_auto_bundle(backend, src, options, compile_variant):
    from . import block_batching as batching
    from . import block_batching_worker as compilation
    from .block_batching_worker import BatchDecline, _baseline_environment
    # Without a bound JIT launch there is no shape/grid contract to score.
    common = {key: value for key, value in options.items() if key not in ("block_auto_contract", "block_batching_grid")}
    baseline_options = dict(common, block_batching_factor=1, block_batching_axis=-1)
    if not options.get("block_auto_contract"):
        return compile_variant(baseline_options)
    contract = json.loads(options["block_auto_contract"])
    normalized = backend.parse_options(options)
    common["block_auto_collect"] = baseline_options["block_auto_collect"] = True
    refusal = compilation._freeze_refusal(normalized)
    frozen = src if refusal else compilation._freeze_source(backend, src, baseline_options,
                                                            normalized.block_batching_backend_revision)
    if isinstance(frozen, BatchDecline):
        refusal, frozen = frozen.reason, src
    baseline = compile_variant(baseline_options, frozen)
    axis = options.get("block_batching_axis", -1)
    cv = getattr(baseline.metadata, "mix_mode", None) in ("aic", "mix")
    if not refusal:
        refusal = compilation._candidate_refusal(backend, src, normalized)
        if not refusal:
            positions = compilation._runtime_positions(src)
            if isinstance(positions, BatchDecline):
                refusal = positions.reason
        if not refusal:
            env_vars = _baseline_environment(frozen, backend, baseline_options, baseline)
            if env_vars is None:
                refusal = "BASELINE_ENVIRONMENT_NOT_PROVEN"
    if refusal:
        plan = dict(candidates=[], reason=refusal)
    elif cv:
        plan = candidate_plan(contract, axis=axis, probe=lambda a, f: _probe(backend, frozen, common, env_vars, a, f))
    elif math.prod(contract["grid"]) < (int(contract["core_floor"])
                                        if contract.get("core_floor") is not None else batching._core_floor()):
        plan = dict(candidates=[], reason="GRID_BELOW_CORE_FLOOR")
    else:
        extents = contract["grid"] if axis == -1 else [contract["grid"][axis]]
        plan = dict(candidates=[dict(axis=axis, factor=f) for f in (2, 4) if max(extents) >= f],
                    reason="VV_F2_F4_AUTOMATIC_AXIS")

    def variant(values, frozen_src=None):
        return baseline if values["block_batching_factor"] == 1 else compile_variant(values, frozen_src)

    candidates = [dict(factor=1, axis=-1, kernel=baseline)]
    bundles = {}
    for spec in plan["candidates"]:
        factor, requested_axis = spec["factor"], spec["axis"]
        values = dict(common, block_batching_factor=factor, block_batching_axis=requested_axis,
                      block_batching_grid=tuple(contract["grid"]))
        bundle = compilation.compile_bundle(backend, frozen, values, variant)
        report = bundle.block_batching
        if bundle.candidate is None:
            candidates.append(
                dict(factor=factor, axis=requested_axis, status="COMPILE_REJECTED",
                     error=report.get("reason", "CANDIDATE_UNAVAILABLE")))
        else:
            selected_axis = report["selected_axis"]
            candidates.append(dict(factor=factor, axis=selected_axis, kernel=bundle.candidate))
            bundles[f"a{selected_axis}_f{factor}"] = bundle
    if bundles:
        factor, selection = select_candidates(candidates, contract)
    else:
        factor, selection = 1, dict(selected_key="unmerged", selected_axis=-1,
                                    reason=plan.get("reason") if not plan["candidates"] else "NO_COMPILED_CANDIDATE",
                                    model_complete=False, device_timing_used=False)
    if factor == 1:
        report = dict(compile_status="DISABLED", owner="none", requested_factor=1, applied_factor=1,
                      reason=selection["reason"])
        result = batching.BatchKernel(baseline, None, batching.BatchRequest(), report, [])
    else:
        result = bundles[selection["selected_key"]]
    result.block_auto_selection = dict(selection, selected_factor=factor, candidate_plan=plan)
    return result
