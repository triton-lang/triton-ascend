# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Connect native CPU selection to the existing guarded batching compiler."""
import json
import math
import os


def _argument(value):
    if value is None or type(value) in (bool, int) or type(value) is float and math.isfinite(value):
        return dict(kind="scalar", value=value)
    if hasattr(value, "dtype") and hasattr(value, "shape") and callable(getattr(value, "stride", None)):
        return dict(kind="tensor", dtype=str(value.dtype), shape=list(value.shape), stride=list(value.stride()),
                    storage_offset=value.storage_offset())
    return dict(kind="unsupported")


def prepare_launch(fn, args, kwargs, grid, warmup, device, backend):
    if "block_batching_factor" in fn.arg_names or kwargs.get("block_batching_factor", 1) != "auto":
        return kwargs, grid
    from .block_batching import _capture_state

    def baseline():
        result = dict(kwargs, block_batching_factor=1)
        result.pop("block_auto_contract", None)
        return result, grid

    if (warmup or not str(backend.target.arch).startswith(("Ascend950", "Ascend910_95"))
            or _capture_state() is not False):
        return baseline()
    bound, specialization, _ = fn.device_caches[device][4](*args, **kwargs)
    if callable(grid):
        grid = grid(bound)
    if not isinstance(grid, (tuple, list)) or not 1 <= len(grid) <= 3 or any(
            type(n) is not int or not 0 < n <= 2147483647 for n in grid):
        return baseline()
    extents = tuple(grid) + (1, ) * (3 - len(grid))
    arguments = {
        name: _argument(value)
        for (name, value), spec in zip(bound.items(), specialization)
        if spec[0] != "constexpr"
    }
    contract = dict(grid=extents, arguments=arguments, argument_order=list(arguments),
                    core_floor=os.getenv("TA_BLOCK_BATCHING_CORE_FLOOR"), device_limit=os.getenv("NPU_DEVICE_LIMIT"))
    return dict(kwargs, block_batching_grid=extents,
                block_auto_contract=json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)), grid


def _probe(backend, frozen, options, axis, factor):
    from triton._C.libtriton import ir
    from .compiler import make_ttir
    from .block_batching import BatchingRejected
    normalized = backend.parse_options(
        dict(options, block_batching_factor=factor, block_batching_axis=axis, block_auto_collect=False,
             enable_graph_optimize=False, debug=False))
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = frozen.make_ir(backend.target, normalized, backend.get_codegen_implementation(normalized),
                            backend.get_module_map(), context)
    metadata = dict(vars(normalized))
    try:
        make_ttir(module, metadata, normalized)
    except BatchingRejected as error:
        raise ValueError(str(error)) from error
    return metadata["block_batching"]


def compile_auto_bundle(backend, src, options, compile_variant):
    from . import block_batching as batching
    from .block_auto import candidate_plan, select_candidates
    # Without a bound JIT launch there is no shape/grid contract to score.
    common = {key: value for key, value in options.items() if key not in ("block_auto_contract", "block_batching_grid")}
    baseline_options = dict(common, block_batching_factor=1, block_batching_axis=-1)
    if not options.get("block_auto_contract"):
        return compile_variant(baseline_options)
    contract = json.loads(options["block_auto_contract"])
    normalized = backend.parse_options(options)
    common["block_auto_collect"] = baseline_options["block_auto_collect"] = True
    refusal = None
    try:
        frozen = batching._freeze_source(backend, src, baseline_options, normalized.block_batching_backend_revision)
    except batching.BatchingRejected as error:
        frozen, refusal = src, str(error)
    baseline = compile_variant(baseline_options, frozen)
    axis = options.get("block_batching_axis", -1)
    cv = getattr(baseline.metadata, "mix_mode", None) in ("aic", "mix")
    if refusal is not None:
        plan = dict(candidates=[], reason=refusal)
    elif cv:
        plan = candidate_plan(contract, axis=axis, probe=lambda a, f: _probe(backend, frozen, common, a, f))
    else:
        plan = dict(candidates=[dict(axis=axis, factor=f) for f in (2, 4)], reason="VV_F2_F4_AUTOMATIC_AXIS")

    def variant(values, frozen_src=None):
        return baseline if values["block_batching_factor"] == 1 else compile_variant(values, frozen_src)

    candidates = [dict(factor=1, axis=-1, kernel=baseline)]
    bundles = {}
    for spec in plan["candidates"]:
        factor, requested_axis = spec["factor"], spec["axis"]
        values = dict(common, block_batching_factor=factor, block_batching_axis=requested_axis,
                      block_batching_grid=tuple(contract["grid"]))
        bundle = batching.compile_bundle(backend, frozen, values, variant)
        report = bundle.block_batching
        if bundle.candidate is None:
            candidates.append(
                dict(factor=factor, axis=requested_axis, status="COMPILE_REJECTED",
                     error=report.get("reason", "CANDIDATE_UNAVAILABLE")))
        else:
            selected_axis = report["selected_axis"]
            candidates.append(dict(factor=factor, axis=selected_axis, kernel=bundle.candidate))
            bundles[f"a{selected_axis}_f{factor}"] = bundle
    factor, selection = select_candidates(candidates, contract)
    if factor == 1:
        report = dict(compile_status="DISABLED", owner="none", requested_factor=1, applied_factor=1,
                      reason=selection["reason"])
        result = batching.BatchKernel(baseline, None, batching.BatchRequest(), report, [])
    else:
        result = bundles[selection["selected_key"]]
    result.block_auto_selection = dict(selection, selected_factor=factor, candidate_plan=plan)
    return result
