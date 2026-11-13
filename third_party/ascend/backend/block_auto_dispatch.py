# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Launch-contract construction and bounded CPU compilation for automatic batching.

The existing Ascend continuation and compile-bundle hook own this path. Candidate
compilation never re-enters JIT, launches a device kernel, or benchmarks a kernel.
"""
from concurrent.futures import Future
import hashlib
import json
import math
import os
from pathlib import Path
import threading

VERSION = "jit-block-auto-v2"
_pending = {}
_pending_lock = threading.Lock()


def _argument(value):
    if value is None or type(value) in (bool, int):
        return {"kind": "scalar", "value": value}
    if type(value) is float and math.isfinite(value):
        return {"kind": "scalar", "value": value}
    # Shape, dtype and strides are host metadata. Never inspect tensor contents,
    # data_ptr(), a device allocation, or a stream here.
    if hasattr(value, "dtype") and hasattr(value, "shape") and callable(getattr(value, "stride", None)):
        shape, strides = tuple(value.shape), tuple(value.stride())
        offset = getattr(value, "storage_offset", lambda: 0)()
        if (all(type(x) is int and x >= 0 for x in shape) and all(type(x) is int for x in strides)
                and type(offset) is int):
            return {
                "kind": "tensor", "dtype": str(value.dtype), "shape": list(shape), "stride": list(strides),
                "storage_offset": offset
            }
    return {"kind": "unsupported", "type": type(value).__module__ + "." + type(value).__qualname__}


def prepare_launch(fn, args, kwargs, grid, warmup, device, backend):
    """Encode selection inputs in the existing JIT key before its cache lookup."""
    # Do not reinterpret a kernel parameter as a compiler option.
    if "block_batching_factor" in fn.arg_names:
        return kwargs, grid
    if "block_batching_factor" in kwargs and kwargs["block_batching_factor"] != "auto":
        return kwargs, grid

    def baseline():
        if kwargs.get("block_batching_factor") != "auto":
            return kwargs, grid
        result = dict(kwargs, block_batching_factor=1)
        result.pop("block_auto_contract", None)
        return result, grid

    if warmup:
        return baseline()
    target = getattr(backend, "target", None)
    if not str(getattr(target, "arch", "")).startswith(("Ascend950", "Ascend910_95")):
        return baseline()
    # A kernel argument with a compiler-option name remains a kernel argument.
    if any(name in fn.arg_names for name in ("block_batching_grid", "block_auto_contract")):
        return baseline()
    from .block_batching import _capture_state
    if _capture_state() is not False:
        # Reuse the normal F1 warmup key; a capture cannot create/load a new
        # guarded bundle. An unavailable query cannot prove eager execution.
        return baseline()
    bound, specialization, compile_options = fn.device_caches[device][4](*args, **kwargs)
    if callable(grid):
        grid = grid(bound)
    if (not isinstance(grid, (tuple, list)) or not 1 <= len(grid) <= 3
            or any(type(n) is not int or n <= 0 or n > 2147483647 for n in grid)):
        return baseline()
    extents = tuple(grid) + (1, ) * (3 - len(grid))
    arguments = {
        name: _argument(value)
        for (name, value), spec in zip(bound.items(), specialization)
        if spec[0] != "constexpr"
    }
    from .block_auto_model import VERSION as model_version
    core_floor = os.getenv("TA_BLOCK_BATCHING_CORE_FLOOR")
    if core_floor is not None and int(core_floor) < 0:
        raise ValueError("TA_BLOCK_BATCHING_CORE_FLOOR must be nonnegative")
    normalized_options = dict(compile_options, block_batching_factor="auto")
    normalized_options.pop("block_auto_contract", None)
    normalized = backend.parse_options(normalized_options)
    normalized_identity = repr(sorted(vars(normalized).items()))
    contract = {
        "version": VERSION,
        "model_version": model_version,
        "target": str(target.arch),
        "device": device,
        "grid": list(extents),
        "arguments": arguments,
        "argument_order": list(arguments),
        "core_floor": core_floor,
        "device_limit": os.getenv("NPU_DEVICE_LIMIT"),
        "compile_options": normalized_identity,
        "cv_empirical_short_batch": True,
    }
    options = dict(kwargs, block_batching_factor="auto", block_batching_grid=extents,
                   block_auto_contract=json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return options, grid


def _contract(options):
    value = options.get("block_auto_contract")
    if not isinstance(value, str):
        raise ValueError("automatic block selection requires an internal launch contract")
    contract = json.loads(value)
    grid = contract.get("grid")
    if (contract.get("version") != VERSION or not isinstance(grid, list) or len(grid) != 3
            or any(type(n) is not int or n <= 0 or n > 2147483647 for n in grid)):
        raise ValueError("invalid automatic block launch contract")
    return contract


def _rejection(error):
    """Only expected admission failures can turn into an unavailable candidate."""
    import subprocess
    from triton.compiler.errors import MLIRCompilationError
    from .block_batching import BatchingRejected, _internal_backend_error
    cause = error
    while isinstance(cause, MLIRCompilationError) and cause.__cause__ is not None:
        cause = cause.__cause__
    if isinstance(cause, BatchingRejected):
        return {"reason": str(cause), "diagnostic": getattr(cause, "diagnostic", None)}
    if isinstance(cause, subprocess.CalledProcessError):
        detail = cause.stderr.decode(errors="replace") if isinstance(cause.stderr, bytes) else str(cause.stderr or "")
        if not _internal_backend_error(cause.returncode, detail):
            return {"reason": "BACKEND_COMPILE_REJECTED", "diagnostic": detail[-16000:]}
    raise error


def _binary_identity(kernel):
    return {"hash": kernel.hash, "binary_sha256": hashlib.sha256(kernel.kernel).hexdigest()}


def _decision_cache(backend, src, normalized, contract, baseline, shortlist):
    from triton.runtime.cache import get_cache_manager
    identity = {
        "policy": VERSION, "source": src.hash(), "target": str(backend.target), "options":
        repr(sorted(vars(normalized).items())), "contract": contract, "baseline": _binary_identity(baseline),
        "shortlist": shortlist
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    key = hashlib.sha256(encoded.encode()).hexdigest()
    return get_cache_manager(key), identity


def _read_decision(cache, identity):
    from triton import knobs
    if getattr(getattr(knobs, "compilation", None), "always_compile", False):
        return None
    get_file = getattr(cache, "get_file", None)
    path = get_file("block_auto.selection.json") if get_file is not None else None
    if path is None:
        return None
    try:
        manifest = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(manifest, dict) or manifest.get("version") != 1 or manifest.get("identity") != identity
            or type(manifest.get("selected_factor")) is not int or manifest["selected_factor"] not in (1, 2, 4)
            or not isinstance(manifest.get("selection"), dict)):
        return None
    return manifest


def _candidate_key(axis, factor):
    return "unmerged" if factor == 1 else f"a{axis}_f{factor}"


def _selected(candidates, chosen, selection):
    matches = [item for item in candidates if type(chosen) is int and item["factor"] == chosen]
    if "selected_key" in selection:
        matches = [
            item for item in matches if _candidate_key(item["axis"], item["factor"]) == selection["selected_key"]
        ]
    if "selected_axis" in selection:
        matches = [item for item in matches if item["axis"] == selection["selected_axis"]]
    if len(matches) != 1:
        raise RuntimeError("CPU block selector returned an unavailable or ambiguous candidate")
    return matches[0]


def _bundle(batching, baseline, native_reports, positions, selected, selection):
    factor, axis = selected["factor"], selected["axis"]
    if factor == 1:
        report = {
            "report_version": 1, "compile_status": "DISABLED", "owner": "none", "requested_factor": 1, "requested_axis":
            -1, "applied_factor": 1, "reason": selection.get("reason", "AUTO_BASELINE")
        }
        candidate, request = None, batching.BatchRequest()
    else:
        candidate = selected["kernel"]
        report = native_reports[(axis, factor)]
        request = batching.BatchRequest(factor, axis)
    report = dict(report, block_auto_selection=selection)
    result = batching.BatchKernel(baseline, candidate, request, report, positions)
    result.block_auto_selection = selection
    return result


def _compile_auto_bundle(backend, src, options, compile_variant):
    from . import block_batching as batching
    from .block_auto_model import select_candidates
    from .utils import backend_policy
    contract = _contract(options)
    normalized = backend.parse_options(options)
    # The launch contract belongs to the decision, never to individual binary
    # identity. All variants, including F1, retain their static evidence.
    common = dict(options, block_auto_collect=True)
    common.pop("block_auto_contract", None)
    common.pop("block_batching_grid", None)
    baseline_options = dict(common, block_batching_factor=1, block_batching_axis=-1)
    rejected = []
    refusal = None
    if not str(backend.target.arch).startswith(("Ascend950", "Ascend910_95")):
        refusal = "TARGET_NOT_VALIDATED"
    elif normalized.is_pure_simt or normalized.compile_mode == "simt_only":
        refusal = "SIMT_NOT_SUPPORTED"
    elif backend_policy not in (None, "torch_npu"):
        refusal = "FRAMEWORK_NOT_VALIDATED"
    elif not hasattr(src, "fn"):
        refusal = "AST_SOURCE_REQUIRED"
    frozen, positions = src, []
    if refusal is None:
        try:
            positions = batching._runtime_positions(src)
            frozen = batching._freeze_source(backend, src, baseline_options, normalized.block_batching_backend_revision)
        except batching.BatchingRejected as error:
            refusal = str(error)
    # Baseline failures always retain their original exception and diagnostics.
    baseline = compile_variant(baseline_options, frozen)
    candidates = [{"factor": 1, "axis": -1, "kernel": baseline}]
    native_reports = {}
    attempted = {}
    requested_axis = options.get("block_batching_axis", -1)
    cv = (getattr(baseline.metadata, "mix_mode", None) in ("aic", "mix")
          and getattr(baseline.metadata, "parallel_mode", None) == "simd")
    if refusal is not None:
        plan = {"candidates": [], "reason": refusal}
    elif cv:
        from .block_auto_cv import candidate_plan
        plan = candidate_plan(contract, baseline, backend=backend, frozen=frozen, options=common, axis=requested_axis)
    else:
        plan = {
            "candidates": [{"axis": requested_axis, "factor": factor} for factor in (2, 4)], "reason":
            "VV_F2_F4_AUTOMATIC_AXIS"
        }
    shortlist = [{"axis": item["axis"], "factor": item["factor"]} for item in plan["candidates"]]
    if (len(shortlist) > 2 or len({(item["axis"], item["factor"])
                                   for item in shortlist}) != len(shortlist) or any(
                                       type(item["factor"]) is not int or item["factor"] not in
                                       (2, 4) or type(item["axis"]) is not int or item["axis"] not in (-1, 0, 1, 2)
                                       for item in shortlist)):
        raise RuntimeError("automatic block candidate plan exceeds the two-candidate contract")

    def obtain(spec):
        factor, axis = spec["factor"], spec["axis"]
        identity = (axis, factor)
        if identity in attempted:
            return attempted[identity]
        request = batching.BatchRequest(factor, axis)
        variant = dict(common, block_batching_factor=factor, block_batching_axis=axis,
                       block_batching_grid=tuple(contract["grid"]))
        try:
            # Each planned (axis, factor) has one backend attempt, including a
            # rejected cache restore. CV may shortlist F4 on two distinct axes.
            kernel = compile_variant(variant, frozen)
            report = batching.validate_candidate(kernel, request)
            selected_axis = report.get("selected_axis")
            if type(selected_axis) is not int or selected_axis not in (0, 1, 2):
                raise batching.BatchingRejected("CANDIDATE_AXIS_MISSING")
            if axis != -1 and selected_axis != axis:
                raise batching.BatchingRejected("CANDIDATE_AXIS_MISMATCH")
            native_reports[(selected_axis, factor)] = report
            item = {"factor": factor, "axis": selected_axis, "kernel": kernel}
            candidates.append(item)
        except Exception as error:
            rejected.append(dict(_rejection(error), factor=factor, axis=axis))
            item = None
        attempted[identity] = item
        return item

    cache, identity = _decision_cache(backend, src, normalized, contract, baseline, shortlist)
    manifest = _read_decision(cache, identity)
    if manifest is not None and (refusal is None or manifest["selected_factor"] == 1):
        chosen = manifest["selected_factor"]
        selected_axis = manifest.get("selected_axis")
        matching = [item for item in shortlist if item["factor"] == chosen and item["axis"] in (-1, selected_axis)]
        selected = candidates[0] if chosen == 1 else obtain(matching[0]) if len(matching) == 1 else None
        if (selected is not None and manifest.get("selected_binary") == _binary_identity(selected["kernel"])
                and manifest.get("selected_axis") == selected["axis"]):
            selection = dict(manifest["selection"], decision_cache_hit=True)
            return _bundle(batching, baseline, native_reports, positions, selected, selection)
    for spec in shortlist:
        obtain(spec)
    if refusal is not None:
        rejected.append({"reason": refusal})
    model_candidates = candidates + [{
        "factor": item["factor"], "axis": item["axis"], "kernel": None, "status": "COMPILE_REJECTED", "error":
        item["reason"]
    } for item in rejected if "factor" in item]
    if cv and not shortlist:
        chosen, selection = 1, {
            "selected_key": "unmerged", "selected_axis": -1, "model_complete": False, "reason":
            plan.get("reason", "CV_CANDIDATE_PLAN_UNAVAILABLE")
        }
    else:
        chosen, selection = select_candidates(model_candidates, contract)
    selected = _selected(candidates, chosen, selection)
    selection = dict(selection, policy=VERSION, selected_factor=chosen, candidate_count=len(model_candidates),
                     compiled_candidates=len(candidates), rejections=rejected, device_executed=False,
                     decision_cache_hit=False, candidate_plan=plan)
    manifest = {
        "version": 1, "identity": identity, "selected_factor": chosen, "selected_axis": selected["axis"],
        "selected_binary": _binary_identity(selected["kernel"]), "selection": selection
    }
    cache.put(json.dumps(manifest, sort_keys=True, allow_nan=False), "block_auto.selection.json")
    return _bundle(batching, baseline, native_reports, positions, selected, selection)


def compile_auto_bundle(backend, src, options, compile_variant):
    """Coalesce concurrent compilation without locking callbacks or compiler work."""
    key = (os.getpid(), str(backend.target), src.hash(), repr(sorted(options.items())))
    owner = threading.get_ident()
    with _pending_lock:
        entry = _pending.get(key)
        if entry is None:
            future = Future()
            _pending[key] = (owner, future)
        else:
            other_owner, future = entry
            if owner == other_owner:
                raise RuntimeError("recursive automatic compilation of the same launch contract")
    if entry is not None:
        return future.result()
    try:
        result = _compile_auto_bundle(backend, src, options, compile_variant)
    except BaseException as error:
        future.set_exception(error)
        raise
    else:
        future.set_result(result)
        return result
    finally:
        with _pending_lock:
            _pending.pop(key, None)
