# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Used-result delta guard for the indirect atomic cost model.
#
# `AtomicMemoryCostModel.cpp` prices a consumed atomic return value as an
# additive delta on top of the unused atomic cost, read from the dedicated
# `<op>.<dtype>.<class>.result_used` profile slot.  This test compiles the same
# kernel twice -- once storing the atomic's returned old value (USE_RESULT=1)
# and once storing the original operand (USE_RESULT=0) -- and checks that the
# *predicted* delta equals the *measured* delta.  Comparing deltas cancels the
# base prediction error, so a pass means the `.result_used` slot is priced.
#
# The `simt_loaded_index` scatter point keeps the same route (all_simt_only) for
# both variants.  Contiguous kernels can flip the whole route when the result
# is used (SIMD becomes expensive), which would make the delta meaningless, so
# they only run when a route-pinned profile is supplied:
#
#   TRITON_UR_PROFILE=/path/to/route_pinned.json   pytest ...
#
# The guard is opt in (needs a real device):
#   TRITON_RUN_ATOMIC_COSTMODEL_GUARDS=1   enable
# =============================================================================

import csv
import json
import os
import statistics
from pathlib import Path

import pytest
import torch
import torch_npu
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    os.getenv("TRITON_RUN_ATOMIC_COSTMODEL_GUARDS") != "1",
    reason="atomic costmodel guards are opt-in",
)

REL_TOLERANCE = 0.15
ABS_TOLERANCE_US = 2.0
CACHE_LINE_BYTES = 128
BLOCKS = (1024, 2048)
UR_REPEATS = 3


@triton.jit
def scatter_add(out_ptr, dst_ptr, idx_ptr, val_ptr, BLOCK: tl.constexpr, LINE: tl.constexpr, USE_RESULT: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * (BLOCK * LINE)
    offs = tl.arange(0, BLOCK)
    idx = tl.load(idx_ptr + offs)
    value = tl.load(val_ptr + offs)
    ret = tl.atomic_add(out_ptr + base + idx, value)
    if USE_RESULT:
        tl.store(dst_ptr + base + idx, ret)
    else:
        tl.store(dst_ptr + base + idx, value)


@triton.jit
def contiguous_add(out_ptr, dst_ptr, val_ptr, BLOCK: tl.constexpr, LINE: tl.constexpr, USE_RESULT: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    value = tl.load(val_ptr + offs)
    ret = tl.atomic_add(out_ptr + pid * BLOCK + offs, value)
    if USE_RESULT:
        tl.store(dst_ptr + pid * BLOCK + offs, ret)
    else:
        tl.store(dst_ptr + pid * BLOCK + offs, value)


def _points():
    points = {"simt_loaded_index": (scatter_add, "scatter")}
    if os.getenv("TRITON_UR_PROFILE"):
        points["simt_contiguous"] = (contiguous_add, "contig")
        points["simd_contiguous"] = (contiguous_add, "contig")
    return points


POINTS = _points()


def _vector_core_count():
    props = triton.runtime.driver.active.utils.get_device_properties(0)
    for key in ("num_vectorcore", "num_vector_core", "vector_core_num"):
        if key in props:
            return int(props[key])
    raise RuntimeError("cannot read the vector core count: %r" % (props, ))


def _aicore_count():
    props = triton.runtime.driver.active.utils.get_device_properties(0)
    return int(props.get("num_aicore", 0))


def _launch_options(report_path, logical_programs):
    options = {
        "num_warps": 4,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "logical_program_count_hint": logical_programs,
        "physical_vector_core_count_hint": _vector_core_count(),
        "physical_aicore_count_hint": _aicore_count(),
    }
    pinned = os.getenv("TRITON_UR_PROFILE", "")
    if pinned:
        options["auto_simt_model_profile"] = pinned
    return options


def _kernel_duration_us(launch, profile_root, iters=20):
    profile_root.mkdir(parents=True, exist_ok=True)
    import torch_npu.profiler as profiler

    samples = []
    for attempt in range(2):
        for _ in range(3):
            with profiler.profile(
                    activities=[profiler.ProfilerActivity.NPU],
                    schedule=profiler.schedule(wait=1, warmup=3, active=iters),
                    on_trace_ready=profiler.tensorboard_trace_handler(str(profile_root)),
            ) as prof:
                for _ in range(iters):
                    launch()
                    prof.step()
            torch.npu.synchronize()
            for path in Path(profile_root).rglob("kernel_details.csv"):
                with path.open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        if row.get("Duration(us)"):
                            samples.append(float(row["Duration(us)"]))
        if samples:
            break
        # A cold profiler context can drop the first session's data on a fresh
        # process; warm the device once and retry.
        launch()
        torch.npu.synchronize()
        import shutil
        for child in Path(profile_root).iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
    if not samples:
        raise RuntimeError("the profiler produced no kernel duration")
    return statistics.median(samples)


def _metadata_field(binary, name):
    metadata = getattr(binary, "metadata", None)
    if metadata is None:
        return None
    if hasattr(metadata, "get"):
        return metadata.get(name)
    return getattr(metadata, name, None)


def _model_report(report_path, binary):
    if report_path.exists():
        return json.loads(report_path.read_text())
    text = _metadata_field(binary, "auto_simt_scope_report")
    if text:
        return json.loads(text)
    return None


def _predicted_us(report):
    prediction = report.get("selected_route_elapsed_prediction") or {}
    predicted = prediction.get("predicted_microseconds")
    if predicted is None:
        return None
    return float(predicted)


@pytest.mark.parametrize("point", sorted(POINTS))
@pytest.mark.parametrize("block", BLOCKS)
def test_used_result_delta_matches_measurement(point, block, tmp_path):
    kernel, pattern = POINTS[point]
    logical_programs = _vector_core_count()
    line = CACHE_LINE_BYTES // 4  # i32

    if pattern == "scatter":
        span = logical_programs * block * line + 64
        out = torch.zeros((span, ), dtype=torch.int32).npu()
        dst = torch.zeros((span, ), dtype=torch.int32).npu()
        idx = (torch.arange(block, dtype=torch.int32) * line).npu()
        value = torch.ones(block, dtype=torch.int32).npu()
        args = (out, dst, idx, value)
    else:
        span = logical_programs * block
        out = torch.zeros((span, ), dtype=torch.int32).npu()
        dst = torch.zeros((span, ), dtype=torch.int32).npu()
        value = torch.ones(block, dtype=torch.int32).npu()
        args = (out, dst, value)

    per_variant = {}
    launches = {}
    for use_result in (0, 1):
        report_path = tmp_path / ("route_%d.json" % use_result)
        options = _launch_options(report_path, logical_programs)

        def launch(use_result=use_result, options=options):
            return kernel[(logical_programs, )](*args, BLOCK=block, LINE=line, USE_RESULT=use_result, **options)

        binary = launch()
        torch.npu.synchronize()
        report = _model_report(report_path, binary)
        assert report is not None, ("no route report for %s USE_RESULT=%d" % (point, use_result))
        predicted = _predicted_us(report)
        assert predicted is not None, "the route model produced no elapsed prediction"
        route = report.get("effective_decision_kind") or report.get("decision_kind")
        per_variant[use_result] = {"predicted": predicted, "route": route}
        launches[use_result] = launch
        del binary
        torch.npu.empty_cache()

    unused, used = per_variant[0], per_variant[1]
    if unused["route"] != used["route"]:
        if os.getenv("TRITON_UR_PROFILE"):
            pytest.fail("%s block %d: USE_RESULT flipped the route %s -> %s" %
                        (point, block, unused["route"], used["route"]))
        pytest.skip("%s block %d: route flipped without a pinned profile "
                    "(set TRITON_UR_PROFILE)" % (point, block))

    # The delta is a difference of two runtime measurements, so interleave the
    # unused/used launches and take the median per-pair delta.  This cancels
    # shared clock/contention drift that a single unused-then-used pair cannot.
    unused_rounds, used_rounds, deltas = [], [], []
    for round_index in range(UR_REPEATS):
        unused_us = _kernel_duration_us(launches[0], tmp_path / ("prof_unused_%d" % round_index))
        used_us = _kernel_duration_us(launches[1], tmp_path / ("prof_used_%d" % round_index))
        unused_rounds.append(unused_us)
        used_rounds.append(used_us)
        deltas.append(used_us - unused_us)

    unused["measured"] = statistics.median(unused_rounds)
    used["measured"] = statistics.median(used_rounds)
    pred_delta = used["predicted"] - unused["predicted"]
    meas_delta = statistics.median(deltas)
    abs_err = abs(pred_delta - meas_delta)
    rel_err = abs_err / abs(meas_delta) if meas_delta else float("inf")

    print("\n%s block=%d route=%s" % (point, block, used["route"]))
    print("  pred unused/used : %.3f / %.3f us" % (unused["predicted"], used["predicted"]))
    print("  rounds unused    : %s us" % ", ".join("%.3f" % value for value in unused_rounds))
    print("  rounds used      : %s us" % ", ".join("%.3f" % value for value in used_rounds))
    print("  meas unused/used : %.3f / %.3f us" % (unused["measured"], used["measured"]))
    print("  pred_delta       : %.3f us" % pred_delta)
    print("  meas_delta       : %.3f us" % meas_delta)
    print("  abs / rel error  : %.3f us / %.1f%%" % (abs_err, 100.0 * rel_err))

    assert abs_err <= ABS_TOLERANCE_US or rel_err <= REL_TOLERANCE, (
        "%s block %d: used-result delta %.3f us but measured %.3f us "
        "(abs %.3f us, rel %.1f%%)" % (point, block, pred_delta, meas_delta, abs_err, 100.0 * rel_err))
