# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# =============================================================================
# Indirect atomic cost model guard.
#
# This test measures the SIMT indirect atomic path the way the model sees it
# and compares the measurement against the model prediction.
#
# Why the measurement is taken here and not from a template probe
# ----------------------------------------------------------------
# A probe that calls the template entry point directly measures the template,
# not the operation.  It is the right instrument for answering "how many
# instructions does this primitive expand to", which is what the instruction
# level evidence is for.  It is not the right instrument for answering "what
# does this operation cost in a compiled kernel", because the compiled kernel
# also pays for the loads, the stores, the mask, the launch and the schedule.
#
# The rate that the profile stores is used by the route model to compare
# candidates, so it has to be fitted from the same quantity the model
# predicts: the end to end cost of the compiled operation.
#
# The test therefore does three things per case:
#   1. compiles the kernel with the route model enabled and keeps its report;
#   2. measures the kernel duration with the profiler and fits the per element
#      cost from the sweep over the block count;
#   3. compares the model prediction against the measurement and requires the
#      relative error to stay within the tolerance.
#
# The fitted per element cost is the number a profile slot has to carry, so the
# test prints it in the form the profile expects.
#
# The guard is opt in because it needs a real device and an idle card.
# =============================================================================

import json
import os
import statistics
from pathlib import Path

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95

pytestmark = pytest.mark.skipif(
    os.getenv("TRITON_RUN_ATOMIC_COSTMODEL_GUARDS") != "1",
    reason="indirect atomic costmodel guards are opt-in",
)

# The model prices cycles; the device counts SYS_CNT ticks.  The frequency is
# a property of the target, so it is measured here instead of assumed.
SYS_CNT_MHZ = float(os.getenv("TRITON_ASCEND_SYS_CNT_MHZ", "0") or 0)

# A slot may not be more than this far away from the measurement it replaces.
RELATIVE_TOLERANCE = 0.10

# One element per cache line keeps the address pattern a scatter instead of a
# line sharing artifact.
CACHE_LINE_BYTES = 128

BLOCKS = (256, 512, 1024, 2048)


@triton.jit
def scatter_add(out_ptr, idx_ptr, val_ptr, BLOCK: tl.constexpr, LINE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * (BLOCK * LINE)
    offs = tl.arange(0, BLOCK)
    idx = tl.load(idx_ptr + offs)
    value = tl.load(val_ptr + offs)
    tl.atomic_add(out_ptr + base + idx, value)


@triton.jit
def scatter_max(out_ptr, idx_ptr, val_ptr, BLOCK: tl.constexpr, LINE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * (BLOCK * LINE)
    offs = tl.arange(0, BLOCK)
    idx = tl.load(idx_ptr + offs)
    value = tl.load(val_ptr + offs)
    tl.atomic_max(out_ptr + base + idx, value)


@triton.jit
def scatter_min(out_ptr, idx_ptr, val_ptr, BLOCK: tl.constexpr, LINE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * (BLOCK * LINE)
    offs = tl.arange(0, BLOCK)
    idx = tl.load(idx_ptr + offs)
    value = tl.load(val_ptr + offs)
    tl.atomic_min(out_ptr + base + idx, value)


@triton.jit
def scatter_xchg(out_ptr, idx_ptr, val_ptr, BLOCK: tl.constexpr, LINE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * (BLOCK * LINE)
    offs = tl.arange(0, BLOCK)
    idx = tl.load(idx_ptr + offs)
    value = tl.load(val_ptr + offs)
    tl.atomic_xchg(out_ptr + base + idx, value)


@triton.jit
def scatter_cas(out_ptr, idx_ptr, cmp_ptr, val_ptr, BLOCK: tl.constexpr,
                LINE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * (BLOCK * LINE)
    offs = tl.arange(0, BLOCK)
    idx = tl.load(idx_ptr + offs)
    compare = tl.load(cmp_ptr + offs)
    value = tl.load(val_ptr + offs)
    tl.atomic_cas(out_ptr + base + idx, compare, value)


KERNELS = {
    "add": scatter_add,
    "max": scatter_max,
    "min": scatter_min,
    "xchg": scatter_xchg,
    "cas": scatter_cas,
}

DTYPES = {
    "f32": (torch.float32, 4),
    "i32": (torch.int32, 4),
}


def _vector_core_count():
    props = triton.runtime.driver.active.utils.get_device_properties(0)
    for key in ("num_vectorcore", "num_vector_core", "vector_core_num"):
        if key in props:
            return int(props[key])
    raise RuntimeError("cannot read the vector core count: %r" % (props,))


def _launch_options(report_path, logical_programs):
    vector_cores = _vector_core_count()
    return {
        "num_warps": 4,
        "compile_mode": "simd_simt_template",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "logical_program_count_hint": logical_programs,
        "physical_vector_core_count_hint": vector_cores,
    }


def _kernel_duration_us(launch, profile_root, iters=20):
    """Median duration of the launched kernel, without the host side cost."""
    profile_root.mkdir(parents=True, exist_ok=True)
    import torch_npu.profiler as profiler

    samples = []
    for _ in range(3):
        with profiler.profile(
                activities=[profiler.ProfilerActivity.NPU],
                schedule=profiler.schedule(wait=1, warmup=3, active=iters),
        ) as prof:
            for _ in range(iters):
                launch()
                prof.step()
        torch.npu.synchronize()
        for path in Path(profile_root).rglob("kernel_details.csv"):
            with path.open(newline="") as stream:
                import csv
                for row in csv.DictReader(stream):
                    if row.get("Duration(us)"):
                        samples.append(float(row["Duration(us)"]))
    if not samples:
        raise RuntimeError("the profiler produced no kernel duration")
    return statistics.median(samples)


def _fit_slope(points):
    """Least squares slope of cycles against the element count."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    den = n * sxx - sx * sx
    if den == 0:
        raise RuntimeError("the sweep has no usable spread")
    slope = (n * sxy - sx * sy) / den
    return slope, (sy - slope * sx) / n


def _model_prediction_us(report):
    """The elapsed prediction the route model produced for this kernel."""
    prediction = report.get("selected_route_elapsed_prediction") or {}
    predicted = prediction.get("predicted_microseconds")
    if predicted is None:
        return None, prediction
    return float(predicted), prediction


@pytest.mark.parametrize("op", sorted(KERNELS))
@pytest.mark.parametrize("dtype_name", sorted(DTYPES))
def test_indirect_atomic_cost_model_matches_measurement(op, dtype_name, tmp_path):
    if not is_compile_on_910_95():
        pytest.skip("the indirect atomic path is only enabled on 910_95/950")

    torch_dtype, elem_bytes = DTYPES[dtype_name]
    line = CACHE_LINE_BYTES // elem_bytes
    vector_cores = _vector_core_count()
    logical_programs = vector_cores

    points = []
    report = None
    for block in BLOCKS:
        address_count = logical_programs * block * line + 64
        out = torch.zeros(address_count, dtype=torch_dtype).npu()
        idx = (torch.arange(block, dtype=torch.int32) * line).npu()
        value = torch.ones(block, dtype=torch_dtype).npu()
        compare = torch.zeros(block, dtype=torch_dtype).npu()

        report_path = tmp_path / ("route_%d.json" % block)
        options = _launch_options(report_path, logical_programs)
        if op == "cas":
            args = (out, idx, compare, value)
        else:
            args = (out, idx, value)

        def launch(args=args, options=options, block=block):
            KERNELS[op][(logical_programs, )](*args, BLOCK=block, LINE=line, **options)

        launch()
        torch.npu.synchronize()

        # The atomic has to have happened once per element.
        expected = float(block * logical_programs)
        got = float(out.float().sum().item())
        assert abs(got - expected) <= max(1.0, 0.01 * expected), (
            "the atomic result is wrong: %r against %r" % (got, expected))

        duration_us = _kernel_duration_us(launch, tmp_path / ("prof_%d" % block))
        cycles = duration_us * (SYS_CNT_MHZ or 1.0)
        points.append((block * logical_programs, cycles))

        if report is None:
            assert report_path.exists(), (
                "the route model produced no report, so the operation did not "
                "reach the route model at all")
            report = json.loads(report_path.read_text())

        del out, idx, value, compare
        torch.npu.empty_cache()

    assert report is not None
    assert report.get("stage_model", {}).get("applied"), (
        "the route model did not apply")

    predicted_us, prediction = _model_prediction_us(report)
    if predicted_us is None:
        pytest.skip("the route model produced no elapsed prediction")

    # The fitted slope is the per element cost of the operation, in the unit
    # the profile stores.  It is reported so that a profile slot can be
    # filled from this measurement.
    slope, intercept = _fit_slope(points)
    print("\n%s.%s" % (op, dtype_name))
    print("  fitted per element cost : %.4f cycles" % slope)
    print("  fitted fixed cost       : %.1f cycles" % intercept)
    print("  model predicted         : %.3f us" % predicted_us)
    print("  measured                : %.3f us" % duration_us)
    print("  decision                : %s" % report.get("decision_kind"))
    print("  formula                 : %s" % prediction.get("formula"))

    relative = abs(predicted_us - duration_us) / duration_us
    print("  relative error          : %.3f" % relative)
    assert relative <= RELATIVE_TOLERANCE, (
        "the route model predicts %.3f us against a measured %.3f us, a "
        "relative error of %.3f above the %.2f tolerance" %
        (predicted_us, duration_us, relative, RELATIVE_TOLERANCE))
