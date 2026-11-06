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
# The rate that the profile stores is consumed by the route model to compare
# candidates, so it has to be fitted from the same quantity the model
# predicts: the end to end cost of the compiled operation.  The sweep over the
# block count gives that quantity as a slope, which removes the fixed per
# launch cost.
#
# The guard is opt in because it needs a real device and an idle card.
#
# Environment:
#   TRITON_RUN_ATOMIC_COSTMODEL_GUARDS=1   enable the guard
#   TRITON_ASCEND_SYS_CNT_MHZ=<freq>       SYS_CNT frequency, when the profile
#                                          does not carry one
#
# The compiler only runs the route model in the ``simd_simt`` compile mode, so
# that is the mode measured here.  ``simd_simt_template`` bypasses the model.
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
from triton.backends.ascend.utils import is_compile_on_910_95

pytestmark = pytest.mark.skipif(
    os.getenv("TRITON_RUN_ATOMIC_COSTMODEL_GUARDS") != "1",
    reason="indirect atomic costmodel guards are opt-in",
)

# A slot may not be more than this far away from the measurement it replaces.
RELATIVE_TOLERANCE = 0.10

# On a shared accelerator the small-block measurements scatter by tens of
# percent between runs, and a sub-microsecond kernel can miss by twenty percent
# while missing by less than half a microsecond in absolute terms. A launch is
# therefore accepted when either bound holds: within ten percent, or within two
# microseconds. The absolute bound only ever admits kernels whose duration is a
# few microseconds, so it cannot hide an error that matters at scale.
ABSOLUTE_TOLERANCE_US = 2.0

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
def scatter_cas(out_ptr, idx_ptr, cmp_ptr, val_ptr, BLOCK: tl.constexpr, LINE: tl.constexpr):
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

# The value the destination is seeded with, chosen so that the update always
# changes it.  A min against zero would leave the destination at zero, so the
# result check could not tell a performed atomic from a skipped one.
SEED = {
    "add": 0.0,
    "max": -1.0,
    "min": 1.0,
    "xchg": 0.0,
    "cas": 0.0,
}


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
    return {
        "num_warps": 4,
        # The route model only runs in this compile mode.  simd_simt_template
        # bypasses it, so the atomic never reaches the model there.
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "logical_program_count_hint": logical_programs,
        "physical_vector_core_count_hint": _vector_core_count(),
        "physical_aicore_count_hint": _aicore_count(),
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
    if not samples:
        raise RuntimeError("the profiler produced no kernel duration")
    print("  [prof %s] n=%d us_min=%.3f us_med=%.3f us_max=%.3f" %
          (profile_root.name, len(samples), min(samples), statistics.median(samples), max(samples)))
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


def _metadata_field(binary, name):
    """KernelMetadata is a namedtuple on this backend, not a mapping."""
    metadata = getattr(binary, "metadata", None)
    if metadata is None:
        return None
    if hasattr(metadata, "get"):
        return metadata.get(name)
    return getattr(metadata, name, None)


def _model_report(report_path, binary):
    """The route model report, from the dump or from the compiled metadata.

    The dump is only written when the route model produced scheduling
    feedback, so the compiled metadata is the fallback.
    """
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
        return None, prediction
    return float(predicted), prediction


@pytest.mark.parametrize("op", sorted(KERNELS))
@pytest.mark.parametrize("dtype_name", sorted(DTYPES))
def test_indirect_atomic_cost_model_matches_measurement(op, dtype_name, tmp_path):
    if not is_compile_on_910_95():
        pytest.skip("the indirect atomic path is only enabled on 910_95/950")

    torch_dtype, elem_bytes = DTYPES[dtype_name]
    line = CACHE_LINE_BYTES // elem_bytes
    logical_programs = _vector_core_count()

    rows = []
    for block in BLOCKS:
        address_count = logical_programs * block * line + 64
        out = torch.full((address_count, ), SEED[op], dtype=torch_dtype).npu()
        idx = (torch.arange(block, dtype=torch.int32) * line).npu()
        value = torch.ones(block, dtype=torch_dtype).npu()
        compare = torch.zeros(block, dtype=torch_dtype).npu()

        report_path = tmp_path / ("route_%d.json" % block)
        options = _launch_options(report_path, logical_programs)
        args = (out, idx, compare, value) if op == "cas" else (out, idx, value)

        def launch(args=args, options=options, block=block):
            return KERNELS[op][(logical_programs, )](*args, BLOCK=block, LINE=line, **options)

        binary = launch()
        torch.npu.synchronize()

        # The atomic has to have happened at every touched address exactly
        # once.  Only the touched addresses carry the updated value; the rest
        # of the destination still holds the seed, so the whole buffer cannot
        # be checked with a sum (a min or a max seed is not zero).
        touched = (torch.arange(logical_programs, dtype=torch.int64, device=idx.device)[:, None] * (block * line) +
                   torch.arange(block, dtype=torch.int64, device=idx.device)[None, :] * line)
        written = out.index_select(0, touched.reshape(-1))
        assert bool((written == 1).all().item()), ("the atomic did not update every touched address exactly once, "
                                                   "%d of %d addresses hold the updated value" % (int(
                                                       (written == 1).sum().item()), written.numel()))

        duration_us = _kernel_duration_us(launch, tmp_path / ("prof_%d" % block))
        rows.append({
            "block": block,
            "elements": block * logical_programs,
            "measured_us": duration_us,
            "report": _model_report(report_path, binary),
        })

        del out, idx, value, compare
        torch.npu.empty_cache()

    # The fitted slope is the per element cost of the compiled operation.  It
    # is the number a profile slot has to carry, so it is printed in the unit
    # the profile uses.
    human_points = [(r["elements"], r["measured_us"]) for r in rows]
    slope_us, intercept_us = _fit_slope(human_points)

    frequency_mhz = None
    for row in rows:
        if row["report"] is None:
            continue
        prediction = row["report"].get("selected_route_elapsed_prediction") or {}
        frequency_mhz = prediction.get("system_counter_frequency_mhz")
        if frequency_mhz:
            break
    if not frequency_mhz:
        frequency_mhz = float(os.getenv("TRITON_ASCEND_SYS_CNT_MHZ", "0") or 0)

    print("\n%s.%s" % (op, dtype_name))
    print("  fitted per element cost : %.6f us" % slope_us)
    if frequency_mhz:
        cycles = slope_us * frequency_mhz
        print("  fitted per element cost : %.4f cycles (at %.1f MHz)" % (cycles, frequency_mhz))
        print("  implied slot rate       : %.6f elements/cycle" % (1.0 / cycles if cycles else 0.0))
    print("  fitted fixed cost       : %.3f us" % intercept_us)

    if any(row["report"] is None for row in rows):
        pytest.skip("the route model produced no report for this operation, so "
                    "the operation did not reach the route model at all")

    # The prediction has to hold for every launch that was measured, not only
    # for the largest one: the fixed and per-element terms have to be right at
    # the same time.
    print("  block   measured us   predicted us   rel err   abs err   decision")
    worst_relative = 0.0
    rejected = []
    for row in rows:
        predicted_us, prediction = _predicted_us(row["report"])
        if predicted_us is None:
            pytest.skip("the route model produced no elapsed prediction")
        absolute = abs(predicted_us - row["measured_us"])
        relative = absolute / row["measured_us"]
        worst_relative = max(worst_relative, relative)
        accepted = (relative <= RELATIVE_TOLERANCE or absolute <= ABSOLUTE_TOLERANCE_US)
        if not accepted:
            rejected.append((row["block"], relative, absolute))
        print("  %-6d %12.3f %14.3f %9.3f %9.3f   %s%s" %
              (row["block"], row["measured_us"], predicted_us, relative, absolute, row["report"].get("decision_kind"),
               "" if accepted else "  REJECTED"))
    print("  worst relative error    : %.3f" % worst_relative)
    print("  tolerance               : rel %.2f or abs %.1f us" % (RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE_US))
    print("  formula                 : %s" % prediction.get("formula"))

    assert not rejected, ("the route model exceeds both tolerances at %s; the worst of them is "
                          "%.3f relative and %.3f us absolute" %
                          (", ".join("block %d" % block for block, _, _ in rejected),
                           max(relative for _, relative, _ in rejected), max(absolute for _, _, absolute in rejected)))
