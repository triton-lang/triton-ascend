# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Numerical references and launch helpers for block batching device tests."""
import time

import torch
import triton
from block_batching_kernels import (_rmsnorm_fwd_fused, _fused_situgluv2_fwd_triton_kernel,
                                    _fused_situgluv2_bwd_triton_kernel)

TOLERANCES = {torch.float32: (2e-4, 2e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}


def prepare_case(kind, rows, cols, dtype, factor=1, axis=0, eps=1e-6):
    torch.manual_seed(42)
    a = torch.randn((rows, cols), device="npu", dtype=dtype)
    b = torch.randn_like(a)
    options = dict(debug=True)
    if factor != 1:
        options.update(block_batching_factor=factor, block_batching_axis=axis)
    if kind == "rmsnorm":
        weight = torch.randn(cols, device="npu", dtype=torch.float32)
        out = torch.full_like(a, float("nan"))
        rrms = torch.full((rows, ), float("nan"), device="npu", dtype=torch.float32)
        grid = (rows, )

        def launch():
            return _rmsnorm_fwd_fused[grid](a, out, weight, rrms, cols, cols, eps,
                                            D_ROUNDED=triton.next_power_of_2(cols), **options)

        inv = torch.rsqrt((a.float()**2).mean(1) + eps)
        expected = [(a.float() * inv[:, None] * weight).to(dtype), inv]
        outputs = [out, rrms]
    else:
        grid = (rows, triton.cdiv(cols, 128))
        out = torch.full_like(a, float("nan"))
        beta = linear_beta = 0.5
        if kind == "situ_fwd":

            def launch():
                return _fused_situgluv2_fwd_triton_kernel[grid](a, b, out, cols, cols, 1, cols, 1, cols, 1, beta,
                                                                1 / beta, linear_beta, 1 / linear_beta, BLOCK=128,
                                                                **options)

            expected = [(beta * torch.tanh(a.float() / beta) * torch.sigmoid(a.float()) * linear_beta *
                         torch.tanh(b.float() / linear_beta)).to(dtype)]
            outputs = [out]
        else:
            dout = torch.randn_like(a)
            db = torch.full_like(b, float("nan"))

            def launch():
                return _fused_situgluv2_bwd_triton_kernel[grid](a, b, dout, out, db, cols, cols, 1, cols, 1, cols, 1,
                                                                cols, 1, cols, 1, beta, 1 / beta, linear_beta,
                                                                1 / linear_beta, BLOCK=128, **options)

            # Independent autograd reference rather than copying the kernel derivative.
            ar, br = a.float().detach().requires_grad_(), b.float().detach().requires_grad_()
            reference = beta * torch.tanh(ar / beta) * torch.sigmoid(ar) * linear_beta * torch.tanh(br / linear_beta)
            expected = [x.to(dtype) for x in torch.autograd.grad(reference, (ar, br), dout.float())]
            outputs = [out, db]
    return {"launch": launch, "options": options, "outputs": outputs, "expected": expected, "grid": grid}


def run_case(kind, rows, cols, dtype, factor=1, axis=0, eps=1e-6, require_applied=False):
    case = prepare_case(kind, rows, cols, dtype, factor, axis, eps)
    launch, outputs, expected, grid = (case[k] for k in ("launch", "outputs", "expected", "grid"))
    first_call_start = time.perf_counter()
    compiled = launch()
    torch.npu.synchronize()
    first_call_wall_ms = (time.perf_counter() - first_call_start) * 1000
    rtol, atol = TOLERANCES[dtype]
    for output, reference in zip(outputs, expected):
        output_rtol, output_atol = TOLERANCES[output.dtype]
        torch.testing.assert_close(output, reference, rtol=output_rtol, atol=output_atol)
    report = getattr(compiled.metadata, "block_batching", None)
    actual = getattr(compiled, "last_launch", {"runtime_status": "BASELINE_SELECTED"})
    if factor > 1:
        assert report is not None, "batching request did not return a report"
        if require_applied:
            assert actual["runtime_status"] == "APPLIED", (report, actual)
            assert actual["applied_factor"] == factor, actual
    result = {
        "kernel":
        kind, "shape": [rows, cols], "dtype":
        str(dtype), "eps":
        eps, "original_grid":
        list(grid), "factor":
        factor, "axis":
        axis, "strides": [cols, 1], "first_call_wall_ms":
        first_call_wall_ms, "constexpr":
        ({"D_ROUNDED": triton.next_power_of_2(cols)} if kind == "rmsnorm" else {"BLOCK": 128}), "scalars":
        ({"d": cols, "eps": eps} if kind == "rmsnorm" else
         {"N": cols, "beta": 0.5, "inv_beta": 2.0, "linear_beta": 0.5, "inv_linear_beta": 2.0}), "correctness":
        True, "rtol":
        rtol, "atol":
        atol, "output_tolerances":
        [{"dtype": str(x.dtype), "rtol": TOLERANCES[x.dtype][0], "atol": TOLERANCES[x.dtype][1]}
         for x in outputs], "compile_report":
        report, "execution":
        actual, "max_abs_error": [float((x.float() - y.float()).abs().max())
                                  for x, y in zip(outputs, expected)], "metadata":
        compiled.metadata._asdict()
    }
    return result, compiled
