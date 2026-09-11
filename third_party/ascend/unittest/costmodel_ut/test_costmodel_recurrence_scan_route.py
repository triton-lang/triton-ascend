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

import json
import os
import tempfile
import unittest

import torch
import torch_npu
import triton
import triton.language as tl

from triton.backends.ascend.utils import is_compile_on_910_95

# Routes whose fast path is SIMT (all SIMT, or SIMD host with a SIMT scope).
SIMT_ROUTES = {"all_simt_only", "mixed_simd_simt"}

# The launch options below are passed explicitly, but these environment
# variables still leak in from the caller's shell: TRITON_ASCEND_COMPILE_MODE
# *overrides* the explicit compile_mode option, and TRITON_ASCEND_AUTO_SIMT_PROFILE
# replaces the calibrated cost profile (the scope mode/dump env vars are only
# fallbacks, cleared here for symmetry).  Removing them keeps the route decision
# deterministic no matter how the caller configured the shell.
ISOLATED_ENV = (
    "TRITON_ASCEND_COMPILE_MODE",
    "TRITON_ASCEND_AUTO_SIMT_SCOPE",
    "TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP",
    "TRITON_ASCEND_AUTO_SIMT_PROFILE",
)


@triton.jit(do_not_specialize=["T"])
def _recurrence_scan_kernel(s_ptr, o_ptr, T, BT: tl.constexpr, H: tl.constexpr):
    """Loop-carried recurrence with a nested prefix scan.

    Mirrors the structure of fla ``chunk_global_cumsum_scalar_kernel``: the
    carried scalar ``b_z`` accumulates the running total across chunks while
    ``tl.cumsum`` scans the chunk.  Guard target: "[sim_costmodel](fix)
    LoopCarriedRecurrence applies prefixScanDependencyFactor to nested scans".
    Before that fix the scan shuffle nested in the recurrence was billed at the
    ideal SIMD rate, under-estimating the SIMD cost and routing the kernel to
    ``all_simd``; the scan-contributed shuffle must now consume the prefix-scan
    dependency factor, which makes SIMT the cheaper candidate.
    """
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    base = i_n * T * H + i_h
    b_z = tl.zeros([], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        offs = i_c * BT + tl.arange(0, BT)
        mask = offs < T
        b_s = tl.load(s_ptr + base + offs * H, mask=mask, other=0.0).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0) + b_z
        b_z += tl.sum(b_s, axis=0)
        tl.store(o_ptr + base + offs * H, b_o.to(o_ptr.dtype.element_ty), mask=mask)


class CostmodelRecurrenceScanRouteTest(unittest.TestCase):

    def setUp(self):
        self._saved_env = {name: os.environ.get(name) for name in ISOLATED_ENV}
        for name in ISOLATED_ENV:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_recurrence_scan_routes_to_simt(self):
        if not is_compile_on_910_95():
            self.skipTest("SIMD/SIMT cost model only supports 910_95")

        batch, sequence, heads, block = 4, 1024, 8, 256
        torch.manual_seed(0)
        source = torch.randn(batch, sequence, heads, dtype=torch.float32, device="npu")
        output = torch.empty_like(source)

        with tempfile.TemporaryDirectory() as tmpdir:
            # The costmodel dumps one JSON line per compile; the directory must exist.
            report_path = os.path.join(tmpdir, "output.json")
            _recurrence_scan_kernel[(batch * heads, )](
                source,
                output,
                sequence,
                BT=block,
                H=heads,
                num_warps=4,
                compile_mode="simd_simt",
                auto_simt_scope_mode="auto",
                auto_simt_scope_dump=report_path,
            )
            torch.npu.synchronize()

            expected = source.cumsum(dim=1)
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

            self.assertTrue(os.path.exists(report_path), "costmodel route report was not written")
            with open(report_path, "r", encoding="utf-8") as stream:
                reports = [json.loads(line) for line in stream if line.strip()]
            self.assertTrue(reports, "costmodel route report is empty")
            report = reports[-1]
            self.assertTrue(report["stage_model"]["applied"])
            self.assertIn(
                report["effective_decision_kind"], SIMT_ROUTES,
                f"expected a SIMT route, got {report['effective_decision_kind']} "
                f"with candidate costs {report['candidate_costs']}")


if __name__ == "__main__":
    unittest.main()
