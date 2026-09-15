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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
import tempfile
import unittest

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from triton.backends.ascend.utils import is_compile_on_910_95

# A user-authored simt scope forbids both single-side routes, so the effective
# decision must be the mixed route; a user-authored simd scope only forbids
# all_simt.  Both cases below also expose an automatic or explicit simt anchor,
# so mixed stays the cheapest legal candidate.
MIXED_ROUTE = "mixed_simd_simt"

# These environment variables override the explicit launch options below and
# would make the route decision depend on the caller's shell.  Clearing them
# keeps the decision deterministic.
ISOLATED_ENV = (
    "TRITON_ASCEND_COMPILE_MODE",
    "TRITON_ASCEND_AUTO_SIMT_SCOPE",
    "TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP",
    "TRITON_ASCEND_AUTO_SIMT_PROFILE",
)


@triton.jit(do_not_specialize=["T"])
def _user_simt_scope_kernel(s_ptr, o_ptr, T, BT: tl.constexpr):
    """A user-authored simt scope wrapping the per-program prefix scan.

    Real multi-program grid: every program owns a distinct BT tile
    (T = grid * BT).  The load and store stay outside the scope (the mixed
    route keeps MTE transfers on the SIMD side); only the scan runs under the
    user contract.  The scope must survive mixed routing pinned to the SIMT
    side instead of being billed into and erased by a single-side route.
    """
    offs = tl.program_id(0) * BT + tl.arange(0, BT)
    mask = offs < T
    b_s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    with al.scope(vector_mode="simt"):
        b_o = tl.cumsum(b_s, axis=0)
    tl.store(o_ptr + offs, b_o.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T"])
def _user_simd_scope_kernel(s_ptr, o_ptr, q_ptr, T, BT: tl.constexpr):
    """A user-authored simd scope competing with an automatic simt anchor.

    The scoped scan is pinned to the SIMD side while the second, unscoped scan
    remains an automatic anchor the cost model may route to the SIMT side.
    The pin must hold: the scoped stage may not be flipped to SIMT even when
    the mixed solver needs a SIMT stage elsewhere.
    """
    offs = tl.program_id(0) * BT + tl.arange(0, BT)
    mask = offs < T
    b_s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    with al.scope(vector_mode="simd"):
        b_o = tl.cumsum(b_s, axis=0)
    b_q = tl.cumsum(b_s * 2.0, axis=0)
    tl.store(o_ptr + offs, b_o.to(o_ptr.dtype.element_ty), mask=mask)
    tl.store(q_ptr + offs, b_q.to(q_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T"])
def _loop_nested_simt_scope_kernel(s_ptr, o_ptr, T, BT: tl.constexpr, CPC: tl.constexpr):
    """A user-authored simt scope nested inside a loop-carried recurrence.

    Mirrors the structure of fla ``chunk_global_cumsum_scalar_kernel`` with the
    scan placed under a user simt scope.  Every program owns CPC consecutive
    chunks (T = grid * CPC * BT); the carried scalar accumulates within the
    program while ``tl.cumsum`` scans each chunk.  The scope must become its
    own stage (not the whole loop) and stay pinned to the SIMT side of the
    mixed route.
    """
    pid = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_c in range(CPC):
        offs = (pid * CPC + i_c) * BT + tl.arange(0, BT)
        mask = offs < T
        b_s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        with al.scope(vector_mode="simt"):
            b_o = tl.cumsum(b_s, axis=0) + b_z
        b_z += tl.sum(b_s, axis=0)
        tl.store(o_ptr + offs, b_o.to(o_ptr.dtype.element_ty), mask=mask)


def _last_report(path):
    with open(path, "r", encoding="utf-8") as stream:
        reports = [json.loads(line) for line in stream if line.strip()]
    assert reports, "costmodel route report is empty"
    return reports[-1]


def _pinned_stage_modes(report, pin_key):
    """Modes the mixed route assigned to the stages carrying the user pin."""
    logical_stages = report["stage_model"]["logical_stages"]
    mixed_stages = report["stage_model"]["routes"]["mixed_simd_simt"]["stages"]
    assert len(logical_stages) == len(mixed_stages)
    pinned = [mixed_stages[i]["implementation"]["mode"] for i, stage in enumerate(logical_stages) if stage.get(pin_key)]
    assert pinned, f"no stage reports {pin_key} in the route report"
    return pinned


class UserScopePinnedMixedRouteTest(unittest.TestCase):

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

    def _compile_and_report(self, kernel, grid, args, kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            # The costmodel dumps one JSON line per compile; the directory must exist.
            report_path = os.path.join(tmpdir, "output.json")
            kwargs = dict(kwargs, compile_mode="simd_simt", auto_simt_scope_mode="auto",
                          auto_simt_scope_dump=report_path)
            kernel[grid](*args, **kwargs)
            torch.npu.synchronize()
            self.assertTrue(os.path.exists(report_path), "costmodel route report was not written")
            return _last_report(report_path)

    # Real multi-program grids: every program owns distinct tiles, so wave
    # counts and the superblock factor enter the route decision.

    def test_user_simt_scope_is_pinned_to_simt(self):
        if not is_compile_on_910_95():
            self.skipTest("SIMD/SIMT cost model only supports 910_95")

        programs, block = 8, 128
        total = programs * block
        torch.manual_seed(0)
        source = torch.randn(total, dtype=torch.float32, device="npu")
        output = torch.empty_like(source)

        report = self._compile_and_report(_user_simt_scope_kernel, (programs, ), (source, output, total),
                                          {"BT": block, "num_warps": 4})

        self.assertTrue(report["stage_model"]["applied"])
        self.assertEqual(report["effective_decision_kind"], MIXED_ROUTE)
        self.assertTrue(report["stage_model"]["routes"]["mixed_simd_simt"]["legal"])
        self.assertEqual(set(_pinned_stage_modes(report, "pinned_to_simt")), {"simt"})

        # Each program scans its own tile.
        expected = source.view(programs, block).cumsum(dim=1).flatten()
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_user_simd_scope_is_pinned_to_simd(self):
        if not is_compile_on_910_95():
            self.skipTest("SIMD/SIMT cost model only supports 910_95")

        # A 4096-lane scan makes the SIMD side of the unscoped scan pay the
        # prefix-scan dependency factor, so the solver routes that Stage to
        # SIMT and the mixed route stays genuine (SIMT + SIMD Stages).
        programs, block = 8, 4096
        total = programs * block
        torch.manual_seed(0)
        source = torch.randn(total, dtype=torch.float32, device="npu")
        output = torch.empty_like(source)
        doubled = torch.empty_like(source)

        report = self._compile_and_report(_user_simd_scope_kernel, (programs, ), (source, output, doubled, total),
                                          {"BT": block, "num_warps": 4})

        self.assertTrue(report["stage_model"]["applied"])
        self.assertEqual(report["effective_decision_kind"], MIXED_ROUTE)
        mixed = report["stage_model"]["routes"]["mixed_simd_simt"]
        self.assertTrue(mixed["legal"])
        self.assertEqual(set(_pinned_stage_modes(report, "pinned_to_simd")), {"simd"})
        # The unscoped scan stays routable: the mixed route must still carry a
        # SIMT stage for it, which is what makes this a genuine mixed route.
        modes = {stage["implementation"]["mode"] for stage in mixed["stages"]}
        self.assertIn("simt", modes)

        expected = source.view(programs, block).cumsum(dim=1).flatten()
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)
        expected_doubled = (source * 2.0).view(programs, block).cumsum(dim=1).flatten()
        torch.testing.assert_close(doubled, expected_doubled, rtol=1e-4, atol=1e-4)

    def test_loop_nested_simt_scope_is_pinned_to_simt(self):
        if not is_compile_on_910_95():
            self.skipTest("SIMD/SIMT cost model only supports 910_95")

        programs, block, chunks = 8, 256, 4
        total = programs * block * chunks
        torch.manual_seed(0)
        source = torch.randn(total, dtype=torch.float32, device="npu")
        output = torch.empty_like(source)

        report = self._compile_and_report(_loop_nested_simt_scope_kernel, (programs, ), (source, output, total),
                                          {"BT": block, "CPC": chunks, "num_warps": 4})

        self.assertTrue(report["stage_model"]["applied"])
        self.assertEqual(report["effective_decision_kind"], MIXED_ROUTE)
        self.assertTrue(report["stage_model"]["routes"]["mixed_simd_simt"]["legal"])
        self.assertEqual(set(_pinned_stage_modes(report, "pinned_to_simt")), {"simt"})

        # Each program cumsum-scans its own chunk sequence.
        expected = source.view(programs, chunks * block).cumsum(dim=1).flatten()
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
