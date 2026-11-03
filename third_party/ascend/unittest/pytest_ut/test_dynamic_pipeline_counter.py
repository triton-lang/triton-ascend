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

import re

import pytest
import torch
import torch_npu
import triton
from triton.backends.ascend.utils import is_register_based_arch

# Python range uses a fixed-width integer induction variable. Supply legal
# TTIR explicitly to exercise index loops without depending on the scaled-dot
# lowering from PR #2280. The regular TTIR-to-device pipeline compiles this
# kernel, including DynamicCVPipeline's outer double-buffer polling.
_INDEX_LOOP_TTIR = """
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  tt.func public @dynamic_pipeline_index_counter(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : i32
    %stride = arith.constant dense<32> : tensor<32x1xi32>
    %quarter = arith.constant dense<2.500000e-01> : tensor<32x32xf16>
    %half = arith.constant dense<5.000000e-01> : tensor<32x32xf32>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %rows = tt.expand_dims %range {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %cols = tt.expand_dims %range {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %row_offsets = arith.muli %rows, %stride : tensor<32x1xi32>
    %row_grid = tt.broadcast %row_offsets : tensor<32x1xi32> -> tensor<32x32xi32>
    %col_grid = tt.broadcast %cols : tensor<1x32xi32> -> tensor<32x32xi32>
    %offsets = arith.addi %row_grid, %col_grid : tensor<32x32xi32>
    scf.for %iv = %c0 to %c4 step %c1 {
      %batch = arith.index_cast %iv : index to i32
      %batch_offset = arith.muli %batch, %c1024 : i32
      %a_base = tt.addptr %arg0, %batch_offset : !tt.ptr<f16>, i32
      %b_base = tt.addptr %arg1, %batch_offset : !tt.ptr<f16>, i32
      %out_base = tt.addptr %arg2, %batch_offset : !tt.ptr<f32>, i32
      %a_splat = tt.splat %a_base : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>
      %b_splat = tt.splat %b_base : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>
      %out_splat = tt.splat %out_base : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
      %a_ptrs = tt.addptr %a_splat, %offsets : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
      %b_ptrs = tt.addptr %b_splat, %offsets : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
      %out_ptrs = tt.addptr %out_splat, %offsets : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
      %a = tt.load %a_ptrs : tensor<32x32x!tt.ptr<f16>>
      %b = tt.load %b_ptrs : tensor<32x32x!tt.ptr<f16>>
      %shifted_a = arith.addf %a, %quarter : tensor<32x32xf16>
      %product = tt.dot %shifted_a, %b, %zero : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
      tt.store %out_ptrs, %product : tensor<32x32x!tt.ptr<f32>>
      %c_base = tt.addptr %arg3, %batch_offset : !tt.ptr<f16>, i32
      %d_base = tt.addptr %arg4, %batch_offset : !tt.ptr<f16>, i32
      %out2_base = tt.addptr %arg5, %batch_offset : !tt.ptr<f32>, i32
      %c_splat = tt.splat %c_base : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>
      %d_splat = tt.splat %d_base : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>
      %out2_splat = tt.splat %out2_base : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
      %c_ptrs = tt.addptr %c_splat, %offsets : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
      %d_ptrs = tt.addptr %d_splat, %offsets : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
      %out2_ptrs = tt.addptr %out2_splat, %offsets : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
      %c = tt.load %c_ptrs : tensor<32x32x!tt.ptr<f16>>
      %d = tt.load %d_ptrs : tensor<32x32x!tt.ptr<f16>>
      %product2 = tt.dot %c, %d, %zero : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
      %result = arith.addf %product2, %half : tensor<32x32xf32>
      tt.store %out2_ptrs, %result : tensor<32x32x!tt.ptr<f32>>
    }
    tt.return
  }
}
"""


def test_dynamic_pipeline_index_counter(tmp_path):
    target = triton.runtime.driver.active.get_current_target()
    if not is_register_based_arch(target.arch):
        pytest.skip("Dynamic CV pipeline requires Ascend 950 (A5)")

    source = tmp_path / "dynamic_pipeline_index_counter.ttir"
    # IRSource does not run the Python CodeGenerator that normally supplies
    # hacc.target. Match the fixture's target attribute to this A5 device.
    source.write_text(_INDEX_LOOP_TTIR.replace("Ascend910_9589", target.arch))
    kernel = triton.compile(
        str(source),
        target=target,
        options={"enable_dynamic_cv_pipeline": True, "buf_slot_num_of_crosscore": 2},
    )

    # Independent products require both V-to-C and C-to-V transfers inside the
    # loop. Reject a fallback or a compilation which no longer exercises index
    # buffer rotation.
    assert kernel.metadata.enable_dynamic_cv_pipeline
    ttadapter = kernel.asm["ttadapter"]
    assert "scope.scope" in ttadapter
    index_twos = re.findall(r"(%[\w.$-]+)\s*=\s*arith\.constant[^\n]*?\b2\s*:\s*index\b", ttadapter)
    assert any(
        re.search(rf"arith\.remsi\s+%[\w.$-]+,\s*{re.escape(two)}(?:\s|\{{)[^\n]*:\s*index\b", ttadapter)
        for two in index_twos), "expected index-typed modulo-two buffer polling"

    # Small binary fractions make the input adjustment, FP32 matmuls and final
    # addition exact. Four distinct batches exercise both buffer slots twice.
    elements = torch.arange(4 * 32 * 32).reshape(4, 32, 32)
    a = ((elements * 3 % 17 - 8).float() / 4).half()
    b = ((elements * 5 % 13 - 6).float() / 8).half()
    c = ((elements * 7 % 11 - 5).float() / 4).half()
    d = ((elements * 9 % 15 - 7).float() / 8).half()
    expected_v2c = (a.float() + 0.25) @ b.float()
    expected_c2v = c.float() @ d.float() + 0.5
    a_device = a.to("npu")
    b_device = b.to("npu")
    c_device = c.to("npu")
    d_device = d.to("npu")
    actual_v2c = torch.full(expected_v2c.shape, float("nan"), dtype=torch.float32, device="npu")
    actual_c2v = torch.full_like(actual_v2c, float("nan"))
    # IRSource reports MLIR type names; the NPU launcher uses Triton names.
    # Normalize only this spelling at the test's launch boundary. The compiled
    # IR, pointer element types and tensor storage stay unchanged.
    assert kernel.src.signature == {0: "*f16", 1: "*f16", 2: "*f32", 3: "*f16", 4: "*f16", 5: "*f32"}
    launch_types = {"*f16": "*fp16", "*f32": "*fp32"}
    kernel.src.signature = {index: launch_types[dtype] for index, dtype in kernel.src.signature.items()}
    kernel[(1, 1, 1)](a_device, b_device, actual_v2c, c_device, d_device, actual_c2v)
    torch.testing.assert_close(actual_v2c.cpu(), expected_v2c, atol=0, rtol=0)
    torch.testing.assert_close(actual_c2v.cpu(), expected_c2v, atol=0, rtol=0)
