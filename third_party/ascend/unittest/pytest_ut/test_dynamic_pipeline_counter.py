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

# Python range uses fixed-width induction variables. Explicit TTIR keeps the
# outer index loop independent of scaled-dot lowering. Both loop bounds are
# runtime arguments: an i32 loop-carried phase controls a vector-side branch,
# while the CUBE scope contains a nested FP32 matmul accumulation loop.
_INDEX_LOOP_TTIR = """
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  tt.func public @dynamic_pipeline_index_counter(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg6: i32, %arg7: i32, %arg8: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %limit = arith.index_cast %arg6 : i32 to index
    %repeat = arith.index_cast %arg8 : i32 to index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %plus = arith.constant 2.500000e-01 : f16
    %minus = arith.constant -2.500000e-01 : f16
    %c1024 = arith.constant 1024 : i32
    %stride = arith.constant dense<32> : tensor<32x1xi32>
    %half = arith.constant dense<5.000000e-01> : tensor<32x32xf32>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %rows = tt.expand_dims %range {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %cols = tt.expand_dims %range {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %row_offsets = arith.muli %rows, %stride : tensor<32x1xi32>
    %row_grid = tt.broadcast %row_offsets : tensor<32x1xi32> -> tensor<32x32xi32>
    %col_grid = tt.broadcast %cols : tensor<1x32xi32> -> tensor<32x32xi32>
    %offsets = arith.addi %row_grid, %col_grid : tensor<32x32xi32>
    %finished = scf.for %iv = %c0 to %limit step %c1 iter_args(%state = %arg7) -> (i32) {
      %parity = arith.remsi %state, %c2_i32 : i32
      %even = arith.cmpi eq, %parity, %c0_i32 : i32
      %adjustment = scf.if %even -> (f16) {
        scf.yield %plus : f16
      } else {
        scf.yield %minus : f16
      }
      %delta = tt.splat %adjustment : f16 -> tensor<32x32xf16>
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
      %shifted_a = arith.addf %a, %delta : tensor<32x32xf16>
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
      %product2 = scf.for %inner = %c0 to %repeat step %c1 iter_args(%acc = %zero) -> (tensor<32x32xf32>) {
        %cd = tt.dot %c, %d, %acc : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
        scf.yield %cd : tensor<32x32xf32>
      }
      %result = arith.addf %product2, %half : tensor<32x32xf32>
      tt.store %out2_ptrs, %result : tensor<32x32x!tt.ptr<f32>>
      %next_state = arith.addi %state, %c1_i32 : i32
      scf.yield %next_state : i32
    }
    tt.return
  }
}
"""


@pytest.mark.parametrize("iterations,phase,repeats", [
    (1, 0, 1),
    (1, 1, 3),
    (3, 0, 2),
    (4, 1, 3),
    (5, 0, 1),
    (6, 1, 2),
])
def test_dynamic_pipeline_index_counter(tmp_path, iterations, phase, repeats):
    target = triton.runtime.driver.active.get_current_target()
    if not is_register_based_arch(target.arch):
        pytest.skip("Dynamic CV pipeline requires Ascend 950 (A5)")

    source = tmp_path / "dynamic_pipeline_index_counter.ttir"
    source.write_text(_INDEX_LOOP_TTIR.replace("Ascend910_9589", target.arch))
    kernel = triton.compile(
        str(source),
        target=target,
        options={"enable_dynamic_cv_pipeline": True, "buf_slot_num_of_crosscore": 2},
    )

    # AnalyzeDataFlow must accept the runtime loop bounds and carried state.
    # A fallback would otherwise hide the counter regression. Keep the nested
    # CUBE accumulation loop and index-typed double-buffer polling observable.
    assert kernel.metadata.enable_dynamic_cv_pipeline
    ttadapter = kernel.asm["ttadapter"]
    assert "triton_ascend.dynamic_cv_pipeline.rc" not in ttadapter
    assert "#hivm.tcore_type<VECTOR>" in ttadapter
    assert "#hivm.tcore_type<CUBE>" in ttadapter
    assert re.search(r"scf\.for[^\n]*iter_args[^\n]*-> \(tensor<32x32xf32>\)", ttadapter)
    index_twos = re.findall(r"(%[\w.$-]+)\s*=\s*arith\.constant[^\n]*?\b2\s*:\s*index\b", ttadapter)
    assert any(
        re.search(rf"arith\.remsi\s+%[\w.$-]+,\s*{re.escape(two)}(?:\s|\{{)[^\n]*:\s*index\b", ttadapter)
        for two in index_twos), "expected index-typed modulo-two buffer polling"

    # Distinct binary-fraction inputs make both branches and all FP32 sums
    # exact. Odd/even trip counts rotate the buffer slots repeatedly; two
    # untouched output batches detect incorrect polling or loop termination.
    storage_batches = iterations + 2
    elements = torch.arange(storage_batches * 32 * 32).reshape(storage_batches, 32, 32)
    a = ((elements * 3 % 17 - 8).float() / 4).half()
    b = ((elements * 5 % 13 - 6).float() / 8).half()
    c = ((elements * 7 % 11 - 5).float() / 4).half()
    d = ((elements * 9 % 15 - 7).float() / 8).half()
    expected_v2c = torch.full((storage_batches, 32, 32), -4096.0)
    expected_c2v = torch.full((storage_batches, 32, 32), -8192.0)
    even = (torch.arange(iterations) + phase) % 2 == 0
    adjustment = torch.where(even, 0.25, -0.25).reshape(iterations, 1, 1)
    expected_v2c[:iterations] = (a[:iterations].float() + adjustment) @ b[:iterations].float()
    product = c[:iterations].float() @ d[:iterations].float()
    accumulated = torch.zeros_like(product)
    for _ in range(repeats):
        accumulated += product
    expected_c2v[:iterations] = accumulated + 0.5

    a_device = a.to("npu")
    b_device = b.to("npu")
    c_device = c.to("npu")
    d_device = d.to("npu")
    actual_v2c = torch.full_like(a_device, -4096.0, dtype=torch.float32)
    actual_c2v = torch.full_like(actual_v2c, -8192.0)
    # Normalize IRSource's MLIR pointer spellings only at the launcher boundary.
    assert kernel.src.signature == {
        0: "*f16",
        1: "*f16",
        2: "*f32",
        3: "*f16",
        4: "*f16",
        5: "*f32",
        6: "i32",
        7: "i32",
        8: "i32",
    }
    launch_types = {"*f16": "*fp16", "*f32": "*fp32"}
    kernel.src.signature = {index: launch_types.get(dtype, dtype) for index, dtype in kernel.src.signature.items()}
    kernel[(1, 1, 1)](a_device, b_device, actual_v2c, c_device, d_device, actual_c2v, iterations, phase, repeats)
    torch.testing.assert_close(actual_v2c.cpu(), expected_v2c, atol=0, rtol=0)
    torch.testing.assert_close(actual_c2v.cpu(), expected_c2v, atol=0, rtol=0)
