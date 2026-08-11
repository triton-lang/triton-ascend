# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

from triton.backends.ascend.utils import _get_triton_opt_path


def _find_triton_opt():
    configured = os.environ.get("TRITON_OPT")
    if configured:
        path = Path(configured)
        if path.is_file():
            return str(path)
        pytest.fail(f"TRITON_OPT does not point to a file: {configured}")
    return _get_triton_opt_path()


def _run_ttolinalg(tmp_path, name, module):
    source = tmp_path / f"{name}.mlir"
    source.write_text(textwrap.dedent(module), encoding="utf-8")
    return subprocess.run(
        [_find_triton_opt(), "--triton-to-linalg=named-ops=True",
         str(source)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _run_unstructure(tmp_path, name, module):
    source = tmp_path / f"{name}.mlir"
    source.write_text(textwrap.dedent(module), encoding="utf-8")
    return subprocess.run(
        [
            _find_triton_opt(),
            "--triton-to-unstructure",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _run_discrete_mask_conversion(tmp_path, name, module):
    source = tmp_path / f"{name}.mlir"
    source.write_text(textwrap.dedent(module), encoding="utf-8")
    return subprocess.run(
        [
            _find_triton_opt(),
            "--discrete-mask-access-conversion",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _run_pipeline(tmp_path, name, module):
    source = tmp_path / f"{name}.mlir"
    source.write_text(textwrap.dedent(module), encoding="utf-8")
    return subprocess.run(
        [
            _find_triton_opt(),
            "--triton-to-unstructure",
            "--triton-to-linalg=named-ops=True",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _assert_clean_failure(result, expected):
    diagnostics = result.stdout + result.stderr
    assert result.returncode > 0, diagnostics
    assert expected in diagnostics
    assert "Assertion" not in diagnostics
    assert "PLEASE submit a bug report" not in diagnostics
    assert "Stack dump" not in diagnostics
    return diagnostics


def _require_same_atomic_lane(output, minimum_uses):
    lane_indices = re.findall(r"tensor\.extract [^\n]*\[(%[^\]\s]+)\]", output)
    assert any(lane_indices.count(iv) >= minimum_uses for iv in set(lane_indices)), output


def test_dynamic_scalar_offset_materializes_exact_byte_address(tmp_path):
    result = _run_ttolinalg(
        tmp_path, "dynamic_ratio4", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @dynamic_ratio4(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %byte_offset: i32) attributes {noinline = false} {
            %byte_ptr = tt.addptr %src, %byte_offset : !tt.ptr<i8>, i32
            %wide_ptr = tt.bitcast %byte_ptr : !tt.ptr<i8> -> !tt.ptr<i32>
            %value = tt.load %wide_ptr : !tt.ptr<i32>
            tt.store %dst, %value : !tt.ptr<i32>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "arith.remsi" not in output, output
    assert "triton_assert" not in output, output
    assert "arith.divsi" not in output, output
    assert "arith.muli" in output, output
    assert "arith.addi" in output, output
    assert "hivm.hir.pointer_cast" in output, output


def test_tensor_byte_offsets_materialize_without_runtime_check(tmp_path):
    result = _run_pipeline(
        tmp_path, "tensor_byte_offsets", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @tensor_byte_offsets(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %four = arith.constant dense<4> : tensor<4xi32>
            %offsets = arith.muli %range, %four : tensor<4xi32>
            %srcs = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %srcs, %offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %values = tt.load %wide_ptrs : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %dst_ptrs = tt.addptr %dsts, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "arith.remsi" not in result.stdout, result.stdout
    assert "triton_assert" not in result.stdout, result.stdout

    assert "builtin.unrealized_conversion_cast" not in result.stdout
    assert "hivm.hir.pointer_cast" in result.stdout


def test_byte_addressed_discrete_mask_load_keeps_scalar_mask_and_other(tmp_path):
    result = _run_unstructure(
        tmp_path, "byte_addressed_discrete_mask_load", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @byte_addressed_discrete_mask_load(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %four = arith.constant dense<4> : tensor<4xi32>
            %byte_offsets = arith.muli %range, %four : tensor<4xi32>
            %bytes = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %bytes, %byte_offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %two = arith.constant dense<2> : tensor<4xi32>
            %zero = arith.constant dense<0> : tensor<4xi32>
            %remainder = arith.remsi %range, %two : tensor<4xi32>
            %mask = arith.cmpi eq, %remainder, %zero : tensor<4xi32>
            %other = arith.constant dense<-7> : tensor<4xi32>
            %values = tt.load %wide_ptrs, %mask, %other
                : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %dst_ptrs = tt.addptr %dsts, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    scalar_loads = [line for line in output.splitlines() if "tt.load" in line and ": !tt.ptr<i32>" in line]
    assert scalar_loads, output
    assert all(line.count(",") >= 2 for line in scalar_loads), output
    assert "tt.pointer_bitcast_pointer_cast" in output, output
    # The constant other may fold directly to a scalar constant in each lane.
    assert re.search(r"tensor\.extract .*tensor<4xi1>", output), output


def test_byte_addressed_discrete_mask_load_without_other_keeps_scalar_mask(tmp_path):
    result = _run_unstructure(
        tmp_path, "byte_addressed_mask_load_without_other", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @byte_addressed_mask_load_without_other(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %four = arith.constant dense<4> : tensor<4xi32>
            %byte_offsets = arith.muli %range, %four : tensor<4xi32>
            %bytes = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %bytes, %byte_offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %two = arith.constant dense<2> : tensor<4xi32>
            %zero = arith.constant dense<0> : tensor<4xi32>
            %remainder = arith.remsi %range, %two : tensor<4xi32>
            %mask = arith.cmpi eq, %remainder, %zero : tensor<4xi32>
            %values = tt.load %wide_ptrs, %mask
                : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %dst_ptrs = tt.addptr %dsts, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    scalar_loads = [line for line in output.splitlines() if "tt.load" in line and ": !tt.ptr<i32>" in line]
    assert scalar_loads, output
    assert all(line.count(",") >= 1 for line in scalar_loads), output
    assert re.search(r"tensor\.extract .*tensor<4xi1>", output), output


def test_different_width_pointer_bitcast_discrete_mask_load_is_rejected(tmp_path):
    result = _run_discrete_mask_conversion(
        tmp_path, "pointer_bitcast_discrete_mask_load", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @pointer_bitcast_discrete_mask_load(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %four = arith.constant dense<4> : tensor<4xi32>
            %byte_offsets = arith.muli %range, %four : tensor<4xi32>
            %bytes = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %bytes, %byte_offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %two = arith.constant dense<2> : tensor<4xi32>
            %zero = arith.constant dense<0> : tensor<4xi32>
            %remainder = arith.remsi %range, %two : tensor<4xi32>
            %mask = arith.cmpi eq, %remainder, %zero : tensor<4xi32>
            %other = arith.constant dense<-7> : tensor<4xi32>
            %values = tt.load %wide_ptrs, %mask, %other
                : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %dst_ptrs = tt.addptr %dsts, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    _assert_clean_failure(
        result,
        "different-width pointer bitcast does not support discrete masked "
        "memory access",
    )


def test_control_flow_pointer_bitcast_discrete_mask_load_is_rejected(tmp_path):
    result = _run_discrete_mask_conversion(
        tmp_path, "control_flow_pointer_bitcast_discrete_mask_load", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @control_flow_pointer_bitcast_discrete_mask_load(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %cond: i1) attributes {noinline = false} {
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %four = arith.constant dense<4> : tensor<4xi32>
            %byte_offsets = arith.muli %range, %four : tensor<4xi32>
            %bytes = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %bytes, %byte_offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %selected = scf.if %cond -> (tensor<4x!tt.ptr<i32>>) {
              scf.yield %wide_ptrs : tensor<4x!tt.ptr<i32>>
            } else {
              %else_ptrs = tt.addptr %wide_ptrs, %range
                  : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
              scf.yield %else_ptrs : tensor<4x!tt.ptr<i32>>
            }
            %two = arith.constant dense<2> : tensor<4xi32>
            %zero = arith.constant dense<0> : tensor<4xi32>
            %remainder = arith.remsi %range, %two : tensor<4xi32>
            %mask = arith.cmpi eq, %remainder, %zero : tensor<4xi32>
            %other = arith.constant dense<-7> : tensor<4xi32>
            %values = tt.load %selected, %mask, %other
                : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %dst_ptrs = tt.addptr %dsts, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    _assert_clean_failure(
        result,
        "different-width pointer bitcast does not support discrete masked "
        "memory access",
    )


def test_different_width_pointer_bitcast_discrete_mask_store_is_rejected(tmp_path):
    result = _run_discrete_mask_conversion(
        tmp_path, "pointer_bitcast_discrete_mask_store", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @pointer_bitcast_discrete_mask_store(
              %dst: !tt.ptr<i8> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %four = arith.constant dense<4> : tensor<4xi32>
            %byte_offsets = arith.muli %range, %four : tensor<4xi32>
            %bytes = tt.splat %dst
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %bytes, %byte_offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %two = arith.constant dense<2> : tensor<4xi32>
            %zero = arith.constant dense<0> : tensor<4xi32>
            %remainder = arith.remsi %range, %two : tensor<4xi32>
            %mask = arith.cmpi eq, %remainder, %zero : tensor<4xi32>
            %values = arith.constant dense<[3, 5, 7, 11]>
                : tensor<4xi32>
            tt.store %wide_ptrs, %values, %mask
                : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    _assert_clean_failure(
        result,
        "different-width pointer bitcast does not support discrete masked "
        "memory access",
    )


def test_different_width_pointer_bitcast_discrete_mask_atomic_is_rejected(tmp_path):
    result = _run_discrete_mask_conversion(
        tmp_path, "pointer_bitcast_discrete_mask_atomic", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @pointer_bitcast_discrete_mask_atomic(
              %dst: !tt.ptr<i8> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %four = arith.constant dense<4> : tensor<4xi32>
            %byte_offsets = arith.muli %range, %four : tensor<4xi32>
            %bytes = tt.splat %dst
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %bytes, %byte_offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %two = arith.constant dense<2> : tensor<4xi32>
            %zero = arith.constant dense<0> : tensor<4xi32>
            %remainder = arith.remsi %range, %two : tensor<4xi32>
            %mask = arith.cmpi eq, %remainder, %zero : tensor<4xi32>
            %values = arith.constant dense<[3, 5, 7, 11]>
                : tensor<4xi32>
            %old = tt.atomic_rmw add, acq_rel, gpu, %wide_ptrs, %values, %mask
                : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>)
                    -> tensor<4xi32>
            tt.return
          }
        }
    """)
    _assert_clean_failure(
        result,
        "different-width pointer bitcast does not support discrete masked "
        "memory access",
    )


def test_different_width_pointer_bitcast_contiguous_mask_is_supported(tmp_path):
    result = _run_discrete_mask_conversion(
        tmp_path, "pointer_bitcast_contiguous_mask_load", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @pointer_bitcast_contiguous_mask_load(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %four = arith.constant dense<4> : tensor<4xi32>
            %byte_offsets = arith.muli %range, %four : tensor<4xi32>
            %bytes = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %bytes, %byte_offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %three = arith.constant dense<3> : tensor<4xi32>
            %mask = arith.cmpi slt, %range, %three : tensor<4xi32>
            %other = arith.constant dense<-7> : tensor<4xi32>
            %values = tt.load %wide_ptrs, %mask, %other
                : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %dst_ptrs = tt.addptr %dsts, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "tt.bitcast" in output, output
    assert "tt.load" in output, output
    assert "arith.select" not in output, output


def test_non_bitcast_discrete_mask_load_keeps_legacy_conversion(tmp_path):
    result = _run_discrete_mask_conversion(
        tmp_path, "legacy_discrete_mask_load", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @legacy_discrete_mask_load(
              %src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %cond: i1)
              attributes {noinline = false} {
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %srcs = tt.splat %src
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %src_ptrs = tt.addptr %srcs, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            %selected = scf.if %cond -> (tensor<4x!tt.ptr<i32>>) {
              scf.yield %srcs : tensor<4x!tt.ptr<i32>>
            } else {
              scf.yield %src_ptrs : tensor<4x!tt.ptr<i32>>
            }
            %two = arith.constant dense<2> : tensor<4xi32>
            %zero = arith.constant dense<0> : tensor<4xi32>
            %remainder = arith.remsi %range, %two : tensor<4xi32>
            %mask = arith.cmpi eq, %remainder, %zero : tensor<4xi32>
            %other = arith.constant dense<-7> : tensor<4xi32>
            %values = tt.load %selected, %mask, %other
                : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %dst_ptrs = tt.addptr %dsts, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "arith.select" in output, output
    assert "tt.load" in output, output
    assert "tt.bitcast" not in output, output
    assert "tt.pointer_bitcast_pointer_cast" not in output, output


def test_pointer_bitcast_mask_guards_the_physical_scalar_load(tmp_path):
    result = _run_pipeline(
        tmp_path, "pointer_bitcast_guarded_scalar_load", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @pointer_bitcast_guarded_scalar_load(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %mask: i1, %fallback: i32) attributes {noinline = false} {
            %wide_ptr = tt.bitcast %src : !tt.ptr<i8> -> !tt.ptr<i32>
            %one = arith.constant 1 : i32
            %other = arith.addi %fallback, %one : i32
            %value = tt.load %wide_ptr, %mask, %other : !tt.ptr<i32>
            tt.store %dst, %value : !tt.ptr<i32>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "scf.if" in output, output
    assert "memref.load" in output, output
    assert output.index("scf.if") < output.index("memref.load"), output


def test_byte_addressed_widening_store_scalarizes_value_and_mask(tmp_path):
    result = _run_pipeline(
        tmp_path, "byte_addressed_widening_store", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @byte_addressed_widening_store(
              %src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i8> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %srcs = tt.splat %src
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %src_ptrs = tt.addptr %srcs, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            %values = tt.load %src_ptrs : tensor<4x!tt.ptr<i32>>
            %four = arith.constant dense<4> : tensor<4xi32>
            %byte_offsets = arith.muli %range, %four : tensor<4xi32>
            %bytes = tt.splat %dst
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %bytes, %byte_offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %three = arith.constant dense<3> : tensor<4xi32>
            %mask = arith.cmpi slt, %range, %three
                : tensor<4xi32>
            tt.store %wide_ptrs, %values, %mask
                : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "scf.for" in output, output
    assert "arith.addi" in output, output
    assert "hivm.hir.pointer_cast" in output, output
    assert "scf.if" in output, output
    assert ("memref.store" in output or "bufferization.materialize_in_destination" in output), output
    assert "tensor.extract_slice" not in output, output
    assert "builtin.unrealized_conversion_cast" not in output, output
    assert re.search(r"tensor\.extract .*tensor<4xi64>", output), output
    assert re.search(r"tensor\.extract .*tensor<4xi32>", output), output
    assert re.search(r"tensor\.extract .*tensor<4xi1>", output), output
    lane_indices = re.findall(r"tensor\.extract [^\n]*\[(%[^\]\s]+)\]", output)
    assert any(lane_indices.count(iv) >= 3 for iv in set(lane_indices)), output


def test_widening_atomic_rmw_scalarizes_value_and_mask(tmp_path):
    result = _run_unstructure(
        tmp_path, "widening_atomic_rmw", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @widening_atomic_rmw(
              %dst: !tt.ptr<i8> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %offsets = arith.constant dense<[0, 4, 8, 12]>
                : tensor<4xi32>
            %values = arith.constant dense<[3, 7, 11, 13]>
                : tensor<4xi32>
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %three = arith.constant dense<3> : tensor<4xi32>
            %mask = arith.cmpi slt, %range, %three : tensor<4xi32>
            %bytes = tt.splat %dst
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %bytes, %offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %old = tt.atomic_rmw add, acq_rel, gpu, %wide_ptrs, %values, %mask
                : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>)
                    -> tensor<4xi32>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "__builtin_indirect_atomic" not in output, output
    assert "tt.atomic_rmw add" in output, output
    assert re.search(r"tensor\.extract .*tensor<4xi1>", output), output
    _require_same_atomic_lane(output, 3)


def test_widening_atomic_cas_scalarizes_compare_value_and_result(tmp_path):
    result = _run_unstructure(
        tmp_path, "widening_atomic_cas", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @widening_atomic_cas(
              %dst: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %old_dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %offsets = arith.constant dense<[0, 4, 8, 12]>
                : tensor<4xi32>
            %compare = arith.constant dense<[10, 99, 30, 77]>
                : tensor<4xi32>
            %values = arith.constant dense<[101, 202, 303, 404]>
                : tensor<4xi32>
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %bytes = tt.splat %dst
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %bytes, %offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %wide_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %old = tt.atomic_cas acq_rel, gpu, %wide_ptrs, %compare, %values
                : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi32>)
                    -> tensor<4xi32>
            %old_dsts = tt.splat %old_dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %old_ptrs = tt.addptr %old_dsts, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %old_ptrs, %old : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "__builtin_indirect_atomic" not in output, output
    assert "tt.atomic_cas" in output, output
    _require_same_atomic_lane(output, 3)


def test_static_pre_and_post_bitcast_offsets_keep_their_units(tmp_path):
    result = _run_pipeline(
        tmp_path, "static_pre_post_offsets", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @static_pre_post_offsets(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %four_bytes = arith.constant 4 : i32
            %two_elements = arith.constant 2 : i32
            %byte_ptr = tt.addptr %src, %four_bytes : !tt.ptr<i8>, i32
            %wide_ptr = tt.bitcast %byte_ptr
                : !tt.ptr<i8> -> !tt.ptr<i32>
            %final_ptr = tt.addptr %wide_ptr, %two_elements
                : !tt.ptr<i32>, i32
            %value = tt.load %final_ptr : !tt.ptr<i32>
            tt.store %dst, %value : !tt.ptr<i32>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "arith.remsi" not in output, output
    assert "triton_assert" not in output, output
    assert "arith.divsi" not in output, output
    assert "hivm.hir.pointer_cast" in output, output


def test_same_width_pointer_bitcast_preserves_existing_path(tmp_path):
    result = _run_pipeline(
        tmp_path, "same_width", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @same_width(
              %src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %offset_src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %one = arith.constant 1 : i32
            %source_ptr = tt.addptr %src, %one : !tt.ptr<i32>, i32
            %float_ptr = tt.bitcast %source_ptr
                : !tt.ptr<i32> -> !tt.ptr<f32>
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %offset_bases = tt.splat %offset_src
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %offset_ptrs = tt.addptr %offset_bases, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            %offsets = tt.load %offset_ptrs : tensor<4x!tt.ptr<i32>>
            %float_bases = tt.splat %float_ptr
                : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
            %float_ptrs = tt.addptr %float_bases, %offsets
                : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
            %values = tt.load %float_ptrs : tensor<4x!tt.ptr<f32>>
            %dst_bases = tt.splat %dst
                : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
            %dst_ptrs = tt.addptr %dst_bases, %range
                : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
            tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<f32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "arith.remsi" not in result.stdout, result.stdout
    assert "triton_assert" not in result.stdout, result.stdout
    assert "arith.divsi" not in result.stdout, result.stdout
    assert "builtin.unrealized_conversion_cast" not in result.stdout


def test_same_width_bitcast_keeps_dynamic_make_block_ptr_shape(tmp_path):
    result = _run_pipeline(
        tmp_path, "same_width_dynamic_block_shape", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @same_width_dynamic_block_shape(
              %src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %rows: i64, %cols: i64, %row_stride: i64)
              attributes {noinline = false} {
            %c0_i32 = arith.constant 0 : i32
            %c1_i64 = arith.constant 1 : i64
            %float_src = tt.bitcast %src : !tt.ptr<i32> -> !tt.ptr<f32>
            %block_ptr = tt.make_tensor_ptr %float_src, [%rows, %cols],
                [%row_stride, %c1_i64], [%c0_i32, %c0_i32]
                {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
            %values = tt.load %block_ptr : !tt.ptr<tensor<4x8xf32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "PointerCastOp requires statically known access sizes" not in (result.stdout + result.stderr)


def test_user_int_to_ptr_does_not_gain_pointer_bitcast_provenance(tmp_path):
    module = r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @user_int_to_ptr(
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %value: i32,
              %cond: i1) attributes {noinline = false} {
            %address = tt.ptr_to_int %dst : !tt.ptr<i32> -> i64
            %restored = tt.int_to_ptr %address : i64 -> !tt.ptr<i32>
            tt.store %restored, %value, %cond : !tt.ptr<i32>
            tt.return
          }
        }
    """
    unstructure_result = _run_unstructure(tmp_path, "user_int_to_ptr", module)
    assert unstructure_result.returncode == 0, unstructure_result.stderr
    assert "tt.int_to_ptr" in unstructure_result.stdout
    assert "tt.pointer_bitcast_pointer_cast" not in unstructure_result.stdout

    result = _run_ttolinalg(tmp_path, "user_int_to_ptr_ttolinalg", module)
    assert result.returncode == 0, result.stderr
    assert "tt.pointer_bitcast_pointer_cast" not in result.stdout


def test_non_byte_addressable_pointee_is_rejected(tmp_path):
    result = _run_ttolinalg(
        tmp_path, "non_byte_addressable", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @non_byte_addressable(
              %src: !tt.ptr<i4> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i16> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %wide_ptr = tt.bitcast %src : !tt.ptr<i4> -> !tt.ptr<i16>
            %value = tt.load %wide_ptr : !tt.ptr<i16>
            tt.store %dst, %value : !tt.ptr<i16>
            tt.return
          }
        }
    """)
    _assert_clean_failure(result, "byte-addressable")


def test_direct_function_argument_different_width_bitcast(tmp_path):
    result = _run_ttolinalg(
        tmp_path, "direct_function_argument", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @direct_function_argument(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %wide_ptr = tt.bitcast %src : !tt.ptr<i8> -> !tt.ptr<i32>
            %value = tt.load %wide_ptr : !tt.ptr<i32>
            tt.store %dst, %value : !tt.ptr<i32>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "arith.remsi" not in result.stdout, result.stdout
    assert "triton_assert" not in result.stdout, result.stdout
    assert "arith.divsi" not in result.stdout, result.stdout


def test_splat_tensor_pointer_bitcast_is_canonicalized(tmp_path):
    result = _run_pipeline(
        tmp_path, "splat_tensor_pointer", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @splat_tensor_pointer(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %srcs = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %wide_ptrs = tt.bitcast %srcs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %load_offsets = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %zero = arith.constant dense<0> : tensor<4xi32>
            %load_ptrs = tt.addptr %wide_ptrs, %load_offsets
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            %values = tt.load %load_ptrs : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %store_ptrs = tt.addptr %dsts, %zero
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %store_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "arith.remsi" not in result.stdout, result.stdout
    assert "triton_assert" not in result.stdout, result.stdout


def test_nested_pointer_bitcasts_are_fused(tmp_path):
    result = _run_ttolinalg(
        tmp_path, "nested_pointer_bitcasts", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @nested_pointer_bitcasts(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %half_ptr = tt.bitcast %src : !tt.ptr<i8> -> !tt.ptr<i16>
            %word_ptr = tt.bitcast %half_ptr : !tt.ptr<i16> -> !tt.ptr<i32>
            %value = tt.load %word_ptr : !tt.ptr<i32>
            tt.store %dst, %value : !tt.ptr<i32>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "arith.remsi" not in result.stdout, result.stdout
    assert "triton_assert" not in result.stdout, result.stdout


def test_legacy_i1_to_i8_pointer_bitcast_is_unchanged(tmp_path):
    result = _run_pipeline(
        tmp_path, "legacy_i1_to_i8", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @legacy_i1_to_i8(
              %src: !tt.ptr<i1> {tt.divisibility = 16 : i32},
              %offset_src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i8> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %byte_ptr = tt.bitcast %src : !tt.ptr<i1> -> !tt.ptr<i8>
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %offset_bases = tt.splat %offset_src
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %offset_ptrs = tt.addptr %offset_bases, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            %offsets = tt.load %offset_ptrs : tensor<4x!tt.ptr<i32>>
            %byte_bases = tt.splat %byte_ptr
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %byte_bases, %offsets
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %values = tt.load %byte_ptrs : tensor<4x!tt.ptr<i8>>
            %dst_bases = tt.splat %dst
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %dst_ptrs = tt.addptr %dst_bases, %range
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<i8>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "arith.remsi" not in result.stdout, result.stdout
    assert "triton_assert" not in result.stdout, result.stdout
    assert "arith.divsi" not in result.stdout, result.stdout
    assert "builtin.unrealized_conversion_cast" not in result.stdout


def test_pointer_bitcast_from_i8_to_i1_is_cleanly_rejected(tmp_path):
    result = _run_ttolinalg(
        tmp_path, "i8_to_i1", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @i8_to_i1(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i1> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %bool_ptr = tt.bitcast %src : !tt.ptr<i8> -> !tt.ptr<i1>
            %value = tt.load %bool_ptr : !tt.ptr<i1>
            tt.store %dst, %value : !tt.ptr<i1>
            tt.return
          }
        }
    """)
    _assert_clean_failure(
        result,
        "pointer bitcast to i1 from a different pointee type is unsupported",
    )


def test_tensor_pointer_bitcast_from_i8_to_i1_is_cleanly_rejected(tmp_path):
    result = _run_unstructure(
        tmp_path, "tensor_i8_to_i1", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @tensor_i8_to_i1(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i8> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %srcs = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %byte_ptrs = tt.addptr %srcs, %range
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            %bool_ptrs = tt.bitcast %byte_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i1>>
            %values = tt.load %bool_ptrs : tensor<4x!tt.ptr<i1>>
            %byte_values = arith.extui %values
                : tensor<4xi1> to tensor<4xi8>
            %dsts = tt.splat %dst
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %dst_ptrs = tt.addptr %dsts, %range
                : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
            tt.store %dst_ptrs, %byte_values : tensor<4x!tt.ptr<i8>>
            tt.return
          }
        }
    """)
    _assert_clean_failure(
        result,
        "pointer bitcast to i1 from a different pointee type is unsupported",
    )


def test_address_space_mismatch_is_rejected(tmp_path):
    result = _run_ttolinalg(
        tmp_path, "address_space_mismatch", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @address_space_mismatch(
              %src: !tt.ptr<i8, 0> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32, 1> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %wide_ptr = tt.bitcast %src : !tt.ptr<i8, 0> -> !tt.ptr<i32, 1>
            %value = tt.load %wide_ptr : !tt.ptr<i32, 1>
            tt.store %dst, %value : !tt.ptr<i32, 1>
            tt.return
          }
        }
    """)
    _assert_clean_failure(result, "different address spaces")


def test_tensor_pointer_argument_without_scalar_root_is_rejected(tmp_path):
    result = _run_unstructure(
        tmp_path, "tensor_argument_without_root", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @tensor_argument_without_root(
              %srcs: tensor<4x!tt.ptr<i8>>,
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %wide_ptrs = tt.bitcast %srcs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %zero = arith.constant dense<0> : tensor<4xi32>
            %load_ptrs = tt.addptr %wide_ptrs, %zero
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            %values = tt.load %load_ptrs : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %store_ptrs = tt.addptr %dsts, %zero
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %store_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    _assert_clean_failure(result, "requires a scalar root pointer")


def test_byte_addressed_pointer_captured_inside_if_is_supported(tmp_path):
    result = _run_pipeline(
        tmp_path, "byte_addressed_pointer_if_capture", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @byte_addressed_pointer_if_capture(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %cond: i1) attributes {noinline = false} {
            %wide_ptr = tt.bitcast %src : !tt.ptr<i8> -> !tt.ptr<i32>
            %value = scf.if %cond -> (i32) {
              %then_value = tt.load %wide_ptr : !tt.ptr<i32>
              scf.yield %then_value : i32
            } else {
              %one = arith.constant 1 : i32
              %else_ptr = tt.addptr %wide_ptr, %one : !tt.ptr<i32>, i32
              %else_value = tt.load %else_ptr : !tt.ptr<i32>
              scf.yield %else_value : i32
            }
            tt.store %dst, %value : !tt.ptr<i32>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "scf.if" in result.stdout, result.stdout


def test_legacy_pointer_control_flow_without_bitcast_is_unchanged(tmp_path):
    result = _run_unstructure(
        tmp_path, "legacy_pointer_control_flow", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @legacy_pointer_control_flow(
              %src: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %upper: i32)
              attributes {noinline = false} {
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %range = tt.make_range {start = 0 : i32, end = 4 : i32}
                : tensor<4xi32>
            %base = tt.splat %src
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %initial = tt.addptr %base, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            %result = scf.for %iv = %c0 to %upper step %c1
                iter_args(%current = %initial)
                -> (tensor<4x!tt.ptr<i32>>) : i32 {
              %next = tt.addptr %current, %range
                  : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
              scf.yield %next : tensor<4x!tt.ptr<i32>>
            }
            %values = tt.load %result : tensor<4x!tt.ptr<i32>>
            %dsts = tt.splat %dst
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %dst_ptrs = tt.addptr %dsts, %range
                : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
            tt.store %dst_ptrs, %values : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    assert result.returncode == 0, result.stderr
    assert "scf.for" in result.stdout, result.stdout
    assert "tt.pointer_bitcast_pointer_cast" not in result.stdout


def test_different_width_pointer_from_control_flow_is_rejected(tmp_path):
    result = _run_ttolinalg(
        tmp_path, "control_flow_pointer_source", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @control_flow_pointer_source(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32},
              %cond: i1) attributes {noinline = false} {
            %four = arith.constant 4 : i32
            %eight = arith.constant 8 : i32
            %selected = scf.if %cond -> (!tt.ptr<i8>) {
              %then_ptr = tt.addptr %src, %four : !tt.ptr<i8>, i32
              scf.yield %then_ptr : !tt.ptr<i8>
            } else {
              %else_ptr = tt.addptr %src, %eight : !tt.ptr<i8>, i32
              scf.yield %else_ptr : !tt.ptr<i8>
            }
            %wide_ptr = tt.bitcast %selected
                : !tt.ptr<i8> -> !tt.ptr<i32>
            %value = tt.load %wide_ptr : !tt.ptr<i32>
            tt.store %dst, %value : !tt.ptr<i32>
            tt.return
          }
        }
    """)
    _assert_clean_failure(
        result,
        "cannot preserve the exact address from unsupported producer 'scf.if'",
    )


def test_different_width_pointer_from_private_helper_argument_is_rejected(tmp_path):
    result = _run_ttolinalg(
        tmp_path, "private_helper_pointer_source", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func private @private_helper_pointer_source(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %dst: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
            %wide_ptr = tt.bitcast %src : !tt.ptr<i8> -> !tt.ptr<i32>
            %value = tt.load %wide_ptr : !tt.ptr<i32>
            tt.store %dst, %value : !tt.ptr<i32>
            tt.return
          }
        }
    """)
    _assert_clean_failure(
        result,
        "requires a public kernel argument",
    )


def test_pointer_if_with_different_roots_is_rejected(tmp_path):
    result = _run_unstructure(
        tmp_path, "pointer_if_different_roots", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @pointer_if_different_roots(
              %left: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %right: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %cond: i1) attributes {noinline = false} {
            %left_wide = tt.bitcast %left : !tt.ptr<i8> -> !tt.ptr<i32>
            %right_wide = tt.bitcast %right : !tt.ptr<i8> -> !tt.ptr<i32>
            %left_ptrs = tt.splat %left_wide
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %right_ptrs = tt.splat %right_wide
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %selected = scf.if %cond -> (tensor<4x!tt.ptr<i32>>) {
              scf.yield %left_ptrs : tensor<4x!tt.ptr<i32>>
            } else {
              scf.yield %right_ptrs : tensor<4x!tt.ptr<i32>>
            }
            %values = tt.load %selected : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    _assert_clean_failure(
        result,
        "byte-addressed pointer cannot cross structured control flow",
    )


def test_pointer_if_with_different_address_units_is_rejected(tmp_path):
    result = _run_unstructure(
        tmp_path, "pointer_if_different_units", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @pointer_if_different_units(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %cond: i1) attributes {noinline = false} {
            %element_ptrs = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %wide_ptrs = tt.bitcast %element_ptrs
                : tensor<4x!tt.ptr<i8>> -> tensor<4x!tt.ptr<i32>>
            %byte_ptrs = tt.bitcast %wide_ptrs
                : tensor<4x!tt.ptr<i32>> -> tensor<4x!tt.ptr<i8>>
            %selected = scf.if %cond -> (tensor<4x!tt.ptr<i8>>) {
              scf.yield %byte_ptrs : tensor<4x!tt.ptr<i8>>
            } else {
              scf.yield %element_ptrs : tensor<4x!tt.ptr<i8>>
            }
            %values = tt.load %selected : tensor<4x!tt.ptr<i8>>
            tt.return
          }
        }
    """)
    _assert_clean_failure(
        result,
        "byte-addressed pointer cannot cross structured control flow",
    )


def test_for_carried_pointer_with_different_roots_is_rejected(tmp_path):
    result = _run_unstructure(
        tmp_path, "for_pointer_different_roots", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @for_pointer_different_roots(
              %left: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %right: !tt.ptr<i8> {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %left_wide = tt.bitcast %left : !tt.ptr<i8> -> !tt.ptr<i32>
            %right_wide = tt.bitcast %right : !tt.ptr<i8> -> !tt.ptr<i32>
            %left_ptrs = tt.splat %left_wide
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %right_ptrs = tt.splat %right_wide
                : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
            %result = scf.for %iv = %c0 to %c1 step %c1
                iter_args(%current = %left_ptrs)
                -> (tensor<4x!tt.ptr<i32>>) : i32 {
              scf.yield %right_ptrs : tensor<4x!tt.ptr<i32>>
            }
            %values = tt.load %result : tensor<4x!tt.ptr<i32>>
            tt.return
          }
        }
    """)
    _assert_clean_failure(
        result,
        "byte-addressed pointer cannot cross structured control flow",
    )


def test_while_carried_pointer_with_different_address_units_is_rejected(tmp_path):
    result = _run_unstructure(
        tmp_path, "while_pointer_different_units", r"""
        module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
          tt.func public @while_pointer_different_units(
              %src: !tt.ptr<i8> {tt.divisibility = 16 : i32},
              %cond: i1) attributes {noinline = false} {
            %element_ptrs = tt.splat %src
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %wide_ptr = tt.bitcast %src : !tt.ptr<i8> -> !tt.ptr<i32>
            %byte_ptr = tt.bitcast %wide_ptr : !tt.ptr<i32> -> !tt.ptr<i8>
            %byte_ptrs = tt.splat %byte_ptr
                : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
            %result = scf.while (%current = %element_ptrs)
                : (tensor<4x!tt.ptr<i8>>) -> (tensor<4x!tt.ptr<i8>>) {
              scf.condition(%cond) %current : tensor<4x!tt.ptr<i8>>
            } do {
            ^bb0(%current_after: tensor<4x!tt.ptr<i8>>):
              scf.yield %byte_ptrs : tensor<4x!tt.ptr<i8>>
            }
            %values = tt.load %result : tensor<4x!tt.ptr<i8>>
            tt.return
          }
        }
    """)
    _assert_clean_failure(
        result,
        "byte-addressed pointer cannot cross structured control flow",
    )
