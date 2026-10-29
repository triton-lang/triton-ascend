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
"""IR contracts for ZeroValuePropagationPass.

The pass is scheduled first in ``ttir_to_linalg``. These tests enter through
the same binding the backend uses, so they see exactly the module that
``make_ttir`` produced. Every rule is an identity on the path it folds, so
numeric output alone cannot tell whether a fold happened; the assertions are
therefore made on the emitted IR.
"""

import pytest
import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl

from triton._C.libtriton import ascend, ir
from triton._C.libtriton.ascend import ir as ascend_ir

if not hasattr(ascend.passes.ttir, "add_zero_value_propagation"):
    pytest.skip(
        "requires the TritonAscend build containing zero-value propagation",
        allow_module_level=True,
    )

WIDTH = 64


def _run(text, tmp_path, *, allow_fp_mul_zero_fold=True):
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    path = tmp_path / "zero_value_propagation.mlir"
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_zero_value_propagation(pm, allow_fp_mul_zero_fold)
    pm.run(module, "zero-value-propagation-test")
    return str(module)


def _integer_fold_module():
    """``x * 0 -> 0`` then ``0 + x -> x`` and ``y - 0 -> y``."""
    return f"""
module {{
  tt.func public @integer_folds(%x: tensor<{WIDTH}xi32>, %y: tensor<{WIDTH}xi32>) -> tensor<{WIDTH}xi32> {{
    %zero = arith.constant dense<0> : tensor<{WIDTH}xi32>
    %prod = arith.muli %x, %zero : tensor<{WIDTH}xi32>
    %sum = arith.addi %prod, %x : tensor<{WIDTH}xi32>
    %diff = arith.subi %y, %zero : tensor<{WIDTH}xi32>
    %out = arith.addi %sum, %diff : tensor<{WIDTH}xi32>
    tt.return %out : tensor<{WIDTH}xi32>
  }}
}}
"""


def _i1_fold_module():
    """``i1`` masks: ``0`` is ``false``, so ``addi``/``subi`` with it are identities."""
    return f"""
module {{
  tt.func public @i1_folds(%mask: tensor<{WIDTH}xi1>) -> tensor<{WIDTH}xi1> {{
    %false = arith.constant dense<false> : tensor<{WIDTH}xi1>
    %sum = arith.addi %mask, %false : tensor<{WIDTH}xi1>
    %diff = arith.subi %sum, %false : tensor<{WIDTH}xi1>
    %and = arith.andi %diff, %mask : tensor<{WIDTH}xi1>
    tt.return %and : tensor<{WIDTH}xi1>
  }}
}}
"""


def _splat_module():
    """A scalar zero stays zero through ``tt.splat``."""
    return f"""
module {{
  tt.func public @splat_chain(%x: tensor<{WIDTH}xi32>) -> tensor<{WIDTH}xi32> {{
    %zero = arith.constant 0 : i32
    %splat = tt.splat %zero : i32 -> tensor<{WIDTH}xi32>
    %sum = arith.addi %x, %splat : tensor<{WIDTH}xi32>
    %out = arith.subi %sum, %splat : tensor<{WIDTH}xi32>
    tt.return %out : tensor<{WIDTH}xi32>
  }}
}}
"""


def _reduce_module():
    """``reduce_add`` of an all-zero integer tile feeds a fold through ``tt.splat``."""
    return f"""
module {{
  tt.func public @reduce_chain(%x: tensor<{WIDTH}xi32>) -> (tensor<{WIDTH}xi32>, i32) {{
    %zero = arith.constant dense<0> : tensor<{WIDTH}xi32>
    %red = "tt.reduce"(%zero) <{{axis = 0 : i32}}> ({{
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      tt.reduce.return %sum : i32
    }}) : (tensor<{WIDTH}xi32>) -> i32
    %splat = tt.splat %red : i32 -> tensor<{WIDTH}xi32>
    %out = arith.addi %x, %splat : tensor<{WIDTH}xi32>
    tt.return %out, %red : tensor<{WIDTH}xi32>, i32
  }}
}}
"""


def _negated_module():
    """``0 - x`` is ``-x``, not ``x``: it must survive (``INT_MIN`` overflow)."""
    return f"""
module {{
  tt.func public @negated(%x: tensor<{WIDTH}xi32>) -> tensor<{WIDTH}xi32> {{
    %zero = arith.constant dense<0> : tensor<{WIDTH}xi32>
    %neg = arith.subi %zero, %x : tensor<{WIDTH}xi32>
    tt.return %neg : tensor<{WIDTH}xi32>
  }}
}}
"""


def _fp_mul_module():
    """``x * 0`` with a non-constant ``x`` is the option-gated rule."""
    return f"""
module {{
  tt.func public @fp_mul_zero(%x: tensor<{WIDTH}xf32>) -> tensor<{WIDTH}xf32> {{
    %zero = arith.constant dense<0.000000e+00> : tensor<{WIDTH}xf32>
    %prod = arith.mulf %x, %zero : tensor<{WIDTH}xf32>
    %sum = arith.addf %prod, %x : tensor<{WIDTH}xf32>
    tt.return %sum : tensor<{WIDTH}xf32>
  }}
}}
"""


def test_integer_mul_add_sub_are_folded(tmp_path):
    text = _run(_integer_fold_module(), tmp_path)

    # The product is a zero, so both following additions collapse onto their
    # non-zero operand; the subtraction drops its zero subtrahend.
    assert "arith.muli" not in text
    assert "arith.subi" not in text
    assert text.count("arith.addi") == 1


def test_integer_folds_do_not_ask_for_the_fp_option(tmp_path):
    """Integer rules are unconditional, so an IEEE-relaxation count must not appear."""
    text = _run(_integer_fold_module(), tmp_path, allow_fp_mul_zero_fold=False)

    assert "arith.muli" not in text
    assert "tt.fp_mul_zero_folds" not in text


def test_i1_zero_mask_keeps_mask_through_add_and_sub(tmp_path):
    text = _run(_i1_fold_module(), tmp_path)

    # `i1` add/sub is xor, so both statements are identities on `%mask`; `andi`
    # is not a zero-value rule and must stay.
    assert "arith.addi" not in text
    assert "arith.subi" not in text
    assert text.count("arith.andi") == 1


def test_scalar_zero_splat_is_folded(tmp_path):
    text = _run(_splat_module(), tmp_path)

    assert "tt.splat" not in text
    assert "arith.addi" not in text
    assert "arith.subi" not in text
    assert "arith.constant" not in text


def test_zero_reduce_result_propagates_through_splat(tmp_path):
    text = _run(_reduce_module(), tmp_path)

    # The reduce stays because its result is returned, but the splat of that
    # zero result and the addition that consumed it are gone; only the reduce
    # combiner's own `addi` is left.
    assert text.count("arith.addi") == 1
    assert '"tt.reduce"' in text
    assert "tt.splat" not in text


def test_zero_minus_value_is_not_folded(tmp_path):
    text = _run(_negated_module(), tmp_path)

    assert "arith.subi" in text


def test_fp_mul_zero_fold_is_opt_in(tmp_path):
    enabled = _run(_fp_mul_module(), tmp_path)
    assert "arith.addf" not in enabled
    assert "tt.fp_mul_zero_folds = 1 : i32" in enabled

    disabled = _run(_fp_mul_module(), tmp_path, allow_fp_mul_zero_fold=False)
    assert "arith.mulf" in disabled
    assert "arith.addf" in disabled
    assert "tt.fp_mul_zero_folds" not in disabled


@triton.jit
def integer_zero_chain_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    false_mask = tl.zeros((BLOCK, ), dtype=tl.int1)
    mask = off < n
    mask = mask + false_mask
    mask = mask - false_mask
    values = tl.load(x_ptr + off, mask=mask, other=-1)
    zero = tl.zeros((BLOCK, ), dtype=tl.int32)
    acc = values * zero
    acc = acc + values
    acc = acc - zero
    tl.store(out_ptr + off, acc)


@pytest.mark.parametrize("n", [0, 1, 37, WIDTH])
def test_integer_zero_chain_kernel_matches_torch(n):
    """End-to-end guard: the folds must not change masked-lane behaviour."""
    x = torch.arange(WIDTH, dtype=torch.int32, device="npu") + 7
    out = torch.full((WIDTH, ), -5, dtype=torch.int32, device="npu")

    integer_zero_chain_kernel[(1, )](x, out, n, WIDTH)
    torch.npu.synchronize()

    lanes = torch.arange(WIDTH, dtype=torch.int32)
    expected = torch.where(lanes < n, x.cpu(), torch.full((WIDTH, ), -1, dtype=torch.int32))
    torch.testing.assert_close(out.cpu(), expected)
