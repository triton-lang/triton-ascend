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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Reject unsupported native register calls before invoking the backend."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import triton.language as tl
from triton._C.libtriton.ascend import ir as ascend_ir
from utils.tla_test_utils import make_context, compile_offline, Options
from predicate_ops import predicate_kernel
from group_reduce import group_kernel


def native_builder(operands, target="Ascend950PR_9579"):
    ctx = make_context()
    builder = ascend_ir.tla_builder(ctx, target)
    module = builder.create_module()
    types = [builder.create_tla_vector_type(lanes, dtype.to_ir(builder)) for lanes, dtype in operands]
    fn = builder.get_or_insert_function(module, "contract", builder.get_function_ty(types, []), "public", False)
    module.push_back(fn)
    builder.set_insertion_point_to_start(fn.add_entry_block())
    return ctx, module, builder, [fn.args(i) for i in range(len(types))]


@pytest.mark.parametrize("lanes,dtype,unpack,part", [(32, tl.int1, False, 0), (256, tl.int1, False, 0),
                                                     (64, tl.int1, True, 0), (128, tl.int16, False, 0),
                                                     (128, tl.int1, False, -1), (128, tl.int1, True, 2)])
def test_pack_rejects_invalid_shape_and_part(lanes, dtype, unpack, part):
    ctx, module, builder, args = native_builder([(lanes, dtype)])
    with pytest.raises(ValueError, match="shape or part"):
        builder.create_simd_pred_pack(args[0], unpack, part)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("left,right,dtype", [(64, 128, tl.int1), (32, 32, tl.int1), (128, 128, tl.int16)])
def test_interleave_rejects_mismatched_predicates(inverse, left, right, dtype):
    ctx, module, builder, args = native_builder([(left, dtype), (right, dtype)])
    with pytest.raises(ValueError, match="matching 64/128/256 lanes"):
        builder.create_simd_pred_interleave(*args, inverse)


@pytest.mark.parametrize("lanes,dtype,mask_lanes,unsigned,kind,message",
                         [(256, tl.int8, 256, False, "add", "unsupported group reduction type"),
                          (128, tl.bfloat16, 128, False, "max", "unsupported group reduction type"),
                          (64, tl.int64, 64, False, "min", "unsupported group reduction type"),
                          (32, tl.float32, 32, False, "add", "unsupported group reduction type"),
                          (128, tl.float16, 128, True, "max", "unsupported group reduction type"),
                          (64, tl.float32, 64, True, "min", "unsupported group reduction type"),
                          (128, tl.int16, 64, False, "add", "mask must match source lanes"),
                          (64, tl.int32, 64, False, "sum", "unknown group reduction kind")])
def test_group_rejects_invalid_contract(lanes, dtype, mask_lanes, unsigned, kind, message):
    ctx, module, builder, args = native_builder([(lanes, dtype), (mask_lanes, tl.int1)])
    with pytest.raises(ValueError, match=message):
        builder.create_simd_group_reduce(kind, *args, unsigned)


@pytest.mark.parametrize("target", ["Ascend960", "Ascend910B1"])
@pytest.mark.parametrize("op", ["ppack", "punpack", "pintlv", "pdintlv", "add", "max", "min"])
def test_rejects_unverified_targets(target, op):
    types = [(128, tl.int1), (128, tl.int1)] if op.startswith("p") else [(128, tl.int16), (128, tl.int1)]
    ctx, module, builder, args = native_builder(types, target)
    with pytest.raises(ValueError, match="supports Ascend950 only"):
        if op in ("ppack", "punpack"):
            builder.create_simd_pred_pack(args[0], op == "punpack", 0)
        elif op in ("pintlv", "pdintlv"):
            builder.create_simd_pred_interleave(*args, op == "pdintlv")
        else:
            builder.create_simd_group_reduce(op, *args, False)


@pytest.mark.parametrize("op", ["ppack", "punpack"])
def test_frontend_rejects_noncontiguous_part(op):
    with pytest.raises(Exception, match="part must be LOWER or HIGHER"):
        compile_offline(predicate_kernel, dict(X="*u8", Y="*u8", O="*i8"), dict(W=16, OP=op, PART="even"))


@pytest.mark.parametrize("op", ["add", "max", "min"])
def test_frontend_rejects_unknown_mode(op):
    with pytest.raises(Exception, match="mode"):
        compile_offline(group_kernel, dict(X="*i32", M="*i32", O="*i32"),
                        dict(DT=tl.int32, W=32, OP=op, MASKED=True, MODE="invalid"))


def test_frontend_checks_target(monkeypatch):
    monkeypatch.setattr(Options, "arch", "Ascend960")
    with pytest.raises(Exception, match="supports Ascend950 only"):
        compile_offline(group_kernel, dict(X="*i32", M="*i32", O="*i32"),
                        dict(DT=tl.int32, W=32, OP="add", MASKED=True, MODE="ZEROING"))
