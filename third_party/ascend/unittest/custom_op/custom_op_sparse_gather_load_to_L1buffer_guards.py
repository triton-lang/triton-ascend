#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Frontend guards on `al.sparse_gather_load_to_L1buffer`.
#
# The template writes block_r * region_size token rows into `dst` without
# bounds checking and addresses a token row as one 32B block, so a destination
# that disagrees with the scalar arguments is silent corruption on the device.
# These are the cases the builtin has to reject before it gets that far.

from __future__ import annotations

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions

REGION_VALID_SHIFT = 24
REGION_ID_MASK = (1 << 24) - 1


@triton.jit
def _gather(dst, k_ptr, packed_ptr, block_r: tl.constexpr, region_size: tl.constexpr, dim_size: tl.constexpr):
    al.sparse_gather_load_to_L1buffer(
        dst,
        k_ptr,
        packed_ptr,
        block_r=block_r,
        valid_region_count=block_r,
        region_size=region_size,
        dim_size=dim_size,
        stride_token=dim_size,
        region_valid_shift=REGION_VALID_SHIFT,
        region_id_mask=REGION_ID_MASK,
    )


@triton.jit
def dst_in_ub(k_ptr, packed_ptr, region_size: tl.constexpr, dim_size: tl.constexpr, BLOCK_R: tl.constexpr):
    dst = bl.alloc(tl.float16, [BLOCK_R * region_size, dim_size], al.ascend_address_space.UB)
    _gather(dst, k_ptr, packed_ptr, BLOCK_R, region_size, dim_size)


@triton.jit
def dst_too_short(k_ptr, packed_ptr, region_size: tl.constexpr, dim_size: tl.constexpr, BLOCK_R: tl.constexpr):
    dst = bl.alloc(tl.float16, [BLOCK_R * region_size // 2, dim_size], al.ascend_address_space.L1)
    _gather(dst, k_ptr, packed_ptr, BLOCK_R, region_size, dim_size)


@triton.jit
def row_not_block_aligned(k_ptr, packed_ptr, region_size: tl.constexpr, dim_size: tl.constexpr,
                          BLOCK_R: tl.constexpr):
    dst = bl.alloc(tl.float16, [BLOCK_R * region_size, dim_size], al.ascend_address_space.L1)
    _gather(dst, k_ptr, packed_ptr, BLOCK_R, region_size, dim_size)


@triton.jit
def well_formed(k_ptr, packed_ptr, region_size: tl.constexpr, dim_size: tl.constexpr, BLOCK_R: tl.constexpr):
    dst = bl.alloc(tl.float16, [BLOCK_R * region_size, dim_size], al.ascend_address_space.L1)
    _gather(dst, k_ptr, packed_ptr, BLOCK_R, region_size, dim_size)


def compile_error(kernel, dim_size: int) -> str | None:
    """Return the message the builtin rejected the kernel with, or None if it compiled."""
    src = ASTSource(kernel, {"k_ptr": "*fp16", "packed_ptr": "*i32"},
                    {"region_size": 8, "dim_size": dim_size, "BLOCK_R": 16})
    ctx = ir.context()
    ir.load_dialects(ctx)
    ascend_ir.load_dialects(ctx)
    try:
        ast_to_ttir(kernel, src, ctx, NPUOptions(arch="Ascend950PR_9579"), {}, {})
        return None
    except Exception as ex:  # noqa: BLE001 - the message is what is under test
        # Triton wraps the guard's error in source context, so walk to the cause.
        root = ex
        while root.__cause__ is not None:
            root = root.__cause__
        return str(root).strip().splitlines()[-1]


CASES = (
    ("dst allocated in UB", dst_in_ub, 192, True),
    ("dst shorter than block_r * region_size", dst_too_short, 192, True),
    ("token row not a whole number of 32B blocks", row_not_block_aligned, 100, True),
    ("well formed", well_formed, 192, False),
)


def main() -> int:
    failures = 0
    for name, kernel, dim_size, want_rejected in CASES:
        message = compile_error(kernel, dim_size)
        rejected = message is not None
        ok = rejected == want_rejected
        failures += not ok
        outcome = f"rejected: {message}" if rejected else "compiled"
        print(f"  {'ok  ' if ok else 'FAIL'} {name:44s} {outcome}")
    print("\nPASS" if not failures else f"\n{failures}/{len(CASES)} case(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
