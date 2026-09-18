#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Demo: `__builtin_sparse_gather_load_to_l1cache`, the builtin behind
# `al.sparse_gather_load_to_l1cache`.
#
# This is the replacement for the per-region gather that sparse attention used to
# run in UB. The kernel below is the gather half of a sparse-attention prefill
# step: for one (query, kv group) row it walks the row's packed region list and
# lands the selected K and V tokens in L1, ready for L0.
#
# What the builtin buys over the Triton-level gather it replaces:
#   * GM -> L1 directly, so UB and the vector pipe stay free and there is no
#     vec -> cube handoff.
#   * Neighbouring region ids collapse into one MTE2 descriptor. The selector
#     emits each row's regions sorted by region offset and a region's tokens are
#     contiguous in the physical token space, so a run of consecutive ids is one
#     contiguous GM range. Triton has to treat every region as an independent
#     access, which is what made the old version issue many short reads.
#
# A5 (dav-c310) only: GM -> L1 is a cube-core MTE2 path there, and no other
# supported target has a GM -> L1 instruction at all.

from __future__ import annotations

import subprocess

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, ttir_to_linalg

# Region metadata packing used by the sparse-attention selector: each slot is
# `region_offset | (valid_tokens << 24)`, with -1 padding the tail.
REGION_VALID_SHIFT = 24
REGION_ID_MASK = (1 << 24) - 1


@triton.jit
def sparse_gather_kernel(
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    stride_k_token,
    stride_k_group,
    stride_v_token,
    stride_v_group,
    stride_packed_row,
    total_q,
    region_size: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    query = tl.program_id(0)
    group = tl.program_id(1)
    row = group * total_q + query
    count = tl.load(counts_ptr + row).to(tl.int32)

    # K and V share the region list, so both land in L1 off the same metadata.
    keys_l1 = bl.alloc(tl.float16, [BLOCK_R * region_size, head_dim], al.ascend_address_space.L1)
    values_l1 = bl.alloc(tl.float16, [BLOCK_R * region_size, head_dim], al.ascend_address_space.L1)

    for base in tl.range(0, count, BLOCK_R):
        # The group offset folds into the base pointer, and the row offset into
        # the metadata pointer, so the builtin only ever sees a token stride.
        al.sparse_gather_load_to_l1cache(
            k_ptr + group * stride_k_group,
            packed_ptr + row * stride_packed_row + base,
            keys_l1,
            block_r=BLOCK_R,
            valid_region_count=count - base,
            region_size=region_size,
            dim_size=head_dim,
            stride_token=stride_k_token,
            region_valid_shift=REGION_VALID_SHIFT,
            region_id_mask=REGION_ID_MASK,
        )
        al.sparse_gather_load_to_l1cache(
            v_ptr + group * stride_v_group,
            packed_ptr + row * stride_packed_row + base,
            values_l1,
            block_r=BLOCK_R,
            valid_region_count=count - base,
            region_size=region_size,
            dim_size=head_dim,
            stride_token=stride_v_token,
            region_valid_shift=REGION_VALID_SHIFT,
            region_id_mask=REGION_ID_MASK,
        )
        # keys_l1 / values_l1 now hold BLOCK_R * region_size token rows in NZ,
        # zero filled on padding slots and on a partial region's tail, so the
        # QK^T and PV mmads can read them straight out of L1.


def compile_to_linalg_mlir(kernel, signature: dict, constants: dict) -> str | None:
    src = ASTSource(kernel, signature, constants)
    ctx = ir.context()
    ir.load_dialects(ctx)
    ascend_ir.load_dialects(ctx)
    options = NPUOptions(arch="Ascend910_9589")
    try:
        ttir = ast_to_ttir(kernel, src, ctx, options, {}, {})
        meta = {**options.__dict__}
        return str(ttir_to_linalg(ttir, meta, options, named_ops=True))
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode())
        print(ex.stderr.decode())
        return None


def main() -> None:
    mlir = compile_to_linalg_mlir(
        sparse_gather_kernel,
        {
            "k_ptr": "*fp16",
            "v_ptr": "*fp16",
            "packed_ptr": "*i32",
            "counts_ptr": "*i32",
            "stride_k_token": "i32",
            "stride_k_group": "i32",
            "stride_v_token": "i32",
            "stride_v_group": "i32",
            "stride_packed_row": "i32",
            "total_q": "i32",
        },
        {"region_size": 8, "head_dim": 192, "BLOCK_R": 16},
    )
    if not mlir:
        print("Compilation failed.")
        return

    hits = [line for line in mlir.splitlines() if "sparse_gather_load_to_l1cache" in line]
    print(f"hivm.hir.custom sites for the builtin: {len(hits)} (expected 2, one for K and one for V)")
    for line in hits:
        print(line.strip())

    # The canonicalizer fills these in from CustomOp::kBuiltins, so seeing them
    # confirms the builtin is registered on the compiler side too.
    for expected in ("tcore_type<CUBE>", "pipe<PIPE_MTE2>"):
        print(f"{expected}: {'found' if expected in mlir else 'MISSING'}")


if __name__ == "__main__":
    main()
