"""Additional A5 standalone MMAD L0 tests, all WITHOUT bias.

From the triton-ascend root (substitute this file's repository-relative path):
  TLA_TEST_STAGE=ir python -m pytest -v -s -x <this-file> -k test_ir
  TLA_TEST_STAGE=compile python -m pytest -v -s -x <this-file> -k test_compile
  TLA_TEST_STAGE=run python -m pytest -v -s -x <this-file> -k test_run
  TLA_TEST_STAGE=run python <this-file>

Case groups for pytest -k: dynamic_clear, subview, capacity, shape, barrier,
l12l0_view and l12l0_batch. These groups read back L1->L0A/B copies with
identity GEMMs, including inactive batch slots and strided views.
The script form runs all enabled stages. All kernels use explicit pipeline
events and grid=(1,), unit_flag=False, sync_solver=False.
Capacity cases exercise legal upper bounds (64 KiB L0A/B, 256 KiB L0C,
and individual M/N/K=2048).
"""
import os
import re
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import Options, compile_offline, lower_offline, check_kernel_binary, get_stages, matrix


class A5Options(Options):
    arch = "Ascend950PR_9589"


def _case(name, m, n, k, part, slots=2, kind="shape"):
    return dict(name=name, kind=kind, M=m, N=n, K=k, L0_K=part, SLOTS=slots)


CONFIGS = [
    _case("shape_tall_k16", 64, 32, 64, 16),
    _case("shape_rectangular_k48", 48, 80, 144, 48),
    _case("shape_square_k64", 128, 128, 192, 64),
    _case("shape_long_k17_tiles", 32, 64, 272, 16),
    # CATLASS inserts PIPE_M barrier below 10 output fractals. Cover both
    # sides and the threshold without adding a PIPE_M barrier in this kernel.
    _case("barrier_9_fractals", 48, 48, 64, 32),
    _case("barrier_10_fractals", 32, 80, 64, 32),
    _case("barrier_12_fractals", 48, 64, 64, 32),
    _case("capacity_single_l0a_64k", 256, 32, 128, 128, slots=1),
    _case("capacity_single_l0b_64k", 32, 256, 128, 128, slots=1),
    _case("capacity_two_slots_l0ab_64k", 128, 128, 256, 128),
    _case("capacity_l0c_256k", 256, 256, 64, 32),
    _case("capacity_m2048", 2048, 16, 16, 16, slots=1),
    _case("capacity_n2048", 16, 2048, 16, 16, slots=1),
    _case("capacity_k2048", 16, 16, 2048, 2048, slots=1),
    _case("dynamic_clear_small", 32, 64, 64, 32, kind="dynamic"),
    _case("dynamic_clear_large", 64, 64, 96, 32, kind="dynamic"),
    dict(name="subview_static", kind="subview", DYNAMIC=False),
    dict(name="subview_dynamic", kind="subview", DYNAMIC=True),
]


def _l12l0_view_cases():
    for dtype in ("float16", "bfloat16"):
        for to_b in (False, True):
            side = "b" if to_b else "a"
            rows, cols, tile_rows, tile_cols = (96, 80, 48, 32) if to_b else (64, 96, 32, 48)
            for endpoint in ("src", "dst"):
                # Source zN's inner physical axis is logical rows. Destination
                # nZ (B) instead has logical columns on that physical axis.
                one_rows, one_cols = (32, 16) if endpoint == "src" or to_b else (16, 32)
                windows = (
                    ("pitch", rows, cols, tile_rows, tile_cols, 16, 16),
                    ("end", rows, cols, tile_rows, tile_cols, rows - tile_rows, cols - tile_cols),
                    ("one_fractal", one_rows, one_cols, 16, 16, one_rows - 16, one_cols - 16),
                )
                for boundary, pr, pc, tr, tc, row, col in windows:
                    yield dict(name=f"l12l0_view_{endpoint}_{side}_{boundary}_{dtype}", kind="l12l0_view",
                               endpoint=endpoint, dtype=dtype, TO_B=to_b, ROWS=pr, COLS=pc, TILE_ROWS=tr, TILE_COLS=tc,
                               ROW=row, COL=col)


CONFIGS += list(_l12l0_view_cases())
CONFIGS += [
    dict(name=f"l12l0_batch_{side}_{layout}", kind="l12l0_batch", TO_B=side == "b", SRC_STEP=src_step,
         DST_STEP=dst_step)
    for side in ("a", "b")
    for layout, src_step, dst_step in (("compact", 1, 1), ("strided", 2, 3))
]


# The shared test helper pins tla.jit to TLAJITFunction. partial supports
# decorator options with that class as well as the public decorator function.
@partial(tla.jit, do_not_specialize=["Clear"])
def plain_mmad(A, B, C, Clear, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, L0_K: tl.constexpr,
               SLOTS: tl.constexpr, SECOND_PASS: tl.constexpr):
    """One GEMM, or two passes whose second initialization uses runtime i1.

    Clear is a Python bool launch scalar, NOT a torch tensor pointer, and is
    not constexpr. Pass one always initializes acc to P=A@B. Pass two yields
    P if Clear is true or 2P if false; no uninitialized accumulator is read.
    """
    tl.static_assert(M > 0 and N > 0 and K > 0 and L0_K > 0)
    tl.static_assert(M % 16 == 0 and N % 16 == 0 and L0_K % 16 == 0)
    tl.static_assert(K % L0_K == 0 and (SLOTS == 1 or SLOTS == 2))
    tl.static_assert(SLOTS * 2 * M * L0_K <= 64 * 1024)
    tl.static_assert(SLOTS * 2 * N * L0_K <= 64 * 1024)
    tl.static_assert(4 * M * N <= 256 * 1024)
    tl.static_assert(2 * K * (M + N) <= 512 * 1024)
    PASSES: tl.constexpr = 2 if SECOND_PASS else 1
    with tla.scope.cube():
        a_l1 = tla.alloc((K // 16, M // 16, 16, 16), tl.float16, space="l1")
        b_l1 = tla.alloc((N // 16, K // 16, 16, 16), tl.float16, space="l1")
        a0 = tla.alloc((L0_K // 16, M // 16, 16, 16), tl.float16, space="l0a")
        b0 = tla.alloc((L0_K // 16, N // 16, 16, 16), tl.float16, space="l0b")
        if SLOTS == 2:
            a1 = tla.alloc((L0_K // 16, M // 16, 16, 16), tl.float16, space="l0a")
            b1 = tla.alloc((L0_K // 16, N // 16, 16, 16), tl.float16, space="l0b")
        acc = tla.alloc((N // 16, M // 16, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(A, (M, K)), a_l1)
        tla.load_gm_to_l1(tla.view(B, (K, N)), b_l1)
        tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        # Keep one free token per slot across both passes, exactly as across
        # K tiles; do not reinitialize event state between passes.
        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 0)
        if SLOTS == 2:
            tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 1)
        for pass_id in tl.static_range(PASSES):
            for ki in tl.static_range(K // L0_K):
                if ki % SLOTS == 0:
                    a_tile = a0
                    b_tile = b0
                else:
                    a_tile = a1
                    b_tile = b1
                tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, ki % SLOTS)
                a_src = tla.subview(a_l1, (ki * (L0_K // 16), 0, 0, 0), (L0_K // 16, M // 16, 16, 16))
                b_src = tla.subview(b_l1, (0, ki * (L0_K // 16), 0, 0), (N // 16, L0_K // 16, 16, 16))
                tla.copy_l1_to_l0(a_src, a_tile)
                tla.copy_l1_to_l0(b_src, b_tile)
                tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
                tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
                if pass_id == 1 and ki == 0:
                    tla.cube.mmad_l0(a_tile, b_tile, acc, m=M, n=N, k=L0_K, clear_accum=Clear,
                                     unit_flag_ctrl=tla.UnitFlagMode.DISABLED)
                else:
                    tla.cube.mmad_l0(a_tile, b_tile, acc, m=M, n=N, k=L0_K, clear_accum=(ki == 0),
                                     unit_flag_ctrl=tla.UnitFlagMode.DISABLED)
                tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, ki % SLOTS)
        # Drain all free tokens, including an unused second slot.
        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 0)
        if SLOTS == 2:
            tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 1)
        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
        tla.store_l0c_to_gm(
            acc, tla.view(C, (M, N)),
            tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND, unit_flag_ctrl=tla.UnitFlagMode.DISABLED))
        tla.sync.barrier(tla.sync.PIPE.ALL)


@partial(tla.jit, do_not_specialize=["a_offset", "b_offset", "c_offset"])
def subview_mmad(A, B, TileOut, FullOut, a_offset, b_offset, c_offset, DYNAMIC: tl.constexpr):
    """Initialize parents, then update only a compact window of L0C.

    A/B parents contain (32,96) and (96,64). Initialize both N=64 halves of
    an N=128 C parent to P=A@B. Then select independent K=32 A/B windows and
    overwrite a middle N=64 C window. FullOut checks updated AND untouched
    columns independently of the store through the subview to TileOut.
    Offsets are indices in physical dimension zero, not bytes or elements.
    """
    with tla.scope.cube():
        a_l1 = tla.alloc((6, 2, 16, 16), tl.float16, space="l1")
        b_l1 = tla.alloc((4, 6, 16, 16), tl.float16, space="l1")
        a_root = tla.alloc((6, 2, 16, 16), tl.float16, space="l0a")
        b_root = tla.alloc((6, 4, 16, 16), tl.float16, space="l0b")
        c_root = tla.alloc((8, 2, 16, 16), tl.float32, space="l0c")
        c_left = tla.subview(c_root, (0, 0, 0, 0), (4, 2, 16, 16))
        c_right = tla.subview(c_root, (4, 0, 0, 0), (4, 2, 16, 16))
        tla.load_gm_to_l1(tla.view(A, (32, 96)), a_l1)
        tla.load_gm_to_l1(tla.view(B, (96, 64)), b_l1)
        # Populate the WHOLE L0 A/B parents. Using the target subviews for
        # the copies could let matching copy/MMAD offset bugs cancel out.
        tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        tla.copy_l1_to_l0(a_l1, a_root)
        tla.copy_l1_to_l0(b_l1, b_root)
        tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
        tla.cube.mmad_l0(a_root, b_root, c_left, m=32, n=64, k=96, clear_accum=True,
                         unit_flag_ctrl=tla.UnitFlagMode.DISABLED)
        tla.cube.mmad_l0(a_root, b_root, c_right, m=32, n=64, k=96, clear_accum=True,
                         unit_flag_ctrl=tla.UnitFlagMode.DISABLED)
        # Complete initialization before a partially overlapping C update.
        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 0)
        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 0)
        tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
        if DYNAMIC:
            a_window = tla.subview(a_root, (a_offset, 0, 0, 0), (2, 2, 16, 16))
            b_window = tla.subview(b_root, (b_offset, 0, 0, 0), (2, 4, 16, 16))
            c_window = tla.subview(c_root, (c_offset, 0, 0, 0), (4, 2, 16, 16))
        else:
            a_window = tla.subview(a_root, (1, 0, 0, 0), (2, 2, 16, 16))
            b_window = tla.subview(b_root, (3, 0, 0, 0), (2, 4, 16, 16))
            c_window = tla.subview(c_root, (2, 0, 0, 0), (4, 2, 16, 16))
        tla.cube.mmad_l0(a_window, b_window, c_window, m=32, n=64, k=32, clear_accum=True,
                         unit_flag_ctrl=tla.UnitFlagMode.DISABLED)
        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
        tla.store_l0c_to_gm(
            c_window, tla.view(TileOut, (32, 64)),
            tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND, unit_flag_ctrl=tla.UnitFlagMode.DISABLED))
        tla.store_l0c_to_gm(
            c_root, tla.view(FullOut, (32, 128)),
            tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND, unit_flag_ctrl=tla.UnitFlagMode.DISABLED))
        tla.sync.barrier(tla.sync.PIPE.ALL)


@tla.jit
def l12l0_source_mmad(Data, Identity, C, TO_B: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
                      TILE_ROWS: tl.constexpr, TILE_COLS: tl.constexpr, ROW: tl.constexpr, COL: tl.constexpr):
    """Copy one L1 window into compact L0, then expose it with an identity GEMM."""
    K: tl.constexpr = TILE_ROWS if TO_B else TILE_COLS
    with tla.scope.cube():
        src = tla.alloc((COLS // 16, ROWS // 16, 16, 16), Data.dtype.element_ty, space="l1")
        identity_l1 = tla.alloc((K // 16, K // 16, 16, 16), Data.dtype.element_ty, space="l1")
        a = tla.alloc((K // 16, TILE_ROWS // 16, 16, 16), Data.dtype.element_ty, space="l0a")
        b = tla.alloc((K // 16, TILE_COLS // 16, 16, 16), Data.dtype.element_ty, space="l0b")
        acc = tla.alloc((TILE_COLS // 16, TILE_ROWS // 16, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(Data, (ROWS, COLS)), src)
        tla.load_gm_to_l1(tla.view(Identity, (K, K)), identity_l1)
        tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        # L1 is zN for both A and B. The source retains the full parent pitch.
        tile = tla.subview(src, (COL // 16, ROW // 16, 0, 0), (TILE_COLS // 16, TILE_ROWS // 16, 16, 16))
        if TO_B:
            tla.copy_l1_to_l0(identity_l1, a)
            tla.copy_l1_to_l0(tile, b)
        else:
            tla.copy_l1_to_l0(tile, a)
            tla.copy_l1_to_l0(identity_l1, b)
        tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
        tla.cube.mmad_l0(a, b, acc, m=TILE_ROWS, n=TILE_COLS, k=K, clear_accum=True,
                         unit_flag_ctrl=tla.UnitFlagMode.DISABLED)
        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
        tla.store_l0c_to_gm(
            acc, tla.view(C, (TILE_ROWS, TILE_COLS)),
            tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND, unit_flag_ctrl=tla.UnitFlagMode.DISABLED))
        tla.sync.barrier(tla.sync.PIPE.ALL)


@tla.jit
def l12l0_destination_mmad(Data, Initial, Identity, C, TO_B: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
                           TILE_ROWS: tl.constexpr, TILE_COLS: tl.constexpr, ROW: tl.constexpr, COL: tl.constexpr):
    """Overwrite an L0 window, but read the WHOLE parent through an identity GEMM.

    MMAD uses zero-offset compact parents, not the copy's destination subview.
    This prevents matching copy/MMAD view bugs from hiding a misplaced write,
    and exposes untouched elements before, between and after the copied blocks.
    """
    K: tl.constexpr = ROWS if TO_B else COLS
    with tla.scope.cube():
        src = tla.alloc((TILE_COLS // 16, TILE_ROWS // 16, 16, 16), Data.dtype.element_ty, space="l1")
        initial_l1 = tla.alloc((COLS // 16, ROWS // 16, 16, 16), Data.dtype.element_ty, space="l1")
        identity_l1 = tla.alloc((K // 16, K // 16, 16, 16), Data.dtype.element_ty, space="l1")
        a = tla.alloc((K // 16, ROWS // 16, 16, 16), Data.dtype.element_ty, space="l0a")
        b = tla.alloc((K // 16, COLS // 16, 16, 16), Data.dtype.element_ty, space="l0b")
        acc = tla.alloc((COLS // 16, ROWS // 16, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(Data, (TILE_ROWS, TILE_COLS)), src)
        tla.load_gm_to_l1(tla.view(Initial, (ROWS, COLS)), initial_l1)
        tla.load_gm_to_l1(tla.view(Identity, (K, K)), identity_l1)
        tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        if TO_B:
            tla.copy_l1_to_l0(identity_l1, a)
            tla.copy_l1_to_l0(initial_l1, b)
            tile = tla.subview(b, (ROW // 16, COL // 16, 0, 0), (TILE_ROWS // 16, TILE_COLS // 16, 16, 16))
        else:
            tla.copy_l1_to_l0(initial_l1, a)
            tla.copy_l1_to_l0(identity_l1, b)
            tile = tla.subview(a, (COL // 16, ROW // 16, 0, 0), (TILE_COLS // 16, TILE_ROWS // 16, 16, 16))
        # Finish parent initialization before the overlapping MTE1 write.
        tla.sync.barrier(tla.sync.PIPE.MTE1)
        tla.copy_l1_to_l0(src, tile)
        tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
        tla.cube.mmad_l0(a, b, acc, m=ROWS, n=COLS, k=K, clear_accum=True, unit_flag_ctrl=tla.UnitFlagMode.DISABLED)
        tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
        tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
        tla.store_l0c_to_gm(
            acc, tla.view(C, (ROWS, COLS)),
            tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND, unit_flag_ctrl=tla.UnitFlagMode.DISABLED))
        tla.sync.barrier(tla.sync.PIPE.ALL)


@partial(tla.jit, do_not_specialize=["Total"])
def l12l0_batch_mmad(Data, Initial, Identity, Out, Total, TO_B: tl.constexpr, SRC_STEP: tl.constexpr,
                     DST_STEP: tl.constexpr):
    # One fractal per matrix lets MMAD read each slot from the same 4D parent.
    SRC_SLOTS: tl.constexpr = 1 + 12 * SRC_STEP
    DST_SLOTS: tl.constexpr = 1 + 4 * DST_STEP
    with tla.scope.cube():
        src = tla.alloc((SRC_SLOTS, 16, 16), tl.float16, space="l1")
        init = tla.alloc((DST_SLOTS, 16, 16), tl.float16, space="l1")
        eye_l1 = tla.alloc((1, 1, 16, 16), tl.float16, space="l1")
        root = tla.alloc((DST_SLOTS, 1, 16, 16), tl.float16, space="l0b" if TO_B else "l0a")
        dst = tla.reshape(root, (DST_SLOTS, 1, 1, 16, 16))
        eye = tla.alloc((1, 1, 16, 16), tl.float16, space="l0a" if TO_B else "l0b")
        acc = tla.alloc((1, 1, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(Data, (SRC_SLOTS, 16, 16)), src)
        tla.load_gm_to_l1(tla.view(Initial, (DST_SLOTS, 16, 16)), init)
        tla.load_gm_to_l1(tla.view(Identity, (16, 16)), eye_l1)
        tla.sync.barrier(tla.sync.PIPE.ALL)
        tla.copy_l1_to_l0(init, dst, tla.CopyL1ToL0Params(actual_batch=DST_SLOTS))
        tla.copy_l1_to_l0(eye_l1, eye)
        tla.sync.barrier(tla.sync.PIPE.ALL)
        d = tla.subview(dst, (1, 0, 0, 0, 0), (4, 1, 1, 16, 16), (DST_STEP, 1, 1, 1, 1))
        for group in tla.static_range(3):
            s = tla.subview(src, (1 + group * 4 * SRC_STEP, 0, 0), (4, 16, 16), (SRC_STEP, 1, 1))
            n = tla.minimum(4, Total - 4 * group)
            tla.copy_l1_to_l0(s, d, tla.CopyL1ToL0Params(actual_batch=n))
            tla.sync.barrier(tla.sync.PIPE.ALL)
            for slot in range(DST_SLOTS):
                tile = tla.subview(root, (slot, 0, 0, 0), (1, 1, 16, 16))
                if TO_B:
                    tla.cube.mmad_l0(eye, tile, acc, m=16, n=16, k=16, clear_accum=True,
                                     unit_flag_ctrl=tla.UnitFlagMode.DISABLED)
                else:
                    tla.cube.mmad_l0(tile, eye, acc, m=16, n=16, k=16, clear_accum=True,
                                     unit_flag_ctrl=tla.UnitFlagMode.DISABLED)
                tla.sync.barrier(tla.sync.PIPE.ALL)
                out = tla.view(Out + (group * DST_SLOTS + slot) * 256, (16, 16))
                tla.store_l0c_to_gm(acc, out, tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                                      unit_flag_ctrl=tla.UnitFlagMode.DISABLED))
                tla.sync.barrier(tla.sync.PIPE.ALL)


def _spec(case):
    if case["kind"] == "l12l0_batch":
        signature = dict(Data="*fp16", Initial="*fp16", Identity="*fp16", Out="*fp32", Total="i32")
        return l12l0_batch_mmad, signature, {key: case[key] for key in ("TO_B", "SRC_STEP", "DST_STEP")}
    if case["kind"] == "l12l0_view":
        element = "bf16" if case["dtype"] == "bfloat16" else "fp16"
        signature = dict(Data="*" + element, Identity="*" + element, C="*fp32")
        fn = l12l0_source_mmad
        if case["endpoint"] == "dst":
            signature["Initial"] = "*" + element
            fn = l12l0_destination_mmad
        constants = {key: case[key] for key in ("TO_B", "ROWS", "COLS", "TILE_ROWS", "TILE_COLS", "ROW", "COL")}
        return fn, signature, constants
    if case["kind"] == "subview":
        signature = dict(A="*fp16", B="*fp16", TileOut="*fp32", FullOut="*fp32", a_offset="i32", b_offset="i32",
                         c_offset="i32")
        return subview_mmad, signature, dict(DYNAMIC=case["DYNAMIC"])
    signature = dict(A="*fp16", B="*fp16", C="*fp32", Clear="i1")
    constants = {name: case[name] for name in ("M", "N", "K", "L0_K", "SLOTS")}
    constants["SECOND_PASS"] = case["kind"] == "dynamic"
    return plain_mmad, signature, constants


def _ops(text):
    return re.findall(r"hivm\.hir\.mmadL0[^\n]*", text)


def _inputs(op):
    return re.search(r"ins\((.*?) :", op).group(1).split(", ")


def _check_l12l0_view_ir(text, case, ops, values, bools):
    destination = case["endpoint"] == "dst"
    to_b = case["TO_B"]
    rows, cols = case["ROWS"], case["COLS"]
    tr, tc, row, col = (case[key] for key in ("TILE_ROWS", "TILE_COLS", "ROW", "COL"))
    elem = "bf16" if case["dtype"] == "bfloat16" else "f16"
    copies = re.findall(r"hivm\.hir\.l12l0[^\n]*", text)
    assert len(copies) == (3 if destination else 2)
    copy = copies[-1] if destination else copies[1 if to_b else 0]
    operands = re.search(r"ins\((.*?)\)\s+outs\((.*?)\)\s*\{", copy)
    assert operands, copy
    src, dst = operands.groups()
    assert f"{tc // 16}x{tr // 16}x16x16x{elem}" in src
    outer, inner = (tr // 16, tc // 16) if to_b else (tc // 16, tr // 16)
    assert f"{outer}x{inner}x16x16x{elem}" in dst
    assert "address_space<cbuf>" in src
    assert ("address_space<cb>" if to_b else "address_space<ca>") in dst
    if destination:
        pitch = (cols if to_b else rows) * 16
        offset = row * cols + col * 16 if to_b else col * rows + row * 16
        descriptor = dst
    else:
        pitch, offset = rows * 16, col * rows + row * 16
        descriptor = src
    assert f"strided<[{pitch}, 256, 16, 1], offset: {offset}>" in descriptor, copy
    tile_k, tile_mn = (tr, tc) if to_b else (tc, tr)
    for name, value in (("k_part_idx", 0), ("k_part_loop", 1), ("k_part", tile_k), ("k_part_ceil", tile_k),
                        ("k_part_actual", tile_k), ("m", tile_mn)):
        assert re.search(rf"\b{name} = {value}\b", copy), copy
    assert "scf.for" not in text and text.count("hivm.hir.fixpipe ") == 1
    assert len(ops) == 1
    args = _inputs(ops[0])
    m, n = (rows, cols) if destination else (tr, tc)
    k = m if to_b else n
    assert tuple(values[v] for v in args[2:5]) == (m, k, n)
    assert bools[args[5]] is False and bools[args[6]] is False and bools[args[7]] is True
    copy_dst = dst.split(" :", 1)[0]
    if destination:
        assert args[1 if to_b else 0] != copy_dst, "MMAD must inspect the whole L0 parent"
    else:
        assert args[1 if to_b else 0] == copy_dst, "MMAD must read the compact copy destination"


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    fn, signature, constants = _spec(case)
    root = output_dir or os.environ.get("TLA_IR_DUMP_DIR")
    dump = Path(root) / case["name"] if root else None
    module, ctx = compile_offline(fn, signature, constants, output_dir=dump, options=A5Options())
    ast = str(module)
    if case["kind"] == "subview":
        assert ast.count("memref.subview") == 5
        assert ast.count("memref.alloc()") == 5, "subviews must alias their parent allocations"
    elif case["kind"] == "l12l0_view":
        assert ast.count("memref.subview") == 1
        assert ast.count("memref.alloc()") == (6 if case["endpoint"] == "dst" else 5)
    text = str(lower_offline(module, ctx, output_dir=dump))
    ops = _ops(text)
    assert "hivm.hir.l12bt" not in text and "address_space<biasbuf>" not in text
    assert "hivm.hir.mmadL1" not in text and "!tt.ptr" not in text and "tensor<" not in text
    assert "hivm.hir.pipe_barrier[<PIPE_M>]" not in text
    assert all(" bias(" not in op and "unit_flag<DISABLED>" in op for op in ops)
    values = {name: int(value) for name, value in re.findall(r"(%[\w]+) = arith\.constant (\d+) : index", text)}
    bools = {name: value == "true" for name, value in re.findall(r"(%[\w]+) = arith\.constant (true|false)", text)}
    if case["kind"] == "l12l0_batch":
        copies = re.findall(r"hivm\.hir\.l12l0[^\n]*", text)
        assert len(copies) == 5 and sum("actual_batch = %" in op for op in copies) == 4
        for op in copies[-3:]:
            assert "4x16x16xf16" in op and "4x1x1x16x16xf16" in op
            assert "k_part = 16" in op and "m = 16" in op
    elif case["kind"] == "l12l0_view":
        _check_l12l0_view_ir(text, case, ops, values, bools)
    elif case["kind"] == "subview":
        assert len(ops) == 3 and text.count("hivm.hir.l12l0 ") == 2
        assert text.count("hivm.hir.fixpipe ") == 2
        for op, k in zip(ops, (96, 96, 32)):
            args = _inputs(op)
            assert tuple(values[v] for v in args[2:5]) == (32, k, 64)
            assert bools[args[5]] is False and bools[args[6]] is False
            assert bools[args[7]] is True
        if case["DYNAMIC"]:
            assert ops[-1].count("offset: ?>") == 3, "runtime subview offsets must stay dynamic"
    else:
        count = case["K"] // case["L0_K"]
        passes = 2 if case["kind"] == "dynamic" else 1
        assert len(ops) == count * passes
        assert text.count("hivm.hir.l12l0 ") == len(ops) * 2
        assert text.count("hivm.hir.fixpipe ") == 1
        dynamic_args = set(re.findall(r"(%[\w]+): i1\b", text))
        for index, op in enumerate(ops):
            args = _inputs(op)
            assert tuple(values[v] for v in args[2:5]) == (case["M"], case["L0_K"], case["N"])
            assert bools[args[5]] is False and bools[args[6]] is False
            if passes == 2 and index == count:
                assert args[7] in dynamic_args and args[7] not in bools, "clear must remain a runtime i1"
            else:
                assert bools[args[7]] == (index % count == 0)
        pairs = [tuple(_inputs(op)[:2]) for op in ops]
        assert len(set(pairs)) == min(case["SLOTS"], count)
    print("IR PASS:", case["name"], "dump:", str(dump) if dump else "disabled")


def _inputs_cpu(case, seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    if case["kind"] == "subview":
        shapes = ((32, 96), (96, 64))
    else:
        shapes = ((case["M"], case["K"]), (case["K"], case["N"]))
    if case["name"].startswith("capacity_"):
        # Dyadic data gives exact FP16 products and FP32 sums even at K=2048.
        # Boundary cases assert exact equality; other groups retain the
        # existing 1e-4 accuracy tolerance on random FP16 data.
        return tuple((torch.randint(-4, 5, shape, generator=generator).float() / 8).half() for shape in shapes)
    return tuple(torch.randn(shape, generator=generator).half() for shape in shapes)


def _guarded_output(rows, cols):
    import torch

    guard = 256  # 1024 bytes, keeps the output pointer aligned.
    sentinel = -8191.0
    storage = torch.full((rows * cols + 2 * guard, ), sentinel, dtype=torch.float32, device="npu")
    output = storage[guard:-guard].view(rows, cols)
    output.fill_(float("nan"))
    return output, storage


def _check_guards(storage):
    host = storage.cpu()
    assert (host[:256] == -8191.0).all() and (host[-256:] == -8191.0).all(), "GM output guard overwritten"


def _invoke(fn, args, constants, compile_only):
    options = dict(unit_flag=False, sync_solver=False)
    if compile_only:
        return fn.warmup(*args, grid=(1, ), **constants, **options)
    return fn[(1, )](*args, **constants, **options)


def _compare(actual, expected, exact=False):
    import torch

    assert torch.isfinite(actual).all(), "unwritten or non-finite MMAD output"
    if exact:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    return (actual - expected).abs().max().item()


def _l12l0_view_inputs(case, seed):
    import torch

    dtype = getattr(torch, case["dtype"])
    rows, cols, tr, tc, row, col = (case[key] for key in ("ROWS", "COLS", "TILE_ROWS", "TILE_COLS", "ROW", "COL"))

    def pattern(r, c, negative=False):
        # All values are exactly representable in both FP16 and BF16. Different
        # row/column weights expose swapped axes and misplaced fractal blocks.
        rr, cc = torch.arange(r)[:, None], torch.arange(c)[None, :]
        values = ((rr * 17 + cc * 7 + seed * 11) % 127 + 1).float() / 16
        return (-values if negative else values).to(dtype)

    if case["endpoint"] == "src":
        data = pattern(rows, cols)
        k = tr if case["TO_B"] else tc
        expected = data[row:row + tr, col:col + tc].float().clone()
        return (data, torch.eye(k, dtype=dtype)), expected
    data, initial = pattern(tr, tc), pattern(rows, cols, negative=True)
    k = rows if case["TO_B"] else cols
    expected = initial.float().clone()
    expected[row:row + tr, col:col + tc] = data.float()
    return (data, initial, torch.eye(k, dtype=dtype)), expected


def _launch_l12l0_view(case, seed, compile_only=False):
    import torch

    fn, _, constants = _spec(case)
    inputs, expected = _l12l0_view_inputs(case, seed)
    out, storage = _guarded_output(*expected.shape)
    args = tuple(value.npu() for value in inputs) + (out, )
    kernel = _invoke(fn, args, constants, compile_only)
    if not compile_only:
        # Keep the device inputs alive until the asynchronous launch completes.
        torch.npu.synchronize()
    return kernel, out, storage, expected


def _launch_l12l0_batch(case, total, compile_only=False):
    import torch

    src_step, dst_step = case["SRC_STEP"], case["DST_STEP"]
    src_slots, dst_slots = 1 + 12 * src_step, 1 + 4 * dst_step
    # Slot and in-matrix position are both visible through the identity GEMM.
    pattern = torch.arange(256).reshape(16, 16) % 31
    data = (32 * torch.arange(src_slots)[:, None, None] + pattern).half()
    initial = -(1 + 32 * torch.arange(dst_slots)[:, None, None] + pattern).half()
    snapshots = []
    slots = initial.clone()
    for start in (0, 4, 8):
        for i in range(min(4, total - start)):
            slots[1 + i * dst_step] = data[1 + (start + i) * src_step]
        snapshots.append(slots.clone())
    expected = torch.stack(snapshots).float().reshape(-1, 16)
    out, storage = _guarded_output(*expected.shape)
    fn, _, constants = _spec(case)
    args = (data.npu(), initial.npu(), torch.eye(16, dtype=torch.float16).npu(), out, total)
    kernel = _invoke(fn, args, constants, compile_only)
    if not compile_only:
        torch.npu.synchronize()
    return kernel, out, storage, expected


@matrix(CONFIGS)
def test_compile(case):
    import torch_npu  # noqa: F401

    if case["kind"] == "l12l0_batch":
        kernel, _, _, _ = _launch_l12l0_batch(case, 10, compile_only=True)
        check_kernel_binary(kernel)
        return
    if case["kind"] == "l12l0_view":
        kernel, _, _, _ = _launch_l12l0_view(case, 7, compile_only=True)
        check_kernel_binary(kernel)
        return
    fn, _, constants = _spec(case)
    a, b = _inputs_cpu(case, 7)
    x, y = a.npu(), b.npu()
    if case["kind"] == "subview":
        out, _ = _guarded_output(32, 64)
        full, _ = _guarded_output(32, 128)
        args = (x, y, out, full, 1, 3, 2)
    else:
        out, _ = _guarded_output(case["M"], case["N"])
        args = (x, y, out, True)  # bool scalar -> runtime i1
    check_kernel_binary(_invoke(fn, args, constants, compile_only=True))


@matrix(CONFIGS)
def test_run(case):
    import torch
    import torch_npu  # noqa: F401

    if case["kind"] == "l12l0_batch":
        for total in (10, 9):  # Same kernel, final group of two or one.
            _, out, storage, expected = _launch_l12l0_batch(case, total)
            _compare(out.cpu(), expected, exact=True)
            _check_guards(storage)
        return
    if case["kind"] == "l12l0_view":
        for seed in (7, 19):
            print("RUN:", case["name"], "seed", seed, flush=True)
            _, out, storage, expected = _launch_l12l0_view(case, seed)
            error = _compare(out.cpu(), expected, exact=True)
            _check_guards(storage)
            print("ACCURACY PASS:", case["name"], "seed", seed, "max_abs_error", error)
        return
    fn, _, constants = _spec(case)
    binary = None
    for seed in (7, 19):
        a, b = _inputs_cpu(case, seed)
        x, y = a.npu(), b.npu()
        product = a.float() @ b.float()
        if case["kind"] == "subview":
            offsets = ((1, 3, 1), (3, 1, 3), (2, 4, 2)) if case["DYNAMIC"] else ((1, 3, 2), )
            out, out_storage = _guarded_output(32, 64)
            full, full_storage = _guarded_output(32, 128)
            for ao, bo, co in offsets:
                out.fill_(float("nan"))
                full.fill_(float("nan"))
                print("RUN:", case["name"], "seed", seed, "offsets", (ao, bo, co), flush=True)
                kernel = _invoke(fn, (x, y, out, full, ao, bo, co), constants, compile_only=False)
                torch.npu.synchronize()
                tile = a[:, ao * 16:ao * 16 + 32].float() @ b[bo * 16:bo * 16 + 32].float()
                expected = torch.cat((product, product), dim=1)
                expected[:, co * 16:co * 16 + 64] = tile
                tile_error = _compare(out.cpu(), tile)
                full_error = _compare(full.cpu(), expected)
                _check_guards(out_storage)
                _check_guards(full_storage)
                if binary is None:
                    binary = kernel.asm["npubin"]
                assert kernel.asm["npubin"] == binary, "runtime offsets unexpectedly changed the binary"
                print("ACCURACY PASS:", case["name"], "seed", seed, "offsets", (ao, bo, co), "tile_max_abs_error",
                      tile_error, "parent_max_abs_error", full_error)
        else:
            # Exercise both dynamic clear values repeatedly using the same
            # input/output pointers and the same compiled binary.
            clears = (True, False, True, False) if case["kind"] == "dynamic" else (True, )
            out, storage = _guarded_output(case["M"], case["N"])
            for clear in clears:
                out.fill_(float("nan"))
                print("RUN:", case["name"], "seed", seed, "clear", clear, flush=True)
                kernel = _invoke(fn, (x, y, out, clear), constants, compile_only=False)
                torch.npu.synchronize()
                expected = product * (2 if case["kind"] == "dynamic" and not clear else 1)
                error = _compare(out.cpu(), expected, exact=case["name"].startswith("capacity_"))
                _check_guards(storage)
                if binary is None:
                    binary = kernel.asm["npubin"]
                assert kernel.asm["npubin"] == binary, "runtime clear unexpectedly changed the binary"
                print("ACCURACY PASS:", case["name"], "seed", seed, "clear", clear, "max_abs_error", error)


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in CONFIGS:
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)


if __name__ == "__main__":
    main()
