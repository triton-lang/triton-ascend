"""Ascend950 same-type Cube DMA, with independent ND and NZ inputs."""
import sys
from pathlib import Path

_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import Options, check_kernel_binary, compile_offline, lower_offline, matrix


class A5Options(Options):
    arch = "Ascend950DT_9582"


DTYPES = {"fp16": (tl.float16, torch.float16), "bf16": (tl.bfloat16, torch.bfloat16),
          "fp32": (tl.float32, torch.float32)}
CONFIGS = [
    dict(name=f"{route}_{dtype}", route=route, dtype=dtype)
    for route in ("gm", "nz", "strided")
    for dtype in DTYPES
]


@tla.jit
def transfer(X, Y, DTYPE: tl.constexpr, C: tl.constexpr, ROUTE: tl.constexpr):
    # Use two M blocks and three N blocks to distinguish tile traversal order.
    M: tl.constexpr = 32
    N: tl.constexpr = 3 * C
    NZ: tl.constexpr = (3, 2, 16, C)
    backing = tla.alloc((4, 2, 16, C), DTYPE, space="l1")
    l1 = tla.subview(backing, (1, 0, 0, 0), NZ)
    out_backing = tla.alloc((M, N + C), DTYPE)
    out = tla.subview(out_backing, (0, C), (M, N))

    if ROUTE != "gm":
        with tla.scope.vector():
            if tla.sub_block_id() == 0:
                if ROUTE == "nz":
                    src = tla.alloc(NZ, DTYPE)
                    tla.load_gm_to_ub(tla.view(X, NZ), src)
                    tla.copy_ub_to_l1(src, l1)
                else:
                    rows: tl.constexpr = 3 * 2 * 16
                    src_backing = tla.alloc((rows, 2 * C), DTYPE)
                    tla.load_gm_to_ub(tla.view(X, (rows, 2 * C)), src_backing)
                    src = tla.subview(src_backing, (0, C), (rows, C))
                    tla.copy_ub_to_l1(src, tla.reshape(l1, (rows, C)))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)

    with tla.scope.cube():
        if ROUTE == "gm":
            src = tla.view(X, (M, N), strides=(N + C, 1), offsets=(0, C))
            tla.load_gm_to_l1(src, l1)
        else:
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
        tla.copy_l1_to_ub(l1, out)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE1, 1)

    with tla.scope.vector():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 1)
        if tla.sub_block_id() == 0:
            tla.store_ub_to_gm(out, tla.view(Y, (M, N)))


def constants(case):
    dtype = case["dtype"]
    return {"DTYPE": DTYPES[dtype][0], "C": 8 if dtype == "fp32" else 16, "ROUTE": case["route"]}


@matrix(CONFIGS)
def test_ir(case):
    ptr = "*" + case["dtype"]
    module, ctx = compile_offline(transfer, {"X": ptr, "Y": ptr}, constants(case), options=A5Options())
    text = str(lower_offline(module, ctx))
    assert "hivm.hir.l12ub " in text
    assert "hivm.hir.sync_block_set" in text and "hivm.hir.sync_block_wait" in text
    expected_op = "nd2nz" if case["route"] == "gm" else "copy"
    assert f"hivm.hir.{expected_op} " in text


def launch(case, compile_only=False):
    args = constants(case)
    c = args["C"]
    dtype = DTYPES[case["dtype"]][1]
    # Exact binary fractions, distinguish rows, columns and regions left unwritten.
    nd = (((torch.arange(32 * 3 * c).reshape(32, 3 * c) * 17) % 1009) / 16).to(dtype)
    if case["route"] == "gm":
        data = torch.full((32, 4 * c), -99, dtype=dtype)
        data[:, c:] = nd
    else:
        nz = nd.reshape(2, 16, 3, c).permute(2, 0, 1, 3).contiguous()
        if case["route"] == "nz":
            data = nz
        else:
            data = torch.full((3 * 2 * 16, 2 * c), -99, dtype=dtype)
            data[:, c:] = nz.reshape(-1, c)
    x = data.npu()
    y = torch.full_like(nd, float("nan"), device="npu")
    if compile_only:
        return transfer.warmup(x, y, grid=(1,), **args)
    transfer[(1,)](x, y, **args)
    torch.npu.synchronize()
    return y.cpu(), nd


@matrix(CONFIGS)
def test_compile(case):
    check_kernel_binary(launch(case, compile_only=True))


@matrix(CONFIGS)
def test_run(case):
    actual, expected = launch(case)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    print(case["name"], "max_abs_error", (actual.float() - expected.float()).abs().max().item())


@tla.jit
def dma_views(src, dst, ROUTE: tl.constexpr):
    if ROUTE == "gm":
        tla.load_gm_to_l1(src, dst)
    elif ROUTE == "ub":
        tla.copy_ub_to_l1(src, dst)
    else:
        tla.copy_l1_to_ub(src, dst)


@pytest.mark.parametrize("route,src,dst,message", [
    ("gm", ((16, 8), "gm", (9, 1), 0), ((1, 1, 16, 8), "l1", None, 0), "row strides"),
    ("gm", ((16, 8), "gm", None, 1), ((1, 1, 16, 8), "l1", None, 0), "offset"),
    ("gm", ((16, 16), "gm", None, 0), ((1, 1, 16, 16), "l1", None, 0), "contiguous NZ destination"),
    ("gm", ((16, 9), "gm", None, 0), ((1, 1, 16, 8), "l1", None, 0), "N divisible by 8"),
    ("ub", ((2, 7), "ub", None, 0), ((2, 7), "l1", None, 0), "32B-aligned width"),
    ("ub", ((2, 8), "ub", None, 0), ((2, 8), "l1", None, 1), "offset"),
    ("ub", ((2, 16), "ub", (8, 1), 0), ((2, 16), "l1", None, 0), "row strides"),
    ("ub", ((8190, 8), "ub", (16, 1), 0), ((8190, 8), "l1", None, 0), "8189 rows"),
    ("ub", ((2, 8), "ub", (65536, 1), 0), ((2, 8), "l1", None, 0), "65535 elements"),
    ("l1", ((1, 1, 16, 8), "l1", None, 0), ((8, 16), "ub", None, 0), "matching 2D ND"),
    ("l1", ((1, 1, 16, 8), "l1", None, 0), ((16, 8), "ub", (8, 2), 0), "contiguous last dimension"),
    ("l1", ((1, 1, 16, 16), "l1", None, 0), ((16, 16), "ub", None, 0), "contiguous .* NZ source blocks"),
    ("l1", ((1, 1, 16, 8), "l1", None, 0), ((16, 8), "ub", (10, 1), 0), "row strides"),
])
def test_invalid_layout(route, src, dst, message):
    def buf(desc):
        shape, space, strides, offset = desc
        return tla.buffer_type(shape, tl.float32, space, strides, offset)
    with pytest.raises(Exception, match=message):
        compile_offline(dma_views, {"src": buf(src), "dst": buf(dst)}, {"ROUTE": route}, options=A5Options())


def test_mixed_dtype():
    src = tla.buffer_type((16, 8), tl.float32, "gm")
    dst = tla.buffer_type((1, 1, 16, 8), tl.float16, "l1")
    with pytest.raises(Exception, match="matching float32 source and destination"):
        compile_offline(dma_views, {"src": src, "dst": dst}, {"ROUTE": "gm"}, options=A5Options())


@tla.jit
def invalid_subview(OFFSET: tl.constexpr):
    backing = tla.alloc((2, 16), tl.float32)
    src = tla.subview(backing, (0, OFFSET), (2, 8))
    dst = tla.alloc((2, 8), tl.float32, space="l1")
    with tla.scope.vector():
        tla.copy_ub_to_l1(src, dst)


@pytest.mark.parametrize("offset,message", [(1, "offset"), (16, "outside the source buffer")])
def test_subview_offset(offset, message):
    with pytest.raises(Exception, match=message):
        compile_offline(invalid_subview, {}, {"OFFSET": offset}, options=A5Options())
