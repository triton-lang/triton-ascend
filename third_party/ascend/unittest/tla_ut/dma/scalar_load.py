"""Scalar reads and writes through single-element GM and UB buffers."""
import sys
from pathlib import Path

_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from triton._C.libtriton import ir
from triton._C.libtriton import ascend as ascend_binding
from triton.language.extra.tla.compiler import check_ttadapter
from utils.tla_test_utils import compile_offline, lower_offline, make_context, check_kernel_binary, get_stages


@tla.jit
def gather_scalar_index(IDS, X, Y, first, second, enabled, MASKED: tla.constexpr):
    with tla.scope.vector():
        # Model DSA's packed i64 index read and its subsequent GM DMA address.
        address = (IDS + first) + second
        if MASKED:
            packed = tl.load(address, enabled, 0)
        else:
            packed = tla.load_scalar(tla.view(address, (1, )))
        index = (packed & 0xFFFFFFFF).to(tl.int32)
        ub = tla.alloc((64, ), tl.float32)
        tla.load_gm_to_ub(tla.view(X + index * 64, (64, )), ub)
        tla.store_ub_to_gm(ub, tla.view(Y, (64, )))


SIGNATURE = {"IDS": "*i64", "X": "*fp32", "Y": "*fp32", "first": "i32", "second": "i32", "enabled": "i1"}


@pytest.mark.parametrize("masked", [False, True])
def test_ir(masked, output_dir=None):
    mod, ctx = compile_offline(gather_scalar_index, SIGNATURE, {"MASKED": masked})
    text = str(lower_offline(mod, ctx, output_dir=output_dir))
    assert "tt.load" not in text and "tt.addptr" not in text
    assert text.count("memref.load ") == 1
    assert "hivm.hir.load" in text and "hivm.hir.store" in text
    if masked:
        # The actual GM read must be inside the active branch.
        assert "memref.load" not in text.split("scf.if", 1)[0]
        then_body, else_body = text.split("scf.if", 1)[1].split("} else {", 1)
        assert "memref.load" in then_body
        assert "memref.load" not in else_body


@pytest.mark.parametrize("operand,error,match", [
    (tl.tensor(None, tl.pointer_type(tl.int32)), TypeError, "expects a GM or UB tla.buffer"),
    (tla.buffer(None, tla.buffer_type((1, ), tl.int32, "l1")), TypeError, "expects a GM or UB tla.buffer"),
    (tla.buffer(None, tla.buffer_type((2, 1), tl.int32)), ValueError, "single-element"),
    (tla.buffer(None, tla.buffer_type((None, ), tl.int32)), ValueError, "single-element"),
    (tla.buffer(None, tla.buffer_type((1, ), tl.int1)), TypeError, "byte-addressable"),
])
def test_scalar_rejects_unsupported_buffer(operand, error, match):
    for operation, args in ((tla.load_scalar, (operand, )), (tla.store_scalar, (operand, 1))):
        with pytest.raises(error, match=match):
            operation(*args, _semantic=object())


def _convert_ir(tmp_path, body):
    path = tmp_path / "input.mlir"
    path.write_text(body)
    ctx = make_context()
    mod = ir.parse_mlir_module(str(path), ctx)
    pm = ir.pass_manager(ctx)
    ascend_binding.passes.ttir.add_tla_to_linalg(pm)
    pm.run(mod, "module")
    assert mod.verify()
    check_ttadapter(mod)
    return str(mod)


@pytest.mark.parametrize("dtype", ["i1", "i8", "i32", "i64", "f16", "bf16", "f32"])
def test_scalar_types(tmp_path, dtype):
    if dtype == "i1":
        # tt.load on !tt.ptr<i1> lowers to a memref.load whose i1 result does
        # not match the i8-backed memref, so the MLIR verifier rejects it.
        # The fix belongs to the shared TritonToLinalg conversion, outside the
        # TLA tree, so this parameter stays parked until that fix lands.
        pytest.xfail("scalar i1 load needs a TritonToLinalg fix")
    # Keep the loaded value live without adding a scalar-store API dependency.
    text = _convert_ir(
        tmp_path, f"""
module {{
  tt.func public @scalar(%p: !tt.ptr<{dtype}>, %mask: i1, %out: memref<1x{dtype}>) {{
    %v = tt.load %p, %mask : !tt.ptr<{dtype}>
    %c0 = arith.constant 0 : index
    memref.store %v, %out[%c0] : memref<1x{dtype}>
    tt.return
  }}
}}
""")
    assert "tt.load" not in text
    assert text.count("memref.load ") == 1
    assert "scf.if" in text
    if dtype == "i1":
        assert "arith.cmpi ne" in text


@pytest.mark.parametrize("body", [
    "tt.store %p, %value : !tt.ptr<i32>",
    """%ptrs = tt.splat %p : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %values = tt.splat %value : i32 -> tensor<4xi32>
    tt.store %ptrs, %values : tensor<4x!tt.ptr<i32>>""",
    """%ptrs = tt.splat %p : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %values = tt.load %ptrs : tensor<4x!tt.ptr<i32>>""",
])
def test_unsupported_memory_operations_still_fail(tmp_path, body):
    # Relaxed dialect legality must not silently accept an unsupported pointer
    # use or a Block Tensor load.
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _convert_ir(
            tmp_path, f"""
module {{
  tt.func public @unsupported(%p: !tt.ptr<i32>, %value: i32) {{
    {body}
    tt.return
  }}
}}
""")


def _launch(masked, compile_only=False):
    ids = torch.tensor([(2 << 32) | i for i in range(8)], dtype=torch.int64, device="npu")
    x = torch.arange(8 * 64, dtype=torch.float32, device="npu")
    y = torch.empty((64, ), dtype=torch.float32, device="npu")
    if compile_only:
        kernel = gather_scalar_index.warmup(ids, x, y, 4, -1, True, MASKED=masked, grid=(1, ))
        check_kernel_binary(kernel)
        return
    gather_scalar_index[(1, )](ids, x, y, 4, -1, True, MASKED=masked)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), torch.arange(3 * 64, 4 * 64, dtype=torch.float32), rtol=0, atol=0)
    if masked:
        # A masked-off address is outside the allocation; it must never be read.
        gather_scalar_index[(1, )](ids, x, y, 1 << 28, 0, False, MASKED=True)
        torch.npu.synchronize()
        torch.testing.assert_close(y.cpu(), torch.arange(64, dtype=torch.float32), rtol=0, atol=0)


@pytest.mark.parametrize("masked", [False, True])
def test_compile(masked):
    _launch(masked, compile_only=True)


@pytest.mark.parametrize("masked", [False, True])
def test_run(masked):
    _launch(masked)


@tla.jit
def scalar_roundtrip(IDS, OUT, first, second):
    with tla.scope.vector():
        packed = tla.load_scalar(tla.view((IDS + first) + second, (1, )))
        tla.store_scalar(tla.view(OUT + first, (1, )), packed)


def test_scalar_roundtrip_ir():
    signature = {"IDS": "*i64", "OUT": "*i64", "first": "i32", "second": "i32"}
    mod, ctx = compile_offline(scalar_roundtrip, signature, {})
    text = str(lower_offline(mod, ctx))
    assert text.count("memref.load ") == text.count("memref.store ") == 1
    assert "tt.load" not in text and "tt.store" not in text
    assert "memref.alloc" not in text


def test_scalar_roundtrip_device():
    if not get_stages()[2]:
        pytest.skip("Requires NPU execution")
    packed = (3 << 32) | 7
    ids = torch.tensor([0, packed, 0], dtype=torch.int64, device="npu")
    out = torch.full((3, ), -1, dtype=torch.int64, device="npu")
    scalar_roundtrip[(1, )](ids, out, 2, -1)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.tensor([-1, -1, packed], dtype=torch.int64), rtol=0, atol=0)


@tla.jit
def scalar_buffer_roundtrip(IN, OUT, source_index, row, write_col, read_col, output_index, DTYPE: tl.constexpr):
    with tla.scope.vector():
        src = tla.view(IN, (8, ), strides=(2, ), offsets=(1, ))
        dst = tla.view(OUT, (8, ), strides=(3, ), offsets=(1, ))
        ub = tla.alloc((4, 160), DTYPE)
        value = tla.load_scalar(tla.subview(src, (source_index, ), (1, )))
        tla.store_scalar(tla.subview(ub, (row, write_col), (1, 1)), value)
        value = tla.load_scalar(tla.subview(ub, (row, tla.cast(read_col, tl.uint8)), (1, 1)))
        tla.store_scalar(tla.subview(dst, (output_index, ), (1, )), value)


def test_scalar_buffer_roundtrip_ir():
    signature = dict(IN="*i32", OUT="*i32", source_index="i32", row="i32", write_col="i32", read_col="i32",
                     output_index="i32")
    mod, ctx = compile_offline(scalar_buffer_roundtrip, signature, {"DTYPE": tl.int32})
    text = str(lower_offline(mod, ctx))
    assert "memref.subview" in text and "arith.extui" in text
    assert text.count("memref.alloc") == 1
    for op in ("memref.load", "memref.store"):
        accesses = [line for line in text.splitlines() if f"{op} " in line]
        assert len(accesses) == 2
        assert sum("#hivm.address_space<ub>" in line for line in accesses) == 1


@pytest.mark.parametrize("dtype",
                         [tl.int8, tl.int16, tl.int32, tl.int64, tl.float16, tl.bfloat16, tl.float32, tl.float64])
def test_scalar_buffer_roundtrip_device(dtype):
    if not get_stages()[2]:
        pytest.skip("Requires NPU execution")
    width = dtype.primitive_bitwidth // 8
    host = torch.arange(32 * width, dtype=torch.int64).to(torch.uint8).reshape(32, width)
    src = host.npu()
    dst = torch.full(tuple(host.shape), 0xA5, dtype=torch.uint8, device="npu")
    scalar_buffer_roundtrip[(1, )](triton.reinterpret(src, dtype), triton.reinterpret(dst, dtype), 3, 2, 130, 130, 2,
                                   DTYPE=dtype)
    torch.npu.synchronize()
    expected = torch.full_like(host, 0xA5)
    expected[9] = host[8]
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
