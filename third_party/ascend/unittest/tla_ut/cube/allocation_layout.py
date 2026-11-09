"""Checked allocation constructors preserve physical buffer and view semantics."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import compile_offline, lower_offline


@tla.jit
def _allocate(SHAPE: tl.constexpr, DTYPE: tl.constexpr, SPACE: tl.constexpr, LAYOUT: tl.constexpr, ALIAS: tl.constexpr,
              PHYSICAL: tl.constexpr):
    with tla.scope.cube():
        if LAYOUT is None:
            buf = tla.alloc(SHAPE, DTYPE, SPACE, 2)
        elif ALIAS:
            buf = tla.alloc_buffer(SHAPE, DTYPE, SPACE, 2, layout=LAYOUT)
        else:
            buf = tla.alloc(SHAPE, DTYPE, SPACE, 2, layout=LAYOUT)
        tl.static_assert(buf.shape == PHYSICAL)


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("space,dtype,logical,physical", [
    (tla.space.l1, tl.float16, (32, 48), (3, 2, 16, 16)),
    (tla.space.l1, tl.bfloat16, (48, 32), (2, 3, 16, 16)),
    (tla.space.l1, tl.float32, (32, 24), (3, 2, 16, 8)),
    (tla.space.l0c, tl.float32, (32, 48), (3, 2, 16, 16)),
    (tla.space.l0c, tl.int32, (48, 32), (2, 3, 16, 16)),
])
def test_nz_matches_physical_allocation_ir(alias, space, dtype, logical, physical):
    allocations = []
    for shape, layout in ((logical, tla.layout.nz), (physical, None)):
        module, _ = compile_offline(
            _allocate, {}, dict(SHAPE=shape, DTYPE=dtype, SPACE=space, LAYOUT=layout, ALIAS=alias, PHYSICAL=physical))
        text = str(module)
        allocations.append(re.findall(r"memref\.alloc\(\)[^\n]*", text))
        assert len(allocations[-1]) == 1
        assert text.count("{mem_unique}") == 1
        assert text.count("hivm.multi_buffer = 2") == 1
    assert allocations[0] == allocations[1]


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("space,shape", [
    (tla.space.ub, (7, 13)),
    (tla.space.l1, (32, 48)),
    (tla.space.l0c, (32, 48)),
    (tla.space.l0c, (3, 2, 16, 16)),
])
def test_nd_preserves_existing_shape_ir(alias, space, shape):
    allocations = []
    for layout in (None, tla.layout.nd):
        module, _ = compile_offline(
            _allocate, {}, dict(SHAPE=shape, DTYPE=tl.float32, SPACE=space, LAYOUT=layout, ALIAS=alias, PHYSICAL=shape))
        allocations.append(re.findall(r"memref\.alloc\(\)[^\n]*", str(module)))
    assert len(allocations[0]) == 1
    assert allocations[0] == allocations[1]


@tla.jit
def _view_product(A, B, C, COMPACT: tl.constexpr):
    with tla.scope.cube():
        root = tla.alloc((128, 64), tl.float16, space=tla.space.l1, layout=tla.layout.nz)
        tla.load_gm_to_l1(tla.view(A, (128, 64)), root)
        if COMPACT:
            flat = tla.reshape(root, (8192, ))
            prefix = tla.subview(flat, offsets=(128, ), sizes=(2048, ))
            a = tla.reshape(prefix, (2, 4, 16, 16))
        else:
            a = tla.subview(root, offsets=(1, 1, 0, 0), sizes=(2, 4, 16, 16))
        b = tla.alloc((32, 48), tl.float16, space=tla.space.l1, layout=tla.layout.nz)
        acc = tla.alloc((64, 48), tl.float32, space=tla.space.l0c, layout=tla.layout.nz)
        tla.load_gm_to_l1(tla.view(B, (32, 48)), b)
        tla.cube.mmad(a, b, acc, clear_accum=True)
        tla.store_l0c_to_gm(acc, tla.view(C, (64, 48)), tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND))


@pytest.mark.parametrize("compact", [False, True])
def test_physical_views_ir(compact):
    module, ctx = compile_offline(_view_product, dict(A="*fp16", B="*fp16", C="*fp32"), dict(COMPACT=compact))
    ast = str(module)
    pitch = 1024 if compact else 2048
    assert f"strided<[{pitch}, 256, 16, 1], offset: ?>" in ast
    assert ast.count("memref.subview") == 1
    assert ("memref.collapse_shape" in ast) == compact
    assert ("memref.expand_shape" in ast) == compact
    text = str(lower_offline(module, ctx))
    assert text.count("memref.alloc()") == 3, "views must alias the original allocation"
    assert text.count("hivm.hir.mmadL1 ") == 1
    assert text.count("hivm.hir.nd2nz ") == 2
    assert text.count("hivm.hir.fixpipe ") == 1
    if compact:
        # expand_shape keeps a dynamic-offset cast of the constant-offset prefix.
        assert "memref<2048xf16, strided<[1], offset: 128>" in text
        assert "strided<[1024, 256, 16, 1], offset: ?>" in text
    else:
        assert "strided<[2048, 256, 16, 1], offset: 2304>" in text


@pytest.mark.parametrize("alloc", [tla.alloc, tla.alloc_buffer])
@pytest.mark.parametrize("layout", ["nz", None, 1, tla.space.l1])
def test_invalid_layout(alloc, layout):
    with pytest.raises(TypeError, match="expected a tla.layout member"):
        alloc((32, 48), tl.float16, space=tla.space.l1, layout=layout, _semantic=object())


@pytest.mark.parametrize("alloc", [tla.alloc, tla.alloc_buffer])
@pytest.mark.parametrize("shape,dtype,space,message", [
    ((512, ), tl.float16, tla.space.l1, "two-dimensional logical shape"),
    ((3, 2, 16, 16), tl.float16, tla.space.l1, "two-dimensional logical shape"),
    ((0, 32), tl.float16, tla.space.l1, "positive compile-time integers"),
    ((16, -32), tl.float16, tla.space.l1, "positive compile-time integers"),
    ((True, 32), tl.float16, tla.space.l1, "positive compile-time integers"),
    ((16, 32.0), tl.float16, tla.space.l1, "positive compile-time integers"),
    ((17, 32), tl.float16, tla.space.l1, "M divisible by 16 and N divisible by 16"),
    ((16, 24), tl.float16, tla.space.l1, "N divisible by 16"),
    ((16, 12), tl.float32, tla.space.l1, "N divisible by 8"),
    ((16, 24), tl.float32, tla.space.l0c, "N divisible by 16"),
    ((16, 32), tl.float16, tla.space.ub, "NZ allocation supports"),
    ((16, 32), tl.float16, tla.space.l0a, "NZ allocation supports"),
    ((16, 32), tl.float16, tla.space.l0b, "NZ allocation supports"),
    ((16, 32), tl.int8, tla.space.l1, "NZ allocation supports"),
    ((16, 32), tl.float16, tla.space.l0c, "NZ allocation supports"),
])
def test_invalid_nz_allocation(alloc, shape, dtype, space, message):
    # Validation must finish before any builder operation is attempted.
    with pytest.raises(ValueError, match=message):
        alloc(shape, dtype, space, layout=tla.layout.nz, _semantic=object())
