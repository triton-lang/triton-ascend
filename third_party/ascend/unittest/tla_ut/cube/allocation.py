"""TLA allocation contracts and Cube/Vector storage validation.

IR checks cover every mapped allocation space and the reserved scale spaces.
Compile/run checks exercise UB, L1 and L0C, which the current backend can plan.
L0A/L0B/SSBUF/BT/FB IR coverage does not imply backend allocation support.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import get_stages, compile_offline, lower_offline, check_kernel_binary


@tla.jit
def _allocate_only(SPACE: tl.constexpr, VERSIONS: tl.constexpr):
    with tla.scope.cube():
        tla.alloc((16, 16), tl.float32, scope=SPACE, versions=VERSIONS)


@tla.jit
def _copy_two_tiles(X, Y):
    with tla.scope.vector():
        first = tla.alloc((16, 32), tl.float32)
        tla.load_gm_to_ub(tla.view(X, (16, 32)), first)
        tla.store_ub_to_gm(first, tla.view(Y, (16, 32)))
        # Separate logical buffers do not force distinct physical storage.
        second = tla.alloc_buffer((16, 32), tl.float32, scope=tla.AddressSpace.UB)
        tla.load_gm_to_ub(tla.view(X + 512, (16, 32)), second)
        tla.store_ub_to_gm(second, tla.view(Y + 512, (16, 32)))


@tla.jit
def _cube_product(A, B, C):
    with tla.scope.cube():
        a = tla.alloc((2, 1, 16, 16), tl.float16, scope=tla.AddressSpace.L1)
        b = tla.alloc_buffer((4, 2, 16, 16), tl.float16, space=tla.AddressSpace.L1)
        acc = tla.alloc((4, 1, 16, 16), tl.float32, scope=tla.AddressSpace.L0C)
        tla.load_gm_to_l1(tla.view(A, (16, 32)), a)
        tla.load_gm_to_l1(tla.view(B, (32, 64)), b)
        tla.cube.mmad(a, b, acc, clear_accum=True)
        tla.store_l0c_to_gm(acc, tla.view(C, (16, 64)), tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND))


def test_ir():
    spaces = ((tla.AddressSpace.UB, "ub"), (tla.AddressSpace.L1, "cbuf"), (tla.AddressSpace.L0A, "ca"),
              (tla.AddressSpace.L0B, "cb"), (tla.AddressSpace.L0C, "cc"), (tla.AddressSpace.SSBUF, "ssbuf"),
              (tla.AddressSpace.BT, "biasbuf"), (tla.AddressSpace.FB, "fixbuf"))
    for space, native in spaces:
        # Inspect frontend IR: dead allocations can disappear during lowering.
        for spelling in (space, space.value):
            module, _ = compile_offline(_allocate_only, {}, dict(SPACE=spelling, VERSIONS=2))
            text = str(module)
            assert f"#hivm.address_space<{native}>" in text
            assert "mem_unique" not in text
            assert "hivm.multi_buffer = 2" in text
            assert "#hivm.tcore_type<CUBE>" in text

    for fn, signature, allocs, core in (
        (_copy_two_tiles, {"X": "*fp32", "Y": "*fp32"}, 2, "VECTOR"),
        (_cube_product, {"A": "*fp16", "B": "*fp16", "C": "*fp32"}, 3, "CUBE"),
    ):
        module, ctx = compile_offline(fn, signature, {})
        text = str(lower_offline(module, ctx))
        assert text.count("memref.alloc()") == allocs
        assert "mem_unique" not in text
        assert f"#hivm.tcore_type<{core}>" in text
        assert "hivm.vector_function" not in text
        assert "!tt.ptr" not in text
    print("IR PASS: allocation spaces, unpinned storage, versions and Cube scope")


@pytest.mark.parametrize("space", [tla.AddressSpace.L0A_SF, tla.AddressSpace.L0B_SF, "l0a_sf", "l0b_sf"])
def test_scale_mapping_is_explicitly_unsupported(space):
    with pytest.raises(NotImplementedError, match="scale-factor buffer mapping"):
        tla.buffer_type((16, 16), tl.float16, space)


def test_enum_and_legacy_types_agree():
    enum_type = tla.buffer_type((16, 16), tl.float32, tla.AddressSpace.L0C)
    old_type = tla.buffer_type((16, 16), tl.float32, "l0c")
    assert enum_type == old_type
    assert enum_type.mangle() == old_type.mangle()
    assert enum_type.space == "l0c"


def test_invalid_allocation_arguments():
    # Errors must precede builder access; these calls need no native builder.
    semantic = object()
    with pytest.raises(ValueError, match="only one of space and scope"):
        tla.alloc((16, 16), tl.float32, "ub", scope=tla.AddressSpace.L1, _semantic=semantic)
    with pytest.raises(ValueError, match="kernel arguments"):
        tla.alloc((16, 16), tl.float32, scope=tla.AddressSpace.GM, _semantic=semantic)
    with pytest.raises(ValueError, match="positive integer"):
        tla.alloc((16, 16), tl.float32, versions=0, _semantic=semantic)


def _launch(compile_only=False, seed=7):
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn((2, 16, 32), generator=generator)
    a = torch.randn((16, 32), generator=generator, dtype=torch.float16)
    b = torch.randn((32, 64), generator=generator, dtype=torch.float16)
    x, an, bn = source.npu(), a.npu(), b.npu()
    y = torch.full_like(x, float("nan"))
    c = torch.full((16, 64), float("nan"), dtype=torch.float32, device="npu")
    if compile_only:
        return [_copy_two_tiles.warmup(x, y, grid=(1, )), _cube_product.warmup(an, bn, c, grid=(1, ))]
    kernels = [_copy_two_tiles[(1, )](x, y), _cube_product[(1, )](an, bn, c)]
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), source, rtol=0, atol=0)
    torch.testing.assert_close(c.cpu(), a.float() @ b.float(), rtol=1e-3, atol=3e-2)
    return kernels


def test_compile():
    for kernel in _launch(compile_only=True):
        check_kernel_binary(kernel)


def test_run():
    for seed in (7, 19):
        _launch(seed=seed)
    print("ACCURACY PASS: independent UB buffers and enum-based L1/L0C Cube GEMM")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
        test_enum_and_legacy_types_agree()
        test_invalid_allocation_arguments()
        for space in (tla.AddressSpace.L0A_SF, tla.AddressSpace.L0B_SF):
            test_scale_mapping_is_explicitly_unsupported(space)
    if do_compile:
        test_compile()
    if do_run:
        test_run()


if __name__ == "__main__":
    main()
