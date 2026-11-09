"""Scalar GM pointer offsets in the dedicated TLA adapter."""
import sys
from pathlib import Path

_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import compile_offline, lower_offline, get_stages, check_kernel_binary


@tla.jit
def offset_copy(X, Y, first, second):
    with tla.scope.vector():
        ub = tla.alloc((64, ), tl.float32)
        # The two addptr operations must each scale by sizeof(f32), while
        # view's own offset remains an element offset on the GM memref.
        src = tla.view((X + first) + second, (64, ), offsets=(3, ))
        dst = tla.view(Y + second, (64, ), offsets=(5, ))
        tla.load_gm_to_ub(src, ub)
        tla.store_ub_to_gm(ub, dst)


def test_ir():
    mod, ctx = compile_offline(offset_copy, {"X": "*fp32", "Y": "*fp32", "first": "i32", "second": "i32"}, {})
    source = str(mod)
    assert source.count("tt.addptr") == 3
    lowered = str(lower_offline(mod, ctx))
    assert "tt.addptr" not in lowered and "tt.ptr_to_int" not in lowered
    assert "hivm.hir.load" in lowered and "hivm.hir.store" in lowered
    # Both source offsets are dynamic and must be scaled independently.
    assert lowered.count("arith.muli") == 3
    assert lowered.count("memref.extract_aligned_pointer_as_index") == 2


def test_run():
    x = torch.arange(128, dtype=torch.float32, device="npu")
    y = torch.full((128, ), -1, dtype=torch.float32, device="npu")
    kernel = offset_copy[(1, )](x, y, 7, 11)
    torch.npu.synchronize()
    check_kernel_binary(kernel)
    expected = torch.full((128, ), -1, dtype=torch.float32)
    expected[16:80] = torch.arange(21, 85, dtype=torch.float32)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)

    # Keep the signed offset in the same compiled kernel's runtime arguments.
    negative = torch.full_like(y, -1)
    offset_copy[(1, )](x, negative, 20, -4)
    torch.npu.synchronize()
    expected = torch.full((128, ), -1, dtype=torch.float32)
    expected[1:65] = torch.arange(19, 83, dtype=torch.float32)
    torch.testing.assert_close(negative.cpu(), expected, rtol=0, atol=0)


def main():
    do_ir, _, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_run:
        test_run()
    print("PASS: TLA scalar GM pointer offsets")


if __name__ == "__main__":
    main()
