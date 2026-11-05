"""Explicit UB destinations and GM subview offsets retain their aliases."""
import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import compile_offline, lower_offline, check_kernel_binary


@tla.jit
def buffer_copy(src, dst, offset):
    source = tla.subview(tla.view(src, (32, )), (offset, ), (16, ))
    target = tla.subview(tla.view(dst, (32, )), (8, ), (16, ))
    source_tiles = tla.make_partition_view(source, [16], [1], [8])
    target_tiles = tla.make_partition_view(target, [16], [1], [8])
    ub = tla.alloc((8, ), tl.float32)
    tile = tl.program_id(0)
    tla.load(source_tiles, index=(tile, ), dst=ub)
    tla.store(target_tiles, ub, index=(tile, ))


def test_ir():
    module, ctx = compile_offline(buffer_copy, {"src": "*fp32", "dst": "*fp32", "offset": "i32"}, {})
    text = str(lower_offline(module, ctx))
    assert "tv.view_load" in text and "tv.view_store" in text
    assert "outs(" in text and "tensor<" not in text
    assert "memref.subview" in text and "offset: ?>" in text


def test_run():
    src = torch.arange(32, dtype=torch.float32, device="npu")
    for offset in (0, 4, 16):
        dst = torch.full_like(src, -17)
        kernel = buffer_copy[(2, )](src, dst, offset)
        check_kernel_binary(kernel)
        expected = torch.full_like(src, -17)
        expected[8:24] = src[offset:offset + 16]
        torch.testing.assert_close(dst, expected, rtol=0, atol=0)
