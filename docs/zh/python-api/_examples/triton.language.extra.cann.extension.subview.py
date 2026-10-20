import triton
import triton.language as tl
import triton.extension.buffer.language as bl


@triton.jit
def test_subview_kernel(XBLOCK: tl.constexpr):
    src_buffer = bl.alloc(tl.float32, [XBLOCK, XBLOCK])
    result_buffer = bl.subview(
        src_buffer,
        offsets=[1, 0],
        sizes=[XBLOCK - 2, XBLOCK],
        strides=[1, 1],
    )
