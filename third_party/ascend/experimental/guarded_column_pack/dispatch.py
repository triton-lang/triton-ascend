"""Isolated experiment for kernel_full.py; not installed in the TA launcher.

ExperimentalColumnPack.cpp requires ascend.experimental_checked_full_rows, a
harness assertion, NOT a compiler-derived proof. Only pass variants compiled
from this exact fixture. Automatic certificate extraction remains unimplemented.
Build the host guard with `MAX_JOBS=1 python setup.py build_ext --inplace`.
"""
try:
    import native_guard
except ImportError:
    native_guard = None


def launch_checked(fast, original, x, cache, y, weight, bias, *,
                   D, N, W, BD, stride_x_n, stride_y_n, num_warps=4):
    """Check each invocation; forward the checked inputs directly to the kernel.

    Unit channel strides, zero chunk offset and SIMD scheduling are fixed parts
    of this fixture's contract. Unknown storage/aliases use the original variant.
    Compile/resource errors propagate; they are not silently hidden as fallback.
    """
    grid = ((D + BD - 1) // BD, N)
    try:
        valid = native_guard is not None and native_guard.check(
            x, cache, weight, bias, y, D, N, W, BD,
            grid[0], grid[1], stride_x_n, stride_y_n)
    except (TypeError, OverflowError):
        valid = False
    kernel = fast if valid else original
    return kernel[grid](
        x, cache, y, weight, bias, stride_x_n, 1, stride_y_n, 1,
        D=D, W=W, BD=BD, HAS_WEIGHT=weight is not None, HAS_BIAS=bias is not None,
        CHUNK_OFFSET=0, N_CHUNK=0, compile_mode="simd_simt_template",
        num_warps=num_warps, superblock_factor=1, num_stages=2,
        multibuffer=False, debug=False)
