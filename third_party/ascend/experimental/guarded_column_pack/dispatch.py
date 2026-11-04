"""Experimental runtime gate for the frozen conv access certificate.

Use only with variants built from the certified source/ABI. An automatic
compiler producer for this certificate is not implemented by this module.
Both callables must use exactly the checked tensors, strides and grid.
"""
try:
    import native_guard
except ImportError:
    native_guard=None


def launch_checked(fast,original,tensors,*,D,N,W,BD,grid,stride_x_n,stride_y_n,
                   stride_x_d=1,stride_y_d=1,chunk_offset=0):
    if native_guard is None or stride_x_d!=1 or stride_y_d!=1 or chunk_offset!=0 or len(grid)!=2:
        return original()
    try:
        ok=native_guard.check(tensors['x'],tensors['cache'],tensors['weight'],tensors['bias'],tensors['y'],
                              D,N,W,BD,grid[0],grid[1],stride_x_n,stride_y_n)
    except (KeyError,TypeError,OverflowError):
        ok=False
    # The certificate result is never cached just by shape/dtype: storage and
    # alias relations can change on the next launch with identical dimensions.
    return fast() if ok else original()
