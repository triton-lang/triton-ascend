"""Finite SD kernels. Importing this file does not import Triton or a device.

Only offline preparation may compile these functions. The production owned
caller accepts exact precompiled artifacts and never invokes JIT compilation.
"""
from __future__ import annotations
import hashlib
import inspect


def _known(X, Index, Values, S: tl.constexpr, D: tl.constexpr,
           PS: tl.constexpr, PD: tl.constexpr, LO0: tl.constexpr,
           HI0: tl.constexpr, LO1: tl.constexpr, HI1: tl.constexpr,
           MASK: tl.constexpr):
    s = tl.arange(0, S)[:, None]
    d = tl.arange(0, D)[None, :]
    index = tl.load(Index + tl.arange(0, D))[None, :]
    value = tl.load(Values + s * D + d)
    pointer = X + s * PS + index * PD
    if MASK:
        tl.store(pointer, value, (s >= LO0) & (s < HI0) & (d >= LO1) & (d < HI1))
    else:
        tl.store(pointer, value)


def _host(X, Index, Values, PS, PD, S: tl.constexpr, D: tl.constexpr,
          LO0: tl.constexpr, HI0: tl.constexpr, LO1: tl.constexpr,
          HI1: tl.constexpr, MASK: tl.constexpr):
    s = tl.arange(0, S)[:, None]
    d = tl.arange(0, D)[None, :]
    index = tl.load(Index + tl.arange(0, D))[None, :]
    value = tl.load(Values + s * D + d)
    pointer = X + s * PS + index * PD
    if MASK:
        tl.store(pointer, value, (s >= LO0) & (s < HI0) & (d >= LO1) & (d < HI1))
    else:
        tl.store(pointer, value)


def _device(X, Index, Values, Coefficients, S: tl.constexpr, D: tl.constexpr,
            LO0: tl.constexpr, HI0: tl.constexpr, LO1: tl.constexpr,
            HI1: tl.constexpr, MASK: tl.constexpr):
    ps = tl.load(Coefficients)
    pd = tl.load(Coefficients + 1)
    s = tl.arange(0, S)[:, None]
    d = tl.arange(0, D)[None, :]
    index = tl.load(Index + tl.arange(0, D))[None, :]
    value = tl.load(Values + s * D + d)
    pointer = X + s * ps + index * pd
    if MASK:
        tl.store(pointer, value, (s >= LO0) & (s < HI0) & (d >= LO1) & (d < HI1))
    else:
        tl.store(pointer, value)


_FUNCTIONS = {"known": _known, "host": _host, "device": _device}
_JIT = {}


def function_source_sha256(visibility):
    return hashlib.sha256(inspect.getsource(_FUNCTIONS[visibility]).encode()).hexdigest()


def builtin_jit(visibility):
    """Create a source object only. Never call it in production/cache-miss."""
    global tl
    if visibility not in _JIT:
        import triton
        import triton.language as tl
        _JIT[visibility] = triton.jit(_FUNCTIONS[visibility])
    return _JIT[visibility]
