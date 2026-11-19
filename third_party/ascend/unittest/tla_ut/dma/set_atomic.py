"""Explicit atomic control IR, object compilation, and NPU accuracy coverage."""
import sys
from pathlib import Path

# Do not let this category's copy.py shadow Python's standard library.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (compile_offline, lower_offline, get_stages, check_kernel_binary)

DTYPES = (
    (tl.float32, torch.float32),
    (tl.float16, torch.float16),
    (tl.bfloat16, torch.bfloat16),
    (tl.int8, torch.int8),
    (tl.int16, torch.int16),
    (tl.int32, torch.int32),
)


@tla.jit
def atomic_store(X, Y, Z, KIND: tl.constexpr, DTYPE: tl.constexpr):
    src = tla.view(X, (64, ))
    dst = tla.view(Y, (64, ))
    plain_dst = tla.view(Z, (64, ))
    ub = tla.alloc((64, ), DTYPE)
    with tla.scope.vector():
        tla.load_gm_to_ub(src, ub)
        tla.sync.barrier(tla.sync.PIPE.ALL)
        tla.set_atomic(KIND, DTYPE)
        tla.store_ub_to_gm(ub, dst)
        tla.sync.barrier(tla.sync.PIPE.ALL)
        tla.set_atomic(tla.AtomicKind.NONE, DTYPE)
        tla.store_ub_to_gm(ub, plain_dst)


def test_ir():
    mod, ctx = compile_offline(atomic_store, {"X": "*fp32", "Y": "*fp32", "Z": "*fp32"},
                               {"KIND": tla.AtomicKind.ADD, "DTYPE": tl.float32})
    text = str(lower_offline(mod, ctx))
    enabled = "hivm.hir.set_atomic kind = <add>[type = f32]"
    reset = "hivm.hir.set_atomic kind = <none>[type = f32]"
    store = "hivm.hir.store"
    assert enabled in text and reset in text and store in text
    first_store = text.index(store)
    second_store = text.index(store, first_store + 1)
    assert text.index(enabled) < first_store < text.index(reset) < second_store
    for kind in (tla.AtomicKind.NONE, tla.AtomicKind.ADD, tla.AtomicKind.MAX, tla.AtomicKind.MIN):
        module, context = compile_offline(atomic_store, {"X": "*fp16", "Y": "*fp16", "Z": "*fp16"},
                                          {"KIND": kind, "DTYPE": tl.float16})
        assert "hivm.hir.set_atomic kind = <" + kind.value + ">[type = f16]" in str(lower_offline(module, context))
    print("IR PASS: set_atomic ADD/MAX/MIN/NONE and reset ordering")


def test_invalid_inputs():
    for kind, dtype, message in ((tla.AtomicKind.AND, tl.float32, "kind must be"), ("add", tl.float32, "kind must be"),
                                 (tla.AtomicKind.ADD, tl.int64, "dtype must be")):
        with pytest.raises(Exception, match=message):
            compile_offline(atomic_store, {"X": "*fp32", "Y": "*fp32", "Z": "*fp32"}, {"KIND": kind, "DTYPE": dtype})
    print("IR PASS: unsupported atomic controls rejected")


def test_compile():
    x = torch.ones(64, dtype=torch.float32, device="npu")
    y = torch.ones_like(x)
    z = torch.empty_like(x)
    compiled = atomic_store.warmup(x, y, z, grid=(1, ), KIND=tla.AtomicKind.ADD, DTYPE=tl.float32)
    check_kernel_binary(compiled)
    print("COMPILE PASS: set_atomic ADD/NONE")


def test_run():
    for tl_dtype, torch_dtype in DTYPES:
        x_cpu = torch.arange(-31, 33, dtype=torch.int32).to(torch_dtype)
        y_cpu = torch.full((64, ), 2, dtype=torch_dtype)
        x = x_cpu.npu()
        for kind in (tla.AtomicKind.ADD, tla.AtomicKind.MAX, tla.AtomicKind.MIN):
            y = y_cpu.npu()
            z = torch.full_like(x, -100)
            atomic_store[(1, )](x, y, z, KIND=kind, DTYPE=tl_dtype)
            torch.npu.synchronize()
            if kind == tla.AtomicKind.ADD:
                expected = y_cpu + x_cpu
            elif kind == tla.AtomicKind.MAX:
                expected = torch.maximum(y_cpu, x_cpu)
            else:
                expected = torch.minimum(y_cpu, x_cpu)
            torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(z.cpu(), x_cpu, rtol=0, atol=0)
            print("ACCURACY PASS:", torch_dtype, kind.name, "and NONE reset", flush=True)


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
        test_invalid_inputs()
    if do_compile:
        test_compile()
    if do_run:
        test_run()


if __name__ == "__main__":
    main()
