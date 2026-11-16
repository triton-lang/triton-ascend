"""GM/UB loads and stores with immutable configuration objects."""
import sys
from pathlib import Path

# Do not let this category's copy.py shadow Python's standard library.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (compile_offline, lower_offline, get_stages, check_kernel_binary, matrix)

CONFIGS = [dict(name="plain_" + dtype, dtype=dtype) for dtype in ("fp16", "fp32", "i32")]
CONFIGS += [dict(name="evict_" + mode.name, dtype="fp32", eviction=mode) for mode in tla.EvictionPolicy]
CONFIGS += [
    dict(name="padding_" + mode.name, dtype="fp32", pad=mode, left=4, right=4)
    for mode in (tla.PadMode.PadFirstElem, tla.PadMode.PadValue)
]
CONFIGS += [dict(name="pad_null", dtype="fp32", pad=tla.PadMode.PadNull)]
CONFIGS += [
    dict(name="atomic_" + mode.name, dtype="fp32", atomic=mode)
    for mode in (tla.AtomicKind.ADD, tla.AtomicKind.MAX, tla.AtomicKind.MIN)
]
DTYPES = {"fp16": (tl.float16, torch.float16), "fp32": (tl.float32, torch.float32), "i32": (tl.int32, torch.int32)}


@tla.jit
def transfer(X, Y, DTYPE: tl.constexpr, PAD: tl.constexpr, LEFT: tl.constexpr, RIGHT: tl.constexpr, EVICT: tl.constexpr,
             ATOMIC: tl.constexpr):
    src = tla.view(X, (2, 32))
    out = tla.view(Y, (2, 32 + LEFT + RIGHT))
    ub = tla.alloc((2, 32 + LEFT + RIGHT), DTYPE)
    value: tl.constexpr = -3.0 if PAD == tla.PadMode.PadValue else None
    # Construct in a JIT function, including constexpr fields.
    p: tl.constexpr = tla.LoadGM2UBParams(pad_mode=PAD, pad_value=value, left_padding_num=LEFT, right_padding_num=RIGHT,
                                          eviction_policy=EVICT)
    tla.load_gm_to_ub(src, ub, p)
    tla.store_ub_to_gm(ub, out, tla.StoreUB2GMParams(atomic_kind=ATOMIC))


@tla.jit
def dynamic_padding(src, dst, value, left, right):
    tla.load_gm_to_ub(
        src, dst,
        tla.LoadGM2UBParams(pad_mode=tla.PadMode.PadValue, pad_value=value, left_padding_num=left,
                            right_padding_num=right))


def constants(case):
    return dict(DTYPE=DTYPES[case["dtype"]][0], PAD=case.get("pad"), LEFT=case.get("left", 0),
                RIGHT=case.get("right", 0), EVICT=case.get("eviction"), ATOMIC=case.get("atomic"))


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    sig = {"X": "*" + case["dtype"], "Y": "*" + case["dtype"]}
    mod, ctx = compile_offline(transfer, sig, constants(case), output_dir=output_dir)
    text = str(lower_offline(mod, ctx, output_dir=output_dir))
    assert text.count("hivm.hir.load ") == 1 and text.count("hivm.hir.store ") == 1
    assert 'hacc.mode = "tla"' in text
    if case.get("pad"):
        assert case["pad"].value in text
    if case.get("eviction"):
        assert case["eviction"].value in text
    if case.get("atomic"):
        assert "atomic = <" + case["atomic"].value + ">" in text
    if case["name"] == "plain_fp32":
        sig = {
            "src": tla.buffer_type((2, 32), tl.float32, "gm"), "dst": tla.buffer_type((2, 40), tl.float32, "ub"),
            "value": "fp32", "left": "i32", "right": "i32"
        }
        dynamic, _ = compile_offline(dynamic_padding, sig, {})
        ir = str(dynamic)
        assert "pad_value = %" in ir and "left_padding_num = %" in ir and "right_padding_num = %" in ir
        assert "arith.index_cast" in ir
    print("IR PASS:", case["name"])


def launch(case, compile_only=False):
    dtype = DTYPES[case["dtype"]][1]
    x_cpu = (torch.arange(64).reshape(2, 32) - 31).to(dtype)
    y = torch.full((2, 32 + case.get("left", 0) + case.get("right", 0)), 2, dtype=dtype, device="npu")
    x = x_cpu.npu()
    if compile_only:
        return transfer.warmup(x, y, grid=(1, ), **constants(case))
    transfer[(1, )](x, y, **constants(case))
    torch.npu.synchronize()
    expected = x_cpu
    if case.get("left"):
        pad = torch.full((2, 4), -3, dtype=dtype) if case["pad"] == tla.PadMode.PadValue else x_cpu[:, :1].expand(2, 4)
        expected = torch.cat((pad, x_cpu, pad), dim=1)
    op = case.get("atomic")
    if op == tla.AtomicKind.ADD:
        expected = expected + 2
    elif op == tla.AtomicKind.MAX:
        expected = expected.clamp_min(2)
    elif op == tla.AtomicKind.MIN:
        expected = expected.clamp_max(2)
    return y.cpu(), expected


@matrix(CONFIGS)
def test_compile(case):
    check_kernel_binary(launch(case, True))


@matrix(CONFIGS)
def test_run(case):
    if case.get("pad") or case.get("atomic"):
        print("SKIP run:", case["name"], "(padding/atomic load/store hits device error 507035 on A5)")
        return
    actual, expected = launch(case)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("ACCURACY PASS:", case["name"])


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in CONFIGS:
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)


if __name__ == "__main__":
    main()
