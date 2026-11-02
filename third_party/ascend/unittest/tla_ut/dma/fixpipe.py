"""MMAD -> explicit Fixpipe store/copy routes and their configuration fields."""
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

CONFIGS = [
    dict(name=target + "_" + dtype, target=target, dtype=dtype, mode=tla.FixpipeDMAMode.NZ2ND)
    for target in ("gm", "ub")
    for dtype in ("fp32", "fp16", "bf16")
]
CONFIGS += [
    dict(name="gm_nz", target="gm", dtype="fp32", mode=tla.FixpipeDMAMode.NZ2NZ),
    dict(name="gm_transpose", target="gm", dtype="fp32", mode=tla.FixpipeDMAMode.NZ2DN),
    dict(name="l1_nz", target="l1", dtype="fp16", mode=tla.FixpipeDMAMode.NZ2NZ),
    dict(name="gm_relu", target="gm", dtype="fp32", mode=tla.FixpipeDMAMode.NZ2ND, relu=True),
    dict(name="gm_scale", target="gm", dtype="fp32", mode=tla.FixpipeDMAMode.NZ2ND, scale=0.5),
    dict(name="gm_unit0", target="gm", dtype="fp32", mode=tla.FixpipeDMAMode.NZ2ND, unit=tla.UnitFlagMode.DISABLED)
]
DTYPES = {
    "fp32": (tl.float32, torch.float32), "fp16": (tl.float16, torch.float16), "bf16": (tl.bfloat16, torch.bfloat16)
}


def params(case):
    cls = {"gm": tla.FixpipeL0C2GMParams, "ub": tla.FixpipeL0C2UBParams, "l1": tla.FixpipeL0C2L1Params}[case["target"]]
    quant = {
        "fp32": tla.FixpipePreQuantMode.NO_QUANT, "fp16": tla.FixpipePreQuantMode.F322F16, "bf16":
        tla.FixpipePreQuantMode.F322BF16
    }[case["dtype"]]
    if case.get("scale") is not None:
        quant = tla.FixpipePreQuantMode.QF322F32_PRE
    return cls(dma_mode=case["mode"], pre_quant=quant,
               pre_relu=tla.FixpipePreReluMode.NORMAL_RELU if case.get("relu") else tla.FixpipePreReluMode.NO_RELU,
               quant_scale=case.get("scale"), unit_flag_ctrl=case.get("unit"))


@tla.jit
def product(A, B, Y, P: tl.constexpr, TARGET: tl.constexpr, DTYPE: tl.constexpr, MODE: tl.constexpr):
    if TARGET == "ub" or TARGET == "l1":
        out = tla.alloc((16, 32), DTYPE)
    with tla.scope.cube():
        a = tla.alloc((1, 1, 16, 16), tl.float16, space="l1")
        b = tla.alloc((2, 1, 16, 16), tl.float16, space="l1")
        acc = tla.alloc((2, 1, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(A, (16, 16)), a)
        tla.load_gm_to_l1(tla.view(B, (16, 32)), b)
        tla.cube.mmad(a, b, acc, clear_accum=True)
        if TARGET == "gm":
            if MODE == tla.FixpipeDMAMode.NZ2NZ:
                dst = tla.view(Y, (2, 1, 16, 16))
            elif MODE == tla.FixpipeDMAMode.NZ2DN:
                dst = tla.view(Y, (32, 16))
            else:
                dst = tla.view(Y, (16, 32))
            tla.store_l0c_to_gm(acc, dst, P)
        elif TARGET == "ub":
            tla.copy_l0c_to_ub(acc, out, P)
        else:
            l1 = tla.alloc((2, 1, 16, 16), DTYPE, space="l1")
            tla.copy_l0c_to_l1(acc, l1, P)
            tla.copy_l1_to_ub(l1, out)
    if TARGET == "ub" or TARGET == "l1":
        with tla.scope.vector():
            tla.store_ub_to_gm(out, tla.view(Y, (16, 32)))


@tla.jit
def config_ir(src, dst, P: tl.constexpr, TARGET: tl.constexpr):
    if TARGET == "gm":
        tla.store_l0c_to_gm(src, dst, P)
    elif TARGET == "l1":
        tla.copy_l0c_to_l1(src, dst, P)
    else:
        tla.copy_l0c_to_ub(src, dst, P)


def check_params():
    cases = [
        ("l1", tla.FixpipeL0C2L1Params(channel_split=True,
                                       c0_pad_en=False), ("channel_split = true", "c0_pad_en = false")),
        ("gm", tla.FixpipeL0C2GMParams(unit_flag_ctrl=tla.UnitFlagMode.ENABLED_WITH_UPDATE, unit_flag_group_id=7),
         ("unit_flag<ENABLED_WITH_UPDATE>", "unit_flag_group_id = 7")),
        ("ub", tla.FixpipeL0C2UBParams(sub_block_idx=tla.FixpipeSubBlock.SUB_BLOCK_1), ("sub_block_1", )),
        ("ub", tla.FixpipeL0C2UBParams(dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                       sub_block_idx=None), ("ROW_SPLIT", )),
        ("ub", tla.FixpipeL0C2UBParams(dual_dst_mode=tla.FixpipeDualDstMode.COLUMN_SPLIT,
                                       sub_block_idx=None), ("COLUMN_SPLIT", )),
    ]
    for target, p, probes in cases:
        signature = {
            "src": tla.buffer_type((2, 1, 16, 16), tl.float32, "l0c"), "dst": tla.buffer_type((2, 1, 16, 16),
                                                                                              tl.float32, target)
        }
        mod, _ = compile_offline(config_ir, signature, {"P": p, "TARGET": target})
        for probe in probes:
            assert probe in str(mod), probe
        if target == "ub" and p.dual_dst_mode == tla.FixpipeDualDstMode.NO_DUAL:
            assert "dual_dst_mode" not in str(mod)


def constants(case):
    return dict(P=params(case), TARGET=case["target"], DTYPE=DTYPES[case["dtype"]][0], MODE=case["mode"])


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    mod, ctx = compile_offline(product, {"A": "*fp16", "B": "*fp16", "Y": "*" + case["dtype"]}, constants(case),
                               output_dir=output_dir)
    text = str(lower_offline(mod, ctx, output_dir=output_dir))
    assert text.count("hivm.hir.fixpipe ") == 1 and text.count("hivm.hir.mmadL1 ") == 1
    assert text.count("hivm.hir.nd2nz ") == 2
    if case["mode"] != tla.FixpipeDMAMode.NZ2NZ:
        assert case["mode"].value in text
    if case.get("scale") is not None:
        assert "quant_scale = %" in text and "QF322F32_PRE" in text
    if case["name"] == "gm_fp32":
        check_params()
    print("IR PASS:", case["name"])


def launch(case, compile_only=False):
    gen = torch.Generator().manual_seed(42)
    a, b = torch.randn(16, 16, generator=gen).half(), torch.randn(16, 32, generator=gen).half()
    y = torch.full((16, 32), float("nan"), dtype=DTYPES[case["dtype"]][1], device="npu")
    an, bn = a.npu(), b.npu()
    if compile_only:
        return product.warmup(an, bn, y, grid=(1, ), **constants(case))
    product[(1, )](an, bn, y, **constants(case))
    torch.npu.synchronize()
    expected = a.float() @ b.float()
    if case.get("scale") is not None:
        expected *= case["scale"]
    if case.get("relu"):
        expected = expected.clamp_min(0)
    if case["mode"] == tla.FixpipeDMAMode.NZ2DN:
        expected = expected.T.contiguous()
    if case["mode"] == tla.FixpipeDMAMode.NZ2NZ and case["target"] == "gm":
        expected = expected.reshape(1, 16, 2, 16).permute(2, 0, 1, 3).contiguous()
    return y.cpu().flatten(), expected.to(y.dtype).flatten()


@matrix(CONFIGS)
def test_compile(case):
    check_kernel_binary(launch(case, True))


@matrix(CONFIGS)
def test_run(case):
    actual, expected = launch(case)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=8e-3, atol=0.1)
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
