"""CTRL write IR and binary coverage; use an idempotent HF32-disable write."""
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

CONFIGS = [dict(name="disable_hf32")]


@tla.jit
def control(X, Y):
    with tla.scope.cube():
        tla.set_ctrl(False, 48)
    with tla.scope.vector():
        ub = tla.alloc((64, ), tl.float32)
        tla.load_gm_to_ub(tla.view(X, (64, )), ub)
        tla.store_ub_to_gm(ub, tla.view(Y, (64, )))


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    mod, ctx = compile_offline(control, {"X": "*fp32", "Y": "*fp32"}, {}, output_dir=output_dir)
    text = str(lower_offline(mod, ctx, output_dir=output_dir))
    assert "set_ctrl false at ctrl[48]" in text
    print("IR PASS:", case["name"])


@matrix(CONFIGS)
def test_compile(case):
    x, y = torch.ones(64, device="npu"), torch.empty(64, device="npu")
    check_kernel_binary(control.warmup(x, y, grid=(1, )))


@matrix(CONFIGS)
def test_run(case):
    x = torch.arange(64, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    control[(1, )](x, y)
    torch.npu.synchronize()
    # This checks the surrounding kernel only, not hardware CTRL readback.
    torch.testing.assert_close(y.cpu(), x.cpu(), rtol=0, atol=0)


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
