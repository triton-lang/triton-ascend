"""Paged MLA lowering, accuracy and benchmark on Ascend 950."""

from pathlib import Path

import pytest

from kernels.mla_reference import make_inputs, precision, reference
from kernels.utils import benchmark, load_kernel
from utils.tla_test_utils import Options, compile_offline, lower_offline

CASES = {"full": (143360, ) * 8, "ragged": (4, 129, 260, 1031)}
SIGNATURE = dict(Q="*bf16", KV="*bf16", BLOCK_TABLE="*i32", KV_LENGTHS="*i32", NUMERATOR="*fp32", MAXIMUM="*fp32",
                 DENOMINATOR="*fp32", OUT="*bf16", LSE="*fp32")


class A5Options(Options):
    arch = "Ascend950PR_9579"


@pytest.fixture(scope="module")
def kernel():
    return load_kernel(Path(__file__).with_name("mla.py"))


@pytest.mark.parametrize("case,pages,split_pages,splits", [("full", 1120, 161, 5), ("ragged", 9, 7, 6)])
def test_ir(kernel, case, pages, split_pages, splits, output_dir):
    constants = dict(BATCH=len(CASES[case]), QUERY_LENGTH=4, QUERY_HEADS=24, MAX_PAGES=pages, SPLIT_PAGES=split_pages,
                     SPLITS=splits, CUBE_CORES=28, SCALE=0.04, Q_TILE_ROWS=96, KV_L2=4, FUSE_MERGE=True,
                     RETURN_LSE=True)
    module, ctx = compile_offline(kernel.attention, SIGNATURE, constants, output_dir, A5Options())
    text = str(lower_offline(module, ctx, output_dir))
    assert "hivm.hir.mmadL0" in text
    assert "hivm.hir.sync_block_set" in text and "hivm.hir.sync_block_wait" in text
    assert "l2_cache_mode = 4 : i32" in text


def _prepare(kernel, case, family, seed):
    inputs = make_inputs(CASES[case], family, seed)
    call = kernel.prepare_call(*inputs, split_pages=2 if case == "ragged" else None)
    precision(call, reference(inputs))
    return call


@pytest.mark.parametrize("case,family,seed", [("full", "normal", 1103), ("ragged", "peaked", 1103)])
def test_run(npu_runtime, kernel, case, family, seed):
    _prepare(kernel, case, family, seed)


def test_perf(perf_output, npu_runtime, kernel):
    benchmark(_prepare(kernel, "full", "normal", 1103), perf_output)
