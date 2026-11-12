"""Offline lowering and Ascend 950 accuracy for explicit L0 Paged MLA."""

from math import gcd
from pathlib import Path

import pytest
import triton

from kernels.utils import load_kernel
from utils.tla_test_utils import Options, compile_offline, get_stages, lower_offline

HERE = Path(__file__).parent
TARGETS = (('Ascend950PR_9579', 28), ('Ascend950DT_9581', 32))
SIGNATURE = dict(Q='*bf16', KV='*bf16', BLOCK_TABLE='*i32', KV_LENGTHS='*i32', NUMERATOR='*fp32', MAXIMUM='*fp32',
                 DENOMINATOR='*fp32', OUT='*bf16', LSE='*fp32')


@pytest.mark.parametrize('target,cores', TARGETS, ids=[target for target, _ in TARGETS])
@pytest.mark.parametrize('batch,pages', ((8, 1120), (4, 9)), ids=('full', 'ragged'))
def test_ir(target, cores, batch, pages):
    kernel = load_kernel(HERE / 'mla_kernel.py')
    options = Options()
    options.arch = target
    groups = cores // gcd(batch, cores)
    split_pages = 2 if batch == 4 else triton.cdiv(pages, groups)
    request_tiles = triton.cdiv(pages * 128, 112)
    tiles_per_core = triton.cdiv(batch * request_tiles, cores)
    splits = min(cores, triton.cdiv(request_tiles + tiles_per_core - 1, tiles_per_core))
    constants = dict(BATCH=batch, QUERY_LENGTH=4, QUERY_HEADS=24, MAX_PAGES=pages,
                     SPLIT_PAGES=triton.cdiv(split_pages, 7) * 7, SPLITS=splits, CUBE_CORES=cores, SCALE=0.04,
                     Q_TILE_ROWS=96, KV_L2=4, FUSE_MERGE=True, RETURN_LSE=True)

    module, context = compile_offline(kernel.attention, SIGNATURE, constants, options=options)
    lowered = lower_offline(module, context)
    assert lowered.verify()
    text = str(lowered)
    assert 'hivm.hir.mmadL0' in text
    assert 'hivm.hir.nd2nz' in text
    assert 'memref<?x576xbf16' in text


@pytest.fixture(scope='module')
def runtime():
    if not get_stages()[2]:
        pytest.skip('Requires an Ascend 950 NPU')
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_npu')
    from triton.backends.ascend.utils import AscendArchGen, arch_generation

    target = triton.runtime.driver.active.get_current_target()
    if arch_generation(target.arch) != AscendArchGen.GEN5:
        pytest.skip('Requires an Ascend 950 NPU')

    benchmark = load_kernel(HERE / 'benchmark.py')
    kernel = load_kernel(HERE / 'mla_kernel.py')
    previous = torch.npu.matmul.allow_hf32
    torch.npu.matmul.allow_hf32 = False
    try:
        yield torch, benchmark, kernel
    finally:
        torch.npu.matmul.allow_hf32 = previous


@pytest.mark.parametrize('case', ('full', 'ragged'))
@pytest.mark.parametrize('family,seed', (('normal', 1103), ('normal', 1104), ('uniform', 1103), ('peaked', 1103)))
def test_run(runtime, case, family, seed):
    _, benchmark, kernel = runtime
    inputs = benchmark.make_inputs(benchmark.CASES[case], family, seed)
    expected = benchmark.reference(inputs)
    call = kernel.prepare_call(*inputs, split_pages=2 if case == 'ragged' else None)
    for _ in range(3):
        benchmark.precision(call, expected)


def test_run_reuses_prepared_call(runtime):
    torch, benchmark, kernel = runtime
    inputs = benchmark.make_inputs(benchmark.CASES['ragged'], 'normal', 8291)
    call = kernel.prepare_call(*inputs, split_pages=2)
    benchmark.precision(call, benchmark.reference(inputs))
    q, kv, table, lengths = inputs
    q.add_(0.25)
    kv.mul_(0.75)
    lengths[-1] = 901
    table[-1, [0, 1]] = table[-1, [1, 0]]
    benchmark.precision(call, benchmark.reference(inputs))
    output = kernel.prepare_call(*inputs, split_pages=2, return_lse=False)()
    torch.npu.synchronize()
    expected, _ = benchmark.reference(inputs)
    assert output.dtype == torch.bfloat16 and output.shape == expected.shape
    assert ((output.float() - expected).abs().amax() / expected.abs().amax()).item() <= 0.005
