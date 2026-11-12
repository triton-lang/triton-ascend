"""Real NPU regression: compiler rejection must not retain autotune inputs.

Run with one explicitly isolated NPU visible. No compiler/allocator mocks.
The baseline is expected to fail the post-error ownership assertions.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import weakref

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.backends.ascend.runtime.autotuner import AutoTilingTuner
from triton.runtime._async_compile import active_mode


@triton.jit
def add_one(x, y, N: tl.constexpr, FAIL: tl.constexpr, BLOCK: tl.constexpr):
    tl.static_assert(not FAIL, 'input-lifetime deliberate compile rejection')
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + offsets, tl.load(x + offsets, offsets < N, 0) + 1, offsets < N)


def reject_prune(configs, named_args, **kwargs):
    raise ValueError('input-lifetime deliberate pruning rejection')


def grid(meta):
    return (triton.cdiv(1024, meta['BLOCK']), )


def snapshot():
    torch.npu.synchronize()
    free, total = torch.npu.mem_get_info()
    return dict(allocated=torch.npu.memory_allocated(), reserved=torch.npu.memory_reserved(), free=free, total=total)


RESULTS = []


def record(row):
    RESULTS.append(row)
    target = os.environ.get('E094_MATRIX_JSON')
    if target:
        import triton.backends.ascend.runtime.autotuner as module
        import triton.runtime._async_compile as async_module
        p = Path(module.__file__)
        async_path = Path(async_module.__file__)
        Path(target).write_text(
            json.dumps(
                dict(
                    torch=torch.__version__,
                    torch_npu=torch_npu.__version__,
                    triton=triton.__version__,
                    source=str(p),
                    source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    async_source=str(async_path),
                    async_source_sha256=hashlib.sha256(async_path.read_bytes()).hexdigest(),
                    visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                    cases=RESULTS,
                ), indent=2) + '\n')


@pytest.fixture(autouse=True)
def restore_test_context():
    # Check the real exit state inside each test. Only tear down afterward so
    # expected baseline failures cannot poison unrelated regression cases.
    assert active_mode.get() is None
    try:
        yield
    finally:
        active_mode.set(None)
        gc.collect()


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('entry', ['run', 'warmup'])
@pytest.mark.parametrize('failure', ['compile', 'prune'])
def test_failed_call_releases_storage(monkeypatch, parallel, entry, failure):
    monkeypatch.setenv('TRITON_AUTOTUNE_PARALLEL_COMPILE', str(int(parallel)))
    monkeypatch.setenv('TRITON_BENCH_METHOD', 'default')
    monkeypatch.setenv('TRITON_ENABLE_UBTUNER', '')
    torch.npu.set_device(0)
    options = {'prune_configs_by': {'early_config_prune': reject_prune}} if failure == 'prune' else {}
    tuner = triton.autotune(configs=[triton.Config({'BLOCK': 64}),
                                     triton.Config({'BLOCK': 128})], key=['FAIL'], **options)(add_one)
    assert isinstance(tuner, AutoTilingTuner)
    before = snapshot()
    # Two 16MiB storages, with only the first 1024 elements used by the tiny
    # kernel. Views make the memory consequence visible without a large launch.
    x = torch.empty(4 * 1024 * 1024, device='npu', dtype=torch.float32)[:1024]
    y = torch.empty(4 * 1024 * 1024, device='npu', dtype=torch.float32)[:1024]
    refs = [weakref.ref(x), weakref.ref(y)]
    error = None
    try:
        if entry == 'run':
            tuner[grid](x, y, 1024, True)
        else:
            tuner.warmup(x, y, 1024, True, grid=grid)
    except Exception as exc:
        # Store strings only; retaining a traceback would retain the inputs.
        error = {'type': type(exc).__name__, 'message': str(exc)}
    del x, y
    gc.collect()
    after = snapshot()
    retained = [ref() is not None for ref in refs]
    cleared = tuner.nargs is None
    # Attribution control also ensures expected failing baseline tests do not
    # accumulate retained NPU allocations across test cases.
    tuner.nargs = None
    gc.collect()
    control = snapshot()
    control_released = all(ref() is None for ref in refs)
    record(
        dict(parallel=parallel, entry=entry, failure=failure, error=error, before=before, after_failure=after,
             retained=retained, nargs_cleared=cleared, after_manual_clear=control,
             manual_clear_released=control_released))
    assert error is not None and 'input-lifetime deliberate' in error['message']
    assert control_released, 'another owner needs investigation'
    assert cleared and not any(retained), 'failed call retains input tensors'


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('failure_entry', ['run', 'warmup'])
def test_success_cache_and_recovery(monkeypatch, parallel, failure_entry):
    monkeypatch.setenv('TRITON_AUTOTUNE_PARALLEL_COMPILE', str(int(parallel)))
    monkeypatch.setenv('TRITON_BENCH_METHOD', 'default')
    monkeypatch.setenv('TRITON_ENABLE_UBTUNER', '')
    torch.npu.set_device(0)
    tuner = triton.autotune(configs=[triton.Config({'BLOCK': 64}),
                                     triton.Config({'BLOCK': 128})], key=['FAIL'])(add_one)
    assert isinstance(tuner, AutoTilingTuner)
    x = torch.arange(1024, dtype=torch.float32, device='npu')
    y = torch.empty_like(x)
    try:
        if failure_entry == 'run':
            tuner[grid](x, y, 1024, True)
        else:
            tuner.warmup(x, y, 1024, True, grid=grid)
    except Exception as exc:
        assert 'input-lifetime deliberate compile rejection' in str(exc)
    else:
        pytest.fail('expected compiler rejection')
    assert active_mode.get() is None, 'failed warmup leaves a stale async context'
    for _ in range(3):
        tuner[grid](x, y, 1024, False)
        torch.npu.synchronize()
        torch.testing.assert_close(y.cpu(), x.cpu() + 1, rtol=0, atol=0)
        assert tuner.nargs is None
    # Successful warmup must not leave input references either; complete any
    # returned futures before observing ownership.
    compiled = tuner.warmup(x, y, 1024, False, grid=grid)
    for item in compiled:
        if hasattr(item, 'result'):
            item.result()
    del compiled
    refs = [weakref.ref(x), weakref.ref(y)]
    del x, y
    gc.collect()
    assert tuner.nargs is None and all(ref() is None for ref in refs)
    record(dict(parallel=parallel, scenario='success_cache_recovery_warmup', failure_entry=failure_entry, passed=True))
