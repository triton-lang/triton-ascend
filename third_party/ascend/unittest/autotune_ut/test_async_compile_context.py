"""No accelerator required: actual AsyncCompileMode and ThreadPoolExecutor."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from triton.runtime._async_compile import AsyncCompileMode, active_mode


def fail():
    raise RuntimeError('deliberate async failure')


@pytest.mark.parametrize('stage', ['compile', 'finalize', 'body', 'success', 'ignored'])
def test_context_is_restored_and_next_compile_works(stage):
    assert active_mode.get() is None
    caught = None
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with AsyncCompileMode(executor, ignore_errors=(stage == 'ignored')) as mode:
                mode.submit('key', fail if stage in ('compile', 'ignored') else lambda: 42,
                            (lambda value: fail()) if stage == 'finalize' else lambda value: None)
                if stage == 'body':
                    fail()
    except RuntimeError as exc:
        caught = str(exc)
    restored = active_mode.get() is None
    # Keep expected baseline failures from contaminating subsequent CPU tests.
    active_mode.set(None)
    assert (caught is not None) == (stage in ('compile', 'finalize', 'body'))
    assert restored, 'failed exit leaves a closed executor in active_mode'
    with ThreadPoolExecutor(max_workers=1) as executor:
        with AsyncCompileMode(executor) as mode:
            future = mode.submit('next', lambda: 7, lambda value: None)
        assert future.result() == 7
    assert active_mode.get() is None


@pytest.mark.parametrize('stage', ['compile', 'finalize'])
@pytest.mark.parametrize('ignore_errors', [False, True])
def test_multiple_futures_preserve_error_and_restore_context(stage, ignore_errors):
    """Finalizer errors still propagate even when compile errors are ignored."""
    assert active_mode.get() is None
    sentinel = RuntimeError('original exception object')

    def reject():
        raise sentinel

    caught = None
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            with AsyncCompileMode(executor, ignore_errors=ignore_errors) as mode:
                good = mode.submit('good', lambda: 41, lambda value: None)
                # Repeated keys must keep the existing future, including on an
                # exit that later fails while processing another future.
                assert mode.submit('good', reject, lambda value: None) is good
                mode.submit('bad', reject if stage == 'compile' else lambda: 42,
                            (lambda value: reject()) if stage == 'finalize' else lambda value: None)
                other = mode.submit('other', lambda: 43, lambda value: None)
    except RuntimeError as exc:
        caught = exc
    restored = active_mode.get() is None
    active_mode.set(None)  # Isolate expected baseline failures.
    assert caught is (None if stage == 'compile' and ignore_errors else sentinel)
    assert good.future.done() and other.future.done()
    assert restored, 'failed multi-future exit leaves a stale active context'
    with ThreadPoolExecutor(max_workers=1) as executor:
        with AsyncCompileMode(executor) as mode:
            next_kernel = mode.submit('recovery', lambda: 44, lambda value: None)
        assert next_kernel.result() == 44
    assert active_mode.get() is None
