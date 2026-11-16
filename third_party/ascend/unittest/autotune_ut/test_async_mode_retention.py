"""A retained exited context must not retain otherwise unowned compile results."""
from concurrent.futures import ThreadPoolExecutor
import gc
import weakref

from triton.runtime._async_compile import AsyncCompileMode


class CompileResult:
    pass


def test_exited_mode_releases_completed_result():
    refs = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        with AsyncCompileMode(executor) as mode:
            mode.submit('key', CompileResult, lambda value: refs.append(weakref.ref(value)))
    gc.collect()
    assert mode is not None  # Keep the exited context alive, as a caller may.
    assert len(refs) == 1 and refs[0]() is None
