import re
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "language", _ROOT / "runtime", _ROOT / "tools", _ROOT / "cuda"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

_CASES = _ROOT / "cases.txt"
_LINE_FN_FILE = re.compile(r"^(?P<fn>(?:Test\w+\.)?test_\w+)\s+(?P<path>\S+\.py)$")


def _allowed_cases() -> set[tuple[str, str]]:
    allowed: set[tuple[str, str]] = set()
    if not _CASES.is_file():
        return allowed
    for raw in _CASES.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _LINE_FN_FILE.match(line)
        if m:
            allowed.add((m.group("path"), m.group("fn")))
    return allowed


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")
    config.addinivalue_line("markers", "backend(name): select the runtime backend for a test module")
    config.addinivalue_line("markers", "forked: run test in a forked subprocess")


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="npu")


def _item_name(item) -> str:
    return getattr(item, "originalname", None) or item.name.split("[", 1)[0]


def _is_a3() -> bool:
    """True on A3/910B. A5 (910_95/950) stays False. Unknown host stays False."""
    try:
        import torch
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            return False
    except Exception:
        return False
    try:
        from triton.backends.ascend.utils import is_compile_on_910_95
        return not is_compile_on_910_95()
    except Exception:
        pass
    try:
        import acl
        name = acl.get_soc_name().lower()
        if any(s in name for s in ("ascend910_95", "ascend950", "910_958b")):
            return False
        return any(s in name for s in ("910b", "910a", "910d", "910_93"))
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    allowed = _allowed_cases()
    if not allowed:
        return
    keep = []
    for item in items:
        rel = Path(item.fspath).resolve().relative_to(_ROOT).as_posix()
        name = _item_name(item)
        cls = getattr(item, "cls", None)
        fn = f"{cls.__name__}.{name}" if cls is not None else name
        if (rel, fn) in allowed:
            keep.append(item)
    if _is_a3():
        skip = pytest.mark.skip(reason="A3: community_ut skipped for now")
        for item in keep:
            item.add_marker(skip)
    items[:] = keep


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        from triton import knobs

        with knobs.cache.scope(), knobs.runtime.scope():
            knobs.cache.dir = tmpdir
            yield tmpdir


@pytest.fixture
def fresh_knobs():
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_knobs_except_libraries():
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl(skipped_attr={"build", "nvidia", "amd"})
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())


@pytest.fixture(scope="module", autouse=True)
def assign_npu(worker_id):
    import torch
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return
    npu_count = torch.npu.device_count()
    if worker_id == "master":
        npu_id = 0
    else:
        npu_id = int(worker_id.replace("gw", "")) % npu_count
    torch.npu.set_device(npu_id)
