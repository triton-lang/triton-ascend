"""Host-only checks for Proton's experimental MSPTI collector."""
import ctypes
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module", autouse=True)
def assign_npu():
    """Override the parent fixture: this module never initializes an NPU."""
    return None


@pytest.fixture
def collector_module():
    path = Path(__file__).resolve().parents[3] / "proton/proton/mspti_instrumentation.py"
    spec = importlib.util.spec_from_file_location("mspti_instrumentation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def collector(collector_module, monkeypatch):
    module = collector_module
    calls = []
    records = (module.ScopeRecord * 1)(module.ScopeRecord(10, 30, 12, 0, 1, 1650))
    status = SimpleNamespace(start=0, stop=0, count=1, flush_error=0)

    def start(device, handle):
        calls.append(("start", device))
        if not status.start:
            ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = 1
        return status.start

    def stop(_handle):
        calls.append(("stop", ))
        return status.stop

    def get_records(_handle, output, count, stats):
        ctypes.cast(output, ctypes.POINTER(ctypes.POINTER(module.ScopeRecord)))[0] = records
        ctypes.cast(count, ctypes.POINTER(ctypes.c_size_t))[0] = status.count
        ctypes.cast(stats,
                    ctypes.POINTER(module.CollectionStats))[0] = module.CollectionStats(flush_error=status.flush_error)
        return 0

    def destroy(_handle):
        calls.append(("destroy", ))
        return 0

    def physical_device(_logical, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint32))[0] = 5
        return 0

    library = SimpleNamespace(msptiInstrumentationStart=start, msptiInstrumentationStop=stop,
                              msptiInstrumentationGetRecords=get_records,
                              msptiInstrumentationGetRawData=lambda *_args: 0, msptiInstrumentationDestroy=destroy)
    runtime = SimpleNamespace(rtGetDevicePhyIdByIndex=physical_device)
    monkeypatch.setattr(module.C, "CDLL", lambda path: runtime if path == "libruntime.so" else library)
    monkeypatch.setitem(
        sys.modules, "torch",
        SimpleNamespace(
            npu=SimpleNamespace(synchronize=lambda: calls.append(("synchronize", )), current_device=lambda: 0)))
    monkeypatch.delenv("PROTON_INSTRUMENTATION_DEBUG_DIR", raising=False)
    with pytest.warns(UserWarning, match="not every Triton program/block"):
        instance = module.MsptiInstrumentationCollector()
    yield instance, status, calls
    if instance.handle:
        instance.stop()


def test_collect_scope_records(collector):
    instance, _status, calls = collector
    instance.start()
    assert instance.stop() == [(12, 0, 1, 10, 30, 1650)]
    assert calls == [("synchronize", ), ("start", 5), ("synchronize", ), ("stop", ), ("destroy", )]
    assert not instance.handle
    assert not instance._lock.locked()
    assert instance.stop() == []


def test_failed_start_releases_lock(collector):
    instance, status, _calls = collector
    status.start = 1
    with pytest.raises(RuntimeError, match="start failed"):
        instance.start()
    assert not instance._lock.locked()
    status.start = 0
    instance.start()
    instance.stop()


@pytest.mark.parametrize("stop_result,count", [(1, 1), (0, 0)])
def test_invalid_capture_releases_session(collector, stop_result, count):
    instance, status, calls = collector
    status.stop, status.count = stop_result, count
    instance.start()
    with pytest.raises(RuntimeError, match="Incomplete/empty"):
        instance.stop()
    assert calls[-1] == ("destroy", )
    assert not instance.handle
    assert not instance._lock.locked()


def test_concurrent_capture_is_rejected(collector, collector_module):
    instance, _status, _calls = collector
    instance.start()
    second = collector_module.MsptiInstrumentationCollector()
    with pytest.raises(RuntimeError, match="serialized"):
        second.start()
    assert instance.handle
    assert instance._lock.locked()
    instance.stop()


def test_missing_api_is_reported(collector_module, monkeypatch):
    monkeypatch.setattr(collector_module.C, "CDLL", lambda _path: SimpleNamespace())
    with pytest.raises(RuntimeError, match="MSPTI instrumentation PoC library"):
        collector_module.MsptiInstrumentationCollector()


def test_flush_failure_is_visible(collector):
    instance, status, _calls = collector
    status.flush_error = 65534
    instance.start()
    with pytest.warns(UserWarning, match="flush returned 65534"):
        assert instance.stop() == [(12, 0, 1, 10, 30, 1650)]
    assert not instance.handle
    assert not instance._lock.locked()


def test_debug_dump_labels_partial_coverage(collector, monkeypatch, tmp_path):
    instance, _status, _calls = collector
    monkeypatch.setenv("PROTON_INSTRUMENTATION_DEBUG_DIR", str(tmp_path))
    instance.start()
    instance.stop()
    paths = list(tmp_path.glob("*/records.json"))
    assert len(paths) == 1
    coverage = json.loads(paths[0].read_text())["coverage"]
    assert coverage == {
        "kind": "sampled_hardware_channels", "requested_channels": list(range(11, 29)), "observed_channels": [12],
        "all_logical_blocks": False
    }


def test_destroy_exception_releases_lock(collector, monkeypatch):
    instance, _status, _calls = collector

    def fail_destroy(_handle):
        raise RuntimeError("destroy exception")

    monkeypatch.setattr(instance.lib, "msptiInstrumentationDestroy", fail_destroy)
    instance.start()
    with pytest.raises(RuntimeError, match="destroy exception"):
        instance.stop()
    assert not instance.handle
    assert not instance._lock.locked()
