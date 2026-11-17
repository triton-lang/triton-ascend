"""Host tests: no device, HAL or MSPTI library is required."""
import json
import struct
import sys
from types import SimpleNamespace

import pytest

monitor = pytest.importorskip("triton.profiler.ascend_perf_monitor")


@pytest.mark.parametrize("collection", ["perf_monitor", "a5_perf_monitor", "mspti"])
def test_perf_monitor_metadata_dispatch(collection, tmp_path, monkeypatch):
    from triton.profiler.hooks import instrumentation

    calls = []
    monkeypatch.setattr(instrumentation, "_get_backend_name", lambda: "ascend")
    monkeypatch.delenv("PROTON_INSTRUMENTATION_WARMUP_LAUNCHES", raising=False)
    monkeypatch.setattr(instrumentation.libproton, "init_function_metadata", lambda *args: None)
    monkeypatch.setattr(instrumentation.libproton, "record_instrumentation_events", lambda f, rows: calls.append(f))
    monkeypatch.setattr(monitor, "PerfMonitorCollector",
                        lambda: SimpleNamespace(start=lambda: calls.append("start"), stop=lambda: []))
    metadata = tmp_path / "add.json"
    metadata.write_text(
        json.dumps(
            {"proton_collection": collection, "proton_metadata": [{"proton": {"format_version": 2, "scopes": []}}]}))
    hook = instrumentation.InstrumentationHook(None)
    hook.init_handle(None, 1, "add", {"add.json": str(metadata)}, "hash")
    launch = SimpleNamespace(data={"function": 1})
    hook.enter(launch)
    hook.exit(launch)
    assert calls == ["start", 1]


def _words(*words):
    return b"".join(struct.pack("<HH", *word) for word in words)


def test_channel_relative_intervals():
    raw = {
        11: _words((1, 0x401), (99, 0x7000), (3, 0xf000), (2, 0xc01)),
        12: _words((4, 0x2401), (1, 0xfff), (3, 0x2c01)),
    }
    records, incomplete = monitor._decode(raw, (1000, 1650))
    assert records == [(11, 0, 1, 1, 6, 1000), (12, 2, 1, 4, 8, 1650)]
    assert incomplete == 0


def test_unpaired_and_truncated_words():
    raw = {12: _words((1, 0x401), (2, 0x401), (1, 0xc02)) + b"x"}
    records, incomplete = monitor._decode(raw, (1000, 1650))
    assert records == [] and incomplete == 4


def test_partial_start_releases_only_owned_channels(monkeypatch):
    operations = []

    def stop(device, channel):
        operations.append(("stop", channel))
        return 0

    driver = SimpleNamespace(physical=lambda *args: 0, clocks=lambda dev: (1000, 1650),
                             set_command=lambda dev, kind: operations.append(
                                 ("command", kind)), start=lambda dev, ch, params: -1 if ch == 13 else 0, stop=stop)
    monkeypatch.setattr(monitor, "_Driver", lambda: driver)
    npu = SimpleNamespace(current_device=lambda: 0, synchronize=lambda dev: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=npu))
    with pytest.warns(UserWarning):
        collector = monitor.PerfMonitorCollector()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="start channel 13"):
            collector.start()
        assert not collector.active
    assert operations == [("command", 1), ("stop", 12), ("stop", 11), ("command", 2)] * 2


def test_stop_reads_pending_data_and_releases_driver(monkeypatch):
    data = [_words((2, 0x401), (5, 0xc01))]
    operations = []

    def read(dev, ch, buf, size):
        if ch == 12 and data:
            payload = data.pop()
            monitor.C.memmove(buf, payload, len(payload))
            return len(payload)
        return 0

    driver = SimpleNamespace(physical=lambda *args: 0, clocks=lambda dev: (1000, 1650),
                             set_command=lambda dev, kind: operations.append(kind), start=lambda *args: 0, read=read,
                             flush=lambda *args: 0, stop=lambda *args: 0)
    monkeypatch.setattr(monitor, "_Driver", lambda: driver)
    monkeypatch.setitem(sys.modules, "torch",
                        SimpleNamespace(npu=SimpleNamespace(current_device=lambda: 0, synchronize=lambda dev: None)))
    with pytest.warns(UserWarning):
        collector = monitor.PerfMonitorCollector()
    collector.start()
    assert collector.stop() == [(12, 0, 1, 2, 7, 1650)]
    assert collector.stop() == []
    collector.start()
    with pytest.raises(RuntimeError, match="empty performance monitor capture"):
        collector.stop()
    assert operations == [1, 2, 1, 2]
