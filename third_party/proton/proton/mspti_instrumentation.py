"""Experimental serialized A5 instrumentation capture through the MSPTI PoC C API."""
import ctypes as C
import json
import os
import threading
import warnings


class ScopeRecord(C.Structure):
    _fields_ = [("start", C.c_uint64), ("end", C.c_uint64), ("channel", C.c_uint32), ("pipe", C.c_uint32),
                ("scope", C.c_uint32), ("frequency", C.c_uint32)]


class CollectionStats(C.Structure):
    _fields_ = [("bytes", C.c_uint64), ("incomplete", C.c_uint64), ("overflow", C.c_uint64),
                ("driver_error", C.c_int32), ("flush_error", C.c_int32)]


class MsptiInstrumentationCollector:
    _lock = threading.Lock()

    def __init__(self):
        selection = os.environ.get("PROTON_INSTRUMENTATION_CHANNELS")
        self.channels = None
        if selection is not None:
            try:
                self.channels = [int(value.strip()) for value in selection.split(",")]
            except ValueError as error:
                raise ValueError(
                    "PROTON_INSTRUMENTATION_CHANNELS must be a comma-separated subset of 11..28") from error
            if not self.channels or len(set(self.channels)) != len(self.channels) or any(channel < 11 or channel > 28
                                                                                         for channel in self.channels):
                raise ValueError("PROTON_INSTRUMENTATION_CHANNELS must contain unique channel IDs in 11..28")
        self.lib = C.CDLL(os.environ.get("PROTON_MSPTI_LIBRARY_PATH", "libmspti.so"))
        signatures = {
            "msptiInstrumentationStart": [C.c_uint32, C.POINTER(C.c_void_p)],
            "msptiInstrumentationStop": [C.c_void_p],
            "msptiInstrumentationGetRecords":
            [C.c_void_p,
             C.POINTER(C.POINTER(ScopeRecord)),
             C.POINTER(C.c_size_t),
             C.POINTER(CollectionStats)],
            "msptiInstrumentationDestroy": [C.c_void_p],
            "msptiInstrumentationGetRawData":
            [C.c_void_p, C.c_uint32, C.POINTER(C.POINTER(C.c_uint8)),
             C.POINTER(C.c_size_t)],
        }
        if self.channels is not None:
            signatures["msptiInstrumentationStartWithChannels"] = [
                C.c_uint32, C.POINTER(C.c_uint32), C.c_size_t,
                C.POINTER(C.c_void_p)
            ]
        for name, args in signatures.items():
            try:
                fn = getattr(self.lib, name)
            except AttributeError as error:
                raise RuntimeError(
                    f"A5 instrumentation requires the MSPTI instrumentation PoC library with {name}") from error
            fn.argtypes, fn.restype = args, C.c_int
        self.handle = C.c_void_p()
        requested = self.channels if self.channels is not None else list(range(11, 29))
        warnings.warn(
            f"A5 instrumentation samples hardware channels {requested}, not every Triton program/block. "
            "Channel IDs are not core IDs; an empty capture can mean the selected channels did not observe the kernel.",
            stacklevel=2)

    @staticmethod
    def _check(rc, operation):
        if rc:
            raise RuntimeError(f"MSPTI instrumentation {operation} failed: {rc}")

    def start(self):
        import torch
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("The A5 instrumentation PoC supports only serialized kernel captures")
        try:
            torch.npu.synchronize()
            runtime = C.CDLL("libruntime.so")
            physical = C.c_uint32()
            convert = runtime.rtGetDevicePhyIdByIndex
            convert.argtypes = [C.c_uint32, C.POINTER(C.c_uint32)]
            convert.restype = C.c_int
            self._check(convert(torch.npu.current_device(), C.byref(physical)), "physical device lookup")
            if self.channels is None:
                rc = self.lib.msptiInstrumentationStart(physical.value, C.byref(self.handle))
            else:
                channels = (C.c_uint32 * len(self.channels))(*self.channels)
                rc = self.lib.msptiInstrumentationStartWithChannels(physical.value, channels, len(channels),
                                                                    C.byref(self.handle))
            self._check(rc, "start")
        except BaseException:
            self._lock.release()
            raise

    def stop(self):
        import torch
        if not self.handle:
            return []
        try:
            torch.npu.synchronize()
            rc = self.lib.msptiInstrumentationStop(self.handle)
            records, count, stats = C.POINTER(ScopeRecord)(), C.c_size_t(), CollectionStats()
            self._check(
                self.lib.msptiInstrumentationGetRecords(self.handle, C.byref(records), C.byref(count), C.byref(stats)),
                "read")
            status = {name: getattr(stats, name) for name, _ in CollectionStats._fields_}
            values = [(r.channel, r.pipe, r.scope, r.start, r.end, r.frequency) for r in records[:count.value]]
            if os.environ.get("PROTON_INSTRUMENTATION_DEBUG_DIR"):
                from pathlib import Path
                import time
                directory = Path(os.environ["PROTON_INSTRUMENTATION_DEBUG_DIR"]) / str(time.time_ns())
                directory.mkdir(parents=True)
                requested = self.channels if self.channels is not None else list(range(11, 29))
                coverage = {
                    "kind": "sampled_hardware_channels", "requested_channels": requested, "observed_channels":
                    sorted({row[0]
                            for row in values}), "all_logical_blocks": False
                }
                (directory / "records.json").write_text(
                    json.dumps({"stats": status, "records": values, "coverage": coverage}, indent=2))
                for channel in requested:
                    raw, size = C.POINTER(C.c_uint8)(), C.c_size_t()
                    self._check(
                        self.lib.msptiInstrumentationGetRawData(self.handle, channel, C.byref(raw), C.byref(size)),
                        "raw read")
                    if size.value:
                        (directory / f"channel-{channel}.bin").write_bytes(C.string_at(raw, size.value))
            if rc or not count.value:
                raise RuntimeError(
                    f"Incomplete/empty MSPTI instrumentation capture: rc={rc}, stats={json.dumps(status)}; "
                    "this is partial hardware sampling, not guaranteed coverage of every logical block")
            if stats.flush_error:
                warnings.warn(
                    f"MSPTI instrumentation flush returned {stats.flush_error}; received records are returned, "
                    "but their presence does not establish complete capture.", stacklevel=2)
            if len({row[5] for row in values}) != 1:
                raise RuntimeError("Mixed AIC/AIV frequencies require separate instrumentation clock domains")
            return values
        finally:
            rc = self.lib.msptiInstrumentationDestroy(self.handle)
            self.handle = C.c_void_p()
            self._lock.release()
            self._check(rc, "destroy")
