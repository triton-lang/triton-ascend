"""Experimental serialized A5 instrumentation capture through the MSPTI PoC C API."""
import ctypes as C
import json
import os
import threading
import time
import warnings
from pathlib import Path

_CHANNEL_IDS = range(11, 29)


class ScopeRecord(C.Structure):
    _fields_ = [("start", C.c_uint64), ("end", C.c_uint64), ("channel", C.c_uint32), ("pipe", C.c_uint32),
                ("scope", C.c_uint32), ("frequency", C.c_uint32)]


class CollectionStats(C.Structure):
    _fields_ = [("bytes", C.c_uint64), ("incomplete", C.c_uint64), ("overflow", C.c_uint64),
                ("driver_error", C.c_int32), ("flush_error", C.c_int32)]


class MsptiInstrumentationCollector:
    _lock = threading.Lock()

    def __init__(self):
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
        for name, args in signatures.items():
            try:
                fn = getattr(self.lib, name)
            except AttributeError as error:
                raise RuntimeError(
                    f"A5 instrumentation requires the MSPTI instrumentation PoC library with {name}") from error
            fn.argtypes, fn.restype = args, C.c_int
        self.handle = C.c_void_p()
        warnings.warn(
            "A5 instrumentation samples hardware channels, not every Triton program/block. "
            "Channel IDs are not core IDs; an empty capture does not prove that the kernel did not execute.",
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
            self._check(self.lib.msptiInstrumentationStart(physical.value, C.byref(self.handle)), "start")
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
            self._dump_debug(status, values)
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
            try:
                rc = self.lib.msptiInstrumentationDestroy(self.handle)
            finally:
                self.handle = C.c_void_p()
                self._lock.release()
            self._check(rc, "destroy")

    def _dump_debug(self, status, records):
        root = os.environ.get("PROTON_INSTRUMENTATION_DEBUG_DIR")
        if not root:
            return
        directory = Path(root) / str(time.time_ns())
        directory.mkdir(parents=True)
        observed = sorted({record[0] for record in records})
        coverage = {
            "kind": "sampled_hardware_channels", "requested_channels": list(_CHANNEL_IDS), "observed_channels":
            observed, "all_logical_blocks": False
        }
        (directory / "records.json").write_text(
            json.dumps({"stats": status, "records": records, "coverage": coverage}, indent=2))
        for channel in _CHANNEL_IDS:
            raw, size = C.POINTER(C.c_uint8)(), C.c_size_t()
            self._check(self.lib.msptiInstrumentationGetRawData(self.handle, channel, C.byref(raw), C.byref(size)),
                        "raw read")
            if size.value:
                (directory / f"channel-{channel}.bin").write_bytes(C.string_at(raw, size.value))
