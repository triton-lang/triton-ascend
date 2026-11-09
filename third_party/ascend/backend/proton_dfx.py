"""Experimental serialized A5 DFX capture through the MSPTI PoC C API."""
import ctypes as C
import json
import os
import threading


class Record(C.Structure):
    _fields_ = [("start", C.c_uint64), ("end", C.c_uint64), ("channel", C.c_uint32), ("pipe", C.c_uint32),
                ("scope", C.c_uint32), ("frequency", C.c_uint32)]


class Stats(C.Structure):
    _fields_ = [("bytes", C.c_uint64), ("incomplete", C.c_uint64), ("overflow", C.c_uint64),
                ("driver_error", C.c_int32), ("flush_error", C.c_int32)]


class DfxCapture:
    _lock = threading.Lock()

    def __init__(self):
        self.lib = C.CDLL(os.environ.get("TRITON_MSPTI_DFX_LIBRARY", "libmspti.so"))
        signatures = {
            "msptiDfxStart": [C.c_uint32, C.POINTER(C.c_void_p)],
            "msptiDfxStop": [C.c_void_p],
            "msptiDfxGetRecords": [C.c_void_p,
                                   C.POINTER(C.POINTER(Record)),
                                   C.POINTER(C.c_size_t),
                                   C.POINTER(Stats)],
            "msptiDfxDestroy": [C.c_void_p],
            "msptiDfxGetRawData": [C.c_void_p, C.c_uint32,
                                   C.POINTER(C.POINTER(C.c_uint8)),
                                   C.POINTER(C.c_size_t)],
        }
        for name, args in signatures.items():
            try:
                fn = getattr(self.lib, name)
            except AttributeError as error:
                raise RuntimeError("A5 instrumentation requires the MSPTI DFX PoC library") from error
            fn.argtypes, fn.restype = args, C.c_int
        self.handle = C.c_void_p()

    @staticmethod
    def _check(rc, operation):
        if rc:
            raise RuntimeError(f"MSPTI DFX {operation} failed: {rc}")

    def start(self):
        import torch
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("The A5 DFX PoC supports only serialized kernel captures")
        try:
            torch.npu.synchronize()
            runtime = C.CDLL("libruntime.so")
            physical = C.c_uint32()
            convert = runtime.rtGetDevicePhyIdByIndex
            convert.argtypes = [C.c_uint32, C.POINTER(C.c_uint32)]
            convert.restype = C.c_int
            self._check(convert(torch.npu.current_device(), C.byref(physical)), "physical device lookup")
            self._check(self.lib.msptiDfxStart(physical.value, C.byref(self.handle)), "start")
        except BaseException:
            self._lock.release()
            raise

    def finish(self):
        import torch
        if not self.handle:
            return []
        try:
            torch.npu.synchronize()
            rc = self.lib.msptiDfxStop(self.handle)
            records, count, stats = C.POINTER(Record)(), C.c_size_t(), Stats()
            self._check(self.lib.msptiDfxGetRecords(self.handle, C.byref(records), C.byref(count), C.byref(stats)),
                        "read")
            status = {name: getattr(stats, name) for name, _ in Stats._fields_}
            values = [(r.channel, r.pipe, r.scope, r.start, r.end, r.frequency) for r in records[:count.value]]
            if os.environ.get("TRITON_DFX_DEBUG_DIR"):
                from pathlib import Path
                import time
                directory = Path(os.environ["TRITON_DFX_DEBUG_DIR"]) / str(time.time_ns())
                directory.mkdir(parents=True)
                (directory / "records.json").write_text(json.dumps({"stats": status, "records": values}, indent=2))
                for channel in range(11, 29):
                    raw, size = C.POINTER(C.c_uint8)(), C.c_size_t()
                    self._check(self.lib.msptiDfxGetRawData(self.handle, channel, C.byref(raw), C.byref(size)),
                                "raw read")
                    if size.value:
                        (directory / f"channel-{channel}.bin").write_bytes(C.string_at(raw, size.value))
            if rc or not count.value:
                raise RuntimeError(f"Incomplete/empty MSPTI DFX capture: rc={rc}, stats={json.dumps(status)}")
            if len({row[5] for row in values}) != 1:
                raise RuntimeError("Mixed AIC/AIV frequencies require separate DFX clock domains")
            return values
        finally:
            rc = self.lib.msptiDfxDestroy(self.handle)
            self.handle = C.c_void_p()
            self._lock.release()
            self._check(rc, "destroy")
