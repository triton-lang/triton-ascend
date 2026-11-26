"""GDB counters in a *separate* run of our test process, not the baseline."""
from collections import Counter
import os

import gdb

hits = Counter()
resolved = set()


def reg(number):
    return int(gdb.parse_and_eval(f"$x{number}"))


class Returned(gdb.FinishBreakpoint):

    def __init__(self, label):
        super().__init__(internal=True)
        self.label = label

    def stop(self):
        rc = reg(0) & 0xffffffff
        hits[(self.label, "returned")] += 1
        if self.label.startswith("read:") and 0 < rc < (1 << 31):
            hits[(self.label, "bytes")] += rc
        elif not self.label.startswith("read:"):
            hits[(self.label, f"rc={rc}")] += 1
        return False


class Observed(gdb.Breakpoint):

    def __init__(self, symbol, kind):
        super().__init__(symbol, internal=True)
        self.kind = kind

    def stop(self):
        label = self.kind
        if label == "marker":
            for probe in probes:
                if probe.locations:
                    resolved.add(probe.kind)
        elif label == "read":
            label = f"read:{reg(1) & 0xffffffff}"
        elif label in ("enqueue", "buffer"):
            # AArch64 ABI: x0=this, x1=msptiActivity*, first field=kind.
            data = gdb.selected_inferior().read_memory(reg(1), 4)
            label += f":kind={int.from_bytes(bytes(data), 'little')}"
        hits[(label, "called")] += 1
        if self.kind in ("read", "enqueue", "buffer", "parse_task"):
            Returned(label)
        return False


gdb.execute("set pagination off")
gdb.execute("set breakpoint pending on")
gdb.execute("set environment LD_PRELOAD " + os.environ["PAIR_MSPTI"])
probes = [
    Observed("prof_channel_read", "read"),
    Observed("Mspti::Parser::KernelParser::ReportStarsSocLog", "soc_parser"),
    Observed("Mspti::Parser::KernelParser::ReportRtTaskTrack", "host_task"),
    Observed("Mspti::Parser::KernelParser::KernelParserImpl::ParseDeviceTask", "parse_task"),
    Observed("Mspti::Activity::ActivityManager::Record", "enqueue"),
    Observed("Mspti::Activity::ActivityBuffer::Record", "buffer"),
    Observed("probe_mark", "marker"),
]
try:
    gdb.execute("run")
finally:
    print(f"STAGE resolved={sorted(resolved)} hits={dict(hits)}")
