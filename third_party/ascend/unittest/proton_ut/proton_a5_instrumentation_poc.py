"""Run directly with the private TA, NPU-IR and MSPTI instrumentation PoC builds."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl

pl.enable_semantic("triton")


@triton.jit
def proton_a5_probe(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with pl.scope("kernel"):
        with pl.scope("load"):
            a = tl.load(x + offsets, offsets < n, other=0)
            b = tl.load(y + offsets, offsets < n, other=0)
        with pl.scope("compute"):
            result = a + b
        with pl.scope("store"):
            tl.store(out + offsets, result, offsets < n)


def validate_intervals(events):
    for pid in {event["pid"] for event in events}:
        local = [event for event in events if event["pid"] == pid]
        outer = [event for event in local if event["name"] == "kernel"]
        assert outer
        for name in ("load", "compute", "store"):
            children = [event for event in local if event["name"] == name]
            assert len(children) == len(outer), (pid, name, len(children), len(outer))
            for parent in outer:
                contained = [
                    child for child in children if child["ts"] >= parent["ts"] - 1e-9 and child["ts"] +
                    child["dur"] <= parent["ts"] + parent["dur"] + 1e-9
                ]
                assert len(contained) == 1, (pid, name, parent, contained)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4097)
    parser.add_argument("--lifecycle", action="store_true")
    parser.add_argument("--warmup-instrumented", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup_instrumented:
        os.environ["PROTON_INSTRUMENTATION_WARMUP_LAUNCHES"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    x = torch.arange(args.n, device="npu", dtype=torch.float32)
    y, out = torch.ones_like(x), torch.empty_like(x)

    def launch():
        kernel = proton_a5_probe[(triton.cdiv(args.n, 256), )](x, y, out, args.n, BLOCK=256, debug=True)
        torch.npu.synchronize()
        return kernel

    launch()
    path = args.output / "proton-instrumentation"
    session = proton.start(str(path), backend="instrumentation", data="trace")
    try:
        if args.warmup_instrumented:
            print("Explicit warmup of the instrumented binary without capture", flush=True)
            launch()
        print("Capturing first launch", flush=True)
        with proton.scope("active_initial"):
            compiled = launch()
        if args.lifecycle:
            proton.deactivate(session)
            print("Launching while inactive", flush=True)
            launch()
            proton.activate(session)
            print("Capturing second launch", flush=True)
            with proton.scope("active_resumed"):
                launch()
        metadata = compiled.metadata._asdict()
        assert metadata["proton_collection"] == "mspti"
        assert metadata["proton_metadata"][0]["proton"]["format_version"] == 2
        assert metadata.get("profile_scratch_size", 0) == 0
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        (args.output / "kernel.ttadapter.mlir").write_text(compiled.asm["ttadapter"])
    finally:
        proton.finalize(session, output_format="chrome_trace")
    torch.testing.assert_close(out, x + y)
    trace_path = path.with_suffix(".chrome_trace")
    trace = json.loads(trace_path.read_text())
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    assert events, "Empty trace"
    assert {e["name"] for e in events} == {"kernel", "load", "compute", "store"}
    assert all(e["dur"] > 0 and e["args"]["Frequency (MHz)"] > 0 for e in events)
    assert all("profiling channel" in str(e["pid"]) and "pipe" in str(e["tid"]) for e in events)
    assert all(e["args"]["coverage"] == "sampled_hardware_channels" for e in events)
    assert all(e["args"]["all_logical_blocks"] is False for e in events)
    capture_ids = {e["pid"].split(" profiling channel")[0] for e in events}
    assert len(capture_ids) == (2 if args.lifecycle else 1), capture_ids
    validate_intervals(events)
    summary = {
        "status": "passed", "n": args.n, "events": len(events), "captures": len(capture_ids), "collector":
        "MSPTI instrumentation API; no msprof op", "trace": str(trace_path), "durations_us":
        {name: [e["dur"]
                for e in events
                if e["name"] == name]
         for name in ("kernel", "load", "compute", "store")}
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
