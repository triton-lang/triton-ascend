"""Validate all outputs, or profile paired kernel-only TLA/AscendC timings."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

from flash_attention import CASES, OPTIONS, prepare_call

FAMILIES = (("normal", 20261030, 10), ("normal", 20261031, 3), ("uniform", 20261101, 3), ("peaked", 20261102, 3))


def make_inputs(s, d, family, seed):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, k, v = (torch.randn(4, 8, s, d, generator=generator).to(torch.bfloat16).npu().contiguous() for _ in range(3))
    if family == "uniform":
        q.zero_()
    elif family == "peaked":
        q.mul_(4)
        k.mul_(4)
    return q, k, v


def native_call(q, k, v, causal):
    import torch
    import torch_npu
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device=q.device), 1) if causal else None

    def call():
        result = torch_npu.npu_fusion_attention(q, k, v, head_num=q.shape[1], input_layout="BNSD", pse=None,
                                                padding_mask=None, atten_mask=mask, scale=q.shape[-1]**-0.5,
                                                keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647,
                                                inner_precise=0, prefix=None, actual_seq_qlen=None,
                                                actual_seq_kvlen=None, sparse_mode=2 if causal else 0)
        return result[:3]

    return call


def reference(q, k, v, causal):
    """Blocked FP32 reference, with HF32 disabled by main()."""
    import torch
    b, h, s, d = q.shape
    qf, kf, vf = (x.reshape(b * h, s, d).float() for x in (q, k, v))
    out = torch.empty_like(qf)
    maxes = torch.empty((b * h, s), dtype=torch.float32, device=q.device)
    sums = torch.empty_like(maxes)
    for lo in range(0, s, 256):
        hi = min(lo + 256, s)
        end = hi if causal else s
        scores = (qf[:, lo:hi] @ kf[:, :end].transpose(1, 2)) * d**-0.5
        if causal:
            mask = torch.arange(end, device=q.device)[None, :] > torch.arange(lo, hi, device=q.device)[:, None]
            scores.masked_fill_(mask, float("-inf"))
        m = scores.amax(-1)
        maxes[:, lo:hi] = m
        sums[:, lo:hi] = (scores - m[..., None]).exp().sum(-1)
        out[:, lo:hi] = scores.softmax(-1) @ vf[:, :end]
    return out.reshape(b, h, s, d), maxes.reshape(b, h, s, 1), sums.reshape(b, h, s, 1)


def check_outputs(actuals, refs):
    import torch
    checks = {}
    expected_shapes = (refs[0].shape, (*refs[1].shape[:3], 8), (*refs[2].shape[:3], 8))
    for name, actual, ref, shape, dtype, tol in zip(("O", "max", "sum"), actuals, refs, expected_shapes,
                                                    (torch.bfloat16, torch.float32, torch.float32),
                                                    (.005, .0001, .0001)):
        assert actual.shape == shape and actual.dtype == dtype, (name, actual.shape, actual.dtype)
        assert bool(torch.isfinite(actual).all()), name
        error = float((actual.float() - ref).abs().max()) / max(float(ref.abs().max()), 1e-30)
        checks[name] = error
        assert error <= tol, (name, error, tol)
    return checks


def binary_receipt(call, case):
    """Use the launched artifact, not an arbitrary cache entry with the same name."""
    expected = json.loads((Path(__file__).parent / "evidence/measured_binaries.json").read_text())[case]
    binary = call.compiled.asm["npubin"]
    digest = hashlib.sha256(binary).hexdigest()
    return dict(binary_sha256=digest, measured_binary_sha256=expected["binary_sha256"],
                matches_measured=digest == expected["binary_sha256"])


def validate(args, report, save):
    import torch
    for case in args.cases:
        s, d, causal = CASES[case]
        families = FAMILIES if not args.smoke else FAMILIES[:1]
        for family, seed, repeats in families:
            q, k, v = make_inputs(s, d, family, seed)
            refs = reference(q, k, v, causal)
            final = prepare_call(q, k, v, causal, args.frontend)
            native = native_call(q, k, v, causal)
            final()
            torch.npu.synchronize()
            receipt = binary_receipt(final, case)
            for model, call in (("final_tla", final), ("ascendc", native)):
                record = dict(case=case, model=model, family=family, seed=seed, checks=[])
                if model == "final_tla":
                    record.update(receipt)
                report["records"].append(record)
                for _ in range(1 if args.smoke else repeats):
                    if hasattr(call, "outputs"):
                        for output in call.outputs:
                            output.fill_(float("nan"))
                    actuals = call()
                    torch.npu.synchronize()
                    record["checks"].append(check_outputs(actuals, refs))
                record["passed"] = True
                save()
            print(case, family, seed, "PASS", "binary_matches=" + str(receipt["matches_measured"]), flush=True)
            del q, k, v, refs, final, native, call, actuals
            torch.npu.empty_cache()


def profile_window(call, path, samples, warmup):
    import torch
    import torch_npu
    if path.exists():
        raise FileExistsError(f"Refusing to mix old/new profiles: {path}")
    path.mkdir(parents=True)
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                                                    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(path))):
        for _ in range(samples):
            call()
        torch.npu.synchronize()
    for raw in path.rglob("PROF_*"):
        if raw.is_dir():
            with (path / "export.log").open("a") as log:
                subprocess.run(["msprof", "--export=on", "--output=" + str(raw)], stdout=log, stderr=subprocess.STDOUT,
                               check=True)
    traces, summaries = list(path.rglob("trace_view.json")), list(path.rglob("op_summary*.csv"))
    assert len(traces) == len(summaries) == 1, (traces, summaries)
    trace = json.loads(traces[0].read_text())
    events = trace if isinstance(trace, list) else trace["traceEvents"]
    kernels = [
        e for e in events
        if e.get("ph") == "X" and str(e.get("args", {}).get("Task Type", "")).startswith(("AI_CORE", "AI_VECTOR",
                                                                                          "MIX_", "KERNEL"))
    ]
    with summaries[0].open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(kernels) == len(rows) == samples, (len(kernels), len(rows), samples)
    assert all((row["Block Num"], row["Mix Block Num"]) == ("28", "56") for row in rows), "core mismatch"
    durations = [float(e["dur"]) for e in kernels]
    # Preserve all samples and cross-check two independently exported formats.
    csv_durations = [float(row["Task Duration(us)"]) for row in rows]
    assert all(abs(a - b) < .002 for a, b in zip(sorted(durations), sorted(csv_durations)))
    return dict(samples_us=durations, median_us=statistics.median(durations), aic=28, aiv=56, samples=len(rows))


def benchmark(args, report, save):
    import torch
    for case in args.cases:
        s, d, causal = CASES[case]
        q, k, v = make_inputs(s, d, "normal", 20261030)
        refs = reference(q, k, v, causal)
        calls = [("baseline_tla_v4", prepare_call(q, k, v, causal, args.frontend, baseline=True)),
                 ("final_tla", prepare_call(q, k, v, causal, args.frontend)), ("ascendc", native_call(q, k, v, causal))]
        for model, call in calls:
            actuals = call()
            torch.npu.synchronize()
            check_outputs(actuals, refs)
        records = {model: dict(case=case, model=model, rounds=[]) for model, _ in calls}
        report["records"].extend(records.values())
        for rnd in range(args.rounds):
            shift = rnd % len(calls)
            order = calls[shift:] + calls[:shift]
            if rnd % 2:
                order = order[::-1]
            for model, call in order:
                path = args.output.parent / (args.output.stem + "_profiles") / f"{case}-{model}-{rnd}"
                result = profile_window(call, path, args.samples, args.warmup)
                record = records[model]
                record["rounds"].append(result)
                medians = [r["median_us"] for r in record["rounds"]]
                record.update(median_us=statistics.median(medians), range_us=[min(medians), max(medians)])
                save()
                print(case, model, rnd, result["median_us"], "us", flush=True)
        del q, k, v, calls, call, refs, actuals
        torch.npu.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "benchmark"))
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--frontend", choices=("compat", "native"), default="compat")
    parser.add_argument("--smoke", action="store_true", help="One normal input/repeat per case")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file to preserve existing evidence")
    if min(args.rounds, args.samples, args.warmup) <= 0:
        parser.error("rounds/samples/warmup must be positive")
    import torch
    import torch_npu
    import importlib.metadata
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    report = dict(completed=False, shared_device=True, frontend=args.frontend, launch_options=OPTIONS,
                  device_name=torch.npu.get_device_name(0),
                  packages={n: importlib.metadata.version(n)
                            for n in ("torch", "torch-npu", "triton-ascend")}, records=[])

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output)

    save()
    try:
        (validate if args.action == "validate" else benchmark)(args, report, save)
        report["completed"] = True
    finally:
        save()


if __name__ == "__main__":
    main()
