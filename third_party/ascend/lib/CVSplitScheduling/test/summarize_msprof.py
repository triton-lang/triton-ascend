"""Summarize measured AI_CORE launches from an msprof task_time CSV."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def find_task_csv(path):
    path = Path(path)
    if path.is_file():
        return path
    matches = sorted(path.rglob("task_time_*.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected one task_time CSV under {path}, found {len(matches)}")
    return matches[0]


def read_kernel_times(path, kernel_name="_attn_fwd_0"):
    csv_path = find_task_csv(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        values = [
            float(row["task_time(us)"])
            for row in rows
            if row["kernel_name"] == kernel_name and row["kernel_type"] == "AI_CORE"
        ]
    if not values:
        raise ValueError(f"no AI_CORE rows for {kernel_name!r} in {csv_path}")
    return csv_path, values


def summarize(path, warmup=3, kernel_name="_attn_fwd_0"):
    csv_path, captured = read_kernel_times(path, kernel_name)
    measured = captured[warmup:]
    if not measured:
        raise ValueError(
            f"warmup={warmup} removes all {len(captured)} captured launches")
    return {
        "task_time_csv": str(csv_path.resolve()),
        "kernel_name": kernel_name,
        "captured_launches": len(captured),
        "discarded_warmups": warmup,
        "measured_launches": len(measured),
        "mean_us": statistics.fmean(measured),
        "median_us": statistics.median(measured),
        "min_us": min(measured),
        "max_us": max(measured),
        "measured_us": measured,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--kernel-name", default="_attn_fwd_0")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = summarize(args.profile, args.warmup, args.kernel_name)
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
