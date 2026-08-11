"""Combine same-kernel msprof summaries into a reproducible comparison."""

import argparse
import csv
import json
from pathlib import Path


def load_results(specs, reference):
    results = []
    for spec in specs:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"expected NAME=SUMMARY.json, got {spec!r}")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["variant"] = name
        results.append(data)

    means = {result["variant"]: result["mean_us"] for result in results}
    if reference not in means:
        raise ValueError(f"reference {reference!r} has no supplied summary")
    reference_mean = means[reference]
    for result in results:
        result["speedup_vs_reference_pct"] = (
            (reference_mean - result["mean_us"]) / reference_mean * 100.0)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", nargs="+")
    parser.add_argument("--reference", default="baseline")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    results = load_results(args.result, args.reference)
    Path(args.json).write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    columns = (
        "variant", "captured_launches", "discarded_warmups",
        "measured_launches", "mean_us", "median_us", "min_us", "max_us",
        "speedup_vs_reference_pct")
    with Path(args.csv).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print("variant,mean_us,speedup_vs_reference_pct")
    for result in results:
        print(
            f"{result['variant']},{result['mean_us']:.3f},"
            f"{result['speedup_vs_reference_pct']:.2f}")


if __name__ == "__main__":
    main()
