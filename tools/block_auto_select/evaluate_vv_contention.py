"""Post-selection replay against existing NPU measurements, without NPU access.

This program cannot change the predictions or parameters. It checks binary
identity and correctness evidence before joining a timing label. Development
examples are reported as replay, never as independently measured accuracy.
"""
import argparse
import csv
import math
from pathlib import Path
from statistics import median

from cpu_resource_gate import sha256
from run_vv_contention_experiment import read, write


def verified_times(prediction, measurement, *, require_static_score=True):
    """Verify same-binary device evidence; optionally include verified fallbacks."""
    reference = measurement.get("reference", {})
    fp64_reference = (reference.get("policy") == "full_output_CPU_FP64_original_case" and "passed" not in reference)
    reference_ok = reference.get("passed") is True
    if fp64_reference:
        # Pointwise fixtures report pass/fail per output, whereas the upstream
        # pytest path has a separate top-level reference.passed record.
        reference_ok = sha256(reference["source"]) == reference["sha256"]
        accuracy = measurement.get("post_profile_correctness", {})
        outputs = accuracy.get("outputs", [])
        reference_ok = reference_ok and accuracy.get("guards") is True and bool(outputs) and all(
            value.get("passed") is True for value in outputs)
    if (measurement.get("status") != "COMPLETE" or not reference_ok
            or measurement.get("input_shape_dtype_grid_match_static") is not True
            or measurement.get("post_profile_correctness", {}).get("passed") is not True):
        return {}
    times = {}
    for key, static in prediction["candidates"].items():
        measured = measurement.get("candidate_results", {}).get(key, {})
        if static.get("resource_checked") is not True:
            continue
        if require_static_score and static["status"] != "STATIC_COST_ESTIMATED":
            continue
        if measured.get("status") != "MEASURED":
            continue
        if measured.get("binary_sha256") != static["binary"]["sha256"]:
            raise ValueError("timing refers to a different binary: " + key)
        if (measured.get("static_binary_loaded") is not True or measured.get("loaded_and_synchronized") is not True
                or measured.get("correctness", {}).get("passed") is not True):
            raise ValueError("missing execution/correctness evidence: " + key)
        if fp64_reference:
            accuracy = measured["correctness"]
            outputs = accuracy.get("outputs", [])
            if accuracy.get("guards") is not True or not outputs or not all(
                    value.get("passed") is True for value in outputs):
                raise ValueError("missing full-output FP64 validation: " + key)
        if key != "unmerged":
            launch = measured.get("last_launch", {})
            if (launch.get("runtime_status") != "APPLIED" or launch.get("applied_factor") != static["factor"]
                    or launch.get("selected_axis") != static["axis"]):
                raise ValueError("requested merge was not executed: " + key)
        samples = measured["samples_us"]
        if len(samples) != 5 or not all(math.isfinite(x) and x > 0 for x in samples):
            raise ValueError("missing five valid timing samples: " + key)
        timing = median(samples)
        if not math.isclose(timing, measured["median_us"], abs_tol=1e-9):
            raise ValueError("stored median differs from samples: " + key)
        times[key] = timing
    return times


def metric(rows):
    eligible = [r for r in rows if r["multiple_measured_choices"]]
    return dict(cases=len(eligible), old_within_3pct=sum(r["old_within_3pct"] for r in eligible),
                new_within_3pct=sum(r["new_within_3pct"] for r in eligible),
                old_exact_best=sum(r["old_exact_best"] for r in eligible), new_exact_best=sum(r["new_exact_best"]
                                                                                              for r in eligible))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True, action="append")
    parser.add_argument("--previous", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completion = read(args.cpu / "static-complete.json")
    if completion["predictions_sha256"] != sha256(args.cpu / "static-predictions.json"):
        raise ValueError("predictions changed after sealing")
    if completion["protocol_sha256"] != sha256(args.cpu / "protocol.json"):
        raise ValueError("profile changed after sealing")
    predictions = read(args.cpu / "static-predictions.json")
    previous, measurements = {}, {}
    for path in args.previous:
        for row in read(path):
            if row["id"] in previous:
                raise ValueError("duplicate prior prediction")
            previous[row["id"]] = row
    # This is the first entry point permitted to open measurement files.
    for path in args.measurement:
        for row in read(path)["results"]:
            if row["case"] in measurements:
                raise ValueError("duplicate measurement")
            measurements[row["case"]] = row
    rows = []
    for row in predictions:
        times = verified_times(row, measurements.get(row["id"], {}))
        old, new = previous[row["id"]]["selected"], row["selected"]
        best = min(times, key=times.get) if times else None
        comparable = len(times) >= 2 and old in times and new in times
        record = dict(id=row["id"], name=row["name"], old_selected=old, new_selected=new, times_us=times,
                      best_measured=best, all_candidates_compared=row["all_candidates_compared"],
                      multiple_measured_choices=comparable, sensitivity_choices=row["sensitivity_choices"])
        for prefix, choice in (("old", old), ("new", new)):
            record[prefix + "_us"] = times.get(choice)
            record[prefix + "_exact_best"] = bool(comparable and times[choice] == times[best])
            record[prefix + "_within_3pct"] = bool(comparable and times[choice] <= times[best] * 1.03)
            record[prefix + "_slowdown_pct"] = (times[choice] / times[best] - 1) * 100 if comparable else None
        rows.append(record)
    result = dict(scope="known-case development replay; no new device execution", rows=rows,
                  all_comparable=metric(rows),
                  fully_covered_comparable=metric([r for r in rows if r["all_candidates_compared"]]),
                  single_measured_choice_cases=sum(len(r["times_us"]) == 1 for r in rows),
                  no_static_choice_cases=sum(r["new_selected"] is None
                                             for r in rows), prediction_sha256=completion["predictions_sha256"],
                  input_files={str(p): sha256(p)
                               for p in args.measurement + args.previous}, device_executed=False)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "evaluation.json", result)
    columns = ["算子", "旧选择", "旧选择耗时", "新选择", "新选择耗时", "已测最快配置", "已测最快耗时", "距最快慢(%)", "备注"]
    with (args.output / "VV自动选参改进对比.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            times = row["times_us"]

            def fmt(value):
                return "" if value is None else f"{value:.2f}"

            def config(value):
                if value == "unmerged":
                    return "不合并"
                if value is None:
                    return "未覆盖"
                axis, factor = value.removeprefix("a").split("_f")
                return f"axis={axis}, factor={factor}"

            note = ("候选覆盖完整" if row["all_candidates_compared"] else "部分候选未覆盖")
            if len(times) == 1:
                note += "；仅一项可用，不计命中率"
            if not times:
                note += "；没有可评价的自动选择"
            writer.writerow(
                dict(
                    zip(columns, [
                        row["name"],
                        config(row["old_selected"]),
                        fmt(row["old_us"]),
                        config(row["new_selected"]),
                        fmt(row["new_us"]),
                        config(row["best_measured"]),
                        fmt(times.get(row["best_measured"])),
                        fmt(row["new_slowdown_pct"]), note
                    ])))
    print(result["all_comparable"])
    print(result["fully_covered_comparable"])


if __name__ == "__main__":
    main()
