"""Host-only audit of the focused package; requires only Python's stdlib."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import statistics

from expand_kernel import VARIANTS, ast_digest, expanded_function, render

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / "evidence" / name).read_text())


def main():
    for variant, record in read("ast_equivalence.json").items():
        assert variant in VARIANTS and record["equal"]
        assert ast_digest(expanded_function(variant)) == record["ast_sha256"]
        for frontend in ("compat", "native"):
            compile(render(variant, frontend), variant, "exec")
    assert len(read("ast_equivalence.json")) == 6

    # Decode the scalar constants, rather than trusting a scheduling comment.
    tree = ast.parse((HERE / "kernel.py").read_text())
    schedule = next(f for f in tree.body
                    if isinstance(f, ast.FunctionDef) and f.name == "inline_balanced_query_schedule")
    query_ids = []
    cores = []
    for branch in schedule.body:
        if isinstance(branch, ast.If):
            cores.append(branch.test.comparators[0].value)
            lo, hi, count = branch.body
            packed = lo.value.args[1].value | (hi.value.args[1].value << 64)
            query_ids.extend((packed >> (8 * i)) & 255 for i in range(count.value.value))
    assert cores == list(range(28))
    assert sorted(query_ids) == list(range(256)), "schedule must cover every query exactly once"

    data = read("performance_samples.json")
    assert data["completed"] and data["shared_device"]
    assert len(data["records"]) == 24
    assert {(r["spec"], r["model"])
            for r in data["records"]} == {(f"FA{i}", model)
                                          for i in range(1, 9)
                                          for model in ("best_v4", "best_v5", "ascendc")}
    count = 0
    for record in data["records"]:
        assert len(record["rounds"]) == 3 and record["core_match"] and not record["accepted_isolated"]
        medians = []
        for rnd in record["rounds"]:
            assert len(rnd["samples_us"]) == rnd["summary_rows"] == 40
            assert all(x > 0 for x in rnd["samples_us"])
            assert rnd["core_metadata"]
            assert all((m["Block Num"], m["Mix Block Num"]) == ("28", "56") for m in rnd["core_metadata"])
            assert statistics.median(rnd["samples_us"]) == rnd["median_us"]
            medians.append(rnd["median_us"])
            count += len(rnd["samples_us"])
        assert statistics.median(medians) == record["median_us"]
        assert [min(medians), max(medians)] == record["round_median_range_us"]
    assert count == 2880
    by = {(r["spec"], r["model"]): r["median_us"] for r in data["records"]}
    with (HERE / "evidence/performance.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    for row in rows:
        for column, model in (("tla_v4_us", "best_v4"), ("tla_final_us", "best_v5"), ("ascendc_us", "ascendc")):
            assert float(row[column]) == by[row["case"], model]

    validation = read("readable_validation.json")
    assert validation["completed"] and len(validation["records"]) == 64
    for model in ("final_tla", "ascendc"):
        records = [r for r in validation["records"] if r["model"] == model]
        assert sum(len(r["checks"]) for r in records) == 152
        for record in records:
            assert record["passed"]
            if model == "final_tla":
                expected = read("measured_binaries.json")[record["case"]]["binary_sha256"]
                assert record["matches_measured"] and record["binary_sha256"] == expected
            for check in record["checks"]:
                assert check["O"] <= .005 and check["max"] <= .0001 and check["sum"] <= .0001

    manifest = HERE / "MANIFEST.sha256"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    print("PASS: 6 exact ASTs, query bijection, 8 identical binaries, 304 precision checks, 2880 timing samples")


if __name__ == "__main__":
    main()
