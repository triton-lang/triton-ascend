"""Freeze CPU-only VV predictions from prepared manifests in a fresh directory.

No device measurements are opened here. A separate evaluation step may compare
the sealed predictions to already measured, identical binaries. Existing runs
and their published tables are never modified.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time

from backend_cost_model import ASSUMPTIONS, Unsupported, read_cost_table
from cpu_resource_gate import inspect_final_ub, sha256
from vv_contention_model import VERSION, ContentionProfile, analyze_candidate, choose, schedule


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def verify_artifacts(candidate, *, ub_inspector=None):
    """Use final allocated IR and the exact binary, never a factor whitelist."""
    if candidate.get("resource_checked") is not True:
        raise ValueError("candidate resource unconfirmed")
    files = candidate["files"]
    for role, record in files.items():
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"{role} changed since compilation")
    proof = read(files["resource"]["path"])
    if proof["status"] != "UB_RESOURCE_CHECKED" or proof.get("resource_checked") is not True:
        raise ValueError("backend resource check missing")
    if proof["artifact"]["sha256"] != files["binary"]["sha256"]:
        raise ValueError("binary differs from resource proof")
    if proof["final_ir_sha256"] != files["final_ir"]["sha256"]:
        raise ValueError("final IR differs from resource proof")
    final_ir = Path(files["final_ir"]["path"]).read_text()
    metadata = read(files["metadata"]["path"])
    target = metadata.get("target", {}).get("arch")
    if not target or proof.get("target", target) != target:
        raise ValueError("resource proof and metadata targets differ")
    usage = (inspect_final_ub(final_ir, target) if ub_inspector is None else ub_inspector(
        final_ir, target, metadata, proof, files))
    if usage["address_extent_bits"] > usage["capacity_bits"]:
        raise ValueError("final UB address exceeds capacity")
    if candidate["factor"] != 1:
        report = candidate["merge_report"]
        if (report["compile_status"] != "REWRITTEN" or report["requested_factor"] != candidate["factor"]
                or report["selected_axis"] != candidate["axis"]):
            raise ValueError("candidate does not match merge report")
    return files, final_ir, metadata, usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--cost-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    profile = ContentionProfile(**read(args.profile)) if args.profile else ContentionProfile()
    table = read_cost_table(args.cost_table.read_text())
    if not table:
        raise ValueError("empty target instruction cost table")
    args.output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).parent
    profiles = {
        f"issue={i};dispatch={d}": ContentionProfile(profile.global_bytes_per_cycle, i, d)
        for i in (profile.dma_issue_interval / 2, profile.dma_issue_interval, profile.dma_issue_interval * 2)
        for d in (profile.block_dispatch_interval / 2, profile.block_dispatch_interval,
                  profile.block_dispatch_interval * 2)
    }
    write(
        args.output / "protocol.json",
        dict(
            model=VERSION, profile=asdict(profile), local_assumptions=ASSUMPTIONS,
            sensitivity_profiles={key: asdict(p)
                                  for key, p in profiles.items()
                                  }, parameter_status="uncalibrated modelling assumptions; not measured hardware rates",
            purpose="development replay; known diagnostic examples, not held-out accuracy", timing_labels_read=False,
            device_executed=False, compilation_executed=False,
            table={"path": str(args.cost_table), "sha256": sha256(args.cost_table)
                   }, inputs=[{"path": str(p), "sha256": sha256(p)} for p in args.manifest], sources={
                       p: sha256(code / p)
                       for p in
                       [Path(__file__).name, "vv_contention_model.py", "backend_cost_model.py", "cpu_resource_gate.py"]
                   }))
    rows, ids = [], set()
    for manifest_path in args.manifest:
        for case in read(manifest_path)["cases"]:
            if case["id"] in ids:
                raise ValueError("duplicate case ID across manifests")
            ids.add(case["id"])
            fixture = read(case["fixture"])
            row = dict(id=case["id"], name=case["name"], candidates={}, manifest=str(manifest_path),
                       fixture_sha256=sha256(case["fixture"]))
            for key, candidate in case["candidates"].items():
                result = dict(status=candidate["status"], resource_checked=False, error=candidate.get("error"),
                              factor=candidate["factor"], axis=candidate["axis"])
                started = time.monotonic()
                if candidate.get("resource_checked") is True:
                    try:
                        if candidate["factor"] not in (1, 2, 4):
                            raise Unsupported("factor outside current 0/2/4 scope")
                        files, final_ir, metadata, usage = verify_artifacts(candidate)
                        result.update(
                            analyze_candidate(
                                Path(files["llvm"]["path"]).read_text(),
                                Path(files["ttir"]["path"]).read_text(), final_ir, metadata, fixture["contract"],
                                candidate["axis"], candidate["factor"], table, profile=profile, keep_traces=True))
                        traces = result.pop("block_traces")
                        result.update(resource_checked=True, binary=files["binary"], resource_proof=files["resource"],
                                      ub_bytes=usage["address_extent_bits"] / 8,
                                      ub_capacity_bytes=usage["capacity_bits"] / 8)
                        result["sensitivity_scores"] = {
                            name: schedule(traces, result["core_count"], p)["score"]
                            for name, p in profiles.items()
                        }
                        trace_path = args.output / "traces" / f"{case['id']}-{key}.json.gz"
                        trace_path.parent.mkdir(exist_ok=True)
                        with gzip.open(trace_path, "wt") as stream:
                            json.dump({"core_count": result["core_count"], "block_traces": traces}, stream)
                        result["trace_artifact"] = {"path": str(trace_path), "sha256": sha256(trace_path)}
                    except (ValueError, KeyError, OSError) as error:
                        result.update(status="STATIC_ANALYSIS_UNSUPPORTED", resource_checked=False, error=str(error))
                result["analysis_seconds"] = time.monotonic() - started
                row["candidates"][key] = result
                write(args.output / "scores" / case["id"] / (key + ".json"), result)
                print(case["id"], key, result["status"], result.get("score", result.get("error")), flush=True)
            row["selected"] = choose(row["candidates"])
            row["all_candidates_compared"] = all(c["status"] in ("STATIC_COST_ESTIMATED", "UB_REJECTED",
                                                                 "COMPILE_REJECTED", "FRONTEND_REJECTED")
                                                 for c in row["candidates"].values())
            row["sensitivity_choices"] = dict(
                Counter(
                    choose({
                        key: dict(value, score=value.get("sensitivity_scores", {}).get(name))
                        for key, value in row["candidates"].items()
                    })
                    for name in profiles))
            row["selection_reason"] = ("LOWEST_STATIC_SCORE"
                                       if row["all_candidates_compared"] else "PARTIAL_COVERAGE_CANNOT_CLAIM_OPTIMAL")
            rows.append(row)
            write(args.output / "static-predictions.json", rows)
    write(
        args.output / "static-complete.json",
        dict(predictions_sha256=sha256(args.output / "static-predictions.json"),
             protocol_sha256=sha256(args.output / "protocol.json"), completed_unix=time.time(),
             timing_labels_read=False, device_executed=False))


if __name__ == "__main__":
    main()
